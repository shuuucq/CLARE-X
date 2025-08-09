import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from util.conf import ModelConf
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
from data.loader import FileIO
from data.ui_graph import Interaction
from utils import *
import logging
import time
from os.path import abspath
from time import strftime, localtime
from datetime import datetime
from model import LightGCN, MoCoMultiViewContrastive
import random
from torch.nn.functional import cosine_similarity
import numpy as np
import os
from logging.handlers import TimedRotatingFileHandler
import torch.nn.functional as F
from collections import deque
import rerank
import json
import asyncio
from collections import defaultdict

GPU = torch.cuda.is_available()
set_random_seed(42)

class AttentionFusion(nn.Module):
    def __init__(self, view_names, emb_dim):
        super().__init__()
        self.view_names = view_names
        self.score_layer = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(emb_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for name in view_names
        })

    def forward(self, view_dict, return_alignment_loss=False):
        scores = []
        embs = []
        for name in self.view_names:
            emb = view_dict[name]  # shape: [B, D]
            score = self.score_layer[name](emb)  # shape: [B, 1]
            scores.append(score)
            embs.append(emb)
        scores = torch.cat(scores, dim=1)  # [B, V]
        weights = F.softmax(scores, dim=1)  # [B, V]
        fused = sum(w.unsqueeze(-1) * e for w, e in zip(weights.t(), embs))  # sum over V

        alignment_loss = 0.0
        if return_alignment_loss:
            num_pairs = 0
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    alignment_loss += F.mse_loss(embs[i], embs[j])
                    num_pairs += 1
            if num_pairs > 0:
                alignment_loss /= num_pairs
        if return_alignment_loss:
            return fused, alignment_loss
        return fused

def setup_logger(log_dir, log_filename="training_log"):
    # 创建logger对象
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 创建文件处理器，将日志写入指定文件
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)  # 设置处理器日志级别

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger中
    logger.addHandler(file_handler)

    return logger

class RecommendationSystem(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.model_cfg = config.config
        self.logger = logger
        self.device = torch.device('cuda:0' if GPU else 'cpu')
        
        # 获取数据集名称
        self.dataset_name = self.model_cfg.get("dataset", "default")
        
        # 创建日志和结果目录
        self.log_dir = f"log/{self.dataset_name}"
        self.result_dir = f"result/{self.dataset_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 设置日志文件路径
        self.log_path = os.path.join(self.log_dir, "training_log")
        self.result_path = os.path.join(self.result_dir, "results.txt")
        self.profile_emb = load_sci_embeddings(self.model_cfg.get("reviewer_profile_bert"))
        self.authored_emb = load_sci_embeddings(self.model_cfg.get("reviewer_write_bert"))
        self.reviewed_emb = load_sci_embeddings(self.model_cfg.get("reviewer_review_bert"))
        self.sub_emb = load_sci_embeddings(self.model_cfg.get("submission_bert"))

        self.training_set = load_data(self.model_cfg.get("train"))
        self.val_set = load_data(self.model_cfg.get("valid"))
        self.test_set = load_data(self.model_cfg.get("test"))
        self.data = Interaction(self.model_cfg, self.training_set, self.val_set, self.test_set)

        self.bestPerformance = []
        self.rerank_bestPerformance = []
        self.item_attn_fuser = AttentionFusion(['profile', 'authored', 'reviewed', 'light_item'], 
                                              emb_dim=self.model_cfg.get('proj_dim', 256)).to(self.device)
        self.user_attn_fuser = AttentionFusion(['submission', 'light_user'], 
                                              emb_dim=self.model_cfg.get('proj_dim', 256)).to(self.device)

        self.lambda_a = self.model_cfg.get('lambda_a', 0.1)
        self.lambda_b = self.model_cfg.get('lambda_b', 100)
        self.best_user_emb = None
        self.best_item_emb = None
        self.batchsize =  self.model_cfg.get('batchsize', 128)
        self.k =  self.model_cfg.get('k', 1)

        input_dims = {
            'profile': self.model_cfg.get('sen_dim'),
            'authored': self.model_cfg.get('sen_dim'),
            'reviewed': self.model_cfg.get('sen_dim'),
            'submission': self.model_cfg.get('sen_dim'),
            'light_user': self.model_cfg.get('str_dim'),
            'light_item': self.model_cfg.get('str_dim'),
        }

        self.contrastive = MoCoMultiViewContrastive(
            input_dims,
            proj_dim=self.model_cfg.get('proj_dim', 256),
            temperature=self.model_cfg.get('temperature', 0.5),
            momentum=self.model_cfg.get('momentum', 0.999),
            queue_size=self.model_cfg.get('queue_size', 1024)
        ).to(self.device)

        self.lightgcn = LightGCN(self.model_cfg, logger, self.data).to(self.device)

        self.f = nn.Sigmoid()
        self.feat_dropout = nn.Dropout(p=self.model_cfg.get('feat_dropout', 0.1))

        self.item_weights = nn.ParameterDict({
            'profile': nn.Parameter(torch.ones(1)),
            'authored': nn.Parameter(torch.ones(1)),
            'reviewed': nn.Parameter(torch.ones(1)),
            'light_item': nn.Parameter(torch.ones(1))
        })
        self.user_weights = nn.ParameterDict({
            'submission': nn.Parameter(torch.ones(1)),
            'light_user': nn.Parameter(torch.ones(1))
        })

        self.max_N = max(int(x) for x in self.model_cfg['item.ranking.topN'])
        self.N = self.model_cfg['item.ranking.topN']
        
        # 记录配置
        self._log_config()

    def _log_config(self):
        """记录配置信息到日志文件和结果文件"""
        # 记录到日志
        self.logger.info("\n=== Configuration ===")
        for key, value in self.model_cfg.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=====================")

        # 记录到 results 文件
        with open(self.result_path, "a") as f:
            f.write("\n=== Configuration ===\n")
            for key, value in self.model_cfg.items():
                f.write(f"{key}: {value}\n")
            f.write("=====================\n")


    def create_optimizer(self):
        return torch.optim.Adam([
            {'params': self.lightgcn.parameters(), 'lr': self.model_cfg['LightGCN_lr']},
            {'params': self.contrastive.parameters(), 'lr': self.model_cfg['contrastive_lr']},
            {'params': self.user_weights.parameters(), 'lr': self.model_cfg.get('view_weight_lr', 0.01)},
            {'params': self.item_weights.parameters(), 'lr': self.model_cfg.get('view_weight_lr', 0.01)},
        ], weight_decay=1e-5)

    def fuse_user_views(self, submission, light, return_alignment=False):
        return self.user_attn_fuser({
            'submission': submission.to(self.device),
            'light_user': light.to(self.device)
        }, return_alignment_loss=return_alignment)

    def fuse_item_views(self, profile, authored, reviewed, light, return_alignment=False):
        return self.item_attn_fuser({
            'profile': profile.to(self.device),
            'authored': authored.to(self.device),
            'reviewed': reviewed.to(self.device),
            'light_item': light.to(self.device)
        }, return_alignment_loss=return_alignment)

    def train_model(self, epochs):
        optim = self.create_optimizer()
        self.train()

        submission2id = {sid: idx for idx, sid in enumerate(self.data.user)}
        reviewer2id = {rid: idx for idx, rid in enumerate(self.data.item)}

        dataset = ReviewSubmissionDataset(
            train_file=self.model_cfg.get("train"),
            profile_emb=self.profile_emb,
            authored_emb=self.authored_emb,
            reviewed_emb=self.reviewed_emb,
            sub_emb=self.sub_emb
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batchsize,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, submission2id, reviewer2id)
        )
        
        # 训练进度条
        pbar = tqdm(range(1, epochs + 1), desc="Training Progress")
        
        for ep in pbar:
            self.zero_grad()
            l1, submission_emb, reviewer_emb = self.lightgcn()
            submission_emb = submission_emb.to(self.device)
            reviewer_emb = reviewer_emb.to(self.device)

            total_batch_loss = 0.0
            total_l2 = 0.0
            total_rec_loss = 0.0

            all_submission_embs = torch.zeros((len(self.data.user), self.model_cfg.get('proj_dim', 256))).to(self.device)
            all_reviewer_embs = torch.zeros((len(self.data.item), self.model_cfg.get('proj_dim', 256))).to(self.device)

            seen_submission_ids = set()
            seen_reviewer_ids = set()

            for batch in loader:
                optim.zero_grad()

                batch_sub_ids = batch['submission_id'].to(self.device)
                batch_rev_ids = batch['reviewer_id'].to(self.device)

                views_q = {
                    'profile': batch['profile'].to(self.device),
                    'authored': batch['authored'].to(self.device),
                    'reviewed': batch['reviewed'].to(self.device),
                    'submission': batch['submission'].to(self.device),
                    'light_user': submission_emb[batch_sub_ids],
                    'light_item': reviewer_emb[batch_rev_ids]
                }
               
                views_k = {k: self.feat_dropout(v) for k, v in views_q.items()}
                l2, views_q_proj = self.contrastive(views_q, views_k)
                views_q_proj = {k: v for k, v in views_q_proj.items()}

                fused_submission_emb = self.fuse_user_views(
                    views_q_proj['submission'], views_q_proj['light_user'], return_alignment=False
                )
                fused_reviewer_emb = self.fuse_item_views(
                    views_q_proj['profile'], views_q_proj['authored'], views_q_proj['reviewed'], 
                    views_q_proj['light_item'], return_alignment=False
                )
                
                # ---------- 硬负样本采样 ----------
                B = fused_submission_emb.size(0)
                pos_mask = torch.zeros(B, B, dtype=torch.bool, device=self.device)
                pos_mask = batch_rev_ids.unsqueeze(0) == batch_rev_ids.unsqueeze(1)
                sim_matrix = torch.matmul(fused_submission_emb, fused_reviewer_emb.t())  # [B, B]
                sim_pos = torch.where(pos_mask, sim_matrix, torch.tensor(0.0, device=self.device))  # [B, B]
                pos_count = pos_mask.sum(dim=1).clamp(min=1)
                l_pos = sim_pos.sum(dim=1) / pos_count

                # 负样本：屏蔽正样本
                sim_matrix[pos_mask] = -1e9
                topk_neg_score, _ = torch.topk(sim_matrix, k=self.k, dim=1)  # [B, 5]

                # rec loss
                pos_score = l_pos.unsqueeze(1).expand_as(topk_neg_score)
                rec_loss = -torch.mean(torch.log(self.f(pos_score - topk_neg_score) + 1e-8))
                loss = self.lambda_a * l2 + rec_loss

                total_batch_loss += loss
                total_l2 += l2
                total_rec_loss += rec_loss
                
                if ep % 10 == 0:
                    for i, sid in enumerate(batch_sub_ids):
                        all_submission_embs[sid] = fused_submission_emb[i]
                        seen_submission_ids.add(sid.item())
                    for i, rid in enumerate(batch_rev_ids):
                        all_reviewer_embs[rid] = fused_reviewer_emb[i]
                        seen_reviewer_ids.add(rid.item())

            # 追加 LGCN loss
            final_loss = total_batch_loss + l1 * self.lambda_b
            final_loss.backward()
            optim.step()

            # 更新进度条描述
            pbar.set_description(f"Epoch {ep}: Loss {final_loss.item():.4f}")
            
            # 记录损失
            self.logger.info(f"Epoch {ep}: Loss {final_loss.item():.4f}, LGCN loss {l1.item():.6f}, Contrastive loss {total_l2.item():.6f}, Rec loss {total_rec_loss.item():.6f}")
            
            # 每10个epoch进行一次评估
            if ep % 10 == 0:
                self.fast_evaluation(ep, all_submission_embs, all_reviewer_embs)

        # 训练结束后进行最终测试
        # self.final_test()
        self.logger.info(f"Training completed. Logs saved to: {self.log_dir}/training_log.txt")

    def predict(self, u, user_emb, item_emb): 
        user_id = self.data.get_user_id(u)
        submission_emb = user_emb[user_id].clone().detach().float().to(self.device) 
        rating = self.f(torch.matmul(submission_emb, item_emb.t()))
        return rating

    def test(self, user_emb, item_emb):
        rec_list = {}
        user_count = len(self.data.test_set)  
        for user in tqdm(self.data.test_set, total=user_count, desc="Processing users"):
            candidates = self.predict(user, user_emb, item_emb)
            rated_list, _ = self.data.user_rated(user)  
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8 
            if isinstance(candidates, torch.Tensor):
                candidates_np = candidates.detach().cpu().numpy().ravel() 
            else:
                candidates_np = candidates.ravel()  
            ids, scores = find_k_largest(self.max_N, candidates_np)    
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
        return rec_list
        
    def rerank_test(self, user_emb, item_emb):
        rec_list = {}
        user_emb = self.best_user_emb
        item_emb = self.best_item_emb
        user_count = len(self.data.test_set)  

        # Process each user in the test set
        for user in tqdm(self.data.test_set, total=user_count, desc="Processing users"):
            candidates = self.predict(user, user_emb, item_emb)
            rated_list, _ = self.data.user_rated(user)  

            for item in rated_list:
                candidates[self.data.item[item]] = -10e8 

            candidates_np = candidates.detach().cpu().numpy().ravel() if isinstance(candidates, torch.Tensor) else candidates.ravel()
            ids, scores = find_k_largest(self.max_N, candidates_np)    
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))

        rerank_inputs_all = []
        for user in self.data.test_set:
            rerank_input = self.collect_rerank_input(
                user_id=user,
                rec_items=rec_list[user],
                reviewer_file=self.model_cfg.get("reviewer"),
                submission_file=self.model_cfg.get("submission"),
                max_candidates=20
            )
            if rerank_input:
                rerank_inputs_all.append(rerank_input)       

        measure = ranking_evaluation(self.data.test_set, rec_list, self.N)
        self.logger.info(f'*Load Performance*: {", ".join([m.strip() for m in  measure])}')  # 保存原始字符串

        rerank_inputs_list = asyncio.run(rerank.run_rerank_with_gpt_batch(rerank_inputs_all))
        # Perform ranking evaluation
        measure = ranking_evaluation(self.data.test_set, rec_list, self.N)
        rerank_measure = ranking_evaluation(self.data.test_set, rerank_inputs_list, self.N)

        # Log performance
        performance = measure
        rerank_performance = rerank_measure

        self.logger.info(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        self.logger.info(f'*Load Performance*: {", ".join([m.strip() for m in performance])}')  # 保存原始字符串
        self.logger.info(f'*Rerank Performance*: {", ".join([m.strip() for m in rerank_performance])}')  # 保存原始字符串

        # 保存结果到文件
        with open(self.result_path, "a") as f:
            f.write("\n=== Final Test Results ===\n")
            f.write(f'*Load Performance*: {", ".join([m.strip() for m in performance])}\n')  # 写入原始格式
            f.write(f"*Rerank Performance*: {', '.join([m.strip() for m in rerank_performance])}\n")

        # 返回结果
        return rec_list
    

    def collect_rerank_input(self, user_id, rec_items, reviewer_file, submission_file, max_candidates=20):
        with open(reviewer_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        reviewer_map = {item["reviewer_id"]: item for item in data}

        with open(submission_file, "r", encoding="utf-8") as f:
            submission_map = {json.loads(line)["paper"]: json.loads(line) for line in f}

        submission = submission_map.get(user_id)
        if not submission:
            return None
        # print(" rec_items:", rec_items)
        candidates = []
        for reviewer_id, scores in rec_items[:max_candidates]:
            reviewer = reviewer_map.get(reviewer_id)
            if reviewer:
                candidates.append({
                    "reviewer_id": reviewer["reviewer_id"],
                    "scores":scores,
                    "specialty": reviewer.get("specialty", []),
                    "user_profile": reviewer.get("user_profile", ""),
                    "publication":reviewer.get("user_profile", "")
                })

        if not candidates:
            return None
        return {
            "user_id": user_id,
            "submission": {
                "title": submission.get("title", ""),
                "abstract": submission.get("abstract", ""),
            },
            "candidates": candidates
        }

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.val_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.val_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch, user_emb, item_emb):
        rec_list = self.test(user_emb, item_emb)
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {}
        for metric_line in measure[1:]:
            parts = metric_line.strip().split(':')
            if len(parts) == 2:
                performance[parts[0]] = float(parts[1])

        if not self.bestPerformance:
            self.bestPerformance = [epoch, performance]
            self.best_user_emb = user_emb.clone().detach()
            self.best_item_emb = item_emb.clone().detach()
            self._save_best_model(epoch)
        else:
            current_score = sum(performance.values())
            best_score = sum(self.bestPerformance[1].values())
            if current_score > best_score:
                self.bestPerformance = [epoch, performance]
                self.best_user_emb = user_emb.clone().detach()
                self.best_item_emb = item_emb.clone().detach()
                self._save_best_model(epoch)

        self.logger.info(f'Evaluation at Epoch {epoch} (Top-{self.max_N})')
        self.logger.info(f'Current: {", ".join([f"{k}: {v:.4f}" for k, v in performance.items()])}')
        self.logger.info(f'Best (Epoch {self.bestPerformance[0]}): {", ".join([f"{k}: {v:.4f}" for k, v in self.bestPerformance[1].items()])}')

        # Save evaluation results
        with open(self.result_path, "a") as f:
            f.write("\n=== Test Results ===\n")
            f.write(f"*Best Performance*: Epoch {self.bestPerformance[0]}, {', '.join([f'{k}: {v:.4f}' for k, v in self.bestPerformance[1].items()])}\n")
            f.write(f"*Current Performance*: {', '.join([f'{k}: {v:.4f}' for k, v in performance.items()])}\n")

        return performance

    def _save_best_model(self, epoch):
        # Save the best embeddings
        torch.save(self.best_user_emb, os.path.join(self.result_dir, f'best_user_emb_{self.dataset_name}.pth'))
        torch.save(self.best_item_emb, os.path.join(self.result_dir, f'best_item_emb_{self.dataset_name}.pth'))

        # Save the model configuration
        config_path = os.path.join(self.result_dir, f'config_{self.dataset_name}.json')
        with open(config_path, "w") as f:
            json.dump(self.model_cfg, f, indent=4)

        # Save the best model parameters (state_dict)
        model_path = os.path.join(self.result_dir, f'best_model_{self.dataset_name}.pth')
        torch.save(self.state_dict(), model_path)  # Save the entire model state_dict


    def final_test(self):
        """训练结束后使用最佳模型进行测试"""
        user_emb_path = os.path.join(self.result_dir, f'best_user_emb_{self.dataset_name}.pth')
        item_emb_path = os.path.join(self.result_dir, f'best_item_emb_{self.dataset_name}.pth')

        # 检查文件是否存在并加载嵌入
        if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
            print("\n=== Running Final Test with Best Model ===")
            
            # 加载最佳嵌入
            self.best_user_emb = torch.load(user_emb_path)
            self.best_item_emb = torch.load(item_emb_path)

            # 运行重新排序测试
            self.rerank_test(self.best_user_emb, self.best_item_emb)
        else:
            print("Warning: No best model found for final test")


def get_logger(name, log_dir):
    """创建并配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # 从命令行参数获取配置文件名
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--epochs", type=int, required=True, help="train epoch")
    args = parser.parse_args()
    
    # 加载配置
    config = ModelConf(args.config)
    
    # 获取数据集名称
    dataset_name = config.config.get("dataset", "default")
    
    # 创建日志记录器
    logger = get_logger("RecSys", f"log/{dataset_name}")
    
    logger.info(f"Starting training for dataset: {dataset_name}")
    logger.info(f"Using configuration: {args.config}")
    
    model = RecommendationSystem(config, logger)
    model.train_model(epochs=args.epochs)
    model.final_test()