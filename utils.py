"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import torch
from torch import nn
from torch import optim
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
import os
import json
import random
from torch.utils.data import Dataset, DataLoader
# ====================Concat Embedding=============================

def set_random_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 



def load_sci_embeddings(filepath):
 # 假设嵌入数据存储在 JSON 文件中
    with open(filepath, 'r') as f:
        data = json.load(f)
    embedding_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in data.items()}
    return embedding_dict


def load_data(train_file):
    training_data = []
    with open(train_file, 'r') as f:
        for line in f:
            data = line.strip().split()  # Assuming user and items are space-separated
            user = data[0]  # First column is user ID
            items = data[1:]  # Remaining columns are item IDs
            for item in items:
                training_data.append((user, item))  # Add each user-item pair to training data
    return training_data

class ReviewSubmissionDataset(Dataset):
    def __init__(self, train_file, profile_emb, authored_emb, reviewed_emb, sub_emb):
        """
        :param train_file: 训练文件路径，格式：submission_id reviewer1 reviewer2 ...
        :param *_emb: 多视图嵌入字典（预加载）
        """
        self.pairs = []  # (submission_id, reviewer_id)
        self.profile_emb = profile_emb
        self.authored_emb = authored_emb
        self.reviewed_emb = reviewed_emb
        self.sub_emb = sub_emb

        print(f"[Dataset Init] Loading from: {train_file}")

        with open(train_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                tokens = line.strip().split()
                if len(tokens) < 2:
                    print(f"[WARN] Skipping invalid line {line_num}: {line.strip()}")
                    continue
                submission_id = tokens[0]
                reviewer_ids = tokens[1:]
                for r_id in reviewer_ids:
                    self.pairs.append((submission_id, r_id))

        print(f"[Dataset Init] Loaded {len(self.pairs)} (submission, reviewer) pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s_id, r_id = self.pairs[idx]
        emb_dim=768
        # print("self.profile_emb[r_id]:",self.profile_emb[r_id])
        return {
            'submission_id': s_id,
            'reviewer_id': r_id,
            'profile': self.profile_emb[r_id] if r_id in self.profile_emb else torch.zeros(emb_dim),
            'authored': self.authored_emb[r_id] if r_id in self.authored_emb else torch.zeros(emb_dim),
            'reviewed': self.reviewed_emb[r_id] if r_id in self.reviewed_emb else torch.zeros(emb_dim),
            'submission': self.sub_emb[s_id] if s_id in self.sub_emb else torch.zeros(emb_dim),

        }


def collate_fn(batch, submission2id, reviewer2id):
    profile = torch.stack([x['profile'] for x in batch])
    authored = torch.stack([x['authored'] for x in batch])
    reviewed = torch.stack([x['reviewed'] for x in batch])
    submission = torch.stack([x['submission'] for x in batch])

    submission_ids = [submission2id[x['submission_id']] for x in batch]
    reviewer_ids = [reviewer2id[x['reviewer_id']] for x in batch]

    return {
        'profile': profile,
        'authored': authored,
        'reviewed': reviewed,
        'submission': submission,
        'submission_id': torch.tensor(submission_ids, dtype=torch.long),
        'reviewer_id': torch.tensor(reviewer_ids, dtype=torch.long)
    }        
