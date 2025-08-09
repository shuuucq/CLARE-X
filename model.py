import torch
from torch import nn
import torch.nn.functional as F
from base.torch_interface import TorchGraphInterface
from collections import deque
class MoCoMultiViewContrastive(nn.Module):
    def __init__(self, input_dims: dict, proj_dim=256, temperature=0.1, momentum=0.999, queue_size=1024):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size

        self.encoder_q = nn.ModuleDict()
        self.encoder_k = nn.ModuleDict()
        for view, dim in input_dims.items():
            proj_q = nn.Sequential(
                        nn.Linear(dim, dim//2),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim//2, proj_dim)
                    )

            proj_k = nn.Sequential(
                        nn.Linear(dim, dim//2),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim//2 , proj_dim)
                    )
            self.encoder_q[view] = proj_q
            self.encoder_k[view] = proj_k

            for param_q, param_k in zip(proj_q.parameters(), proj_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        self.queues = {view: deque(maxlen=queue_size) for view in input_dims}

    @torch.no_grad()
    def _momentum_update_key_encoders(self):
        for view in self.encoder_q:
            for param_q, param_k in zip(self.encoder_q[view].parameters(), self.encoder_k[view].parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    def forward(self, views_q: dict, views_k: dict) -> torch.Tensor:
        batch_size = next(iter(views_q.values())).size(0)

        # 新增支持：包括 light_user 和 light_item
        contrastive_pairs = [
            # reviewer 多视图
            ("reviewed", "profile"),
            ("reviewed", "authored"),
            ("reviewed", "submission"),

            ("reviewed","reviewed"),
            ("profile", "profile"),
            ("authored", "authored"),
            ("submission", "submission"),

            # # 新增 LightGCN 输出对比
            ("light_user", "submission"),
            ("light_item", "reviewed"),
            ("light_user", "light_user"),
            ("light_item", "light_item")
        ]

        # 仅保留必要视图
        views = set(v for pair in contrastive_pairs for v in pair)
        views_q = {v: views_q[v] for v in views if v in views_q}
        views_k = {v: views_k[v] for v in views if v in views_k}

        # 编码器投影 + normalize
        q = {v: F.normalize(self.encoder_q[v](views_q[v]), dim=1) for v in views_q}

        with torch.no_grad():
            self._momentum_update_key_encoders()
            k = {v: F.normalize(self.encoder_k[v](views_k[v]), dim=1) for v in views_k}

        loss = 0.0
        pair_count = 0

        for view_q, view_k in contrastive_pairs:
            if view_q not in q or view_k not in k:
                continue

            qi = q[view_q]  # [B, D]
            ki = k[view_k]  # [B, D]

            l_pos = torch.einsum('nc,nc->n', [qi, ki])[:, None]

            # 当前 batch 负样本
            current_neg = ki
            if len(self.queues[view_k]) > 0:
                queue_neg = torch.stack(list(self.queues[view_k]), dim=0).to(qi.device)
                negatives = torch.cat([current_neg, queue_neg], dim=0)
            else:
                negatives = current_neg

            l_neg = torch.einsum('nc,kc->nk', [qi, negatives])  # [B, K]
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            labels = torch.zeros(batch_size, dtype=torch.long).to(qi.device)

            loss += F.cross_entropy(logits, labels)
            pair_count += 1

                # 更新负样本队列
            for view in k:
                self.queues[view].extend(k[view].detach().cpu())

        return loss / pair_count if pair_count > 0 else 0.0, q

class LightGCN(nn.Module):
    def __init__(self, conf, logger,data):
        super(LightGCN, self).__init__()
        self.config = conf
        self.data=data
        args = self.config['LightGCN']
        self.n_layers = int(args['n_layer'])
        # self.LightGCN_emb_size = int(args['embedding_dim'])  # 如果你忘记了
        self.norm_adj = self.data.norm_adj
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Graph = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(self.device)

        self.users_emb = nn.Embedding(num_embeddings=self.data.user_num, embedding_dim=int(args['LightGCN_emb_size']))
        self.items_emb = nn.Embedding(num_embeddings=self.data.item_num, embedding_dim=int(args['LightGCN_emb_size']))

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def computer(self):
        users_emb = self.users_emb.weight  # [n_user, D]
        items_emb = self.items_emb.weight  # [n_item, D]
        all_emb = torch.cat([users_emb, items_emb], dim=0)  # [n_user + n_item, D]

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)  # [n_user + n_item, n_layer+1, D]
        light_out = torch.mean(embs, dim=1)  # [n_user + n_item, D]

        user_all_embeddings = light_out[:self.data.user_num]
        item_all_embeddings = light_out[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
    def forward(self):
        user_emb, item_emb = self.computer()
        scores = torch.matmul(user_emb, item_emb.T)  # logits

        labels = torch.tensor(self.data.interaction_mat.toarray(), dtype=torch.float32, device=self.device)


        loss = nn.BCEWithLogitsLoss()(scores, labels)

        return loss, user_emb, item_emb

