#!/usr/bin/env python3
"""
LightGCN 训练脚本

独立的 LightGCN 训练脚本，适配 GRAM 数据格式。
基于官方 LightGCN 实现，简化并适配到 GRAM 项目。

使用方法：
    python scripts/train_lightgcn.py --data_path rec_datasets --dataset Beauty
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class GRAMLightGCNDataset:
    """GRAM 格式的 LightGCN 数据集"""
    
    def __init__(self, data_path: str, dataset: str):
        self.data_path = os.path.join(data_path, dataset, 'lightgcn_data')
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"LightGCN data not found at {self.data_path}. "
                f"Please run convert_to_lightgcn.py first."
            )
        
        self.n_user = 0
        self.m_item = 0
        self.train_data_size = 0
        self.test_data_size = 0
        
        # 加载数据
        self._load_data()
        
        # 构建图
        self.Graph = None
        self._build_graph()
        
        logger.info(f"Dataset loaded: {self.n_user} users, {self.m_item} items")
        logger.info(f"Train interactions: {self.train_data_size}")
        logger.info(f"Test interactions: {self.test_data_size}")
    
    def _load_data(self):
        """加载训练和测试数据"""
        train_file = os.path.join(self.data_path, 'train.txt')
        test_file = os.path.join(self.data_path, 'test.txt')
        
        # 加载训练数据
        train_users, train_items = [], []
        self.train_user_items: Dict[int, List[int]] = {}
        
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user = int(parts[0])
                items = [int(i) for i in parts[1:]]
                
                self.train_user_items[user] = items
                for item in items:
                    train_users.append(user)
                    train_items.append(item)
                    self.n_user = max(self.n_user, user + 1)
                    self.m_item = max(self.m_item, item + 1)
        
        self.train_users = np.array(train_users)
        self.train_items = np.array(train_items)
        self.train_data_size = len(train_users)
        
        # 加载测试数据
        self.test_dict: Dict[int, List[int]] = {}
        
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user = int(parts[0])
                items = [int(i) for i in parts[1:]]
                
                self.test_dict[user] = items
                self.test_data_size += len(items)
                for item in items:
                    self.n_user = max(self.n_user, user + 1)
                    self.m_item = max(self.m_item, item + 1)
        
        # 构建 user-item 交互矩阵
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.train_users)), (self.train_users, self.train_items)),
            shape=(self.n_user, self.m_item)
        )
        
        # 预计算所有用户的正样本
        self._all_pos = [self.UserItemNet[u].nonzero()[1] for u in range(self.n_user)]
    
    def _build_graph(self):
        """构建归一化的邻接矩阵"""
        logger.info("Building adjacency matrix...")
        start = time()
        
        # 构建邻接矩阵
        adj_mat = sp.dok_matrix(
            (self.n_user + self.m_item, self.n_user + self.m_item), 
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.n_user, self.n_user:] = R
        adj_mat[self.n_user:, :self.n_user] = R.T
        adj_mat = adj_mat.todok()
        
        # 归一化
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        
        # 转换为 PyTorch sparse tensor
        coo = norm_adj.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        self.Graph = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
        logger.info(f"Adjacency matrix built in {time() - start:.2f}s")
    
    def get_user_pos_items(self, users: List[int]) -> List[np.ndarray]:
        """获取用户的正样本"""
        return [self._all_pos[u] for u in users]
    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item


class LightGCN(nn.Module):
    """LightGCN 模型"""
    
    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Embedding 层
        self.embedding_user = nn.Embedding(n_users, embedding_dim)
        self.embedding_item = nn.Embedding(n_items, embedding_dim)
        
        # 初始化
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        logger.info(f"LightGCN initialized:")
        logger.info(f"  - Users: {n_users}, Items: {n_items}")
        logger.info(f"  - Embedding dim: {embedding_dim}")
        logger.info(f"  - Layers: {n_layers}")
    
    def computer(self, graph: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        图卷积传播
        
        Returns:
            users_emb: [n_users, embedding_dim] 多层聚合后的用户 embedding
            items_emb: [n_items, embedding_dim] 多层聚合后的物品 embedding
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        
        # 多层 embedding 取平均
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def get_embedding(
        self, 
        graph: torch.sparse.FloatTensor,
        users: torch.Tensor, 
        pos_items: torch.Tensor, 
        neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """获取 BPR 训练所需的 embedding"""
        all_users, all_items = self.computer(graph)
        
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(
        self, 
        graph: torch.sparse.FloatTensor,
        users: torch.Tensor, 
        pos_items: torch.Tensor, 
        neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 BPR 损失"""
        (users_emb, pos_emb, neg_emb, 
         users_emb_ego, pos_emb_ego, neg_emb_ego) = self.get_embedding(
            graph, users, pos_items, neg_items
        )
        
        # L2 正则化损失
        reg_loss = (1/2) * (
            users_emb_ego.norm(2).pow(2) + 
            pos_emb_ego.norm(2).pow(2) + 
            neg_emb_ego.norm(2).pow(2)
        ) / float(len(users))
        
        # BPR 损失
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss


class BPRSampler:
    """BPR 负采样器"""
    
    def __init__(self, dataset: GRAMLightGCNDataset):
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.m_items
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """采样一个 batch 的 (user, pos_item, neg_item)"""
        users = np.random.randint(0, self.n_users, batch_size)
        pos_items = []
        neg_items = []
        
        for user in users:
            pos_list = self.dataset._all_pos[user]
            if len(pos_list) == 0:
                pos_items.append(0)
                neg_items.append(0)
                continue
            
            pos_item = pos_list[np.random.randint(0, len(pos_list))]
            pos_items.append(pos_item)
            
            # 负采样
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if neg_item not in pos_list:
                    break
            neg_items.append(neg_item)
        
        return np.array(users), np.array(pos_items), np.array(neg_items)


def evaluate(
    model: LightGCN, 
    dataset: GRAMLightGCNDataset, 
    graph: torch.sparse.FloatTensor,
    device: torch.device,
    k: int = 20
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    with torch.no_grad():
        users_emb, items_emb = model.computer(graph)
        
        # 计算所有用户的评分
        rating = torch.matmul(users_emb, items_emb.t())
        
        # 计算 Recall@K
        recall_sum = 0.0
        ndcg_sum = 0.0
        n_users = 0
        
        for user, test_items in dataset.test_dict.items():
            if user >= rating.shape[0]:
                continue
            
            user_rating = rating[user].cpu().numpy()
            
            # 排除训练集中的物品
            train_items = dataset._all_pos[user]
            user_rating[train_items] = -np.inf
            
            # 获取 top-k
            top_k_items = np.argsort(user_rating)[-k:][::-1]
            
            # 计算 Recall
            hits = len(set(top_k_items) & set(test_items))
            recall_sum += hits / min(len(test_items), k)
            
            # 计算 NDCG
            dcg = 0.0
            for i, item in enumerate(top_k_items):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            ndcg_sum += dcg / idcg if idcg > 0 else 0
            
            n_users += 1
    
    model.train()
    
    return {
        'recall': recall_sum / n_users if n_users > 0 else 0,
        'ndcg': ndcg_sum / n_users if n_users > 0 else 0
    }


def train_lightgcn(
    data_path: str,
    dataset: str,
    embedding_dim: int = 64,
    n_layers: int = 3,
    epochs: int = 1000,
    batch_size: int = 2048,
    lr: float = 0.001,
    reg: float = 1e-4,
    eval_every: int = 10,
    save_path: Optional[str] = None,
    seed: int = 2020
) -> LightGCN:
    """
    训练 LightGCN 模型
    
    Args:
        data_path: 数据目录路径
        dataset: 数据集名称
        embedding_dim: embedding 维度
        n_layers: GCN 层数
        epochs: 训练轮数
        batch_size: batch 大小
        lr: 学习率
        reg: L2 正则化系数
        eval_every: 每隔多少 epoch 评估一次
        save_path: 模型保存路径
        seed: 随机种子
    
    Returns:
        训练好的 LightGCN 模型
    """
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载数据
    data = GRAMLightGCNDataset(data_path, dataset)
    graph = data.Graph.to(device)
    
    # 创建模型
    model = LightGCN(
        n_users=data.n_users,
        n_items=data.m_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 采样器
    sampler = BPRSampler(data)
    
    # 训练
    best_recall = 0.0
    best_epoch = 0
    n_batches = data.train_data_size // batch_size + 1
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        
        for _ in range(n_batches):
            users, pos_items, neg_items = sampler.sample(batch_size)
            
            users = torch.LongTensor(users).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)
            neg_items = torch.LongTensor(neg_items).to(device)
            
            loss, reg_loss = model.bpr_loss(graph, users, pos_items, neg_items)
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            
            optimizer.zero_grad()
            (loss + reg * reg_loss).backward()
            optimizer.step()
        
        avg_loss = total_loss / n_batches
        avg_reg_loss = total_reg_loss / n_batches
        
        # 评估
        if epoch % eval_every == 0:
            metrics = evaluate(model, data, graph, device)
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {avg_loss:.4f} | Reg: {avg_reg_loss:.4f} | "
                f"Recall@20: {metrics['recall']:.4f} | NDCG@20: {metrics['ndcg']:.4f}"
            )
            
            # 保存最佳模型
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_epoch = epoch
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'recall': best_recall,
                        'n_users': data.n_users,
                        'n_items': data.m_items,
                        'embedding_dim': embedding_dim,
                        'n_layers': n_layers
                    }, save_path)
                    logger.info(f"Best model saved to {save_path}")
    
    logger.info(f"Training finished. Best Recall@20: {best_recall:.4f} at epoch {best_epoch}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train LightGCN for GRAM')
    parser.add_argument('--data_path', type=str, default='rec_datasets',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='Beauty',
                        help='Dataset name')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GCN layers')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--reg', type=float, default=1e-4,
                        help='L2 regularization weight')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 模型保存路径
    save_path = os.path.join(
        args.data_path, args.dataset, 'lightgcn_data', 
        f'lightgcn_model_dim{args.embedding_dim}_layer{args.n_layers}.pt'
    )
    
    train_lightgcn(
        data_path=args.data_path,
        dataset=args.dataset,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        eval_every=args.eval_every,
        save_path=save_path,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
