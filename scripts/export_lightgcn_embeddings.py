#!/usr/bin/env python3
"""
LightGCN Embedding 导出脚本

从训练好的 LightGCN 模型导出 item embeddings。

索引对齐：
- LightGCN 训练使用 0-based 索引
- 导出时在前面插入零向量，使得：
  - 第 0 行：零向量（padding/unknown）
  - 第 1 ~ N 行：对应 LightGCN 的 item 0 ~ N-1

使用方法：
    python scripts/export_lightgcn_embeddings.py --data_path rec_datasets --dataset Beauty
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_embeddings(
    data_path: str,
    dataset: str,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    导出 LightGCN item embeddings
    
    Args:
        data_path: 数据目录路径
        dataset: 数据集名称
        model_path: 模型文件路径（可选，默认自动查找）
        output_path: 输出文件路径（可选）
    
    Returns:
        导出的 embedding 文件路径
    """
    dataset_path = os.path.join(data_path, dataset)
    lightgcn_data_dir = os.path.join(dataset_path, 'lightgcn_data')
    
    # 查找模型文件
    if model_path is None:
        # 自动查找最新的模型文件
        model_files = [f for f in os.listdir(lightgcn_data_dir) if f.startswith('lightgcn_model') and f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(f"No LightGCN model found in {lightgcn_data_dir}")
        model_path = os.path.join(lightgcn_data_dir, sorted(model_files)[-1])
    
    logger.info(f"Loading model from: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    n_users = checkpoint['n_users']
    n_items = checkpoint['n_items']
    embedding_dim = checkpoint['embedding_dim']
    n_layers = checkpoint['n_layers']
    
    logger.info(f"Model info:")
    logger.info(f"  - Users: {n_users}")
    logger.info(f"  - Items: {n_items}")
    logger.info(f"  - Embedding dim: {embedding_dim}")
    logger.info(f"  - Layers: {n_layers}")
    logger.info(f"  - Best Recall@20: {checkpoint.get('recall', 'N/A')}")
    
    # 重建模型以获取 embedding
    from train_lightgcn import LightGCN, GRAMLightGCNDataset
    
    # 加载数据集以获取图
    data = GRAMLightGCNDataset(data_path, dataset)
    graph = data.Graph
    
    # 创建模型并加载权重
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取多层聚合后的 embedding
    with torch.no_grad():
        users_emb, items_emb = model.computer(graph)
    
    # items_emb: [n_items, embedding_dim]，索引 0 ~ n_items-1
    items_emb = items_emb.cpu()
    
    logger.info(f"Raw item embeddings shape: {items_emb.shape}")
    
    # 在前面插入零向量（索引 0 用于 padding/unknown）
    zero_vector = torch.zeros(1, embedding_dim)
    embeddings_with_padding = torch.cat([zero_vector, items_emb], dim=0)
    
    # 验证
    assert embeddings_with_padding.shape[0] == n_items + 1, \
        f"Expected {n_items + 1} rows, got {embeddings_with_padding.shape[0]}"
    assert torch.allclose(embeddings_with_padding[0], torch.zeros(embedding_dim)), \
        "Row 0 should be zero vector"
    
    # 验证与映射文件的一致性
    mapping_path = os.path.join(dataset_path, 'item_id_to_gcn_index.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            item_mapping = json.load(f)
        max_gcn_index = max(item_mapping.values())
        
        if max_gcn_index >= embeddings_with_padding.shape[0]:
            logger.warning(
                f"Mapping file has max index {max_gcn_index}, "
                f"but embedding has only {embeddings_with_padding.shape[0]} rows"
            )
        else:
            logger.info(f"Mapping file validation passed (max index: {max_gcn_index})")
    
    # 保存
    if output_path is None:
        output_path = os.path.join(lightgcn_data_dir, 'lightgcn_item_embeddings.pt')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_with_padding, output_path)
    
    logger.info("=" * 50)
    logger.info("Export Summary:")
    logger.info(f"  - Output file: {output_path}")
    logger.info(f"  - Embedding shape: {embeddings_with_padding.shape}")
    logger.info(f"  - Row 0: zero vector (padding/unknown)")
    logger.info(f"  - Rows 1-{n_items}: item embeddings (LightGCN index 0 to {n_items-1})")
    logger.info("=" * 50)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export LightGCN item embeddings')
    parser.add_argument('--data_path', type=str, default='rec_datasets',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='Beauty',
                        help='Dataset name')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (optional)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for embeddings (optional)')
    
    args = parser.parse_args()
    
    export_embeddings(
        data_path=args.data_path,
        dataset=args.dataset,
        model_path=args.model_path,
        output_path=args.output_path
    )


if __name__ == '__main__':
    main()
