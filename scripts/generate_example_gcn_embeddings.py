#!/usr/bin/env python
"""
生成示例 GCN Embedding 文件用于 GRAM-C 测试

此脚本读取 item_plain_text.txt 获取所有 item IDs，
然后生成随机的 GCN embeddings 和 ID 映射文件。

使用方法:
    python scripts/generate_example_gcn_embeddings.py --dataset Beauty --gcn_dim 64

输出文件:
    - rec_datasets/{dataset}/gcn_item_embeddings.pt
    - rec_datasets/{dataset}/item_id_to_gcn_index.json
"""

import argparse
import json
import os
import torch
import numpy as np


def load_item_ids(data_path: str, dataset: str) -> list:
    """
    从 item_plain_text.txt 加载所有 item IDs
    
    文件格式: item_id \t plain_text
    """
    item_file = os.path.join(data_path, dataset, "item_plain_text.txt")
    
    if not os.path.exists(item_file):
        raise FileNotFoundError(f"Item file not found: {item_file}")
    
    item_ids = []
    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                item_ids.append(parts[0])
    
    print(f"Loaded {len(item_ids)} item IDs from {item_file}")
    return item_ids


def generate_gcn_embeddings(num_items: int, gcn_dim: int, seed: int = 42) -> torch.Tensor:
    """
    生成随机 GCN embeddings
    
    Args:
        num_items: 物品数量（不包括 padding）
        gcn_dim: GCN embedding 维度
        seed: 随机种子
        
    Returns:
        [num_items + 1, gcn_dim] 的 tensor，第 0 行为零向量
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建 embedding 矩阵，第 0 行预留为零向量
    embeddings = torch.zeros(num_items + 1, gcn_dim)
    
    # 生成随机 embeddings（从 1 开始）
    # 使用正态分布，模拟真实 GCN 输出
    embeddings[1:] = torch.randn(num_items, gcn_dim) * 0.1
    
    # L2 归一化（可选，但通常 GCN embeddings 会做归一化）
    embeddings[1:] = torch.nn.functional.normalize(embeddings[1:], p=2, dim=1)
    
    print(f"Generated GCN embeddings with shape: {embeddings.shape}")
    print(f"  - Row 0 (padding): {embeddings[0].sum().item()} (should be 0)")
    print(f"  - Row 1 norm: {embeddings[1].norm().item():.4f}")
    
    return embeddings


def create_id_mapping(item_ids: list) -> dict:
    """
    创建 item_id -> gcn_index 映射
    
    Args:
        item_ids: 原始 item ID 列表
        
    Returns:
        {item_id: gcn_index} 字典，gcn_index 从 1 开始
    """
    mapping = {}
    for idx, item_id in enumerate(item_ids):
        mapping[item_id] = idx + 1  # 从 1 开始，0 预留给 padding
    
    print(f"Created ID mapping with {len(mapping)} items")
    print(f"  - Index range: 1 ~ {len(mapping)}")
    
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Generate example GCN embeddings for GRAM-C")
    parser.add_argument("--data_path", type=str, default="rec_datasets",
                        help="Path to rec_datasets folder")
    parser.add_argument("--dataset", type=str, default="Beauty",
                        help="Dataset name (Beauty, Sports, Toys, Yelp)")
    parser.add_argument("--gcn_dim", type=int, default=64,
                        help="GCN embedding dimension")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 加载 item IDs
    item_ids = load_item_ids(args.data_path, args.dataset)
    
    # 生成 GCN embeddings
    embeddings = generate_gcn_embeddings(len(item_ids), args.gcn_dim, args.seed)
    
    # 创建 ID 映射
    id_mapping = create_id_mapping(item_ids)
    
    # 保存文件
    output_dir = os.path.join(args.data_path, args.dataset)
    
    # 保存 embeddings
    emb_path = os.path.join(output_dir, "gcn_item_embeddings.pt")
    torch.save(embeddings, emb_path)
    print(f"Saved GCN embeddings to: {emb_path}")
    
    # 保存 ID 映射
    mapping_path = os.path.join(output_dir, "item_id_to_gcn_index.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2)
    print(f"Saved ID mapping to: {mapping_path}")
    
    # 验证
    print("\n=== Verification ===")
    loaded_emb = torch.load(emb_path)
    with open(mapping_path, 'r') as f:
        loaded_mapping = json.load(f)
    
    print(f"Loaded embeddings shape: {loaded_emb.shape}")
    print(f"Loaded mapping size: {len(loaded_mapping)}")
    print(f"Row 0 is zero: {torch.allclose(loaded_emb[0], torch.zeros(args.gcn_dim))}")
    
    # 测试一个 item
    test_item = item_ids[0]
    test_idx = loaded_mapping[test_item]
    print(f"Test item '{test_item}' -> index {test_idx}")
    print(f"  Embedding norm: {loaded_emb[test_idx].norm().item():.4f}")


if __name__ == "__main__":
    main()
