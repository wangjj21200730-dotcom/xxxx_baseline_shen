"""
GCN Embedding 加载和验证工具

用于 GRAM-C 的协同过滤信号处理
"""

import torch
import json
import pickle
import logging
import os
from typing import Dict, Optional, Tuple


def load_gcn_embeddings(
    gcn_emb_path: str,
    expected_dim: Optional[int] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    从文件加载 GCN Item Embeddings
    
    Args:
        gcn_emb_path: GCN embedding 文件路径 (.pt 或 .npy)
        expected_dim: 期望的 embedding 维度（用于验证）
        device: 目标设备
        
    Returns:
        torch.Tensor: shape [Num_Items, GCN_Dim]
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 维度不匹配或格式不支持
    """
    if not os.path.exists(gcn_emb_path):
        raise FileNotFoundError(f"GCN embedding file not found: {gcn_emb_path}")
    
    if gcn_emb_path.endswith('.pt'):
        gcn_embeddings = torch.load(gcn_emb_path, map_location=device)
    elif gcn_emb_path.endswith('.npy'):
        import numpy as np
        gcn_embeddings = torch.from_numpy(np.load(gcn_emb_path)).to(device)
    else:
        raise ValueError(f"Unsupported GCN embedding file format: {gcn_emb_path}")
    
    # 确保是 2D tensor
    if gcn_embeddings.dim() != 2:
        raise ValueError(
            f"GCN embeddings should be 2D tensor, got {gcn_embeddings.dim()}D"
        )
    
    # 验证维度
    if expected_dim is not None and gcn_embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"GCN embedding dimension mismatch: "
            f"expected {expected_dim}, got {gcn_embeddings.shape[1]}"
        )
    
    logging.info(f"Loaded GCN embeddings from {gcn_emb_path}")
    logging.info(f"  - Shape: {gcn_embeddings.shape}")
    logging.info(f"  - Dtype: {gcn_embeddings.dtype}")
    
    return gcn_embeddings


def load_item_id_mapping(mapping_path: str) -> Dict[str, int]:
    """
    加载 raw_item_id 到 GCN index 的映射表
    
    Args:
        mapping_path: 映射文件路径 (.json 或 .pkl)
        
    Returns:
        Dict[str, int]: raw_item_id -> gcn_index 映射
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式不支持
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Item ID mapping file not found: {mapping_path}")
    
    if mapping_path.endswith('.json'):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    elif mapping_path.endswith('.pkl'):
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
    else:
        raise ValueError(f"Unsupported mapping file format: {mapping_path}")
    
    # 验证映射格式
    if not isinstance(mapping, dict):
        raise ValueError(f"Mapping should be a dict, got {type(mapping)}")
    
    logging.info(f"Loaded item ID mapping from {mapping_path}")
    logging.info(f"  - Number of items: {len(mapping)}")
    
    return mapping


def validate_gcn_embeddings(
    gcn_embeddings: torch.Tensor,
    item_id_mapping: Dict[str, int],
    check_zero_row: bool = True
) -> Tuple[bool, str]:
    """
    验证 GCN embeddings 和 ID 映射的一致性
    
    Args:
        gcn_embeddings: GCN embedding 矩阵
        item_id_mapping: ID 映射表
        check_zero_row: 是否检查第 0 行为零向量
        
    Returns:
        Tuple[bool, str]: (是否通过验证, 错误信息)
    """
    num_embeddings = gcn_embeddings.shape[0]
    
    # 检查映射表中的最大索引
    if item_id_mapping:
        max_index = max(item_id_mapping.values())
        if max_index >= num_embeddings:
            return False, (
                f"Max index in mapping ({max_index}) exceeds "
                f"embedding matrix rows ({num_embeddings})"
            )
    
    # 检查第 0 行是否为零向量
    if check_zero_row:
        if not torch.allclose(gcn_embeddings[0], torch.zeros_like(gcn_embeddings[0])):
            return False, "Row 0 of GCN embeddings is not a zero vector"
    
    return True, "Validation passed"


def prepare_gcn_embeddings_with_padding(
    gcn_embeddings: torch.Tensor,
    ensure_zero_row: bool = True
) -> torch.Tensor:
    """
    准备 GCN embeddings，确保第 0 行为零向量（用于 padding/unknown）
    
    Args:
        gcn_embeddings: 原始 GCN embedding 矩阵
        ensure_zero_row: 是否确保第 0 行为零向量
        
    Returns:
        处理后的 GCN embedding 矩阵
    """
    if ensure_zero_row:
        # 检查第 0 行是否已经是零向量
        if not torch.allclose(gcn_embeddings[0], torch.zeros_like(gcn_embeddings[0])):
            logging.warning("Row 0 is not zero vector, forcing it to zero")
            gcn_embeddings = gcn_embeddings.clone()
            gcn_embeddings[0] = 0
    
    return gcn_embeddings


def create_dummy_gcn_embeddings(
    num_items: int,
    gcn_dim: int,
    save_path: Optional[str] = None
) -> torch.Tensor:
    """
    创建虚拟的 GCN embeddings（用于测试）
    
    Args:
        num_items: 物品数量
        gcn_dim: embedding 维度
        save_path: 保存路径（可选）
        
    Returns:
        torch.Tensor: shape [num_items + 1, gcn_dim]
    """
    # 创建随机 embeddings
    gcn_embeddings = torch.randn(num_items + 1, gcn_dim)
    
    # 第 0 行设为零向量
    gcn_embeddings[0] = 0
    
    # 归一化（可选，模拟真实 GCN 输出）
    gcn_embeddings[1:] = torch.nn.functional.normalize(gcn_embeddings[1:], dim=1)
    
    if save_path:
        torch.save(gcn_embeddings, save_path)
        logging.info(f"Saved dummy GCN embeddings to {save_path}")
    
    return gcn_embeddings


def create_dummy_item_id_mapping(
    item_ids: list,
    save_path: Optional[str] = None
) -> Dict[str, int]:
    """
    创建虚拟的 ID 映射表（用于测试）
    
    Args:
        item_ids: 物品 ID 列表
        save_path: 保存路径（可选）
        
    Returns:
        Dict[str, int]: raw_item_id -> gcn_index 映射
    """
    # 从 1 开始编号（0 保留给 unknown）
    mapping = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        logging.info(f"Saved dummy item ID mapping to {save_path}")
    
    return mapping
