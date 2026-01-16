#!/usr/bin/env python3
"""
LightGCN 数据转换脚本

将 GRAM 的 user_sequence.txt 转换为 LightGCN 训练格式。

索引策略：
- LightGCN 训练文件（train.txt, test.txt）使用 0-based 索引
- item_id_to_gcn_index.json 使用 1-based 索引（0 保留给 padding/unknown）
- 映射关系：gcn_index = lightgcn_index + 1

数据拆分（与 GRAM 一致）：
- train: items[:-2]
- validation: items[-2]（不写入文件，由 GRAM 自行处理）
- test: items[-1]
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGCNDataConverter:
    """将 GRAM 数据格式转换为 LightGCN 格式"""
    
    def __init__(self, data_path: str, dataset: str):
        """
        Args:
            data_path: 数据根目录（如 rec_datasets）
            dataset: 数据集名称（如 Beauty）
        """
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_path = os.path.join(data_path, dataset)
        
        self.user_sequences: Dict[str, List[str]] = {}  # user_id -> [item_id1, item_id2, ...]
        self.item_to_lightgcn_index: Dict[str, int] = {}  # raw_item_id -> lightgcn_index (0-based)
        self.user_to_index: Dict[str, int] = {}  # raw_user_id -> user_index (0-based)
        
        # 统计信息
        self.num_train_interactions = 0
        self.num_test_interactions = 0
        self.skipped_users = 0
    
    def load_user_sequences(self) -> None:
        """从 user_sequence.txt 加载用户序列"""
        seq_file = os.path.join(self.dataset_path, 'user_sequence.txt')
        
        if not os.path.exists(seq_file):
            raise FileNotFoundError(f"User sequence file not found: {seq_file}")
        
        with open(seq_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Line {line_num}: Invalid format, skipping")
                    continue
                
                user_id = parts[0]
                item_ids = parts[1:]
                
                # 序列长度至少为 3（train 至少 1 个，validation 1 个，test 1 个）
                if len(item_ids) < 3:
                    logger.warning(f"Line {line_num}: User {user_id} has less than 3 items, skipping")
                    self.skipped_users += 1
                    continue
                
                self.user_sequences[user_id] = item_ids
        
        logger.info(f"Loaded {len(self.user_sequences)} users from {seq_file}")
        if self.skipped_users > 0:
            logger.warning(f"Skipped {self.skipped_users} users with less than 3 items")
    
    def build_index_mappings(self) -> None:
        """
        构建 item 和 user 的索引映射
        
        Note:
            - LightGCN 训练使用 0-based 索引
            - 导出到 GRAM 时，item 索引 +1（0 保留给 padding）
        """
        # 收集所有唯一的 items（只从训练集收集，确保一致性）
        all_items = set()
        for user_id, items in self.user_sequences.items():
            # 训练集：items[:-2]
            train_items = items[:-2]
            all_items.update(train_items)
            # 也包含 test item，确保所有 item 都有映射
            all_items.add(items[-1])
        
        # 构建 item 映射（0-based for LightGCN）
        for idx, item_id in enumerate(sorted(all_items)):
            self.item_to_lightgcn_index[item_id] = idx
        
        # 构建 user 映射（0-based）
        for idx, user_id in enumerate(sorted(self.user_sequences.keys())):
            self.user_to_index[user_id] = idx
        
        logger.info(f"Built mappings: {len(self.item_to_lightgcn_index)} items, {len(self.user_to_index)} users")
    
    def convert_to_lightgcn_format(self) -> Tuple[str, str]:
        """
        转换为 LightGCN 格式
        
        Returns:
            train_path: train.txt 路径
            test_path: test.txt 路径
            
        Note:
            - train.txt 包含 items[:-2]（与 GRAM 训练集一致）
            - test.txt 包含 items[-1]（与 GRAM 测试集一致）
            - 验证集 items[-2] 不写入文件，由 GRAM 自行处理
        """
        # 创建 LightGCN 数据目录
        lightgcn_data_dir = os.path.join(self.dataset_path, 'lightgcn_data')
        os.makedirs(lightgcn_data_dir, exist_ok=True)
        
        train_path = os.path.join(lightgcn_data_dir, 'train.txt')
        test_path = os.path.join(lightgcn_data_dir, 'test.txt')
        
        # 写入 train.txt
        with open(train_path, 'w', encoding='utf-8') as f:
            for user_id in sorted(self.user_sequences.keys()):
                items = self.user_sequences[user_id]
                train_items = items[:-2]  # 与 GRAM 一致
                
                user_idx = self.user_to_index[user_id]
                item_indices = []
                for item_id in train_items:
                    if item_id in self.item_to_lightgcn_index:
                        item_indices.append(str(self.item_to_lightgcn_index[item_id]))
                
                if item_indices:
                    f.write(f"{user_idx} {' '.join(item_indices)}\n")
                    self.num_train_interactions += len(item_indices)
        
        # 写入 test.txt
        with open(test_path, 'w', encoding='utf-8') as f:
            for user_id in sorted(self.user_sequences.keys()):
                items = self.user_sequences[user_id]
                test_item = items[-1]  # 与 GRAM 一致
                
                user_idx = self.user_to_index[user_id]
                if test_item in self.item_to_lightgcn_index:
                    item_idx = self.item_to_lightgcn_index[test_item]
                    f.write(f"{user_idx} {item_idx}\n")
                    self.num_test_interactions += 1
        
        logger.info(f"Generated train.txt: {self.num_train_interactions} interactions")
        logger.info(f"Generated test.txt: {self.num_test_interactions} interactions")
        
        return train_path, test_path
    
    def save_mappings(self) -> Tuple[str, str]:
        """
        保存索引映射文件
        
        Returns:
            item_mapping_path: item_id_to_gcn_index.json 路径（索引从 1 开始）
            user_mapping_path: user_id_to_index.json 路径（索引从 0 开始）
            
        Note:
            - item 索引在保存时 +1，使得 0 保留给 padding/unknown
            - user 索引保持 0-based
        """
        # item 映射：LightGCN index + 1 = GCN index（1-based）
        item_id_to_gcn_index = {
            item_id: lightgcn_idx + 1 
            for item_id, lightgcn_idx in self.item_to_lightgcn_index.items()
        }
        
        item_mapping_path = os.path.join(self.dataset_path, 'item_id_to_gcn_index.json')
        with open(item_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(item_id_to_gcn_index, f, indent=2)
        
        user_mapping_path = os.path.join(self.dataset_path, 'user_id_to_index.json')
        with open(user_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.user_to_index, f, indent=2)
        
        logger.info(f"Saved item mapping to {item_mapping_path}")
        logger.info(f"  - Max GCN index: {max(item_id_to_gcn_index.values())}")
        logger.info(f"  - Index range: 1 to {len(item_id_to_gcn_index)} (0 reserved for padding)")
        logger.info(f"Saved user mapping to {user_mapping_path}")
        
        return item_mapping_path, user_mapping_path
    
    def convert(self) -> None:
        """执行完整的转换流程"""
        logger.info(f"Starting conversion for dataset: {self.dataset}")
        logger.info(f"Dataset path: {self.dataset_path}")
        
        # Step 1: 加载用户序列
        self.load_user_sequences()
        
        # Step 2: 构建索引映射
        self.build_index_mappings()
        
        # Step 3: 转换为 LightGCN 格式
        train_path, test_path = self.convert_to_lightgcn_format()
        
        # Step 4: 保存映射文件
        item_mapping_path, user_mapping_path = self.save_mappings()
        
        # 打印摘要
        logger.info("=" * 50)
        logger.info("Conversion Summary:")
        logger.info(f"  - Users: {len(self.user_sequences)}")
        logger.info(f"  - Items: {len(self.item_to_lightgcn_index)}")
        logger.info(f"  - Train interactions: {self.num_train_interactions}")
        logger.info(f"  - Test interactions: {self.num_test_interactions}")
        logger.info(f"  - Skipped users: {self.skipped_users}")
        logger.info("Output files:")
        logger.info(f"  - {train_path}")
        logger.info(f"  - {test_path}")
        logger.info(f"  - {item_mapping_path}")
        logger.info(f"  - {user_mapping_path}")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Convert GRAM data to LightGCN format')
    parser.add_argument('--data_path', type=str, default='rec_datasets',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='Beauty',
                        help='Dataset name')
    
    args = parser.parse_args()
    
    converter = LightGCNDataConverter(args.data_path, args.dataset)
    converter.convert()


if __name__ == '__main__':
    main()
