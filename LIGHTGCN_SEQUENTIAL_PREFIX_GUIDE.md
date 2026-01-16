# LightGCN Sequential Prefix 使用指南

## 概述

本功能将 LightGCN 集成到 GRAM-C，并实现序列化协同前缀（Sequential Prefix）。核心改进是将最近 K 个 item 的 LightGCN embedding 分别通过 Adapter，得到 K 个独立的 Soft Tokens，让 T5 的 Self-Attention 机制自己学习哪些 item 更重要。

## 快速开始

### 1. 数据转换

将 GRAM 数据格式转换为 LightGCN 格式：

```bash
cd GRAM_baseline_shen
python scripts/convert_to_lightgcn.py --data_path rec_datasets --dataset Beauty
```

输出文件：
- `rec_datasets/Beauty/lightgcn_data/train.txt` - LightGCN 训练数据
- `rec_datasets/Beauty/lightgcn_data/test.txt` - LightGCN 测试数据
- `rec_datasets/Beauty/item_id_to_gcn_index.json` - Item ID 到 GCN 索引映射（1-based）
- `rec_datasets/Beauty/user_id_to_index.json` - User ID 到索引映射

### 2. 训练 LightGCN

```bash
python scripts/train_lightgcn.py \
    --data_path rec_datasets \
    --dataset Beauty \
    --embedding_dim 64 \
    --n_layers 3 \
    --epochs 1000 \
    --lr 0.001
```

模型保存到：`rec_datasets/Beauty/lightgcn_data/lightgcn_model_dim64_layer3.pt`

### 3. 导出 Embeddings

```bash
python scripts/export_lightgcn_embeddings.py --data_path rec_datasets --dataset Beauty
```

输出文件：`rec_datasets/Beauty/lightgcn_data/lightgcn_item_embeddings.pt`

### 4. 训练 GRAM-C with Sequential Prefix

```bash
bash command/train_gram_c_sequential_beauty.sh
```

## 关键参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--use_collaborative_prefix` | 1 | 启用协同前缀 |
| `--use_sequential_prefix` | 0 | 启用序列化前缀（1=K tokens, 0=1 token） |
| `--recent_k` | 5 | 使用最近 K 个物品 |
| `--lightgcn_emb_path` | "" | LightGCN embedding 文件路径 |
| `--gcn_item_emb_path` | "" | 回退的 GCN embedding 路径 |
| `--prefix_dropout_prob` | 0.2 | 整段 prefix dropout 概率 |

## 索引策略

- LightGCN 训练使用 **0-based** 索引（item 0 ~ N-1）
- `item_id_to_gcn_index.json` 使用 **1-based** 索引（0 保留给 padding）
- Embedding 文件：第 0 行是零向量，第 1~N 行对应 LightGCN item 0~N-1

## 两种模式对比

| 特性 | 均值池化模式 | 序列化模式 |
|-----|------------|----------|
| `use_sequential_prefix` | 0 | 1 |
| Prefix tokens 数量 | 1 | K |
| 池化方式 | Mean pooling | 无池化 |
| Dropout 方式 | 单 token | 整段 K tokens |
| 注意力学习 | 无 | T5 自己学习重要性 |

## 文件结构

```
GRAM_baseline_shen/
├── scripts/
│   ├── convert_to_lightgcn.py    # 数据转换
│   ├── train_lightgcn.py         # LightGCN 训练
│   └── export_lightgcn_embeddings.py  # Embedding 导出
├── command/
│   └── train_gram_c_sequential_beauty.sh  # 训练脚本
├── src/
│   ├── model/gram_c.py           # GRAM-C 模型（已修改）
│   ├── arguments.py              # 参数定义（已修改）
│   └── main_generative_gram.py   # 主训练脚本（已修改）
└── rec_datasets/Beauty/
    ├── lightgcn_data/
    │   ├── train.txt
    │   ├── test.txt
    │   ├── lightgcn_model_*.pt
    │   └── lightgcn_item_embeddings.pt
    ├── item_id_to_gcn_index.json
    └── user_id_to_index.json
```
