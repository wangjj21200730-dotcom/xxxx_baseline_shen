# GRAM_baseline_shen 快速启动

## 一键运行

### GRAM-C (协同增强版本) - 推荐

```bash
cd GRAM_baseline_shen
bash command/train_gram_c_beauty.sh
```

### 原始 GRAM (基线版本)

```bash
cd GRAM_baseline_shen
bash command/train_gram_beauty.sh
```

---

## 单卡训练（最简单）

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0 python src/main_generative_gram.py \
    --datasets Beauty \
    --distributed 0 \
    --gpu 0 \
    --rec_batch_size 16 \
    --rec_epochs 10 \
    --use_collaborative_prefix 1 \
    --gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt \
    --item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json \
    --hierarchical_id_type hierarchy_v1_c128_l7_len32768_split \
    --item_id_type split
```

---

## 多卡训练（加速）

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main_generative_gram.py \
    --datasets Beauty \
    --distributed 1 \
    --gpu 0,1,2,3 \
    --master_port 12345 \
    --rec_batch_size 32 \
    --rec_epochs 10 \
    --use_collaborative_prefix 1 \
    --gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt \
    --item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json \
    --hierarchical_id_type hierarchy_v1_c128_l7_len32768_split \
    --item_id_type split
```

---

## 快速测试（Debug 模式）

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0 python src/main_generative_gram.py \
    --datasets Beauty \
    --distributed 0 \
    --gpu 0 \
    --rec_batch_size 4 \
    --rec_epochs 1 \
    --debug_train_100 1 \
    --debug_test_100 1 \
    --use_collaborative_prefix 1 \
    --gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt \
    --item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json \
    --hierarchical_id_type hierarchy_v1_c128_l7_len32768_split \
    --item_id_type split
```

---

## 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--use_collaborative_prefix` | 启用协同前缀 | 1 (启用) |
| `--gcn_dim` | GCN embedding 维度 | 64 |
| `--recent_k` | 最近 K 个物品 | 5 |
| `--prefix_init_scale` | 前缀缩放因子 | 0.1 |
| `--prefix_dropout_prob` | 前缀 dropout | 0.2 |
| `--rec_batch_size` | 批次大小 | 16 (单卡), 32 (多卡) |
| `--rec_lr` | 学习率 | 1e-4 |

---

## 常见问题

### 1. CUDA Out of Memory
```bash
# 减小批次大小
--rec_batch_size 8
--gradient_accumulation_steps 4
```

### 2. 找不到 GCN 文件
```bash
# 禁用协同前缀
--use_collaborative_prefix 0
```

### 3. 找不到本地模型
```bash
# 使用在线模型（需要网络）
--backbone t5-small
```

---

## 输出位置

- **日志**: `log/${DATASET}_gram_c_${TIMESTAMP}/train.log`
- **模型**: `log/${DATASET}_gram_c_${TIMESTAMP}/checkpoint_epoch_*.pt`

---

详细文档请查看: [RUNNING_GUIDE.md](RUNNING_GUIDE.md)
