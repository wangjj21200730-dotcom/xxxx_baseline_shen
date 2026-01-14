# GRAM_baseline_shen 运行指南

## 目录结构验证

在运行之前，请确保以下文件和目录存在：

### ✅ 必需文件检查清单

```bash
# 1. 数据集文件
GRAM_baseline_shen/rec_datasets/Beauty/
├── user_sequence.txt                    # 用户交互序列
├── item_plain_text.txt                  # 物品文本信息
├── similar_item_sasrec.txt              # 协同过滤相似物品
├── item_generative_indexing_*.txt       # 生成式索引文件
├── gcn_item_embeddings.pt               # GCN 物品 embedding (GRAM-C 必需)
└── item_id_to_gcn_index.json            # 物品 ID 映射表 (GRAM-C 必需)

# 2. 本地模型文件
GRAM_baseline_shen/models/
├── t5_small/                            # T5-small 本地模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── t5-small-machine-articles-tag-generation/  # Tag generation 模型
    └── ...

# 3. 源代码
GRAM_baseline_shen/src/
├── main_generative_gram.py              # 主入口
├── arguments.py                         # 参数定义
├── model/
│   ├── gram.py                          # GRAM 基础模型
│   ├── gram_c.py                        # GRAM-C 模型
│   └── collaborative_adapter.py         # 协同适配器
├── data/
│   └── multi_task_dataset_gram.py       # 数据集
└── ...
```

---

## 运行方式

### 方式 1: 使用训练脚本（推荐）

#### 1.1 运行 GRAM-C (协同增强版本)

```bash
cd GRAM_baseline_shen
bash command/train_gram_c_beauty.sh
```

**特点**:
- ✅ 启用协同前缀注入 (`use_collaborative_prefix=1`)
- ✅ 使用 GCN embedding 增强语义表示
- ✅ Recent-aware Prefix (最近 5 个物品)
- ✅ Prefix Scaling + Dropout 防止过拟合

#### 1.2 运行原始 GRAM (基线版本)

```bash
cd GRAM_baseline_shen
bash command/train_gram_beauty.sh
```

**特点**:
- 不使用协同前缀
- 纯语义推荐

---

### 方式 2: 直接使用 Python 命令

#### 2.1 GRAM-C 完整命令

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0 python src/main_generative_gram.py \
    --datasets Beauty \
    --data_path rec_datasets \
    --backbone models/t5_small \
    --local_model_dir models \
    --distributed 0 \
    --gpu 0 \
    --seed 2023 \
    --train 1 \
    --rec_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --rec_lr 1e-4 \
    --rec_epochs 10 \
    --test_epoch_rec 1 \
    --save_rec_epochs 5 \
    --max_his 20 \
    --item_prompt_max_len 256 \
    --target_max_len 32 \
    --item_prompt all_text \
    --item_id_type split \
    --hierarchical_id_type hierarchy_v1_c128_l7_len32768_split \
    --id_linking 1 \
    --beam_size 10 \
    --use_collaborative_prefix 1 \
    --gcn_dim 64 \
    --gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt \
    --item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json \
    --prefix_init_scale 0.1 \
    --prefix_dropout_prob 0.2 \
    --adapter_dropout 0.1 \
    --recent_k 5 \
    --metrics "hit@5,hit@10,ndcg@5,ndcg@10"
```

#### 2.2 原始 GRAM 命令（禁用协同前缀）

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0 python src/main_generative_gram.py \
    --datasets Beauty \
    --data_path rec_datasets \
    --backbone models/t5_small \
    --local_model_dir models \
    --distributed 0 \
    --gpu 0 \
    --seed 2023 \
    --train 1 \
    --rec_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --rec_lr 1e-4 \
    --rec_epochs 10 \
    --test_epoch_rec 1 \
    --save_rec_epochs 5 \
    --max_his 20 \
    --item_prompt_max_len 256 \
    --target_max_len 32 \
    --item_id_type split \
    --hierarchical_id_type hierarchy_v1_c128_l7_len32768_split \
    --use_collaborative_prefix 0 \
    --metrics "hit@5,hit@10,ndcg@5,ndcg@10"
```

---

## 关键参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | Beauty | 数据集名称 |
| `--data_path` | rec_datasets | 数据集根目录 |
| `--backbone` | models/t5_small | T5 模型路径 |
| `--rec_batch_size` | 16 | 训练批次大小 |
| `--rec_lr` | 1e-4 | 学习率 |
| `--rec_epochs` | 10 | 训练轮数 |
| `--max_his` | 20 | 最大历史长度 |

### GRAM-C 协同前缀参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_collaborative_prefix` | 1 | 启用协同前缀 (1=启用, 0=禁用) |
| `--gcn_dim` | 64 | GCN embedding 维度 |
| `--gcn_item_emb_path` | "" | GCN embedding 文件路径 (.pt 或 .npy) |
| `--item_id_to_gcn_index_path` | "" | 物品 ID 映射文件路径 (.json 或 .pkl) |
| `--prefix_init_scale` | 0.1 | 前缀初始缩放因子（防止早期训练崩溃） |
| `--prefix_dropout_prob` | 0.2 | 前缀 dropout 概率（防止过度依赖） |
| `--adapter_dropout` | 0.1 | Adapter dropout 概率 |
| `--recent_k` | 5 | 使用最近 K 个物品构建协同前缀 |
| `--local_model_dir` | models/ | 本地模型基础目录 |

### 分布式训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--distributed` | 0 | 是否使用分布式训练 (1=是, 0=否) |
| `--gpu` | 0 | GPU 设备 ID (多卡用逗号分隔，如 "0,1,2,3") |
| `--master_port` | 12345 | 分布式训练端口 |

---

## 分布式训练（多卡）

如果有多张 GPU，可以使用分布式训练加速：

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main_generative_gram.py \
    --datasets Beauty \
    --distributed 1 \
    --gpu 0,1,2,3 \
    --master_port 12345 \
    --rec_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --use_collaborative_prefix 1 \
    --gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt \
    --item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json \
    [其他参数...]
```

---

## 输出和日志

### 训练日志

训练过程会输出到：
- **控制台**: 实时训练进度
- **日志文件**: `log/${DATASET}_gram_c_${TIMESTAMP}/train.log`

### 模型检查点

模型会保存到：
```
log/${DATASET}_gram_c_${TIMESTAMP}/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
└── train.log
```

### 评估指标

训练过程中会输出：
- Hit@5, Hit@10, Hit@20
- NDCG@5, NDCG@10, NDCG@20

---

## 常见问题排查

### 1. 找不到 GCN embedding 文件

**错误信息**:
```
FileNotFoundError: GCN embedding file not found: rec_datasets/Beauty/gcn_item_embeddings.pt
```

**解决方案**:
- 确保 `gcn_item_embeddings.pt` 文件存在
- 或者禁用协同前缀: `--use_collaborative_prefix 0`

### 2. 找不到本地模型

**错误信息**:
```
FileNotFoundError: Local model path not found: models/t5_small
```

**解决方案**:
- 下载 T5-small 模型到 `models/t5_small/`
- 或者使用在线模型: `--backbone t5-small` (需要网络)

### 3. CUDA Out of Memory

**解决方案**:
- 减小批次大小: `--rec_batch_size 8`
- 增加梯度累积: `--gradient_accumulation_steps 4`
- 减少历史长度: `--max_his 10`
- 减少 item prompt 长度: `--item_prompt_max_len 128`

### 4. 维度不匹配错误

**错误信息**:
```
ValueError: GCN embedding dimension mismatch: expected 64, got 128
```

**解决方案**:
- 修改 `--gcn_dim` 参数匹配 embedding 文件的维度
- 或者重新生成 GCN embedding 文件

---

## 快速测试（Debug 模式）

如果只想快速测试代码是否能运行：

```bash
cd GRAM_baseline_shen

CUDA_VISIBLE_DEVICES=0 python src/main_generative_gram.py \
    --datasets Beauty \
    --distributed 0 \
    --gpu 0 \
    --rec_batch_size 4 \
    --rec_epochs 1 \
    --test_epoch_rec 1 \
    --debug_train_100 1 \
    --debug_test_100 1 \
    --use_collaborative_prefix 1 \
    --gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt \
    --item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json \
    --hierarchical_id_type hierarchy_v1_c128_l7_len32768_split \
    --item_id_type split
```

**Debug 参数说明**:
- `--debug_train_100 1`: 只使用 100 个训练样本
- `--debug_test_100 1`: 只使用 100 个测试样本
- `--rec_epochs 1`: 只训练 1 轮

---

## 性能对比实验

### 实验 1: GRAM vs GRAM-C

```bash
# 运行 GRAM (基线)
bash command/train_gram_beauty.sh

# 运行 GRAM-C (协同增强)
bash command/train_gram_c_beauty.sh
```

### 实验 2: 不同 Recent K 值

```bash
# K=3
python src/main_generative_gram.py --recent_k 3 [其他参数...]

# K=5 (默认)
python src/main_generative_gram.py --recent_k 5 [其他参数...]

# K=10
python src/main_generative_gram.py --recent_k 10 [其他参数...]
```

### 实验 3: 不同 Prefix Scaling

```bash
# Scale=0.05 (更保守)
python src/main_generative_gram.py --prefix_init_scale 0.05 [其他参数...]

# Scale=0.1 (默认)
python src/main_generative_gram.py --prefix_init_scale 0.1 [其他参数...]

# Scale=0.2 (更激进)
python src/main_generative_gram.py --prefix_init_scale 0.2 [其他参数...]
```

---

## 其他数据集

### Sports & Outdoors

```bash
bash command/train_gram_sports.sh
```

### Toys & Games

```bash
bash command/train_gram_toys.sh
```

### Yelp

```bash
bash command/train_gram_yelp.sh
```

**注意**: 确保对应数据集目录下有必需的文件（user_sequence.txt, item_plain_text.txt 等）

---

## 总结

### 推荐运行方式

1. **首次运行**: 使用训练脚本
   ```bash
   bash command/train_gram_c_beauty.sh
   ```

2. **调参实验**: 使用 Python 命令
   ```bash
   python src/main_generative_gram.py [自定义参数...]
   ```

3. **快速测试**: 使用 Debug 模式
   ```bash
   python src/main_generative_gram.py --debug_train_100 1 --debug_test_100 1 [...]
   ```

### 核心优势

- ✅ **完全离线运行**: 所有模型从本地加载
- ✅ **协同增强**: GCN embedding 作为上下文前缀
- ✅ **Recent-aware**: 聚焦短期兴趣
- ✅ **稳定训练**: Prefix Scaling + Dropout
- ✅ **向后兼容**: 可以禁用协同前缀回退到原始 GRAM

---

**最后更新**: 2026-01-14  
**维护者**: GRAM-C Team
