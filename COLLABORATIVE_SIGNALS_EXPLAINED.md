# GRAM-C 协同信号机制说明

## 概述

GRAM-C 使用**两种**协同过滤信号来增强推荐：

1. **Similar Items (传统方式)** - 在 item prompt 中拼接相似物品
2. **Collaborative Prefix (新方式)** - 使用 GCN embedding 作为上下文前缀

---

## 两种协同信号对比

### 1. Similar Items（传统 GRAM 方式）

**机制**: 在每个 item 的文本描述前拼接其 Top-K 相似物品的 ID

**示例**:
```
原始 item prompt:
"nail polish red color"

添加 similar items 后:
"similar items: polish_a12, polish_b34, polish_c56; nail polish red color"
```

**配置参数**:
```bash
--top_k_similar_item 10    # Top-K 相似物品数量
--cf_model sasrec          # 协同过滤模型（sasrec/lightgcn）
```

**数据来源**: `rec_datasets/Beauty/similar_item_sasrec.txt`

**优点**:
- ✅ 简单直接
- ✅ 显式告诉模型哪些物品相似
- ✅ 可解释性强

**缺点**:
- ❌ 增加序列长度（每个 item 多 K 个 token）
- ❌ 只能用离散的 item ID，信息有限
- ❌ 静态相似度（预计算，不考虑用户历史）

---

### 2. Collaborative Prefix（GRAM-C 新方式）

**机制**: 使用最近 K 个物品的 GCN embedding 均值池化，作为一个"协同上下文前缀"注入到 encoder 输入

**示例**:
```
用户历史: [item_1, item_2, item_3, item_4, item_5]

1. 取最近 5 个物品的 GCN embedding
2. Mean pooling → 单个向量 [64-dim]
3. 通过 Adapter 映射到 T5 空间 [512-dim]
4. 作为第 0 个 token 拼接到 encoder 输入最前面

最终序列: [CF_PREFIX] [Token_1] [Token_2] ... [Token_N]
```

**配置参数**:
```bash
--use_collaborative_prefix 1           # 启用协同前缀
--gcn_dim 64                           # GCN embedding 维度
--recent_k 5                           # 使用最近 K 个物品
--prefix_init_scale 0.1                # 初始缩放因子
--prefix_dropout_prob 0.2              # Dropout 概率
--gcn_item_emb_path <path>             # GCN embedding 文件
--item_id_to_gcn_index_path <path>     # ID 映射文件
```

**数据来源**: 
- `rec_datasets/Beauty/gcn_item_embeddings.pt` (GCN embedding)
- `rec_datasets/Beauty/item_id_to_gcn_index.json` (ID 映射)

**优点**:
- ✅ 不增加序列长度（只有 1 个 prefix token）
- ✅ 使用连续向量，信息更丰富
- ✅ 动态考虑用户最近兴趣（Recent-aware）
- ✅ 通过 Adapter 学习最优对齐

**缺点**:
- ❌ 需要预训练 GCN embedding
- ❌ 可解释性较弱（黑盒向量）
- ❌ 增加模型复杂度

---

## 两种方式可以同时使用！

GRAM-C 的设计允许**同时启用**两种协同信号：

```bash
# 同时启用两种协同信号
--top_k_similar_item 10              # ✅ 启用 Similar Items
--use_collaborative_prefix 1         # ✅ 启用 Collaborative Prefix
```

### 组合效果

```
输入序列结构:
[CF_PREFIX] [User_Prompt] [Item_1_with_similar] [Item_2_with_similar] ...

其中:
- CF_PREFIX: 来自 GCN embedding 的协同上下文
- Item_X_with_similar: "similar items: ...; item description"
```

**优势**:
- 结合两种协同信号的优点
- Similar Items 提供显式相似度
- Collaborative Prefix 提供隐式用户兴趣

**劣势**:
- 序列更长（可能需要更大的 `item_prompt_max_len`）
- 训练更慢

---

## 配置建议

### 场景 1: 只用 Similar Items（传统 GRAM）

```bash
--top_k_similar_item 10
--use_collaborative_prefix 0
```

**适用**: 
- 没有 GCN embedding
- 追求可解释性
- 计算资源有限

---

### 场景 2: 只用 Collaborative Prefix（纯 GRAM-C）

```bash
--top_k_similar_item 0
--use_collaborative_prefix 1
--gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt
--item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json
```

**适用**:
- 有预训练 GCN embedding
- 追求性能（序列更短）
- 关注动态用户兴趣

---

### 场景 3: 同时使用两种（最强组合）

```bash
--top_k_similar_item 10
--use_collaborative_prefix 1
--gcn_item_emb_path rec_datasets/Beauty/gcn_item_embeddings.pt
--item_id_to_gcn_index_path rec_datasets/Beauty/item_id_to_gcn_index.json
```

**适用**:
- 追求最佳性能
- 有充足计算资源
- 数据和 embedding 都齐全

---

## 当前脚本配置

### train_gram_beauty.sh（原始 GRAM）

```bash
--top_k_similar_item 10              # ✅ 启用
--use_collaborative_prefix 0         # ❌ 禁用（默认）
```

### train_gram_c_beauty.sh（GRAM-C）

**已修正**:
```bash
--top_k_similar_item 10              # ✅ 启用
--use_collaborative_prefix 1         # ✅ 启用
```

**之前的问题**: 缺少 `--top_k_similar_item` 参数，导致 similar items 功能被禁用

---

## 实验对比建议

### 消融实验（Ablation Study）

1. **Baseline**: 不用任何协同信号
   ```bash
   --top_k_similar_item 0
   --use_collaborative_prefix 0
   ```

2. **GRAM**: 只用 Similar Items
   ```bash
   --top_k_similar_item 10
   --use_collaborative_prefix 0
   ```

3. **GRAM-C (Pure)**: 只用 Collaborative Prefix
   ```bash
   --top_k_similar_item 0
   --use_collaborative_prefix 1
   ```

4. **GRAM-C (Full)**: 同时使用两种
   ```bash
   --top_k_similar_item 10
   --use_collaborative_prefix 1
   ```

---

## 数据文件说明

### Similar Items 文件

**路径**: `rec_datasets/Beauty/similar_item_sasrec.txt`

**格式**:
```
anchor_item_id similar_item_1 similar_item_2 similar_item_3 ...
B00005N7P0 B00KAL5JAU B00KHGIK54 B004ZT0SSG ...
B00006HAXW B00KHH2VOY B00KHGIK54 B00KAL5JAU ...
```

### GCN Embedding 文件

**路径**: `rec_datasets/Beauty/gcn_item_embeddings.pt`

**格式**: PyTorch tensor, shape `[Num_Items + 1, GCN_Dim]`
- 第 0 行: 零向量（用于 padding/unknown）
- 第 1~N 行: 实际物品的 GCN embedding

### ID 映射文件

**路径**: `rec_datasets/Beauty/item_id_to_gcn_index.json`

**格式**:
```json
{
  "B00005N7P0": 1,
  "B00006HAXW": 2,
  ...
}
```

---

## 总结

| 特性 | Similar Items | Collaborative Prefix |
|------|---------------|---------------------|
| 序列长度 | 增加 K*N tokens | 增加 1 token |
| 信息类型 | 离散 ID | 连续向量 |
| 可解释性 | 高 | 低 |
| 动态性 | 静态 | 动态（Recent-aware） |
| 数据需求 | similar_item_*.txt | GCN embeddings |
| 计算开销 | 低 | 中 |

**推荐**: 同时使用两种方式（`--top_k_similar_item 10 --use_collaborative_prefix 1`）以获得最佳性能。

---

**最后更新**: 2026-01-14
