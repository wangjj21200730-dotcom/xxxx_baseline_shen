#!/bin/bash
# GRAM-C Training Script for Beauty Dataset (Multi-GPU Version)
# 
# GRAM-C: Collaborative-Enhanced GRAM
# 核心改进：将协同过滤信号（来自 LightGCN）作为"协同上下文前缀"注入到 T5 Encoder
#
# 关键 Tricks:
# 1. Prefix Scaling (init_scale=0.1) - 防止早期训练注意力崩溃
# 2. Recent-aware Prefix - 使用最近 K 个物品的 GCN embedding 均值池化
# 3. Prefix Dropout (prob=0.2) - 防止过度依赖协同信号

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# ============================================================
# GPU 配置 - 多卡训练需要修改这 3 个地方
# ============================================================
# 1. 设置可见的 GPU（例如：使用 GPU 0,1,2,3）
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 2. 启用分布式训练
DISTRIBUTED=1

# 3. 指定 GPU 列表（必须与 CUDA_VISIBLE_DEVICES 一致）
GPU_LIST="2,3,4,5"

# 4. 设置分布式训练端口（避免冲突）
MASTER_PORT=12345
# ============================================================

# 数据集配置
DATASET="Beauty"
DATA_PATH="${REPO_ROOT}/rec_datasets"

# 日志目录
LOG_DIR="${REPO_ROOT}/log"

# 模型配置 - 使用本地模型
LOCAL_MODEL_DIR="${REPO_ROOT}/models"
BACKBONE="${LOCAL_MODEL_DIR}/t5_small"

# GRAM-C 协同前缀配置
USE_COLLABORATIVE_PREFIX=1      # 启用协同前缀
GCN_DIM=64                      # GCN embedding 维度
PREFIX_INIT_SCALE=0.01          # 前缀初始缩放因子（降低初始幅度，减小早期噪声）
PREFIX_DROPOUT_PROB=0.2         # 前缀 dropout 概率
ADAPTER_DROPOUT=0.1             # Adapter dropout 概率
RECENT_K=5                      # 使用最近 K 个物品

# GCN 文件路径
GCN_ITEM_EMB_PATH="${DATA_PATH}/${DATASET}/gcn_item_embeddings.pt"
ITEM_ID_TO_GCN_INDEX_PATH="${DATA_PATH}/${DATASET}/item_id_to_gcn_index.json"

# 协同过滤配置（Similar Items）
TOP_K_SIMILAR_ITEM=10           # Top-K 相似物品数量（0=禁用）
CF_MODEL="sasrec"               # 协同过滤模型（sasrec/lightgcn）

# 训练配置（多卡可以增大 batch size）
BATCH_SIZE=32                   # 多卡训练可以用更大的 batch size
EVAL_BATCH_SIZE=16
REC_EPOCHS=30                   # 训练 30 个 epoch
REC_LR=1e-4
WARMUP_PROP=0.1
GRADIENT_ACCUMULATION_STEPS=2

# 其他配置
MAX_HIS=20
ITEM_PROMPT_MAX_LEN=128
TARGET_MAX_LEN=32
BEAM_SIZE=10

# hierarchical id 文件后缀（对应: rec_datasets/Beauty/item_generative_indexing_${HIERARCHICAL_ID_TYPE}.txt）
HIERARCHICAL_ID_TYPE="hierarchy_v1_c128_l7_len32768_split"

# 输出路径
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="${REPO_ROOT}/log/${DATASET}_gram_c_${TIMESTAMP}"
mkdir -p ${MODEL_PATH}

# 运行训练
python "${REPO_ROOT}/src/main_generative_gram.py" \
    --datasets ${DATASET} \
    --tasks sequential \
    --data_path ${DATA_PATH} \
    --log_dir ${LOG_DIR} \
    --backbone ${BACKBONE} \
    --local_model_dir ${LOCAL_MODEL_DIR} \
    --model_path ${MODEL_PATH} \
    --distributed ${DISTRIBUTED} \
    --gpu ${GPU_LIST} \
    --master_port ${MASTER_PORT} \
    --batch_size ${BATCH_SIZE} \
    --rec_batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --rec_epochs ${REC_EPOCHS} \
    --rec_lr ${REC_LR} \
    --warmup_prop ${WARMUP_PROP} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_his ${MAX_HIS} \
    --item_prompt_max_len ${ITEM_PROMPT_MAX_LEN} \
    --target_max_len ${TARGET_MAX_LEN} \
    --beam_size ${BEAM_SIZE} \
    --item_prompt all_text \
    --cf_model ${CF_MODEL} \
    --top_k_similar_item ${TOP_K_SIMILAR_ITEM} \
    --use_collaborative_prefix ${USE_COLLABORATIVE_PREFIX} \
    --gcn_dim ${GCN_DIM} \
    --gcn_item_emb_path ${GCN_ITEM_EMB_PATH} \
    --item_id_to_gcn_index_path ${ITEM_ID_TO_GCN_INDEX_PATH} \
    --prefix_init_scale ${PREFIX_INIT_SCALE} \
    --prefix_dropout_prob ${PREFIX_DROPOUT_PROB} \
    --adapter_dropout ${ADAPTER_DROPOUT} \
    --recent_k ${RECENT_K} \
    --metrics "hit@5,hit@10,ndcg@5,ndcg@10" \
    --test_epoch_rec 5 \
    --save_rec_epochs 5 \
    --item_id_type "split" \
    --hierarchical_id_type "${HIERARCHICAL_ID_TYPE}" \
    --prompt_file "${REPO_ROOT}/prompt.txt" \
    --alt_style "rec_first" \
    --rounds 1 \
    --id_epochs 0 \
    2>&1 | tee ${MODEL_PATH}/train.log

echo "Training completed. Model saved to: ${MODEL_PATH}"
