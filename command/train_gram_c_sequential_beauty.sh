#!/bin/bash
# GRAM-C Sequential Prefix Training Script for Beauty Dataset
# 
# GRAM-C with Sequential Prefix: 
# 核心改进：将最近 K 个物品的 LightGCN embedding 分别通过 Adapter，
# 得到 K 个独立的 Soft Tokens，让 T5 的 Self-Attention 自己学习哪些物品更重要。
#
# 与均值池化模式的区别：
# - 均值池化模式 (use_sequential_prefix=0): K 个 embedding 池化为 1 个 token
# - 序列化模式 (use_sequential_prefix=1): K 个 embedding 分别映射为 K 个 tokens
#
# 关键 Tricks:
# 1. Sequential Prefix - 保留 K 个独立 tokens，让 T5 自己学习重要性
# 2. Whole-Prefix Dropout - 整个 K 个 tokens 一起置零（而非逐 token dropout）
# 3. LightGCN Embeddings - 使用多层聚合后的 item embeddings

 set -euo pipefail
 
 SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
 
 # ============================================================
 # GPU 配置 - 多卡训练需要修改这 3 个地方
 # ============================================================
 export CUDA_VISIBLE_DEVICES=2,3,4,5
 DISTRIBUTED=1
 GPU_LIST="0,1,2,3"
 MASTER_PORT=2342
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
 USE_SEQUENTIAL_PREFIX=1         # 启用序列化前缀（K 个独立 tokens）
 GCN_DIM=64                      # GCN embedding 维度
 PREFIX_INIT_SCALE=0.01          # 前缀初始缩放因子（降低初始幅度，减小早期噪声）
 PREFIX_DROPOUT_PROB=0.2         # 前缀 dropout 概率（整段置零）
 ADAPTER_DROPOUT=0.1             # Adapter dropout 概率
 RECENT_K=5                      # 使用最近 K 个物品（也是 prefix token 数量）

# LightGCN 文件路径（优先使用 LightGCN embeddings）
LIGHTGCN_EMB_PATH="${DATA_PATH}/${DATASET}/lightgcn_data/lightgcn_item_embeddings.pt"
GCN_ITEM_EMB_PATH="${DATA_PATH}/${DATASET}/gcn_item_embeddings.pt"  # 回退路径
ITEM_ID_TO_GCN_INDEX_PATH="${DATA_PATH}/${DATASET}/item_id_to_gcn_index.json"

# 协同过滤配置（Similar Items）
TOP_K_SIMILAR_ITEM=10           # Top-K 相似物品数量（0=禁用）
CF_MODEL="sasrec"               # 协同过滤模型（sasrec/lightgcn）

 # 训练配置
 BATCH_SIZE=32
 EVAL_BATCH_SIZE=1
 REC_EPOCHS=30
 REC_LR=1e-3
 WARMUP_PROP=0.05
 GRADIENT_ACCUMULATION_STEPS=2

 # 其他配置
 MAX_HIS=20
 ITEM_PROMPT_MAX_LEN=128
 TARGET_MAX_LEN=32
 BEAM_SIZE=50
 
 # generative indexing file
 HIERARCHICAL_ID_TYPE="hierarchy_v1_c128_l7_len32768_split"
 ITEM_ID_PATH="item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt"
 
 # Evaluation metrics / outputs
 METRICS="hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50"
 SAVE_PREDICTIONS=1

 # 输出路径
 TIMESTAMP=$(date +%Y%m%d_%H%M%S)
 MODEL_PATH="${REPO_ROOT}/log/${DATASET}_gram_c_sequential_${TIMESTAMP}"
 mkdir -p ${MODEL_PATH}

# 运行训练
 python ${REPO_ROOT}/src/main_generative_gram.py \
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
    --use_sequential_prefix ${USE_SEQUENTIAL_PREFIX} \
    --gcn_dim ${GCN_DIM} \
    --lightgcn_emb_path ${LIGHTGCN_EMB_PATH} \
    --gcn_item_emb_path ${GCN_ITEM_EMB_PATH} \
    --item_id_to_gcn_index_path ${ITEM_ID_TO_GCN_INDEX_PATH} \
    --prefix_init_scale ${PREFIX_INIT_SCALE} \
    --prefix_dropout_prob ${PREFIX_DROPOUT_PROB} \
     --adapter_dropout ${ADAPTER_DROPOUT} \
     --recent_k ${RECENT_K} \
     --metrics ${METRICS} \
     --test_epoch_rec 5 \
     --save_rec_epochs 5 \
     --save_predictions ${SAVE_PREDICTIONS} \
     --item_id_type "split" \
     --hierarchical_id_type "${HIERARCHICAL_ID_TYPE}" \
     --item_id_path "${ITEM_ID_PATH}" \
     --prompt_file "${REPO_ROOT}/prompt.txt" \
     --alt_style "rec_first" \
     --rounds 1 \
     --id_epochs 0 \
     2>&1 | tee ${MODEL_PATH}/train.log

echo "Training completed. Model saved to: ${MODEL_PATH}"
echo ""
echo "Configuration:"
echo "  - Sequential Prefix: ${USE_SEQUENTIAL_PREFIX} (K=${RECENT_K} tokens)"
echo "  - LightGCN Embeddings: ${LIGHTGCN_EMB_PATH}"
echo "  - Prefix Dropout: ${PREFIX_DROPOUT_PROB} (whole-prefix dropout)"
