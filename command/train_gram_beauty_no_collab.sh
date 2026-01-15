#!/bin/bash
# Ablation: GRAM without collaborative prefix (Beauty, multi-GPU)
# Purpose: disable collaborative prefix (use_collaborative_prefix=0, recent_k=0)
#          to check whether misaligned/low-quality GCN signal is hurting metrics.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# =============================
# GPU / Distributed settings
# =============================
export CUDA_VISIBLE_DEVICES=2,3,4,5
DISTRIBUTED=1
GPU_LIST="2,3,4,5"
MASTER_PORT=12345

# =============================
# Dataset & paths
# =============================
DATASET="Beauty"
DATA_PATH="${REPO_ROOT}/rec_datasets"
LOG_DIR="${REPO_ROOT}/log"
LOCAL_MODEL_DIR="${REPO_ROOT}/models"
BACKBONE="${LOCAL_MODEL_DIR}/t5_small"

# =============================
# Collaborative prefix (disabled)
# =============================
USE_COLLABORATIVE_PREFIX=0
RECENT_K=0
GCN_ITEM_EMB_PATH="${DATA_PATH}/${DATASET}/gcn_item_embeddings.pt"  # still pass a path to satisfy arg parser
ITEM_ID_TO_GCN_INDEX_PATH="${DATA_PATH}/${DATASET}/item_id_to_gcn_index.json"
PREFIX_INIT_SCALE=0.01
PREFIX_DROPOUT_PROB=0.2
ADAPTER_DROPOUT=0.1
GCN_DIM=64

# =============================
# Training hyperparameters
# =============================
BATCH_SIZE=32
EVAL_BATCH_SIZE=16
REC_EPOCHS=30
REC_LR=1e-4
WARMUP_PROP=0.1
GRADIENT_ACCUMULATION_STEPS=2
MAX_HIS=20
ITEM_PROMPT_MAX_LEN=128
TARGET_MAX_LEN=32
BEAM_SIZE=10
HIERARCHICAL_ID_TYPE="hierarchy_v1_c128_l7_len32768_split"

# =============================
# Output
# =============================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="${REPO_ROOT}/log/${DATASET}_gram_no_collab_${TIMESTAMP}"
mkdir -p ${MODEL_PATH}

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
    --cf_model sasrec \
    --top_k_similar_item 10 \
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
