#!/bin/bash
# Evaluate GRAM-C (Beauty) using a specific checkpoint (epoch 20) in distributed mode.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# ============================================================
# GPU / distributed config (edit if needed)
# ============================================================
export CUDA_VISIBLE_DEVICES=2,3,4,5
DISTRIBUTED=1
GPU_LIST="0,1,2,3"
MASTER_PORT=2341
# ============================================================

DATASET="Beauty"
DATA_PATH="${REPO_ROOT}/rec_datasets"
LOG_DIR="${REPO_ROOT}/log"
LOCAL_MODEL_DIR="${REPO_ROOT}/models"
BACKBONE="${LOCAL_MODEL_DIR}/t5_small"
LOCAL_MODEL_DIR_ABS="${LOCAL_MODEL_DIR}"

# GRAM-C collaborative prefix config (must match training)
USE_COLLABORATIVE_PREFIX=1
GCN_DIM=64
PREFIX_INIT_SCALE=0.01
PREFIX_DROPOUT_PROB=0.2
ADAPTER_DROPOUT=0.1
RECENT_K=5
GCN_ITEM_EMB_PATH="${DATA_PATH}/${DATASET}/gcn_item_embeddings.pt"
ITEM_ID_TO_GCN_INDEX_PATH="${DATA_PATH}/${DATASET}/item_id_to_gcn_index.json"

# CF / similar-item retrieval config (must match training)
TOP_K_SIMILAR_ITEM=10
CF_MODEL="sasrec"

# Evaluation config
EVAL_BATCH_SIZE=1
MAX_HIS=20
ITEM_PROMPT_MAX_LEN=128
TARGET_MAX_LEN=32
BEAM_SIZE=50
HIERARCHICAL_ID_TYPE="hierarchy_v1_c128_l7_len32768_split"
METRICS="hit@1,hit@3,hit@5,hit@10,hit@20,hit@50,ndcg@1,ndcg@3,ndcg@5,ndcg@10,ndcg@20,ndcg@50"
SAVE_PREDICTIONS=1

# Point to the epoch-20 checkpoint you want to test
# Note: depending on the runner/save logic, the file name might be:
# - model_rec_phase_1_epoch_20.pt
# - scheduler_rec_phase_1_epoch_20.pt
REC_MODEL_PATH="${REPO_ROOT}/log/Beauty/96_20260115_1214/id_0_rec_30/model_rec_phase_1_epoch_20.pt"

PYTHONPATH="${REPO_ROOT}/src" python -u ${REPO_ROOT}/src/main_generative_gram.py \
  --distributed ${DISTRIBUTED} \
  --gpu ${GPU_LIST} \
  --master_addr localhost \
  --master_port ${MASTER_PORT} \
  --local_model_dir ${LOCAL_MODEL_DIR_ABS} \
  --train 0 \
  --rec_model_path ${REC_MODEL_PATH} \
  --rec_epochs 0 \
  --batch_size 1 \
  --rec_batch_size 1 \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --log_dir ${LOG_DIR} \
  --data_path ${DATA_PATH} \
  --datasets ${DATASET} \
  --tasks sequential \
  --prompt_file ${REPO_ROOT}/prompt.txt \
  --max_his ${MAX_HIS} \
  --his_sep " ; " \
  --skip_empty_his 1 \
  --reverse_history 1 \
  --user_id_without_target_item 1 \
  --id_linking 1 \
  --backbone ${BACKBONE} \
  --item_prompt all_text \
  --cf_model ${CF_MODEL} \
  --top_k_similar_item ${TOP_K_SIMILAR_ITEM} \
  --item_prompt_max_len ${ITEM_PROMPT_MAX_LEN} \
  --target_max_len ${TARGET_MAX_LEN} \
  --hierarchical_id_type ${HIERARCHICAL_ID_TYPE} \
  --item_id_type split \
  --beam_size ${BEAM_SIZE} \
  --metrics ${METRICS} \
  --save_predictions ${SAVE_PREDICTIONS} \
  --use_position_embedding 1 \
  --use_collaborative_prefix ${USE_COLLABORATIVE_PREFIX} \
  --gcn_dim ${GCN_DIM} \
  --gcn_item_emb_path ${GCN_ITEM_EMB_PATH} \
  --item_id_to_gcn_index_path ${ITEM_ID_TO_GCN_INDEX_PATH} \
  --prefix_init_scale ${PREFIX_INIT_SCALE} \
  --prefix_dropout_prob ${PREFIX_DROPOUT_PROB} \
  --adapter_dropout ${ADAPTER_DROPOUT} \
  --recent_k ${RECENT_K}
