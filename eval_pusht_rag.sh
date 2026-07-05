#!/bin/bash
# eval_pusht_rag.sh — evaluate the residual RAG PushT policy across 9 visual configs.
#
# Live retrieval from the demo pool (no precomputed retrieval files needed at eval).
# Parallel: one process per GPU.
#
# Env:
#   CKPT               path to model_*.pt          (required)
#   BASE_DATASETS_DIR  dataset root (has PushT-Cosmos-Policy/)   default: ./data
#   NUM_GPUS           number of GPUs to use        default: 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT="${CKPT:-$SCRIPT_DIR/checkpoints/ReCAP-Cosmos2.5-pusht/model_000007000.pt}"
BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$SCRIPT_DIR/data}"
NUM_GPUS="${NUM_GPUS:-8}"
COSMOS_HF_CHECKPOINT_ROOT="${COSMOS_HF_CHECKPOINT_ROOT:-/DATA1/wxs/_HUGGINGFACE}"
PREDICT2P5_DISTILLED_CKPT="${PREDICT2P5_DISTILLED_CKPT:-/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt}"
PREDICT2P5_POSTTRAINED_CKPT="${PREDICT2P5_POSTTRAINED_CKPT:-$PREDICT2P5_DISTILLED_CKPT}"
PREDICT2P5_TOKENIZER_CKPT="${PREDICT2P5_TOKENIZER_CKPT:-/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth}"
COSMOS_SKIP_HF_AUTO_DOWNLOAD="${COSMOS_SKIP_HF_AUTO_DOWNLOAD:-1}"
export COSMOS_HF_CHECKPOINT_ROOT
export COSMOS_SKIP_HF_AUTO_DOWNLOAD
export COSMOS_PREDICT2P5_DISTILLED_CKPT="$PREDICT2P5_DISTILLED_CKPT"
export COSMOS_PREDICT2P5_POSTTRAINED_CKPT="$PREDICT2P5_POSTTRAINED_CKPT"
export COSMOS_PREDICT2P5_TOKENIZER_CKPT="$PREDICT2P5_TOKENIZER_CKPT"

DATA="$BASE_DATASETS_DIR/PushT-Cosmos-Policy/success_only"
EVAL_CFG=cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only
LOG_ROOT=cosmos_policy/experiments/robot/pusht_ret/logs_pusht_rag_eval
TAG=residual_top100
mkdir -p "$LOG_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "$CKPT"
require_file "$COSMOS_PREDICT2P5_DISTILLED_CKPT"
require_file "$COSMOS_PREDICT2P5_POSTTRAINED_CKPT"
require_dir "$DATA"
require_file "$DATA/t5_embeddings.pkl"
require_file "$DATA/dataset_statistics.json"
require_file "$DATA/delta_dataset_statistics.json"

VISUAL_CONFIGS=(tri_default tri_goal_flipped tri_rot0 tri_rot15 tri_rot-15 tri_rot30 tri_rot-30 tri_rot60 tri_rot-60)

COMMON_ARGS=(
  --config "$EVAL_CFG"
  --ckpt_path "$CKPT"
  --config_file cosmos_policy/config/config.py
  --t5_text_embeddings_path "$DATA/t5_embeddings.pkl"
  --dataset_stats_path      "$DATA/dataset_statistics.json"
  --retrieval_data_dir      "$DATA"
  --use_residual_actions True
  --delta_stats_path        "$DATA/delta_dataset_statistics.json"
  --num_trials 50 --chunk_size 8 --num_open_loop_steps 8
  --num_denoising_steps_action 5 --predict_future_states True --seed 42
)

declare -a GPU_PIDS
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[$i]=0; done
get_free_gpu() {
  while true; do
    for ((i=0; i<NUM_GPUS; i++)); do
      pid=${GPU_PIDS[$i]}
      if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then echo "$i"; return; fi
    done
    sleep 2
  done
}

for V in "${VISUAL_CONFIGS[@]}"; do
  GPU_ID=$(get_free_gpu)
  LOGFILE="$LOG_ROOT/${TAG}--${V}.log"
  echo "GPU=$GPU_ID  $V  -> $LOGFILE"
  CUDA_VISIBLE_DEVICES=$GPU_ID uv run --python 3.10 --extra cu128 --group pusht \
    -m cosmos_policy.experiments.robot.pusht_ret.run_eval \
    "${COMMON_ARGS[@]}" --visual_config "$V" \
    --local_log_dir "$LOG_ROOT/$TAG" > "$LOGFILE" 2>&1 &
  GPU_PIDS[$GPU_ID]=$!
done

wait
echo ""
echo "=== Summary ==="
grep -h "Success rate" "$LOG_ROOT/${TAG}"--*.log 2>/dev/null | sort || true
