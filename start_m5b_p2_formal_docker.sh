#!/usr/bin/env bash
set -euo pipefail

# This only opens the full M5B-P2 Docker shell.  It does not create a formal
# launch-activation/final-acceptance artifact and does not launch any cell.
mkdir -p "$HOME/.cache" "$HOME/.local/share/uv"

# Select exactly four physical host GPUs.  Docker remaps them to logical
# devices 0,1,2,3 inside the container, which is the frozen v5 runtime view.
M5B_P2_GPU_DEVICES="${M5B_P2_GPU_DEVICES:-0,1,2,3}"
if [[ ! "$M5B_P2_GPU_DEVICES" =~ ^([0-9]+,){3}[0-9]+$ ]]; then
  echo "M5B_P2_GPU_DEVICES must contain exactly four comma-separated GPU indices" >&2
  exit 2
fi
IFS=',' read -r -a m5b_p2_gpu_ids <<< "$M5B_P2_GPU_DEVICES"
declare -A m5b_p2_seen_gpu_ids=()
for gpu_id in "${m5b_p2_gpu_ids[@]}"; do
  if [[ -n "${m5b_p2_seen_gpu_ids[$gpu_id]:-}" ]]; then
    echo "M5B_P2_GPU_DEVICES contains a duplicate GPU index: $gpu_id" >&2
    exit 2
  fi
  m5b_p2_seen_gpu_ids[$gpu_id]=1
done
M5B_P2_DOCKER_GPU_REQUEST="\"device=${M5B_P2_GPU_DEVICES}\""

docker run --rm -it \
  --name recap_m5b_p2_formal \
  --gpus "$M5B_P2_DOCKER_GPU_REQUEST" \
  --ipc=host \
  --network=host \
  -e HOST_USER_ID="$(id -u)" \
  -e HOST_GROUP_ID="$(id -g)" \
  -e COSMOS_SKIP_HF_AUTO_DOWNLOAD=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e WANDB_MODE=disabled \
  -e WANDB_DISABLED=true \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=65536 \
  -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  -e TORCH_NCCL_DESYNC_DEBUG=1 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=COLL \
  -e HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS=5 \
  -v "$HOME/.cache:/home/cosmos/.cache" \
  -v "$HOME/.local/share/uv:/home/cosmos/.local/share/uv" \
  -v /DATA1:/DATA1:rw \
  -v "$PWD:/workspace" \
  -w /workspace \
  cosmos-policy
