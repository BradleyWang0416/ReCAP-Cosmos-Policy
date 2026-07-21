#!/usr/bin/env bash
set -euo pipefail

# Stage-0 launcher only: record the Docker session, run the v04 preflight, and
# open a shell. It never starts data preparation, training, or evaluation.
v04_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
v04_gpu_devices="${HUMAN2ROBOT_V04_GPU_DEVICES:-0,1,2,3}"

if [[ ! "$v04_gpu_devices" =~ ^([0-9]+,){3}[0-9]+$ ]]; then
  echo "HUMAN2ROBOT_V04_GPU_DEVICES must contain exactly four comma-separated GPU indices" >&2
  exit 2
fi
IFS=',' read -r -a v04_gpu_ids <<< "$v04_gpu_devices"
declare -A v04_seen_gpu_ids=()
for v04_gpu_id in "${v04_gpu_ids[@]}"; do
  if [[ -n "${v04_seen_gpu_ids[$v04_gpu_id]:-}" ]]; then
    echo "HUMAN2ROBOT_V04_GPU_DEVICES contains a duplicate GPU index: $v04_gpu_id" >&2
    exit 2
  fi
  v04_seen_gpu_ids[$v04_gpu_id]=1
done

v04_image="cosmos-policy:latest"
v04_image_id="$(docker image inspect "$v04_image" --format '{{.Id}}')"
if [[ ! "$v04_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Unable to bind the local cosmos-policy:latest image ID" >&2
  exit 2
fi
v04_started="$(date -u +%Y%m%dT%H%M%SZ)"
v04_session_id="docker_session_${v04_started}_$RANDOM"
v04_container_name="recap_human2robot_v04_${v04_started,,}_$RANDOM"
v04_gpu_request="\"device=${v04_gpu_devices}\""

mkdir -p "$HOME/.cache" "$HOME/.local/share/uv"

docker run --rm -it \
  --pull=never \
  --name "$v04_container_name" \
  --gpus "$v04_gpu_request" \
  --ipc=host \
  --network=none \
  -e HOST_USER_ID="$(id -u)" \
  -e HOST_GROUP_ID="$(id -g)" \
  -e HUMAN2ROBOT_V04_IMAGE="$v04_image" \
  -e HUMAN2ROBOT_V04_IMAGE_ID="$v04_image_id" \
  -e HUMAN2ROBOT_V04_CONTAINER_NAME="$v04_container_name" \
  -e HUMAN2ROBOT_V04_GPU_DEVICES="$v04_gpu_devices" \
  -e RECAP_WORKSPACE=/workspace \
  -e HUMAN2ROBOT_ROOT=/workspace/data/Human2Robot \
  -e HUMAN2ROBOT_V04_RUN_ROOT=/DATA1/wxs/ReCAP_M5B_V04_RUNS \
  -e COSMOS_HF_CHECKPOINT_ROOT=/DATA1/wxs/_HUGGINGFACE \
  -e COSMOS_SKIP_HF_AUTO_DOWNLOAD=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e WANDB_MODE=disabled \
  -e WANDB_DISABLED=true \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_DEBUG=WARN \
  -v "$v04_repo_root:/workspace:rw" \
  -v /DATA1:/DATA1:rw \
  -v "$HOME/.cache:/home/cosmos/.cache:rw" \
  -v "$HOME/.local/share/uv:/home/cosmos/.local/share/uv:rw" \
  -w /workspace \
  "$v04_image" \
  bash --noprofile --norc -c ".venv/bin/python tools/human2robot_v04.py session-receipt --run-id '$v04_session_id'; .venv/bin/python tools/human2robot_v04.py preflight --run-id '${v04_session_id}_preflight' || true; exec bash --noprofile --norc"
