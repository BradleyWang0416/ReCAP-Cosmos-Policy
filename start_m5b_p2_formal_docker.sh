#!/usr/bin/env bash
set -euo pipefail

# This only opens the full M5B-P2 Docker shell.  It does not create a formal
# launch-activation/final-acceptance artifact and does not launch any cell.
mkdir -p "$HOME/.cache" "$HOME/.local/share/uv"

docker run --rm -it \
  --name recap_m5b_p2_formal \
  --gpus all \
  --ipc=host \
  --network=host \
  -e HOST_USER_ID="$(id -u)" \
  -e HOST_GROUP_ID="$(id -g)" \
  -e COSMOS_SKIP_HF_AUTO_DOWNLOAD=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e WANDB_MODE=disabled \
  -e WANDB_DISABLED=true \
  -v "$HOME/.cache:/home/cosmos/.cache" \
  -v "$HOME/.local/share/uv:/home/cosmos/.local/share/uv" \
  -v /DATA1:/DATA1:rw \
  -v "$PWD:/workspace" \
  -w /workspace \
  cosmos-policy
