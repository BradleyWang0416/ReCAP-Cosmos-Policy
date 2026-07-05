mkdir -p "$HOME/.cache" "$HOME/.local/share/uv"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v "$HOME/.cache:/home/cosmos/.cache" \
  -v "$HOME/.local/share/uv:/home/cosmos/.local/share/uv" \
  -v /DATA1:/DATA1:ro \
  -v "$PWD:/workspace" \
  -w /workspace \
  cosmos-policy
