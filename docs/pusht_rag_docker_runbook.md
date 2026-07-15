# PushT RAG Docker 运行手册

本文档记录从启动 Docker、构建镜像、进入容器、同步依赖，到执行 PushT RAG 评测的完整流程。默认仓库位于：

```bash
/home/wxs/ReCAP-Cosmos-Policy
```

以下命令默认在仓库根目录执行。

## 1. 宿主机 Docker 准备

### 1.1 让 `wxs` 用户可以使用 Docker

如果出现：

```text
permission denied while trying to connect to the Docker daemon socket
```

说明当前用户没有访问 Docker daemon socket 的权限。由于 `wxs` 用户没有 sudo 权限，需要使用管理员账户执行：

```bash
sudo usermod -aG docker wxs
sudo systemctl restart docker
```

然后 `wxs` 用户重新登录，或临时执行：

```bash
newgrp docker
```

验证：

```bash
docker ps
```

### 1.2 配置 Docker daemon 代理或 registry mirror

如果构建镜像时拉取 Docker Hub 超时，例如：

```text
Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection
```

用管理员账户编辑：

```bash
sudo nano /etc/docker/daemon.json
```

示例代理配置：

```json
{
  "proxies": {
    "http-proxy": "http://127.0.0.1:7890",
    "https-proxy": "http://127.0.0.1:7890",
    "no-proxy": "localhost,127.0.0.1"
  }
}
```

如果同时配置 registry mirror，需要保持 JSON 合法，例如：

```json
{
  "registry-mirrors": [
    "https://your-mirror.example.com"
  ],
  "proxies": {
    "http-proxy": "http://127.0.0.1:7890",
    "https-proxy": "http://127.0.0.1:7890",
    "no-proxy": "localhost,127.0.0.1"
  }
}
```

保存后重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

验证 Docker 是否能访问外网：

```bash
docker pull hello-world
```

## 2. 安装 Docker GPU runtime

宿主机能运行 `nvidia-smi` 只说明 NVIDIA 驱动正常，不代表 Docker 已经能使用 GPU。

如果运行容器时报错：

```text
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

先检查：

```bash
which nvidia-ctk nvidia-container-runtime nvidia-container-cli
sudo docker info | grep -i -A5 runtimes
```

如果 runtime 里只有 `runc`，需要管理员安装 NVIDIA Container Toolkit：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

这里先执行 `sudo apt-get update` 是为了刷新本机 apt 软件包索引，让 apt 认识刚加入的 NVIDIA 软件源；否则安装时可能找不到 `nvidia-container-toolkit`，或者安装到旧版本。

验证 Docker GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

能在容器内看到 GPU 表格，说明 Docker GPU runtime 已可用。

## 3. 构建项目镜像

当前 `docker/Dockerfile` 已经兼容 legacy Docker builder，不再依赖 BuildKit 专属的 `COPY --chmod` 或 `RUN --mount`。

构建镜像：

```bash
docker build -t cosmos-policy docker
```

成功时会看到：

```text
Successfully built ...
Successfully tagged cosmos-policy:latest
```

如果仍遇到 BuildKit 相关错误，也可以显式启用 BuildKit：

```bash
DOCKER_BUILDKIT=1 docker build -t cosmos-policy docker
```

## 4. 启动容器

仓库中已有启动脚本：

```bash
bash start_docker_byBrad.sh
```

当前脚本核心内容：

```bash
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
```

进入容器后应看到：

```bash
cosmos@user:/workspace$ whoami
cosmos

cosmos@user:/workspace$ pwd
/workspace
```

### 4.1 `/workspace` 实际在哪里

`/workspace` 是宿主机当前仓库目录的 bind mount：

```bash
-v "$PWD:/workspace"
```

如果宿主机在：

```bash
/home/wxs/ReCAP-Cosmos-Policy
```

那么容器里的：

```bash
/workspace
```

就是宿主机的：

```bash
/home/wxs/ReCAP-Cosmos-Policy
```

容器内改 `/workspace` 下的文件，宿主机仓库会同步变化。

### 4.2 数据、权重和缓存实际存放位置

当前启动脚本挂载：

```bash
-v /DATA1:/DATA1:ro
```

因此容器可以读取 `/DATA1`，但默认不能写入 `/DATA1`。

本项目中常用路径：

```bash
/workspace/data/PushT-Cosmos-Policy
/DATA1/wxs/DATASETS/PushT-Cosmos-Policy
/DATA1/wxs/_HUGGINGFACE
```

当前仓库里 `data/PushT-Cosmos-Policy` 是指向 `/DATA1/wxs/DATASETS/PushT-Cosmos-Policy` 的 symlink。因为 `/DATA1` 是只读挂载，所以如果需要下载新数据或权重，建议在宿主机下载到 `/DATA1`；如果必须在容器内下载，需要把启动脚本改成可写挂载：

```bash
-v /DATA1:/DATA1
```

uv 和 HuggingFace 等缓存位置：

```bash
宿主机: $HOME/.cache
容器内: /home/cosmos/.cache

宿主机: $HOME/.local/share/uv
容器内: /home/cosmos/.local/share/uv
```

项目虚拟环境位于：

```bash
/workspace/.venv
```

注意：`.venv/bin/python` 指向容器内的 uv Python，例如 `/home/cosmos/.local/share/uv/.../python3.10`。所以它在容器内可用，但在宿主机上可能显示为断开的 symlink，这是正常现象。

### 4.3 M5B-P2 正式输出专用容器（只打开 shell，不启动训练）

上面的通用 PushT 容器故意使用 `/DATA1:/DATA1:ro`。M5B-P2 的冻结正式输出根是
`/DATA1/wxs/ReCAP_M5B_P2_RUNS`，因此不能在该只读容器中创建 checkpoint、cell artifact
或 source snapshot。M5B-P2 在 successor contract 获批后应使用独立脚本：

```bash
bash start_m5b_p2_formal_docker.sh
```

v3 successor 固定只向容器暴露 4 张宿主机 GPU。默认使用宿主机 `0,1,2,3`；若稳定卡是
其他编号，启动时显式指定恰好四个、互不重复的物理卡号，例如：

```bash
M5B_P2_GPU_DEVICES=2,3,4,5 bash start_m5b_p2_formal_docker.sh
```

容器内这四张卡会重新编号为逻辑 `0,1,2,3`。正式训练命令固定
`torchrun --nproc_per_node=4`，禁止再用 `--gpus all` 向正式容器暴露额外 GPU。

该脚本只打开名为 `recap_m5b_p2_formal` 的完整 Docker shell，并执行以下约束：

- `/DATA1` 可写挂载，`/workspace` 仍绑定当前仓库；
- 固定 offline/no-download 环境，W&B disabled；
- 使用已经存在的 `cosmos-policy:latest`、本地权重和缓存；
- 不下载、不创建 launch activation/final acceptance，也不自动启动训练。

进入 shell 后首先运行只读 preflight：

```bash
.venv/bin/python -m tools.human2robot_m5b_p2_preflight
```

在可写正式容器中，冻结与授权顺序必须是：

```bash
# 1. 物化当前候选代码的 immutable source snapshot；不启动 cell
.venv/bin/python -m tools.human2robot_m5b_p2 prepare

# 2. 对同一 candidate code 运行冻结的完整 Docker suite 并写 receipt
.venv/bin/python -m tools.human2robot_m5b_p2_activation run-docker-suite

# 3. 复核 mount/GPU/storage/weights/snapshot/receipt 后只签发 launch authorization
.venv/bin/python -m tools.human2robot_m5b_p2_activation issue-launch

# 4. launch artifact 存在后重跑 preflight；此时才允许返回 passed
.venv/bin/python -m tools.human2robot_m5b_p2_preflight

# 5. 仍只查看计划，不会自动 run all
.venv/bin/python -m tools.human2robot_m5b_p2_dag plan
```

第 4、5 步都必须显示 `formal_queue_allowed=true`、`formal_queue_started=false`，且
preflight 的 `blockers=[]`、DAG plan 的 `matrix_blockers=[]`。preflight 与单-cell
dispatcher 会重新计算当前源码 SHA256，并核对 source snapshot 与 Docker-suite receipt；
签发后若修改任何受控源码，旧 activation 会立即失效，必须重新执行上述 1–4 步。

正式 cell 仍须逐个显式执行 `human2robot_m5b_p2_dag run-cell <cell_id>`；任何父产物未完成都会拒绝执行。

只有 preflight 的 mount、恰好 4 GPU、存储、权重、source snapshot 与 successor-contract blocker
全部清零，且 `/DATA1/wxs/ReCAP_M5B_P2_RUNS/launch_activation_v3.json` 按冻结 schema 独立签发，
才允许执行 203-cell DAG。launch activation 只开启队列，仍固定
`p2_acceptance_allowed=false`；第 203 个终止报告通过后，才能另行生成
`final_acceptance_v3.json`。v2 activation 和旧 `run_manifest.json` 只保留为 8 卡历史记录，
不能授权 v3 四卡运行。因此仅把 `/DATA1` 改为可写并不构成启动授权。

## 5. 同步 Python 环境

进入容器后执行：

```bash
uv sync --python 3.10 --extra cu128 --group pusht
```

含义：

- `uv sync`：根据 `pyproject.toml` 和 lock 文件同步项目虚拟环境。
- `--python 3.10`：强制使用 Python 3.10，避免 uv 自动下载或选择 Python 3.13。
- `--extra cu128`：安装 CUDA 12.8 对应的 PyTorch/CUDA 依赖组合。
- `--group pusht`：安装 PushT 实验需要的依赖组。

如果看到：

```text
warning: Ignoring existing virtual environment linked to non-existent Python interpreter
```

通常是 `.venv` 里的 Python symlink 指向容器内 uv Python，而对应 uv Python 没有持久化。当前启动脚本已经挂载：

```bash
-v "$HOME/.local/share/uv:/home/cosmos/.local/share/uv"
```

重新进入容器后再执行一次：

```bash
uv sync --python 3.10 --extra cu128 --group pusht
```

即可恢复。

## 6. 提前准备 HuggingFace 权重

为了避免程序运行时自动联网下载，推荐提前在宿主机下载到：

```bash
/DATA1/wxs/_HUGGINGFACE
```

如果模型是 gated repo，需要先申请访问权限，然后登录：

```bash
hf auth login
```

### 6.1 Cosmos Predict2.5 distilled checkpoint

评测脚本默认读取：

```bash
/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt
```

下载命令：

```bash
mkdir -p /DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B

hf download nvidia/Cosmos-Predict2.5-2B \
  base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt \
  --repo-type model \
  --local-dir /DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B
```

### 6.2 Cosmos Predict2.5 tokenizer

评测脚本默认读取：

```bash
/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth
```

下载命令：

```bash
hf download nvidia/Cosmos-Predict2.5-2B \
  tokenizer.pth \
  --repo-type model \
  --local-dir /DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B
```

### 6.3 Cosmos Predict2 2B Video2World checkpoint

部分配置导入时会引用：

```bash
/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt
```

下载命令：

```bash
mkdir -p /DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2-2B-Video2World

hf download nvidia/Cosmos-Predict2-2B-Video2World \
  model-480p-16fps.pt \
  --repo-type model \
  --local-dir /DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2-2B-Video2World
```

### 6.4 可选 post-trained checkpoint

当前 `eval_pusht_rag.sh` 默认把 `PREDICT2P5_POSTTRAINED_CKPT` 指向 distilled checkpoint，避免导入无关配置时强制下载 post-trained 权重。

如果确实要使用 post-trained 权重，可下载：

```bash
hf download nvidia/Cosmos-Predict2.5-2B \
  base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
  --repo-type model \
  --local-dir /DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B
```

然后运行时覆盖：

```bash
PREDICT2P5_POSTTRAINED_CKPT=/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
NUM_GPUS=1 bash eval_pusht_rag.sh
```

## 7. 检查 PushT 数据

评测需要以下文件存在：

```bash
/workspace/data/PushT-Cosmos-Policy/success_only/t5_embeddings.pkl
/workspace/data/PushT-Cosmos-Policy/success_only/dataset_statistics.json
/workspace/data/PushT-Cosmos-Policy/success_only/delta_dataset_statistics.json
```

容器内检查：

```bash
ls -lh /workspace/data/PushT-Cosmos-Policy/success_only/t5_embeddings.pkl
ls -lh /workspace/data/PushT-Cosmos-Policy/success_only/dataset_statistics.json
ls -lh /workspace/data/PushT-Cosmos-Policy/success_only/delta_dataset_statistics.json
```

如果报：

```text
Dataset stats do not exist at path: ./data/PushT-Cosmos-Policy/success_only/dataset_statistics.json
```

优先确认当前路径是 `/workspace`：

```bash
pwd
```

以及 symlink 在容器内可见：

```bash
ls -lh /workspace/data
ls -lh /workspace/data/PushT-Cosmos-Policy/success_only
```

当前 `eval_pusht_rag.sh` 已经在脚本开头自动切换到脚本所在目录，并使用绝对路径，正常情况下不会再因为工作目录错误找不到数据。

## 8. 执行 PushT RAG 评测

进入容器：

```bash
bash start_docker_byBrad.sh
```

容器内同步依赖：

```bash
uv sync --python 3.10 --extra cu128 --group pusht
```

建议先单 GPU 验证：

```bash
NUM_GPUS=1 bash eval_pusht_rag.sh
```

确认无报错后，再使用 8 张 GPU：

```bash
NUM_GPUS=8 bash eval_pusht_rag.sh
```

日志目录：

```bash
cosmos_policy/experiments/robot/pusht_ret/logs_pusht_rag_eval
```

查看单个配置日志：

```bash
tail -f cosmos_policy/experiments/robot/pusht_ret/logs_pusht_rag_eval/residual_top100--tri_default.log
```

停止正在运行的评测进程：

```bash
pkill -f cosmos_policy.experiments.robot.pusht_ret.run_eval
```

## 9. 当前评测脚本的离线逻辑

`eval_pusht_rag.sh` 默认设置：

```bash
COSMOS_HF_CHECKPOINT_ROOT=/DATA1/wxs/_HUGGINGFACE
COSMOS_SKIP_HF_AUTO_DOWNLOAD=1
COSMOS_PREDICT2P5_DISTILLED_CKPT=/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt
COSMOS_PREDICT2P5_TOKENIZER_CKPT=/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth
```

含义：

- 优先从 `/DATA1/wxs/_HUGGINGFACE` 找本地权重。
- 跳过导入无关配置时的 HuggingFace 自动下载。
- PushT 当前需要的 Predict2.5 distilled checkpoint 和 tokenizer 使用本地路径。

如果看到无关模型下载，例如 ALOHA 或 LIBERO checkpoint，通常说明新脚本没有生效，或者旧评测进程仍在运行。先停止旧进程，再重新运行：

```bash
pkill -f cosmos_policy.experiments.robot.pusht_ret.run_eval
NUM_GPUS=1 bash eval_pusht_rag.sh
```

## 10. 常见问题

### 10.1 `natten` wheel 下载超时

报错示例：

```text
Failed to download natten==0.21.0+cu128.torch27
operation timed out
```

这是依赖安装阶段访问 GitHub release 超时。可选处理：

- 配置宿主机或容器网络代理。
- 在网络较好的机器提前下载 wheel，再放入可访问路径。
- 重试 `uv sync --python 3.10 --extra cu128 --group pusht`。

### 10.2 HuggingFace gated repo 无权限

报错示例：

```text
Cannot access gated repo
Access to model nvidia/Cosmos-Predict2.5-2B is restricted
```

处理：

1. 浏览器访问对应 HuggingFace repo，申请访问权限。
2. 宿主机或容器内执行：

```bash
hf auth login
```

3. 重新执行 `hf download ...`。

### 10.3 程序仍然访问 `google-t5/t5-11b`

报错示例：

```text
https://huggingface.co/google-t5/t5-11b/resolve/main/tokenizer_config.json
```

正常 PushT 评测应直接加载：

```bash
/workspace/data/PushT-Cosmos-Policy/success_only/t5_embeddings.pkl
```

如果没有加载成功，程序会现场计算 T5 embedding，从而访问 `google-t5/t5-11b`。

之前的根因是 `/DATA1` 只读挂载导致 lock 文件创建失败：

```text
Read-only file system: '/workspace/data/PushT-Cosmos-Policy/success_only/t5_embeddings.pkl.lock'
```

当前代码已兼容只读路径：如果 `.lock` 无法创建，会直接读取已有 `t5_embeddings.pkl`。新日志里应看到：

```text
Loaded T5 text embeddings from read-only path ...
```

如果仍然联网，检查：

```bash
grep -n "Loaded T5\\|Error loading T5\\|Computing T5" \
  cosmos_policy/experiments/robot/pusht_ret/logs_pusht_rag_eval/residual_top100--tri_default.log
```

### 10.4 `/workspace` 下文件存在，但程序说不存在

常见原因：

- 没有从仓库根目录启动 Docker，导致 `-v "$PWD:/workspace"` 挂错目录。
- 容器内 `/DATA1` 没有挂载，导致 `data/PushT-Cosmos-Policy` symlink 断开。
- 脚本不是最新版，仍使用相对路径。

检查：

```bash
pwd
ls -lh /workspace
ls -lh /workspace/data/PushT-Cosmos-Policy/success_only/dataset_statistics.json
ls -lh /DATA1/wxs/DATASETS/PushT-Cosmos-Policy/success_only/dataset_statistics.json
```

### 10.5 `.venv/bin/python` 在宿主机不可用

这是正常现象。当前 `.venv/bin/python` 指向容器内 uv 管理的 Python：

```bash
/home/cosmos/.local/share/uv/python/...
```

在宿主机上可能不存在，但容器内可用。需要在容器里执行：

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```

或者使用：

```bash
uv run --python 3.10 python -c "import torch; print(torch.__version__)"
```

## 11. 推荐的最短复现流程

宿主机：

```bash
cd /home/wxs/ReCAP-Cosmos-Policy
docker build -t cosmos-policy docker
bash start_docker_byBrad.sh
```

容器内：

```bash
cd /workspace
uv sync --python 3.10 --extra cu128 --group pusht
NUM_GPUS=1 bash eval_pusht_rag.sh
```

确认单 GPU 正常后：

```bash
NUM_GPUS=8 bash eval_pusht_rag.sh
```
