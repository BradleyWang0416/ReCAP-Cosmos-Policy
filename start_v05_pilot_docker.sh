#!/usr/bin/env bash
# v05 探索期专用 Docker launcher（E0.4）。
#
# 职责边界（总计划 §2.1）：本脚本只负责在宿主机上挑选 GPU、绑定镜像身份、检查磁盘、
# 建立 run/attempt 记录、启动容器并打开 shell。它**不启动任何计算**。
#
# 用法：
#   V05_PILOT_GPU_COUNT=1 bash start_v05_pilot_docker.sh      # 常规：声明张数，自动选空闲卡
#   V05_PILOT_GPU_COUNT=0 bash start_v05_pilot_docker.sh      # 纯 CPU（E1/E2.0/B1–B3/E5）
#   V05_PILOT_GPU_DEVICES=2,3,6,7 bash start_v05_pilot_docker.sh   # 例外：复现旧运行时显式指定
#
# 可选环境变量：
#   V05_PILOT_RUN_ID              复用某个 run id（会新建 attempt_000N，不覆盖旧 attempt）
#   V05_PILOT_GPU_POLL_SECONDS    等待空闲卡的轮询间隔，默认 30
#   V05_PILOT_GPU_WAIT_SECONDS    等待上限，默认 0 = 无限等待（不得降级为更少卡数）
#   V05_PILOT_ACK_IMAGE_CHANGE=1  确认镜像 ID 已变更且已登记进 PILOT_LOG
#   V05_PILOT_ALLOW_NETWORK=1     §2.2 一次性资产获取例外；默认 --network=none
#   V05_PILOT_NONINTERACTIVE=1    只跑 session 回执与 preflight 后退出，不开 shell、不分配 TTY。
#                                 用于自检与远程非交互调用；**不改变任何检查项或科学语义**，
#                                 preflight 的 blocker 判定与交互模式完全相同。
set -euo pipefail

pilot_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
pilot_run_root="/DATA1/wxs/ReCAP_M5B_V05_PILOT"
pilot_log_root="${pilot_run_root}/orchestrator_logs"
pilot_image="cosmos-policy:latest"
# 总计划 §2.1 登记的镜像 ID。变更必须在 PILOT_LOG 登记后用 V05_PILOT_ACK_IMAGE_CHANGE=1 放行。
pilot_pinned_image_id="sha256:4fc8db9f70eeb96fee271ef282385163ec1da220dfed35da9c832fb6769891e8"
pilot_min_free_bytes=$((100 * 1024 * 1024 * 1024))
pilot_poll_seconds="${V05_PILOT_GPU_POLL_SECONDS:-30}"
pilot_wait_seconds="${V05_PILOT_GPU_WAIT_SECONDS:-0}"

pilot_die() { echo "[v05-pilot][FATAL] $*" >&2; exit 2; }
pilot_warn() { echo "[v05-pilot][WARN ] $*" >&2; }
pilot_info() { echo "[v05-pilot][INFO ] $*"; }

# ---------------------------------------------------------------- GPU 需求解析
# 张数与显式编号二选一。显式编号只用于复现旧运行（§2.4）。
pilot_gpu_devices="${V05_PILOT_GPU_DEVICES:-}"
pilot_gpu_count="${V05_PILOT_GPU_COUNT:-}"
pilot_gpu_selection_mode=""

if [[ -n "$pilot_gpu_devices" && -n "$pilot_gpu_count" ]]; then
  pilot_die "V05_PILOT_GPU_DEVICES 与 V05_PILOT_GPU_COUNT 不能同时设置"
fi
if [[ -n "$pilot_gpu_devices" ]]; then
  [[ "$pilot_gpu_devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || pilot_die "V05_PILOT_GPU_DEVICES 必须是逗号分隔的 GPU 编号"
  pilot_gpu_selection_mode="explicit"
elif [[ -n "$pilot_gpu_count" ]]; then
  [[ "$pilot_gpu_count" =~ ^[0-9]+$ ]] || pilot_die "V05_PILOT_GPU_COUNT 必须是非负整数"
  pilot_gpu_selection_mode="count"
else
  pilot_die "必须设置 V05_PILOT_GPU_COUNT=<n>（可为 0）或 V05_PILOT_GPU_DEVICES=<idx,...>"
fi

# --------------------------------------------------------------- 镜像身份绑定
command -v docker >/dev/null 2>&1 || pilot_die "宿主机找不到 docker"
pilot_image_id="$(docker image inspect "$pilot_image" --format '{{.Id}}' 2>/dev/null || true)"
[[ "$pilot_image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || pilot_die "无法绑定本地 ${pilot_image} 的镜像 ID"

pilot_image_id_matches_pin="true"
if [[ "$pilot_image_id" != "$pilot_pinned_image_id" ]]; then
  pilot_image_id_matches_pin="false"
  if [[ "${V05_PILOT_ACK_IMAGE_CHANGE:-0}" != "1" ]]; then
    echo "[v05-pilot][FATAL] 镜像 ID 与总计划 §2.1 登记值不一致：" >&2
    echo "                   登记值 = ${pilot_pinned_image_id}" >&2
    echo "                   实际值 = ${pilot_image_id}" >&2
    echo "                   按 §2.1，镜像 ID 变化必须先登记进 PILOT_LOG，" >&2
    echo "                   然后用 V05_PILOT_ACK_IMAGE_CHANGE=1 放行。" >&2
    exit 2
  fi
  pilot_warn "镜像 ID 已变更并被显式放行；请确认已登记进 PILOT_LOG（实际值 ${pilot_image_id}）"
fi

# ------------------------------------------------------------------- 磁盘检查
[[ -d /DATA1 ]] || pilot_die "/DATA1 不存在"
pilot_free_bytes="$(df -B1 --output=avail /DATA1 | tail -1 | tr -d ' ')"
pilot_free_gib=$((pilot_free_bytes / 1024 / 1024 / 1024))
pilot_storage_status="passed"
if (( pilot_free_bytes < pilot_min_free_bytes )); then
  pilot_storage_status="failed"
  pilot_warn "/DATA1 可用 ${pilot_free_gib} GiB < 100 GiB 门槛；容器内 preflight 将返回 BLOCKED_ENVIRONMENT"
elif (( pilot_free_bytes < pilot_min_free_bytes + 50 * 1024 * 1024 * 1024 )); then
  pilot_warn "/DATA1 可用 ${pilot_free_gib} GiB，高出 100 GiB 门槛不足 50 GiB。"
  pilot_warn "  E4 短训每方法约需 11 GiB checkpoint；按总计划 §四 E0.2 应先追加清理。"
else
  pilot_info "/DATA1 可用 ${pilot_free_gib} GiB"
fi

# --------------------------------------------------------------- 空闲 GPU 探测
# 判定规则（§2.4）：该卡上无其他用户的 compute process，且已用显存 < 总显存的 10%。
pilot_current_user="$(id -un)"

pilot_free_gpu_indices() {
  # 输出：每行一个空闲 GPU 的 index
  local apps busy_uuids="" line uuid pid owner
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    uuid="$(echo "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')"
    pid="$(echo "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')"
    owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    # owner 为空表示进程已退出或不可见；保守起见按"他人占用"处理。
    if [[ "$owner" != "$pilot_current_user" ]]; then
      busy_uuids="${busy_uuids} ${uuid}"
    fi
  done <<< "$apps"

  nvidia-smi --query-gpu=index,uuid,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local idx guuid total used
      idx="$(echo "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
      guuid="$(echo "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')"
      total="$(echo "$line" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
      used="$(echo "$line" | awk -F',' '{gsub(/ /,"",$4); print $4}')"
      [[ " ${busy_uuids} " == *" ${guuid} "* ]] && continue
      # 已用显存 < 总显存 10%
      (( used * 10 < total )) || continue
      echo "$idx"
    done
}

pilot_selected_indices=()
if [[ "$pilot_gpu_selection_mode" == "explicit" ]]; then
  IFS=',' read -r -a pilot_selected_indices <<< "$pilot_gpu_devices"
  pilot_info "显式指定 GPU：${pilot_gpu_devices}（复现模式，跳过空闲判定）"
elif (( pilot_gpu_count == 0 )); then
  pilot_info "声明 0 张 GPU（纯 CPU 环节）"
else
  command -v nvidia-smi >/dev/null 2>&1 || pilot_die "需要 GPU 但宿主机找不到 nvidia-smi"
  pilot_waited=0
  while true; do
    mapfile -t pilot_free_list < <(pilot_free_gpu_indices)
    if (( ${#pilot_free_list[@]} >= pilot_gpu_count )); then
      pilot_selected_indices=("${pilot_free_list[@]:0:$pilot_gpu_count}")
      break
    fi
    # §2.4：不得自动降级为更少卡数或改用 CPU，只能等待。
    echo "[v05-pilot][HEARTBEAT] $(date -u +%Y-%m-%dT%H:%M:%SZ) 需要 ${pilot_gpu_count} 张空闲卡，当前 ${#pilot_free_list[@]} 张；等待中（已等 ${pilot_waited}s）"
    if (( pilot_wait_seconds > 0 && pilot_waited >= pilot_wait_seconds )); then
      pilot_die "等待空闲 GPU 超时（${pilot_wait_seconds}s）。不降级为更少卡数，直接退出。"
    fi
    sleep "$pilot_poll_seconds"
    pilot_waited=$((pilot_waited + pilot_poll_seconds))
  done
  pilot_gpu_devices="$(IFS=','; echo "${pilot_selected_indices[*]}")"
  pilot_info "已选中空闲 GPU：${pilot_gpu_devices}"
fi

# ------------------------------------------- GPU 映射（宿主机编号 → 容器逻辑编号）
pilot_gpu_map_json="[]"
if (( ${#pilot_selected_indices[@]} > 0 )); then
  pilot_gpu_map_json="$(
    logical=0
    printf '['
    for idx in "${pilot_selected_indices[@]}"; do
      row="$(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits -i "$idx" 2>/dev/null || true)"
      [[ -z "$row" ]] && pilot_die "无法读取 GPU ${idx} 的信息"
      guuid="$(echo "$row" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')"
      gname="$(echo "$row" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')"
      gmem="$(echo "$row" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
      (( logical > 0 )) && printf ','
      printf '{"host_index":%s,"container_index":%s,"uuid":"%s","name":"%s","memory_total_mib":%s}' \
        "$idx" "$logical" "$guuid" "$gname" "$gmem"
      logical=$((logical + 1))
    done
    printf ']'
  )"
  pilot_info "GPU 映射：$pilot_gpu_map_json"
fi

# ---------------------------------------------------------- run / attempt 记录
pilot_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
pilot_started_compact="$(date -u +%Y%m%dT%H%M%SZ)"
pilot_run_id="${V05_PILOT_RUN_ID:-pilot_session_${pilot_started_compact}_$$}"
pilot_run_dir="${pilot_log_root}/${pilot_run_id}"
mkdir -p "$pilot_run_dir" || pilot_die "无法创建 run 目录：${pilot_run_dir}"

# 重试必须新建整套 attempt_000N，不得覆盖或追加进旧 attempt（§2.5）。
pilot_attempt=1
while [[ -e "${pilot_run_dir}/attempt_$(printf '%04d' "$pilot_attempt").runtime.json" ]]; do
  pilot_attempt=$((pilot_attempt + 1))
done
pilot_attempt_tag="attempt_$(printf '%04d' "$pilot_attempt")"
pilot_attempt_prefix="${pilot_run_dir}/${pilot_attempt_tag}"

pilot_network_mode="none"
if [[ "${V05_PILOT_ALLOW_NETWORK:-0}" == "1" ]]; then
  pilot_network_mode="bridge"
  pilot_warn "已启用网络（§2.2 一次性资产获取例外）。本次 session 必须单独登记进 PILOT_LOG。"
fi

pilot_container_name="recap_v05_pilot_${pilot_started_compact,,}_$$"

# 原子落盘：先写 .partial，再 rename（§2.6）。
pilot_write_atomic() {
  local target="$1"
  local tmp="${target}.$$.partial"
  cat > "$tmp"
  mv -f "$tmp" "$target"
}

pilot_write_atomic "${pilot_attempt_prefix}.command.txt" <<EOF
${BASH_SOURCE[0]}
V05_PILOT_GPU_COUNT=${V05_PILOT_GPU_COUNT:-}
V05_PILOT_GPU_DEVICES=${V05_PILOT_GPU_DEVICES:-}
selected_gpu_devices=${pilot_gpu_devices}
V05_PILOT_RUN_ID=${pilot_run_id}
V05_PILOT_ALLOW_NETWORK=${V05_PILOT_ALLOW_NETWORK:-0}
V05_PILOT_ACK_IMAGE_CHANGE=${V05_PILOT_ACK_IMAGE_CHANGE:-0}
EOF

pilot_write_atomic "${pilot_attempt_prefix}.runtime.json" <<EOF
{
  "schema_version": "human2robot-v05-pilot-launcher-v1",
  "run_id": "${pilot_run_id}",
  "attempt": ${pilot_attempt},
  "started_at_utc": "${pilot_started_utc}",
  "host": {
    "hostname": "$(hostname)",
    "user": "${pilot_current_user}",
    "uid": $(id -u),
    "gid": $(id -g)
  },
  "image": {
    "reference": "${pilot_image}",
    "id": "${pilot_image_id}",
    "pinned_id": "${pilot_pinned_image_id}",
    "matches_pinned_id": ${pilot_image_id_matches_pin},
    "change_acknowledged": ${V05_PILOT_ACK_IMAGE_CHANGE:-0}
  },
  "container": {
    "name": "${pilot_container_name}",
    "network": "${pilot_network_mode}",
    "ipc": "host",
    "workdir": "/workspace"
  },
  "gpu": {
    "selection_mode": "${pilot_gpu_selection_mode}",
    "requested_count": "${pilot_gpu_count}",
    "selected_devices": "${pilot_gpu_devices}",
    "mapping": ${pilot_gpu_map_json}
  },
  "storage": {
    "path": "/DATA1",
    "free_bytes": ${pilot_free_bytes},
    "minimum_free_bytes": ${pilot_min_free_bytes},
    "status": "${pilot_storage_status}"
  },
  "mounts": [
    {"source": "${pilot_repo_root}", "target": "/workspace", "mode": "rw"},
    {"source": "/DATA1", "target": "/DATA1", "mode": "rw"},
    {"source": "/DATA1/wxs/DATASETS/Human2Robot/data/v1", "target": "/DATA1/wxs/DATASETS/Human2Robot/data/v1", "mode": "ro"},
    {"source": "/DATA1/wxs/ReCAP_M5B_P2_RUNS", "target": "/DATA1/wxs/ReCAP_M5B_P2_RUNS", "mode": "ro"},
    {"source": "/DATA1/wxs/ReCAP_M5B_V04_RUNS", "target": "/DATA1/wxs/ReCAP_M5B_V04_RUNS", "mode": "ro"},
    {"source": "/DATA1/wxs/_HUGGINGFACE", "target": "/DATA1/wxs/_HUGGINGFACE", "mode": "ro"}
  ]
}
EOF

pilot_write_atomic "${pilot_run_dir}/latest_log.json" <<EOF
{"run_id": "${pilot_run_id}", "attempt": ${pilot_attempt}, "log_path": "${pilot_attempt_prefix}.log", "started_at_utc": "${pilot_started_utc}"}
EOF

pilot_write_atomic "${pilot_attempt_prefix}.status.json" <<EOF
{"run_id": "${pilot_run_id}", "attempt": ${pilot_attempt}, "status": "RUNNING", "started_at_utc": "${pilot_started_utc}"}
EOF
cp -f "${pilot_attempt_prefix}.status.json" "${pilot_run_dir}/status.json"

pilot_write_atomic "${pilot_attempt_prefix}.progress.json" <<EOF
{"run_id": "${pilot_run_id}", "attempt": ${pilot_attempt}, "current_unit": "docker_session", "updated_at_utc": "${pilot_started_utc}"}
EOF
cp -f "${pilot_attempt_prefix}.progress.json" "${pilot_run_dir}/progress.json"

# 启动后立即输出 run ID、时间、命令、日志绝对路径、产物根、代码哈希、GPU 映射（§2.5）。
pilot_code_sha="$(cd "$pilot_repo_root" && git rev-parse HEAD 2>/dev/null || echo "unavailable")"
cat <<EOF
================ v05 pilot session ================
run_id        : ${pilot_run_id}
attempt       : ${pilot_attempt_tag}
started_utc   : ${pilot_started_utc}
log_path      : ${pilot_attempt_prefix}.log
run_root      : ${pilot_run_root}
code_sha      : ${pilot_code_sha}
image_id      : ${pilot_image_id}
gpu_devices   : ${pilot_gpu_devices:-<none>}
gpu_mapping   : ${pilot_gpu_map_json}
network       : ${pilot_network_mode}
===================================================
EOF

# ------------------------------------------------------------------- 容器启动
mkdir -p "$HOME/.cache" "$HOME/.local/share/uv" "$pilot_run_root"

pilot_interactive=1
[[ "${V05_PILOT_NONINTERACTIVE:-0}" == "1" ]] && pilot_interactive=0

pilot_docker_args=(run --rm)
(( pilot_interactive == 1 )) && pilot_docker_args+=(-it)
pilot_docker_args+=(
  --pull=never
  --name "$pilot_container_name"
  --ipc=host
  -e HOST_USER_ID="$(id -u)"
  -e HOST_GROUP_ID="$(id -g)"
  -e V05_PILOT_IMAGE="$pilot_image"
  -e V05_PILOT_IMAGE_ID="$pilot_image_id"
  -e V05_PILOT_CONTAINER_NAME="$pilot_container_name"
  -e V05_PILOT_RUN_ID="$pilot_run_id"
  -e V05_PILOT_ATTEMPT="$pilot_attempt"
  -e V05_PILOT_GPU_DEVICES="${pilot_gpu_devices:-}"
  -e V05_PILOT_GPU_MAP_JSON="$pilot_gpu_map_json"
  -e V05_PILOT_NETWORK_MODE="$pilot_network_mode"
  -e RECAP_WORKSPACE=/workspace
  -e HUMAN2ROBOT_ROOT=/DATA1/wxs/DATASETS/Human2Robot/data/v1
  -e V05_PILOT_RUN_ROOT="$pilot_run_root"
  -e COSMOS_HF_CHECKPOINT_ROOT=/DATA1/wxs/_HUGGINGFACE
  -e COSMOS_SKIP_HF_AUTO_DOWNLOAD=1
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e WANDB_MODE=disabled
  -e WANDB_DISABLED=true
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e NCCL_DEBUG=WARN
  # /DATA1 整体 rw，但把只读资产以 ro 覆盖挂载在其上，把 §2.3 的权限约定变成强制约束。
  -v "$pilot_repo_root:/workspace:rw"
  -v /DATA1:/DATA1:rw
  -v /DATA1/wxs/DATASETS/Human2Robot/data/v1:/DATA1/wxs/DATASETS/Human2Robot/data/v1:ro
  -v /DATA1/wxs/ReCAP_M5B_P2_RUNS:/DATA1/wxs/ReCAP_M5B_P2_RUNS:ro
  -v /DATA1/wxs/ReCAP_M5B_V04_RUNS:/DATA1/wxs/ReCAP_M5B_V04_RUNS:ro
  -v /DATA1/wxs/_HUGGINGFACE:/DATA1/wxs/_HUGGINGFACE:ro
  -v "$HOME/.cache:/home/cosmos/.cache:rw"
  -v "$HOME/.local/share/uv:/home/cosmos/.local/share/uv:rw"
  -w /workspace
)

if [[ "$pilot_network_mode" == "none" ]]; then
  pilot_docker_args+=(--network=none)
fi
if [[ -n "${pilot_gpu_devices:-}" ]]; then
  pilot_docker_args+=(--gpus "\"device=${pilot_gpu_devices}\"")
fi

# 两个容器内命令各用独立 run_id，使 attempt 文件名保持 §2.5 规定的 attempt_000N.*，
# 且不覆盖本 session run 目录下的 latest_log.json / status.json。
pilot_inner_cmd=".venv/bin/python tools/pilot/v05_pilot_session.py session-receipt --run-id '${pilot_run_id}_session_receipt'; \
 .venv/bin/python tools/pilot/v05_pilot_session.py preflight --run-id '${pilot_run_id}_preflight' || true"
if (( pilot_interactive == 1 )); then
  # 交互模式：preflight 后把控制权交给人；脚本本身不启动任何计算（§2.1）。
  pilot_inner_cmd="${pilot_inner_cmd}; exec bash --noprofile --norc"
fi

pilot_docker_args+=("$pilot_image" bash --noprofile --norc -c "$pilot_inner_cmd")

set +e
docker "${pilot_docker_args[@]}" 2>&1 | tee -a "${pilot_attempt_prefix}.log"
pilot_exit=${PIPESTATUS[0]}
set -e

pilot_finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
pilot_final_status="COMPLETED"
(( pilot_exit != 0 )) && pilot_final_status="FAILED"

pilot_write_atomic "${pilot_attempt_prefix}.status.json" <<EOF
{"run_id": "${pilot_run_id}", "attempt": ${pilot_attempt}, "status": "${pilot_final_status}", "exit_code": ${pilot_exit}, "started_at_utc": "${pilot_started_utc}", "finished_at_utc": "${pilot_finished_utc}"}
EOF
cp -f "${pilot_attempt_prefix}.status.json" "${pilot_run_dir}/status.json"

pilot_info "session ${pilot_final_status}（exit=${pilot_exit}），日志：${pilot_attempt_prefix}.log"
exit "$pilot_exit"
