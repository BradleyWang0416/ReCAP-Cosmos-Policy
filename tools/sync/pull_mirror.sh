#!/usr/bin/env bash
# B 层：把远端实验产出中的「小文件」单向拉到本地 .mirror/。
#
# 目的：让训练日志、metrics、eval 结果成为本地真实文件，
# 从而可以被本地工具（Read / Grep / diff）直接读取，而不必每次 ssh 现取。
#
# 单向：远端 -> 本地。本地 .mirror/ 视为只读快照，不入 git。
# 用法：tools/sync/pull_mirror.sh [--dry-run]

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
# shellcheck source=sync.conf
source "$script_dir/sync.conf"

dry_run=()
[ "${1:-}" = "--dry-run" ] && dry_run=(--dry-run) && echo "[dry-run] 只列出将要传输的文件，不实际写入"

mirror_root="$repo_root/.mirror"
mkdir -p "$mirror_root"

# 组装 rsync 过滤规则。顺序要紧：先排除重目录，再放行目录，再放行白名单类型，最后兜底排除。
filters=()
for d in "${MIRROR_EXCLUDE_DIRS[@]}"; do
    filters+=(--exclude="$d/")
done
filters+=(--include='*/')
for pat in "${MIRROR_INCLUDE[@]}"; do
    filters+=(--include="$pat")
done
filters+=(--exclude='*')

total_before=$(find "$mirror_root" -type f 2>/dev/null | wc -l)

for entry in "${MIRROR_ROOTS[@]}"; do
    remote_path="${entry%%:*}"
    local_name="${entry##*:}"
    dest="$mirror_root/$local_name"

    if ! ssh "$REMOTE_HOST" "test -d '$remote_path'"; then
        echo "跳过（远端不存在）: $remote_path"
        continue
    fi

    echo "==> $remote_path  ->  .mirror/$local_name"
    mkdir -p "$dest"
    rsync -az --info=stats1 \
        "${dry_run[@]}" \
        --prune-empty-dirs \
        --max-size="$MIRROR_MAX_SIZE" \
        --delete --delete-excluded \
        "${filters[@]}" \
        "$REMOTE_HOST:$remote_path/" "$dest/"
done

if [ ${#dry_run[@]} -eq 0 ]; then
    total_after=$(find "$mirror_root" -type f 2>/dev/null | wc -l)
    size=$(du -sh "$mirror_root" 2>/dev/null | cut -f1)
    echo
    echo "镜像完成：$total_before -> $total_after 个文件，共 $size"
    echo "位置：$mirror_root（已 gitignore）"
fi
