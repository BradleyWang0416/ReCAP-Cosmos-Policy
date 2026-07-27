#!/usr/bin/env bash
# C 层：把「无法同步的东西」转成「可同步的描述」。
#
# 数据集和权重本身传不过来，但做判断真正需要的是它们的元信息：
# 软链接指向何处、目录多大、HDF5 里有哪些字段、每个任务多少条 episode。
# 这些体积极小，入 git 后本地离线可读。
#
# 本脚本在【本地】运行，通过 ssh 采集远端信息，写入 $META_DIR/。
# 用法：tools/sync/gen_meta.sh

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
# shellcheck source=sync.conf
source "$script_dir/sync.conf"

meta="$repo_root/$META_DIR"
mkdir -p "$meta"
stamp=$(date '+%F %T')

echo "==> 采集软链接清单"
{
    echo "# 远端仓库软链接清单"
    echo
    echo "生成时间：$stamp　来源：\`$REMOTE_HOST:$REMOTE_REPO\`"
    echo
    echo "这些软链接指向仓库外的大文件，不随 git 同步；本文件记录它们的指向，"
    echo "以便在本地离线判断某个路径实际落在哪块存储上。"
    echo
    echo '```'
    ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && find . -type l -not -path './.git/*' -printf '%p -> %l\n' | sort"
    echo '```'
} > "$meta/软链接清单.md"

echo "==> 采集大目录概览"
{
    echo "# 远端大目录概览"
    echo
    echo "生成时间：$stamp"
    echo
    for root in "${META_TREE_ROOTS[@]}"; do
        echo "## \`$root\`"
        echo
        if ssh "$REMOTE_HOST" "test -d '$root'"; then
            echo '```'
            ssh "$REMOTE_HOST" "du -h --max-depth=2 '$root' 2>/dev/null | sort -k2"
            echo '```'
        else
            echo "_（远端不存在）_"
        fi
        echo
    done
} > "$meta/远端目录概览.md"

echo "==> 采集数据集清单"
{
    echo "# Human2Robot 数据集清单"
    echo
    echo "生成时间：$stamp　根目录：\`/DATA1/wxs/DATASETS/Human2Robot/data/v1/\`"
    echo
    echo "| 任务目录 | episode 数 |"
    echo "|---|---:|"
    ssh "$REMOTE_HOST" '
        base=/DATA1/wxs/DATASETS/Human2Robot/data/v1
        for d in "$base"/*/; do
            [ -d "$d" ] || continue
            n=$(find "$d" -maxdepth 1 -name "*.hdf5" | wc -l)
            printf "| %s | %s |\n" "$(basename "$d")" "$n"
        done | sort
    '
} > "$meta/数据集清单.md"

echo "==> dump HDF5 结构"
ssh "$REMOTE_HOST" "
    py='$META_PYTHON'
    [ -x \"\$py\" ] || py=python3
    \"\$py\" - '$META_HDF5_SAMPLE_GLOB' <<'PYEOF'
import json, sys
try:
    import h5py
except ImportError:
    print(json.dumps({'error': 'h5py 不可用，跳过 schema dump'}, ensure_ascii=False)); sys.exit(0)

path = sys.argv[1]
out = {'sample': path, 'datasets': {}, 'attrs': {}}

def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        out['datasets'][name] = {'shape': list(obj.shape), 'dtype': str(obj.dtype)}

try:
    with h5py.File(path, 'r') as f:
        f.visititems(visit)
        out['attrs'] = {k: str(v) for k, v in f.attrs.items()}
except Exception as e:
    out['error'] = f'{type(e).__name__}: {e}'

print(json.dumps(out, ensure_ascii=False, indent=2))
PYEOF
" > "$meta/HDF5结构样本.json" 2>/dev/null || echo '{"error":"采集失败"}' > "$meta/HDF5结构样本.json"

echo
echo "元信息已写入 $META_DIR/："
ls -1sh "$meta"
echo
echo "下一步：git add $META_DIR && git commit && git push"
