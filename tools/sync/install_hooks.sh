#!/usr/bin/env bash
# 让本仓库使用受版本控制的 hooks（tools/git-hooks/），而不是 .git/hooks/。
# 本地和远端各跑一次即可；之后 hook 的改动随 git 同步。
#
# 用法：tools/sync/install_hooks.sh

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

git config core.hooksPath tools/git-hooks
chmod +x tools/git-hooks/* 2>/dev/null || true

# 旧的 .git/hooks/pre-commit 已被 core.hooksPath 架空，改名留档避免混淆
if [ -f .git/hooks/pre-commit ]; then
    mv .git/hooks/pre-commit .git/hooks/pre-commit.superseded
    echo "已停用旧 hook：.git/hooks/pre-commit -> .git/hooks/pre-commit.superseded"
fi

echo "core.hooksPath = $(git config core.hooksPath)"
echo "生效的 pre-commit：$repo_root/tools/git-hooks/pre-commit"
