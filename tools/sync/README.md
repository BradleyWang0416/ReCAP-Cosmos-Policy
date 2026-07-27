# 本地 ↔ 远端服务器 同步方案

本项目在两处存在：

| | 路径 | 角色 |
|---|---|---|
| 本地 | `/home/cc/wxs/ReCAP-Cosmos-Policy` | 读代码、写文档、分析结果 |
| 远端 | `wxs:/home/wxs/ReCAP-Cosmos-Policy` | 跑训练与评测（GPU、数据集在此） |

数据集与权重（`/DATA1/wxs/` 下）体积过大，**永远不同步**；仓库内的软链接也只在远端有效。
为了让本地仍能拥有足够上下文，采用三层结构。

## 三层

| 层 | 内容 | 传输 | 入 git |
|---|---|---|---|
| **A 源码** | `.py` / `.md` / `.json` / 配置 | git（双向，经 GitHub） | ✅ |
| **B 镜像** | 远端产出的小文件：日志、metrics、eval 结果 | `pull_mirror.sh`（远→本地，单向） | ❌ `.mirror/` |
| **C 元信息** | 大文件的**描述**：软链接指向、目录体量、HDF5 字段、episode 数 | `gen_meta.sh` 生成后随 git 走 | ✅ `元信息/` |

关键点：B 层让远端产出变成**本地真实文件**，可以直接 Read / Grep / diff，
不必每次 ssh 现取；C 层让传不动的大文件仍以可读形式存在于本地。

## 首次安装

本地和远端**各执行一次**：

```bash
tools/sync/install_hooks.sh
```

将 `core.hooksPath` 指向 `tools/git-hooks/`，使 pre-commit hook 随 git 同步。

## 日常用法

```bash
# 拉取远端实验产出的小文件到 .mirror/（随时可重跑，幂等）
tools/sync/pull_mirror.sh
tools/sync/pull_mirror.sh --dry-run   # 先看会传什么

# 刷新 元信息/（数据集或软链接变动后跑）
tools/sync/gen_meta.sh
git add 元信息 && git commit -m "刷新元信息" && git push
```

配置集中在 [`sync.conf`](sync.conf)：要增删镜像目录、放宽文件类型白名单、
调整大小保险丝（默认 `2M`），改那个文件即可，不必动脚本。

## 避免历史分叉

两端提交前一律：

```bash
git pull --rebase
```

不要用默认的 merge —— 曾因此产生合并提交 `8533837`，且把 `.gitignore` 的内容
重复贴了一遍。

## pre-commit hook

[`tools/git-hooks/pre-commit`](../git-hooks/pre-commit) 自动把仓库内软链接写入
`.gitignore` 并取消其跟踪。

原先的 `.git/hooks/pre-commit` 有两个缺陷，已在现版本修复：

1. **`grep -qS` 中 `-S` 不是合法选项**，每次都返回 exit 2，`!` 取反后恒为真，
   于是每次 commit 都无条件重复追加一遍条目 —— `.gitignore` 因此从 69 条有效规则
   膨胀到 293 行、7 个条目各重复 4 次。现改用 `git check-ignore` 判定，它同时能
   识别 `.venv/` 这类父目录规则，因此 `.venv/bin/python` 之流不会再被冗余写入。
2. **`for link in $SYMLINKS` 按空格分词**，含空格的路径会被拆断。现用
   `find -print0` 读取，中文与空格路径均安全。
