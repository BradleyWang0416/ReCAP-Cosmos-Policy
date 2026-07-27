# RECAP × Human2Robot v05 探索期执行总计划

本文件是 v05 冻结确认实验**之前**的探索期总计划，覆盖 E0～E6 六个环节的目标、判据、否决条件，以及全部与实现细节无关但对运行必要的通用规范。

配套文件：

- `方案/v05_pilot/PILOT_LOG.md` — 探索期流水账，记录每一条带基线的数字；
- `tools/pilot/` — 探索期只读诊断脚本与 `README.md`；
- `方案/v04/阶段5后_科学前提偏差记录_20260727.md` — 触发本探索期的四项偏差；
- `方案/v04/CHANGELOG.md` — v04 变更记录（只追加）。

---

## 一、定位与边界

### 1.1 为什么有探索期

v03 建了 203-cell 矩阵后停在 4/203；v04 建了七阶段冻结协议后跑到阶段 5 的 2/3，才发现「人手池」实际是机器人指令池、且检索在 K=8 下是净负收益。两次的共同点不是协议不够严，而是**协议投入永远发生在科学前提被验证之前**。

探索期的唯一职责，是在冻结任何东西之前把科学前提钉死。

### 1.2 探索期的产出与非产出

**产出**：一份可回答六个问题的前提结论（见 §四 E6），以及支撑它的、可复现的数字。

**非产出**：不发验收报告，不签发 immutable receipt，不冻结协议 SHA，不产生任何可引用的性能声明。探索期的一切结论在进入 v05 冻结协议前必须在冻结容器内复算一次（见 §2.1）。

### 1.3 与 v04 的关系

v04 的**运行基建**整体保留并复用：Docker launcher 模式、离线 preflight、attempt 级日志、原子落盘、immutable receipt、来源身份与 partition 隔离、task-balanced sampler、测试骨架。这些未被任何偏差证伪。

v04 的**冻结产物**（阶段 1 投影/manifest、geometry 统计、WAN cache、smoke plan、两个 step-7000 checkpoint）在探索期一律**只读引用**，不修改、不删除、不作为 v05 输入资格。

v04 的运行根 `/DATA1/wxs/ReCAP_M5B_P2_RUNS` 与 `/DATA1/wxs/ReCAP_M5B_V04_RUNS` 在探索期**只读**。

---

## 二、统一运行、环境与可审计性规范

本节适用于探索期的所有数据处理、测试、推理和训练命令，不因某个脚本能在宿主机启动而豁免。

### 2.1 运行入口和 Docker 边界

所有需要 GPU、加载模型权重、或产生将写入 v05 冻结协议的数字的程序，**必须**在 `cosmos-policy:latest` 完整镜像中运行。当前镜像 ID 为 `sha256:4fc8db9f70eeb96fee271ef282385163ec1da220dfed35da9c832fb6769891e8`；镜像 ID 变化必须在 PILOT_LOG 中登记。

宿主机只负责启动 Docker、检查磁盘与 GPU、查看日志、停止容器。

探索期把执行分为两类，规则不同：

| 类别 | 定义 | 允许的运行方式 | 义务 |
|---|---|---|---|
| **A：只读 CPU 诊断** | 只读原始数据与 v04 冻结产物；不占 GPU；不加载模型权重；只写探索期临时根 | 允许宿主机 Python | 必须标记 `NONFORMAL_DIAGNOSTIC`，在 `tools/pilot/README.md` 与 PILOT_LOG 中披露；结论进入 v05 冻结协议前必须在容器内复算一次 |
| **B：其余全部** | 任何 GPU 任务、任何模型推理或训练、任何写入探索期正式产物的操作 | **仅限容器内** | 完整 preflight、attempt 级日志、receipt |

`tools/pilot/` 下现有的 16 个脚本全部属于类别 A，并已在其 README 中披露不满足 Docker-only 约束。E3、E4 属于类别 B。

标准启动方式（E0.4 交付后）：

```bash
cd /home/wxs/ReCAP-Cosmos-Policy
V05_PILOT_GPU_COUNT=1 bash start_v05_pilot_docker.sh
```

该脚本只打开完整环境的 shell，不自动启动任何计算。容器内从 `/workspace` 执行具体命令。

专用容器必须满足：

- 宿主机 `/home/wxs/ReCAP-Cosmos-Policy` 以读写方式绑定为容器 `/workspace`；
- `/DATA1` 以读写方式绑定，但探索期所有新写入严格限制在 `/DATA1/wxs/ReCAP_M5B_V05_PILOT`；
- `/home/wxs/.cache` 与 `/home/wxs/.local/share/uv` 分别绑定为容器用户的缓存与 uv 根；
- 工作目录为 `/workspace`，使用仓库内 `/workspace/.venv`；
- 使用 `--ipc=host`；
- 默认 `--network=none`；仅在 §2.2 允许的一次性资产获取时例外，且必须单独记录；
- 容器名、镜像 ID、挂载读写属性、用户 UID/GID、实际分配的 GPU 与启动时间写入运行回执。

### 2.2 完整环境、离线和禁止降级

正式运行不得执行 `docker build`、`uv sync`、`pip install`、`hf download`，也不得在程序内触发 Hugging Face 自动下载。

容器统一设置并在回执中记录：

```bash
RECAP_WORKSPACE=/workspace
HUMAN2ROBOT_ROOT=/DATA1/wxs/DATASETS/Human2Robot/data/v1
V05_PILOT_RUN_ROOT=/DATA1/wxs/ReCAP_M5B_V05_PILOT
COSMOS_HF_CHECKPOINT_ROOT=/DATA1/wxs/_HUGGINGFACE
COSMOS_SKIP_HF_AUTO_DOWNLOAD=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
WANDB_MODE=disabled
WANDB_DISABLED=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**严禁**为了"先跑起来"采用下列残次版本：

- 改用宿主机 Python、CPU、错误 CUDA 环境或未登记镜像跑类别 B 任务；
- 自动下载、临时替换、跳过或随机初始化缺失权重；
- 静默减少 GPU、数据、query、step、分辨率、top-k 或模型模块；
- 用 synthetic、截断样本、旧 cache 或部分输出冒充完整运行；
- 在依赖或扩展加载失败后切换到语义不同的 fallback 实现；
- 把诊断运行或提前 checkpoint 的结果写进结论表。

少量数据或较少 GPU 只可用于明确标记 `NONFORMAL_DIAGNOSTIC` 的诊断，须使用独立目录与日志，且不能解除任何 blocker。

**资产缺失时的唯一合法路径**：停止、在 PILOT_LOG 登记缺失资产、由人工在容器外一次性获取并落到 `/DATA1/wxs/_HUGGINGFACE` 对应位置、记录来源与 SHA256，然后恢复离线运行。不得让程序自己联网。E3 的 DINOv2 权重目前正处于这一状态（见 §四 E3.0）。

### 2.3 路径和资产登记

所有配置与回执使用解析后的绝对路径。

| 资产 | 固定位置 | 权限 |
|---|---|---|
| 宿主机仓库 | `/home/wxs/ReCAP-Cosmos-Policy` | rw |
| 容器仓库 | `/workspace` | rw |
| Human2Robot 原始数据 | `/DATA1/wxs/DATASETS/Human2Robot/data/v1` | **只读** |
| v04 冻结派生数据 | `/workspace/data/Human2Robot/derived/v04` | **只读** |
| v04 冻结特征缓存 | `/DATA1/wxs/ReCAP_M5B_V04_RUNS/features` | **只读** |
| v03 / v04 历史运行根 | `/DATA1/wxs/ReCAP_M5B_P2_RUNS`、`/DATA1/wxs/ReCAP_M5B_V04_RUNS` | **只读** |
| 本地模型资产根 | `/DATA1/wxs/_HUGGINGFACE` | 只读 |
| **探索期运行根** | `/DATA1/wxs/ReCAP_M5B_V05_PILOT` | rw（唯一写入边界） |
| 探索期日志根 | `/DATA1/wxs/ReCAP_M5B_V05_PILOT/orchestrator_logs` | rw |
| 类别 A 临时产物 | `/tmp`（缓存）、`tools/pilot/`（脚本） | rw |

模型资产登记（preflight 必须检查存在、可读、大小与 SHA256，不能只检查目录名）：

| 用途 | 本地文件 | 状态 |
|---|---|---|
| Cosmos Predict2.5 2B 初始化 | `/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-…_ema_bf16.pt` | ✅ 已登记 |
| Predict2.5 tokenizer/VAE | `/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth` | ✅ 已登记 |
| SAM ViT-H | `/DATA1/wxs/_HUGGINGFACE/sam/sam_vit_h_4b8939/sam_vit_h_4b8939.pth` | ✅ 本地可用（2.56 GB） |
| DINOv2 | — | ❌ **本地缺失**，E3 阻塞项 |
| V-JEPA2 ViT-L | `/DATA1/wxs/_HUGGINGFACE/facebook/vjepa2-vitl-fpc64-256` | ✅ 本地可用（备选） |
| Depth-Anything | `/DATA1/wxs/_HUGGINGFACE/depth-anything` | ✅ 本地可用（其骨干为 DINOv2，可作备选来源） |

探索期使用 zero text embedding，`google-t5/t5-11b` 不是依赖。若日志出现联网访问或现场计算 T5 embedding，应立即失败并排查配置。

### 2.4 GPU 需求与选择

**按需求张数声明，不绑定具体编号。** 每项任务只声明需要几张空闲 GPU；启动器从当前空闲卡中自动选择，任何空闲卡都可以用。

各环节 GPU 需求：

| 环节 | GPU 张数 | 说明 |
|---|---:|---|
| E1、E2（B1–B3）、E5 | **0** | 纯 CPU |
| E3 | **1** | 只做特征提取推理，显存需求小 |
| E4 | **4** | 短训，需要一组四卡；不要求是相邻或特定编号 |

"空闲"的判定规则（启动器实现，写入回执）：

- 该卡上无其他用户的 compute process；
- 已用显存低于总显存的 10%；
- 若空闲卡数不足声明张数，则**等待资源**，不得自动降级为更少卡数或改用 CPU。

启动器接口：

```bash
# 常规：声明张数，自动挑选任意空闲卡
V05_PILOT_GPU_COUNT=4 bash start_v05_pilot_docker.sh

# 例外：需要复现某次运行时才显式指定
V05_PILOT_GPU_DEVICES=2,3,6,7 bash start_v05_pilot_docker.sh
```

启动日志与回执必须同时写出：宿主机物理编号、GPU UUID、型号、显存、以及容器内逻辑编号（Docker 会把所选卡重编号为 `0..N-1`）。多卡训练固定使用 `torchrun --nproc_per_node=<N>`。

复现旧运行时以回执中记录的 UUID 为准，而非编号——编号会随其他用户的作业变化。

### 2.5 实时日志和运行状态

每次容器 session、以及每个类别 B 命令，都必须先创建唯一 run ID 与 attempt 记录：

```text
/DATA1/wxs/ReCAP_M5B_V05_PILOT/orchestrator_logs/<run_id>/
  attempt_0001.log
  attempt_0001.command.txt
  attempt_0001.runtime.json
  attempt_0001.status.json
  attempt_0001.progress.json
  latest_log.json
  status.json
  progress.json
```

重试必须新建整套 `attempt_0002.*`，不得覆盖或追加进旧 attempt。`latest_log.json` 与汇总状态原子更新。

统一入口必须启用行缓冲输出，并用 `tee` 同时写终端和文件，同时保留被执行程序的真实退出码。**禁止**只在终端显示、重定向到 `/dev/null`、或等程序退出后才一次性写日志。

程序启动后必须立即输出：run ID、开始时间、完整命令、**日志绝对路径**、产物根、代码哈希、GPU 映射。长任务至少每 60 秒输出 heartbeat；训练至少每 10 个 optimizer step 输出 step、loss、LR、耗时与显存摘要；长预处理至少每完成一个 shard 输出计数。

默认日志级别：项目 `INFO`、第三方 `WARNING`、`NCCL_DEBUG=WARN`。不得默认启用 `NCCL_DEBUG_SUBSYS`、逐样本 tensor dump 或完整环境变量 dump。重复告警应限频并在末尾汇总。需要 DEBUG 时使用独立诊断 attempt。

运行状态以 `status.json`、`progress.json`、最新日志和已验证产物**共同**判定，不能只看 PID。`ps -p <pid>` 为空只说明该 PID 不存在，不足以区分正常完成、失败或容器退出。

人工检查最小流程：读 `latest_log.json` → 对其中绝对路径 `tail -F` → 核对 `status.json` 与最新完成的产物。只有退出码为 0、预期产物齐全且校验通过，状态才可写为 `COMPLETED`。

### 2.6 原子落盘、断点续跑和完整性

所有长耗时预处理和推理必须按可验证的 **episode / shard / query 分片**保存进度。单次 `np.savez_compressed` 或 `pickle.dump` 的最终文件不能作为唯一心跳；程序必须在 `progress.json` 中持续更新已完成数量、总数、当前单元和更新时间。

文件先写到同文件系统的 `.partial` 临时路径，完成 flush、关闭、重新读取校验和哈希后再原子 `rename`。`.partial`、缺失 footer、哈希不符或回执不完整的文件不得进入下游。

重启时只复用通过 schema、输入哈希、配置哈希和内容校验的完整单元；失败单元从最近有效边界续跑，不得为了方便重算或覆盖无关的完整证据。

训练恢复必须验证方法、seed、optimizer step、模型/optimizer 状态、配置、数据、feature、world size 和代码哈希。任何不匹配都 hard-fail，不能只加载模型权重就宣称"断点续训"。

对类别 A 脚本，同样要求：中间缓存（如 `/tmp/h2r_*.pkl`）必须可由脚本重跑重建，且 `README.md` 必须写明依赖链与执行顺序。

### 2.7 变更记录和文档同步

任何代码、配置、路径、GPU 策略、日志、缓存、重试或判据调整，都必须在**下一次受影响运行之前**更新相关文档。不允许运行事实长期领先于文档。

每次变更至少记录：

- 时间、原因、提出者和实施者；
- 修改前后值、涉及文件和 Git SHA；
- 影响的数据、cache、checkpoint、评估与结论；
- 是否改变科学语义、是否使旧产物失效；
- 已执行验证、结果和剩余 blocker。

文档职责固定为：

| 文档 | 职责 | 写入规则 |
|---|---|---|
| 本文件 | 探索期目标、规范、判据、进度 | 可修订，修订记入 §七 |
| `方案/v05_pilot/PILOT_LOG.md` | 每一条带基线的数字与其结论 | 追加条目；已发布条目的数字若更正，须显式标注新旧对照 |
| `tools/pilot/README.md` | 脚本执行顺序、依赖链、合规披露、性能 | 随脚本同步 |
| `方案/v04/CHANGELOG.md` | v04 侧的变更与偏差 | **只追加**，不改写既有条目 |

发现文档与实际运行不一致时，立即暂停新任务，保存现场并生成偏差记录；修正文档和实现后才能继续，不得事后补写成"原本就是这样"。

数字口径更正的处理方式：可修订文档直接改正，只追加文档另起条目登记新旧对照。**结论方向发生变化的更正，必须同时更新 §五 进度表。**

---

## 三、统一指标与判据

### 3.1 主指标

对齐后残差 `|robot_future − aligned_plan|`（位置，mm），即 RECAP 的残差学习目标。

- **基线**：不检索 / 保持当前位姿（把 plan 当作"机器人不动"）；
- 记 `ρ = 残差 / 基线`。`ρ < 1` 表示检索为正收益，`ρ > 1` 表示检索比不检索更差；
- **oracle 上界**：在候选池全部合法窗口中用真实未来挑最优（不可实现，仅作天花板参考）。

### 3.2 纪律

1. 每条结论必须给出**一个数字**和**一个基线**。没有基线的数字不进 PILOT_LOG。
2. 允许小数据、单卡、脏代码。不允许无基线的定性判断。
3. 每条记录必须能由 `tools/pilot/` 下的脚本重跑。
4. 探索期结论不得引用为正式结果。

### 3.3 数据使用纪律

探索期用于**规则选择**的评估集，必须是 seen-validation 或 dev；**不得打开 v04 的 final 集**。若因故打开，v05 必须重新划分 final，并在 PILOT_LOG 与 §七 显式登记。

---

## 四、分项实施 E0～E6

### E0 — 现场保全与资源（前置）

| 编号 | 内容 | 判据 | 状态 |
|---|---|---|---|
| E0.1 | 诊断脚本归档进 `tools/pilot/`，建立 `PILOT_LOG.md` | 脚本可重跑并复现记录中的数字 | ✅ **已完成**（2026-07-27，commit `d85425a`；随后完成并行化与 GEMM 改造，输出逐位不变） |
| E0.2 | `/DATA1` 清理至 ≥ 300 GiB | 正式 preflight 不再 `BLOCKED_ENVIRONMENT` | ⬜ 未完成（当前 171 GiB） |
| E0.3 | 明确 GPU 可用时间窗 | E3 能拿到 1 张卡、E4 能拿到 4 张卡 | ⬜ 未完成（8 卡全占，其中 6 张为本人 LlamaFactory 作业） |
| E0.4 | 交付 `start_v05_pilot_docker.sh`（按张数自动选卡，见 §2.4） | 能以 `V05_PILOT_GPU_COUNT=N` 启动并在回执写出实际 GPU UUID | ⬜ 未开始（E3 的前置） |

### E1 — 锁定 K 与检索规则（GPU 0 张）

| 步骤 | 内容 | 状态 |
|---|---|---|
| E1.1 | 把消融从 160 个 dev query 扩到 **seen-validation** 全量（82 episode）。**不得使用 final 集**，见 §3.3 | ⬜ |
| E1.2 | K ∈ {16, 32, 48, 64, 96} × 特征 {geom, abs-pos, geometry-only 加权变体} × 窗口 {phase, search} | 🟡 首轮已在 K ∈ {8,16,32,64} 上完成 36 格 |
| E1.3 | 训练侧同口径测量（seen-train 内跨 episode 检索） | ⬜ |

**判据**：存在配置使 `ρ ≤ 0.75`，且训练侧与评估侧 `ρ` 差距 < 0.1。
**当前**：K=32 时现配置已达 0.84、最佳 0.78；K=64 时最佳 0.68。K=8 下 36 格无一取得实质正收益（最好 0.96）。
**产出**：选定的 (K, 特征, 窗口规则) 三元组。

### E2 — 真人手通道能否既是人手又有效（GPU 0 张；B4 需 1 张）**← 否决项**

按成本递增试，任一档达标即停：

| 档 | 方案 | 状态 |
|---|---|---|
| B0 | 全局相似变换 | ❌ **已否决**：ρ = 0.95–0.96 |
| B1 | 逐 seen-task 相似变换，对 held-out 任务外推（拟合只能用 seen-train，不得触碰 held-out 机器人数据） | ⬜ |
| B2 | 在 seen-train 上监督学一个人手窗口 → 机器人窗口的小 MLP | ⬜ |
| B3 | 内蕴手部描述子（相对手腕的关节构型 + 手腕速度方向 + 开合度），不做坐标映射 | ⬜ |
| B4 | 语义视觉检索（与 E3 合并） | ⬜ |

**判据**：某档达到 `ρ ≤ 0.85`，且候选侧只读 human-only 字段。
**否决条件**：若 B1–B4 全部无法达到 `ρ ≤ 0.90`，判定 **Human2Robot 不适合作为 RECAP 人手→机器人实验的替代数据集**，停止本路线，转为 (a) 换数据集，或 (b) 把课题重新定位为同具身跨 episode 检索。
**参考锚点**：B 通道 oracle 上界在 K=8 时为 7.6 mm（基线 11.1 mm），信息存在，缺的是表征。

### E3 — 视觉分支（GPU 1 张，类别 B）

已证伪：16 维 WAN 空间均值（ρ 1.59–1.70）、576 维稠密灰度（ρ 1.32–1.75）。二者在每个 K 上都劣于 geometry-only。

| 步骤 | 内容 | 状态 |
|---|---|---|
| E3.0 | **前置**：DINOv2 权重本地缺失。按 §2.2 由人工一次性获取并登记 SHA256；或改用本地已有的 SAM ViT-H 图像编码器 / V-JEPA2 / Depth-Anything 的 DINOv2 骨干 | ⬜ **阻塞** |
| E3.1 | 提取 human/robot 当前帧语义特征，测 ρ | ⬜ |
| E3.2 | 若全局特征仍差，试 patch-level + 前景掩码（SAM 或背景差分），只保留操作区域 | ⬜ |
| E3.3 | 与 geometry-only 及 E2 最佳通道做加权组合搜索 | ⬜ |

**判据**：某视觉方案单独或组合后优于 geometry-only 至少 0.05（ρ）。
**否决则**：v05 主检索模态定为 **geometry-only**，并在 v05 计划中写明"visual 分支经证伪后移除"及其证据。

### E4 — 离线指标能否反映策略性能（GPU 4 张，类别 B）

前三项只证明学习目标的条件数变好，不等于策略变好。

| 步骤 | 内容 | 状态 |
|---|---|---|
| E4.1 | 实现最小评估：canonical 误差 + episode 宏平均（可从 `tools/human2robot_m5b_p2_evaluation.py` 改；v05 阶段 6 无论如何都要用） | ⬜ |
| E4.2 | 用 E1–E3 选定配置短训 `no_retrieval` 与 `recap_hand_ret` 各 1500 step（约 8 GPU·小时/方法） | ⬜ |
| E4.3 | 在 seen-validation 与 dev 上对比 | ⬜ |

**判据**：recap < no_retrieval，且方向与 ρ 一致（ρ 越低差距越大）。
**否决则**：离线 next-state proxy 无法反映 RECAP 收益，需重新设计评估指标或承认必须上闭环 rollout——该结论必须在 v05 之前定下。

### E5 — co_training 重定义（GPU 0 张）

改为论文定义：把 human pool 窗口作为**额外训练样本**并入，各自 target 为其自身的绝对未来；不再把配对人手未来作为上下文。顺带消除同 episode 夹爪泄漏与训练/评估口径不一致。

**判据**：训练/评估口径一致；同 episode 未来不进入任何输入。
**状态**：⬜ 未开始。

### E6 — 决策关口

落一份一页纸的前提结论，明确回答：

1. K = ?
2. 人手通道 = ?（含 ρ）
3. 主检索模态 = ?
4. co_training 定义 = ?
5. 离线指标是否可用？
6. → **开 v05 冻结确认实验**，或 **换数据集 / 换选题**

只有 1–5 全部有数字支撑，才允许撰写 v05 总计划并进入其阶段 0。

**v05 相对 v04 的唯一结构性新增**：阶段 −1 科学前提门禁——在任何数据派生或 GPU 任务之前，必须提供只读证据证明 (a) 每个语义字段的物理含义经独立验证（不能只验 SHA）；(b) 检索计划相对无检索基线为正收益，且有 oracle 上界对照；(c) 各基线定义与被复现论文逐条对齐。任一项不通过，不得进入阶段 0。

---

## 五、当前进度总览

截至 2026-07-27。

| 环节 | GPU | 状态 | 关键数字 |
|---|---:|---|---|
| E0.1 脚本归档 | 0 | ✅ 完成 | 16 脚本 + `_pilot_util.py`；`d14` 约 9 min → 29.5 s、`d15` 27.3 → 4.9 s、`d16` 33.9 → 5.5 s，输出全部逐位不变 |
| E0.2 存储 | — | ⬜ | `/DATA1` 171 GiB < 300 GiB 门槛 |
| E0.3 GPU 窗口 | — | ⬜ | 8 卡全占 |
| E0.4 pilot launcher | 0 | ⬜ | E3 前置 |
| E1 K 与检索规则 | 0 | 🟡 首轮完成 | K=8/16/32/64 现配置 ρ = 1.19 / 1.03 / 0.84 / 0.74；各 K 最佳 0.96 / 0.91 / 0.78 / 0.68 |
| E2 真人手通道 | 0（B4 需 1） | 🟡 B0 已否 | B0 ρ = 0.95–0.96；oracle 上界 K=8 为 7.6 mm vs 基线 11.1 mm |
| E3 视觉分支 | 1 | ⬜ 阻塞 | 16 维 WAN ρ 1.59–1.70；576 维灰度 ρ 1.32–1.75；DINOv2 权重缺失 |
| E4 策略验证 | 4 | ⬜ | — |
| E5 co_training | 0 | ⬜ | 同 episode 夹爪泄漏 5–8 倍（0.018–0.029 vs 0.141–0.152） |
| E6 决策关口 | — | ⬜ | — |

**关键路径**：E2 是唯一可能否决整个路线的环节，且 B1–B3 纯 CPU、不依赖 GPU 释放，应优先。E3（B4）只需一张卡半小时，是 E2 中最可能过线的一档，建议与 B1–B3 并行推进。四卡组留给 E4。

---

## 六、停止条件与默认假设

**停止条件**：

- E2 的 B1–B4 全部无法达到 `ρ ≤ 0.90`；
- E4 判定离线指标无法反映策略性能，且找不到替代离线指标；
- 探索期发现新的、使前提再次失效的数据语义问题；
- 存储或 GPU 长期不可用，导致 E3/E4 无法执行。

**默认假设**：

- 原始 Human2Robot v1 保持只读；
- v03/v04 的全部产物保持只读，不删除、不改写；
- 探索期不产生任何性能声明；
- 探索期不重训 v04 的三个方法，不打开 v04 的 final 集；
- 探索期通过后仍需完整的 v05 冻结确认实验，才能得出可发表结论。

---

## 七、本文件变更记录

| 日期 | 变更 | 影响判断 |
|---|---|---|
| 2026-07-27 | 建立本文件；纳入 E0～E6、统一运行/环境/日志/落盘/变更规范；GPU 改为按张数声明、自动选择任意空闲卡并以 UUID 记录，不再绑定具体编号；同步 E0.1 完成、E1 首轮、E2 B0 否决、E3 阻塞的当前进度 | 探索期规范首次成文；不改变任何已记录数字，不产生新产物 |
