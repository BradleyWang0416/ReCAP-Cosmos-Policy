# RECAP × Human2Robot v04 无泄漏离线复现执行计划

## 一、目标与冻结决策

本阶段终点限定为：

> 在 Human2Robot 上完成无同源 episode 泄漏、多 robot episode、单训练 seed 的离线 RECAP 主效应对比。

明确不包含真实机器人部署、command adapter、闭环 rollout、RoboTwin、三 seed 鲁棒性或大规模消融。

固定决策：

- 冻结 v03 和当前 203-cell 矩阵，不再启动任何剩余 cell。
- v03 的代码、manifest、checkpoint 和非正式结果只作为历史证据，不删除、不改写。
- v04 使用新数据切分、新运行目录和新验收协议，不把结果写回 v03 DAG。
- 重训 no-retrieval、co-training、RECAP 三个学习方法，各使用同一个 seed 20260711。
- 主检索为 geometry_plus_visual；phase 只作为 oracle 上界。
- 采用严格统计门槛；失败时输出“未复现”，不进行事后 checkpoint 或超参数挑选。

2026-07-21 经用户明确批准，阶段 1 协议修订为 v04.1：原 held-out `grab_pencil1_v1` 因 102/102 个来源与 seen `grab_pencil_v1` 字节相同而移除，替换为正式预审通过的 `push_plate_v1`。替代候选审计 attempt 0002 receipt SHA256 为 `6e6072ff085bc831010d9adb7a0a13edd143e6beed873e1d355acdbf7892d20d`。其余 seen 任务、seed、切分、角色隔离和验收规则不变。

## 二、统一运行、环境与可审计性规范

本节适用于阶段 0～7 的所有数据处理、测试、训练、评估和报告命令。它是正式结果的前置门禁，不因某个脚本能在宿主机或不完整环境中启动而豁免。

### 2.1 运行入口和 Docker 边界

Docker 操作以 `docs/pusht_rag_docker_runbook.md` 为基础。v04 的专用约束以本节为准；v03 的 launcher、activation、cell registry 和输出目录只作历史参考，不得直接授权 v04。

所有 Python 程序必须在 `cosmos-policy:latest` 完整镜像中运行。宿主机只负责启动 Docker、选择物理 GPU、检查磁盘、查看日志和停止容器，禁止直接用宿主机 Python 或宿主机失效的 `.venv/bin/python` 运行实验。

阶段 0 必须先提供并验收专用入口 `start_human2robot_v04_docker.sh`。标准启动方式为：

```bash
cd /home/wxs/ReCAP-Cosmos-Policy
HUMAN2ROBOT_V04_GPU_DEVICES=0,1,2,3 \
  bash start_human2robot_v04_docker.sh
```

该脚本只打开完整环境的 shell，不自动启动 prepare、train 或 evaluate。正式命令进入容器后从 `/workspace` 执行，且必须通过 `tools/human2robot_v04.py` 的统一入口和 preflight。

专用容器必须满足：

- 宿主机 `/home/wxs/ReCAP-Cosmos-Policy` 以读写方式绑定为容器 `/workspace`；
- `/DATA1` 以读写方式绑定，所有 v04 新写入严格限制在 `/DATA1/wxs/ReCAP_M5B_V04_RUNS`；
- `/home/wxs/.cache` 和 `/home/wxs/.local/share/uv` 分别绑定为容器用户的缓存与 uv 根目录；
- 工作目录为 `/workspace`，使用仓库内 `/workspace/.venv`；
- 使用 `--ipc=host`，共享内存、CUDA、NCCL 和项目编译扩展全部通过 preflight；
- 容器、镜像 ID、挂载读写属性、用户 UID/GID 和启动时间写入运行回执。

### 2.2 完整环境、离线和禁止降级

当前环境和权重均已准备好，本计划不包含构建镜像、同步依赖或下载新文件。正式运行不得执行 `docker build`、`uv sync`、`pip install`、`hf download`，也不得在程序内触发 Hugging Face 自动下载。

容器统一设置并在回执中记录以下环境：

```bash
RECAP_WORKSPACE=/workspace
HUMAN2ROBOT_ROOT=/workspace/data/Human2Robot
HUMAN2ROBOT_V04_RUN_ROOT=/DATA1/wxs/ReCAP_M5B_V04_RUNS
COSMOS_HF_CHECKPOINT_ROOT=/DATA1/wxs/_HUGGINGFACE
COSMOS_SKIP_HF_AUTO_DOWNLOAD=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
WANDB_MODE=disabled
WANDB_DISABLED=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

启动前必须验证镜像、`.venv`、Python 3.10、CUDA/NCCL、编译扩展、数据、权重、挂载权限、磁盘门槛和 GPU 数量。任一项缺失时以 `BLOCKED_ENVIRONMENT` 退出，并在日志和 preflight receipt 中写明缺项。

严禁为了“先跑起来”而采用下列残次版本：

- 改用宿主机 Python、CPU、错误 CUDA 环境或未验收镜像；
- 自动下载、临时替换、跳过或随机初始化缺失权重；
- 静默减少 GPU、数据、query、step、分辨率、top-k 或模型模块；
- 用 synthetic、截断样本、旧 cache 或部分输出冒充正式运行；
- 在依赖或扩展加载失败后切换到语义不同的 fallback 实现；
- 将诊断运行、冒烟运行或提前 checkpoint 的结果写入正式验收报告。

少量数据或较少 GPU 只可用于明确标记为 `NONFORMAL_DIAGNOSTIC` 的诊断。诊断必须使用独立目录和 receipt，且不能解除 formal preflight blocker。

### 2.3 路径和权重登记

所有配置与回执使用解析后的绝对路径，不依赖当前 shell 的相对路径。宿主机与容器内的固定位置如下：

| 资产 | 固定位置 |
|---|---|
| 宿主机仓库 | `/home/wxs/ReCAP-Cosmos-Policy` |
| 容器仓库 | `/workspace` |
| Human2Robot 数据入口 | `/workspace/data/Human2Robot` |
| v04 派生数据 | `/workspace/data/Human2Robot/derived/v04` |
| 本地 Hugging Face 根 | `/DATA1/wxs/_HUGGINGFACE` |
| v04 正式运行根 | `/DATA1/wxs/ReCAP_M5B_V04_RUNS` |
| v04 统一日志根 | `/DATA1/wxs/ReCAP_M5B_V04_RUNS/orchestrator_logs` |
| v03 只读历史证据根 | `/DATA1/wxs/ReCAP_M5B_P2_RUNS` |
| 非正式诊断根 | `/DATA1/wxs/ReCAP_M5B_V04_DIAGNOSTICS` |

本地模型资产固定登记如下。preflight 必须检查文件存在、可读、大小和 SHA256，不能只检查目录名：

| 用途 | 本地文件 | v04 约束 |
|---|---|---|
| 2B 正式初始化 | `/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt` | 三个学习方法使用同一文件与同一 SHA256 |
| Predict2.5 tokenizer/VAE | `/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth` | 冻结，不进入 optimizer |
| Predict2 Video2World 兼容权重 | `/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt` | 仅在受控配置确实引用时允许加载 |
| Predict2.5 distilled 兼容权重 | `/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt` | 不是 v04 正式初始化权重，不得静默替换 post-trained |

阶段 4 使用的三个 v03 step-7000 checkpoint 从 `/DATA1/wxs/ReCAP_M5B_P2_RUNS` 读取。其精确绝对路径和 SHA256 必须由阶段 0 冻结 manifest 提供，禁止用目录 glob 或“最新 checkpoint”自动选择。

v04 新 checkpoint 只能写入 v04 正式运行根。每个 checkpoint receipt 必须登记 method、seed、step、绝对路径、文件清单、总字节数和 SHA256；配置不得把 distilled、post-trained、v03 和 v04 checkpoint 混为同一种资产。

本计划使用 zero text embedding，在线 `google-t5/t5-11b` 不是 v04 依赖。若日志出现联网访问或现场计算 T5 embedding，应立即失败并排查配置，不得等待下载完成。

### 2.4 四卡选择和容器内映射

正式训练固定使用四张稳定物理 GPU。宿主机通过 `HUMAN2ROBOT_V04_GPU_DEVICES` 显式提供恰好四个互不重复的编号；禁止对正式容器使用 `--gpus all`。

Docker 会把所选物理卡重新编号为容器逻辑 `0,1,2,3`。启动日志必须同时写出宿主机物理编号、GPU UUID/型号/显存和容器逻辑编号，训练固定使用 `torchrun --nproc_per_node=4`。

若稳定 GPU 少于四张，则等待资源，不改为单卡或双卡正式训练。诊断可另用空闲卡，但必须遵守 `NONFORMAL_DIAGNOSTIC` 隔离规则。

### 2.5 实时日志和运行状态

每次 Docker session、`prepare-data`、`audit-data`、`prepare-features`、`preflight`、`train`、`evaluate` 和 `report` 都必须先创建唯一 run ID 与 attempt 记录：

```text
/DATA1/wxs/ReCAP_M5B_V04_RUNS/orchestrator_logs/<run_id>/
  attempt_0001.log
  attempt_0001.command.txt
  attempt_0001.runtime.json
  attempt_0001.status.json
  attempt_0001.progress.json
  latest_log.json
  status.json
  progress.json
```

重试必须新建整套 `attempt_0002.*`，不得覆盖或继续追加旧 attempt。`latest_log.json`、汇总状态和可选的 `latest.log` 符号链接必须原子更新，指向当前或最近一次 attempt。

统一入口必须启用无缓冲或行缓冲输出，并用 `tee` 同时写终端和文件，同时保留被执行程序的真实退出码。禁止只在终端显示、重定向到 `/dev/null`，或等程序退出后才一次性写日志。

程序启动后立即输出 run ID、开始时间、完整命令、日志路径、产物根、代码/协议哈希和 GPU 映射。长任务至少每 60 秒输出 heartbeat；训练至少每 10 个 optimizer step 输出 step、loss、LR、耗时和显存摘要。

每次 checkpoint、cache shard、评估分片和报告落盘时记录路径、完成状态、计数和哈希。异常必须保留 traceback、退出码、最后完成单元与恢复入口，不能只留下“进程消失”。

默认日志级别为项目 `INFO`、第三方依赖 `WARNING`、`NCCL_DEBUG=WARN`。不得默认启用 `NCCL_DEBUG_SUBSYS`、逐样本 tensor dump、完整环境变量 dump 或重复 warning 刷屏。

需要 DEBUG 时使用独立诊断 attempt 和日志，不污染正式进度日志。重复告警应限频并在末尾汇总；正式日志既要足够判断健康状态，也要避免无关信息造成超大文件。

运行状态以 `status.json`、`progress.json`、最新日志和已验证产物共同判定，不能只看 PID。`ps -p <pid>` 为空只说明该 PID 当前不存在，不足以区分正常完成、失败或容器退出。

人工检查的最小流程为：先读 `latest_log.json`，再对其中的绝对日志路径执行 `tail -F`，最后核对 `status.json` 和最新完成的 checkpoint/cache shard。只有退出码为 0、预期产物齐全且 receipt 通过，状态才可写为 `COMPLETED`。

### 2.6 原子落盘、断点续跑和完整性

所有长耗时预处理和推理必须按可验证的 episode、shard 或 query 分片保存进度。单次 `np.savez_compressed` 的最终文件不能作为唯一心跳；程序必须在 `progress.json` 中持续更新已完成数量、总数、当前单元和更新时间。

文件先写到同文件系统的 `.partial` 临时路径，完成 flush、关闭、重新读取校验和哈希后再原子 rename。`.partial`、缺失 footer、哈希不符或 receipt 不完整的文件不得进入下游。

重启时只复用通过 schema、输入哈希、配置哈希和内容校验的完整单元。已有完整 cell、feature shard 或 inference receipt 必须保留；失败单元从最近有效边界续跑，不得为了方便重算或覆盖无关完整证据。

训练恢复必须验证 method、seed、optimizer step、模型/optimizer 状态、配置、数据、feature、world size 和代码哈希。任何不匹配都 hard-fail，不能只加载模型权重后宣称“断点续训”。

### 2.7 变更记录和文档同步

任何代码、配置、路径、GPU、日志、缓存、重试或协议调整，都必须在下一次受影响运行前更新相关文档，不允许运行事实长期领先于文档。

每次变更至少记录：

- 时间、原因、提出者和实施者；
- 修改前后值、涉及文件和 source snapshot/Git SHA；
- 影响的数据、cache、checkpoint、评估与报告；
- 是否改变科学语义、是否使旧产物失效；
- 已执行测试、receipt 路径、结果和剩余 blocker。

文档职责固定为：

- 本总计划记录冻结目标、阶段、硬门禁和正式口径；
- `docs/pusht_rag_docker_runbook.md` 记录可复用的 Docker 操作与故障处理；
- `方案/v04/` 下的阶段报告记录当次数据、训练、评估和异常事实；
- v04 变更记录按时间追加，不改写旧决策；
- immutable receipt 绑定实际命令、环境、输入、代码、协议、日志和输出哈希。

只影响日志降噪、物理 GPU 映射或等价重试的操作变更，也必须记录 attempt，但不冒充科学协议变更。影响 split、字段、特征、训练或评估语义的修改必须升级协议版本并重新哈希，显式作废和重跑所有受影响下游产物。

发现文档与实际运行不一致时，立即暂停新正式任务，保存现场并生成偏差记录。修正文档和实现、重新 preflight 后才能继续；不得事后补写成“原本就是这样”。

上述规范本身也进入阶段 0 冻结范围。所有正式 receipt 必须记录本文件和 Docker runbook 的 SHA256，从而使后续改动可检测、可解释、可追溯。

## 三、分阶段实施

### 阶段 0：冻结 v03

1. 创建实施分支 codex/recap-v04-offline-clean。
2. 生成只读的 v03 冻结记录，绑定：
   - Git HEAD 9a6aaca；
   - 3 个已训练 checkpoint、1 个 retrieval-only 产物；
   - 当前 4/203、3/48 training、0/147 formal evaluation 状态；
   - 最新非正式 full-149 指标；
   - v03 split、P1 pool、协议、代码和 checkpoint SHA256；
   - /DATA1 当前存储快照。
3. 停止旧 dispatcher/orchestrator 的自动续跑能力，但不删除旧输出。
4. README 将 v03 标为 LEGACY_ORACLE_PHASE_PILOT，不得用于新任务泛化结论。
5. 实现并验收 v04 专用 Docker launcher、离线环境门禁、统一日志/状态目录和文档变更记录。
6. 在任何 v04 数据写入或 GPU 任务前，生成镜像、挂载、GPU、权重、数据、磁盘和文档哈希的 preflight receipt。

通过条件：v03 所有引用可追溯、没有活动训练进程、旧产物哈希未变化。

---

### 阶段 1：建立 v04 数据与来源身份协议

#### 1.1 Seen-task 训练数据

沿用现有 16 个训练任务，扫描到的原始候选共 737 个 HDF5。

先执行结构、字段、时间段和 H/K 窗口有效性检查，再在每个任务内部按：

SHA256("20260711:" + source_relative_path)

排序并切分：

- seen train：前 90%；
- seen validation：后 max(1, ceil(10%))；
- 小于 5 个有效 episode 的任务直接中止，不静默合并任务；
- validation 只监控训练，不参与 checkpoint 选择。

训练采样采用任务平衡策略：

1. 均匀选择 16 个任务；
2. 在任务内均匀选择 episode；
3. 在 episode 内均匀选择合法 window；
4. 分布式 rank 间不得重复同一 global sample index；
5. set_epoch() 后结果可重复。

#### 1.2 Held-out 数据

固定四个 held-out 任务：

- grab_cube2_v1
- push_plate_v1
- grab_to_plate1_v1
- push_box_random_v1

四任务原始候选共 258 个。`push_plate_v1` 的 51/51 个来源通过字段、finite、时间 gap、H=8/K=8 与完整文件 SHA 独立性预审。

每任务按以下顺序建立互斥集合：

1. legacy_quarantine：严格取 v03 canonical 和 P1 pool 曾使用过的全部 source SHA；四任务计数依次为 10、0、10、10，总数 30，不为新任务伪造历史来源；
2. v04_human_pool：10 个新的 human-only episode；
3. v04_robot_dev：5 个新的 robot-only episode；
4. v04_robot_final：20 个新的 robot-only episode；
5. 其余进入 reserve。

候选同样按固定 SHA 排序，选择“第一个满足数据契约的 episode”，所有拒绝原因必须写入 manifest。任一任务在 quarantine 后不足 35 个合法新 episode，立即中止。

物理数据边界：

- human pool 文件只允许 human camera、human hand state/action、时间字段；
- robot dev/final 文件只允许 robot camera、robot observed EEF、gripper、时间字段；
- robot query 中不得保留可被检索器访问的人手配对字段；
- raw 数据只读，所有派生文件可由 manifest 重建。

#### 1.3 新来源身份字段

每个 episode/window 必须携带：

- source_relative_path
- source_sha256
- source_partition
- task
- episode_id
- role
- human_content_sha256 或 robot_content_sha256

所有 manifest 同时保存生成代码、原始清单和协议哈希。

通过条件：

- train/validation/pool/dev/final/quarantine 两两 source SHA 不重叠；
- pool 每任务恰好 10 个，dev 恰好 5 个，final 恰好 20 个；
- 最终共 20 个 dev robot episode 和 80 个 final robot episode；
- 所有文件通过时间段、字段白名单、finite、H=8/K=8 窗口检查。

---

### 阶段 2：修正候选过滤和主检索

#### 2.1 强制来源隔离

P2Window 增加原始来源身份。held-out 候选过滤必须无条件拒绝：

- candidate.source_sha256 == query.source_sha256
- candidate.source_relative_path == query.source_relative_path
- candidate 不属于 v04_human_pool
- query 不属于对应的 dev/final partition
- candidate 使用任何 robot-only 字段
- retrieval feature 读取 query future rows 或 target action

原来仅比较派生 candidate.path == query.path 的逻辑不得作为 v04 的安全边界。

每条 retrieval record 必须同时记录 query/candidate source SHA、partition、rank、distance 和 feature provenance。

#### 2.2 主检索定义

主方法固定为 geometry_plus_visual：

- geometry：H=8 的历史 10D state，相对当前状态、按 seen-train 统计量标准化、L2 normalize；
- query geometry 只来自当前及过去 robot observed state；
- candidate geometry 只来自 human hand state；
- visual：当前 robot frame 与候选 human frame 的冻结 WAN latent；
- geometry 与 visual 各自 L2 normalize，按现有 1/sqrt(2) 等权拼接；
- 在同任务的 active human pool 内计算距离；
- top-k 固定为 3；
- 距离相同使用 SHA256(seed, query_id, human_content_sha256) 决定顺序。

phase 改名为 oracle_phase，只能在主结果完成后运行诊断，任何 primary result、训练配置或验收报告引用 phase 都必须 hard-fail。

#### 2.3 Pool-growth 条件

human pool 顺序由 v04 manifest 固定，使用嵌套集合：

- pool1：rank 1
- pool2：rank 1–2
- pool4：rank 1–4
- pool8：rank 1–8
- pool10：rank 1–10

主方法比较使用 pool10；pool-growth 仅对同一个 RECAP checkpoint 运行，不重新训练。

---

### 阶段 3：实现 v04 最小实验接口

提供单一入口，例如 tools/human2robot_v04.py。

固定子命令：

- prepare-data
- audit-data
- prepare-features
- preflight
- train --method {no_retrieval,co_training,recap_hand_ret}
- evaluate --split {dev,final} --method ... --pool-size ...
- evaluate-oracle-phase
- report

所有命令默认 dry-run；真正写产物或启动 GPU 任务需要显式 --execute。每次执行生成独立 manifest 和 immutable receipt。

v04 运行根目录固定为：

/DATA1/wxs/ReCAP_M5B_V04_RUNS

派生数据和轻量 manifest 使用：

data/Human2Robot/derived/v04/

不得复用 /DATA1/wxs/ReCAP_M5B_P2_RUNS 的 cell registry 或 artifact 路径。

---

### 阶段 4：预检和旧 checkpoint 冒烟

1. 在 synthetic HDF5 上跑数据、身份、检索和评估测试。
2. 物化 seen-train、seen-validation、held-out pool/dev/final manifest。
3. 生成 geometry statistics 和冻结 visual feature cache。
4. 使用现有三个 v03 step-7000 checkpoint，在 v04 dev 集运行只读冒烟：
   - 5 robot episode/任务；
   - 每 episode 固定 8 个合法 query window；
   - 共 160 query；
   - 每 query top-3，共 480 inference receipts/方法。
5. 冒烟只验证运行闭环、来源隔离、输出完整和数值有限，不用于选择模型或判断 RECAP 优越性。
6. 锁定 v04 数据、检索、训练和最终评估协议 SHA。

任一 provenance、future-target independence、非有限数值、缺失 receipt 或跨 gap 错误出现时，禁止训练。

---

### 阶段 5：单 seed 三方法重训

三种方法共享：

- Cosmos Predict2.5 2B 初始化权重；
- seed 20260711；
- 相同 seen-train/validation split；
- task-balanced sampler；
- 4 GPU；
- batch size 25/GPU；
- gradient accumulation 2；
- effective batch 200；
- 7000 optimizer steps；
- H=8、K=8、top-k=3；
- 224×224；
- LR 1e-4；
- action loss multiplier 16；
- text conditioning 为 zero embedding；
- fixed step-7000 为最终 checkpoint；
- held-out dev/final 不参与训练和 checkpoint 选择。

方法差异仅限：

| 方法 | 检索 | Target |
|---|---|---|
| no-retrieval | masked | absolute robot future |
| co-training | 对应的共同训练输入 | absolute robot future |
| RECAP | geometry+visual human retrieval | robot future − aligned human plan |
| retrieval-only | 不训练 | aligned human plan |

训练顺序固定为 no-retrieval → co-training → RECAP。

Checkpoint 保存策略：

- 每 1000 step 保存一个滚动恢复点；
- 最多保留最近两个滚动点；
- step-7000 审计、哈希后标记为 immutable final；
- 不删除任何 v03 checkpoint；
- 每方法预计长期保留约一个最终 checkpoint。

启动前存储门槛：

- /home 可用空间不少于 150 GiB；
- /DATA1 可用空间不少于 100 GiB；
- 若不满足则中止，不通过压缩或删除旧正式证据绕过。

单方法通过条件：

- 7000 step 正常结束；
- loss 全程 finite；
- final checkpoint 可独立加载；
- config、data、feature、code、checkpoint 哈希齐全；
- seen-validation 仅报告，不选择更早 checkpoint。

---

### 阶段 6：锁定后的 dev 评估

新训练的四种方法在 20 个 dev episode 上统一评估：

- 每 episode 选择 8 个按合法窗口索引等分的 query；
- 不足 8 个窗口的 episode 由 reserve 中下一个合法 episode 替换；
- 共 160 query；
- top-3 聚合方式沿用现有 rank prediction aggregation；
- RECAP 主条件为 pool10 + geometry+visual。

dev 只允许发现工程错误，不允许根据数值：

- 更换 checkpoint；
- 调整检索权重；
- 调整 loss；
- 修改任务或 episode；
- 改变最终指标。

工程错误修正后必须升级协议版本和重新生成全部受影响哈希。若只是 RECAP 数值较差，仍按锁定协议进入 final，避免选择性报告。

---

### 阶段 7：一次性 final 评估

final 集在代码、checkpoint 和协议全部锁定后才允许打开。

固定规模：

- 4 个任务；
- 20 robot episode/任务；
- 8 query/episode；
- 共 80 episode、640 query；
- 每 query top-3，共 1920 inference receipts/学习方法条件。

一次性运行：

1. no-retrieval；
2. co-training；
3. retrieval-only pool10；
4. RECAP geometry+visual pool10；
5. RECAP pool1/2/4/8/10；
6. primary report 完成后，单独运行 oracle-phase diagnostic。

不得运行 step-1000～6000 checkpoint，不得根据 final 结果重新训练。

#### 主要指标

macro_episode_canonical_error：

1. 每个 query 对 K=8 的 canonical 误差取均值；
2. 每 episode 对 8 个 query 取均值；
3. 每任务对 20 个 episode 取均值；
4. 四任务等权宏平均。

次要指标：

- position error
- orientation error
- gripper error
- final-position error
- residual norm/saturation
- workspace violation
- 每任务结果
- pool-size 曲线

#### 统计方法

使用相同 episode/query 的配对比较。按任务分层、在每任务 20 个 episode 内有放回采样：

- bootstrap 次数：10,000；
- bootstrap seed：20260711；
- 差值定义：RECAP error − baseline error；
- 报告均值差、百分比差和 percentile 95% CI。

#### 严格通过条件

必须同时满足：

1. 所有 source overlap、target-feature、gap、nonfinite、workspace、clipping、residual-saturation guardrail 为 0；
2. 80 episode、640 query、所有 top-3 receipts 完整；
3. RECAP pool10 宏平均低于 no-retrieval；
4. RECAP pool10 宏平均低于 co-training；
5. 对两个学习基线的 bootstrap 差值 95% CI 上界都小于 0；
6. RECAP pool10 相对 pool1 的差值 95% CI 上界小于 0；
7. 至少 3/4 个任务优于最佳学习基线；
8. 任一任务不得比最佳基线恶化超过 10%；
9. gripper 宏平均不得比最佳学习基线恶化超过 10%。

任何条件失败，最终结论固定为：

NEEDS_REPRODUCTION — RECAP 未在 v04 单 seed 无泄漏离线协议下复现

不允许通过改用 step-5000、phase 主结果或删除困难任务改变结论。

## 四、接口与产物变更

主要新增接口：

- 来源身份：所有 window/candidate/query 暴露原始 source SHA 和 partition；
- v04 split manifest：显式描述 quarantine/train/validation/pool/dev/final/reserve；
- retrieval provenance：记录特征来源、query/candidate source 和禁止字段计数；
- task-balanced distributed sampler；
- episode-level evaluation 和 stratified paired bootstrap；
- v04 独立状态机，仅包含 prepare、preflight、3 training、dev、final、report，不建立新的 203-cell DAG。
- v04 专用 Docker launcher 和 fail-closed 环境 preflight；
- attempt 级实时日志、结构化状态、原子 cache/checkpoint 和可校验断点续跑；
- 文档与 immutable receipt 的双向哈希绑定。

必须产出的报告：

1. v03 冻结报告；
2. v04 数据切分与来源独立性报告；
3. retrieval feature provenance 报告；
4. 三方法训练回执；
5. dev engineering report；
6. final 主结果 JSON/Markdown；
7. oracle-phase 附录；
8. 最终限制说明。

最终报告必须明确：

- 单训练 seed，不支持训练鲁棒性结论；
- offline next-state proxy，不是机器人 command；
- 无真实机器人 rollout；
- visual feature 使用冻结 WAN latent，不等同论文完整 DINO/SAM 检索；
- Human2Robot 是替代数据集，因此属于机制复现，不是论文原数据严格复刻。

## 五、测试计划

必须新增并通过：

- 同一 source SHA、不同派生路径仍被拒绝；
- episode_0 或其他 quarantine 来源不能进入 v04 pool/dev/final；
- pool 文件不存在 robot datasets；
- robot query 文件不存在可被检索访问的人手字段；
- 修改 held-out future robot target 后，retrieval ranking 完全不变；
- geometry 只读取 current/history，visual 只读取 current frame；
- primary config 使用 geometry+visual，phase primary hard-fail；
- pool1/2/4/8/10 严格嵌套；
- split 和排序在相同 seed 下字节级确定；
- task-balanced sampler 在四个 rank 间无重复且可恢复；
- 三训练方法除预声明差异外配置一致；
- 8-window episode sampling 确定且不跨时间 gap；
- episode/task 宏平均不受 episode 长度影响；
- bootstrap 在固定 seed 下可重复；
- final report 拒绝缺失 receipt 或非锁定 checkpoint；
- 宿主机运行、非 Docker Python、GPU 数量错误、只读输出挂载或任一权重缺失时 hard-fail；
- offline 环境下不产生网络请求，缺失资产时不自动下载或切换 fallback；
- 每个子命令启动即产生可 `tail -F` 的 attempt 日志、status/progress 和真实退出码；
- 日志默认级别受控、重复告警限频，DEBUG 诊断与正式日志相互隔离；
- cache/checkpoint 使用 partial、校验、原子 rename，重启只复用完整且哈希匹配的单元；
- receipt 能检测本计划、Docker runbook、代码、配置或协议的未登记变更；
- v03 原有 161 项测试继续通过；
- 完整 Docker suite 生成新的只读回执。

## 六、停止条件与默认假设

停止条件：

- held-out 任一任务无法提供 quarantine 之外的 35 个合法 episode；
- source disjointness 不为零；
- visual/query feature 使用 future target；
- Docker、挂载、GPU、离线环境、依赖或本地权重任一 preflight blocker 非零；
- 无法生成实时日志、结构化状态或 immutable receipt；
- 存储低于门槛；
- checkpoint 无法独立加载；
- final 已打开后发现科学结果不通过。

默认假设：

- raw Human2Robot v1 保持只读；
- 16 seen 与经 v04.1 修订后的 4 个 held-out 任务定义保持不变；
- 当前 4 GPU 和本地 Cosmos 2B 权重继续可用；
- seed 固定为 20260711；
- 不启动 seed 20260712/20260713；
- 不执行 203-cell 剩余矩阵；
- 不实现真实机器人部署；
- 不进行 phase、视觉、H/K、分辨率或 representation 大规模消融；
- v04 通过后仍只能宣称“单 seed、无泄漏、多 episode 的离线跨具身证据”。

最终推荐状态：NEEDS_REPRODUCTION。只有阶段 7 的全部严格条件通过，才升级为 VERIFIED_FOR_SINGLE_SEED_OFFLINE_PILOT。

## 七、本文件变更记录

| 日期 | 变更 | 影响判断 |
|---|---|---|
| 2026-07-21 | 新增统一 Docker、完整离线环境、权重路径、四卡映射、实时日志、状态判定、原子落盘、断点续跑和文档同步规范；同步扩充阶段 0、接口、测试与停止条件 | 属于运行与审计门禁增强；不改变 v04 的数据切分、模型方法、训练预算、评估规模或严格通过条件 |
| 2026-07-21 | 用户批准 v04.1 held-out 修订：`grab_pencil1_v1` 替换为 `push_plate_v1`；legacy quarantine 改为真实历史并集 10/0/10/10 | 解除原任务与 seen 来源字节相同造成的硬 blocker；改变 held-out 任务身份与候选总数，但不改变每任务 10/5/20 新 episode 配额、seed、方法或验收门槛 |
