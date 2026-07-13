# RECAP 人手示范用于上下文学习指导真实机器人复现实验方案 v02

日期：2026-07-11

主数据集：Human2Robot v1
目标：冻结策略后，仅通过新增人手示范检索池，使真实机器人执行未见任务。

## 0. v02 决策摘要

v02 撤销 v01 中“canonical 数据必须统一到 10 Hz 或 16 Hz”的要求。新的原则是：canonical 层保留源时间轴；检索、残差和执行层必须具有可比的物理时间或任务阶段语义。

M2-v01 的结构转换成功，但 30 Hz 源帧率假设没有得到数据支持。其“严格 10 Hz”只保证人工时间轴内部一致，不能证明与真实采集时间一致，因此原 M2 验收结论撤销。

撤销不等于删除。旧代码、HDF5、统计和视频全部保留为历史产物，用于回归测试和对照实验；它们不得进入 v02 的 M3 检索、M4 训练或论文结论。

v02 只使用新的 `canonical/v2` 作为主线。任何 derived time view 都必须由显式配置生成，并记录源数据版本、时间策略、插值规则和配置哈希。

### 0.1 当前里程碑状态

| 里程碑 | v02 状态 | 决策 |
|---|---|---|
| M0：PushT 复现 | 已完成 | 保留；代码或 checkpoint 变化时做 smoke test |
| M1：Human2Robot 访问验收 | 已完成 | 数据可访问；许可、单位和动作语义仍需补证 |
| M2-v01：固定 30→10 Hz canonical | 验收撤销 | 保留历史产物，禁止作为 v02 训练数据 |
| M2-v02：native-time canonical | 待重新实现和验收 | v02 当前第一优先级 |
| M3：检索与离线 sanity check | 需重写 | 必须基于 v2 时间语义重新构造 |
| M4：paired bridge 训练 | 需调整 | 增加时间视图、对齐和 horizon 消融 |
| M5：消融与压力测试 | 待执行 | 增加 temporal mismatch 专项 |
| M6：真实机器人实验 | 待执行 | 先标定真实 policy/control clock |

### 0.2 硬性门禁

- `data/Human2Robot/canonical/v1/` 不得进入新的 retrieval index 或 dataloader。
- M2-v02 未通过前，不启动 M3 正式索引。
- M3 未证明 aligned residual 优于 absolute action 前，不启动 M4 主训练。
- 真实机器人 policy clock 未标定前，不固定最终 `H`、`K` 或 policy Hz。

## 1. 相比 v01 的核心改动

| 维度 | v01 | v02 | 改动原因 |
|---|---|---|---|
| 主数据源 | 方案正文仍以 MIME/RH20T 为主，后补 Human2Robot | Human2Robot 是唯一 public paired 主线 | M1 已确认本地 1,316 条 episode 可访问 |
| canonical 时间轴 | 假定源 30 Hz，离线下采样到 10 Hz | 保留 native frame axis，不伪造物理时间 | 源 timestamp 只有整秒精度，部分 episode 有 gap |
| 时间统一位置 | M2 存储层 | M3/M4 sample-construction 与部署层 | 模型需要可比时间语义，不要求所有源文件同一 Hz |
| `H/K` 定义 | 主要按 frame/step 理解 | 同时记录 `H_steps/K_steps` 与物理持续时间 | residual 和 future-state 对 temporal mismatch 敏感 |
| 速度特征 | 每帧位移可直接使用 | 仅在 `dt` 可信时使用 `Δx/Δt` | 不同采样率下每帧差分不可比较 |
| residual | 同下标 query/pool action 直接相减 | 先建立时间或阶段映射，再相减 | 同下标不等于同一物理时刻或任务阶段 |
| M2 delta stats | `action-state` 也写作 delta stats | M2 不生成 retrieval delta stats | RECAP residual 只能在 M3 检索和对齐后定义 |
| normalization stats | pilot 全体统计 | 只用 train split 统计 | 防止 held-out task 信息泄漏 |
| M2-v01 状态 | 通过 | 时间轴验收撤销 | validator 只验证人工时间轴，不验证真实时间准确性 |
| 旧产物处理 | 未定义 | 冻结、标记、禁止误用、可复现 legacy | 防止静默覆盖和结论污染 |

## 2. M2-v01 为什么必须回滚

### 2.1 论文没有要求固定 Hz

RECAP 论文以离散 step 定义 `H` 和 `K`。RoboTwin 的配置为 `H=16 frames`、`K=1`，检索时间项权重为 `0.0`，论文没有规定全数据必须统一到 10 Hz 或 16 Hz。

论文同时明确指出：当来源动作与目标动作在执行速度或时间尺度上差异过大时，residual 会失效。真正的约束是 query 与 pool chunk 的时间语义可比，而不是 HDF5 文件的标称 FPS 相同。

### 2.2 当前代码隐式假设 step duration 一致

PushT 检索把最近若干帧的位置差直接作为速度，没有除以 `Δt`。训练按相同数组下标计算 query action 与 retrieved action 的 residual，推理也按相同下标加回。

这套实现能工作，是因为 PushT query、pool 和环境都共享 10 Hz control step。代码中的 `fps=16` 只是保持训练和推理 conditioning 配置一致，不能作为真实机器人数据应为 16 Hz 的依据。

### 2.3 Human2Robot 不支持全局 30 Hz 假设

对 M2-v01 选择的 20 条源 episode 做首尾整秒 timestamp 粗测，表观频率中位数约为 29.2 Hz，但范围约为 2.96–31.21 Hz；其中 8/20 条低于 20 Hz。

相邻 timestamp 的重复比例约为 89.4%–96.8%。部分 episode 还出现明显 step jump；最大观测跳变为 232。低表观频率可能来自采集暂停、处理阻塞或 timestamp 语义差异。

这些现象不能证明真实相机频率是多少，但足以否定“所有 episode 都连续稳定地以 30 Hz 采集”。每三帧取一帧只能称为 fixed-stride derived view，不能称为物理严格 10 Hz。

### 2.4 回滚范围

撤销以下结论：

- `source_fps=30` 是已验证事实。
- 每三帧取一帧后得到物理准确的 10 Hz。
- 基于人工 `0.1 s` 时间轴计算的速度上限具有真实物理含义。
- `canonical/v1` 可以直接进入 M3/M4。

保留以下成果：

- Human2Robot HDF5 字段访问和 time-axis 一致性检查。
- `uint8/uint16` RGB 容器兼容与 gzip 写入。
- xyz、rot6d、gripper 的 schema 框架。
- 原始 qpos/qvel/action/end pose 和 hand trajectory 的审计字段。
- 原子写入、manifest、validator、统计框架、视频和单元测试基础。

## 3. v02 的三层数据架构

v02 将“原始事实”“canonical 表示”和“训练时间视图”分离。任何时间处理都必须发生在可追溯的 derived 层，不能覆盖 canonical 事实。

```text
Human2Robot source HDF5
        │
        │  只做字段映射、单位/表示转换、质量标记
        ▼
canonical/v2 native-time episodes
        │
        │  显式 time-view config + hash
        ▼
derived/timeviews/<view_id>/
        │
        ├── M3 retrieval index
        └── M4 train/eval samples
```

### 3.1 Source 层

Source 层是 `/DATA1/wxs/DATASETS/Human2Robot/data/v1/`。它保持只读，不修改原 HDF5，不用新的人工 timestamp 覆盖源 `step/timestamp`。

### 3.2 Canonical v2 层

Canonical v2 统一字段、单位和 action/state 表示，但不统一固定 Hz。每条 canonical frame 必须能追溯到唯一 source frame。

目标目录：

```text
data/Human2Robot/canonical/v2/
```

### 3.3 Derived time-view 层

Derived view 用于具体实验，可以是 native-index、固定 policy clock、阶段归一化或 DTW 对齐。每个 view 都必须拥有独立 ID、manifest、统计和 validator 报告。

建议路径：

```text
data/Human2Robot/derived/timeviews/<view_id>/
```

`view_id` 至少绑定：source manifest hash、canonical schema、时间策略、目标 policy clock、插值规则、gap 规则、`H/K` 和代码版本。

## 4. 里程碑实施方案

### M0：PushT 训练/推理链路复现

状态：已完成。

验收依据保持 v01 结论：当前 residual RAG 的 unseen average 为 35.1%，论文为 34.9%。M0 只在代码、环境、checkpoint 或时间处理公共模块变化时重跑。

M0 的 10 Hz 是 PushT 环境自身的 control clock，不外推为 Human2Robot 或真实机器人的默认时钟。

### M1：Human2Robot 下载与访问确认

状态：已完成访问验收，保留补证项。

已确认：本地 1,316 条 episode、33 个任务目录、约 112.5 GB；paired human/robot RGB、robot state/action、hand trajectory 和 source step/timestamp 可访问。

进入 M4 前必须补证：数据许可与引用、xyz 单位、Euler 顺序、action 语义、gripper 极性，以及 source `step/timestamp` 的采集含义。

### M2-v02：native-time canonical HDF5

状态：已于 2026-07-11 完成并通过验收。M2-v01 的“通过”状态不继承；v02 验收依据见 [`M2_Human2Robot_native_time_验收报告.md`](M2_Human2Robot_native_time_验收报告.md)。

#### M2-v02 目标

把 Human2Robot 转为可追溯、无人工时间假设的 canonical HDF5。M2 负责表示统一和质量审计，不负责把不同节奏强行变成同一控制频率。

#### M2-v02 pilot 范围

- 第一阶段仍选 20 条 episode，优先覆盖 20 个不同任务。
- 每条 canonical episode 保留全部 source frame，不做 fixed-stride 抽帧。
- pilot 通过后再决定是否转换全部 1,316 条。
- train/held-out task split 在统计前固化，防止 normalization 泄漏。

#### M2-v02 schema

```text
data/demo_0/
├── obs/
│   ├── images                    uint8   (T,H,W,3)  robot RGB
│   └── states                    float32 (T,10)     xyz(m)+rot6d+gripper
├── actions                       float32 (T,10)     robot absolute EE target
└── metadata/
    ├── source_indices            int64   (T,)
    ├── source_step               int64   (T,)
    ├── source_timestamp          int64   (T,)
    ├── segment_id                int32   (T,)
    ├── gap_mask                  bool    (T,)
    ├── qpos_raw/qvel_raw/end_position_raw/action_raw
    └── human/
        ├── images                uint8   (T,H,W,3)
        ├── hand_coords           float32 (T,24,3)
        └── hand_frames           float32 (T,4,3)
```

metadata attrs 必须包含：

- `timebase_status`: `trusted | coarse | discontinuous | unknown`。

- `timestamp_resolution`: 已知时写真实分辨率；当前预期为整秒。

- `source_fps`: 只有被上游资料确认后才填写数值，否则为空。

- `frame_selection`: v2 必须为 `all_source_frames`。

- 单位、Euler 顺序、gripper/action 语义及其证据状态。

- source 文件、schema、转换代码和配置哈希。

#### 时间质量处理

- `source_indices` 必须严格递增并与 source 一一对应。
- `source_step/timestamp` 原样保存，不生成伪造的严格时间轴。
- 大 step jump、timestamp jump、重复和回退全部进入报告。
- gap 不被插值跨越；后续 derived view 只能在同一 `segment_id` 内取 chunk。
- timestamp 不可信时，速度只报告 per-step displacement，不标记为 m/s。

#### action/state 表示

pilot 继续使用单臂 10D：`xyz(m) + rot6d + gripper`。源 pose/action 同时原样保存，便于单位或 Euler 假设更新后无损重转。

源单位和 Euler 约定未证实时，canonical attrs 必须写 `assumed`，不能写 `verified`。进入 M4 前必须升级为 verified，或通过消融证明备选解释不会改变结论。

#### M2-v02 统计

只用 train split 生成：

- `dataset_statistics.json`
- `dataset_statistics_post_norm.json`
- `data_quality_statistics.json`

M2 不生成名为 `delta_dataset_statistics.json` 的 RECAP residual 统计。若需要 action-state 控制偏移诊断，使用明确名称 `control_target_offset_statistics.json`。

#### M2-v02 validator

validator 分为两类：

1. 结构验证：schema、shape、dtype、time axis、finite、rot6d、gripper、workspace、RGB 解码。
2. 时间真实性验证：source frame 一一映射、timestamp 质量、gap、segment、是否存在人工时间轴。

时间真实性 validator 不要求固定 Hz。它要求每个时间结论都有证据等级，并禁止将 `coarse/unknown` 时间轴用于 m/s 或固定秒数 horizon 的计算。

#### M2-v02 验收标准

- 至少 20 条 episode，覆盖至少 20 个任务。

- 每条 `canonical_frames == source_frames`。

- human/robot/state/action 第一维一致，所有必要数值 finite。

- 不存在人工 `0, 0.1, 0.2, ...` 时间轴，除非真实 source 证据支持。

- 每条 episode 有 `timebase_status`、gap 报告和 segment 划分。

- train/held-out split 固化，统计只来自 train split。

- 10 条 native-time human/robot/action 可视化通过人工抽查。

- 新契约、新自动报告和人工验收报告均保存到 `方案/v02/`。

### M3-v02：时间感知检索索引与 residual sanity check

状态：需在 M2-v02 通过后重新实现。

#### M3-v02 目标

构造 query robot 到 pool human 的 subframe 检索，并证明匹配后的 human pseudo-action 可以成为比 absolute robot action 更稳定的粗计划。

M3 必须先把 human hand trajectory 提升为共享 10D pool state/action。lifting 的单位、坐标系、gripper proxy 和置信度必须独立记录。

#### 时间视图候选

至少比较四种时间处理：

1. `native_index`：保留 source frame index，仅作基线。
2. `legacy_stride3`：复现 M2-v01，仅作负面对照。
3. `policy_clock`：依据可信 timestamp 重采样到候选 policy clock。
4. `phase_or_dtw`：按任务阶段、关键事件或 DTW 对齐不同执行节奏。

若 timestamp 仍不可信，`policy_clock` 不得作为主结果。此时优先比较 native-index 与 phase/DTW，并禁用需要物理速度的检索项。

#### Derived view 插值规则

- RGB：最近邻取帧，不做跨 gap 插值。
- xyz：线性插值。
- rotation：SLERP。
- gripper：zero-order hold，并保留切换事件。
- absolute EE action：xyz 线性插值，orientation 使用 SLERP。
- velocity/delta action：根据物理 `dt` 重算，不直接插值其数值。

#### 检索索引格式

```text
query_ids:       (N,)    view/segment/demo/t_or_phase
match_ids:       (N,K)   view/segment/demo/t_or_phase
match_sims:      (N,K)   float32
alignment_meta:  (N,K)   pool time/phase、warp scale、confidence
```

index manifest 必须记录 `view_id`。训练和推理若使用不同检索代价，也要分别记录，例如 training 使用 action term，inference 不使用未来 query action。

#### Stage 1

按 task language、初始物体状态、初始 robot proprio 和数据质量筛候选 episode。不同 split、低 lifting 置信度和含严重 gap 的 episode 不进入主候选集。

#### Stage 2

按 object pose、robot proprio、human wrist/grip、视觉 embedding 和阶段信息做 subframe matching。只有 `dt` 可信时才加入物理速度 `Δx/Δt`。

检索不允许返回跨 segment chunk。时间/阶段对齐后的 pool action 才能与 query action 构造 residual。

#### residual 定义

```text
residual(τ) = robot_action(query_time + τ)
              - warped_human_action(pool_time + warp(τ))
```

这里的 `τ` 必须是物理时间或已定义的任务阶段坐标。禁止直接假定 query 第 i 帧与 pool 第 i 帧具有相同时间含义。

#### M3-v02 指标

- same-task top-k 命中率和 random retrieval 对照。

- 关键阶段一致率、warp scale、alignment confidence。

- absolute action norm 与 aligned residual norm 分布。

- position/orientation/gripper residual 分项。

- future-state DTW、phase error 和检索切换率。

- gap crossing count，主结果必须为 0。

#### M3-v02 验收标准

- 每个有效 query step 至少有 top-10 候选。

- random retrieval 在语义、阶段或 residual 指标上明显更差。

- 主时间视图的 residual norm 中位数低于 absolute action norm。

- 主时间视图优于或至少不差于 native-index 与 legacy-stride3。

- 不使用 held-out robot action 构造 inference-time retrieval feature。

- 若 residual 不稳定，暂停 M4，继续修正 lifting、时间对齐或检索。

### M4-v02：Human2Robot paired bridge 训练

状态：需在 M3-v02 通过后调整实现。

#### M4-v02 目标

验证模型能否在 seen paired task 上学习 human-to-robot 修正，并在 held-out task 上通过新增 human pool 改善 robot action reconstruction。

#### 数据与 split

- 只使用 Human2Robot 作为 public paired 主线。
- split 以 task 为单位，不能随机打散同任务 episode 到 train 和 held-out。
- normalization 只使用 train robot split。
- held-out robot action 仅用于离线评测，不参与训练或 inference retrieval feature。

#### dataloader 时间契约

每个训练配置必须显式记录：

```text
view_id
policy_dt 或 phase coordinate
H_steps / H_seconds
K_steps / K_seconds
gap policy
interpolation policy
alignment version
```

训练与评测必须使用相同时间视图语义。Cosmos batch 中的 `fps` conditioning 值要与训练配置一致，但不能替代真实 `policy_dt` 记录。

#### 固定对比方法

| 方法 | 目的 |
|---|---|
| `No retrieval` | 测量目标端 seen-data 泛化能力 |
| `Retrieval Only / hand playback` | 测量 human pseudo-action 本身的有效性 |
| `Co-training` | 检查简单混合数据能否替代 test-time retrieval |
| `RECAP hand-ret` | 检索条件化 residual + future-state prediction |

#### 必做时间消融

- native-index vs legacy-stride3 vs 主时间视图。
- fixed policy clock vs phase/DTW 对齐。
- per-frame displacement vs 可信 `Δx/Δt` velocity feature。
- 不同 `H_seconds` 与 `K_seconds`。
- 每 chunk 重检索 vs 更小 K 的高频闭环。

#### 主指标

- robot action MAE。
- position、orientation、gripper error。
- DTW trajectory distance。
- future-state prediction error。
- aligned residual norm 和 saturation rate。
- workspace clipping、velocity violation、gap crossing。

#### Pool-growth

对 held-out task 使用 `0, 1, 3, 5, 10` 条 human pool demo。模型参数全程冻结，观察 action/trajectory error 是否随 pool 增长总体下降。

#### M4-v02 验收标准

- `RECAP hand-ret` 优于 `No retrieval` 和 `Retrieval Only`。

- 与 `Co-training` 相比，在 held-out task 上不低于或更稳定。

- 主时间视图明显优于或至少不差于 legacy-stride3。

- pool-growth 呈总体改善趋势。

- residual 没有长期 saturated delta、gap crossing 或严重 workspace clipping。

- 所有结论都绑定具体 `view_id`、split 和配置哈希。

### M5-v02：机制消融与 temporal mismatch 压力测试

状态：待执行。

保留 v01 的 residual/absolute、future-state、retrieval modality dropout、geometry/visual 和 top-k 消融。

新增 temporal mismatch 压力测试：

- 对 pool chunk 人为做 `0.5×、0.75×、1.25×、1.5×` 时间拉伸。
- 注入 frame drop、timestamp jitter、短暂停顿和 gap。
- 比较固定 index、policy-clock 和 phase/DTW 的鲁棒性。
- 记录 residual norm、alignment confidence、future-state error 和失败类别。

human-only 数据仍只用于 trajectory extraction 和检索表示压力测试，不直接支撑真实机器人主结论。

### M6-v02：同场景真实机器人实验

状态：待执行。

任务继续使用：seen `open_cabinet`，unseen `close_cabinet` 和 `put_bottle_in_box`。

#### M6 前置时钟标定

真实采集前必须测量：

- camera capture rate 与 timestamp 分辨率。
- policy observation/query rate。
- action waypoint rate。
- low-level servo rate。
- inference latency 与 jitter。
- action queue 实际执行时长。

最终 `policy_dt/H/K` 根据这组测量决定，而不是沿用 10 Hz、16 Hz 或 PushT 的 `H=8,K=8`。

#### 采集协议

- `open_cabinet`：25 条 human + 25 条 robot paired demo。
- 两个 unseen task：各 10 条 human demo，不采新 robot training demo。
- 所有流使用可验证的高分辨率单调时钟。
- human/robot 的关键阶段、接触和 gripper event 单独记录。

#### 执行协议

- 只在 seen paired task 上训练一次，随后冻结模型。
- unseen task 只追加 human pool。
- 每个控制 chunk 重新检索，或按 M4 选出的 K 执行闭环重检索。
- policy waypoint 由低层控制器插值到 servo rate。
- 所有速度、加速度、workspace、force 和急停约束在物理时间上定义。

#### M6 验收标准

- unseen task 上 `RECAP hand-ret` 明显优于 `No retrieval`。
- 优于 hand playback/retarget-only。
- pool 从 0 到 10 条时总体改善。
- 全程不更新模型参数。
- 日志能还原每个观测、检索、预测和执行动作的真实 timestamp。

## 5. M2-v01 代码和中间数据的正确处理

### 5.1 总原则

不删除、不静默覆盖、不继续作为主线使用。先冻结历史产物，再在新路径实现 v2。

### 5.2 旧代码分类

| 文件/能力 | 处理方式 | v02 用途 |
|---|---|---|
| `tools/human2robot_m2.py` | 原地重构，但保留显式 legacy 模式 | 复用 schema、dtype、rot6d、原子写入和可视化 |
| `resample_indices()` 与固定 30→10 | 重命名为 legacy-only，默认禁用 | 只用于对照实验和 v1 可复现 |
| `convert_human2robot_m2.py` | 默认改为 `preserve_native`，输出 v2 | v02 主 CLI |
| `validate_canonical_hdf5.py` | 拆分结构验证与时间真实性验证 | v2 自动验收 |
| `human2robot_m2_test.py` | 保留旧回归，新增 native/gap/timebase 测试 | 防止回滚功能丢失 |

legacy 模式必须要求显式参数，例如：

```text
--timebase-policy legacy_fixed_stride3_assumed30
```

默认执行不得再接受隐式 `source_fps=30,target_fps=10`。

### 5.3 旧数据分类

历史目录保持：

```text
data/Human2Robot/canonical/v1/
```

计划增加 `DEPRECATED_M2_V01.json`，至少记录：

- 状态：`acceptance_withdrawn`。
- 原因：unverified 30 Hz assumption and synthetic 10 Hz timeline。
- 原验收报告与 manifest/hash。
- 允许用途：legacy 回归、dtype/schema 测试、可视化、时间消融。
- 禁止用途：v02 retrieval、训练统计、主实验和论文结论。

不把 v1 移动到新目录，避免旧报告路径失效。只有在 v2 完成、磁盘确有压力且用户明确批准后，才考虑删除或外部归档。

### 5.4 旧统计文件

以下文件只描述 v1 derived view，不得复制到 v2：

- `dataset_statistics.json`
- `dataset_statistics_post_norm.json`
- `delta_dataset_statistics.json`

其中旧 `delta_dataset_statistics.json` 是同帧 action-state offset，不是 RECAP retrieval residual。v02 不再沿用该命名。

### 5.5 旧 validator 和速度结论

旧 validator 证明 HDF5 结构和人工时间轴内部一致，但没有证明真实 10 Hz。旧速度上限依赖人工 `dt=0.1 s`，不能作为物理速度证据。

这些结果可以保留在历史报告中，但报告顶部计划增加“验收已撤销”的醒目标记，并链接 v02 方案和新的 M2-v02 报告。

### 5.6 旧视频

10 条 MP4 保留为图像配对、dtype 转换和 overlay 的诊断样例。其播放 10 fps 只表示视频编码速度，不证明源动作物理频率。

### 5.7 迁移顺序

1. 冻结 v1 manifest、统计、报告和 SHA-256。

2. 为 v1 增加 deprecated marker，不修改 HDF5 内容。

3. 为旧 CLI 增加显式 legacy timebase policy。

4. 实现 `preserve_native`，只写 `canonical/v2`。

5. 重写 validator 和测试。

6. 用相同 20 条 source episode 运行 v2，便于逐项对比。

7. M2-v02 通过后再开始 M3，v1 继续只读保留。

## 6. 更新后的实验矩阵

| 阶段 | 数据/视图 | 训练对象 | 测试对象 | 核心问题 |
|---|---|---|---|---|
| M0 | PushT 10 Hz env | 已完成 | 已完成 | RECAP residual RAG 是否复现 |
| M1 | Human2Robot source | 不训练 | 数据访问 | paired 字段是否可访问 |
| M2-v02 | Human2Robot native-time | 不训练 | schema/timebase QA | 是否无损、可追溯、无伪时间轴 |
| M3-v02 | 多个 derived time view | 不训练主策略 | held-out paired data | 哪种时间/阶段对齐能稳定 residual |
| M4-v02 | 选定主 time view | seen paired task | held-out task | human-to-robot bridge 是否可学 |
| M5-v02 | 时间扰动与 human-only | 消融 | robustness | 收益是否来自机制且耐时间错配 |
| M6-v02 | 自采真实时钟数据 | open_cabinet | 两个 unseen task | 冻结模型+新 human pool 是否有效 |

## 7. 公平性和数据泄漏要求

- train/held-out 必须按 task 划分。

- normalization、time-view 参数学习和 alignment 模板只能使用 train split。

- held-out robot action 只能用于评测，不能进入 inference retrieval feature。

- 所有 temporal baseline 使用相同 episode、action/state 表示和模型预算。

- legacy-stride3 只能作为标记清晰的 baseline，不能混入主 time view。

- pool-growth 各方法必须看到相同 human pool。

- 所有结果记录 `view_id`、split hash、index hash、model config 和 seed。

## 8. 阶段门禁

### Gate A：M2-v02 → M3-v02

通过条件：native frame 全保留、无伪时间轴、时间质量分级完成、gap 已分段、train/held-out split 固化、单位和动作语义风险有明确状态。

### Gate B：M3-v02 → M4-v02

通过条件：无 gap crossing；主对齐策略优于 random 和 legacy；aligned residual 比 absolute action 更稳定；human pseudo-action lifting 置信度达标。

### Gate C：M4-v02 → M6-v02

通过条件：RECAP 优于 No retrieval/Retrieval Only；时间消融支持主 view；pool-growth 有改善；部署 clock、H/K、延迟和安全约束已经标定。

## 9. 关键风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| source timestamp 只有整秒 | 无法恢复可靠物理 dt | 寻找采集代码/日志；否则使用 phase/DTW，不伪造 Hz |
| episode 中有暂停或 step jump | chunk 跨越无效区间 | gap mask + segment，禁止跨 segment 取样 |
| 30 Hz 只对部分任务成立 | fixed stride 产生不同真实时长 | canonical 保留 native，derived view 显式处理 |
| human pseudo-action 不可靠 | residual 噪声大 | lifting confidence、关键事件对齐、低置信度剔除 |
| action/Euler/gripper 语义未证 | 共享动作空间可能错误 | M4 前补上游证据或做备选解释消融 |
| temporal warp 过度平滑 | 接触/gripper event 丢失 | event-aware 插值、ZOH、切换点保护 |
| `H/K` 与真实 latency 不匹配 | 开环误差和安全风险 | 真实 clock 标定后再定，优先更小 K |
| v1/v2 被混用 | 统计和结论污染 | 不同路径、schema、marker、view_id 和 loader hard check |
| public 数据与真实场景差异大 | 离线收益不能外推 | public 阶段只证明工程与 bridge 可行性 |

## 10. v02 交付物

### M2-v02

- `human2robot_canonical_hdf5_contract_v2.md`

- native-time converter 与 timebase audit。

- v2 HDF5 validator 和新增单元测试。

- train/held-out split manifest。

- train-only dataset statistics。

- 20 条 native-time pilot HDF5 和 10 条可视化。

- `M2_Human2Robot_native_time_验收报告.md`。

- v1 deprecated marker 与历史报告撤销说明。

### M3-v02

- time-view builder 与 view manifest。
- human wrist/grip 到共享 10D trajectory 的 lifting 工具。
- retrieval NPZ、alignment metadata 和 index hash。
- temporal baseline 对比、residual/absolute 分布和检索可视化。
- M3 验收报告。

### M4-v02

- Human2Robot paired bridge 配置。
- 四个固定 baseline。
- 时间处理与 `H/K` 消融。
- pool-growth 曲线。
- action、future-state、residual、gap/safety 指标报告。

### M5-v02

- temporal mismatch 压力测试。
- retrieval modality 和 residual/future-state 消融。
- human-only lifting 质量与失败案例。

### M6-v02

- 真实机器人 clock/latency 标定报告。
- 采集与安全协议。
- rollout 日志、成功率和 95% CI。
- success/failure 视频与最终结论边界。

## 11. 下一步执行顺序

1. 为 v1 增加 deprecated marker，并在旧报告顶部写明验收撤销。

2. 重构 M2 工具，默认 `preserve_native`，保留显式 legacy 模式。

3. 创建 canonical v2 契约、split manifest 和时间质量审计。

4. 用与 v1 相同的 20 条 episode 生成 v2 pilot。

5. 完成 M2-v02 自动与人工验收。

6. 实现 M3 的 native、legacy、policy-clock、phase/DTW 四个 time view。

7. 通过 residual sanity gate 后再进入 M4。

## 12. 结论与不能过度声称的边界

v02 不把“统一帧率”当作数据清洗目标，而把“可追溯且可比较的时间语义”当作 RECAP residual 成立的前提。

M2-v01 仍证明了字段映射、HDF5 工程、dtype 兼容和可视化链路可用，但没有证明 30→10 Hz 的物理正确性。它是有价值的历史实验，不是 v02 的训练数据。

M2-v02 通过只代表 canonical 数据无损、可追溯、质量已审计。M3 通过才代表检索与 residual 在某个 time view 下合理；M4 通过才代表 public paired bridge 可学。

只有 M6 的同场景真实机器人 rollout 才能支持“冻结模型后，通过新增人手示范执行未见任务”的最终复现结论。
