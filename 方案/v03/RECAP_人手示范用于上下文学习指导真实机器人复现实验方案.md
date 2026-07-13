# RECAP 人手示范用于上下文学习指导真实机器人复现实验方案 v03

日期：2026-07-11  
主数据集：Human2Robot / H&R v1  
目标：冻结策略后，仅通过新增人手示范检索池，使真实机器人执行未见任务。

修订状态：v03.1 action-gate erratum。M2 schema 与已验收产物不变；本次仅修正 M3/M4 的 pool/query action 角色和 M6 deployment command 门禁。

修订基线：[v02 实验方案](../v02/RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md)。

## 0. v03 决策摘要

v03 依据 Human2Robot 论文版本记录、最新版 v4 附录和官方数据集卡重新打开 M2 Gate A。M2-v02 的工程验收仍有价值，但其 FPS 与 action 角色需要修订；M2-v03 现已完成并通过 Gate A1。

论文 v4 已确认两台 D435 的标称采集配置为 30 Hz。这个证据支持 `nominal_camera_fps=30`，但不自动证明发布 HDF5 每两行都严格相隔 `1/30 s`。

本地 pilot 仍存在 step jump、timestamp jump 和整秒 timestamp。v03 因此继续保留 native frame、gap 和 segment，不把 30 Hz 写成全局无间断的测量时间轴。

论文 v2 曾将 30 FPS 降为四分之一用于第二阶段训练。该 7.5 FPS 方案属于版本化 derived view，不属于 canonical 数据，也不作为 v03 的默认 policy clock。

官方数据集卡将 v1 `/action` 定义为“机器人坐标系中的人手空间姿态”。M2-v02 却把它标成 robot absolute EE target；这一语义冲突会直接污染 residual，因此必须修订 schema 并重新验收。

数据集卡同时明确 v0 的 `/end_position + /gripper_state` 可作为 action。v03.1 因此将 `/action` 固定为 pool-side human plan，将 future robot EE trajectory 作为 query-side BC proxy；原始低层 command 是否发布不再阻塞离线 M3/M4。

真实机器人仍必须在 rollout 前标定 executable command adapter。离线 BC proxy 不能自动升级为真实控制器 command。

v03 不覆盖 v1 或 v2。新主线写入 `canonical/v3`；v1 继续作为固定 stride 历史对照，v2 作为 native-time 工程与时间审计基线，但二者都不得直接进入 v03 的 M3/M4。

### 0.1 当前里程碑状态

| 里程碑 | v03 状态 | 决策 |
|---|---|---|
| M0：PushT 复现 | 已完成 | 保留；公共训练/检索代码变化时 smoke test |
| M1：Human2Robot 访问与证据 | 已完成 | evidence manifest 已保存；xyz 物理单位与原始控制链路保留为未解决项 |
| M2-v01：固定 30→10 | 验收撤销 | 只读保留，作为 `legacy_stride3` 负面对照 |
| M2-v02：native-time canonical | 语义验收重开 | 结构、时间审计和 split 可复用；FPS 与 action 角色结论被 v03 取代 |
| M2-v03：semantic-safe canonical | 已完成，通过 Gate A1 | 20 条、20 个任务、9,039 rows；三类 validator 与人工可视化通过 |
| M3-v02 | 暂停 | 未生成正式主索引；由 M3-v03 替代 |
| M3-v03：action/time view sanity | 已完成，通过 Gate B | 分离 pool/query action view；批准严格 future `t+1` BC proxy，lag-calibrated proxy 因弱相关仅作诊断 |
| M4-v03：paired bridge | 已启动，Gate C pending | 已完成 ridge smoke 闭环；正式 Cosmos/RECAP、多 seed 与扩充 human pool 待执行 |
| M5-v03：机制与压力测试 | M5-A 首轮通过；M5B-P0 通过，P1/P2 pending | 正式 2B Human2Robot adapter、9 个 3-seed 配置与 Docker/CUDA 单批契约测试已通过；独立 human pool 与正式训练运行仍待完成 |
| M6-v03：真实机器人 | 待执行 | 真实 clock 独立标定，不继承 public 30 Hz |

### 0.2 硬性门禁

- `canonical/v1`、`canonical/v2` 不得进入 v03 正式 retrieval index 或 dataloader。
- M2-v03 已通过；正式索引只能读取 `canonical/v3` 和通过 Gate A2b 的 derived view。
- `/action` 固定为 pool-side human plan，不要求它是 robot command。
- Gate A2b 未批准 query-side future EE BC proxy、pool/query alignment 和 residual sanity 前，不启动 M4 主训练。
- deployment command adapter、真实 clock、latency 和安全门禁未通过前，不执行 M6 rollout，也不固定最终 `H/K/policy_hz`。

## 1. 相比 v02 的核心改动

| 维度 | v02 | v03 | 改动原因 |
|---|---|---|---|
| 相机采集率 | `source_fps` 为空、状态 unknown | `nominal_camera_fps=30`，状态 `verified_upstream` | 论文 v4 附录明确 D435 以 30 Hz 采集 |
| HDF5 行时间 | 主要标为 coarse/discontinuous | 与相机标称 clock 分开记录 | 标称采集率不等于每行无丢帧、无暂停 |
| 7.5 FPS | 未纳入主候选 | `paper_v2_stride4_nominal7p5` derived baseline | 四分之一采样明确出现在论文 v2，v3/v4 不再声明 |
| `/action` 角色 | robot absolute EE target | pool-side human pose in robot frame，不要求其为 robot command | 官方数据集卡与 RECAP pool 角色共同约束 |
| robot trajectory | `obs/states` 与 generic `actions` | observed robot EE；future chunk 可作为 dataset-card-approved BC proxy | 数据集卡允许 `/end_position + /gripper_state` 作为 action |
| Euler / gripper | assumed / unknown | dataset-card verified | 数据集卡明确 degree XYZ、1=open、0=close |
| xyz 单位 | assumed millimetre | 继续 assumed，单独补证 | 数据集卡未给出 xyz 单位 |
| canonical schema | `human2robot-canonical-hdf5-v2` | 新 schema v3，不提供含糊 generic action | 语义变化影响下游 residual |
| M2 状态 | 已通过 Gate A | Gate A 重开，需重新实现和验收 | v02 报告中的关键证据状态已过时 |
| v02 产物 | v02 主线 | 冻结为 superseded baseline | 禁止静默修改已验收哈希和结论 |
| M3 主问题 | 时间对齐与 lifting | 分离 pool plan、query BC proxy、alignment 与 residual | RECAP 两侧 action 角色不同，但必须共享低层表示 |
| M4 dataloader | 绑定 `view_id` | 绑定 time、pool/query action、alignment 与证据版本 | 防止角色、时间和 residual 语义漂移 |
| M5 消融 | temporal mismatch 为主 | 增加 action-role、paper-version 和 424/426 处理 | 新证据揭示新的混杂因素 |

## 2. 新证据、版本和证据等级

### 2.1 论文与数据集卡证据

arXiv 当前记录为 v4，最后修订于 2025-11-15。v4 附录说明两台 Intel RealSense D435 以 240×424、30 Hz 配置采集同步人手和机器人视频。

来源：[Human2Robot arXiv 记录](https://arxiv.org/abs/2502.16587)；[v4 PDF](https://arxiv.org/pdf/2502.16587v4)。

论文 v2 明确写明第二阶段的 dense 30 FPS 会造成生成视频抖动，因此将帧率降为原来的四分之一。这个陈述对应 nominal 7.5 FPS。

来源：[Human2Robot v2 PDF](https://arxiv.org/pdf/2502.16587v2)。v3/v4 不再保留这段实现说明，因此 v03 必须把它标记为 `paper_v2`，不能冒充最新版统一规范。

官方数据集卡说明 `/end_position` 是机器人末端 6DoF pose，Euler 单位为 degree、顺序 XYZ；`gripper_state` 中 1=open、0=close。

数据集卡的版本说明明确指出，v0 的 `/end_position` 与 `/gripper_state` 可以作为 action。该证据足以批准 dataset-card-defined BC label，但不证明它是控制器实际收到的低层 command。

数据集卡同时说明 v1 `/action` 是机器人坐标系中的人手空间姿态。它没有直接证明该字段就是实际下发给机器人的无延迟 absolute EE target。

来源：[HumanAndRobot 数据集卡](https://huggingface.co/datasets/dannyXSC/HumanAndRobot)。

### 2.2 v03 证据等级

| 等级 | 含义 | 可支持的结论 |
|---|---|---|
| `verified_upstream` | 论文、数据集卡或上游代码明确说明 | 可写具体值和来源版本 |
| `verified_local` | 对本地 source/HDF5 全量或抽样实测 | 可描述本地序列事实，不外推采集机制 |
| `inferred` | 由数值、命名或对齐关系推断 | 只能作为候选解释或消融 |
| `unknown` | 缺少足够证据 | 不得进入依赖该事实的主结论 |

### 2.3 字段级证据表

| 字段/结论 | v03 状态 | 证据 | 限制 |
|---|---|---|---|
| D435 nominal capture FPS | 30，`verified_upstream` | 论文 v4 附录 | 不证明 HDF5 行间无丢帧或暂停 |
| human/robot video frame pairing | `verified_upstream + verified_local` | 论文、数据集卡、相同 T | 不证明所有状态的控制采样率高于或等于视觉流 |
| released row effective dt | `unknown/coarse/discontinuous` | 整秒 timestamp、step jump | 不得全局设为严格 `1/30 s` |
| Euler 单位与顺序 | degree + XYZ，`verified_upstream` | 数据集卡 | 仍需数值转换单元测试 |
| gripper 极性 | 1=open、0=close，`verified_upstream` | 数据集卡 | 需要检查所有 episode 的取值范围 |
| `/action` 字段内容 | pool-side human hand pose in robot frame，`verified_upstream` | 数据集卡 | 不要求其为 robot command；是否为实际 retarget command 仍 unknown |
| `/end_position + /gripper_state` | observed robot EE；可作 BC action label，`verified_upstream` | 数据集卡 | command status unknown；必须派生 future/lag proxy，不能同帧复制 observation |
| xyz 物理单位 | `unknown`；当前数值上推断 mm | 本地数值审计 | 离线 M4 需验证 pool/query 共同数值尺度；真实物理单位在 M6 command gate 前解决 |
| source timestamp 单位/语义 | 预期整秒，未完全确认 | 本地 dtype/range | 只用于审计 jump，不单独恢复逐帧 dt |
| 论文相机尺寸 | 240×424 | 论文 v4 | 本地 HDF5 曾观测 240×426，需单独解释 padding/crop |

### 2.4 证据对时间结论的影响

v03 将 `nominal capture clock` 与 `serialized record continuity` 分开。前者可以是 verified 30 Hz，后者仍可能是 coarse 或 discontinuous。

在无 gap、`source_step` 连续且 step 语义已确认时，可构造 nominal `1/30 s` derived view。该时间仍要标为 nominal，不替代 source timestamp。

若 step 语义未确认，只能用 row index 或 phase/DTW。任何 30→7.5、30→10 或目标 policy clock 都必须在 derived 层生成。

## 3. v03 四层数据架构

v03 将原始事实、semantic-safe canonical、derived action/time view 和训练样本分为四层。generic `actions` 不再出现在未解决语义的 canonical 层。

```text
Human2Robot source HDF5 (read-only)
        │
        │  字段逐值保留 + evidence manifest
        ▼
canonical/v3 semantic-safe episodes
        │
        │  pool/query action views + alignment + time-view config
        ▼
derived/views/<time_view_id>/<pool_action_view_id>/<query_action_view_id>/<action_alignment_id>/
        │
        ├── M3 retrieval/alignment index
        └── M4 train/eval samples
```

### 3.1 Source 层

Source 根目录保持只读：

```text
/DATA1/wxs/DATASETS/Human2Robot/data/v1/
```

不得修改 source HDF5，不得把论文 30 Hz 或 derived timestamp 回写到 source。所有上游证据保存 URL、arXiv version、访问日期和证据摘要。

### 3.2 Canonical v3 层

目标目录：

```text
data/Human2Robot/canonical/v3/
```

Canonical v3 保留全部 source row，统一 dtype、pose 表示和字段命名。它描述“字段是什么”，不提前决定“哪个字段是策略监督 action”。

### 3.3 Derived action/time view 层

建议目录：

```text
data/Human2Robot/derived/views/<time_view_id>/<pool_action_view_id>/<query_action_view_id>/<action_alignment_id>/
```

每个 view 必须绑定 canonical manifest hash、论文/数据卡证据版本、gap policy、时间策略、pool/query action role、alignment、插值策略、`H/K` 和代码版本。

### 3.4 Train/eval 层

训练样本只能读取已通过 M3 sanity gate 的 `time_view_id + pool_action_view_id + query_action_view_id + action_alignment_id`。loader 必须拒绝 generic、未批准 proxy 或与 checkpoint manifest 不一致的 view。

## 4. 里程碑实施方案

### M0：PushT 训练/推理链路复现

状态：已完成。

保留 v02 结果。只有公共 residual、future-state、retrieval 或 normalization 代码变化时，才重跑 smoke test 或完整复现。

PushT 10 Hz 只代表该环境 control clock，不用于证明 Human2Robot 或真实机器人 clock。

### M1-v03：Human2Robot 访问与证据注册

状态：已完成。证据 manifest 已保存；xyz 物理单位和原始控制链路明确保留为 unknown。

本地 1,316 条 episode、33 个任务目录和 paired 字段访问结论保留。v03 新增 `source_evidence_manifest_v3.json`，集中记录论文和数据集卡证据。

M1-v03 已写明：30 Hz 是 camera nominal capture rate；Euler/gripper 已验证；`/action` 是 human hand pose in robot frame；xyz 物理单位和控制链路仍待确认。

### M2-v03：semantic-safe native canonical HDF5

状态：**已完成，通过 Gate A1**。验收报告为 `M2_Human2Robot_semantic_safe_验收报告.md`。

#### M2-v03 重开原因

M2-v02 的 frame preservation、source hash、gap、segment、split 和 finite 检查仍有效，不因新证据失效。

M2-v02 把 `source_fps` 留空并让 validator 拒绝 numeric FPS。论文 v4 已提供上游证据，因此这一 evidence contract 已过时。

M2-v02 将 source `/action` 写入 canonical `actions`，并标为 robot absolute EE target。数据集卡只证明它是 robot-frame human hand pose，因此该标签不能继续作为 residual ground truth。

M2-v02 把 Euler 约定标为 assumed、gripper 极性标为 unknown。数据集卡已足以升级这两项，但 xyz 单位仍不能升级。

由于 action 角色会改变 schema、统计和下游 residual，不能只改报告或 HDF5 attrs。必须生成新 schema、重跑 pilot 和验收。

#### M2-v03 pilot 范围

- 继续使用 v02 相同的 20 条 source episode，覆盖相同 20 个任务。
- 继续使用原 task split；新 manifest 记录 v02 split hash 作为 parent。
- 每条保留全部 source row，不做 stride 抽帧。
- 重新生成 train-only、角色明确的统计，不复制 v02 generic action statistics。
- 生成 10 条新可视化，overlay 分别显示 robot observed state 与 human/retarget pose。

#### M2-v03 schema

```text
data/demo_0/
├── obs/
│   ├── robot_images                 uint8   (T,H,W,3)
│   └── robot_state_10d              float32 (T,10)
├── trajectories/
│   ├── robot_ee_observed_10d        float32 (T,10)
│   └── human_hand_robot_frame_10d   float32 (T,10)
└── metadata/
    ├── source_indices               int64   (T,)
    ├── source_step                  int64   (T,)
    ├── source_timestamp             int64   (T,)
    ├── segment_id                   int32   (T,)
    ├── gap_mask                     bool    (T,)
    ├── qpos_raw/qvel_raw
    ├── end_position_raw/action_raw/gripper_state_raw
    └── human/
        ├── images                   uint8   (T,H,W,3)
        ├── hand_coords              float32 (T,24,3)
        └── hand_frames              float32 (T,4,3)
```

Canonical v3 默认不写含糊的 `actions` dataset。若兼容现有 loader，必须由 derived adapter 写 `policy_actions`，同时强制绑定 `query_action_view_id`、proxy role 和 alignment evidence。

`robot_state_10d` 与 `robot_ee_observed_10d` 都源自 `/end_position + /gripper_state`。前者服务 observation API；后者强调它是观测轨迹，不自动等于控制目标。

实现时可在同一 HDF5 内用 alias 表达这两个逻辑角色，避免复制数值。该 alias 只允许在 v3 文件内部使用，不能用跨版本 hard link 连接 v2 与 v3。

`human_hand_robot_frame_10d` 源自 v1 `/action`。其 verified role 是 pool-side robot-frame human hand pose；是否等同原采集系统的 retarget command 不影响该 pool 角色，保持 unknown。

#### M2-v03 时间 metadata

```text
nominal_camera_fps: 30.0
nominal_camera_fps_status: verified_upstream
nominal_camera_fps_source: arxiv:2502.16587v4 Appendix A
record_timebase_status: coarse | discontinuous | unknown
frame_selection: all_source_rows
timestamp_resolution: source integer field; semantics partially unknown
```

不得用 `nominal_camera_fps=30` 把全部 episode 写成严格 `0,1/30,2/30,...`。M2 继续只保存 source step/timestamp、gap 和 segment。

#### M2-v03 语义 metadata

```text
euler_unit: degree
euler_order: XYZ
euler_evidence_status: verified_upstream
gripper_open_value: 1
gripper_closed_value: 0
gripper_evidence_status: verified_upstream
source_action_role: human_hand_pose_in_robot_frame
source_action_role_status: verified_upstream
source_action_as_robot_command_status: unknown
xyz_source_unit_status: unknown
```

每项证据必须记录 URL、文档版本和访问日期，不能只写 `verified` 而没有 provenance。

#### M2-v03 统计

只读取 train split，分别生成：

- `robot_observed_statistics.json`
- `human_hand_robot_frame_statistics.json`
- `dataset_statistics_post_norm_by_role.json`
- `data_quality_statistics.json`

M2 不生成 generic `dataset_statistics.json` 供训练直接消费，也不生成 residual delta statistics。真正的 policy action statistics 在 M3 选定 pool/query action view 与 alignment 后生成。

#### M2-v03 validator

validator 继续分为结构、时间真实性和证据/角色三类。

1. 结构验证：schema、shape、dtype、共享 T、finite、rot6d、gripper、workspace 和 RGB 解码。
2. 时间真实性：source row 一一映射、source hash、step/timestamp、gap、segment，以及无全局伪时间轴。
3. 证据/角色：30 Hz provenance、Euler/gripper provenance、禁止 generic action、字段 role 与统计 provenance 一致。

validator 必须允许并要求 verified nominal 30 Hz，同时继续拒绝把 coarse/discontinuous record timebase 标成全局 trusted。

#### M2-v03 单元测试

- nominal camera FPS 与 record timebase 状态可以同时存在。
- numeric 30 Hz 有证据时通过，无证据时失败。
- `/action` 不得自动写成 generic robot action。
- Euler degree XYZ 和 gripper 极性转换正确。
- gap、rollback、step jump 和 segment 逻辑保持 v02 回归。
- v1/v2 loader 被 v3 正式 dataloader hard reject。
- source action 与 robot observed trajectory 的字段不可交换。

#### M2-v03 验收标准

- 20 条 episode、20 个任务、source row 100% 保留。
- human/robot/trajectory/raw metadata 第一维一致，必要数值 finite。
- 20/20 source step、timestamp 和 SHA-256 回查一致。
- nominal 30 Hz 绑定 v4 证据，但不存在全局伪造的严格时间轴。
- 每条 episode 有 record continuity、gap 和 segment 报告。
- `/action` 和 `/end_position` 使用不同、正确的 role 名称。
- generic `actions` 在 canonical v3 中不存在或被 validator 拒绝。
- split 固化；所有 M2 统计只来自 train split 并按 role 分开。
- 10 条新可视化通过人工抽查，overlay 不再误标 action。
- 新契约、自动报告和人工报告保存到 `方案/v03/`。

### M3-v03：action-role、time-view 与 residual sanity check

状态：**已完成，通过 Gate A2a、Gate A2b 与 Gate B**。验收报告为 `M3_action_time_residual_验收报告.md`；M3-v02 方案被本节取代。

#### M3 Stage 0A：pool/query 字段角色

RECAP 分别需要 pool-side action plan 与 query-side robot target。两者角色不同，不能继续共用一个含糊的 `action_view_id`。

Gate A2a 依据官方数据集卡直接通过：

- `human_hand_robot_frame_10d` 是 pool-side human plan，来源为 v1 `/action`。
- `/end_position + /gripper_state` 是 observed robot EE trajectory，官方允许其作为 BC action label。
- 发布数据未证明任何字段是原控制器实际收到的低层 command；该结论不阻塞离线 M3/M4。

Stage 0A 生成 `action_role_audit.json`。新增证据写入该 M3 产物，不修改已被 M2 HDF5 哈希绑定的 `source_evidence_manifest_v3.json`。

#### M3 Stage 0B：离线 action proxy 与 alignment

在连续 segment 内计算 human plan 与 observed robot trajectory 的跨相关、最佳 lag，以及 position/orientation/gripper 误差。所有阈值只由 train split 确定。

Pool action view 候选：

| `pool_action_view_id` | 定义 | 角色 |
|---|---|---|
| `human_hand_robot_frame_raw` | canonical `/action` 轨迹 | 主 pool-side coarse plan |
| `human_hand_phase_aligned` | human plan 经 phase/DTW 重参数化 | temporal mismatch 候选 |

Query action view 候选：

| `query_action_view_id` | 定义 | 角色 |
|---|---|---|
| `robot_ee_observed_t` | 当前 row 的 observed EE | 诊断基线；不得作为主同帧监督 |
| `robot_ee_observed_t_plus_1_bc_proxy` | 下一连续 row 的 observed EE | dataset-card-approved BC proxy |
| `robot_ee_future_horizon_lag_calibrated_proxy` | 连续 segment 内经 train-only lag 标定的 future EE chunk | 主候选 |

每个组合另行绑定 `action_alignment_id`，记录 lag、尺度变换、坐标表示、gripper 规则、末帧策略和 gap policy。

离线阶段不要求知道原控制器 API 或真实物理单位名称，但必须证明 pool/query 处于相同 canonical 数值尺度。任何尺度假设都要进入 manifest 和消融，且不得据此报告 m/s。

Gate A2b 只批准明确的 BC proxy，不把它升级为 `verified executable command`。若 future/lag proxy 均不能产生稳定 residual，则暂停 residual 主线，转为 trajectory/future-state 任务。

#### M3 时间视图候选

至少比较：

1. `native_row_index`：保留 source row，仅作基线。
2. `nominal_camera_30hz_segmented`：只在连续 segment 内使用 nominal 30 Hz。
3. `paper_v2_stride4_nominal7p5`：复现论文 v2 的四分之一采样。
4. `legacy_v01_stride3_nominal10`：复现旧 M2，仅作负面对照。
5. `policy_clock_<hz>`：重采样到候选 policy clock，不能跨 gap。
6. `phase_or_dtw`：按关键事件或任务阶段对齐不同执行节奏。

`paper_v2_stride4_nominal7p5` 必须记录 `paper_version=v2`。它不能被写成最新版论文强制设置，也不能替代 v3 canonical。

#### M3 derived view 规则

- RGB：最近邻选帧。
- xyz：连续 segment 内线性插值。
- rotation：连续 segment 内 SLERP。
- gripper：zero-order hold，并保留切换事件。
- observed trajectory：按角色标记，不自动提升为 command。
- query action chunk 必须严格指向当前 observation 之后的 future rows；同帧值只作泄漏诊断。
- pool/query chunk 经 alignment 后必须具有相同 horizon 和 10D canonical 表示。
- M3/M4 manifest 中 `deployment_command_adapter_id` 固定为空；不得伪装已具备真实执行语义。
- velocity：只有 nominal clock 和 step 语义门禁通过时才报告 m/s。
- 任何 view 的 gap crossing count 必须为 0。

#### M3 检索与 residual

每个 retrieval index 必须同时绑定 `time_view_id`、`pool_action_view_id`、`query_action_view_id` 与 `action_alignment_id`。

```text
residual(τ; time_view_id, pool_action_view_id,
             query_action_view_id, action_alignment_id)
  = selected_query_bc_target(query, τ)
    - aligned_pool_human_plan(pool, τ)
```

若 query target 不是 Gate A2b 批准的 proxy，或 pool/query 不共享明确的 canonical 表示，则 residual 只能标为 exploratory，不能进入 M4 主训练。

#### M3-v03 验收标准

- pool/query action role 分离，Gate A2a 证据完整。
- Gate A2b 批准一个 future/lag query BC proxy；同帧 `robot_ee_observed_t` 只作诊断。
- human plan 与 observed robot trajectory 的 lag、尺度和误差报告完整。
- 每个有效 query 至少有 top-10 候选。
- 主 view gap crossing 为 0。
- random retrieval 在语义、阶段或 residual 指标上明显更差。
- 主 pool/query/time/alignment 组合的 residual norm 中位数低于 absolute target norm。
- stride4、stride3、native、nominal30 和 phase/DTW 均以相同 episode 与 pool/query role 比较。
- held-out robot trajectory 不进入 inference retrieval feature。
- 错误 role、同帧复制、错误 lag 和尺度扰动必须显著恶化指标。
- 若 residual 不稳定或 query proxy 不成立，暂停 residual M4，转为 future-state 路线。

### M4-v03：Human2Robot paired bridge 训练

状态：可启动；M3-v03 已通过 Gate B。

M4 目标保持不变：在 seen paired task 上学习 human-to-robot bridge，在 held-out task 上只增加 human pool，检查冻结模型是否改善机器人轨迹预测。

每个 dataloader 和 checkpoint 必须显式绑定：

```text
canonical_schema
source_evidence_manifest_hash
time_view_id
pool_action_view_id
query_action_view_id
action_alignment_id
pool_action_role
query_action_role: dataset_card_approved_bc_proxy
query_command_status: unverified
policy_dt 或 phase coordinate
H_steps / H_seconds
K_steps / K_seconds
gap_policy
alignment_version
split_hash
```

M4 不得根据 generic `actions` 自动推断监督字段。checkpoint 加载时，pool/query action、alignment 或 time view 不一致必须 hard fail。

固定对比仍包括 `No retrieval`、`Retrieval Only`、`Co-training` 和 `RECAP hand-ret`。所有方法必须使用同一 pool/query role、alignment、time view、split 和模型预算。

新增 action-view 消融：同帧 observed、`t+1` BC proxy、lag-calibrated future proxy，以及 raw/phase-aligned human plan。只有 Gate A2b 批准的组合可进入主结果。

新增论文版本消融：`paper_v2_stride4_nominal7p5` 与 v03 主 view。该对比用于判断作者旧训练节奏是否适合 RECAP，不用于声称 v4 必须使用 7.5 FPS。

M4 验收标准继续要求 RECAP 优于 `No retrieval` 和 `Retrieval Only`，pool-growth 总体改善，且无长期 residual saturation、gap crossing 或 workspace clipping。

### M5-v03：机制消融与压力测试

状态：**M5-A 已完成首轮数据/契约压力测试；M5B-P0-IMPLEMENTATION 已通过，M5B-P1/P2 与后续模型门禁 pending**。M5-A 启动报告为 `M5A_data_contract_stress_启动报告.md`，P0 报告为 `M5B_P0_IMPLEMENTATION_验收报告.md`。

M5-A 绑定 M3 Gate B 与 M4 冻结契约，只验证错误 role、lag、scale、时间和分辨率处理能否被数据/契约检测器发现。首轮结果中四类 action stress、六种 time view、frame drop/jitter/pause/step jump 检测和 426↔424 crop/pad 契约均通过；所有 time view 和 temporal stress 的 gap crossing 均为 0。

M5-A 不证明最终模型收益或鲁棒性。M5-B 仍需正式 M4 多 seed checkpoint、独立或扩充 human-only pool，才能执行 residual/absolute、future-state、retrieval modality、geometry/visual、top-k 和正式 pool-growth 消融。Gate C 继续保持 pending。

M5-B 正式验收协议 v1 已冻结为 `M5B_formal_acceptance_protocol_v1.json`，人类可读版本为 `M5B_formal_acceptance_protocol_v1.md`，锁文件记录协议与 validator SHA256。协议固定使用 seeds `20260711/20260712/20260713`，统计单位为 held-out task×seed，并预注册四方法主比较、全部模型消融、统计检验、非劣界限、guardrails 和 M5 完全通过规则。P0 已在完整项目 Docker/CUDA 环境中通过：formal dataset/model adapter、3 learned methods×3 seeds 配置、Hydra 加载、15 项 pytest、本地初始化 checkpoint SHA256 绑定，以及真实 2B 同批固定噪声 50-step overfit 均通过；真实 2B loss 从 0.3335 降至 0.0039。该诊断不替代 2B 正式训练；每个 held-out task 的 10 条独立 human demonstrations、正式运行统一 data-parallel world size、step-7000 checkpoint 与全部统计门禁仍是硬前置。

保留 residual/absolute、future-state、retrieval modality、geometry/visual、top-k、pool-growth 和 temporal mismatch 消融。

新增 action-role 压力测试：交换 pool/query role、注入 lag/尺度错误、使用同帧 observed state 作为监督，并检查 residual 和 future-state 指标是否暴露错误。

新增 FPS/version 测试：nominal30、paper-v2 stride4、legacy stride3、policy clock 和 phase/DTW；同时注入 frame drop、timestamp jitter、暂停和 step jump。

新增分辨率处理测试：论文 240×424 与本地 HDF5 240×426 的 crop/pad 策略必须显式记录，并验证不会改变检索或 action 结论。

### M6-v03：同场景真实机器人实验

状态：待执行。

任务继续使用 seen `open_cabinet`，以及 unseen `close_cabinet` 和 `put_bottle_in_box`。

Human2Robot 的 nominal 30 Hz 不外推到 M6。真实系统必须重新测量 camera capture、observation query、policy、waypoint、servo、latency、jitter 和 action queue 时长。

真实数据必须记录 command、observed EE、human pose 和每条流的单调 timestamp。字段 role 必须由采集代码定义，而不是事后根据数组名称推断。

M6 rollout 前必须通过 deployment command gate，并生成唯一的 `deployment_command_adapter_id`。adapter 把 M4 的 canonical query action 映射为目标机器人的可执行接口。

adapter 必须声明 control API、absolute/relative 模式、base/tool frame、m/mm、rad/degree、控制频率、command queue、gripper 语义、workspace、速度/加速度限制和急停策略。

Human2Robot 的 query BC proxy 不自动证明真实 command 语义。只有目标机器人 command→observed 的阶跃、轨迹与 gripper 标定通过后，才允许执行模型 rollout。

M6 继续要求模型只在 seen paired task 上训练一次；unseen task 只追加 human pool；所有速度、加速度、workspace、force 和急停约束按实测物理时间定义。

#### M6-v03 验收标准

- unseen task 上 `RECAP hand-ret` 明显优于 `No retrieval`。
- 优于 hand playback/retarget-only。
- pool 从 0 到 10 条时总体改善。
- 全程不更新模型参数。
- 日志能还原 observation、retrieval、prediction、command 和 execution 的真实 timestamp 与 role。
- `deployment_command_adapter_id`、安全配置和 calibration report 与 rollout manifest 一致。

## 5. M2-v01 与 M2-v02 代码和中间数据的正确处理

### 5.1 总原则

不删除、不静默覆盖、不把旧验收报告改写成“从未发生”。冻结每代 schema、代码 hash、配置、统计和报告，再在新路径生成 v3。

### 5.2 版本状态

| 版本 | 状态 | 允许用途 | 禁止用途 |
|---|---|---|---|
| canonical/v1 | `acceptance_withdrawn` | 固定 stride 回归、dtype/schema、负面对照 | v03 训练、主索引、论文主结论 |
| canonical/v2 | `semantic_acceptance_reopened` | native frame、RGB、gap/segment、split、validator 回归 | v03 action stats、residual、M4 主训练 |
| canonical/v3 | `passed_gate_a1` | 唯一正式 canonical 主线；通过 approved derived views 进入 M3/M4 | 不得绕过 Gate A2b 直接生成训练 action |

### 5.3 v02 代码处理

| 文件/能力 | v03 处理 |
|---|---|
| `tools/human2robot_m2.py` | 重构为 v3 semantic-safe converter；保留显式 v1/v2 legacy 读取或独立冻结入口 |
| `convert_human2robot_m2.py` | 默认输出 `canonical/v3`；禁止默认复用 v2 HDF5 |
| `validate_canonical_hdf5.py` | 新增 evidence/role validator；保留 v2 validator 用于历史复验 |
| `human2robot_m2_test.py` | 保留 v2 回归，新增 FPS provenance、role 隔离和 loader hard-fail 测试 |
| gap/segment 与 source hash | 原样复用并增加回归测试 |
| v02 generic action statistics | 只读冻结，不导入 v3 |

在修改前先保存当前代码 SHA-256、CLI help、测试结果和 v02 conversion config。若使用 git，先提交或打 tag；若不提交，至少生成不可变 code manifest。

### 5.4 v02 HDF5 处理

不得原地修改 `data/Human2Robot/canonical/v2/pilot/*.hdf5`。原地修改会破坏既有 report/hash，并使历史结论无法复现。

在 `canonical/v2/` 新增 `SUPERSEDED_M2_V02.json`，状态为 `semantic_acceptance_reopened`。marker 至少记录新证据、受影响字段、原报告/hash、允许用途和禁止用途。

v3 使用相同 20 条 source episode 重新生成。即使图像内容相同，也优先完整原子重写，避免 hard link 或共享 inode 导致 v2 被间接修改。

只有磁盘压力明确、reflink 行为经过验证且用户批准时，才考虑 copy-on-write 优化。不得删除 v1/v2 来为 v3 腾空间。

### 5.5 v02 manifest 与 split

v02 episode selection 和 task split 可作为 v3 的 parent evidence，因为它们不依赖 action 语义。

v3 必须生成新 split manifest，记录 `parent_v2_split_sha256`、v3 schema 和新统计 hash。不得直接复制后宣称为 v3 manifest。

### 5.6 v02 统计

以下 v02 统计含有 generic action 解释，不得进入 v3 dataloader：

- `dataset_statistics.json`
- `dataset_statistics_post_norm.json`
- `data_quality_statistics.json` 中与 generic action 相关的位移项

其中 robot state、timebase 和 gap 的数值可用于回归对比，但 v3 必须从 role 明确的 train 数据重新计算统计。

### 5.7 v02 validator 与报告

v02 validator 的结构、source mapping、RGB、finite、rot6d、split 和 gap 结果继续有效。

[v02 验收报告](../v02/M2_Human2Robot_native_time_验收报告.md)顶部应增加“语义验收重开”说明，链接 v03 方案。原来的“允许进入 M3”结论暂停，但历史 20/20 结构验收不改写。

### 5.8 v02 可视化

v02 MP4 可用于 paired RGB、native frame、source step/timestamp 和 segment 诊断。

其中 action overlay 的文字标签不再可信，只能作为旧实现回归。v3 重新生成 role-separated overlay，不覆盖旧视频。

### 5.9 迁移顺序

1. 冻结 v02 code/config/report/statistics SHA-256。
2. 添加 `SUPERSEDED_M2_V02.json` 和旧报告醒目标记。
3. 创建 v3 evidence manifest 与 canonical contract。
4. 重构 converter、validator、statistics 和 tests。
5. 用同一 20 条 source episode 生成 `canonical/v3`。
6. 完成三类 validator 和 10 条人工可视化验收。
7. M2-v03 通过后，启动 M3 Stage 0A/0B 的双侧 action-view 与 alignment calibration。
8. Gate A2b 批准 query BC proxy 后，才生成正式 residual index。

## 6. v03 实验矩阵

| 阶段 | 数据/视图 | 训练对象 | 测试对象 | 核心问题 |
|---|---|---|---|---|
| M0 | PushT 10 Hz env | 已完成 | 已完成 | RECAP residual RAG 是否复现 |
| M1-v03 | source + evidence | 不训练 | 证据审计 | 字段、FPS、单位和角色证据是否足够 |
| M2-v03 | semantic-safe native rows | 不训练 | schema/time/role QA | 是否无损、可追溯且不误标 action |
| M3 Stage 0A | canonical role evidence | 不训练主策略 | paired pilot | pool human plan 与 query robot trajectory 角色是否分离 |
| M3 Stage 0B | pool/query action proxy + alignment | 不训练主策略 | paired pilot | 哪个 future/lag proxy 可产生稳定 residual |
| M3-v03 | pool/query/time/alignment views | 不训练主策略 | held-out paired | 哪个组合能稳定 residual |
| M4-v03 | selected pool/query/time/alignment view | seen paired | held-out task | human-to-robot bridge 是否可学 |
| M5-v03 | role/time/resolution 扰动 | 消融 | robustness | 收益是否来自机制且耐语义错配 |
| M6-v03 | 自采真实 clock/role 数据 | open_cabinet | 两个 unseen task | 冻结模型+新 human pool 是否有效 |

## 7. 公平性和数据泄漏要求

- train/held-out 必须按 task 划分。
- normalization、alignment、lag calibration 和 time-view 参数只能使用 train split。
- held-out robot trajectory 只用于评测，不能进入 inference retrieval feature。
- action proxy、lag、尺度和 alignment 阈值只能用 train task 确定。
- 所有 temporal baseline 必须使用相同 episode、pool/query role、模型预算和 gap policy。
- stride4 与 stride3 必须标记论文/历史版本，不能混入主 view。
- pool-growth 各方法必须看到相同 human pool。
- 结果必须记录 evidence、canonical、split、time、pool/query action、alignment、index、model 和 seed hash。

## 8. 阶段门禁

### Gate A1：M2-v03 canonical 验收

通过条件：全 source row 保留；30 Hz evidence 正确；record continuity 独立分级；role-safe schema；split 和 role-separated statistics 固化；无 generic action。

### Gate A2a：M3 Stage 0A 字段角色验收

通过条件：`/action` 固定为 pool-side human plan；`/end_position + gripper_state` 固定为 observed robot trajectory 和 dataset-card-approved BC label。A2a 现已具备上游证据。

### Gate A2b：M3 Stage 0B 离线 action proxy 验收

通过条件：query future/lag BC proxy、pool/query canonical 数值尺度、alignment、gripper 规则和 gap policy 明确；错误 role/lag/scale 显著恶化指标。不能成立时停止 residual 主线，转 future-state。

### Gate B：M3-v03 → M4-v03

通过条件：无 gap crossing；主 pool/query/time/alignment view 优于 random 与 legacy；aligned residual 比 absolute target 更稳定；lifting 和 alignment confidence 达标。

### Gate C：M4-v03 → M6-v03

通过条件：RECAP 优于 `No retrieval/Retrieval Only`；role/time 消融支持主 view；pool-growth 改善。该 gate 只批准离线 bridge，不批准真实执行。

### Gate D：M6 deployment command 与安全验收

通过条件：`deployment_command_adapter_id`、control API、单位、坐标系、absolute/relative、clock、H/K、latency、queue、gripper 和安全约束已标定；command→observed 测试通过后才允许 rollout。

## 9. 关键风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 把 nominal 30 Hz 当成严格 record clock | 速度和 horizon 错误 | clock/continuity 分离，segment 内显式 derived view |
| `/action` 被误标为 query robot target | residual 物理意义错误 | 固定为 pool-side human plan，pool/query view 分离 |
| 同帧 observed state 被当作未来 target | 状态复制或时序泄漏 | `t+1`/future-lag proxy、同帧负向基线 |
| BC proxy 被误称为 executable command | 离线结果无法安全部署 | query command status 固定 unknown，M6 独立 adapter gate |
| v2 generic stats 被复用 | normalization 语义污染 | loader hard reject，v3 按 role 重算 |
| 7.5 FPS 被当成最新版规范 | 论文版本误读 | 标记 `paper_v2`，只作 derived baseline |
| step/timestamp 语义未证 | 不能恢复实测 dt | 寻找采集代码；否则只用 nominal 或 phase |
| episode jump/pause | chunk 跨无效区间 | gap mask + segment，crossing=0 |
| 240×424 与 240×426 不一致 | crop/pad 改变视觉特征 | 记录 source shape，做处理消融 |
| xyz 物理单位未证 | 物理速度和部署尺度错误 | M4 前验证共同数值尺度并做尺度消融；M6 前确认物理单位 |
| v1/v2/v3 混用 | 结论不可复现 | 独立路径、schema marker、manifest 和 hard check |
| public 30 Hz 外推真实机器人 | 部署 H/K 与 latency 不匹配 | M6 独立 clock/latency 标定 |

## 10. v03 交付物

### M1-v03

- `source_evidence_manifest_v3.json`
- 论文版本与数据集卡证据摘要
- 未解决证据清单

### M2-v03

- `human2robot_canonical_hdf5_contract_v3.md`
- semantic-safe converter、validator 和 tests
- v3 split manifest，绑定 parent v2 split hash
- role-separated train-only statistics
- 20 条 canonical v3 pilot HDF5
- 10 条 role-separated 可视化
- `M2_Human2Robot_semantic_safe_验收报告.md`
- `SUPERSEDED_M2_V02.json` 与 v02 报告重开说明

### M3-v03

- `action_role_audit.json`
- pool-action-view builder 与 manifest
- query-action-view builder 与 manifest
- `action_alignment_id` 配置、lag/scale/gripper 报告
- time-view builder 与 manifest
- retrieval NPZ、alignment metadata 和 index hash
- stride4/stride3/native/nominal30/phase 对比
- residual/absolute 分布与检索可视化
- M3-v03 验收报告

### M4-v03

- 绑定 pool/query action、alignment 和 time view 的 paired bridge 配置
- 四个固定 baseline
- action-role、时间处理和 `H/K` 消融
- pool-growth 曲线
- action、trajectory、future-state、residual 和 safety 报告

### M5-v03

- M5-A（已生成）：实验协议与 claim-to-experiment 映射
- M5-A（首轮通过）：action-role/lag/scale/same-frame 压力测试
- M5-A（首轮通过）：FPS/version 与 frame drop/jitter/pause/step-jump 测试
- M5-A（首轮通过）：240×426 ↔ 240×424 center-crop/edge-pad 契约审计
- M5-A 启动报告与自动 JSON 报告
- M5-B（已冻结）：3-seed 正式验收 JSON/Markdown 协议、validator 与 lock
- M5-B P0（已通过）：formal 2B Human2Robot adapter、9 个 learned-method×seed 配置、本地 checkpoint 绑定、15 项 contract tests 与真实 2B 同批 50-step overfit
- M5-B（待执行）：retrieval modality、geometry/visual、top-k、pool-growth 与 residual/future-state 模型消融

### M6-v03

- 真实机器人 clock/latency/role 标定报告
- `deployment_command_adapter_id`、接口契约与 command→observed 标定报告
- 采集与安全协议
- rollout 日志、成功率和 95% CI
- success/failure 视频与最终结论边界

## 11. 执行顺序

1. 冻结 v02 代码、HDF5、统计、报告和哈希。
2. 为 v02 增加 superseded marker 和语义验收重开说明。
3. 完成 M1-v03 evidence manifest。
4. 编写 canonical v3 contract。
5. 重构 M2 converter、validator、统计和测试。
6. 用相同 20 条 source episode 生成 v3 pilot。
7. 完成 M2-v03 自动与人工验收。
8. 执行 M3 Stage 0A/0B，固化 pool/query action view、future/lag proxy 和 alignment。
9. Gate A2b 通过后构建各 time/pool/query/alignment view 和 retrieval index。
10. residual sanity gate 通过后进入 M4；完成 ridge smoke 闭环并保持 Gate C pending。
11. 启动 M5-A 数据/契约压力测试，冻结扰动、检测器、crop/pad 策略和 claim boundary。
12. 冻结 M5-B 3-seed 正式验收协议；完成 P0 formal adapter 与本地 checkpoint SHA 绑定；下一步将每个 held-out task 的独立 human pool 从 1 条扩充到 10 条，并在正式 launch 时冻结统一 data-parallel world size。
13. 按冻结协议完成正式 M4 训练和 M5-B 模型依赖型消融。
14. M4/M5 证据满足 Gate C 后，实现并标定 M6 deployment command adapter。
15. Gate C 与 Gate D 均通过后执行真实 rollout。

## 12. 结论与不能过度声称的边界

v03 接受“D435 nominal capture rate 为 30 Hz”，但拒绝把它等同于发布 HDF5 的全局严格行时间。native row、gap、segment 和 evidence provenance 继续是 canonical 的核心。

论文 v2 的 7.5 FPS 是有价值的复现基线，但不是 canonical 事实，也不是最新版论文对 RECAP 的强制设置。

M2-v02 仍证明了全帧映射、RGB/dtype、source hash、split、gap/segment、原子写入和可视化链路可用。它的工程成果被继承，但 FPS 与 action 语义结论被 v03 取代。

M2-v03 已通过，代表字段事实、时间证据和角色命名正确。Gate A2a 已确定 pool/query 字段角色；Gate A2b 才批准具体 future/lag BC proxy 与 alignment。

M3-v03 通过才代表 residual 在指定 pool/query/time/alignment view 下合理；M4 通过才代表 public paired bridge 可学。

M4 的 BC proxy 不等于 executable command。只有 Gate D 与 M6 同场景真实机器人 rollout 通过，才能支持“冻结模型后，通过新增人手示范执行未见任务”的最终结论。
