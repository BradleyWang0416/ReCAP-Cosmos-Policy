# RECAP 人手示范用于上下文学习指导真实机器人复现实验方案 v01

日期：2026-07-05

## 1. 总体目标

目标是从当前已经完成的 PushT 复现继续推进，验证 RECAP 论文中更关键的真实机器人命题：

> 只在一个 paired 任务上训练一次，冻结模型后，通过新增人手示范检索池，让机器人执行未见任务。

当前 `笔记/RECAP_论文实验结果复现与对比公平性评估.md` 已经说明：

- PushT residual RAG 复现的 unseen average 是 35.1%。
- 论文 Table 2 的 RECAP/Ours unseen average 是 34.9%。
- 当前训练/推理链路与论文 PushT 主结果基本对齐。

因此，v01 方案不再把“RECAP retrieval-conditioned residual policy 在当前环境中是否跑通”作为待验证主任务，而是将它视为 **M0 已完成**。后续重点转向公开 paired human-robot 数据上的桥接验证、human-only 数据的轨迹抽取压力测试，以及最终同场景真实机器人实验。

## 2. 小里程碑

### M0：PushT 训练/推理链路复现

状态：已完成。

验收依据：

- 当前 PushT residual RAG 主线已能评估冻结模型。
- unseen average 35.1%，与论文 34.9% 基本一致。
- seen average 53.0%，论文为 50.0%，差异在 50-trial rollout 随机波动范围内。

后续只在代码、环境或 checkpoint 发生变化时重跑 PushT smoke test。

### M1：数据下载与访问确认

目标：先用公开 paired human-robot 数据验证桥接假设，不立刻自采。

优先下载：

1. **MIME**
   - 作用：首个 public paired pipeline pilot。
   - 优先任务：`Place objects in box`、`Open Bottles`、`Close Book`。
   - 原因：每个数据点包含 human demonstration 和 Baxter robot demonstration，适合验证 human video / robot trajectory 对齐与 residual target 构造。

2. **RH20T**
   - 作用：规模化 public paired bridge 验证。
   - 先下载 `Task Description File`。
   - pilot 数据优先取 `RH20T_cfg3` 的 320x180 RGB、LowDim、Calibration。
   - 只有在需要更稳定的 3D human/object lifting 时，再补 cfg3 Depth。

3. **H&R / Human2Robot**
   - 作用：如果能获取，是最接近 perfectly aligned human-robot pair 的公开数据。
   - 风险：当前只确认论文中描述了 2600 条 paired synchronized episodes，未确认稳定公开下载入口。
   - 决策：尝试获取，但不让主线依赖它。

暂缓下载：

- **HOI4D**
- **ARCTIC**
- **TACO**

这些数据集只有人手/物体侧或主要用于 hand-object annotation，不优先作为 bridge 训练集。它们放到 M5，用于测试 wrist pose、object pose、contact、articulated state 抽取是否稳定。

M1 验收标准：

- 明确每个数据集的下载位置、许可、体量、所需磁盘空间。
- 成功打开至少一个 MIME 任务的数据样本。
- 成功打开至少一个 RH20T episode 的 robot low-dim、human video、camera calibration。
- 产出 `data_inventory.md`，记录每个候选数据源是否可用、是否需要授权、是否进入后续处理。

### M2：统一预处理与 canonical HDF5

目标：把 public paired 数据转换成当前 RECAP/Cosmos Policy 代码容易接入的 episode 格式。

统一 schema：

```text
data/<split>/demo_x.hdf5
└── data/demo_i/
    ├── obs/images
    ├── obs/states
    ├── actions
    └── metadata
```

默认 action/state 表示：

- public pilot 阶段默认使用单臂 10D EE 表示：`xyz + rot6d + gripper`。
- 如果任务或平台明确需要双臂，再扩展为 20D：左臂 10D + 右臂 10D。
- 单臂任务在 20D 表示中将 inactive arm 填 no-op，并在 metadata 记录 active arm。

MIME 预处理：

- 从 Baxter joint angle 做 FK，得到 robot EE pose。
- 从 human RGB-D 中抽取 wrist pose 与 grip aperture。
- 用任务阶段、关键物体状态或 DTW 对齐 human 与 robot episode。
- 对齐置信度低的样本只进入可视化检查，不进入训练。

RH20T 预处理：

- 从 `tcp_base.npy` 得到 gripper Cartesian pose。
- 从 `gripper.npy` 得到 gripper width / command。
- 从 paired human video、timestamps、calibration 抽取 human wrist/grip。
- 优先保留 calibration quality 和 completion quality 较高的样本。

必须生成的统计文件：

- `dataset_statistics.json`
- `dataset_statistics_post_norm.json`
- `delta_dataset_statistics.json`

注意：不要依赖训练第一次启动时隐式生成这些文件。统计文件应作为预处理产物显式生成并记录版本。

M2 验收标准：

- 每个 pilot 数据集至少转换 20 条 episode。
- HDF5 可被轻量脚本逐条读取。
- 无 NaN、时间戳单调、帧率统一到 10 Hz 或 16 Hz。
- action/proprio 维度一致，gripper 范围、workspace 范围、速度范围全部有校验。
- 随机抽样 10 条 episode 生成 human/robot/action 可视化视频。

### M3：检索索引与离线 sanity check

目标：构造 RECAP 所需的 query-to-pool subframe 检索索引。

输出格式兼容当前 `PushTRetDataset` 风格：

```text
query_ids:  (N,)    str, 形如 query_split/demo_i/t
match_ids:  (N, K)  str, 形如 pool_split/demo_j/t'
match_sims: (N, K)  float32
```

检索策略：

- Stage 1：按 task language、初始物体状态、初始 proprio 筛候选 episode。
- Stage 2：按 object pose、robot proprio、human wrist/grip、速度项、视觉 embedding 做 subframe nearest-neighbor。
- pilot 版本先实现 geometry-only 检索，再加入 DINOv2/object crop embedding。

离线 sanity 指标：

- same-task top-k 命中率。
- random retrieval 对照。
- retrieved action 与 target robot action 的 raw residual norm 分布。
- residual target 是否比 absolute action 更稳定。
- 检索片段可视化是否语义一致。

M3 验收标准：

- 每个 query step 至少有 top-10 候选。
- 随机检索 baseline 明显更差。
- residual norm 中位数低于 absolute action norm 中位数。
- 如果 residual action 不比 absolute action 稳定，暂停进入 M4，先修检索或对齐。

### M4：公开 paired bridge 训练

目标：验证“human/pool 到 robot/query 的 bridge 是否能学出来”。

训练顺序：

1. MIME pilot：小规模确认 dataloader、normalization、retrieval context、residual target、forward/backward 全链路。
2. RH20T pilot：验证更复杂任务与多相机/多机器人配置下的泛化。
3. 如果 H&R 获取成功，再作为最接近 RECAP 假设的 paired 数据加入主实验。

固定对比方法：

| 方法 | 目的 |
|---|---|
| `No retrieval` | 验证只靠 seen robot query 数据的泛化能力 |
| `Retrieval Only / hand playback` | 验证直接执行检索到的人手伪动作是否足够 |
| `Co-training` | 验证简单混合 query/pool 数据是否能替代 test-time retrieval |
| `RECAP hand-ret` | 检索条件化 residual policy + future-state prediction |

主指标：

- robot action MAE。
- position error。
- orientation error。
- gripper error。
- DTW trajectory distance。
- future-state prediction error。
- retrieval-conditioned residual norm。

Pool-growth 设置：

- 对每个 held-out task 使用 `0, 1, 3, 5, 10` 条 human pool demos。
- 模型参数全程冻结。
- 观察 action reconstruction 和 trajectory error 是否随 pool 增长总体下降。

M4 验收标准：

- `RECAP hand-ret` 优于 `No retrieval` 和 `Retrieval Only`。
- `RECAP hand-ret` 不低于或优于 `Co-training`，至少在 held-out task 上更稳定。
- pool-growth 曲线呈总体改善趋势。
- 残差预测没有出现长期 saturated delta 或 workspace clipping。

### M5：消融与 human-only 压力测试

目标：确认收益来自 RECAP 机制，而不是数据量、检索巧合或动作平滑。

消融实验：

- residual action vs absolute action。
- with future-state prediction vs without future-state prediction。
- retrieved image / state / action dropout。
- geometry-only retrieval vs geometry + visual embedding retrieval。
- top-1 retrieval vs top-k random / weighted retrieval。

human-only 数据测试：

- HOI4D：测试 egocentric RGB-D 下的 hand pose、object pose、action segmentation。
- ARCTIC：测试 articulated object、contact、bimanual hand-object state。
- TACO：测试 tool-action-object triplet 与 bimanual motion forecasting。

human-only 数据的使用边界：

- 只用于轨迹抽取、检索特征和鲁棒性调试。
- 不直接支撑“RECAP 真实机器人复现成功”的主结论。
- 如果无法稳定 lifting 成 state/action trajectory，不进入 final pool。

M5 验收标准：

- 每个关键模块消融都有明确差异。
- human-only 抽取失败率、轨迹平滑前后误差、workspace violation 都有记录。
- 明确哪些 human-only 数据可以进入检索特征调试，哪些只能作为失败案例。

### M6：小规模同场景真实机器人实验

目标：复现 RECAP 论文真实机器人设置的核心结论。

任务设置：

- seen task：`open_cabinet`
- unseen task 1：`close_cabinet`
- unseen task 2：`put_bottle_in_box`

采集协议：

- `open_cabinet`：25 条 human demos + 25 条 robot demos，作为 train-time paired data。
- `close_cabinet`：只采 10 条 human demos，作为 test-time pool。
- `put_bottle_in_box`：只采 10 条 human demos，作为 test-time pool。
- unseen tasks 不采新的 robot training demos。

训练/测试协议：

- 只在 seen paired task 上训练一次。
- 训练完成后冻结模型参数。
- unseen task 到来时，只向 retrieval pool 追加对应 human demos。
- 每个控制 chunk 重新检索，预测 normalized delta，反归一化后执行 `retrieved_action + predicted_delta`。

真实机器人安全约束：

- workspace clipping。
- 速度与加速度限制。
- gripper force / width 限制。
- 碰撞检测或人工急停。
- 每次 rollout 记录是否触发 safety stop。

真实 rollout 指标：

- success rate。
- binomial 95% CI。
- completion time。
- safety interruption count。
- retrieval switch rate。
- residual norm。
- failure category。
- representative success/failure videos。

M6 验收标准：

- unseen tasks 上 `RECAP hand-ret` 明显优于 `No retrieval`。
- `RECAP hand-ret` 优于直接 hand playback / retarget only。
- pool 从 0 到 10 条 human demos 时，成功率或关键离线指标总体改善。
- 全程不更新模型参数。

## 3. 实验矩阵

| 阶段 | 数据 | 训练对象 | 测试对象 | 是否真实机器人 | 主要问题 |
|---|---|---|---|---:|---|
| M0 | PushT | 已完成 | 已完成 | 否 | RECAP PushT 链路是否复现 |
| M1-M3 | MIME | paired human/robot | held-out MIME tasks | 否 | canonical 格式与检索索引是否成立 |
| M4 | MIME/RH20T | paired human/robot | held-out public tasks | 否 | human-to-robot bridge 是否可学 |
| M5 | HOI4D/ARCTIC/TACO | 不训练主策略 | trajectory extraction | 否 | human-only 能否稳定 lifting |
| M6 | 自采同场景 | open_cabinet paired | close_cabinet / put_bottle_in_box | 是 | 冻结模型 + 新增 human pool 是否有效 |

## 4. 公平对比要求

所有主对比必须满足：

- 相同 train split。
- 相同 held-out split。
- 相同可见 human pool。
- 相同 action/state 表示。
- 相同模型 backbone 或明确说明差异。
- `RECAP hand-ret` 与 baselines 都不能在 unseen robot data 上训练。
- `Co-training` 只能使用 train-time 可见数据，不能使用 test-time unseen robot demos。

不要声称：

- public-only 结果等价于真实机器人复现。
- human-only 数据直接证明 RECAP 主结论。
- 10 rollouts 足以支撑强统计结论。
- 当前 PushT GT-state retrieval 等价于完整真实机器人视觉检索系统。

## 5. 关键风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| H&R 数据无法公开获取 | 失去最像 RECAP paired 假设的数据 | 主线改用 MIME + RH20T |
| MIME/RH20T human 与 robot 不够严格同步 | residual target 噪声大 | 用 alignment confidence 筛样本，低置信度只可视化 |
| human wrist/grip 抽取不稳定 | 检索匹配错误 | 先做 geometry-only + 手工检查，再加入视觉 embedding |
| 单臂/双臂表示过早复杂化 | 工程耦合过重 | pilot 默认单臂 10D，final robot 再扩 20D |
| public 数据与最终机器人场景差异大 | public 指标不能外推 | public 阶段只证明工程可行性，最终结论用自采 |
| residual 不优于 absolute | RECAP 核心机制未成立 | 暂停训练，优先检查检索质量和 action alignment |

## 6. 交付物清单

M1 交付：

- `data_inventory.md`
- 数据下载脚本或下载说明。
- 每个数据集的许可、体量、任务列表、可用字段表。

M2 交付：

- canonical HDF5 转换脚本。
- HDF5 validator。
- dataset statistics / delta statistics。
- 10 条样本可视化视频。

M3 交付：

- retrieval npz builder。
- 检索质量报告。
- top-k 检索可视化。
- residual norm 对比图。

M4 交付：

- public paired bridge 训练配置。
- 四个 baseline 的离线指标表。
- pool-growth 曲线。
- action reconstruction / future-state prediction 分析。

M5 交付：

- 消融实验表。
- HOI4D/ARCTIC/TACO 抽取质量报告。
- failure cases 汇总。

M6 交付：

- 真实机器人采集协议。
- 真实 rollout 日志。
- 成功率表与置信区间。
- representative videos。
- 最终结论与不能过度声称的边界。

## 7. 默认假设

- 当前 PushT 复现已经完成，不重复作为主任务。
- MIME 和 RH20T 是 public paired 主线数据。
- H&R 如果可获取，则作为更强 public paired 数据加入。
- HOI4D、ARCTIC、TACO 是 human-only 辅线数据。
- public-only 结果只支持工程可行性与桥接证据。
- 论文级真实机器人结论必须来自同场景自采 paired data 和真实 rollout。
