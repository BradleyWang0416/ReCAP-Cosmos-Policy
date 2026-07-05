# RECAP 人手示范用于上下文学习指导真实机器人复现方案

## 1. Summary

- 目标：复现论文 “只在一个 paired 任务上训练一次，冻结模型后通过新增人手检索池让真实机器人执行未见任务” 的结论。
- 默认假设：真实机器人优先；最终结论用自己采集的小规模同场景人手池验证；公开数据只用于预处理、检索和离线调试。
- 关键边界：公开人手视频不能直接证明论文真实机器人结论；必须有少量目标机器人 paired demos 来训练 “人手源端动作/状态 -> 机器人残差动作” 的转换。

## 2. Data Plan

### 2.1 可用数据集

- MIME：首选离线调试数据，含人手视频和 Baxter kinesthetic robot trajectories，适合验证 paired human-robot 数据管线。
- HOI4D：人手 RGB-D、3D hand pose、object pose、分割标注，适合调试人手/物体状态抽取。
- ARCTIC：双手、关节物体、手/物体 mesh 和 contact，适合柜门、盒盖、剪刀等 articulated-object 检索特征。
- TACO：双手 tool-action-object 组合，含 egocentric/third-person views 和 3D hand-object annotations，适合工具类任务。
- DROID / BridgeData V2：机器人数据，不是人手池；用于对比机器人检索、动作统计和离线基线。

### 2.2 自采数据默认协议

- 训练 seen task：`open_cabinet`，采集 25 条人手示范 + 25 条目标机器人示范。
- 测试 unseen tasks：`put_bottle_in_box`、`close_cabinet`，每个任务只采 10 条人手示范，不采新机器人训练数据。
- 每条轨迹统一 16 Hz，保存 RGB/RGB-D、相机内外参、任务语言、机器人 proprio、机器人 action、人手 wrist pose、grip aperture、物体 pose 或任务进度量。

### 2.3 统一 HDF5 格式

采用与当前仓库 retrieval 数据一致的层次：

```text
data/<split>/demo_x.hdf5
└── data/demo_i/
    ├── obs/images
    ├── obs/states
    └── actions
```

- 人手池的 `states/actions` 写成与 repo `REAL_ROBOT_CONSTANTS` 对齐的 20D bimanual EE 表示。
- 20D 格式：左臂 `xyz(3) + rot6d(6) + gripper(1)`，右臂同样 10D。
- 单臂任务将另一臂填 no-op，并在 metadata 中记录 active arm。

## 3. Implementation Changes

### 3.1 数据转换与校验

- 新增 `scripts/handret/convert_to_handret_hdf5.py`，支持 `custom,mime,hoi4d,arctic,taco` 输入，输出 canonical HDF5。
- 新增 `scripts/handret/validate_handret_hdf5.py`，检查帧率、长度、NaN、action/proprio 维度、相机标定和任务标签。
- 新增 `scripts/handret/compute_dataset_stats.py` 和 `compute_delta_stats.py`，生成 `dataset_statistics.json` 与 `delta_dataset_statistics.json`。

### 3.2 人手动作/状态预处理

- wrist pose 统一到 robot base frame；旋转转为 rot6d；gripper 用 thumb-index distance 或手指开合估计。
- 用低通或 Savitzky-Golay 平滑 wrist/action，裁剪到机器人 workspace 和速度/加速度安全范围。
- 对 public 数据无法标定到真实 robot base 的样本，只用于检索特征调试，不进入最终真实机器人结论。

### 3.3 检索模块

- 新增 `cosmos_policy/experiments/robot/real_robot_ret/retrieval.py`，实现两阶段检索。
- Stage 1：按任务语言、初始视觉或物体初始状态筛 demo。
- Stage 2：按 object pose、robot proprio、hand wrist pose、DINOv2/object crop embedding、速度项做 subframe nearest-neighbor。
- 输出与现有 PushT RAG 一致：`retrieved_frames`、`retrieved_actions`、`retrieved_proprio`、match metadata。
- 新增 `scripts/handret/build_retrieval_npz.py`，输出兼容当前 `PushTRetDataset` 风格的 `query_ids/match_ids/match_sims`。

### 3.4 训练/推理代码

- 新增 `RealRobotHandRetDataset`，复用 `PushTRetDataset` 的 residual target、retrieval slot 和 future-state layout。
- 新增 generic retrieval model alias，复用 `policy_video2world_model_pusht_ret.py` 的 mask 与 retrieved action/state 注入逻辑。
- 新增 `real_robot_handret` config：
  - `chunk_size=16`
  - `action_dim=20`
  - `proprio_dim=20`
  - `state_t=12`
  - `min_num_conditional_frames=9`
  - `use_residual_actions=True`
  - `predict_future_states=True`
- 更新 `cosmos_utils.py` 支持 `suite=real_robot_ret`，并按 retrieval layout 设置 future decode replacement indices。
- 新增 `cosmos_policy/experiments/robot/real_robot_ret/run_eval.py` 和 `eval_real_robot_handret.sh`，每个 control chunk 检索一次，预测 normalized delta，反归一化后执行 `retrieved_action + delta`，并加安全限幅。

## 4. Experiment Plan

### 4.1 Sanity check

- 先跑现有 PushT RAG，确认环境、checkpoint、`retrieved_*` 条件链路正常。
- 对照本地笔记：
  - `笔记/RECAP_论文代码对应关系.md`
  - `笔记/RECAP_检索池与检索详解.md`

### 4.2 Main comparison

- `No retrieval`：只用 seen task 训练的冻结策略。
- `Hand playback/retarget only`：直接执行最近人手伪动作，不预测残差。
- `RECAP hand-ret`：人手检索 + residual delta + future-state prediction。
- Optional oracle：对 unseen task 采少量机器人 demo 微调，作为上界。

### 4.3 Pool-growth 验证

- 对每个 unseen task 用 `0,1,3,5,10` 条人手 demo 建池。
- 模型全程冻结。
- 画 success-rate 曲线，验证新增人手池是否带来单调或总体上升趋势。

### 4.4 消融实验

- residual vs absolute action。
- with vs without future-state prediction。
- with vs without retrieved image/state/action。
- geometry-only retrieval vs geometry + visual embedding retrieval。

### 4.5 指标

- 每任务至少 10 次真实 rollout，建议 20 次。
- 报告 success rate、binomial 95% CI、完成时间、安全中断次数、retrieval switch rate、residual norm、失败类别。
- 论文结论成立标准：
  - unseen tasks 上 `RECAP hand-ret` 明显高于 `No retrieval` 和 `Hand playback`。
  - 检索池增长带来总体上升趋势。
  - 加入新任务人手池后不更新模型参数。

## 5. Assumptions

- 默认使用 repo 现有 `REAL_ROBOT_CONSTANTS`：20D bimanual absolute EE pose in rot6d + gripper。
- 默认真实机器人只有单个 primary camera；如有 wrist cameras，可后续扩成 ALOHA/RobotWin 类三相机 retrieval layout。
- 最终论文结论必须基于自采同场景人手池 + 真实机器人 rollout；公开数据结果只能作为工程可行性或离线证据。

