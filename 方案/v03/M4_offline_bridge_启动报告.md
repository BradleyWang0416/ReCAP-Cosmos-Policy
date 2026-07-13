# M4-v03 离线 paired bridge 启动报告

日期：2026-07-11T07:09:22.639855+00:00

结论：**M4-v03 离线 bridge 已启动；Gate C 仍为 pending。**

本报告只确认 action-space 离线闭环、四个固定 baseline、pool-growth smoke 与 checkpoint 契约已经可运行；不批准 M6 或真实机器人 command。

## 冻结契约

- time view：`nominal_camera_30hz_segmented`
- pool action：`human_hand_robot_frame_raw`
- query action：`robot_ee_observed_t_plus_1_bc_proxy`
- alignment：`train_only_tplus1_query_anchor_se3_identity_scale_v1`
- H/K：8/8
- deployment command adapter：`null`

## 数据与泄漏门禁

- train windows：968
- held-out windows：153
- gap crossing：0
- retrieval feature：normalized segment phase + task pool membership
- held-out robot trajectory：只用于离线 target 评测

## 全池 smoke 指标

| 方法 | position median | orientation median rad | gripper median |
|---|---:|---:|---:|
| `no_retrieval` | 0.006334 | 0.000609 | 0.020790 |
| `retrieval_only` | 0.005114 | 0.000001 | 0.000000 |
| `co_training` | 0.006075 | 0.000503 | 0.005719 |
| `recap_hand_ret` | 0.006076 | 0.000504 | 0.005719 |

## 当前边界

- 当前模型是确定性 ridge smoke bridge，用于打通数据、baseline、评测和 checkpoint 契约，不是最终 Cosmos/RECAP 主训练配置。
- pilot 每个 held-out task 只有一个 paired episode；human-only pool 与 robot target 来自同一发布 pair，但 retrieval 代码不读取 held-out robot target。正式 Gate C 需要独立/扩充 human pool 与多 seed 训练。
- pool-growth 是否总体改善、RECAP 是否稳定优于 No retrieval/Retrieval Only，必须在正式训练后验收；本报告不提前通过 Gate C。
- `query_command_status=unverified`，不得用于真实机器人执行。
