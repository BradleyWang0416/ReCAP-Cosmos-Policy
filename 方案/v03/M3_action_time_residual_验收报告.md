# M3-v03 action-role、time-view 与 residual sanity check 验收报告

日期：2026-07-11T06:58:19.458870+00:00

结论：**通过 M3-v03；允许启动 M4-v03 离线 bridge**。Gate A2a=passed，Gate A2b=passed，Gate B=passed。

## 验收摘要

- Canonical 输入：`human2robot-canonical-hdf5-v3`，20 条 pilot；旧 v1/v2 未进入正式 M3。
- Pool role：`human_hand_robot_frame_raw`（人手 plan）；query role：`robot_ee_observed_t_plus_1_bc_proxy`（严格 future BC proxy）。
- 选择的 future offset：1 个 view step；motion cross-correlation 最优 lag=5，相关系数=0.1133，因此 lag-calibrated proxy 仅保留诊断。
- 主 view residual norm 中位数=0.009234，absolute target norm 中位数=1.768861。
- legacy stride3 residual norm 中位数=0.014618；主 view 更低。
- Held-out query=153，top-10 覆盖=153/153，gap crossing=0。
- 检索 phase error 中位数=0.002429，random=0.275350。
- 检索 residual norm 中位数=0.025296，random=0.078882。
- Retrieval feature 只有 normalized segment phase；held-out robot trajectory 仅用于离线 target 评测。
- `deployment_command_adapter_id=null`；本报告不批准真实机器人 command 或 M6 rollout。

## Action-role 与 Gate A2a

`/action` 对应 pool-side human plan；`/end_position + /gripper_state` 仅作为 observed robot trajectory 与数据集卡允许的 BC label 来源。canonical v3 中不存在 generic action。

## Gate A2b：proxy、alignment 与扰动

主 query 使用下一连续 view row，所有不完整末帧窗口被丢弃，不跨 segment。identity numeric scale 来自同一 canonical 10D 转换；xyz 物理单位仍未确认，未报告 m/s。

- `wrong_role`：metric=role_contract_violation_rate，baseline=0.000000，perturbed=1.000000，significant_worsening=True。
- `same_frame_copy`：metric=temporal_leakage_rate，baseline=0.000000，perturbed=1.000000，significant_worsening=True。
- `wrong_lag`：metric=paired_position_error_median_canonical，baseline=0.007382，perturbed=0.062440，ratio=8.458，significant_worsening=True。
- `scale_x2`：metric=paired_position_error_median_canonical，baseline=0.007382，perturbed=0.494242，ratio=66.949，significant_worsening=True。

## Time-view 对比

| time_view_id | samples | gap crossing | residual median | absolute median |
|---|---:|---:|---:|---:|
| `native_row_index` | 7785 | 0 | 0.009234 | 1.768861 |
| `nominal_camera_30hz_segmented` | 7785 | 0 | 0.009234 | 1.768861 |
| `paper_v2_stride4_nominal7p5` | 1940 | 0 | 0.017461 | 1.768851 |
| `legacy_v01_stride3_nominal10` | 2590 | 0 | 0.014618 | 1.768810 |
| `policy_clock_10hz` | 2590 | 0 | 0.014618 | 1.768810 |
| `phase_or_dtw` | 1008 | 0 | 0.026997 | 1.769564 |

## 产物与下一门禁

正式 view：`data/Human2Robot/derived/views/nominal_camera_30hz_segmented/human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy/train_only_tplus1_query_anchor_se3_identity_scale_v1`。检索索引、view manifest、action statistics 和自动验收 JSON 均已保存。

Gate B 仅允许启动 M4-v03 离线 bridge。真实控制仍需 M6 deployment command adapter、clock、latency 和安全验收。
