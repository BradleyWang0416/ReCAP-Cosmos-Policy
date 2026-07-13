# M5-A-v03 数据与契约压力测试启动报告

日期：2026-07-11T07:32:01.861282+00:00

结论：**M5-A-v03 已启动并完成首轮数据/契约压力测试；M5-B 与 Gate C 保持 pending**

M5-A 只验证数据与契约检测能力，不声明最终 Cosmos/RECAP 模型收益、鲁棒性或可部署性。

## 前置状态

- M3 Gate B：`passed`
- M4：`launched`；Gate C=`pending`
- query command：`unverified`；M6 rollout：`false`

## Action-role / lag 压力测试

| 扰动 | 主指标 | baseline | perturbed | detector |
|---|---|---:|---:|---|
| `wrong_role` | `role_contract_violation_rate` | 0.000000 | 1.000000 | triggered |
| `same_frame_copy` | `temporal_leakage_rate` | 0.000000 | 1.000000 | triggered |
| `wrong_lag` | `paired_position_error_median_canonical` | 0.007382 | 0.062440 | triggered |
| `scale_x2` | `paired_position_error_median_canonical` | 0.007382 | 0.494242 | triggered |

## FPS/version 数据级对比

| time view | samples | gap crossing | residual median | position median |
|---|---:|---:|---:|---:|
| `native_row_index` | 7785 | 0 | 0.009234 | 0.007382 |
| `nominal_camera_30hz_segmented` | 7785 | 0 | 0.009234 | 0.007382 |
| `paper_v2_stride4_nominal7p5` | 1940 | 0 | 0.017461 | 0.013990 |
| `legacy_v01_stride3_nominal10` | 2590 | 0 | 0.014618 | 0.011762 |
| `policy_clock_10hz` | 2590 | 0 | 0.014618 | 0.011762 |
| `phase_or_dtw` | 1008 | 0 | 0.026997 | 0.022555 |

## Temporal mismatch 注入

| 扰动 | pairs | detector | triggered | gap crossing | position median |
|---|---:|---|---|---:|---:|
| `frame_drop` | 7013 | `source_row_jump_count` | true | 0 | 0.007586 |
| `timestamp_jitter` | 7785 | `timestamp_jitter_exceed_count` | true | 0 | 0.007382 |
| `pause` | 7785 | `pause_count` | true | 0 | 0.007382 |
| `step_jump` | 7785 | `logical_step_jump_count` | true | 0 | 0.007382 |

## 分辨率契约

- canonical streams：`{'human/images': [[240, 426, 3]], 'robot_images': [[240, 426, 3]]}`
- paper 对比主策略：`center_crop_width_426_to_424_v1`，左右各裁 1 列，不 resize。
- canonical 兼容策略：`edge_pad_width_424_to_426_v1`，左右各复制 1 列，不 resize。
- action contract hash unchanged：`true`
- phase retrieval image-independent：`true`
- visual retrieval crop/pad robustness：`NEEDS_EXPERIMENT`（M5-B）。

## 启动边界与下一步

- M5-A 已执行并产出协议、四类数据/契约检查和自动报告。
- M5-B 仍为 pending：需正式 M4 多 seed checkpoint 后执行模型依赖型消融。
- Gate C 仍为 pending；不得据此批准 M6 或真实机器人 command。

## 产物

- `experiment_protocol`：`data/Human2Robot/derived/m5a_v03/experiment_protocol.json`
- `action_role_stress`：`data/Human2Robot/derived/m5a_v03/action_role_stress.json`
- `time_view_matrix`：`data/Human2Robot/derived/m5a_v03/time_view_matrix.json`
- `temporal_stress`：`data/Human2Robot/derived/m5a_v03/temporal_stress.json`
- `resolution_stress`：`data/Human2Robot/derived/m5a_v03/resolution_stress.json`
- `automatic_report`：`data/Human2Robot/derived/m5a_v03/m5a_launch_report.json`
