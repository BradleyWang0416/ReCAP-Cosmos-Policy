# M2 Human2Robot semantic-safe native canonical HDF5 验收报告

验收日期：2026-07-11（Asia/Shanghai）  
数据源：Human2Robot v1  
schema：`human2robot-canonical-hdf5-v3`  
结论：**通过 Gate A1**

## 1. 结论与边界

M2-v03 pilot 已通过：20 条 episode 覆盖 20 个不同任务，9,039 个 source row 全部一一保留；structure、time truth、evidence/role 三类 validator 均为 20/20 通过，独立 validator 复验结果相同。

Canonical v3 不含 generic `actions`/`policy_actions`。source `/end_position + gripper_state` 只写为 `robot_ee_observed_10d`，source `/action` 只写为 `human_hand_robot_frame_10d`。`obs/robot_state_10d` 是 observed trajectory 的同文件 hard-link alias，没有复制数值，也没有与 v2/v3 跨文件共享 inode。

30 Hz 已按论文 v4 记录为 `nominal_camera_fps=30.0/verified_upstream`；所有 episode 同时保持 `record_timebase_globally_trusted=false`，没有生成 `0,1/30,...` 或其他全局人工时间轴。

本报告只通过 M2-v03 Gate A1。source xyz 单位和 `/action` 是否为实际下发 robot command 仍为 unknown；因此尚未定义 residual，也不代表 M3 Stage 0、M4 或真实机器人 gate 已通过。

## 2. 实现与交付物

| 交付物 | 路径 | 状态 |
|---|---|---|
| semantic-safe converter、统计、可视化、三类 validator | `tools/human2robot_m2.py` | 已实现并实跑 |
| 默认 canonical/v3 CLI | `tools/convert_human2robot_m2.py` | 已实现 |
| standalone validator | `tools/validate_canonical_hdf5.py` | 20/20 独立复验通过 |
| v3/FPS/role/legacy-reject 回归测试 | `tools/human2robot_m2_test.py` | 11/11 通过 |
| v3 数据契约 | `方案/v03/human2robot_canonical_hdf5_contract_v3.md` | 已保存 |
| source evidence manifest | `方案/v03/source_evidence_manifest_v3.json` | URL、版本、访问日期完整 |
| v02 code/config/report 冻结清单 | `方案/v03/m2_v02_frozen_code_manifest.json` | 已保存 |
| v02 superseded marker | `data/Human2Robot/canonical/v2/SUPERSEDED_M2_V02.json` | 已保存 |
| v02 报告语义重开说明 | `方案/v02/M2_Human2Robot_native_time_验收报告.md` | 已添加，不改写历史数值 |
| canonical v3 pilot | `data/Human2Robot/canonical/v3/pilot/demo_*.hdf5` | 20 条，约 1.9 GiB |
| role-separated train-only 统计 | `data/Human2Robot/canonical/v3/*.json` | 4 份，provenance 通过 |
| role-separated 可视化 | `data/Human2Robot/canonical/v3/visualizations/sample_*.mp4` | 10 条，人工抽查通过 |
| 自动/独立报告 | `data/Human2Robot/canonical/v3/m2_validation_report.json`、`m2_independent_validation_report.json` | 均通过 |
| 自动验收报告 | `方案/v03/M2_Human2Robot_semantic_safe_自动验收报告.json` | 已保存 |

## 3. 验收矩阵

| M2-v03 标准 | 实测 | 结果 |
|---|---:|---|
| 20 条 episode、20 个任务 | 20 / 20 | 通过 |
| source row 100% 保留 | canonical/source = 9,039 / 9,039 | 通过 |
| 所有 stream 第一维共享 T | 20/20 | 通过 |
| 必要数值 finite、rot6d、gripper/workspace 合法 | 20/20 | 通过 |
| source step/timestamp/SHA-256 回查 | 20/20 | 通过 |
| record continuity 独立于 nominal FPS | 17 coarse、3 discontinuous、0 trusted | 通过 |
| gap/segment | 3 gap / 23 segment | 通过 |
| nominal 30 Hz 有 v4 provenance | 20/20 | 通过 |
| 无全局人工 timeline | 20/20 不含 `metadata/timestamps` | 通过 |
| `/action` 与 `/end_position` role 分离 | 20/20 source-to-role 回查 | 通过 |
| generic `actions`/`policy_actions` 不存在 | 20/20 | 通过 |
| split 继承 v2 assignment 并生成新 hash | parent 文件 hash 已绑定 | 通过 |
| 统计仅使用 train 且按 role 分开 | 16 task / 7,803 rows | 通过 |
| held-out 与 generic action 未进入统计 | 两者均为 false | 通过 |
| 10 条 role-separated 可视化 | 10/10 人工通过 | 通过 |
| v1/v2 正式 reader hard reject | 单元测试通过 | 通过 |

## 4. schema 与证据验收

每条 v3 episode 写入：

```text
obs/robot_images
obs/robot_state_10d                       # alias
trajectories/robot_ee_observed_10d        # /end_position + /gripper_state
trajectories/human_hand_robot_frame_10d   # /action
metadata/source_*/segment_id/gap_mask/raw/human
```

validator 对每条 source 重新计算两条 10D trajectory，并分别与 canonical 字段逐值比较；交换字段的负向测试会失败。`source_evidence_manifest_sha256=bc7e0af7...f98ff8` 已进入每条 HDF5 和全部统计 provenance。

证据状态为：Euler degree XYZ、gripper 1=open/0=close、source `/action` 为 robot-frame human hand pose，均 `verified_upstream`；xyz source unit 与 `/action`-as-command 为 `unknown`。

## 5. 时间、split 与统计

v3 split hash 为 `1d3ef2377aa19938b06646f6d5fc31ec9f275fc9f37e253e1e9aa5eecdc5a968`。其 parent v2 split 文件 SHA-256 为 `d16b0da6267b2bc1c5a7234d14cdbefa9b8ba49fd6e8495c1b83f154cf367871`，20 个 task assignment 与 v2 完全相同：16 train、4 held-out。

train-only 统计来自 7,803 rows。连续 segment 内 per-source-step 位移仅作质量诊断，不解释为 m/s：human pose max/p99 为 0.19998/0.06714 m，robot observed max/p99 为 0.12853/0.02021 m。

以下 generic/residual 文件在 v3 中均不存在：`dataset_statistics.json`、`dataset_statistics_post_norm.json`、`delta_dataset_statistics.json`。

## 6. 人工可视化验收

seed `20260711` 随机抽取 10 条。每条检查 frame 0/50/100/150 联系表，确认 human/robot RGB 对应、任务进程可见、source step/timestamp/segment 可读，并且 overlay 使用 `ROBOT OBS` 与 `HUMAN POSE`，不再出现 generic action 标签。

ffprobe 复验 10/10 为 852×312、10 fps 编码，视频帧数均等于 canonical T。10 fps 只代表播放速度。逐样本结果保存在 `data/Human2Robot/canonical/v3/visualizations/manual_review/manual_review_manifest.json`。

## 7. 自动验证与复验

```text
python -m pytest -q tools/human2robot_m2_test.py
-> 11 passed

.venv/bin/ruff check tools/human2robot_m2.py \
  tools/convert_human2robot_m2.py \
  tools/validate_canonical_hdf5.py \
  tools/human2robot_m2_test.py
-> All checks passed!

python -m py_compile tools/human2robot_m2.py \
  tools/convert_human2robot_m2.py \
  tools/validate_canonical_hdf5.py \
  tools/human2robot_m2_test.py
-> passed

python tools/validate_canonical_hdf5.py ... --minimum-episodes 20 --minimum-tasks 20
-> 20/20 passed; structure/time/evidence-role all passed; split passed
```

## 8. 关键 SHA-256

```text
52988de59862ffab51882e9734ae30c3b56d092c64f2b47dcd20bb1607f3fdc7  preprocessing_manifest.json
21e241cef98ec1a470605d8e54b9560aff5aad7d571372e74d92b9d4939e876d  task_split_manifest.json
d10c464602e2bc692d591ba408920420b86ca3dbf5cd9784610dec2db332666e  timebase_audit_report.json
cd352e1d3db569f4d8f55d1681915bd37bb6c6ad0bb5d96e0f9a3b5a4213ba40  robot_observed_statistics.json
122667dafb269e6bcdd498c745564718ba39b64b364f80480d2f48e48f0001dd  human_hand_robot_frame_statistics.json
8caab15ef05a3c432d738d2000059ef420e2b4bbead3e1038cf8660c325cc46c  dataset_statistics_post_norm_by_role.json
521b128b07b1536e4f1e9f93c0c8ebb034744bb1dceee3f901da34b4acd08f38  data_quality_statistics.json
fdeed7b5055608cb9413d416dc15ddcf401a5f2b0d7a5708d88f923c862440a0  m2_validation_report.json
33889daca5018e7fe71027e9815f887a38e5b7b5ed4d983562cc3e43ad5d80a7  m2_independent_validation_report.json
11731dbdd01264ae45bea3c1f7ca9fd3d4506ec3cae0da44e5167f7b3a9fd5c3  visualization_manifest.json
cfe488e62dba057d94274e46a0160fe852227f4e194043a79a43e699076be0cc  manual_review_manifest.json
bc7e0af72e310df0dba6c51f19d774644365f6501f86ece84ca0447fa4f98ff8  source_evidence_manifest_v3.json
```

## 9. 后续门禁

M2-v03 Gate A1 已通过，可以开始 M3 Stage 0 的 action-role calibration。M3 必须先解决或明确代理化 `/action` 的 command 角色、lag、xyz 单位与尺度；在 Gate A2 通过前不得构造正式 residual，也不得启动 M4 主训练。
