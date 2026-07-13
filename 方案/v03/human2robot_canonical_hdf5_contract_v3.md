# Human2Robot semantic-safe native canonical HDF5 契约 v3

版本：`human2robot-canonical-hdf5-v3`  
里程碑：M2-v03  
状态：实现契约

## 1. 核心约束

Canonical v3 保存 source 事实和已验证的字段角色，不选择 policy action。每条 source row 必须一一保留；固定 Hz、抽帧、插值、lag 与 `policy_actions` 只能由后续 derived view 产生。正式 v3 reader 必须拒绝 v1/v2。

30 Hz 只表示论文 v4 中 D435 的 nominal capture configuration。它不得生成全局 `0,1/30,...` 时间轴，也不得把 coarse/discontinuous record 标成 trusted。

## 2. HDF5 schema

```text
data/demo_0/
├── obs/
│   ├── robot_images                         uint8   (T,H,W,3)
│   └── robot_state_10d                      float32 (T,10) [in-file alias]
├── trajectories/
│   ├── robot_ee_observed_10d                float32 (T,10)
│   └── human_hand_robot_frame_10d           float32 (T,10)
└── metadata/
    ├── source_indices/source_step/source_timestamp
    ├── segment_id/gap_mask
    ├── qpos_raw/qvel_raw/end_position_raw/action_raw/gripper_state_raw
    └── human/images/hand_coords/hand_frames
```

`robot_state_10d` 与 `robot_ee_observed_10d` 是同一 HDF5 object 的内部 hard link。禁止 `actions`、`policy_actions` 或跨 v2/v3 hard link。

`robot_ee_observed_10d = /end_position + /gripper_state`；其角色为 `observed_robot_ee_pose`。`human_hand_robot_frame_10d = /action[:6] + /action[6]`；其已验证角色为 `human_hand_pose_in_robot_frame`，是否等于机器人实际 command 仍为 `unknown`。两者均表示为 `xyz(m, conversion assumption) + rot6d + gripper`。

## 3. evidence metadata

每条 episode 必须绑定 `source_evidence_manifest_sha256`，且 manifest 的每个上游证据含 URL、版本和访问日期。必需结论：

- `nominal_camera_fps=30.0`、`nominal_camera_fps_status=verified_upstream`、来源为 arXiv v4 Appendix A；
- `euler_unit=degree`、`euler_order=XYZ`、`euler_evidence_status=verified_upstream`；
- `gripper_open_value=1`、`gripper_closed_value=0`、状态 `verified_upstream`；
- source `/action` 角色为 `human_hand_pose_in_robot_frame/verified_upstream`；
- `/action` 作为 robot command 的状态为 `unknown`；
- xyz source unit 状态为 `unknown`。

## 4. 时间真实性

`source_indices == 0..T-1`，source step/timestamp 与 source HDF5 逐值一致。`gap_mask[0]=false`；后续位置的边界规则为 step 或 timestamp diff `<0` 或 `>1`。`segment_id=cumsum(gap_mask)`。

`record_timebase_status` 取 `coarse|discontinuous|unknown`（validator 兼容集合中保留 `trusted`，但本 pilot 不授予）。`record_timebase_globally_trusted=false`。禁止 `metadata/timestamps`。

## 5. split 与统计

v3 使用与 v2 相同的 20 条 episode 和 task assignment；新 split manifest 必须记录 `parent_v2_split_sha256` 和 v3 自身 hash。仅 train split 可生成：

- `robot_observed_statistics.json`
- `human_hand_robot_frame_statistics.json`
- `dataset_statistics_post_norm_by_role.json`
- `data_quality_statistics.json`

禁止 generic `dataset_statistics.json`、`dataset_statistics_post_norm.json` 和 residual/delta statistics。所有统计 provenance 必须记录 role、split hash、evidence hash、`heldout_data_used=false` 与 `generic_action_used=false`。

## 6. validator 与验收

validator 分三类报告：structure、time truth、evidence/role。它检查 schema/shape/dtype/shared-T/finite/rot6d/gripper/workspace/RGB，source row/hash/step/timestamp/gap/segment/no synthetic timeline，以及 FPS provenance、Euler/gripper provenance、source-to-role 映射和 generic action 禁止项。

M2-v03 pilot 需达到 20 episode、20 task、source row 100% 保留、20/20 source SHA-256 回查、10 条 role-separated 可视化，并保存自动与人工验收报告。

## 7. 参考命令

```bash
python tools/convert_human2robot_m2.py \
  --output-root data/Human2Robot/canonical/v3 \
  --report-root 方案/v03 \
  --episodes 20 --heldout-task-count 4 --visualizations 10 --overwrite

python tools/validate_canonical_hdf5.py \
  --input data/Human2Robot/canonical/v3/pilot \
  --source-root /DATA1/wxs/DATASETS/Human2Robot/data/v1 \
  --split-manifest data/Human2Robot/canonical/v3/task_split_manifest.json \
  --minimum-episodes 20 --minimum-tasks 20
```
