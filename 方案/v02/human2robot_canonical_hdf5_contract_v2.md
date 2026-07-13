# Human2Robot native-time canonical HDF5 契约 v2

版本：`human2robot-canonical-hdf5-v2`  
适用里程碑：M2-v02  
状态：实现契约

## 1. 目的与边界

v2 canonical 层统一 Human2Robot 的表示、数据类型和审计字段，但不统一物理采样率。默认策略 `preserve_native` 必须保留每一帧 source 数据；任何固定步长或目标 Hz 视图都属于后续 derived time view，不得写成 native-time canonical 数据。

`legacy_fixed_stride3_assumed30` 仅用于复现已撤销的 M2-v01 结果。它必须由 CLI 显式指定，不具备 M2-v02 验收资格。

## 2. 文件与时间轴

每个 source episode 对应一个 HDF5 文件，核心结构为：

```text
data/demo_0/
├── obs/
│   ├── images                    uint8   (T,H,W,3)
│   └── states                    float32 (T,10)
├── actions                       float32 (T,10)
└── metadata/
    ├── source_indices            int64   (T,)
    ├── source_step               int64   (T,)
    ├── source_timestamp          int64   (T,)
    ├── segment_id                int32   (T,)
    ├── gap_mask                  bool    (T,)
    ├── qpos_raw                  float32 (T,7)
    ├── qvel_raw                  float32 (T,7)
    ├── end_position_raw          float32 (T,6)
    ├── action_raw                float32 (T,7)
    └── human/
        ├── images                uint8   (T,H,W,3)
        ├── hand_coords           float32 (T,24,3)
        └── hand_frames           float32 (T,4,3)
```

对 `preserve_native`：

- `source_indices` 必须等于 `0..T-1`，且 `canonical_frame_count == source_frame_count == T`。
- `source_step` 和 `source_timestamp` 必须与 source HDF5 逐值一致。
- 禁止生成 `metadata/timestamps` 或其他人工等间隔时间轴。
- `source_fps` 在没有上游证据时必须为空，不能写入估计值或默认值。

## 3. 时间质量与 segment

`gap_mask[0]` 固定为 `false`；`gap_mask[i]` 表示 source frame `i-1` 与 `i` 之间存在边界。pilot 的默认边界规则为：

- `diff(source_step) < 0` 或 `> 1`；
- `diff(source_timestamp) < 0` 或 `> 1`。

`segment_id = cumsum(gap_mask)`。后续 chunk、插值和检索不得跨越不同 `segment_id`。

每条 episode 必须记录 step/timestamp 的重复、jump、rollback、gap 和 segment 数量。`timebase_status` 取值：

- `trusted`：上游时钟和分辨率有可核验依据；本 pilot 不主动授予该等级。
- `coarse`：时间字段单调但分辨率不足以支持逐帧物理 `dt`。
- `discontinuous`：存在 gap、jump 或 rollback，且已正确分段。
- `unknown`：时间字段不能支持采样率或持续时间结论。

在 `trusted` 以外的状态，质量报告只能使用 `per source step` 位移，不得标记为 m/s，也不得把固定秒数 horizon 作为已知事实。

## 4. 状态、动作与证据状态

`obs/states` 与 `actions` 都使用 10D：`xyz(m) + rot6d + gripper`。state 来自 `end_position + gripper_state`，action 暂按 `action[:6] + action[6]` 的 absolute EE target 解释。

source xyz 单位、Euler 角顺序、action 语义和 gripper 极性尚未由上游文档确认，因此 attrs 必须分别标为 `assumed` 或 `unknown`；不得标为 `verified`。`qpos/qvel/end_position/action` 原值保留，便于证据更新后无损重转。

## 5. 可追溯性

每条 canonical episode 必须记录：

- source relative path 与 SHA-256；
- schema version；
- 转换代码 SHA-256；
- 影响数据内容的转换配置 SHA-256；
- frozen task split SHA-256；
- source/canonical frame count；
- 图像源 dtype 与 canonical 转换方式。

写入采用临时文件加原子替换；已有文件若 source、schema、策略、配置或 split 不一致，必须拒绝复用。

## 6. task split 与统计

split 必须按 task 固化。默认 pilot 用 `SHA256("<seed>:<task>")` 排序后选择 4 个 held-out task，其余为 train；episode 不得跨 task split。

以下统计只允许读取 metadata 中 `task_split=train` 且 split hash 一致的 episode：

- `dataset_statistics.json`
- `dataset_statistics_post_norm.json`
- `data_quality_statistics.json`

每份统计必须保存来源 task、episode/frame 数、split hash 和 `heldout_data_used=false`。M2-v02 禁止生成 `delta_dataset_statistics.json`；同帧 action-state offset 若将来需要，只能使用 `control_target_offset_statistics.json`。

## 7. validator

结构验证检查 schema、必需路径、shape、dtype、共享 time axis、finite、rot6d、gripper、workspace、压缩和 RGB 首尾帧解码。

时间真实性验证检查：

- native source index 一一映射；
- canonical/source frame 数一致；
- source step/timestamp 逐值回查；
- source SHA-256；
- 不存在人工 timestamp；
- gap、segment、timebase status 与重算结果一致；
- frozen task split 与统计 provenance 无 held-out 泄漏。

时间真实性验证不要求固定 Hz。

## 8. 可视化

可视化必须同时显示 native human RGB、robot RGB、state/action、source step/timestamp、segment 和 gap 边界。MP4 的 `playback_fps` 只代表编码播放速度，manifest 必须明确它不是 source capture frequency 的证据。

## 9. 参考命令

```bash
python tools/convert_human2robot_m2.py \
  --timebase-policy preserve_native \
  --selection-manifest data/Human2Robot/canonical/v1/preprocessing_manifest.json \
  --output-root data/Human2Robot/canonical/v2 \
  --episodes 20 \
  --heldout-task-count 4 \
  --visualizations 10 \
  --overwrite
```

```bash
python tools/validate_canonical_hdf5.py \
  --input data/Human2Robot/canonical/v2/pilot \
  --source-root /DATA1/wxs/DATASETS/Human2Robot/data/v1 \
  --split-manifest data/Human2Robot/canonical/v2/task_split_manifest.json \
  --minimum-episodes 20 \
  --minimum-tasks 20
```
