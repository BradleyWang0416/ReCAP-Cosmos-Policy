# Human2Robot canonical HDF5 数据契约（M2）

版本：`human2robot-canonical-hdf5-v1`

本契约将 Human2Robot v1 的 paired human/robot episode 转成 RECAP/Cosmos Policy pilot 可直接逐条读取的格式。M2 固定使用单臂 10D 表示，并把不能由本地数据说明文件确认的语义作为显式假设记录，而不是静默猜测。

## 目录与 HDF5 树

```text
data/Human2Robot/canonical/v1/
├── pilot/demo_00000.hdf5
├── ...
├── preprocessing_manifest.json
├── dataset_statistics.json
├── dataset_statistics_post_norm.json
├── delta_dataset_statistics.json
├── m2_validation_report.json
└── visualizations/
    ├── sample_*.mp4
    └── visualization_manifest.json
```

每个 episode 文件的内部结构：

```text
data/demo_0/
├── obs/
│   ├── images                    uint8   (T,H,W,3)  robot RGB
│   └── states                    float32 (T,10)     xyz(m)+rot6d+gripper
├── actions                       float32 (T,10)     absolute EE target+gripper command
└── metadata/
    ├── timestamps                float64 (T,)       canonical 相对时间（秒）
    ├── source_indices            int64   (T,)
    ├── source_step               int64   (T,)
    ├── source_timestamp          int64   (T,)       原始整秒时间戳
    ├── qpos_raw                  float32 (T,7)
    ├── qvel_raw                  float32 (T,7)
    ├── end_position_raw          float32 (T,6)
    ├── action_raw                float32 (T,7)
    └── human/
        ├── images                uint8   (T,H,W,3)  paired human RGB
        ├── hand_coords           float32 (T,24,3)
        └── hand_frames           float32 (T,4,3)
```

来源、任务、split、帧率、单位、欧拉角顺序、gripper/action 语义和转换时间保存在 `metadata.attrs`。

## 10D 表示

- 位置：源前三维按毫米解释，乘 `0.001` 转为米。
- 姿态：源后三维按 degree、`xyz` Euler 顺序解释，先转 rotation matrix，再按“第一列 3D + 第二列 3D”保存为 rot6d。
- state gripper：来自 `gripper_state`。
- action gripper：来自源 `action[:, 6]`。
- action 是绝对 EE target，不是时间差分或 retrieval residual。

源数据本地没有 README/LICENSE，因此毫米、degree、Euler 顺序、30 Hz 和 gripper 极性仍是待上游材料确认的假设。转换器完整保留原始低维字段，后续确认语义后可以无损重转。

源 RGB 在不同任务中同时存在 `uint8` 和“0..255 数值装在 `uint16` 容器中”两种 dtype。转换器对后者先用均匀抽样帧判定量程，再无损 cast 到 canonical `uint8`；如果探测为真正 16-bit 全量程 RGB，则统一除以 257。每路相机的源 dtype 与实际转换方式保存在 metadata attrs，若同一 episode 内量程不一致则拒绝转换。

## 时间轴

源 `timestamp` 只有整秒精度，部分 episode 还包含较长 wall-clock 间隔，不能用它构造稳定控制时间轴。M2 使用已知/假设的 30 Hz frame sequence，通过最近 frame-index 下采样到严格 10 Hz；禁止上采样。canonical `timestamps` 必须严格递增且相邻差恒为 0.1 秒，源 `step/timestamp/index` 同时保留供审计。

## 统计定义

- `dataset_statistics.json`：canonical raw actions 与 states 的 min/max/mean/std/median；states 使用 Cosmos Policy 兼容键名 `proprio_*`。
- `dataset_statistics_post_norm.json`：逐维 min-max 到 `[-1,1]` 后的同组统计；常量维映射到 0。
- `delta_dataset_statistics.json`：同一时刻 `canonical action - canonical state` 的统计。这是 M2 控制目标差，不冒充 M3 才能定义的 query-to-retrieval residual。

## validator 硬检查

- schema、必需字段、dtype、shape 和所有 time axis 一致。
- state/action 均为 finite 10D；rot6d 两列单位正交。
- canonical timestamp 严格单调且严格 10 Hz。
- source index 严格单调，source step/timestamp 单调非递减。
- state/action gripper 位于 `[0,1]`。
- state/action xyz 位于配置 workspace，默认 `[-1,-1,-0.25]` 到 `[1,1,1]` 米。
- 默认线速度上限 2.5 m/s、角速度上限 12 rad/s；报告同时保存观测最大值。
- paired human/robot RGB shape 一致且使用压缩存储。
