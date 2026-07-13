# M2 Human2Robot 统一预处理与 canonical HDF5 验收报告

> **验收已撤销（2026-07-11）**：本报告依赖未经证实的 30 Hz source 假设和人工生成的 10 Hz 时间轴，只能用于 legacy 回归、dtype/schema 测试、可视化和时间消融，不得用于 v02 retrieval、训练统计、主实验或论文结论。替代方案见 [`方案/v02/RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md`](../v02/RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md)；新验收见 [`方案/v02/M2_Human2Robot_native_time_验收报告.md`](../v02/M2_Human2Robot_native_time_验收报告.md)。

验收日期：2026-07-10（Asia/Shanghai）  
数据源：Human2Robot v1  
schema：`human2robot-canonical-hdf5-v1`  
结论：**通过**

## 1. 验收范围

本次按已确认的数据源决策，只验收 Human2Robot，不执行原方案中的 MIME/RH20T 转换。输入根目录为：

```text
/DATA1/wxs/DATASETS/Human2Robot/data/v1/
```

输出根目录为：

```text
data/Human2Robot/canonical/v1/
```

pilot 使用 deterministic task-round-robin 选择 20 条 episode，覆盖 20 个不同任务；没有候选因 schema 或数值检查被拒绝。输出 20 个 canonical HDF5，共 3,021 帧，HDF5 合计 662,756,974 bytes。

## 2. 交付物

| 交付物 | 路径 | 状态 |
|---|---|---|
| 全流水线 CLI | `tools/convert_human2robot_m2.py` | 已实现并实跑 |
| 转换/统计/可视化核心 | `tools/human2robot_m2.py` | 已实现 |
| 独立 HDF5 validator | `tools/validate_canonical_hdf5.py` | 已实现并独立复验 |
| 单元测试 | `tools/human2robot_m2_test.py` | 4/4 通过 |
| canonical 数据契约 | `方案/v01/human2robot_canonical_hdf5_contract.md` | 已保存 |
| 预处理 manifest | `data/Human2Robot/canonical/v1/preprocessing_manifest.json` | 已生成 |
| canonical episodes | `data/Human2Robot/canonical/v1/pilot/demo_*.hdf5` | 20 条 |
| raw statistics | `data/Human2Robot/canonical/v1/dataset_statistics.json` | 已生成 |
| post-norm statistics | `data/Human2Robot/canonical/v1/dataset_statistics_post_norm.json` | 已生成 |
| delta statistics | `data/Human2Robot/canonical/v1/delta_dataset_statistics.json` | 已生成 |
| 自动验收明细 | `data/Human2Robot/canonical/v1/m2_validation_report.json` | 20/20 通过 |
| 可视化 manifest | `data/Human2Robot/canonical/v1/visualizations/visualization_manifest.json` | 已生成 |
| human/robot/action MP4 | `data/Human2Robot/canonical/v1/visualizations/sample_*.mp4` | 10 条 |

## 3. canonical 映射结果

每条输出使用以下核心结构：

```text
data/demo_0/
├── obs/images                 uint8   (T,240,426,3)
├── obs/states                 float32 (T,10)
├── actions                    float32 (T,10)
└── metadata/
    ├── timestamps             float64 (T,)
    ├── source_indices/step/timestamp
    ├── qpos_raw/qvel_raw/end_position_raw/action_raw
    └── human/images/hand_coords/hand_frames
```

10D 定义为 `xyz(m) + rot6d + gripper`。state 使用源 `end_position + gripper_state`，action 使用源 `action[:6] + action[6]` 并按绝对 EE target 解释。源低维字段完整保留，便于上游语义确认后无损重转。

40 路 paired 相机流中，8 路源 dtype 是 `uint8`；32 路是数值仍位于 0..255 的 `uint16` 容器，已按探测结果无损 cast 成 canonical `uint8`。所有流均使用 gzip 压缩，转换方式逐路记录在 metadata attrs。

## 4. 验收结果

| M2 标准 | 实测 | 结果 |
|---|---:|---|
| 至少 20 条 episode | 20 条，覆盖 20 个任务 | 通过 |
| HDF5 可逐条读取 | 独立 validator 打开 20/20，并实际解码每条首尾 paired RGB chunk | 通过 |
| 无 NaN/Inf | actions、states、qpos/qvel、raw pose/action、human hand 全量 finite | 通过 |
| 时间戳单调、帧率统一 | canonical timestamp 严格递增，相邻差为 0.1 s，即 10 Hz | 通过 |
| action/proprio 维度一致 | 两者均为 10D | 通过 |
| gripper 范围 | state/action 均在 `[0,1]` | 通过 |
| workspace 校验 | 全部位于默认硬范围；实测范围见下表 | 通过 |
| 速度范围校验 | 线速度低于 2.5 m/s、角速度低于 12 rad/s | 通过 |
| 显式统计产物 | raw、post-norm、delta 三份 JSON 均已生成 | 通过 |
| 随机 10 条可视化 | seed `20260710`；10 个 MP4 均经 ffprobe 验证为 10 fps | 通过 |

全局数值范围与速度：

| 流 | xyz 最小值 (m) | xyz 最大值 (m) | 最大线速度 (m/s) | 最大角速度 (rad/s) |
|---|---|---|---:|---:|
| `obs/states` | `[0.00444, 0.19774, -0.000001]` | `[0.37420, 0.62292, 0.16995]` | 1.3286 | 0.8902 |
| `actions` | `[0.00444, 0.19772, 0.00000]` | `[0.37482, 0.62292, 0.16995]` | 2.4476 | 1.3702 |

post-normalization 的 actions/proprio 每个非恒定维度均覆盖 `[-1,1]`。`delta_dataset_statistics.json` 的 delta 定义为同一时刻 `canonical action - canonical state`，不是 M3 才能构造的 retrieval residual。

## 5. 执行与复验命令

正式转换：

```bash
python tools/convert_human2robot_m2.py \
  --output-root data/Human2Robot/canonical/v1 \
  --episodes 20 \
  --visualizations 10 \
  --overwrite
```

独立复验：

```bash
python tools/validate_canonical_hdf5.py \
  --input data/Human2Robot/canonical/v1/pilot \
  --minimum-episodes 20 \
  --output data/Human2Robot/canonical/v1/m2_validation_report.json
```

代码质量检查：

```text
python -m pytest -q tools/human2robot_m2_test.py     -> 4 passed
ruff format --check ...                             -> passed
ruff check ...                                      -> passed
python -m py_compile ...                            -> passed
```

关键 JSON 的 SHA-256：

```text
516c01a98be680e21a34582727b622c5d25aaa3cbf839bf88b9a76b66f6475f5  dataset_statistics.json
e578cf6c3230825d996f9d152e5a98496059bb3ded55538c4ab89806d83c8a22  dataset_statistics_post_norm.json
73aee94fefb972205ca99a3d8c368bc825f5954d76d51f3ca18bc63bb5f7b536  delta_dataset_statistics.json
ab3f90d788681057db978fe2208f7bb3887ce66c420769392b1d168f57ca513d  preprocessing_manifest.json
fd4161f0ec10a4ce14fe96631f143ce3c5d8ecd4e73409d90fed57e71d4e48b2  m2_validation_report.json
```

## 6. 已知假设与验收边界

- 源 `timestamp` 只有整秒精度，且部分 episode 有 wall-clock 间隔；不能直接用于稳定重采样。本次按 30 Hz source frame sequence 下采样到严格 10 Hz，并保留原始 index/step/timestamp 供审计。
- 根据 M1 数值范围，源 xyz 按毫米、Euler 按 degree 和 `xyz` 顺序解释。由于本地没有上游 README，这三项仍需官方采集/预处理代码最终确认。
- gripper 数值范围已验证为 `[0,1]`，但 open/close 极性尚未由上游文档确认。
- M1 已记录本地缺少 Human2Robot README/LICENSE；本验收只证明本地工程与数据质量链路通过，不解决引用和再分发许可。
- 本次 M2 不构造 query-to-pool retrieval residual，也不声称已完成 M3 检索或 M4 bridge 训练。

在这些边界内，M2 的代码、canonical 数据、显式统计、validator 和 10 条可视化交付均已完成，可以进入 M3 检索索引与离线 sanity check。
