# tools/pilot — v05 探索期只读诊断脚本

本目录是 **探索期（pilot）** 脚本，不是 v04 正式管线的一部分：

- **不**纳入 `tools/human2robot_v04_experiment.py` 的 `_controlled_bindings()` 哈希绑定；
- **不**签发 receipt，**不**产生正式产物，**不**解除任何 preflight blocker；
- 全部只读：只读取原始 HDF5、v04 已冻结的投影/manifest、已冻结的 WAN 特征缓存；
- 中间结果一律写 `/tmp`，不写 `/DATA1/wxs/ReCAP_M5B_V04_RUNS` 或 `data/Human2Robot/derived`。

**合规性披露**：这些脚本在宿主机 `/home/wxs/anaconda3/bin/python` 下运行，**不满足总计划 §2.2 的 Docker-only 约束**。它们属 `NONFORMAL_DIAGNOSTIC`，结论仅用于探索期决策；若需正式化，须在冻结镜像内重做并签发 receipt。

依赖：`h5py`、`numpy`、`scipy`。不需要 GPU、不需要 torch。

结论与数字记录在 `方案/v05_pilot/PILOT_LOG.md`。

## 执行顺序

脚本之间通过 `/tmp` 下的中间缓存串联，必须按序执行：

```bash
bash tools/pilot/run_all.sh
```

或手工逐个执行：

| 顺序 | 脚本 | 作用 | 产出 |
|---:|---|---|---|
| 1 | `d01_probe_hand_fields.py` | 验证 `transformed_hand_frames` 是否合法 SE(3)；`hand_coords` 拓扑 | 仅打印 |
| 2 | `d02_action_is_robot_command.py` | **P0-1 主证据**：`action` 是机器人指令而非人手 | 仅打印 |
| 3 | `d03_gripper_leakage.py` | **P0-2 证据**：co_training 同 episode 夹爪泄漏量化 | 仅打印 |
| 4 | `d04_calibration_global.py` | 人手→机器人全局相似变换；识别指尖点对 | `/tmp/h2r_calib.npy` |
| 5 | `d05_calibration_points.py` | 对应点定义搜索 + 时间 lag 扫描 | 仅打印 |
| 6 | `d06_pairing_control.py` | 正确配对 vs 错配对照，判定人机流是否同步 | 仅打印 |
| 7 | `d07_delta_calibration.py` | K=8 位移的全局旋转+尺度标定（`align_pool_chunk` 只用增量） | `/tmp/h2r_delta_calib.npy` |
| 8 | `d08_delta_magnitudes.py` | 各通道增量幅度与相关性 | 仅打印 |
| 9 | `d09_variant_ab_retrieval.py` | 在冻结的 160 个 dev query 上构建 A/B 两版检索 | `/tmp/h2r_cmp.npy` |
| 10 | `d10_residual_analysis.py` | **P0-4 主证据**：排序一致性 + 对齐后残差 | 仅打印 |
| 11 | `d11_timebase.py` | 时基核对（H=8/K=8 实际时长） | 仅打印 |
| 12 | `d12_oracle_ceiling.py` | 池内全窗口最优的 oracle 上界 | 仅打印 |
| 13 | `d13_ablation_cache.py` | 缓存消融所需的全部特征 | `/tmp/h2r_abl.pkl` |
| 14 | `d15_ablation_run.py` | 消融网格（通道×窗口×特征×K） | 仅打印 |
| 15 | `d14_ablation_pixel.py` | 追加 576 维稠密灰度描述子（约 6.5 GB 图像 I/O，较慢） | `/tmp/h2r_abl2.pkl` |
| 16 | `d16_ablation_run_pixel.py` | 含像素特征的完整消融 | 仅打印 |

> 编号 14/15 的执行顺序与文件名不一致是有意的：`d15` 只依赖 `d13`，可先看结果；`d14` 的图像 I/O 很慢，`d16` 依赖它。

## 硬编码路径

脚本内固定引用以下位置，换机器需同步修改：

```
/home/wxs/ReCAP-Cosmos-Policy/data/Human2Robot/derived/v04/   v04 冻结投影与 manifest
/DATA1/wxs/DATASETS/Human2Robot/data/v1/                      原始只读数据
/DATA1/wxs/ReCAP_M5B_V04_RUNS/features/                       冻结 geometry 统计与 WAN 缓存
/tmp/                                                          中间缓存
```

`d09` 及其下游读取 `stage4_smoke_plan.json`，因此对照使用的是**管线真实使用过的那 160 个 query**，而非重新抽样。
