# M1 Human2Robot 数据下载与访问确认 inventory

记录日期：2026-07-08（Asia/Shanghai）

本文件单独记录 `RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md` 中 M1 的 H&R / Human2Robot 数据源访问状态，不与 RH20T inventory 混放。

## 总览

| 数据源 | 本地状态 | M1 访问验收 | 后续处理决策 |
|---|---|---|---|
| H&R / Human2Robot | 已下载 v1 HDF5，本地 1,316 条 episode | 访问通过 | 可进入本地 M2 预处理 pilot；主线仍不依赖它 |

## 来源

- 论文描述该数据集包含 synchronized human-robot paired episodes，适合验证最接近 perfectly aligned human-robot pair 的桥接假设。
- 本地目录带有 Hugging Face 下载缓存痕迹，但当前数据目录下未发现 `README` 或 `LICENSE` 文件。
- 因此本次结论只覆盖 M1 的本地访问验收；引用格式、再分发限制需要在进入正式训练结论前补充确认。

## 下载位置与体量

本地根目录：

```text
/DATA1/wxs/DATASETS/Human2Robot/data/v1/
```

本地 HDF5 inventory：

| 项 | 数量 |
|---|---:|
| HDF5 episode files | 1,316 |
| Task directories | 33 |
| HDF5 total bytes | 120,743,378,270 |
| HDF5 total size | 112.5GB |
| `du -sh` 本地占用 | 113G |

任务目录计数记录在自动验收报告 `data/Human2Robot/m1_access_check_v1.json` 的 `inventory.task_episode_counts` 中。

## 自动验收结果

验收脚本：

```bash
python tools/check_human2robot_m1_access.py \
  --root /DATA1/wxs/DATASETS/Human2Robot/data/v1 \
  --output data/Human2Robot/m1_access_check_v1.json
```

报告文件：

```text
data/Human2Robot/m1_access_check_v1.json
```

运行结果：`passed`

检查时间：2026-07-08T02:46:03Z，也就是 2026-07-08 10:46:03（Asia/Shanghai）。

自动选择的 paired HDF5 episode：

```text
roll/episode_0.hdf5
```

通过的 M1 Human2Robot 检查项：

- HDF5 树可访问：
  - 找到 1,316 个 `episode_*.hdf5` 文件，分布在 33 个任务目录中。
  - 自动选择的 `roll/episode_0.hdf5` 大小为 144.3MB，包含 549 帧。
- Paired human/robot RGB 可打开：
  - `cam_data/human_camera`：人手示范侧的 RGB 观察流；第一维是时间帧，后面三维是 `height x width x RGB channel`。本次样本为 `549 x 240 x 426 x 3`，`uint8`，gzip 压缩。
  - `cam_data/robot_camera`：机器人执行侧的 RGB 观察流，与 `human_camera` 使用同一 episode 时间轴和 frame index；用于检查 human/robot 是否为 paired 视角。本次样本为 `549 x 240 x 426 x 3`，`uint8`，gzip 压缩。
- Paired human/robot depth 可打开：
  - `depth_data/human_camera`：人手示范侧的逐像素深度图，时间轴与 `cam_data/human_camera` 对齐；二维空间分辨率为 `240 x 426`。本次样本为 `549 x 240 x 426`，`uint16`，gzip 压缩。
  - `depth_data/robot_camera`：机器人执行侧的逐像素深度图，时间轴与 `cam_data/robot_camera` 对齐；可用于后续 3D object/hand lifting 或 paired depth sanity check。本次样本为 `549 x 240 x 426`，`uint16`，gzip 压缩。
- Robot state/action 可打开：
  - `action`：每帧对应的机器人动作/控制目标，维度为 7。根据数值范围和 `end_position` 的前 6 维对应关系，当前可按 `6D 末端目标/增量 + 1D gripper` 理解；具体单位、坐标系和第 7 维含义需要官方 README 或采集代码最终确认。本次样本为 `549 x 7`，数值 finite。
  - `qpos`：robot joint position，即机器人关节位置/构型向量；`q` 是机器人学里常用的 generalized coordinate 记号，`pos` 表示 position。本数据中每帧 7 维，通常对应 7-DoF 机械臂的 7 个关节角/关节位置。本次样本为 `549 x 7`，数值 finite。
  - `qvel`：robot joint velocity，即与 `qpos` 对应的关节速度向量；每帧 7 维，通常对应 7 个关节的一阶速度。本次样本为 `549 x 7`，数值 finite。
  - `end_position`：机器人末端执行器位姿，按 6D 表示记录末端 Cartesian position 与 orientation。样本值显示前三维为位置量，后三维为姿态角量；具体单位和欧拉角顺序仍需官方说明校准。本次样本为 `549 x 6`，数值 finite。
  - `gripper_state`：夹爪状态标量，样本中取值范围为 0 到 1，可作为 open/close 或归一化夹爪状态使用。本次样本为 `549`，数值 finite。
- Human hand trajectory 可打开：
  - `transformed_hand_coords`：每帧的人手关键点/控制点 3D 坐标，已经过数据集预处理变换到统一坐标表达；`24 x 3` 表示每帧 24 个点、每个点 3D 坐标。它是后续 human trajectory retrieval、human-to-robot alignment 和 wrist/grip lifting 的主要候选输入。本次样本为 `549 x 24 x 3`，数值 finite。
  - `transformed_hand_frames`：每帧的人手局部坐标系/姿态 frame 表达；`4 x 3` 可理解为 4 个 3D frame 向量或 frame anchor，用于描述手部朝向与局部几何关系。具体 4 个 frame 的语义需要官方 README 或预处理代码确认。本次样本为 `549 x 4 x 3`，数值 finite。
- 时间同步字段可检查：
  - `timestamp` 长度为 549，单调非递减。
  - `step` 长度为 549，单调非递减。
  - 所有被检查数据集第一维均为 549，time axis 一致。

## 风险与待补

- 当前本地 episode 总数为 1,316，与方案中先前根据论文记录的 2,600 条 paired synchronized episodes 不一致；需要确认是否只下载了 v1 子集、是否还有 train/test split 或额外版本未下载。
- HDF5 schema 与 v01 M2 的 canonical HDF5 schema 不同，需要在 M2 明确字段映射：`cam_data/*` 到 `obs/images`，`qpos/end_position/gripper_state` 到 `obs/states`，`action` 到 `actions`，`transformed_hand_*` 到 human retrieval/pool metadata。
