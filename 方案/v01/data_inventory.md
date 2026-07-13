# M1 数据下载与访问确认 inventory

记录日期：2026-07-08（Asia/Shanghai）

本文件记录 `RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md` 中 M1 的数据源访问状态。当前文件保留 RH20T 相关访问验收；H&R / Human2Robot 已单独记录在 `data_inventory_human2robot.md`；MIME 仍待单独补充。

## 总览

| 数据源 | 本地状态 | M1 访问验收 | 后续处理决策 |
|---|---|---|---|
| MIME | 待补 | 待补 | 继续作为首个 public paired pipeline pilot |
| RH20T | 已下载 cfg3 的 320x180 RGB、LowDim、Calibration | 通过 | 进入 M2 预处理 pilot |

## RH20T

### 来源与许可

- 官方页面：https://rh20t.github.io/
- 数据集论文：`RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot`
- 许可按 episode 名称中的 scene 划分：
  - `scene_0001` 到 `scene_0005` 属于 RH20T-C，许可为 CC BY-SA 4.0。
  - `scene_0006` 到 `scene_0010` 属于 RH20T-NC，许可为 CC BY-NC 4.0；商业使用 RH20T-NC 或其训练模型不允许。
- 本次自动验收选中的 episode 是 `task_0001_user_0016_scene_0001_cfg_0003`，属于 RH20T-C 范围。

### 下载位置与体量

本地根目录：

```text
/DATA1/wxs/DATASETS/RH20T/320x180/
```

已下载并解压的 cfg3 模态：

| 模态 | 本地路径 | 本地解压占用 | 本地压缩包 | 官方 320x180 cfg3 标称压缩包 |
|---|---|---:|---:|---:|
| RGB | `/DATA1/wxs/DATASETS/RH20T/320x180/RGB/RH20T_cfg3` | 9.1G | 4.5G | 4.4GB |
| LowDim | `/DATA1/wxs/DATASETS/RH20T/320x180/LowDim/RH20T_cfg3` | 32G | 12G | 11.3GB |
| Calibration | `/DATA1/wxs/DATASETS/RH20T/320x180/Calibration/RH20T_cfg3` | 1.1G | 335M | 334.7MB |

本地 `320x180` 总占用：43G。

Task Description File：

| 文件 | 本地路径 | 本地大小 | 任务数 |
|---|---|---:|---:|
| Task Description File | `data/RH20T/task_description.json` | 25,754 bytes | 149 |

暂未下载：

- `RH20T_cfg3` Depth，官方 320x180 标称压缩包 71.3GB。

### 自动验收结果

验收脚本：

```bash
python tools/check_rh20t_m1_access.py \
  --root /DATA1/wxs/DATASETS/RH20T/320x180 \
  --task-description data/RH20T/task_description.json \
  --output data/RH20T/m1_access_check_cfg3.json
```

报告文件：

```text
data/RH20T/m1_access_check_cfg3.json
```

运行结果：`passed`

检查时间：2026-07-05T16:36:38Z，也就是 2026-07-06 00:36:38（Asia/Shanghai）。

自动选择的 paired episode：

```text
robot: task_0001_user_0016_scene_0001_cfg_0003
human: task_0001_user_0016_scene_0001_cfg_0003_human
```

顶层计数：

| 项 | 数量 |
|---|---:|
| LowDim robot episode dirs | 800 |
| LowDim human episode dirs | 775 |
| RGB robot episode dirs | 800 |
| RGB human episode dirs | 775 |
| Calibration dirs | 29 |

通过的 M1 RH20T 检查项：

- Robot low-dim 可打开：
  - `tcp_base.npy` 可读，样例 camera 数 8，样例 camera 记录数 221，`tcp` 为 7D 且 finite。
  - `gripper.npy` 可读，样例 camera 数 8，样例 camera 记录数 221，包含 `gripper_command` 与 `gripper_info`。
- Paired human video 可打开：
  - `cam_036422060909/color/color.mp4` 可被 `ffprobe` 打开。
  - codec 为 `h264`，分辨率 `320x180`，帧数 36。
  - `timestamps.npy` 可读，长度 36，时间戳单调非递减。
- Robot camera calibration 可打开：
  - metadata 指向 calibration timestamp `1631153393825`，目录存在。
  - `devices.npy`、`intrinsics.npy`、`extrinsics.npy`、`tcp.npy` 均可读。
  - 样例 intrinsics 为 `3x4`，extrinsics 为 `4x4`，数值 finite。
- Task Description File 可打开：
  - `data/RH20T/task_description.json` 可读，包含 149 个任务。
  - 当前验收 episode 对应的 `task_0001` 条目存在。
  - `task_0001` 英文描述为 `Press the button from top to bottom`，中文描述为 `自上而下按下按钮`。

### 风险与待补

- Paired human metadata 指向 calibration timestamp `1640674052372`，但该目录不在本地 `Calibration/RH20T_cfg3/calib/` 中；同时 human metadata 的 `calib_quality` 为 `-1`。M1 的 robot calibration 访问已通过，但 M2 若要稳定做 human/object 3D lifting，需要先解决 human calibration 可用性或改用无需 human calibration 的 lifting 路径。
- 当前未下载 Depth。按 v01 方案，Depth 只在需要更稳定的 3D human/object lifting 时补充。
