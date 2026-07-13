# RH20T 下游开源论文与代码审计

记录日期：2026-07-06（Asia/Shanghai）

## 结论摘要

- 严格同时使用 RH20T human 侧和 robot 侧的论文里，最贴近的是 `Toward Aligning Human and Robot Actions via Multi-Modal Demonstration Learning`。论文声称代码开源，仓库为 `utkauraslab/aligning_hr_actions`。但代码发布得比较“实验脚本化”：没有通用 RH20T loader，也没有直接读取 `task_*_human/metadata.json`；它假设 RH20T 已被整理成作者自己的 `icra25_align_human_robot/labeled/...` 中间目录和 CSV 标签。
- 公开代码里真正可复用的 RH20T 处理实现主要有两类：
  - RH20T-P API：读 RH20T 图像/深度加 RH20T-P annotation pickle，输出 primitive/action 层面的 robot 末端位姿、2D 投影和轨迹标签；主要是 robot 侧。
  - Aligning H/R Actions：robot 侧 RGB-D/Depth Anything V2 到 point cloud/voxel，再用 Perceiver 分类 8 类动作；human 侧方法在论文里写清楚，但仓库中没有完整 human 训练代码。
- `human metadata.calib_quality = -1` 在官方格式说明范围外。官方只定义了 robot metadata 的 calibration quality：`0` 表示有相机未标定，`1-5` 表示标定精度且越低越好；human metadata 只说包含 calibration timestamp and quality。官方 API 遇到 `metadata["calib"] == -1` 会直接 `raise NotImplementedError`。因此在工程上应把 human 的 `calib_quality=-1` 当作“不可用/未评估/不要信任的 calibration sentinel”，不能当作可用的质量等级。

## 严格 RH20T 相关项目

### 官方 RH20T API

来源：

- Dataset/API: https://rh20t.github.io/
- API repo: https://github.com/rh20t/rh20t_api

代码要点：

- `RH20TScene(folder, robot_confs)` 面向单个 scene folder 动态读取数据。它不自动配对 robot scene 和 `_human` scene。
- robot scene 可读 `transformed/tcp.npy`、`tcp_base.npy`、`joint.npy`、`gripper.npy`、`force_torque*.npy`，并通过 timestamp 插值获得 aligned TCP、joint、gripper。
- 图像侧通过各 camera 下的 `timestamps.npy` 或文件名时间戳做 nearest timestamp 匹配。
- calibration 逻辑只认 `metadata["calib"]` 指向的 `calib/<timestamp>/intrinsics.npy`、`extrinsics.npy`、`tcp.npy`。若 `calib == -1`，`_load_calib()` 和 `_load_calib_tcp()` 都直接抛 `NotImplementedError`。
- 没有看到 human metadata 的特殊处理逻辑；human folder 主要能作为视频/timestamp 容器读取，不能自然给出 robot action。

对本项目的含义：

- robot 侧应优先走官方 API 或复刻其 timestamp interpolation。
- human 侧如果 `calib=-1` 或 `calib_quality=-1`，不要用它做 camera-to-base 3D lifting；先作为 RGB/video prompt、2D hand/object cue 或语义标签来源。

### RH20T-P

论文：

- https://arxiv.org/abs/2403.19622

代码：

- https://github.com/rh20tp-rap/rh20tp-api

代码要点：

- 数据入口是：

```python
LazyRH20TPrimitiveDataset(
    data_root="/path/to/rh20t/dataset",
    anno_path="/path/to/rh20tp/annotation")
```

- `data_root` 期望已经提取好的 RH20T 图像/深度路径，格式类似：

```text
RH20T_cfg{cfg_id}/{task_id}/cam_{camera_id}/{color|depth}/{timestamp}.{jpg|png}
```

- `anno_path` 是 RH20T-P 自己的 pickle annotation。代码并不重新解析 RH20T 原始 `metadata.json`；它假设 annotation 内已有：
  - `camera`
  - `img_timestamps`
  - `intrinsics`
  - `extrinsics`
  - `tcp_base`
  - `gripper_command`
  - `gripper_info`
  - `base_aligned_timestamps`
  - `action_list`
- `LazyRH20TPrimitiveDataset` 按 primitive boundary 采样，返回：
  - current primitive action text
  - historical actions
  - current/target end-effector 3D pose
  - current/target end-effector 2D projection
  - end-effector trajectory
- `LazyRH20TActionDataset` 按固定采样率从 robot base-aligned timestamps 中生成 dense future action labels，并过滤静止动作。

对本项目的含义：

- RH20T-P 是最值得借鉴的 robot-side action label/primitive label 实现。
- 它没有使用 RH20T human 侧；若我们要做 human prompt + robot policy，可以把 RH20T-P 风格的 robot primitive/action label 和 human video context 自己接起来。

### Toward Aligning Human and Robot Actions via Multi-Modal Demonstration Learning

论文：

- https://arxiv.org/abs/2504.11493

代码：

- https://github.com/utkauraslab/aligning_hr_actions

论文要点：

- 任务：RH20T pick/pick-and-place 数据，5 users，10 scenes。
- human branch：从 human demonstration video 提取 RGB frames，Resize/Normalize 后进 ResNet-18，再进 LSTM + MLP，分类 8 类 intention/action：
  - Reaching
  - Grasping
  - Lifting
  - Holding
  - Transporting
  - Placing
  - Releasing
  - Nothing
- robot branch：RGB 通过 Depth Anything V2 生成深度，形成 RGB-D，Open3D 反投影为 point cloud，再 voxelization，Perceiver Transformer 分类 robot action。

代码审计要点：

- 仓库 README 只有一句说明。没有通用安装/数据准备说明。
- 代码中没有直接出现 RH20T 官方目录名、`_human`、`metadata.json`、`calib_quality` 或官方 API 调用。
- 数据路径硬编码为作者本地目录：

```text
/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004
```

- robot voxel 生成流程：
  - `Labelled_Robot_004.csv` 第一列提供 timestamp。
  - 在整理后的 robot root 下按同名 `{timestamp}.jpg` 搜索 `rgb/`、`depth/`、`depth_anything_v2/`。
  - camera intrinsic 写死为 640x360 版本：

```python
[[908.49682617/2, 0, 640.63098145/2],
 [0, 907.65515137/2, 351.33010864/2],
 [0, 0, 1]]
```

  - Depth Anything V2 输出被归一化成 8-bit depth image。
  - Open3D 用 RGB-D 生成 point cloud，代码内将 depth 映射到约 `0.4-1.2m`。
  - point cloud 投影回 image plane，给 3D 点取 RGB feature。
  - voxel bounds 写死为 `[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]`。
  - voxel feature 由 xyz/RGB/位置编码/occupancy 组成，Perceiver classifier 训练时 `input_dim` 从数据最后一维读取，论文中是 10。
- label 生成：
  - `label_dict.py` 读取 CSV 的 `Timestamp` 和 `ID`，生成 `{timestamp_stem: label_index}` JSON。
  - `rename_robot_testing_file.py` 手写 label map，顺序是 `nothing/reaching/grasping/lifting/transporting/holding/placing/releasing`。
- 训练：
  - `VoxelDatasetPtFile` 读取 `.pt` voxel 文件，展开到 `[N, C]`，最多随机采样 50,000 points。
  - `PerceiverClassifier` 用 learnable latents + cross-attention + MLP 做 8 类分类。
  - loss 是 class-weighted cross entropy，权重来自训练集类别计数。
- human 侧：
  - 论文描述了 ResNet-18 + LSTM + MLP。
  - 仓库中未看到对应 human video dataset、frame extraction、ResNet/LSTM 训练脚本。只有混淆矩阵绘图脚本里手写了 human confusion matrix。

对本项目的含义：

- 这是 strict human+robot RH20T 方向最相关的论文，但代码复现价值主要在 robot voxel branch。
- 它的 human/robot “对齐”更像共享 8 类语义动作标签，而不是从 human metadata 恢复 3D action。
- 对我们的 M2 更现实的复刻路径：
  - 先把 RH20T robot side 处理成 timestamp-level/action-level label 表。
  - human side 只用视频帧序列和同一套语义标签，不依赖 human calibration。
  - 若做 robot voxel branch，尽量不要硬编码 intrinsic/bounds，要从 RH20T calibration 或本地统计配置生成。

## 相关但非 RH20T 直接代码

### Motion Tracks / MT-PI

论文/项目：

- https://arxiv.org/abs/2501.06994
- https://portal.cs.cornell.edu/motion_track_policy/
- https://github.com/jren03/mt_pi_codebase

代码要点：

- 不是 RH20T 数据代码，但很适合参考 human/robot 双侧中间表示设计。
- robot demo：
  - 从 robot proprio/gripper 状态构造 gripper 上的 5 个几何点。
  - 用相机内参/外参把这些点投影到 agent1/agent2 图像平面。
  - 写入 `dataset.zarr`：`img1/img2`、`depth*_gs/colored`、`track1/track2`、`robot_action`、`proprio`、`gripper_open`、`episode_ends`。
- human demo：
  - 用 HaMeR 或 MediaPipe 提取 21 个 hand keypoints。
  - 检测失败的中间帧用线性插值补上，末尾无法补的帧丢弃。
  - 写入同样风格的 `dataset.zarr`：`img*`、`depth*`、`track*`、`is_right*`、`episode_ends*`。
- 训练时：
  - `ImageTrackDataset` 根据路径中是否含 `hand` 区分 domain。
  - 交替采样 human/robot，`label=0` 表示 hand，`label=1` 表示 robot。
  - 可加 keypoint mapping、KL/mean-var alignment、domain adversarial loss。

对本项目的含义：

- 如果想绕开 RH20T human calibration 的不确定性，一个可行方向是把 robot TCP 轨迹和 human hand keypoints 都投影/归一化成 2D motion tracks。
- 这比试图把 `human metadata.calib_quality=-1` 的视频直接 lift 到 robot base frame 更稳。

### UniSkill

项目：

- https://unikill-cvpr.github.io/
- https://github.com/UniSkill/UniSkill

代码要点：

- 方法上是 human/robot/general video 的技能表示学习，相关但不是 RH20T loader。
- 仓库搜索未见 RH20T、`RH20T_cfg`、`_human`、RH20T metadata 的专门处理。
- 它的数据接口更通用：按 video path + `metadata.json` 组织 sequence。

## 对 RECAP 当前 RH20T 方案的建议

1. `human metadata.calib_quality=-1` 不应阻塞 M2，但应改变路线：先把 human 侧作为 RGB/video context 或 2D motion source，不把它当可靠 3D calibration source。
2. robot 侧用官方 API/RH20T-P 思路稳定拿：
   - RGB frame nearest timestamp
   - `tcp_base`
   - `gripper_command/info`
   - joint
   - optional depth
3. human/robot pairing 用 episode name：

```text
task_XXXX_user_YYYY_scene_ZZZZ_cfg_CCCC
task_XXXX_user_YYYY_scene_ZZZZ_cfg_CCCC_human
```

4. 如果短期目标是“人手示范作为上下文学习指导机器人复现”，建议优先两条轻量路线：
   - Semantic alignment：人工/弱监督切 8 类动作阶段，human RGB -> intention，robot RGB-D/lowdim -> action。
   - Track alignment：robot TCP/gripper 投影成 2D/3D tracks，human hand/object 提取成 2D tracks，以 motion tracks 作为共享表示。
5. 若后续要做 human 侧 3D lifting，需要单独补齐 human calibration 可用性：检查全量 calibration 包是否缺 timestamp、是否可从相邻 robot calibration 复用、或改用单目手部/物体方法替代相机外参。
