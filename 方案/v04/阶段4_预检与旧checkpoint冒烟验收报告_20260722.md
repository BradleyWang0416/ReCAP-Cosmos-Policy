# Human2Robot v04 阶段 4 预检与旧 checkpoint 冒烟验收报告

结论：**VERIFIED_STAGE4 / PASSED**。阶段 4 已完成 synthetic 数据/身份/检索/评估闭环测试、五个冻结 partition manifest、seen-train geometry statistics、只读当前帧 WAN visual feature cache，以及三个 v03 step-7000 checkpoint 在 v04 robot-dev 上的只读冒烟。正式 attempt 0005 和独立逐条审计均通过，协议锁已设置 `training_allowed=true`、`stage5_allowed=true`。

本阶段的 1,440 条预测仅证明旧 checkpoint 能在冻结 v04 接口下闭环运行、来源隔离、收据完整且数值有限。它们的 `formal_result=false`、`performance_claim_allowed=false`，不得用于模型选择、方法排序、RECAP 优越性或任何泛化/部署结论。

## 1. 正式边界与冻结输入

正式执行入口为：

```bash
.venv/bin/python tools/human2robot_v04_experiment.py prepare-features \
  --execute --run-id stage4_prepare_20260721 --visual-batch-size 32
```

运行继续固定在 `cosmos-policy:latest` image ID `sha256:4fc8db9f70eeb96fee271ef282385163ec1da220dfed35da9c832fb6769891e8`、物理 GPU `0,2,5,6`、容器逻辑 GPU `0,1,2,3` 和 `--network=none`。attempt 0005 的完整 preflight 为 `PASSED`：四 rank NCCL all-reduce=6.0，PyTorch 2.7.0+cu128、CUDA 12.8、NCCL 2.26.2、编译扩展、本地权重 SHA、离线环境、v03 freeze 与存储门禁全部通过。

正式调用重新绑定以下上游证据：

| 输入 | SHA256 |
|---|---|
| 阶段 1 source split manifest | `7869a078b19ba18aaa6a92c22bec26998a81d412165a02eb8cb6c6aec1c879ed` |
| 阶段 2 retrieval contract | `27eceb61565d01297d4ec4ff19d166b5ff5c8d5e9af7916d92d5d9837af651d9` |
| 阶段 3 experiment interface | `10eab34a3479cdf54da5125c1de6d8035b372631f529298c11589ac3032a7e6b` |
| 阶段 3 full-suite receipt | `2717f2ff31b8a1cac8afc64ec34caeb77f5da9ea9f7b2d9ce7ca907074223a26` |
| v03 frozen manifest | `d20eae44a2b1d0e1287dc8ae0973e2f713413aab904ef2dff1e304d927db1ab4` |
| protocol / split / raw inventory | `55854766...` / `8ca7c835...` / `d9a2f361...` |

attempt 0005 invocation manifest SHA256 为 `56f07e507e92adda62655885fde6104475d0a1355be2329472eee30e765fd2f6`，绑定了阶段 4 orchestrator、四卡 feature worker、checkpoint smoke worker 及实际复用的 v03 checkpoint inference/diagnostic backend；manifest 和 receipt 均为 `0444`。

## 2. Synthetic 闭环与测试覆盖

新增 5 项阶段 4 测试，覆盖：

- synthetic HDF5 的来源身份、role/partition、gap-safe window 与主检索闭环；
- 送入 `validate_human2robot_batch` 的完整模型 batch metadata；
- population geometry mean/std 的参考重算、有限性与非退化性；
- WAN cache 只读取 current visual frame、禁止 future/target 读取、有限值及断点复用；
- 每 episode 固定 8 个等距合法窗口、协议锁的非性能边界与阶段 5 授权。

冻结四卡离线 Human2Robot 全量套件自动纳入上述测试，结果为 **211 passed、3 warnings**，高于 206 项门槛，返回码 0。三个 warning 均来自 Megatron/SWIG 的第三方 deprecation 提示。full-suite receipt SHA256 为 `d024e50c4628946ba1e810c79c9ff45fb4d9ac5180c52d1eb322d29d02b03afa`，权限为 `0444`。

## 3. Partition manifest 物化

五个 manifest 均为 `FROZEN`、`0444`，且绑定相同 split SHA256 `8ca7c8352e031292975779fd254db9591963b2cd3612b1f1b5ab65b6e79ebeaf`：

| Partition | episode | 合法窗口 | manifest SHA256 |
|---|---:|---:|---|
| seen_train | 654 | 244,372 | `fc368ed780d0ff7fa055e83246cbaf1c5d4dc561aa30ffcede38a8b87d557cb5` |
| seen_validation | 82 | 31,636 | `d0940bb38da5cc248b05e5ae2748968c8d3489c4c3b8ffa01d22907c37ebf98f` |
| v04_human_pool | 40 | 13,426 | `02557a497e25d44794188f27026ca73d21120cef63cee1c3032dd3bf9998d0b7` |
| v04_robot_dev | 20 | 6,238 | `bbad7bb3a9fdd89f5e7d8d6087be6ca8b0937e460aed5a4df8cf45710ae3c959` |
| v04_robot_final | 80 | 25,363 | `cbec468f1a33efca9d45057099484c94489aa5d00f39d60ebfd169d0fa70ddd9` |

## 4. Geometry 与冻结 WAN cache

geometry statistics 只使用 `seen_train` 的 gap-safe 合法窗口，覆盖 human/robot 两种 role，共 3,909,952 条 relative 10D row；十维 mean/std 全部有限且每维 std 大于 0，`future_rows_read=0`、`target_datasets_read=0`。产物 SHA256 为 `d16fda28986f4860a237fabfb598792fb4630764bc3766735a97d94198e630eb`，状态 `FROZEN`、权限 `0444`。

visual cache 绑定 tokenizer SHA256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`，按 current frame 生成 1,612 个只读 shard、597,043 个 feature：

| Partition | shard | feature |
|---|---:|---:|
| seen_train | 1,308 | 488,744 |
| seen_validation | 164 | 63,272 |
| v04_human_pool | 40 | 13,426 |
| v04_robot_dev | 20 | 6,238 |
| v04_robot_final | 80 | 25,363 |

cache index 的 `future_frames_read=0`、`target_datasets_read=0`，shard bundle SHA256 为 `be5c6e03c9d5f7eb3bfc8195571729088333d863f16fdabacff21ddd771faba6`；index SHA256 为 `48e895a1c664790cab85d837506e63facf7c97eca207e3d15f20d4e55c589d17`，状态 `FROZEN`、权限 `0444`。

## 5. 旧 checkpoint 冒烟设计与结果

smoke plan 从每个 held-out task 固定选择 5 个 robot-dev episode，每 episode 等距选择 8 个合法 query window：4 task × 5 episode × 8 window = 160 query。每个 query 使用冻结 `geometry_plus_visual` 检索的 top-3 human candidate，因此每方法 480 条 inference receipt。plan 的 future-row/target-dataset 读取均为 0，SHA256 为 `3a2c123da9c3f08bc9c6a0ab8cba2e75f0668a46d9ed0744c4cd5197cf4a47e6`。

| 方法 | v03 step-7000 payload SHA256 | query × rank | finite / nonfinite | 缺失 / provenance / gap | receipt bundle SHA256 |
|---|---|---:|---:|---:|---|
| no_retrieval | `6be4056171acc96c11bb85618feffb2e6dbabb78e29c2f74f0a6a3ce11ed787a` | 160 × 3 | 480 / 0 | 0 / 0 / 0 | `c136da0ba42af563c7adfb6a3d54feab9b43610a7df7c4af397f772a3d4c688d` |
| co_training | `a8a3d2da5b71c7efc21ec68435955b397476682c5ac62210451184ad1f244713` | 160 × 3 | 480 / 0 | 0 / 0 / 0 | `b9385d58561ffb5aa13b85a11e7eafc9098bc03ec12758ffebb57f6e971c8a24` |
| recap_hand_ret | `06c5e0abf9c7507fce09a033772179f655f944e255be80130ab4252e5a6ef82b` | 160 × 3 | 480 / 0 | 0 / 0 / 0 | `994c4288a54107224b48170b391860cd09c31901785fd25da428f5d203a2b7d2` |

独立审计逐条读取全部 1,440 个 JSON，而非只信任 summary：验证精确文件集合 `q0000_r0`～`q0159_r2`、160 个唯一 query、480 个 query-rank pair/方法、方法与 checkpoint payload 绑定、query/candidate source SHA 与 partition、canonical retrieval record SHA、`[8,10]` shape、80 个实际浮点值、原始 float32 prediction SHA/min/max、未来帧/target/gap 计数和 summary bundle。结果为 0 error、0 `.partial`；每方法 480 个 prediction SHA 均唯一，全部 receipt/summary 权限为 `0444`。

## 6. 协议锁与阶段门禁

`stage4_protocol_lock.json` 状态为 `VERIFIED_STAGE4`，SHA256 为 `834fe271bfe964c0b4f6e6c3b8ba899191609643acd166df0fa8e2811fca7ade`，锁定：

| 协议 | SHA256 |
|---|---|
| data | `8c214c6612bd70bff05b51747d6ebeed0d60d334715c88587ca51a3cac899525` |
| retrieval | `fd255610265f55bec6347d512a6b031c7ff9235257fa5e1a4e43efa7b3c147be` |
| training | `c3667592906de61afebeaf56559c65d6e580c1a5a175bda773e9d277b15c884f` |
| final evaluation | `12d6d89efc873bfb8c384ebd5af0a0d3ec6fa4b0972c84211295b49b39d8fafd` |

锁内 provenance、future-target independence、nonfinite、missing receipt 和 gap-crossing 五项计数均为 0，因此 `training_allowed=true`、`stage5_allowed=true`。若上述任一计数非零，构建协议锁会 hard-fail，不能授权训练。

正式 attempt 0005 receipt 状态为 `PASSED`，SHA256 为 `00a15f9a69c08ffe7a39a98af3cbc4fec5b551cc022a296f3fd43d2084d94d9e`；全局 stage4 orchestrator 状态已收口为 `COMPLETED` 并指向 attempt 0005。

## 7. 偏差与处置

- attempt 0001 在 geometry 完成后，WAN 四卡 worker 使用 `torchrun --standalone`，与 `--network=none` 下的 hostname 解析不兼容；人工中断，没有正式 receipt。后续改为显式 loopback master address/port，WAN cache 从头完成。
- attempt 0002 完成 WAN cache 后，旧 inference backend 把 `/workspace/...` 绝对 config 路径传给 Hydra，no_retrieval 在推理前失败；receipt SHA256 `3fa809cdef8988f840de6d870048fba05dd0389a09451ac7b53330e5adfd9817`。改用已审计的 intermediate-checkpoint backend，以 package-relative config 加载 step7000 DCP。
- attempt 0003 因启动命令漏传 `HUMAN2ROBOT_V04_GPU_DEVICES` 在 preflight 阻断，未启动数据、模型或推理；receipt SHA256 `3019d8a417370dd028db4baa1d0610d704a5b86a1d9426178b520a1b8ca1e41f`。
- attempt 0004 已加载 checkpoint，但 strict Human2Robot adapter 检出 smoke item 缺少冻结 metadata，在第一条推理前失败；receipt SHA256 `a813a9ab84724c89ac4ea22561fed6a13d22d2b1ed6d89e61adb785f52da4c5a`。补齐 method/experiment/variant、H/K、future-offset/gap/heldout/query-command/deployment-adapter/protocol SHA metadata，并新增 synthetic adapter 验证后通过。
- 另有镜像 `tag@digest` 本地引用、Docker `--gpus` 引号、一次错误 pytest node ID 等命令装配偏差，均在 Python/数据/模型正式执行前失败，不生成正式产物。attempt 0005 仍绑定同一冻结 image ID、源代码 SHA 和完整 preflight。
- 三个 smoke worker 退出时各出现一次未显式调用 `destroy_process_group()` 的 PyTorch 清理 warning；进程退出码为 0，summary、1,440 条 receipt、数值及 bundle 独立审计均通过。该 warning 不改变收据或科学语义。

## 8. Claim-to-evidence 边界与下一阶段

- `VERIFIED`：阶段 4 的 synthetic 闭环、partition 物化、geometry/WAN cache、三方法旧 checkpoint smoke、五项 fail-closed guardrail、协议 SHA 和阶段 5 授权均有正式证据。
- `NEEDS_EXPERIMENT`：三方法 v04 单 seed 重训、dev、final、pool growth、oracle-phase 诊断和最终报告仍未开始。
- `OVERCLAIM_RISK`：旧 checkpoint smoke 不是正式性能实验，不允许声称 RECAP 优于 baseline、提升泛化性/鲁棒性或具备部署能力。

建议：**进入阶段 5，严格按已锁定顺序 no_retrieval → co_training → recap_hand_ret 重训；不要复用本阶段旧 checkpoint 预测作为训练选择或论文指标。**
