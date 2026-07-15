# M5B-P2 I/O successor v5 冻结与重新验收报告（2026-07-15）

## 1. 结论

本轮已完成正式训练继续运行前的下一步：修复 Human2Robot-P2 数据读取中的整段 HDF5 图像加载，冻结新的 I/O successor v5，完成完整 Docker 环境回归与 4-GPU 启动前验收，并使用同一正式 cell 配置重新启动训练。

截至本报告记录时，新会话已完成 Iteration 1–5；四个 rank 的 NCCL collective 持续同步推进，没有出现本轮慢样本告警、CUDA OOM、NCCL timeout 或 rank 退出。该结果证明新 successor 已通过启动和早期训练验证，但不等同于整个 cell 已完成，也不改变 M5B-P2 仍未最终验收的事实。

## 2. 上一轮故障及根因判断

上一轮正式 cell 在运行过程中发生 `ProcessGroupNCCL` 的 `_ALLGATHER_BASE` 600 秒超时。故障前没有发现 CUDA OOM、GPU Xid 或显存分配失败；超时表现为 rank 之间的 collective 进度失配。

数据通路检查发现 `Human2RobotP2Dataset` 对每个样本使用 `images[:]` 读取完整 episode，再从内存中选择少量帧。实际训练每个样本只需要：

- robot episode：当前帧和未来末帧，共 2 帧；
- canonical/P1 human episode：future rows 对应帧。

在大 episode 上，整段图像读取可能超过 60 秒，而读取所需的两帧约为 0.0635 秒、0.585 MiB。该 I/O 放大足以产生 DataLoader rank straggler，进而使其他 rank 在 collective 中等待并最终触发 NCCL watchdog。由于故障日志不能单独证明唯一根因，本报告把它表述为“与现有证据最一致且已被直接消除的主要根因”，而不是无条件的唯一根因。

上一轮原始日志已作为只读失败证据归档：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/failure_evidence/20260715_M5B-MAIN-01_seed20260711_nccl_timeout.log`

归档 SHA256：

`ba40bc6d466a66168ab63d7a7f939078d3ec44a4a36d5be766ea904e8f94fddd`

## 3. 实施内容

### 3.1 精确索引图像读取

`cosmos_policy/datasets/human2robot_p2_dataset.py` 新增 `_read_image_rows(images, rows)`：

- 禁止空索引和越界索引；
- 对索引排序、去重后执行 HDF5 fancy indexing；
- 使用 inverse index 恢复调用方要求的原始顺序和重复项；
- robot 路径只读取 `[current_row, future_rows[-1]]`；
- canonical/P1 human 路径只读取 `future_rows`；
- 正式样本路径不再出现 `images[:]` 整段读取。

### 3.2 慢样本可观测性

每次 `__getitem__` 计时；当耗时达到 `HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS` 阈值时，输出一条 `[H2R-P2-DATA-TIMING]` JSON 记录，包含：

- global/local rank；
- DataLoader worker id；
- PID；
- sample/query/episode；
- 耗时或异常。

正式阈值冻结为 5 秒。

### 3.3 分布式故障诊断环境

v5 将下列环境值纳入精确运行时绑定：

| 变量 | 冻结值 |
| --- | --- |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | `65536` |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` | `1` |
| `TORCH_NCCL_DESYNC_DEBUG` | `1` |
| `NCCL_DEBUG` | `INFO` |
| `NCCL_DEBUG_SUBSYS` | `COLL` |
| `HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS` | `5` |

运行时绑定 schema 升级为 `human2robot-m5b-p2-runtime-binding-v2`。训练入口和 orchestrator 会共同验证这些值，避免 successor 声明与实际进程环境不一致。

## 4. Contract/overfit 回归覆盖

新增或加强的关键 contract tests 包括：

- fake image dataset 对任意 slice 直接报错，确保 robot 路径只读取两行；
- canonical human 只读取 future rows；
- P1 human 只读取 future rows；
- 慢样本日志包含 rank、worker 和样本定位字段；
- v5 successor、lock、launch/final schema、runtime binding 和诊断环境精确匹配；
- v3/v4 历史冻结产物保持不变。

修复前，针对整段读取的三个红测按预期失败；修复后，4 个定向测试全部通过。完整正式 Docker 测试结果为：

`158 passed, 3 warnings in 17.28s`

Docker suite receipt：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/docker_suite_receipt_v5.json`

receipt SHA256：

`352d753dbbc6f6663d5021e6cec3349ce5b7eb10fd9d37c8bf0d90c73f701be0`

## 5. 冻结产物

| 产物 | SHA256 |
| --- | --- |
| `M5B_P2_io_successor_v5.json` | `844f44c8e39178582f4a1cf7dcc5d16d510aad262e22104ff11eb93666d8fde2` |
| I/O successor lock | `67dad1418e97fe30e751fdf720a40e99cf30c96b51f9e52b61945bb058d5cb52` |
| launch schema v5 | `aef9326caf8056fc00e289a67e1fbc12148b8c5d9b6484ef0c356dbb26bc1c03` |
| launch schema lock | `5f4efb57d5e350cefab97d4c023b47d5aae5f93cd82c2e43a1daffac884ed094` |
| final schema v5 | `a77f9f4800ee697aeba532f76842e49023e014ff03c86ba61392837f6effb01f` |
| final schema lock | `f4eb5f418297745e51c2a3122fc6e41d25d999b01971afd75ff85d61c283b1fa` |

v5 successor 绑定的数据集实现 SHA256：

`298f93f4b64a1411e6f56e14bf94a616031af44878ce16417e318ae80503af81`

v5 successor 绑定的数据集测试 SHA256：

`599f712f459faadb1b9cab753c80bad2346ed17087c1ff974bddb4df5e6bcf25`

正式 source snapshot SHA256：

`07fe5ee3a9b90037ed724a94da76d9d87476e6eaea554af8c3ddb420ff3da511`

snapshot 共 731 个受控文件，位置为：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/source_snapshots/07fe5ee3a9b90037ed724a94da76d9d87476e6eaea554af8c3ddb420ff3da511`

## 6. 启动前重新验收

v5 preflight：

`方案/v03/M5B_P2_io_successor_v5_preflight_20260715.json`

验收结果：

- status：`passed`；
- blockers：空；
- `formal_queue_allowed=true`；
- 4 张可见 GPU；
- 权重哈希通过；
- `/DATA1` 可用空间约 1116.21 GiB；
- Docker 完整环境测试通过；
- source、successor、schema 和 runtime binding 均精确绑定。

launch activation：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/launch_activation_v5.json`

activation SHA256：

`3dfd5ab3133b4a7a8126a203ba2b10d1f358bd58eeac783dc42844196fc08e76`

## 7. 正式 cell 复跑状态

复跑 cell：

`learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711`

运行配置保持冻结：

- GPU：0–3，共 4 卡；
- world size / DP / FSDP：`4 / 4 / 4`；
- per-rank batch：25；
- gradient accumulation：2；
- global batch：200；
- seed：20260711；
- memory successor 与既有冻结值一致；
- I/O successor：v5；
- run manifest：`data/Human2Robot/derived/m5b_v03/run_manifest_v5.json`。

新会话已记录：

| Iteration | Loss | Time |
| ---: | ---: | ---: |
| 1 | 0.5939 | 67.55s |
| 2 | 0.5600 | 20.03s |
| 3 | 0.6149 | 36.63s |
| 4 | 0.5740 | 18.34s |
| 5 | 0.6688 | 18.41s |

Iteration 1 包含首次图构建和缓存等冷启动成本；后续迭代已进入稳定范围。当前没有本轮 `[H2R-P2-DATA-TIMING]`、OOM、NCCL timeout 或 Python traceback。

实时日志：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711.log`

该文件采用追加写入，前部仍包含历史失败会话。判断本轮状态时应以 `2026-07-15 02:57:51 UTC` 之后的记录为准；历史故障的不可变副本见第 2 节归档路径。

实时查看命令：

```bash
tail -n 100 -F /DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711.log
```

## 8. 验收边界与后续门槛

本轮已经完成的是“修复、冻结、完整环境验收、同一 cell 重新启动并确认早期迭代正常”。尚未完成的是：

1. 等待本 cell 产出冻结要求的正式 checkpoint 和训练结束状态；
2. 对本 cell 执行相应评估并生成 cell acceptance；
3. 按队列顺序完成其余 seed/condition cells；
4. 汇总 learned 与 deterministic baselines、paired comparisons 和最终验收材料；
5. 只有全部硬门槛通过后，才能声明 M5B-P2 或 M5 完全通过。

因此，本报告的准确状态是：**I/O successor v5 已冻结并通过完整 Docker 启动前验收；第一个正式 cell 正在正常运行，尚未完成。**
