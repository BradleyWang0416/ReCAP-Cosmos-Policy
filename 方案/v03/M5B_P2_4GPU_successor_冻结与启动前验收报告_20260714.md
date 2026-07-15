# M5B-P2 四卡 successor 冻结与启动前验收报告

日期：2026-07-14  
正式输出根：`/DATA1/wxs/ReCAP_M5B_P2_RUNS`  
结论：**四卡 v3 successor 已冻结，启动前门禁已通过；首个正式训练 cell 已在宿主 GPU 0–3 上执行，但在完成 2 个 optimizer iteration 后因 CUDA OOM 失败。当前为 1 failed、0 running、0/203 completed。**

> 启动后更新（2026-07-14 23:35 CST）：首个 cell
> `learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711`
> 已产生 `Iteration 1/2`，loss 分别为 `0.5931/0.5595`，第二步耗时 `20.04s`；随后 WAN VAE
> encoder 尝试分配 `7.18 GiB` 时 OOM（当时 GPU 0 可用 `7.02 GiB`）。该 attempt 已按
> `formal_result=false`、`no_imputation=true` 记录，未自动减 batch、改协议或启动第二个 cell。

## 1. 证据边界

- **VERIFIED**：由冻结 JSON/lock、当前源码、完整四卡 Docker、receipt、preflight 或 DAG plan 直接验证。
- **DERIVED**：由 VERIFIED 数值直接计算。
- **NEEDS_EXPERIMENT**：必须由首个正式四卡训练 cell 产生，本文不提前宣称。

## 2. successor 冻结内容

四卡运行时冻结件为 `M5B_P2_4gpu_successor_v3.json`，SHA256：
`6f333136b343cee87dca3c0328a73ffd441d3059633159d02e6f514573b809ab`。

其运行时精确值为：

| 项目 | v2 八卡历史值 | v3 四卡正式值 |
|---|---:|---:|
| visible GPU / world size | 8 | 4 |
| data-parallel world size | 8 | 4 |
| FSDP shard size | 8 | 4 |
| 每 DP rank batch | 25 | 25 |
| gradient accumulation | 1 | 2 |
| 有效全局 batch | 200 | 200 |
| optimizer updates / cell | 7000 | 7000 |
| checkpoint 间隔 | 1000 updates | 1000 updates |
| seeds | 20260711/12/13 | 不变 |

有效全局 batch 的计算为 `4 × 25 × 2 = 200`。因此每个 cell 仍呈现
`7000 × 200 = 1,400,000` 个训练 example（DERIVED），但四卡梯度归约顺序、rank-local
随机流和微批次分组不保证与未执行的八卡方案逐 bit 相同。所有 48 个 learned cell 必须统一使用
v3，禁止混入 v2 checkpoint；当前只有首个 v3 cell 正在运行，不存在跨 successor 混合结果。

v3 只覆盖运行时并行、梯度累积和 activation 版本。以下内容复用且不重做：203-cell registry、
48 条 prepared learned-cell 输入、三 seed、147 evaluation、5 aggregate report、数据 split、
retrieval/index/statistics、优化器/学习率/调度器（梯度累积项除外）和验收统计。

## 3. 代码与 fail-closed 约束

当前候选源码 SHA256 为
`caa481ea3b9ad585997ca5e53f02858e264e09688d71da02103da4a4180480e7`，共 731 个受控源码文件。
不可变 snapshot manifest SHA256 为
`fdb5a07a2f3c7f496265b3417d7573db1fef6e284112612234c056902d129b6a`。

代码已强制：

- `torchrun --nproc_per_node=4`，子进程只见逻辑 GPU `0,1,2,3`；
- formal config 固定 `fsdp_shard_size=4`、`grad_accum_iter=2`、batch/rank=25；
- runtime binding 同时核对 world/DP/FSDP/grad-accum/effective-batch/visible-device-count；
- DCP auditor 要求四个 rank 的完整组件文件；
- dispatcher 若从旧八卡或 `--gpus all` 容器调用，会在启动 subprocess 前拒绝；
- v2 activation、receipt、run manifest 与 final acceptance 只读保留，不能授权 v3；
- 新 manifest 为 `data/Human2Robot/derived/m5b_v03/run_manifest_v3.json`。

宿主物理 GPU 通过 `M5B_P2_GPU_DEVICES` 选择，必须恰好四个互不重复的编号。默认示例为
`0,1,2,3`；本次验收使用宿主 0–3，容器内实测四张均为 RTX 4090。

## 4. Docker 与 activation 证据

四卡完整 Docker suite：**150 passed、3 warnings**。warnings 均为第三方依赖 deprecation，
没有测试失败。receipt：

- 路径：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/docker_suite_receipt_v3.json`
- SHA256：`696524101d4963d7e0f6f9a1874edf16e94e4f1be3b18e17e9ac58775f9b24b0`
- `visible_gpu_count=4`
- `cell_execution_started=false`（receipt 在正式 launch 前签发；这是签发时刻属性）

launch activation：

- 路径：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/launch_activation_v3.json`
- SHA256：`f50feb8ceea96515f7d1eebaed1112a826f515c6ca1a0c3a4386cf569a01fd16`
- `formal_queue_allowed=true`
- `p2_acceptance_allowed=false`
- world/DP/FSDP=`4/4/4`，batch/rank=25，grad accumulation=2，有效 batch=200

签发后的最终 preflight 保存在
`/DATA1/wxs/ReCAP_M5B_P2_RUNS/final_preflight_before_first_cell_v3.json`（SHA256
`ea5c64ae684d14d1f37f0a6a30ff284af1a32c5245ebe361fefe479c9d653ccd`），其结果为：

- blockers=`[]`；
- visible/expected GPU=`4/4`；
- `/DATA1` 挂载可写；
- 初始化权重与 tokenizer 完整内容 SHA256 均通过；
- source snapshot 与 Docker receipt 均绑定当前源码；
- 当时可用空间 `1,199,359,959,040 bytes = 1116.99 GiB`；
- `formal_queue_allowed=true`、`formal_queue_started=false`。

DAG plan 在启动前的结果为：203 个 cell 中 completed=0、invalid=0、missing=203，matrix blockers=`[]`，
首层 ready cell 共 51 个（48 learned checkpoint + 3 nonlearned retrieval-only artifact）。随后仅调用一次
`run-cell` 启动首个 cell；run manifest 当前为 `status=failed`、`attempt_count=1`、
`formal_result=false`，失败类型为 `P2Error`，未启动第二个 cell。

首个 cell 的实时证据：

- log：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711.log`；
- runtime binding：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/cosmos_policy/human2robot_m5b_formal/cosmos_predict2p5_2b_human2robot_no_retrieval_seed20260711/m5b_p2_runtime_binding.json`；
- container dispatcher PID：`7623`（不是宿主 PID，不能直接用宿主 `ps -p 7623` 检查）；
- 实测初始化/首步观测值约 `44.7–44.9 GiB/GPU`，第二步后一次采样约 `19.7–20.2 GiB/GPU`；
- runtime binding 实测 world/DP/FSDP=`4/4/4`、batch/rank=`25`、grad accumulation=`2`、有效 batch=`200`、bfloat16、seed=`20260711`。
- OOM traceback：rank 0 的 WAN VAE encoder 需要连续 `7.18 GiB`，总显存 `47.37 GiB`、可用
  `7.02 GiB`；PyTorch 已分配 `27.48 GiB`、保留但未分配 `11.87 GiB`。退出后 GPU 0–3 均已
  回到 `0 MiB`，无训练进程残留。

## 5. 仍需真实训练回答的项目

以下全部是 **NEEDS_EXPERIMENT**：

- 冻结 batch/rank=25 的四卡方案已实测不能稳定越过第 3 个 batch；在新的内存 successor 决策前
  不得把本 attempt 当成 cell 结果，也不得自动重跑；
- 稳态 optimizer-step 吞吐、单 cell/全队列 wall-clock（只有第二步 `20.04s`，样本不足）；
- 第一份四 rank step-1000 DCP 的总字节数和实际长期存储预算；
- loss 有限性、首个 step-1000 checkpoint 的可恢复性；
- 48 个 learned cell、147 个评估和最终统计结果。

因此“门禁通过”只表示可以按 v3 启动第一个正式 cell，不表示 P2/M5/Gate C 已通过。正式开始前应
使用真正稳定的四张物理卡启动 `start_m5b_p2_formal_docker.sh`；如果物理卡选择不是本次验收所用的
0–3，建议在该映射下重新运行 receipt、issue-launch 和 post-activation preflight，再执行首个 cell。

## 6. 后续 memory-successor

本报告记录的 v3 首次 OOM 仍作为历史证据保留。针对该 OOM，现已按用户指令冻结仅增加
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 的 memory-successor v4，并重新完成完整 Docker
验收。当前有效启动工件已升级为 v4；v3 activation 不再授权后续重试。详见
`方案/v03/M5B_P2_memory_successor_v4_冻结与重新验收报告_20260714.md`。
