# M5B-P2 memory-successor v4 冻结与重新验收报告

日期：2026-07-14

## 1. 结论

本次 memory-successor 已冻结并完成重新验收，结论为：

- 唯一获准的训练运行时变化是
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
- 四卡、batch、梯度累积、有效全局 batch、精度、optimizer steps、checkpoint 周期、模型、数据、
  seed 和评估协议均继承 v3，未改变；
- 完整四卡 Docker 验收为 **152 passed、3 warnings、0 failed**；
- 最终 post-activation preflight 为 `passed`，`blockers=[]`；
- v4 DAG 已可调度，但 `formal_queue_started=false`，本次没有启动任何训练 cell；
- `p2_acceptance_allowed=false`。这次通过的是“可以按 memory-successor 重试首个 cell”的启动前门禁，
  不是 P2、M5 或 Gate C 的实验结果验收。

## 2. 冻结决策与边界

冻结文件：

- `方案/v03/M5B_P2_memory_successor_v4.json`
  - SHA256：`c5f3334e4fecc81b38466046917d7aefdf1d6eaf7b0e8344458b05cf02455bc2`
- `方案/v03/M5B_P2_memory_successor_v4.lock.json`
  - SHA256：`a60f8def2b6a8ed08e50b7469c5689ebedf8b23ffd74786605171e8866242bc0`

其父协议为 `M5B_P2_4gpu_successor_v3.json`，SHA256
`6f333136b343cee87dca3c0328a73ffd441d3059633159d02e6f514573b809ab`。

继承且保持不变的关键训练参数：

- visible/world/DP/FSDP/checkpoint ranks：`4/4/4/4/4`；
- batch per DP rank：`25`；
- gradient accumulation：`2`；
- effective global batch：`200`；
- precision：`bfloat16`；
- max optimizer steps：`7000`；
- save every optimizer steps：`1000`；
- seeds：`20260711`、`20260712`、`20260713`。

明确未授权改变：模型结构或可训练参数、数据与 retrieval 工件、优化器、学习率、batch 语义、精度、
step 数、checkpoint 语义、评估矩阵和统计门限。

## 3. 冻结依据

v3 首个正式 cell 在完成 2 个 optimizer iteration 后于 WAN VAE encoder 发生
`torch.OutOfMemoryError`：申请 `7.18 GiB` 时 GPU 0 仅余 `7.02 GiB`；PyTorch 已分配
`27.48 GiB`，保留但未分配 `11.87 GiB`。因此本 successor 只针对 allocator 碎片/segment 增长策略，
不把失败 attempt 当成 cell 结果，也不声称该环境变量已经证明能够稳定完成训练。

原失败证据仍保持原哈希：

- v3 run manifest：
  `979d8172a52688ae9e6f83d5409e0c7f7e5e7a68960944d3ae5bc3ae40faa526`；
- training log：
  `b22e2f20473558cc3dfb5c13514e0f5807189d5104548de90255f148bef8b8ba`；
- runtime binding：
  `863ff4f307313876494a61d1c5939899a01e4d6ccfc229187927b8f7dba07be2`。

为防止未来重试同名 cell 时覆盖原路径，log 与 runtime binding 已复制到：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/failure_evidence/v3_oom_memory_successor/`

副本 SHA256 与原文件逐字节一致。v3 run manifest 继续保留在
`data/Human2Robot/derived/m5b_v03/run_manifest_v3.json`。

## 4. 实现与 fail-closed 约束

执行面已升级为 v4：

- matrix 在加载时验证 memory-successor 与 lock 的固定 SHA256、父 successor、唯一 delta、继承参数和
  OOM 失败事实；
- run manifest、每个训练 cell binding、训练命令和 runtime binding 都绑定
  `memory_successor_sha256` 与精确 allocator 字符串；
- launcher、handler 计划、Docker suite、preflight 和训练子进程均显式设置
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
- 训练入口记录并核对实际 allocator 值；缺失或不同值会在正式结果成立前 fail closed；
- v4 receipt、activation、run manifest、DAG、preflight 和 final-acceptance schema 使用独立版本；
- v3 activation 即使其它字段正确，也不能授权 v4 执行；
- final acceptance 仍要求全部 203 个 cell 和终端报告真实完成，本次不能生成 P2 通过结论。

冻结 schema：

- launch schema v4 SHA256：
  `3f3d1e7b55f67ebbee736b6875a64868b8cf1db4c10f7e1eab25ab8d67bace2d`；
- launch schema lock SHA256：
  `a5297b0e62e81d42faae7c679c9cb079cc135d79cfe9428c3150e46a12ac7884`；
- final acceptance schema v4 SHA256：
  `072b71a45c4566dbb82de972dc46c0c9cfde27b110519ccfada70ad9596fc0b1`；
- final acceptance schema lock SHA256：
  `4908978f1428187bfb0f705dd27dbc750555eb8421f9ff8263942bc230063e38`。

## 5. Docker 全量验收

验收严格在现有完整四卡 Docker 环境中完成，没有使用宿主残缺环境，也没有下载新文件。

- 测试结果：`152 passed, 3 warnings in 24.13s`；
- warnings：第三方依赖 deprecation，不含测试失败；
- visible GPU：`4`；
- receipt 中的 allocator：`expandable_segments:True`；
- receipt 中的 `cell_execution_started=false`；
- 最终候选源码 SHA256：
  `edd17f1dfa032a094a7cfe00f1fbbf4e4d4e2ad65b70fdac4f911e3612b3956b`；
- 受控源码文件数：`731`。

正式证据：

- v4 run manifest：
  - 路径：`data/Human2Robot/derived/m5b_v03/run_manifest_v4.json`
  - SHA256：`764b38e60a46dfefd21da4a0e3e82254ced5d8892f8bd5ec6a7c0de5aa0dd322`
- Docker receipt：
  - 路径：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/docker_suite_receipt_v4.json`
  - SHA256：`3d9cea1909d3e5530fff1c12e31656e233183de8223d2d4d15cabb8fc3465b2a`
- launch activation：
  - 路径：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/launch_activation_v4.json`
  - SHA256：`c8eab162355f61f48608a76ef9bdf5099db2feff095aff8f3da8327c8f8766c4`
- post-activation preflight：
  - 路径：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/final_preflight_before_memory_retry_v4.json`
  - SHA256：`8a381ca2650a74df0f26568f14c7e5783512414101639ddcbd43d8100218fadd`
- source snapshot manifest：
  - 路径：`/DATA1/wxs/ReCAP_M5B_P2_RUNS/source_snapshots/edd17f1dfa032a094a7cfe00f1fbbf4e4d4e2ad65b70fdac4f911e3612b3956b/source_snapshot_manifest.json`
  - SHA256：`81432f7c1fc6be5d36b547a55b15e3f051b65b94ca4076c5e128e3b6cf313a13`

中途为补齐“v3 activation 必须被拒绝”的显式回归测试而产生的旧候选源码工件已添加
`33d0d422.explicit_v3_rejection_pretest.superseded` 标记并保留，不能授权最终候选源码。

## 6. 最终 preflight 与 DAG 状态

最终 post-activation preflight：

- `status=passed`；
- `blockers=[]`、`infrastructure_blockers=[]`；
- visible/expected GPU=`4/4`；
- `/DATA1` ext4 挂载可写；
- 可用空间约 `1116.96 GiB`，高于冻结的 `35 GiB` 最低启动门限；
- 初始化 checkpoint 和 tokenizer 均完成内容 SHA256 核对；
- 当前进程环境、计划 handler 环境和 required 环境中的 allocator 三者完全一致；
- source snapshot、Docker receipt、activation 和当前源码哈希一致；
- `formal_queue_allowed=true`、`formal_queue_started=false`。

DAG plan：

- schema：`human2robot-m5b-p2-dag-plan-v4`；
- completed=`0`、invalid=`0`、missing=`203`；
- matrix blockers=`[]`；
- 首层 ready cell=`51`（48 个 learned checkpoint + 3 个 nonlearned retrieval-only）；
- v4 manifest 中 48 个 learned training cell 全部为 `pending`，没有 running/completed/failed；
- `nvidia-smi --query-compute-apps` 为空，当前没有 GPU 训练进程。

容器中仍可见上轮 OOM 后由 `sleep infinity` PID 1 未回收的若干 `defunct` 历史进程；它们不是活动
训练进程，不占 GPU 显存或 CPU 执行时间，不构成本次门禁 blocker。正式重试前如果希望清理进程表，
可以另行重建容器，但这不是 memory-successor 的运行语义要求。

## 7. 下一步

当前状态可以进入“只重试第一个正式 cell”的阶段，但本报告没有启动它。下一次得到明确启动指令后，
应只启动：

`learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711`

启动后首先核对 runtime binding 中的 memory-successor SHA256 和 allocator 实测值，再观察是否稳定越过
上次失败的第 3 个 batch，并继续检查 loss、显存曲线、首个 step-1000 四 rank DCP 完整性和可恢复性。
在这些真实训练证据出现之前，allocator 是否足以消除 OOM 仍然是 `NEEDS_EXPERIMENT`。
