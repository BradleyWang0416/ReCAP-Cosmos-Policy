# M5B-P2 正式训练前就绪报告

日期：2026-07-14  
状态：**READY_TO_START_FORMAL_TRAINING；正式队列已授权但尚未启动，0/203 cell 完成。**

> **运行时 successor 提示（2026-07-14）**：本报告正文的容量与运行时数字基于已撤销
> 启动效力的八卡 v2。当前授权已迁移到四卡 v3：`world_size=DP=fsdp_shard_size=4`、
> batch/rank=25、gradient accumulation=2、有效全局 batch=200。当前证据与启动方式以
> `M5B_P2_4GPU_successor_冻结与启动前验收报告_20260714.md` 为准。

## 1. 结论

正式训练开始前的全部步骤已经完成。最终 preflight 与 DAG plan 均满足：

- `status=passed`；
- `blockers=[]`、`matrix_blockers=[]`；
- `formal_queue_allowed=true`；
- `formal_queue_started=false`；
- 203 个 cell 中 completed=0、invalid=0、missing=203；
- 51 个无父依赖根 cell 已 ready；
- `p2_acceptance_allowed=false`，M6 rollout 仍禁止。

这表示可以开始显式执行第一个正式训练 cell，但不表示 M5B-P2 已通过。全部尚未运行的模型指标继续标记为 `NEEDS_EXPERIMENT`。

## 2. 已完成步骤与冻结证据

| 步骤 | 结果 | 正式证据 |
|---|---|---|
| 可写正式 Docker | `recap_m5b_p2_formal` 正常运行；输出根固定为 `/DATA1/wxs/ReCAP_M5B_P2_RUNS`；离线/禁下载/W&B disabled | 最终 preflight 中 mount=`rw`、8/8 GPU、offline env passed |
| immutable source snapshot | 731 个候选源码文件已物化 | code SHA256 `a11d63d52944b718279ab4454371efcfc111a536e1299bdf4f3b74bcab565bec`；snapshot manifest SHA256 `fa4a17426f65947df3674a0d88c74bde5f944b4df3586405f18391e41351cc9e` |
| 正式 run manifest | 与最终源码、协议、203-cell registry 和本地权重绑定 | SHA256 `82528287a2de8a1f9419cde6adb29db6e8f991c9c277e4c4a9d5ce56c319f6ce` |
| Docker full suite | 141 passed、3 个第三方弃用 warning；无测试失败；未执行 cell | receipt SHA256 `697180f8f0cff11cb9d4d42f3c79ab295a477ded52cbcf607d8988cd81b9823d` |
| launch activation | approved；开放队列但不开放最终验收 | SHA256 `33d491f81cd4ab825a03728de15cc5709f986c87161f850b9089bd163e8e4682` |
| post-activation preflight | passed；blocker=0；queue allowed；queue not started | SHA256 `36bec9e69dbe1060072786e5699da7dab7dabf9f20aaa90da0db5d3a12b617bb` |
| DAG plan | queue allowed；queue not started；0/203；51 root-ready | SHA256 `9dc6ada439d285979849f2638b9dec2740e090c3e2521c82a8cc97e6d1a0131c` |

初始化 checkpoint 与 tokenizer 的内容 SHA256 分别验证为：

- `565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47`；
- `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。

正式盘剩余约 417.90 GiB，高于 35 GiB 冻结下限。准备过程未下载任何文件。

## 3. 启动门禁修复

首次 post-activation 验证发现 preflight 与 DAG plan 将 `formal_queue_allowed` 硬编码为 false。该问题已修复并新增 fail-closed 约束：

1. 仅当 activation 已批准且 blocker 为空时，preflight/plan 才返回 queue allowed；
2. activation 必须绑定当前候选代码 SHA256、对应 snapshot manifest 和 Docker suite receipt；
3. snapshot 严格核对 schema、代码 SHA256 和完整文件表，仅允许物化器增加审计时间戳；
4. 任意源码、snapshot 或 receipt 漂移都会在 plan/run-cell 前拒绝执行。

与中间源码哈希 `638c189b...`、`cad81848...` 绑定的产物均保留在 `/DATA1/wxs/ReCAP_M5B_P2_RUNS/superseded/`，不得用于正式执行。最终 active activation 只绑定 `a11d63d5...`。

## 4. 当前运行状态与边界

- `recap_m5b_p2_formal` 内仅有保活进程 `sleep infinity`，无 Python/torchrun/训练进程；
- 没有执行 `human2robot_m5b_p2_dag run-cell`；
- 没有 checkpoint、evaluation 或 report 被计为完成；
- launch activation 不是 final acceptance；
- 第 203 个终止报告通过前，不得生成 `final_acceptance_v2.json` 或声称 M5B-P2/M5-v03 通过。

下一步已经从“准备”切换为“正式执行”：按 DAG 逐个显式启动 root cell，并保持单-cell dispatcher、日志、artifact 和断点续跑检查。
