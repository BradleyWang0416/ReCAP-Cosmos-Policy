# M5B-P2 日志洪泛修复与 v6 successor 验收报告

日期：2026-07-17  
状态：**修复完成；v6 launch activation 已签发；未启动新训练 cell**

## 1. 结论

正式训练的日志洪泛已修复，并以 `human2robot-m5b-p2-logging-successor-v6` 冻结。

- 正常正式训练固定 `NCCL_DEBUG=WARN`。
- 正常正式训练不设置 `NCCL_DEBUG_SUBSYS`，因此不再输出逐 collective 的 `INFO/COLL` 记录。
- 保留 `TORCH_NCCL_TRACE_BUFFER_SIZE=65536`、`TORCH_NCCL_DUMP_ON_TIMEOUT=1`、`TORCH_NCCL_DESYNC_DEBUG=1` 与慢样本诊断。
- 每次执行尝试写入独立且不可覆盖的 `attempt_0001.log`、`attempt_0002.log` 等文件。
- `latest_log.json` 实时指向当前或最近一次 attempt 日志。
- 旧 v5 日志、checkpoint 和首个正式 cell artifact 均未改写。
- v6 preflight 通过，DAG 仍为 `completed=1, missing=202, invalid=0`，队列尚未启动。

## 2. 原问题与证据

旧日志：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711.log`

观测结果：

| 项目 | 数值 |
|---|---:|
| 文件大小 | 1,324,701,780 bytes |
| 总行数 | 6,956,943 |
| NCCL INFO 行数 | 6,936,345 |
| NCCL INFO 字节数 | 1,322,687,585 |
| NCCL INFO 字节占比 | 99.848% |
| 正式进度行 | 24 |
| 非 NCCL 诊断行 | 399 |

根因是 v5 将 `NCCL_DEBUG=INFO` 与 `NCCL_DEBUG_SUBSYS=COLL` 固定到每个正式训练进程，同时 orchestrator 对同一路径使用追加写入。前者导致每个 rank 输出逐 collective 记录，后者把多次重试混入同一个大文件。

仅去掉 NCCL 洪泛后，旧日志的其余内容约为 2 MB；该值是基于旧日志构成的估算，不是对后续每个 cell 的硬性大小保证。

## 3. 冻结的 v6 日志契约

正常训练环境：

```text
NCCL_DEBUG=WARN
TORCH_NCCL_TRACE_BUFFER_SIZE=65536
TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_DESYNC_DEBUG=1
HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS=5
```

`NCCL_DEBUG_SUBSYS` 必须不存在。`training_command` 会先从继承环境中删除它，再写入冻结的正常训练环境。

逐 collective 的 `INFO/COLL` 仅允许用于单独命名、显式发起、短时运行的诊断 replay；不得用于普通正式 cell。

日志路径契约：

```text
/DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/<cell_id>/
├── attempt_0001.log
├── attempt_0002.log
└── latest_log.json
```

- attempt 文件以 exclusive-create 模式创建；若路径已存在则 fail closed，不追加、不覆盖。
- `latest_log.json` 记录 cell、attempt 编号、日志路径与 running/completed/failed 状态。
- 训练 stdout/stderr 仍实时直写 attempt 文件，可直接 `tail -f`。

下一个计划 cell 启动后，首个日志将是：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260712/attempt_0001.log`

实时指针：

`/DATA1/wxs/ReCAP_M5B_P2_RUNS/orchestrator_logs/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260712/latest_log.json`

## 4. 版本与兼容性

v6 只改变日志诊断和重试日志文件语义，不改变：

- 模型、训练模块或冻结模块；
- 数据、图像行读取、目标、检索排序或增强；
- optimizer、学习率、batch、梯度累积；
- 四卡 world size、FSDP shard；
- seed、7000 step、1000 step checkpoint 间隔；
- 正式评估协议。

首个 v5 完成 cell 以只读方式导入 v6 manifest：

`learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711`

其 artifact SHA256 仍为：

`3ddddef35e3c193ff1ed7ba332c54e164be047cdb46a64ebdf6057bbc79e2197`

v6 manifest 中该 cell 保留其原始 v5 runtime binding、`INFO/COLL` 历史环境与旧 source SHA；其余 47 个 learned cell 使用 v6 source 与 `WARN` 日志契约。旧 `launch_activation_v5.json` 不授权 v6 后续运行。

## 5. 冻结产物

| 产物 | SHA256 |
|---|---|
| `M5B_P2_logging_successor_v6.json` | `ca391656442f18d1881143988f13c51aacbe9a749844c79844527adf6cb55d18` |
| `run_manifest_v6.json` | `484d5f16ee5272d2c830688f505211553292dc42ae9235266e460e4b7d41043d` |
| candidate source | `1fe054a3e1d5d26dfdce9e41859322dd228c98a916e543c6a4143532ee6f467e` |
| source snapshot manifest | `254a6e96484a134c4a787959e7b8569bd8892fdd1ba6ec3fc99a2513b29cf658` |
| Docker suite receipt v6 | `586e403dc7f14d9894d631d389b263ddd6b2a214f196fdd3912e5e4a4750e470` |
| launch activation v6 | `930beb8e75d96ae8653ba1abb9e661e616ccdb26cd4f937a7511db23c472b6ef` |

## 6. Docker 验证

正式容器：

- 用户：`cosmos`
- Python：3.10
- PyTorch：`2.7.0+cu128`
- CUDA runtime：12.8
- 可见 GPU：4
- 网络下载：正式恢复与验证均使用已有本地缓存和本地权重

测试结果：

- 日志/runtime 针对性回归：`40 passed`
- 冻结完整 Human2Robot Docker suite：`161 passed, 3 warnings`
- Docker receipt：`passed`
- source SHA 与 snapshot：一致
- 权重内容哈希：通过
- formal output mount：可写
- preflight blockers：`[]`
- activation：`approved`
- `formal_queue_allowed=true`
- `formal_queue_started=false`

环境完整性说明：一次错误的 root 身份 `uv run` 被立即中止，但已使 `.venv` 暂时切换到不匹配的 PyTorch 2.9.1。随后只修复 `.venv` 所有权，并以正式 `cosmos` 用户、`--offline --frozen` 和已有缓存恢复到锁文件规定的 PyTorch 2.7/cu128；PyTorch 与 Transformer Engine 联合导入及完整 Docker suite 均已通过。

## 7. 当前队列状态

```text
completed = 1
missing   = 202
invalid   = 0
ready     = 51
queue_started = false
```

下一正式训练 cell 仍是：

`learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260712`

本次修复没有启动该 cell。
