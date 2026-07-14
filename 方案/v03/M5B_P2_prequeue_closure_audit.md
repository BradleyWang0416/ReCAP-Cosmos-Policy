# M5B-P2 正式队列前执行/评估闭环审计

日期：2026-07-14（Asia/Shanghai）

状态：**实现闭环已大幅补齐，但正式队列仍必须保持关闭。**

机器可读的最终只读检查摘要见 `M5B_P2_prequeue_preflight_20260714.json`。

## 已完成

- 48 个 learned checkpoint spec 与 48 个 prepared entry 一一对应；prepared manifest、registry、protocol、supplement、split、pool、tokenizer 哈希全部一致。
- 冻结 202-cell registry 已形成无环 DAG，并为 48 training、3 nonlearned、147 evaluation、4 report cell 分配唯一 handler。
- 原 orchestrator 已从 9 个 main cell 升级为 48 个冻结 training cell；variant、H/K、top-k、pool、retrieval modality、time view、query offset、representation 都进入 runtime binding。
- held-out 评估层已实现 deterministic per-rank seed、inverse normalization、residual/absolute/future-state/retrieval-only 重建、canonical projection、top-k 等权聚合和 task×seed 汇总。
- 统计层已实现 10,000 次 hierarchical paired bootstrap、精确 `2^12` one-sided sign-flip、两项主比较 Holm 校正以及 no-imputation 检查。
- 4 个注册 QUAL report builder 和额外的 202-cell/147-evaluation completion report builder 已实现。
- 3 个 `retrieval_only` cell 已有真正的 nonlearned artifact materializer：绑定 code、pool、feature/index、alignment、projection、seed tie-break、逐 rank provenance、逐窗口输出和 payload SHA256；其 linked evaluation 必须复核父 payload，不能伪装 checkpoint。
- P0→P2 窗口迁移已审计：train 968→954、heldout 153→149；48-cell prepared 数据遵循冻结 P2 语义。
- 202 条 handler command 已逐条解析，48 个训练 config 全部解析到唯一冻结 spec；eval command 显式绑定 workspace、artifact root、activation 和 workspace bounds，并统一携带 offline/no-download 环境。
- 训练 audit 现在会额外生成统一 registry artifact，固定 `checkpoint_path` 与 `model_payload_sha256`，147 个 evaluator 不再依赖不匹配的 `primary_checkpoint_*` 私有字段。
- 202-cell DAG 提供 inventory、plan 和显式单-cell dispatcher；无 `run all` 命令，所有 subprocess 之前再次验证 activation 和 parent artifact。
- 完整 Human2Robot/M5B Docker 非正式回归为 **127 passed**，新增模块 `py_compile` 通过；没有启动正式训练。
- 全量只读 preflight 已验证：容器内 8/8 GPU、正式盘约 504 GiB 可用、初始化 checkpoint/tokenizer 完整内容 SHA256 与冻结值一致、48 prepared entries 与 202 handlers 完整。

## 仍然阻塞正式队列的问题

1. 冻结 registry 本身仍声明 `formal_queue_allowed=false`、`p2_acceptance_allowed=false`，没有冻结从 pending 到 activated 的状态转换工件。
2. 冻结推理指定 `solver=2ab`，但实际 2B Human2Robot 模型是 rectified flow，方法签名没有 `solver_option`，当前会被 `**kwargs` 静默吞掉；runner 现已在加载模型前硬失败。
3. `workspace_clipping_count=0` 没有冻结 canonical xyz workspace bounds，无法计算或验收。
4. 4 个冻结 QUAL reports 只传递覆盖 27/147 个 evaluation；全矩阵 completion report 虽已实现，但尚未进入冻结 registry/supplement。
5. lag-calibrated variant 使用 offset 5，却仍绑定 t+1 view manifest。
6. temporal corruption 只有名称/强度，缺少精确 model-input transform、mask/reject threshold 和 mild/severe materialization。
7. G6 要求 visual top-k Jaccard，但 RES cells 复用 phase retrieval parent，没有注册 per-resolution visual ranking evidence，也没有冻结 Jaccard aggregation rule。
8. runbook/现有容器把 `/DATA1` 挂为 `ro`，而冻结正式输出根是 `/DATA1/wxs/ReCAP_M5B_P2_RUNS`；因此正式 checkpoint、cell artifact 和 source snapshot 均不可写。
9. 候选 source manifest 已覆盖新增 runtime modules，候选 code SHA256 为 `1fbf1bd9e81d3ba2d39934ceb699f3322c5500f56da88ca6b5d5a877374db5d8`，但在可写正式根出现前尚未物化和锁定，不能写成 `source_snapshot_frozen=true`。

这些不是训练结果可以补救的问题；若直接开队列，会产生无法按冻结协议验收的昂贵 checkpoint。

## 距离最终目标

- 工程实现层：核心 DAG、训练绑定、评估数学和报告框架已具备，剩余主要是上述冻结契约修正后的具体 materializer 适配。
- 运行基础层：GPU、容量、本地权重和 handler offline 环境已通过；正式输出挂载和 immutable source snapshot 未通过。
- 正式 M5B 证据层：仍为 **0/202 formal cells completed**；prepared 48/48 只是非正式输入工件，不是训练结果。
- 最终研究目标层：即使 M5B 全部通过，仍需 Gate C 审查、M6 deployment command adapter/clock/latency/安全标定、真实机器人 rollout 与 Gate D，才能支持“新增人手示范指导真实机器人复现未见任务”的最终结论。

## 下一步

先审阅并批准或修改 `M5B_P2_execution_correction_v2.proposed.json` 的 7 项决定；随后生成 successor supplement/registry（或独立 completion/activation lock）。正式启动容器必须把 `/DATA1` 以可写方式挂载，在启动任何 cell 前物化并复核 candidate source snapshot，再重跑同一全量 preflight。202-handler smoke、8-GPU、存储和权重哈希已经通过，不需要重做“残次环境”替代实验；只有 mount、snapshot 和 7 项 successor 契约全部通过，才允许生成 formal activation artifact。

## 只读 preflight 摘要

| 检查 | 结果 |
|---|---|
| Docker | passed；现有完整容器 `recap_m5b_p1_v2` |
| GPU | passed；8/8 visible |
| `/DATA1` free space | passed；约 504 GiB，阈值 35 GiB |
| 初始化 checkpoint | passed；4,118,575,006 bytes，SHA256 `565bbb2c...ff47` |
| tokenizer | passed；507,609,880 bytes，SHA256 `38071ab5...981` |
| prepared/handler | passed；48/48 prepared，202/202 handlers |
| Docker regression | passed；127 tests |
| 正式输出 mount | **blocked**；`/DATA1` 为 `ro` |
| source snapshot | **pending**；candidate 已算 hash，尚未物化 |
| 正式队列 | **closed**；`formal_queue_started=false` |
