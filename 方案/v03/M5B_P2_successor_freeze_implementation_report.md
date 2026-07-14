# M5B-P2 successor 七项冻结实施报告

日期：2026-07-14  
状态：**七项 successor 决定已冻结并实现；正式训练/评估尚未启动（0/203），模型结果仍为 `NEEDS_EXPERIMENT`。**

## 1. 结论

本轮只完成协议冻结、数据派生、执行器、报告器、启动/验收门禁和完整 Docker 回归，不签发 launch activation，也不运行任何正式 cell。既有 v1 registry 保持不变，其 SHA256 仍为 `4664d036bcf6bc41e8a44fac2afe04ff6de62c2a180a29d3433bd83e46604df5`；所有修正均进入 v2 successor。

完整环境回归结果为 **138 passed、3 warnings**。刷新后的 `M5B_P2_successor_preflight_20260714.json` 表明：native sampler 语义、203-cell 矩阵、8 GPU、本地权重内容哈希和存储空间均通过；正式队列仍被可写挂载、候选源码快照和 launch activation 三项条件阻断。这是预期的 fail-closed 状态，不是正式实验失败。

## 2. 七项冻结及实现证据

| # | 冻结决定 | 已实施内容 | 主要证据 |
|---|---|---|---|
| 1 | 使用原生 deterministic Rectified Flow，废弃 legacy `2ab` | 固定 guidance=1.5、num_steps=35（34 ODE steps + final clean x0）、shift=5、Karras sigma 开启、variance scale 关闭；preflight 校验正式模型方法签名 | `tools/human2robot_m5b_p2_inference.py`、`cosmos_policy/config/experiment/human2robot_experiment_configs.py`、successor preflight |
| 2 | train-only canonical xyz 全局范围，各轴扩 5%，只计越界、不裁剪 | 仅用 16 个 train episode 计算范围；输出 `workspace_violation_count` 与恒为 0 的 `workspace_clipping_applied_count`；明确不替代 M6 物理安全边界 | `M5B_P2_workspace_bounds_v1.json`，SHA256 `29e0fd8d4b58beabcf7cea7ba50488a0775a79b6f429596a3573a0bbb007eb6a` |
| 3 | 新增注册的第 203 个终止聚合报告 | v2 registry 为 48 learned + 3 nonlearned + 147 evaluation + 5 reports；终止 cell 依赖前 202 个 cell，并覆盖全部 147 个 evaluation | `M5B_P2_cell_registry_v2.json`，SHA256 `502cc57d41c7e4829e872ac95a258d7dc1e8d0d8a27ddfc3cf0315d4d31ef2d6` |
| 4 | offset=5 必须物化为真实 lag view | 3 个 lag seed 均改用独立 t+5 query-anchor view；train=943、heldout=147、gap crossing=0；其余 45 个 prepared entry 与 v1 index/statistics 哈希一致 | lag `view_manifest.json`，SHA256 `53ab59227f865767f07fd4b8c6cea52689b7c22ec1359cedb975308644fe806d`；v2 prepared manifest SHA256 `15a1bd6cc378079b04a821fe691fe293739acc827e183caa44633b76b6a629cd` |
| 5 | temporal corruption 作用于 tokenizer 前的真实 model video | mild 变体修改实际 uint8 `C,T,H,W` 输入后再进入 backend/tokenizer；severe 变体在模型前拒绝，要求 `model_call_count=0`，禁止用 mask 伪装通过 | `tools/human2robot_m5b_p2_inference.py`、`tools/human2robot_m5b_p2_reports.py` |
| 6 | resolution gate 保存逐 query visual ranking | 使用冻结 WAN encoder 对相应分辨率的 query/candidate 视觉输入编码；保存逐 query top-k；主门禁 mean Jaccard >=0.90，同时报告 median/min/identical ratio，指标退化不得超过 5% | resolution inference/report builders 与对应 contract tests |
| 7 | launch activation 与 final acceptance 两阶段拆分 | launch 只开放正式队列，强制 `p2_acceptance_allowed=false`；只有第 203 个终止报告通过且 203/203 完成后才可生成 final acceptance | `M5B_P2_launch_activation_schema_v2.json`，SHA256 `1e2b5f3e245c87d9c9a9f65ffadf2b191ff3dafc20546fb0beecb3e0b23b4ba2`；`M5B_P2_final_acceptance_schema_v2.json`，SHA256 `edb257d7770a2dd0cbd8c7270cef5718afc3fb11411878f6de6eab32b09247eb` |

v2 execution supplement 的 SHA256 为 `17d9fc308c50b9b7899793a4c8d3bca1eeba217053fbacb368e2f9a2e390d7ab`。registry、supplement、workspace bounds 以及两阶段 schema 均配有 lock 文件。

## 3. 验证结果

- 运行环境：完整 `recap_m5b_p1_v2` Docker 容器；未使用宿主机残缺环境，未下载新文件。
- 语法检查：通过。
- Human2Robot 全量测试：`138 passed, 3 warnings in 22.37s`。
- 矩阵：203/203 cell 有显式 handler；147/147 evaluation 被报告层覆盖；语义 blocker 为 0。
- GPU：8/8 可见，通过。
- 权重：初始化 checkpoint 与 tokenizer 均存在且内容 SHA256 匹配，通过。
- 存储：约 417.94 GiB 可用，高于 35 GiB 下限。
- 候选源码：720 个文件，候选代码 SHA256 为 `e14713b3f928c7f592d6096dad6bd0fc4de0d70203dc771a2d6c321f1b9b8487`。
- preflight 文件 SHA256：`57537a76e673e7bca251a8fa26f88aacd684f12ac37845ce3120d5639a9f300a`。

测试中的 3 个 warning 均来自第三方依赖的弃用提示，不是 M5B-P2 contract 或测试失败。

## 4. 当前阻断与后续执行顺序

当前 preflight 精确保留三项 blocker：

1. `formal_output_mount_is_read_only`：现有容器将 `/DATA1` 挂为只读；
2. `candidate_source_snapshot_not_materialized`：不可在只读正式盘上物化候选源码快照；
3. `launch_activation_v2_not_issued`：前两项未满足前不会签发启动凭证。

后续必须按以下顺序继续：

1. 用 `start_m5b_p2_formal_docker.sh` 启动 `/DATA1:/DATA1:rw` 的完整环境；
2. 执行 `tools.human2robot_m5b_p2 prepare`，物化并锁定本报告所列候选源码快照；
3. 执行 `tools.human2robot_m5b_p2_activation run-docker-suite`，在正式盘写入与源码哈希绑定的测试回执；
4. 执行 `tools.human2robot_m5b_p2_activation issue-launch`；只有全部前置通过才会签发 `launch_activation_v2.json`；
5. 再次运行 preflight，要求 blocker 为空且 `formal_queue_allowed=true`；
6. 先生成 DAG plan，再逐 cell 显式启动正式训练、评估与报告；不得绕过单-cell dispatcher；
7. 只有终止聚合 cell 通过后，才运行 `build-final-acceptance` 生成最终验收凭证。

因此，当前可以进入“可写正式容器与启动签发准备”，但不能宣称 M5B-P2 已通过，也不能把任何尚未运行的模型结果写成结论。
