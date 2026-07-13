# M5B-P2 正式运行前完整性审计

日期：2026-07-13  
状态：**EXECUTION SPEC/REGISTRY 已冻结；NEEDS_FULL_IMPLEMENTATION；正式队列关闭。**

## 1. 结论

目前仍不能启动“完整 M5B-P2”正式队列。现有代码已经具备
`M5B-MAIN-01` 中三个 learned method × 三个冻结 seed 的 9 个 2B 训练单元，
且用户已授权冻结最小 claim-centered 执行补充规范与 202-cell registry；但其余 7 个
实验族、`retrieval_only` 非学习基线、held-out 正式推理、统计与质性导出的 handler
尚未形成可执行闭环。

因此：

- P2、M5-v03、Gate C 均保持 `pending`；
- step-1 容量探针只证明 8 GPU、每 rank batch=25 的配置可运行，不是正式实验结果；
- supplement/registry 已冻结并通过 Docker 锁链验证，但在 202 个 cell 全部有 handler、
  契约测试和全矩阵 smoke 前，不打开正式队列；
- 不允许对缺失的 method × experiment × seed 单元插值、复用后冒充独立 checkpoint，
  或用 host/不完整环境替代 Docker 正式运行。

## 2. 已具备的运行基础

| 项目 | 当前证据 | 状态 |
|---|---|---|
| 冻结 seed | `20260711/20260712/20260713` | 已绑定 |
| 正式主配置 | 3 learned methods × 3 seeds = 9 | 已实现 |
| 模型与 batch 容量 | 8 GPU、DP=8、每 rank batch=25、真实 2B、step-1 完成 | 仅 preflight |
| 训练预算 | 7,000 optimizer steps、每 1,000 step 保存 | 已写入配置与运行时门禁 |
| 离线运行 | 本地 checkpoint/tokenizer；HF/Transformers offline；W&B disabled | 已写入运行环境 |
| checkpoint 验收 | model/optim/scheduler/trainer 的 8-rank DCP、step-7000 与哈希 | 主训练审计器已通过 Docker 契约测试 |
| 防重复运行 | 固定 8 GPU 全局非阻塞文件锁 | 已通过 Docker 契约测试 |
| 防过度宣称 | 局部队列改名并要求显式确认；完整 P2 永不由 9 单元自动判定通过 | 已通过 Docker 契约测试 |
| 执行补充规范 | `M5B_P2_execution_supplement_v1.json` + lock | 已冻结并验证 |
| 完整 cell registry | 48 learned + 3 nonlearned + 147 eval + 4 report = 202 | 已冻结为 pending，队列关闭 |
| Docker 回归 | Human2Robot 全回归 | 62 passed；1 个 Megatron deprecation warning |
| P2 定向契约 | adapter/config/orchestrator/registry | 18 passed；后续锁链接入定向集 18 passed |
| 存储 | `/DATA1` 运行前可用约 535 GiB | 满足冻结的单次启动下限 35 GiB |

## 3. 冻结矩阵与实现覆盖

| 冻结实验 | 当前可执行性 | 缺口 |
|---|---|---|
| `M5B-MAIN-01` | 部分 | 只有三个 learned method 的主设置训练；pool-growth、`retrieval_only` 与 held-out 评测未接入 |
| `M5B-REP-01` | 否 | 缺少与 absolute future query 明确区分的 `future_state` target/loss/decoder |
| `M5B-ACTION-01` | 否 | adapter 只接受冻结主 action view；其他 action view 与负控制未物化到 loader |
| `M5B-RET-01` | 否 | P1 human-only pool 未被正式 adapter/检索器读取；geometry/visual 特征和 encoder 未冻结 |
| `M5B-SENS-01` | 否 | adapter 硬限制 H=8、K=8；无 top-k 聚合路径 |
| `M5B-TIME-01` | 否 | adapter 只接受 nominal30 主 view；无模型级 perturb/reject/mask runner |
| `M5B-RES-01` | 否 | loader 只接受 224 主预处理；无三种 source/crop/pad 变体与共同 encoder 绑定 |
| `M5B-QUAL-01` | 否 | 无正式 task-seed 指标表、best/worst 选择器和失败分类导出器 |

## 4. 代码事实

1. `Human2RobotFormalDataset` 的构造器硬要求 H=8、window stride=8、最终输入 224，
   `_validate_parent_contract()` 又硬绑定 nominal30、raw human plan、t+1 query 和主 alignment。
2. `__getitem__()` 从同一 canonical paired episode 同时读取 human 与 robot trajectory；
   它没有读取 `p1_human_only_pool/pool_manifest.json`，因此当前不能证明 pool-growth 或
   跨 episode held-out retrieval。
3. `_normalized_targets()` 只实现 residual、absolute 与非学习占位分支；没有独立的
   `future_state` 表示协议。
4. 当前正式 config 生成器只生成 9 个 learned-method 配置，并关闭 validation；项目中没有
   Human2Robot 正式 held-out inference/evaluator。

## 5. 已冻结的执行补充决策

以下内容已在任何正式 P2 结果产生前冻结：

1. 每个实验变体是重新训练还是复用主 checkpoint 做 eval-only，以及完整 cell ID 枚举；
2. 非学习 `retrieval_only` 用什么不可变 artifact 替代逻辑上不存在的 optimizer checkpoint；
3. `future_state` 的 target、loss、decoder、normalization 与 canonical trajectory 映射；
4. random/phase/geometry/visual 的特征、冻结 encoder SHA、index、top-k 聚合与 tie-break；
5. action/time/HK/resolution 变体的数据物化与 train/eval 规则；
6. held-out 推理、task × seed 聚合、guardrail、统计和质性导出的文件 schema。

其中 `retrieval_only` 明确使用不可变 nonlearned artifact，不要求不存在的 optimizer
checkpoint；learned 变体仍严格要求 step-7000。该消歧已写入 supplement 和 registry，
不允许运行脚本动态更改。

## 6. 下一执行顺序

1. 实现 P1 pool-backed retrieval 与正式 held-out evaluator；
2. 实现并验证 REP/ACTION/RET/SENS/TIME/RES/QUAL 变体；
3. 为 202 个 registry cell 建立 fail-closed handler/report builder 与 artifact schema；
4. Docker 中跑小规模、不计入正式证据的全矩阵 smoke；
5. 生成不可变 source snapshot 和主 manifest，并再次核验 8 GPU 与存储下限；
6. 按 registry 拓扑顺序运行所有 3-seed 正式单元，逐单元验收 step-7000 或其冻结 artifact；
7. 生成完整统计与 P2 验收报告。任何缺失单元保持 `pending`，不 impute。

## 7. Docker 与存储检查结果

用户已明确授权，检查均在现有 `recap_m5b_p1_v2` 完整 Docker 环境内完成；没有下载、
构建镜像或同步依赖，也没有使用 host Python 替代。目标契约集为 `18 passed`，完整
Human2Robot 回归为 `62 passed`。存储只读检查显示 `/workspace` 约 216 GiB 可用、`/DATA1`
约 535 GiB 可用。现有真实 2B step-1 容量探针约 11 GiB，仅作为容量证据，不作为正式结果。

## 8. 冻结 artifact 与哈希

- supplement：`M5B_P2_execution_supplement_v1.json`，SHA256
  `be6ca3cdeb7d725221cbefa4664a44f33531edea1b66a74ea2405bff54dfc4ba`；
- supplement lock：SHA256
  `bdea172ada310f421c2398c57eb9536a8636ca3ba51302932de2e7b589fcff77`；
- registry：`M5B_P2_cell_registry_v1.json`，SHA256
  `4664d036bcf6bc41e8a44fac2afe04ff6de62c2a180a29d3433bd83e46604df5`；
- registry lock：SHA256
  `a349b799e0364945a438d400cf74348ee9ba8c300a72e8f9cdb6b4202e9c3fba`；
- registry cells payload：SHA256
  `6945695513ecf2bacbebf3f7af8cb476d0c89e073f32533098eb065c9ce486bf`。

上述 artifact 只冻结未来执行语义和单元拓扑，不包含实验结果，不通过 P2。当前
`formal_queue_allowed=false`、`p2_acceptance_allowed=false`。
