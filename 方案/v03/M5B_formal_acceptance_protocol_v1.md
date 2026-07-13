# M5-B-v03 正式验收预注册协议 v1

状态：**已冻结并通过机器校验；实验尚未执行。**

协议 ID：`m5b_v03_preregistered_3seed_formal_v1`

协议 SHA256：`7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4`

本协议定义正式 M4 多 seed 比较和 M5-B 模型依赖型消融。它只冻结未来实验、统计方法和通过条件，不表示任何 M5-B 实验已经完成或通过。

## 1. 研究主张与证据边界

规范化主张：在冻结的 Human2Robot v03 split、pool/query action role、time view 和 alignment 下，RECAP residual bridge 相比 No Retrieval 与 Retrieval Only 改善 held-out robot trajectory prediction，并且该收益得到表示、检索、时间、超参数和分辨率消融支持。

必须降级的主张：

- 主比较未通过：不得声称 RECAP 优于必要 baseline。
- pool-growth 未通过：不得声称新增 human pool 总体改善。
- residual/retrieval 消融未通过：不得把收益归因于 residual 或 retrieval 机制。
- 时间或分辨率实验未通过：不得声称相应鲁棒性或不变性。
- M5 通过也不代表 executable command、物理安全或 M6 rollout 已获批准。

## 2. 数据与任务假设

- Canonical：`human2robot-canonical-hdf5-v3`。
- Frozen split：16 个 train tasks、4 个 held-out tasks。
- 统计单位：`heldout_task × seed`，不是 window/chunk。
- Held-out robot trajectory：只能作为离线 target 评测。
- 正式 pool-growth 要求：每个 held-out task 至少 10 条独立 human demonstrations。
- 独立 demonstration 必须来自不同 source episode；同一 episode 的多个 window 不增加样本量。
- 当前每个 held-out task 只有 1 条独立 demonstration，因此数据状态为 `NEEDS_DATA`。

## 3. 正式模型与训练协议

| 项目 | 冻结值 |
|---|---|
| Model family | Cosmos-Predict2.5 retrieval-conditioned rectified flow |
| Model class | `CosmosPolicyPushTRetModelRectifiedFlow` |
| Base config | `cosmos_predict2p5_2b_480p_pusht_ret_100` |
| Parameter scale | 2B |
| Initialization | `PREDICT2P5_POSTTRAINED_CKPT`；运行前绑定实际 SHA256 |
| Action/proprio dim | 10 / 10 |
| H/K | 8 / 8 |
| Resolution | 224 |
| Main image path | 426→center-crop 424→resize 224 |
| Optimizer | AdamW，lr=`1e-4`，weight decay=`0.1`，betas=`[0.9,0.999]` |
| Steps | 7,000 optimizer steps，无 early stopping |
| Batch | 每个 data-parallel rank 25，gradient accumulation=1 |
| Precision | bf16 |
| Save interval | 每 1,000 steps |
| Primary checkpoint | 固定 step 7,000；不得按 held-out 结果选 checkpoint |

Human2Robot 2B dataset/model adapter 当前尚未实现，状态为 `NEEDS_IMPLEMENTATION`。实现必须输出协议、代码、初始化 checkpoint、数据 view、method、experiment、seed、batch/world-size 和 H/K 的完整绑定；缺少 step-7000 checkpoint 的 run 必须重跑，不能插值或填补。

## 4. 三个冻结 seed

```text
20260711
20260712
20260713
```

每个 seed 必须同时控制 Python、NumPy、Torch CPU/CUDA、distributed sampler、retrieval tie-breaking 和 augmentation。Sampler seed 必须等于 run seed，不得继续使用所有 run 共享的硬编码 `0`。

## 5. Baseline 与公平性

| method | 定义 | learned |
|---|---|---:|
| `no_retrieval` | 同一正式模型，train/eval 都 mask retrieval conditioning，预测 absolute query target | 是 |
| `retrieval_only` | 对齐后的 retrieved human plan 直接投影到 canonical query representation | 否 |
| `co_training` | 同一正式模型，读取 retrieved plan，直接预测 absolute query target | 是 |
| `recap_hand_ret` | 同一正式模型，读取 retrieved plan，预测 aligned plan→query target residual | 是 |

所有 learned 方法必须使用相同模型参数预算、optimizer steps、每 rank batch、data-parallel world size、augmentation 数量和训练样本。各 pool size 必须看到相同的独立 human pool。

## 6. 指标与统计协议

主指标：`position_error_median_canonical`，越低越好。先对每个 held-out task × seed 计算一个 median；window 只用于形成该 median，不能直接进入显著性检验。

次指标：orientation、gripper、final-position、canonical error、residual norm 和 pool-growth slope。

Guardrails：

- non-finite prediction=0；
- gap crossing=0；
- workspace clipping=0；
- held-out target 进入 retrieval feature=0；
- 连续至少 5 步超过 train-only residual-norm P99 的 saturation sequence=0。

统计方法：

- 3 seeds × 4 held-out tasks = 每个方法 12 个 primary units；
- hierarchical paired bootstrap：先采样 task，再在 task 内采样 seed，10,000 次；
- paired randomization test；
- 主比较的两个 baseline 使用 Holm 校正，`alpha=0.05`；
- 必须公开所有 task-level、seed-level 数值、paired difference、95% CI、raw/adjusted p、median 和 IQR；
- 缺失 run 不允许 imputation。

Efficiency 只报告 GPU hours、峰值显存、参数量和单 query latency；本阶段不声称 lightweight/efficient/deployable。

## 7. 最小必需实验矩阵

| ID | 实验 | 冻结主设置 | 必需变体 |
|---|---|---|---|
| `M5B-MAIN-01` | 四方法主比较与 pool-growth | RECAP，pool=0/1/2/4/8/10 | 四个固定方法 |
| `M5B-REP-01` | 输出表示 | residual | residual / absolute / future-state |
| `M5B-ACTION-01` | action view | raw plan + t+1 query | phase plan、lag proxy、same-frame、swapped role、scale×2 |
| `M5B-RET-01` | retrieval modality | phase | random / phase / geometry / visual / geometry+visual |
| `M5B-SENS-01` | top-k 与 H/K | top-k=3，H/K=8/8 | k=1/3/5/10；4/4、8/8、16/8 |
| `M5B-TIME-01` | FPS/time robustness | nominal30 | stride4、stride3、policy10、phase/DTW；drop/jitter/pause/jump 分级 |
| `M5B-RES-01` | visual resolution | crop424→224 | source426、crop424、crop+pad426 |
| `M5B-QUAL-01` | 质性与失败案例 | 每 task 3 best + 3 worst | 预注册失败分类 |

Temporal severity 固定为：frame drop=`5/10/20%`，jitter std=`5/10/20 ms`，pause=`0.2/0.5/1.0 s`，step jump=`1/5/20`。

## 8. M5 完全通过条件

全部条件必须同时通过：

1. 正式 Human2Robot adapter 完成，一批 overfit 与契约测试通过。
2. 每个 held-out task 至少 10 条独立 human demonstrations。
3. 所有 method × experiment × 3 seed 单元都有合法 step-7000 checkpoint。
4. RECAP−No Retrieval 和 RECAP−Retrieval Only 的 primary error 95% CI 上界均小于 0，Holm-adjusted `p<0.05`，且至少 3/4 held-out tasks 改善。
5. pool10−pool0 和误差随 pool size 的拟合 slope 95% CI 上界均小于 0；最多允许一个相邻 pool step 恶化。
6. Residual 分别优于 absolute 与 future-state；phase retrieval 优于 random retrieval；所有负 action controls 被检测并排除。
7. 主 top-k/HK 相对最佳备选不超过 5% 非劣界限。
8. mild drop/jitter 的主指标退化不超过 10%；严重无效输入在 inference 前被检测并 reject/mask。
9. 424 主视觉预处理的 top-k Jaccard≥0.90，主指标相对 426 的退化≤5%。
10. 所有 guardrail count 达到通过值，且完整报告、曲线、失败案例和哈希齐备。

任何一项失败或缺失，M5 和 Gate C 都保持 pending。

## 9. 质性分析和失败案例

每个 held-out task 按预注册主指标选择 3 个最好和 3 个最差 task-seed case。失败分类固定为 wrong retrieval phase、role/alignment mismatch、gripper mismatch、residual saturation、workspace violation 和 temporal discontinuity。

## 10. 当前缺失证据与风险

- `NEEDS_IMPLEMENTATION`：正式 2B Human2Robot dataset/model adapter 与配置。
- `NEEDS_DATA`：每个 held-out task 的 10 条独立 human demonstrations。
- `RUN_BOUND_REQUIRED`：预训练 checkpoint SHA256 和统一 data-parallel world size。
- 只有 4 个 held-out tasks，结论不得外推到冻结 split 之外。
- 3 seeds 只估计优化方差，不能替代 task/demo 层面的独立重复。
- XYZ 物理单位和 executable command 不属于 M5 证据。

## 11. 下一阶段

下一步执行 `M5B-P0-IMPLEMENTATION`：实现正式 2B Human2Robot adapter、config 和一批 overfit/contract tests；同时按 `M5B-P1-DATA` 扩充独立 held-out human-only pool。

机器可读协议：`M5B_formal_acceptance_protocol_v1.json`。

锁文件：`M5B_formal_acceptance_protocol_v1.lock.json`。
