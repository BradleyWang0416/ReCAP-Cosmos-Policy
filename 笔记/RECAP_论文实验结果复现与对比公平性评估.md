# RECAP 论文实验结果复现与对比公平性评估

日期：2026-07-05

本文整理前面对 **Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time** 的三个问答：

1. 当前 PushT 复现实验和论文 Table 2 是否一致。
2. 论文主要对比了哪些方法。
3. 这些对比是否可以看作公平对比。

## 1. 总体结论

当前仓库的 PushT residual RAG 复现结果和论文主表基本对上。

- 论文 Table 2 中 RECAP/Ours 的七个 unseen angle 平均成功率是 **34.9%**。
- 当前 `logs_pusht_rag_eval/residual_top100--*.log` 的七个 unseen angle 平均成功率是 **35.1%**。
- 差值只有 **+0.2 percentage point**，可以看作复现成功。
- seen angle 平均值当前复现是 **53.0%**，论文是 **50.0%**，略高 **+3.0 pp**。
- 单个角度最大差值是 **8 pp**，在 50 trials 设置下等价于 4 个 episode 的波动。

因此，当前结果不表现为系统性跑偏，更像单次 50-trial rollout 下的正常随机波动。

一个注意点：`SUMMARY.txt` 是多个配置的逐 episode 行汇总，缺少 visual config 标签，直接用它肉眼对照容易串列。对论文数值做精确比较时，应以每个 `residual_top100--*.log` 的 `=== FINAL RESULTS ===` 为准。

## 2. 角度映射

论文 Table 2 的角度列是：

| 论文角度 | 当前评测配置 | 当前检索池 |
|---:|---|---|
| -60 | `tri_rot-60` | `rot-60_*` |
| -45 | `tri_goal_flipped` | `goal_flipped_*` |
| -30 | `tri_rot-30` | `rot-30_*` |
| -15 | `tri_rot-15` | `rot-15_*` |
| 0 | `tri_rot0` | `rot0_*` |
| +15 | `tri_rot15` | `rot15_*` |
| +30 | `tri_rot30` | `rot30_*` |
| +45 | `tri_default` | `base_*` |
| +60 | `tri_rot60` | `rot60_*` |

这里最容易错的是两个 seen angle：

- `tri_default` 对应 `base_*`，即 **+45 degree**。
- `tri_goal_flipped` 对应 `goal_flipped_*`，即 **-45 degree**。

## 3. 论文结果与当前复现结果对照

论文 Table 2 的 RECAP/Ours 结果为：

| 角度 | 论文 Ours |
|---:|---:|
| -60 | 28.0 |
| -45 | 40.0 |
| -30 | 44.0 |
| -15 | 26.0 |
| 0 | 28.0 |
| +15 | 36.0 |
| +30 | 48.0 |
| +45 | 60.0 |
| +60 | 34.0 |

当前复现日志的最终结果为：

| 角度 | 当前配置 | 成功数 / trials | 当前复现 | 论文 Ours | 差值 |
|---:|---|---:|---:|---:|---:|
| -60 | `tri_rot-60` | 18 / 50 | 36.0 | 28.0 | +8.0 |
| -45 | `tri_goal_flipped` | 23 / 50 | 46.0 | 40.0 | +6.0 |
| -30 | `tri_rot-30` | 22 / 50 | 44.0 | 44.0 | 0.0 |
| -15 | `tri_rot-15` | 14 / 50 | 28.0 | 26.0 | +2.0 |
| 0 | `tri_rot0` | 12 / 50 | 24.0 | 28.0 | -4.0 |
| +15 | `tri_rot15` | 15 / 50 | 30.0 | 36.0 | -6.0 |
| +30 | `tri_rot30` | 26 / 50 | 52.0 | 48.0 | +4.0 |
| +45 | `tri_default` | 30 / 50 | 60.0 | 60.0 | 0.0 |
| +60 | `tri_rot60` | 16 / 50 | 32.0 | 34.0 | -2.0 |

平均值：

| 指标 | 论文 | 当前复现 | 差值 |
|---|---:|---:|---:|
| Seen avg, -45 and +45 | 50.0 | 53.0 | +3.0 |
| Unseen avg, other seven angles | 34.9 | 35.1 | +0.2 |

解释：

- 当前复现的 unseen average 与论文几乎完全一致。
- 个别角度有上下浮动，例如 -60 高 8 pp、+15 低 6 pp，但没有改变整体结论。
- 50 trials 下每个 episode 对成功率的影响是 2 pp，因此这些差异大多可以理解为 1 到 4 个 episode 级别的波动。

## 4. 论文对比的方法

### 4.1 PushT 主表

PushT Table 2 主要比较：

| 方法 | 含义 | 对比目的 |
|---|---|---|
| `Cosmos Policy` | 无检索的 Cosmos Policy baseline | 检验只靠目标端训练、不使用测试时检索时的泛化能力 |
| `Retrieval Only` | 直接回放最近的来源端检索动作 | 检验“只有检索轨迹、不学习残差修正”是否足够 |
| `Co-train (all)` | 把来源端和目标端数据混合训练 | 检验“把 pool 数据吞进参数”能否替代测试时检索 |
| `RECAP (Ours)` | 检索条件化、残差动作、未来状态预测 | 论文主方法 |

### 4.2 PushT 额外消融

Figure 5 和 Table 3 还做了几类消融：

| 对比 | 说明 |
|---|---|
| `pi0.5 no retrieval` vs `pi0.5 + RAG` | 检验检索是否也能提升普通动作策略 backbone |
| `Cosmos no retrieval` vs `Cosmos + RAG` | 检验检索对 WAM backbone 的提升 |
| `Absolute action` vs `Residual action` | 检验动作参数化是否应该相对 retrieved action 预测 delta |
| `w/o future prediction` vs `w/ future prediction` | 检验未来状态/图像预测目标是否提供额外监督 |
| `RECAP` vs `no-retrieval baseline` 的注意力图 | 检验 L10/L15 等层是否真的读取检索片段 |

核心结论是：检索对 `pi0.5` 和 Cosmos 都有帮助，但在 Cosmos Policy / WAM 上增益更大；未来状态预测只有和 retrieval 结合时才明显有效；残差动作比绝对动作更适合 RECAP。

### 4.3 RoboTwin 主表

RoboTwin Table 1 主要比较：

| 方法 | 含义 | 对比目的 |
|---|---|---|
| `Baseline [7]` | Cosmos Policy，只用目标端 Aloha-Agilex 示范训练，不使用 UR5 pool | 无检索、无 pool baseline |
| `Retrieval Only` | 直接执行最近 UR5 pool 轨迹 | 检验直接回放来源具身动作是否足够 |
| `Co-training` | 目标端与 UR5 pool 轨迹联合训练 | 检验简单跨具身混训是否足够 |
| `RECAP (Ours)` | 检索条件化残差策略 | 论文主方法 |

其中 `[7]` 是 Cosmos Policy，不是另一个独立新算法。RoboTwin 表里最强非本文基线是 `Retrieval Only`，unseen average 为 26.0%，RECAP 为 31.5%。

### 4.4 真实机器人实验

真实机器人实验主要比较：

| 方法 | 含义 |
|---|---|
| `Baseline` | 无检索基线，倾向复现训练任务轨迹 |
| `RECAP (Ours)` | 冻结策略，通过新增人手示范检索池暴露未见任务 |

真实机器人每个任务只有 10 rollouts，因此更像补充验证，不如 PushT / RoboTwin 的统计强。

## 5. 是否可以看作公平对比

结论：可以，但要加限定语。

更准确的说法是：

> 论文的比较在其定义的 **test-time retrieval adaptation protocol** 下是公平的；但它不是对所有可能跨具身适配方法的穷尽式 strongest-baseline 比较。

### 5.1 为什么可以认为公平

| 对比对象 | 公平性判断 | 原因 |
|---|---|---|
| `Cosmos Policy` 无检索 | 公平 | 同 backbone，用来证明 retrieval 是否提供额外泛化能力 |
| `Retrieval Only` | 公平 | 和 RECAP 一样访问检索池，但没有学习目标具身残差 |
| `Co-train / Co-training` | 基本公平 | 用来检验“把来源端数据混入训练”是否能替代外部检索记忆 |
| `pi0.5 + RAG` vs `Cosmos + RAG` | 公平 | 用来证明 retrieval 不是只服务于某一个 backbone |
| 真实机器人 Baseline vs RECAP | 方向公平 | 对比无检索和测试时加入人手示范池的差别 |

这些比较共同服务于论文主张：新任务不通过逐任务重训进入策略，而是通过测试时扩展检索池进入上下文。

### 5.2 为什么不能说是完全充分的最强公平对比

有几条边界需要在汇报里讲清楚：

1. `Cosmos Policy` baseline 没有测试时检索池信息，所以这是“无外部记忆 vs 有外部记忆”的协议对比，不是信息量完全相同的对比。
2. `Co-training` 是相对简单的跨具身数据利用方式，不代表所有 domain adaptation、action translation、embodiment-conditioned policy 或 per-task fine-tuning 方法。
3. `Retrieval Only` 更像机制消融，用来说明直接回放来源端动作不够强，而不是一个经过充分优化的策略学习 baseline。
4. PushT 设置比较干净：来源端和目标端动作空间都可以表示为二维位置动作，残差相加成立，因此对 RECAP 假设更友好。
5. 当前仓库的 PushT live retrieval 是工程化特例，主要用 10 维仿真 GT-state 特征做最近邻，而不是完整实现论文通用描述中的视觉 DINO/SAM/language 检索项。
6. 真实机器人实验 rollouts 少，样本量更适合展示趋势，不适合单独支撑很强统计结论。

## 6. 汇报时建议表述

可以这样说：

> 这组对比是 protocol-fair 的。RECAP 和 baselines 的差别正好对应论文要验证的变量：是否使用测试时检索池、是否直接回放检索动作、是否用简单混训替代外部记忆、是否使用残差动作和未来状态预测。结果说明，在冻结策略、只通过追加来源端示范扩展任务覆盖的设定下，RECAP 优于无检索、检索回放和简单混训。

同时补一句边界：

> 但这不是 strongest-baseline-exhaustive 的比较。更强的跨具身适配方法，例如显式动作翻译、per-task fine-tuning、domain adaptation、embodiment-conditioned policy，仍可能构成更严格的未来对比。

## 7. 对当前复现的最终判断

当前 PushT 复现最能支持的结论是：

- 当前代码和论文 PushT 主线高度一致：检索条件化 Cosmos Policy、残差动作、未来状态预测、冻结模型评测。
- 50-trial 复现实验的 unseen average 为 35.1%，与论文 34.9% 几乎一致。
- seen average 略高，单角度有小幅波动，但没有破坏主结论。
- 因此可以在汇报中说：**PushT 主结果已基本复现，论文 Table 2 的核心数值可信地对齐。**

不应过度声称的是：

- 当前仓库没有完整复现 RoboTwin、真实机器人、人手示范实验。
- 当前仓库没有完整复现注意力探针、L10/L15 遮蔽等机制分析。
- 当前 PushT 检索实现是论文方法在仿真状态可用条件下的工程化特例，不等价于完整真实机器人检索系统。

## 8. 关联文件

- 论文深读笔记：`论文阅读/RECAP/DeepPaperNote - Retrieve_Dont_Retrain_Extending_Vision_Language_Action_Models_to_New_Tasks_at_Test_Time/DeepPaperNote - Retrieve_Dont_Retrain_Extending_Vision_Language_Action_Models_to_New_Tasks_at_Test_Time.md`
- 论文代码对应关系：`笔记/RECAP_论文代码对应关系.md`
- 检索池与检索详解：`笔记/RECAP_检索池与检索详解.md`
- 当前 PushT 复现日志目录：`cosmos_policy/experiments/robot/pusht_ret/logs_pusht_rag_eval/`
- 注意：`SUMMARY.txt` 只适合快速扫逐 episode 结果，严肃对表应使用每个 `residual_top100--*.log` 的最终结果块。
