# M5B P0 → P2 窗口语义迁移审计

状态：**passed_with_migration_boundary**（非正式实验结果）

P0 与冻结 P2 的窗口语义并不相同。P0 把 `current` 放在向前 pool block 的首行；P2 使用截至当前行的 H 步历史，并只把当前行之后的 K 步作为 query target。

| split | P0 旧语义 | P2 冻结语义 |
|---|---:|---:|
| train | 968 | 954 |
| heldout | 153 | 149 |

首个连续 segment 的例子最清楚：两者都读取 rows 0–7，但 P0 把 row 0 当作 current，并预测 rows 1–8；P2 把 row 7 当作 current，并预测 rows 8–15。因此两套 anchor 没有相同含义。

P0 overfit 证据继续证明正式 2B adapter/action-latent 链路可学习真实 Human2Robot batch，但不能替代 P2 窗口语义验收。已生成的 48-cell prepared inputs 与冻结 P2 语义计数完全一致；正式训练只能使用 `Human2RobotP2Dataset` 和这 48 个 prepared entries。
