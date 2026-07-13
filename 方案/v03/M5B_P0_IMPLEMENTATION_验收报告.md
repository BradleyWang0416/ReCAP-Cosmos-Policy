# M5B-P0-IMPLEMENTATION 验收报告

日期：2026-07-11T09:39:41.818876+00:00

结论：**M5B-P0-IMPLEMENTATION 通过；M5-B、M5-v03、Gate C 与 M6 均未通过。**

## 完整环境证据

- 仅在项目 Docker `/workspace` 中执行，Torch `2.7.0+cu128`，CUDA `12.8`。
- GPU：`NVIDIA GeForce RTX 4090`，容器可见 8 张。
- 未下载文件、未构建镜像、未同步或降级环境。

## P0 通过项

- 正式 Human2Robot dataset adapter：train `968` windows，held-out `153` windows，严格 t+1、gap crossing=0。
- 正式 model adapter 是现有 retrieval-conditioned rectified-flow 模型的真实子类；3 个 learned methods × 3 seeds 共 `9` 个 2B 配置。
- 配置固定 7,000 optimizer steps、每 rank batch 25、10D action/proprio、H/K=8、37-frame tokenizer chunk。
- 真实 2B 单批 overfit：50 optimizer steps，loss `0.3335` → `0.0039`，下降 `98.83%`，峰值 PyTorch GPU memory `15.40 GiB`。
- 补充 CUDA adapter-I/O 梯度探针：loss `0.0656578317` → `7.63656072e-09`，ratio `1.16e-07`。
- 初始化 checkpoint SHA256：`565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47`。
- tokenizer SHA256：`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。

## 证据边界

- 真实 2B 单批 overfit 固定同一 train window 与 rectified-flow 噪声，并为诊断关闭 warmup；它是实现/连通性测试，不是模型质量证据，也不是 7,000-step 正式训练替代品。
- Human2Robot 尚无冻结的 T5 embedding artifact；P0 使用显式 `disabled_zero_embedding`，没有复用 PushT 文本语义，也没有下载新文件，因此不提出语言条件能力结论。
- 每个 held-out task 当前只有 1 条独立 human demonstration，P1 要求 10 条；正式训练 checkpoint、全实验矩阵与统计门禁仍未完成。
- `query_command_status=unverified`，`deployment_command_adapter_id=null`；禁止真实机器人 rollout。
