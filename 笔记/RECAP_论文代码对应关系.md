# RECAP 论文与当前代码项目对应关系

日期：2026-07-04

## 1. 总体判断

当前仓库 `/home/wxs/ReCAP-Cosmos-Policy` 是论文 **Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time** 的 **PushT + Cosmos Policy / Predict2.5** 方向实现与发布仓库。

它覆盖了论文的核心 PushT 路线：

- 测试时从示范池检索相似片段。
- 把检索到的未来帧、动作块和状态作为世界动作模型条件。
- 模型预测相对检索动作的残差 delta。
- 推理时将 `retrieved_action + predicted_delta` 作为最终动作。
- 在多个 PushT visual config 上评测冻结模型。

但它不是论文全量复现仓库。RoboTwin 2.0、真实机器人、人手示范，以及论文附录里的注意力探针、L10/L15 遮蔽机制分析，在当前仓库中没有完整可运行闭环。

## 2. 论文核心思想到代码主链路

论文的一句话方法是：

> 新任务不通过重新训练模型进入策略，而是通过向检索池追加来源端示范，在每个控制步检索相似轨迹片段，并让冻结策略基于检索片段预测目标端动作残差。

当前代码中的对应执行链路是：

1. `eval_pusht_rag.sh` 指定 checkpoint、数据目录、检索池和评测 visual configs。
2. `cosmos_policy/experiments/robot/pusht_ret/run_eval.py` 创建 PushT 环境和 retrieval 对象。
3. `cosmos_policy/experiments/robot/pusht_ret/retrieval.py` 从 live demo pool 中检索未来帧、动作块和 proprio。
4. `cosmos_policy/experiments/robot/cosmos_utils.py` 把检索内容拼入 Cosmos Policy 的 latent 输入布局。
5. `cosmos_policy/models/policy_video2world_model_pusht_ret.py` 将 retrieved action/state 注入 condition 和 latent state。
6. 模型输出动作 delta。
7. `run_eval.py` 用 `delta_dataset_statistics.json` 反归一化 delta，并加回 raw retrieved action，执行前 `num_open_loop_steps` 步。

## 3. 模块级对应表

| 论文概念 | 代码位置 | 对应说明 |
|---|---|---|
| 测试时检索池 | `eval_pusht_rag.sh`, `DATA_MANIFEST.md` | 评测时不使用预计算 npz，而是从 `success_only/base_*`, `goal_flipped_*`, `rot*_*` live retrieval。 |
| 检索池选择 | `cosmos_policy/experiments/robot/pusht_ret/retrieval.py::resolve_retrieval_split` | 根据 `visual_config`、`goal_angle` 或显式 `retrieval_pool_split` 选择池子。 |
| 当前状态到检索片段 | `retrieval.py::PushTRetrieval.get_retrieved_data` | 用 block/agent 位置、yaw、速度组成 10 维特征，在候选 demo 内找最近邻。 |
| 训练时检索索引 | `cosmos_policy/datasets/pusht_dataset_ret.py::_build_retrieval_lookup` | 读取预计算 `retrieval_results_state_action_*.npz`，将 query step 映射到 source step。 |
| 检索条件序列 | `pusht_dataset_ret.py::__getitem__` | 构造 `blank | ret_frame | ret_state | ret_action | cur_frame | cur_state | pred_action | pred_frame | pred_state`。 |
| retrieved action/state 注入 | `policy_video2world_model_pusht_ret.py::_inject_retrieved_actions`, `_inject_retrieved_state` | 把数值 action/proprio 写入对应 latent slot。 |
| 残差动作训练 | `pusht_dataset_ret.py::__getitem__` | `raw_delta = raw_action_chunk - ret_actions_raw`，再用 delta stats 归一化为监督目标。 |
| 残差动作推理 | `run_eval.py::run_episode` | 模型输出 normalized delta，反归一化后执行 `raw_delta + ret_actions`。 |
| 未来状态预测 | `pusht_experiment_configs.py::pusht_ret_dataset_top100_residual` | `predict_future_states=True`，对应论文中的动作和未来观测联合生成。 |
| 评测入口 | `eval_pusht_rag.sh` | 对 9 个 visual configs 并行评测，输出成功率和视频。 |

## 4. 训练配置与论文设置的关系

README 推荐的主配置是：

- 训练：`cosmos_predict2p5_2b_480p_pusht_ret_top100_residual`
- 推理：`cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only`

关键参数：

- `chunk_size=8`
- `num_open_loop_steps=8`
- `state_t=10`
- `tokenizer.chunk_duration=37`
- `use_residual_actions=True`
- `predict_future_states=True`
- `retrieval_top_k_choice=1`
- `episode_allowlist_top_k=100`

这和论文中 PushT 主线高度对应：检索条件化 Cosmos Policy，残差动作参数化，加未来状态预测目标。

## 5. 数据文件与论文实验的关系

`DATA_MANIFEST.md` 说明了数据发布边界。

评测需要：

- `success_only/base_{0..4}`
- `success_only/goal_flipped_{0..4}`
- `success_only/rot0_{0..4}`
- `success_only/rot15_{0..4}`, `rot-15_{0..4}`
- `success_only/rot30_{0..4}`, `rot-30_{0..4}`
- `success_only/rot60_{0..4}`, `rot-60_{0..4}`
- `t5_embeddings.pkl`
- `dataset_statistics.json`
- `delta_dataset_statistics.json`

重新训练额外需要：

- `base_5`, `goal_flipped_5`
- triangle query splits，例如 `tri_default_predict2*`, `tri_goal_*`
- allowlist JSON
- 两个预计算 retrieval npz：
  - `retrieval_results_state_action_tri_default_p_base.npz`
  - `retrieval_results_state_action_tri_goal_goal_flipped.npz`

重要边界：当前仓库没有带生成这些 retrieval npz 的脚本本体，只保留了数据集读取格式和 manifest 说明。

## 6. 与论文描述的差异

当前 PushT 实现比论文方法描述更工程化、更受控：

- 检索特征主要来自仿真 GT state：block 位置、agent 位置、yaw、速度。
- 没有实现论文通用描述里的 DINO 视觉特征检索、语言目标检索或多对象检索距离。
- PushT 中来源端和目标端是圆形推手与三角形推手，动作空间都是二维位置动作，因此残差相加比较直接。
- RoboTwin 和真实机器人相关代码不是当前仓库的完整 ReCAP 复现重点。
- 注意力探针、L10/L15 intake/commit、单层遮蔽实验没有看到专门脚本。

## 7. 当前仓库能复现什么

最适合复现：

- PushT residual RAG policy。
- 检索池增长或不同 visual config 下的冻结策略表现。
- residual vs absolute、predict future state vs no prediction 等配置消融。
- live retrieval 评测视频：real rollout、generated future、retrieved frames 的并排观察。

不适合直接复现：

- RoboTwin 2.0 全部未见任务结果。
- 真实机器人与人手示范结果。
- 论文附录中的注意力机制解释和遮蔽因果实验。
- 从原始轨迹重新生成 retrieval npz 的完整流程。

## 8. 读代码建议

如果要沿着论文方法读代码，推荐顺序如下：

1. `README.md`：确认仓库定位、下载数据、训练和评测命令。
2. `DATA_MANIFEST.md`：理解哪些数据用于评测，哪些数据用于训练。
3. `eval_pusht_rag.sh`：看实际评测入口和 9 个 visual configs。
4. `cosmos_policy/experiments/robot/pusht_ret/run_eval.py`：看闭环评测、检索调用、残差动作加回。
5. `cosmos_policy/experiments/robot/pusht_ret/retrieval.py`：看 live retrieval 的状态特征和池选择。
6. `cosmos_policy/datasets/pusht_dataset_ret.py`：看训练样本如何构造，以及 delta 监督如何形成。
7. `cosmos_policy/models/policy_video2world_model_pusht_ret.py`：看 retrieved action/state 如何注入模型 latent。
8. `cosmos_policy/config/experiment/pusht_experiment_configs.py`：看主配置和各种消融配置。

## 9. 一句话总结

这个仓库把 RECAP 论文的核心思想落在了 PushT 上：用检索池提供粗粒度未来计划，用 Cosmos Policy 预测目标具身残差，并通过冻结模型加 live retrieval 的方式扩展到新的目标角度和视觉配置。
