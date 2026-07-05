# RECAP 检索池与检索代码详解

日期：2026-07-04

本文是在 `RECAP_论文代码对应关系.md` 的基础上，专门展开 **检索池**、**检索规则**、**检索条件如何进入 Cosmos Policy**，以及它们和论文公式的对应关系。

主要参考：

- 论文：Retrieve, Don't Retrain: Extending Vision-Language-Action Models to New Tasks at Test Time, arXiv:2606.15631, https://arxiv.org/abs/2606.15631
- 项目说明：`README.md`, `DATA_MANIFEST.md`
- 代码主线：`eval_pusht_rag.sh`, `cosmos_policy/experiments/robot/pusht_ret/retrieval.py`, `cosmos_policy/experiments/robot/pusht_ret/run_eval.py`, `cosmos_policy/datasets/pusht_dataset_ret.py`, `cosmos_policy/experiments/robot/cosmos_utils.py`, `cosmos_policy/models/policy_video2world_model_pusht_ret.py`

## 0. 先给结论

当前仓库实现的是论文 RECAP 在 PushT 上的一个工程化特例：

- 论文中的 **query side** 是目标具身，PushT 里对应三角推手 `tri_*`。
- 论文中的 **pool side** 是便宜来源具身，PushT 里对应圆形推手或旋转目标池 `base_*`, `goal_flipped_*`, `rot*_*`。
- 论文中的主动检索池 `D_pool` 在评测时不是一个向量数据库文件，而是 `success_only/<pool>_<shard>/...hdf5` 这些 HDF5 demo 目录。
- 评测时每个控制 chunk 重新 live retrieval：从 pool 中找最相近的子帧 `t'`，取从 `t'` 开始的未来帧、动作块、proprio，喂给冻结的 Cosmos Policy。
- 训练时不 live 计算最近邻，而是读取预计算好的 retrieval `.npz`；`.npz` 存的是 query step 到 pool step 的 best-first 匹配列表。
- 论文 Eq. (5)-(8) 的两阶段检索，在当前 PushT live eval 中被简化为：
  - Stage 1：按初始 block 位置选 top-100 demo；
  - Stage 2：在这些 demo 的 subframes 上做 10 维 GT-state 特征的平方 L2 最近邻；
  - 没有 DINO/SAM/language 视觉检索项，也没有 inference-time action term。
- 因此“最 match 的 demo”不是靠视觉 embedding 找的，而是靠仿真 GT pose / proprio 状态：block 位置、agent 位置、block 姿态角，以及 block/agent 的短时速度。
- 论文 Eq. (3) 的 residual action 是当前仓库最忠实的一条公式对应：训练预测 `delta = target_action - retrieved_action`，推理执行 `retrieved_action + predicted_delta`。

## 1. 论文里的符号先翻译成代码变量

| 论文符号 | 直觉 | 当前 PushT 代码里的对应 |
|---|---|---|
| `D_pool` | 活跃检索池 | 评测时 `cfg.retrieval_data_dir` 下被选中的 pool dirs；训练时 `retrieval_source_splits` 加预计算 `.npz` |
| `s_t^query` | 当前目标具身状态 | 评测 `run_episode()` 里 env 当前 block pose、agent proprio、angle、历史速度 |
| `s_{t'}^pool` | pool 里被匹配到的状态 | `retrieval.py` 中某个 subframe 的 `feat` / `t_last` |
| `a_{t':t'+H}^pool` | 检索到的 pool 动作块 | `ret_actions`, `retrieved_actions` |
| `H` | action horizon / chunk | `chunk_size=8` |
| `K` | 每次执行的 open-loop stride | `num_open_loop_steps=8`，当前主评测等于 `H` |
| `t'` | 检索出来的 pool 子帧索引 | `sf["t_last"]` 或训练 `.npz` 的 `m_start` |
| `r_t` | 检索上下文 | `retrieved_frames`, `retrieved_actions`, `retrieved_proprio` |
| $\Delta a$ | 目标动作相对检索动作的残差 | dataset 中 `raw_delta`；eval 中 `norm_delta -> raw_delta` |

## 2. 检索池：评测时的 live pool

### 2.1 pool 从哪里来

`eval_pusht_rag.sh` 将数据目录设成：

```bash
DATA="$BASE_DATASETS_DIR/PushT-Cosmos-Policy/success_only"
```

然后给 `run_eval.py` 传：

```bash
--retrieval_data_dir "$DATA"
--visual_config "$V"
```

`DATA_MANIFEST.md` 说明了评测需要的 live retrieval pools：

| visual_config | 默认 pool dirs |
|---|---|
| `tri_default` | `base_0` 到 `base_4` |
| `tri_goal_flipped` | `goal_flipped_0` 到 `goal_flipped_4` |
| `tri_rot0` | `rot0_0` 到 `rot0_4` |
| `tri_rot15` / `tri_rot-15` | `rot15_*` / `rot-15_*` |
| `tri_rot30` / `tri_rot-30` | `rot30_*` / `rot-30_*` |
| `tri_rot60` / `tri_rot-60` | `rot60_*` / `rot-60_*` |

代码中的默认映射在 `run_eval.py::RETRIEVAL_SPLIT_MAP`：

```python
"tri_default":      [f"base_{i}" for i in range(5)]
"tri_goal_flipped": [f"goal_flipped_{i}" for i in range(5)]
"tri_rot0":         [f"rot0_{i}" for i in range(5)]
...
```

这就是论文中 “test-time pool holds demonstrations and can be extended without retraining” 的代码落点：评测换任务时，模型参数不变，只换 `retrieval_split` 指向哪些 pool dirs。

### 2.2 `resolve_retrieval_split()` 如何选池

入口在 `run_eval.py::eval_pusht_ret()`：

```python
retrieval_split = resolve_retrieval_split(
    data_dir=cfg.retrieval_data_dir,
    visual_config=cfg.visual_config,
    goal_angle=cfg.goal_angle,
    explicit_split=cfg.retrieval_pool_split,
    k=cfg.retrieval_pool_k,
    fallback_map=RETRIEVAL_SPLIT_MAP,
    pool_pattern=cfg.retrieval_pool_pattern,
)
```

`retrieval.py::resolve_retrieval_split()` 的优先级是：

1. `--retrieval_pool_split` 显式指定：如 `rot30_0,rot30_1,rot60_0`。
2. `--goal_angle` 指定目标角：扫描 `data_dir` 下符合 `rot{angle}_{id}` 的目录，取角度最近的 K 组；若存在几乎精确匹配，则只取这一组。
3. 回退到 `RETRIEVAL_SPLIT_MAP[visual_config]`。

有两个实现细节值得注意：

- `_POOL_PATTERNS` 支持 `plain`, `flipcolor`, `both`，默认只识别 `rot(-?\d+)_(\d+)`。
- 显式 pool 中如果有 `base_*` 或 `goal_flipped_*`，它们会被 `_NON_ROT_POOL_ANGLES` 当作固定角度：`base -> 45`, `goal_flipped -> -45`。这用于“累计 pool”或“邻近角 pool”实验。

所以代码里的“检索池扩展”可以有两层含义：

- 数据层扩展：往 `success_only/` 增加新的 pool dirs。
- 运行层扩展：用 `retrieval_pool_split` 或 `goal_angle + retrieval_pool_k` 控制活跃 pool 子集。

### 2.3 `PushTRetrieval._load_pool()` 读入什么

`PushTRetrieval.__init__()` 会调用 `_load_pool(data_dir)`。它从选中的 HDF5 文件中读取：

- `obs/images`：用于返回 `retrieved_frames`。
- `actions`：用于返回 `ret_actions`，最终 residual add-back。
- `obs/states[:, :2]`：agent xy，作为 retrieved proprio。
- `obs/states[:, 2:4]`：block xy。
- `obs/states[:, 4]`：block yaw。

`_base_data[(suite, demo_key)]` 保存完整 demo：

```python
{"images": imgs, "actions": acts, "proprio": prop}
```

而真正用于最近邻的是一个 subframe 列表 `self._subframes`。每个 subframe 包含：

```python
{
    "split": suite,
    "demo": demo_key,
    "start": t,
    "t_last": t_last,
    "block_x0": initial block x,
    "block_y0": initial block y,
    "end_x": block x at t_last,
    "end_y": block y at t_last,
    "feat": 10-dim retrieval feature,
}
```

subframe 有两类：

- early subframes：`t_last=0..WINDOW_SIZE-2`，用于覆盖 episode 开头。
- sliding windows：`WINDOW_SIZE=8`, `STRIDE=2`，每 2 帧建一个窗口，`t_last=t+7`。

加载完成后还会构造三个索引：

- `self._feat`: shape `(N, 10)`，所有 subframe 的检索特征。
- `self._demo_indices`: `(split, demo)` 到 subframe index 列表。
- `self._demo_init_block_pos`: 每个 demo 的初始 block 位置，用于 Stage 1 预筛。

## 3. 检索特征：10 维状态向量

论文 Eq. (6)/(8) 写的是多组件距离：object pose、proprio、DINO image feature、训练时 action chunk。当前 PushT live eval 把这件事压成了一个 10 维 GT-state 特征。

`PushTRetrieval._build_feature()` 和 `get_retrieved_data()` 都用同一套构造：

```text
q = [
  W_BLOCK_POS * block_x / 512,
  W_BLOCK_POS * block_y / 512,
  W_AGENT_POS * agent_x / 512,
  W_AGENT_POS * agent_y / 512,
  W_YAW * sin(yaw),
  W_YAW * cos(yaw),
  W_BLOCK_VEL * block_vx,
  W_BLOCK_VEL * block_vy,
  W_AGENT_VEL * agent_vx,
  W_AGENT_VEL * agent_vy,
]
```

默认权重在 `retrieval.py::PushTRetrieval` 类常量里：

```python
SIM_SCALE   = 512.0
W_BLOCK_POS = 2.0
W_AGENT_POS = 2.5
W_YAW       = 1.5
W_BLOCK_VEL = 1.0
W_AGENT_VEL = 1.0
VEL_WINDOW  = 2
```

速度项是最近 `VEL_WINDOW + 1 = 3` 个位置的平均有限差分。评测时 `run_episode()` 用 `block_pos_history` 和 `agent_pos_history` 维护这段历史。

`block_rel=True` 时，agent xy 不用世界坐标，而是先转到 block 局部坐标系；默认 `block_rel=False`。

从论文符号看，这个 10 维向量大致对应：

```text
phi_obj  = block position + block yaw + block velocity
phi_prop = agent position + agent velocity
phi_vis  = not used in this PushT live retrieval path
phi_act  = not used at inference
```

也就是说，代码将论文中的加权距离：

$$
\begin{aligned}
d_{\mathrm{inf}}(t,t') =
&\, w_{\mathrm{obj}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{obj}}^t, \phi_{\mathrm{obj}}^{t'}\right)
 + w_{\mathrm{prop}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{prop}}^t, \phi_{\mathrm{prop}}^{t'}\right) \\
&+ w_{\mathrm{vis}} d_{\cos}\!\left(\phi_{\mathrm{vis}}^t, \phi_{\mathrm{vis}}^{t'}\right)
\end{aligned}
$$

实现成：

$$
d_{\mathrm{code}}(t,t') = \left\lVert q_t - q_{t'} \right\rVert_2^2
$$

其中各个 `w_*` 已经乘进 `q` 的分量里。因此代码的平方 L2 实际等价于“分量加权后的 L2”，但权重经过平方进入最终距离。例如 `W_AGENT_POS=2.5` 会在平方距离里产生 `6.25` 的相对权重。

### 3.1 “最 match 的 demo”到底看什么

一句话：当前 PushT live eval 里，检索最相似 demo / subframe 的依据是 **10 维 GT-state 特征的平方 L2 距离**，不是视觉图像相似度。

query 端的状态来自 `run_episode()` 当前环境：

- `unwrapped.block.position`：当前 block xy；
- `unwrapped.block.angle`：当前 block yaw；
- `observation["proprio"]`：当前 agent xy；
- `block_pos_history` / `agent_pos_history`：最近几帧位置，用来算速度。

pool 端的状态来自 HDF5 的 `obs/states`：

```text
obs/states = [agent_x, agent_y, block_x, block_y, block_angle]
```

两边都被转成同一套 10 维向量后，再计算：

```python
dists = ((self._feat[sub_idx] - q_feat) ** 2).sum(axis=1)
best_i = int(sub_idx[np.argmin(dists)])
```

所以匹配依据可以拆成：

| 特征 | 是否用于匹配 | 说明 |
|---|---:|---|
| block 位置 | 是 | `block_x, block_y`，先除以 `512`，权重 `W_BLOCK_POS=2.0` |
| agent 位置 / proprio | 是 | 默认世界坐标，权重 `W_AGENT_POS=2.5`；`block_rel=True` 时转为 block 局部坐标 |
| block 姿态角 | 是 | 用 `sin(yaw), cos(yaw)` 表示，权重 `W_YAW=1.5` |
| block 短时速度 | 是 | 最近 3 帧位置的平均有限差分，权重 `W_BLOCK_VEL=1.0` |
| agent 短时速度 | 是 | 最近 3 帧位置的平均有限差分，权重 `W_AGENT_VEL=1.0` |
| 图像 / retrieved frame | 否 | 图像只是在匹配完成后作为 `retrieved_frames` 返回给 policy，不参与 nearest-neighbor 打分 |
| action | eval 时否 | 推理时没有未来 query action；训练 `.npz` 的离线生成可能包含 action term，但仓库 live eval 不用 |

以 `residual_top100--tri_rot0.log` 这次运行为例：

```text
retrieval_strategy='standard'
block_rel=False
retrieval split=['rot0_0', 'rot0_1', 'rot0_2', 'rot0_3', 'rot0_4']
```

因此它是在 `rot0_*` 池里，先用当前 block 位置选出初始位置最接近的 top-100 demo，再用上述 10 维状态向量找最近的 subframe。`retrieved_frames` 虽然会被喂给 Cosmos Policy，但它们是“检索结果”，不是“检索依据”。

## 4. 检索流程：live eval 的两阶段近邻

### 4.1 论文的两阶段检索

论文附录 E 把检索写成两阶段：

1. Stage 1：用初始场景描述 `psi_0` 做 trajectory-level top-K，得到候选轨迹集合 `C_t^traj`。
2. Stage 2：在候选轨迹内部做 subframe-level matching，得到最优 `t'`。

公式可以用当前文档符号重写为：

$$
C_t^{\mathrm{traj}}
= \operatorname{TopK}_{\tau \in D_{\mathrm{pool}}}
\left(
-\left\lVert \psi_0(\mathrm{query}) - \psi_0(\tau) \right\rVert_2^2
\right)
$$

训练时 subframe cost：

$$
\begin{aligned}
d_{\mathrm{tr}}(t,t') =
&\, w_{\mathrm{obj}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{obj}}^t, \phi_{\mathrm{obj}}^{t'}\right)
 + w_{\mathrm{prop}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{prop}}^t, \phi_{\mathrm{prop}}^{t'}\right) \\
&+ w_{\mathrm{vis}} d_{\cos}\!\left(\phi_{\mathrm{vis}}^t, \phi_{\mathrm{vis}}^{t'}\right)
 + w_{\mathrm{act}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{act}}^t, \phi_{\mathrm{act}}^{t'}\right)
\end{aligned}
$$

推理时没有未来 query action，所以去掉 action term：

$$
\begin{aligned}
d_{\mathrm{inf}}(t,t') =
&\, w_{\mathrm{obj}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{obj}}^t, \phi_{\mathrm{obj}}^{t'}\right)
 + w_{\mathrm{prop}} d_{\mathrm{L2}}\!\left(\phi_{\mathrm{prop}}^t, \phi_{\mathrm{prop}}^{t'}\right) \\
&+ w_{\mathrm{vis}} d_{\cos}\!\left(\phi_{\mathrm{vis}}^t, \phi_{\mathrm{vis}}^{t'}\right)
\end{aligned}
$$

然后：

$$
t' = \arg\min_{\tilde{t} \in C_t^{\mathrm{traj}}}
d_{\mathrm{inf}}(t,\tilde{t})
$$

### 4.2 当前代码的 Stage 1：按初始 block 位置筛 demo

`PushTRetrieval.get_retrieved_data()` 里先构造当前状态的 `q_feat` 和 `q_pos=(block_x, block_y)`。

然后按 demo 初始 block 位置做预筛：

```python
cent_d2 = ((self._demo_init_block_pos - q_pos) ** 2).sum(axis=1)
top_demos = {self._demo_keys[i] for i in np.argsort(cent_d2)[:self.N_DEMO_FILTER]}
sub_idx = np.array([i for dk in top_demos for i in self._demo_indices[dk]])
```

默认：

```python
N_DEMO_FILTER = 100
```

这对应论文 Eq. (5)，但只用了 `psi_0` 的一个非常窄的 PushT 特例：初始 block xy。论文里提到的 goal-language embedding、SAM object positions、initial proprio 在这条 PushT live eval 路径没有显式实现。

### 4.3 当前代码的 Stage 2：10 维特征平方 L2

预筛后对候选 subframes 计算：

```python
dists = ((self._feat[sub_idx] - q_feat) ** 2).sum(axis=1)
best_i = int(sub_idx[np.argmin(dists)])
sf = self._subframes[best_i]
```

这对应论文 Eq. (8) 的 $\arg\min d_{\mathrm{inf}}$，但有三个工程化变化：

- 没有 DINO image cosine 项。
- 没有 SAM/object detector，直接使用仿真 GT block pose。
- 所有权重通过 `q_feat` 的分量缩放体现，而不是在公式中单独相加。

代码注释里写了 “3-stage retrieval”，但当前 `PushTRetrieval.get_retrieved_data()` 实际主链路可以理解为：

1. 构造 query 10 维特征；
2. demo-level 初始 block 位置筛选；
3. subframe-level 10 维 L2 最近邻。

其中第 2 步对应论文 Stage 1，第 3 步对应论文 Stage 2。

### 4.4 检索结果如何取 chunk

找到 best subframe 后：

```python
key = (sf["split"], sf["demo"])
t_last = min(sf["t_last"], T - 1)
ret_chunk = self.chunk_size * self.ret_context_multiplier
all_frames  = [imgs[t_last + i clipped] for i in range(ret_chunk)]
all_actions = [acts[t_last + i clipped] for i in range(ret_chunk)]
ret_proprio = prop[t_last]
frames = all_frames[::self.ret_image_subsample]
```

返回：

```python
return np.stack(frames), np.stack(all_actions), ret_proprio
```

默认 `chunk_size=8`, `ret_context_multiplier=1`, `ret_image_subsample=1`，所以检索返回：

- `retrieved_frames`: 8 帧图像；
- `retrieved_actions`: 8 步 raw action；
- `retrieved_proprio`: 匹配帧的 agent xy。

这就是论文 Eq. (1) 里：

$$
\left(s_{t':t'+H}^{\mathrm{pool}}, a_{t':t'+H}^{\mathrm{pool}}\right)
$$

在代码里的具体数据形状。

## 5. 训练时的检索：预计算 `.npz`

训练数据集不调用 `PushTRetrieval` live search，而是 `PushTRetDataset` 读取 retrieval `.npz`。

主配置 `pusht_ret_dataset_top100_residual` 使用：

```python
retrieval_npz_path=[
  ".../retrieval_results_state_action_tri_default_p_base.npz",
  ".../retrieval_results_state_action_tri_goal_goal_flipped.npz",
]
task_split=["tri_default_p", "tri_goal"]
retrieval_source_splits=["base", "goal_flipped"]
episode_allowlist_top_k=100
use_residual_actions=True
predict_future_states=True
```

`_build_retrieval_lookup()` 将 `.npz` 里的：

- `query_ids`: `(N,)`，形如 `suite/demo/start`
- `match_ids`: `(N,K)`，best-first source ids
- `match_sims`: `(N,K)`

合并成：

```python
self._retrieval_lookup[(suite, demo_key, start)] = [
    (source_suite, source_demo, source_start, sim),
    ...
]
```

训练取样时 `__getitem__()` 调 `_get_retrieved_data(suite, demo_key, step_idx)`：

1. 先用 `(suite, demo_key, step_idx)` 查表；
2. 如果没有，就按 `0, -1, +1, -2, +2, ...` 找附近 key；
3. 在 `matches[:retrieval_top_k_choice]` 里随机选一个，主配置常用 `retrieval_top_k_choice=1`；
4. 从 source demo 取 frames/actions/proprio；
5. actions/proprio 会用 dataset stats 归一化；
6. 同时返回一份 `all_acts_raw`，供 residual target 计算。

因此训练时的论文 Eq. (6)/(7) 并不在这个仓库里重新计算；它已经离线固化在 `.npz` 里。仓库保留的是 `.npz` 的消费格式，而不是生成 `.npz` 的脚本。

## 6. 检索条件如何进入模型输入序列

论文 Eq. (1) 写的是：

$$
\pi_\theta\!\left(
s_t^{\mathrm{query}},
s_{t':t'+H}^{\mathrm{pool}},
a_{t':t'+H}^{\mathrm{pool}}
\right)
\rightarrow
\left(
\hat{a}_{t:t+H}^{\mathrm{query}},
\hat{s}_{t+H}^{\mathrm{query}}
\right)
$$

代码把这个输入组织成一个视频 latent 序列。训练路径在 `PushTRetDataset.__getitem__()`；评测路径在 `cosmos_utils.get_action()` 的 `elif "retrieved_frames" in obs:` 分支。

### 6.1 实际 PushT retrieval layout

训练 dataset 的实际布局是：

```text
blank
ret_frame
ret_state
ret_action
cur_frame
cur_state
pred_action
pred_frame
pred_state
```

当 `chunk_size=8`, `num_duplicates_per_image=4`, `predict_future_states=True` 时，latent 级别通常是：

| latent idx | 内容 | 条件还是预测 |
|---|---|---|
| 0 | blank sentinel | condition |
| 1-2 | retrieved frames, 8 frames 压成 2 latents | condition |
| 3 | retrieved state placeholder，后续注入 proprio | condition |
| 4 | retrieved action placeholder，后续注入 action chunk | condition |
| 5 | current frame | condition |
| 6 | current state placeholder，后续注入 proprio | condition |
| 7 | predicted action slot | generated |
| 8 | predicted future frame | generated, if `predict_future_states=True` |
| 9 | predicted future state | generated, if `predict_future_states=True` |

评测 `cosmos_utils.get_action()` 的 PushT retrieval 分支也按这个顺序创建 `image_sequence`，然后把以下索引放进 `data_batch`：

```python
"retrieved_video_start_latent_idx"
"retrieved_video_end_latent_idx"
"retrieved_action_latent_idx"
"retrieved_state_latent_idx"
"current_image_latent_idx"
"current_proprio_latent_idx"
"action_latent_idx"
"future_image_latent_idx"
"future_proprio_latent_idx"
```

一个小坑：`policy_video2world_model_pusht_ret.py` 顶部模块注释里有一版旧的 latent 顺序，和当前 dataset / eval 代码的实际顺序不完全一致。实际运行以 `data_batch` 里的 `*_latent_idx` 为准，因为 mask 和注入逻辑全部读这些索引，而不是读注释里的固定顺序。

### 6.2 mask：哪些 retrieval slot 被当作条件

`policy_video2world_model_pusht_ret.py::_apply_ret_mask()` 修改 Cosmos condition mask：

- retrieved video：`retrieved_video_start_latent_idx:retrieved_video_end_latent_idx`，受 `has_ret_image` 控制。
- retrieved action：`retrieved_action_latent_idx`，受 `has_ret_data` 控制。
- retrieved state：`retrieved_state_latent_idx`，受 `has_ret_data` 控制。
- current image / current proprio：始终 condition。

这对应论文 Eq. (1) 中 “retrieved chunk and current observation condition the WAM”。在代码里，条件化不是改网络结构，而是把 retrieval slots 的 mask 打开。

### 6.3 action/state 的数值注入

retrieved action 和 proprio 不是普通图像像素，代码先放 blank placeholder，再写入 latent：

```python
_inject_retrieved_actions(condition.gt_frames, data_batch)
_inject_retrieved_state(condition.gt_frames, data_batch)
_inject_retrieved_actions(latent_state, data_batch)
_inject_retrieved_state(latent_state, data_batch)
```

作用：

- `condition.gt_frames`：让模型在 conditioning side 看见 retrieved action/state。
- `latent_state`：让 flow-matching 的 clean target `x0` 也带上这些条件槽位。

action 用 `replace_latent_with_action_chunk()` 写入；proprio 用 `replace_latent_with_proprio()` 写入。

评测时 `cosmos_utils.get_action()` 注意了一点：live retrieval 返回的是 raw action/proprio，所以在注入前会用 dataset stats 归一化到训练尺度。

## 7. Residual action：论文 Eq. (3) 的代码实现

论文 residual 公式可以写成：

$$
\hat{a}_{t:t+H}^{\mathrm{query}}
= a_{t':t'+H}^{\mathrm{pool}} + \Delta a_{t:t+H}
$$

这在当前仓库有训练和推理两处强对应。

### 7.1 训练 target：`action_chunk` 被改成 normalized delta

`PushTRetDataset.__getitem__()` 中：

```python
raw_action_chunk = unnormalize(action_chunk)
raw_delta = raw_action_chunk - ret_actions_raw[: self.chunk_size]
action_chunk = normalize_with_delta_stats(raw_delta)
```

也就是模型的 `actions` target 不再是 absolute action，而是 $\Delta a$。

`_load_or_compute_delta_statistics()` 会用 top-1 retrieval 遍历训练集，统计：

```python
delta_actions_min = all_deltas.min(axis=0)
delta_actions_max = all_deltas.max(axis=0)
```

并缓存到：

```text
success_only/delta_dataset_statistics.json
```

### 7.2 推理 add-back：`raw_delta + ret_actions`

`run_eval.py::run_episode()` 中，拿到检索结果后：

```python
ret_frames, ret_actions, ret_proprio = retrieval.get_retrieved_data(...)
observation["retrieved_actions"] = ret_actions
```

如果 `cfg.use_residual_actions=True`：

1. 临时关闭 `cfg.unnormalize_actions`，让 `get_action()` 返回 normalized delta；
2. 用 `delta_stats` 反归一化成 raw delta；
3. 加回 raw retrieved action：

```python
raw_actions = raw_delta + ret_actions[:len(raw_delta)]
```

这就是 Eq. (3) 在执行侧的完整闭环。

### 7.3 为什么 eval 里同一个 `ret_actions` 有两种尺度

这点很容易读混：

- `retrieval.get_retrieved_data()` 返回给 `run_episode()` 的 `ret_actions` 是 raw sim action，用于最后 add-back。
- `cosmos_utils.get_action()` 接收 `obs["retrieved_actions"]` 后，会用 `dataset_stats["actions_min/max"]` 归一化，再注入模型 latent。

所以 raw retrieved action 同时承担两个角色：

1. 归一化后作为条件输入；
2. 原始尺度下作为 residual add-back 的基底。

## 8. 论文 Eq. (2)：联合 action / future-state 目标

论文 Eq. (2) 可理解为：

$$
\mathcal{L}(\theta)
= \lambda \mathcal{L}_{\mathrm{act}}(\hat{a}, a)
 + \mathcal{L}_{\mathrm{state}}(\hat{s}, s)
$$

当前仓库不在 retrieval 专用文件里重写这个 loss，而是继承 Cosmos Policy 的 rectified-flow / flow-matching 训练逻辑：

- `CosmosPolicyDiffusionModelRectifiedFlow.training_step()`
- `CosmosPolicyDiffusionModelRectifiedFlow.compute_loss_rectified_flow()`

PushT retrieval 配置中：

- `action_loss_multiplier=16`，对应论文里的动作损失权重 $\lambda$。
- `predict_future_states=True` 的配置会保留 `future_image_latent_idx` 和 `future_proprio_latent_idx`，让 future image/state 槽位参与目标。
- `predict_future_states=False` 的 no-pred 配置只保留 action prediction，作为消融。

主 README 推荐的训练配置是：

```text
cosmos_predict2p5_2b_480p_pusht_ret_top100_residual
```

它挂到：

```text
pusht_ret_dataset_top100_residual
```

关键参数：

```text
chunk_size=8
state_t=10
tokenizer.chunk_duration=37
action_loss_multiplier=16
predict_future_states=True
use_residual_actions=True
episode_allowlist_top_k=100
retrieval_top_k_choice=1
```

## 9. 训练检索和评测检索的差异

| 维度 | 训练 | 评测 |
|---|---|---|
| 检索来源 | 预计算 `.npz` | HDF5 live pool |
| 主要类 | `PushTRetDataset` | `PushTRetrieval` |
| 是否现场算距离 | 否，直接查 `_retrieval_lookup` | 是，构造 10 维特征并 argmin |
| pool 配置 | `retrieval_source_splits=["base","goal_flipped"]` | `visual_config/goal_angle/retrieval_pool_split` 解析出的 dirs |
| top-k 行为 | `random.choice(matches[:retrieval_top_k_choice])` | 标准策略 top-1 argmin |
| action term | 可能已经体现在离线 `.npz` 生成逻辑里 | 推理不可用，当前代码不使用 |
| 输出尺度 | ret actions 归一化 + raw copy | raw actions，进模型前再归一化 |
| fallback | 无匹配则 zeros | 如果 entry 缺失返回 zeros；正常 live pool 应都有 |

这个差异解释了为什么 README 说：eval 不需要预计算 retrieval files，训练需要 `.npz`。

## 10. `consistent` 和 `cumulative` 检索策略

`PolicyEvalConfig.retrieval_strategy` 支持：

```text
standard
consistent
cumulative
```

默认是 `standard`，也就是 `PushTRetrieval`，最接近论文每步重新检索的 Algorithm 1。

两个变体用于稳定检索轨迹：

- `PushTConsistentRetrieval`：维护 `top_n` 条 track。如果上一条 track 往后推进 `chunk_size` 后仍然和当前 query 足够近，就继续 follow；否则重新检索。
- `PushTCumulativeRetrieval`：维护累计距离，类似对“继续原轨迹”和“fresh start”做带 `gamma` 的动态选择，并可加 `switch_cost`。

它们不是论文主公式的直接实现，更像是为减少逐 chunk 抖动而加的评测策略变体。主文档里应优先按 `standard` 理解。

## 11. 公式到代码的逐项对应

| 论文公式 / 算法 | 论文含义 | 当前代码对应 | PushT 特例与差异 |
|---|---|---|---|
| Section 3: $t' = \arg\min d(s_t^{\mathrm{query}}, s_{t'}^{\mathrm{pool}})$ | 每步找最相似 pool 状态 | `PushTRetrieval.get_retrieved_data()` 中 `np.argmin(dists)` | live eval 用 10 维 GT-state L2 |
| Eq. (1) | 当前 query + retrieved state/action chunk 条件化 policy | `run_episode()` 写 `observation["retrieved_*"]`; `cosmos_utils.get_action()` 建 `data_batch`; `_apply_ret_mask()` 打开条件槽 | 无网络结构改动，靠 latent sequence 和 mask |
| Eq. (2) | action 与 future-state 联合 flow-matching loss | 继承 `compute_loss_rectified_flow()`；配置 `action_loss_multiplier=16`, `predict_future_states=True` | retrieval 文件只负责条件注入，不重写主 loss |
| Eq. (3) | residual action parameterization | dataset 中 `raw_delta = raw_action_chunk - ret_actions_raw`; eval 中 `raw_actions = raw_delta + ret_actions` | 当前仓库最强对应 |
| Eq. (5) | trajectory-level top-K prefilter | `cent_d2` 按初始 block position 选 `N_DEMO_FILTER=100` demos | 只实现 PushT 的简化 `psi_0` |
| Eq. (6) | training subframe distance with action term | 训练 `.npz` 离线固化匹配结果，代码只消费 | 仓库没有 `.npz` 生成脚本 |
| Eq. (7) | training retrieved index argmin | `_build_retrieval_lookup()` 读取 best-first matches；`_get_retrieved_data()` 选 top-k | 可随机 top-k，主配置 top-1 |
| Eq. (8) | inference subframe distance without action term | $d_{\mathrm{code}}=\lVert q_t-q_{t'}\rVert_2^2$ | 无 DINO，几何项主导 |
| Algorithm 1 | observe -> retrieve -> predict -> execute K actions -> repeat | `run_episode()` 的 action queue 为空时检索和预测，然后执行 `num_open_loop_steps` | 当前 `K=H=8`，每 8 步重新检索 |

## 12. 读代码顺序建议

如果只想理解检索池和检索，推荐按这个顺序读：

1. `DATA_MANIFEST.md`：确认哪些 dirs 是 eval pool，哪些 `.npz` 是训练 retrieval。
2. `eval_pusht_rag.sh`：确认实际评测参数和 9 个 `visual_config`。
3. `run_eval.py::PolicyEvalConfig`：看 retrieval 参数默认值。
4. `run_eval.py::RETRIEVAL_SPLIT_MAP` 和 `retrieval.py::resolve_retrieval_split()`：看 pool dirs 怎么选。
5. `retrieval.py::PushTRetrieval._load_pool()`：看 pool 如何变成 subframes 和 feature matrix。
6. `retrieval.py::PushTRetrieval.get_retrieved_data()`：看 live retrieval 最近邻。
7. `pusht_dataset_ret.py::_build_retrieval_lookup()` 和 `_get_retrieved_data()`：看训练 `.npz` 如何被消费。
8. `pusht_dataset_ret.py::__getitem__()`：看检索结果如何拼成训练样本，尤其 residual target。
9. `cosmos_utils.py::get_action()` 的 `retrieved_frames` 分支：看评测时如何拼同样的输入布局。
10. `policy_video2world_model_pusht_ret.py`：看 mask 和 retrieved action/state 注入。

## 13. 一句话总结

论文把 RECAP 描述成“用可增长的 pool memory 做 test-time adaptation”。当前仓库把它落成 PushT 的具体机制：评测时从 HDF5 pool 现场构建 10 维状态最近邻，取未来帧/动作/proprio 作为 Cosmos Policy 的条件；模型不直接生成绝对动作，而是生成相对 retrieved action 的 delta，最后在 raw action 空间加回去执行。
