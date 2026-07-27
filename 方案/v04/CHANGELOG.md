# Human2Robot v04 变更记录

本文件只追加，不改写既有记录。科学语义变更必须升级协议版本并显式作废受影响的下游产物。

## 2026-07-21 — 阶段 0 实施

- 原因：按 `RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md` 冻结 v03 并建立 v04 独立运行边界。
- 提出者：用户；实施者：Codex。
- 代码基线：v03 commit `9a6aacaffcacdd71a5ff6fd3fc92bf30eb711f2a`；v04 实施分支 `codex/recap-v04-offline-clean`。
- 新增：`start_human2robot_v04_docker.sh`、`tools/human2robot_v04.py`、阶段 0 测试、本地资产 SHA256 注册表、v03 冻结 manifest/lock（由 `freeze-v03` 生成）。
- 运行语义：v03 标为 `LEGACY_ORACLE_PHASE_PILOT`；禁止自动续跑；v04 仅允许完整 Docker、四卡、离线环境和 `/DATA1/wxs/ReCAP_M5B_V04_RUNS` 写边界。
- 科学影响：不更改 v03 数据、checkpoint 或指标；不产生 v04 数据、cache、checkpoint 或评估结果。
- 验证：launcher `bash -n` 通过；Docker 内 `py_compile` 与单元测试 4/4 通过；v03 freeze/verify 通过；四卡 CUDA/NCCL all-reduce、编译扩展、数据和四个权重 SHA256 通过。
- 环境 blocker：`/DATA1` 可用 267,942,281,216 bytes，低于冻结的 300 GiB 门槛；正式 preflight 按协议返回 `BLOCKED_ENVIRONMENT`，禁止进入阶段 1。详见 `阶段0_v03冻结与环境门禁验收报告_20260721.md`。

## 2026-07-21 — 阶段 0 磁盘 blocker 解除

- 原因：用户释放 `/DATA1` 空间后，重新执行完整离线四卡 preflight。
- Docker session attempt 0005：`passed`；回执 SHA256 `c5b96421a1c357393a001f0641fe0198642d549b8aa7c13c8c8abe6848cf2fad`。
- Formal preflight attempt 0005：`PASSED`、`formal_v04_allowed=true`、`blockers=[]`；回执 SHA256 `bb1fe4dea6f4f3cf443942bed8986c99b5e0444d0ee7d1a3ab0e6fcc5f436bc6`。
- storage 实测：`/DATA1` 可用 338,502,508,544 bytes（约 315.25 GiB），超过 300 GiB 门槛。
- 其余门禁：四卡 CUDA/NCCL all-reduce、四个本地权重 SHA256、编译扩展、离线环境与 v03 freeze 复核全部通过。
- 科学影响：无协议或科学语义变更；未启动阶段 1、数据派生或训练。

## 2026-07-21 — 阶段 1 实现与来源 SHA 阻断

- 原因：按 v04 总计划实施阶段 1 的数据切分、来源身份、物理字段隔离、审计与任务均衡分布式采样。
- 新增：`tools/human2robot_v04_data.py`、对应测试、`human2robot_v04_sampler.py` 及对应测试；统一入口新增 dry-run 默认的 `prepare-data`、`audit-data` 和 `--execute` 门禁。
- 验证：冻结镜像离线单元测试 17/17 通过；formal prepare 三次均先通过完整 Docker/四卡/权重/v03 freeze preflight。
- attempt 0001：额外的全局 SHA 唯一门禁过严，失败且未写产物；receipt SHA256 `e42eb354bb775ddbf36ff1040e976e0333924a1caa00cb9cd1b8b14a0f332d46`。
- attempt 0002：改为冻结协议要求的 partition 间 SHA 零交集后，确认真实重叠；receipt SHA256 `002f155fe9e00d0514c4914fcc9e95a65159234b537813b8c12326ed29ffd2a0`。
- attempt 0003：明确定位 `grab_pencil1_v1` 的 102/102 个文件与 seen `grab_pencil_v1` 一一具有相同完整文件 SHA；receipt SHA256 `edebc04076e726dd8eacb2679a6959bea71aae7c44e1643a10c7cfc28cfdefb6`。
- 产物影响：`data/Human2Robot/derived/v04` 未生成 manifest、lock、audit report、HDF5 投影或 partial；阶段 2 与训练仍禁止。
- 科学影响：发现冻结任务定义本身违反无泄漏来源独立性；尚未擅自更换 seen/held-out 任务。当前状态 `BLOCKED_PROTOCOL`，详见 `阶段1_来源SHA重叠阻断报告_20260721.md`。

## 2026-07-21 — 阶段 1 Held-out 替代候选预审

- 原因：在不修改冻结协议的前提下，为 `grab_pencil1_v1` 来源重叠 blocker 收集替代任务的完整数据证据。
- 代码：统一入口新增只读 `assess-heldout-replacement`；审计候选容量、字段、finite、时间 gap、H=8/K=8 窗口、完整文件 SHA、v03 历史来源和冻结基线重叠；正式 receipt 明确保持 `protocol_change_authorized=false` 与 `stage1_prepare_allowed=false`。
- 测试：宿主机与冻结离线 Docker 的阶段 0/1 聚焦套件均为 19/19 通过；dry-run 回执通过。首次 Docker 测试命令漏挂 UV 运行时而在启动前失败，补齐正式 launcher 的运行时挂载后通过，不计作测试失败。
- 正式审计：冻结镜像、四卡 `0,2,5,6`、离线环境及完整 preflight 通过；扫描冻结 20 任务和两个候选共 1,147 个文件；attempt 0001 receipt SHA256 `1e7e30b948b3c8c3748f07f10ddd86560c09f2cef12385bbe06dec5ffed3807f`。
- 结果：`push_plate_v1` 51/51 合法且与基线零 SHA 重叠；`push_box_two_v1` 49/50 合法且零重叠；二者互相零重叠，v03 历史来源均为 0。数据质量排序推荐 `push_plate_v1`。
- 科学影响：仅收敛候选证据，未批准或实施任务替换，未生成阶段 1 manifest/HDF5；详见 `阶段1_Heldout替代候选预审报告_20260721.md`。

## 2026-07-21 — 阶段 1 候选审计口径加强

- 原因：attempt 0001 的冻结基线重叠字段只统计契约合法候选；为避免对拒绝文件作未证明的零重叠声明，审计改为同时报告全部候选原始文件与合法文件的重叠数。
- 验证：聚焦套件 19/19 通过；attempt 0002 再次通过完整离线四卡 preflight 并扫描 1,147 个文件。
- 结果：两个候选的全部原始文件重叠数与合法文件重叠数都为 0；原始清单摘要保持 `3d1008bc879604a3168132317e2560e55fce3cfd0d65dc24ded923e9cd004bf9`；attempt 0002 receipt SHA256 `6e6072ff085bc831010d9adb7a0a13edd143e6beed873e1d355acdbf7892d20d`。
- 科学影响：推荐不变，仍为 `push_plate_v1`；任务替换未获授权，阶段 1 继续 `BLOCKED_PROTOCOL`。

## 2026-07-21 — 用户批准 v04.1 Held-out 协议修订

- 授权：用户明确回复“批准推荐方案”。
- 修订：held-out `grab_pencil1_v1` 替换为 `push_plate_v1`；四任务 legacy quarantine 严格按 v03 历史来源并集计为 10/0/10/10；held-out 原始候选总数由 309 改为 258。
- 绑定证据：替代候选最终预审 attempt 0002 receipt SHA256 `6e6072ff085bc831010d9adb7a0a13edd143e6beed873e1d355acdbf7892d20d`。
- 代码协议：stage-1 schema 升级为 `human2robot-v04-stage1-data-v2`，保留 16 个 seen、seed 20260711、每任务 human/dev/final 10/5/20、任务均衡采样和所有无泄漏门禁。
- 科学影响：held-out 任务身份与来源清单发生预注册修订；不得与原 blocked split 混用，必须从 formal `prepare-data` 与 `audit-data` 重新生成全部阶段 1 证据。

## 2026-07-21 — 阶段 1 v04.1 数据与来源身份协议验收通过

- 结论：`VERIFIED_STAGE1 / PASSED`；可进入阶段 2 数据读取与检索实现，仍保持 `training_allowed=false`。
- 聚焦验证：宿主机与冻结离线 Docker 均为 21/21 通过。完整 Human2Robot Docker suite 为 182 passed、3 个第三方 deprecation warnings；receipt SHA256 `a92d27db84eb5f62f519e74c0d5a4d2ea4471b2cdde9ec13f8a187834b368ece`。
- prepare attempt 0001 虽完成生成，但收口审查发现 verifier 尚未独立重算被拒绝来源、split 和 raw inventory，因此未接纳；其产物原子移至可恢复目录 `data/Human2Robot/derived/v04_superseded_prepare_attempt_0001/`，不属于正式输入。
- 审计器补强：独立重算 protocol/split/raw inventory，重哈希 995/995 个原始来源（接收 994、拒绝 1），并复核 140/140 个物理投影。
- 正式 prepare attempt 0002：PASSED；receipt SHA256 `be624fe1b862fa7dcbba962c413e5ac82355b83efcef54877911acc7fa11e379`。正式 audit attempt 0001：PASSED；receipt SHA256 `a527136c363cbce815ef47d9167121c9044f67aaea5b0fbf09e7b17377cfaea2`。三份回执均冻结为 `0444`。
- 冻结摘要：protocol SHA256 `55854766bb7a36468aa0db73bd3a99dfca941dead26a452d2e000ed7bec8d3c2`；split SHA256 `8ca7c8352e031292975779fd254db9591963b2cd3612b1f1b5ab65b6e79ebeaf`；raw inventory SHA256 `d9a2f361862d9266bbd101700f99e1a629c86786bab5a279b6c2df62a388ce43`。
- 冻结产物：manifest SHA256 `7869a078b19ba18aaa6a92c22bec26998a81d412165a02eb8cb6c6aec1c879ed`；lock SHA256 `69632457fcfc935c06413224382da914cd1e43607976ff3a7e44029572ee0577`；audit report SHA256 `f17e797f5af5695a9ac6b036012d1efcc1d16afa8c49b250a7a8067d28bcb85d`，均为 `0444`。
- 数据结果：seen train/validation 为 654/82；held-out quarantine/human/dev/final/reserve 为 30/40/20/80/88；21 个 partition pair 全部来源 SHA 零交集，且无 `.partial`。
- 验收证据：`stage1_v04.1数据与来源身份协议验收报告_20260721.md`。

## 2026-07-21 — 阶段 2 候选过滤与主检索验收通过

- 结论：`VERIFIED_STAGE2 / PASSED`；阶段 3 最小实验接口可以开始，仍保持 `training_allowed=false`。
- 新增 v04 专用 `human2robot_v04_retrieval.py`：`P2Window` 绑定 source SHA/path/partition/role；候选无条件执行来源、partition、role、字段白名单、active pool 和同任务过滤；主方法固定 `geometry_plus_visual`、top-k=3、pool10 与冻结 hash tie-break。
- feature provenance：geometry 只读 H=8 history、visual 只读 current frame；每条 record 记录 query/candidate 来源、partition、rank、distance、tie 和精确 dataset/row provenance；future、target/action、opposite-role 读取均 hard-fail。
- pool growth：每任务 rank 1～10 完整，pool1/2/4/8/10 严格嵌套；只允许同一 checkpoint 评估，不允许因 pool size 重训。
- oracle：原 `phase` 主配置 hard-fail；诊断接口命名为 `oracle_phase`，必须绑定已经完成的 primary receipt SHA256。
- 正式审计：完整离线四卡 preflight 通过，140/140 个 role-only projection、40 个 human pool 与 100 个 robot dev/final episode 通过；source SHA/path overlap、future-row、target/action、opposite-role read count 均为 0。
- 正式 stage-2 report SHA256：`27eceb61565d01297d4ec4ff19d166b5ff5c8d5e9af7916d92d5d9837af651d9`；audit receipt SHA256：`80a1ba97679b404528110aa9917658dd6e5835fc4f5ce6d66dfb4dfd3215f919`。
- 完整 Docker suite：194 passed、3 个第三方 deprecation warnings；receipt SHA256 `ee36ac7348e1d9a8269c21ba137f18fe6142587175deadd1e02e16cfbfb2e375`，明确 `stage3_authorized_by_this_receipt=true`、`training_allowed=false`。
- 非正式偏差：一次宿主机纯静态 `py_compile` 未遵守 Docker-only 约束，未读实验数据、未写正式产物且不作为证据；随后全部语法/单测/正式审计均在冻结 Docker 重做。另有一次 pytest 同名文件收集错误和一次 HDF5 测试夹具广播错误，均无正式产物，最终已由 194 项完整套件覆盖。
- 阶段边界：未生成实际 geometry statistics、WAN feature cache、checkpoint 或评估结果；这些仍分别属于阶段 4/5 及后续阶段。详见 `阶段2_候选过滤与主检索验收报告_20260721.md`。

## 2026-07-21 — 阶段 3 v04 最小实验接口验收通过

- 结论：`VERIFIED_STAGE3 / PASSED`；阶段 4 可以开始，继续保持 `training_allowed=false`。
- 冻结兼容：阶段 1 manifest 绑定的 `tools/human2robot_v04.py` SHA256 保持 `8cbf7f5f...` 不变；新增 `tools/human2robot_v04_experiment.py` 作为阶段 3 起唯一公开实验入口，旧文件只作阶段 0/1 冻结后端。
- 接口：完整提供 `prepare-data`、`audit-data`、`prepare-features`、`preflight`、`train`、`evaluate`、`evaluate-oracle-phase`、`report`；全部默认 dry-run，真实操作必须显式 `--execute`。
- 审计：每次调用独立生成 attempt manifest 与 immutable receipt；后续阶段未授权时返回 `BLOCKED_STAGE_GATE`，不启动 GPU 工作。v04 状态机固定为 prepare、preflight、三方法训练、dev、final、report 共八状态，不复用 203-cell registry。
- 正式证据：preflight receipt SHA256 `3b9e03319...`；stage-3 contract SHA256 `10eab34a...`；完整 Docker suite 206 passed、3 warnings，receipt SHA256 `2717f2ff...`。
- 行为探针：`prepare-features` 默认返回 `DRY_RUN`，receipt SHA256 `abcd94ce...`；`train --execute` 在阶段 5 前返回 `BLOCKED_STAGE_GATE` 且 `training_started=false`，receipt SHA256 `cfde108b...`。
- 偏差：首次 contract 审计因阶段 1 lock 字段名适配错误在写产物前失败；按真实 `lock.manifest.sha256` 修正后通过。没有生成 partial、feature、checkpoint 或评估结果，科学语义不变。
- 验收报告：`阶段3_v04最小实验接口验收报告_20260721.md`。

## 2026-07-22 — 阶段 4 预检与旧 checkpoint 冒烟验收通过

- 结论：`VERIFIED_STAGE4 / PASSED`；五项 fail-closed guardrail 全部为 0，`training_allowed=true`、`stage5_allowed=true`。
- 实现：公开入口的 `prepare-features --execute` 接入阶段 4 orchestrator；新增四卡 WAN feature worker、三个旧 checkpoint 的 strict-adapter smoke worker、协议锁和 5 项 synthetic/协议测试。
- 分区物化：seen-train/validation、held-out human-pool/robot-dev/robot-final 分别为 654/82/40/20/80 episode，manifest 全部 `FROZEN`、`0444`。
- geometry：seen-train human+robot 共 3,909,952 条 relative 10D row，全部有限且非退化；SHA256 `d16fda28986f4860a237fabfb598792fb4630764bc3766735a97d94198e630eb`。
- WAN cache：绑定 tokenizer SHA256 `38071ab...`，生成 1,612 个只读 shard、597,043 个 current-frame feature，future/target read 均为 0；index SHA256 `48e895a1c664790cab85d837506e63facf7c97eca207e3d15f20d4e55c589d17`。
- 旧 checkpoint smoke：每方法 4 task × 5 episode × 8 query × top3 = 480 receipt；三方法合计 1,440/1,440 `PASSED`，实际浮点逐条复核全部有限，0 缺失、0 provenance、0 gap、0 `.partial`，全部 receipt/summary 为 `0444`。
- 正式证据：attempt 0005 manifest/receipt SHA256 为 `56f07e50...` / `00a15f9a...`；protocol lock SHA256 `834fe271...`；冻结四卡离线全量 suite 为 211 passed、3 个第三方 deprecation warnings，receipt SHA256 `d024e50c4628946ba1e810c79c9ff45fb4d9ac5180c52d1eb322d29d02b03afa`。
- 偏差：attempt 0001 的离线 torchrun hostname、attempt 0002 的 Hydra 绝对 config、attempt 0003 的 GPU 映射环境变量、attempt 0004 的 strict adapter metadata 均在最终 attempt 前暴露并修正；失败/阻断收据保留，未作为正式结论。三个 worker 退出有 PyTorch process-group 清理 warning，但退出码、全部收据和独立 bundle 审计均通过。
- 科学边界：所有 smoke 均为 `formal_result=false`、`performance_claim_allowed=false`；不用于模型选择、方法排序或 RECAP 优越性结论。详见 `阶段4_预检与旧checkpoint冒烟验收报告_20260722.md`。

## 2026-07-22 — 阶段 5 双四卡并行调度与启动前实现

- 用户资源授权：物理 GPU `0,1,2,3` 可运行一个实验，物理 GPU `4,5,6,7` 可同时运行另一个实验。
- 调度解释：保持科学方法和启动顺序 `no_retrieval → co_training → recap_hand_ret`；允许前两种方法先后启动后在互不重叠的四卡组上并行，不要求 no-retrieval 完成后才启动 co-training。RECAP 仍为第三个启动的方法。
- 新增：v04 stage-5 seen-train 数据适配、task-balanced sampler 的训练入口接线、三方法冻结配置、方法级训练输入/统计物化、阶段 4 锁与 stage-5 suite 门禁、loss finite 集体门禁、每 10 optimizer step 结构化进度、滚动 checkpoint 保留及 step-7000 独立审计。
- 运行隔离：两方法使用独立容器、日志、training input、训练输出、checkpoint 和回执目录；容器内逻辑 GPU 均为 `0,1,2,3`，回执额外绑定宿主机物理 GPU 编号。
- 科学影响：并发只改变墙钟调度，不改变 seed、seen split、task-balanced sampling、batch、gradient accumulation、optimizer step、LR、H/K、top-k、分辨率、loss multiplier、初始化权重或 fixed-step checkpoint 选择。
- 当前边界：本条记录发生在 stage-5 Docker 全量 suite 和正式训练启动之前；实际 suite/容器/attempt/启动证据必须由后续追加记录给出，不能以本条代替。

## 2026-07-24 — 阶段 5 前两方法运行事实补记（补记于 2026-07-27）

- 补记原因：2026-07-22「阶段 5 双四卡并行调度与启动前实现」条目自述「实际 suite/容器/attempt/启动证据必须由后续追加记录给出」，该记录此前缺失，违反总计划 §2.7「不允许运行事实长期领先于文档」。本条按实际落盘证据补齐，不改写既有记录。
- 代码基线：Git HEAD `d5c2dc6d38e5cf339304959c4a5ea54bd8052318`，分支 `codex/recap-v04-offline-clean`。
- 阶段 5 门禁套件：`stage5_full_suite_20260722` attempt 0002 为 215 passed、attempt 0003 为 217 passed（门槛 215），均 `returncode=0`、`formal_result=false`、`performance_claim_allowed=false`；attempt 0003 receipt 文件 SHA256 `c4c856ed5a654d7a768efef64a14cb5f64db2a27f87efda91dcc34712c5df7b3`。
- `no_retrieval`：run id `stage5_train_no_retrieval_20260722`，attempt 0001 `BLOCKED_ENVIRONMENT`（`offline_env_mismatch` 系列）、attempt 0002/0003 `FAILED`（receipt 仅记录 `Stage-5 training exited with code 1`，未捕获根因——该点属日志缺口）、attempt 0004 `COMPLETED`。容器 `recap-v04-stage5-no-retrieval-20260722-a4`，宿主机物理 GPU `0,1,2,3`，2026-07-22T08:42:43Z 起、2026-07-23T21:19:04Z 止，耗时 131,222.999 s（约 36.45 h），退出码 0。
- `co_training`：run id `stage5_train_co_training_20260722`，attempt 0001 `BLOCKED_ENVIRONMENT`、attempt 0002 `FAILED`（同上，根因未捕获）、attempt 0003 `COMPLETED`。容器 `recap-v04-stage5-co-training-20260722-a3`，宿主机物理 GPU `4,5,6,7`，2026-07-22T07:11:58Z 起、2026-07-23T19:56:55Z 止，耗时 131,836.088 s（约 36.62 h），退出码 0。
- 训练结果：两方法均 `optimizer_step=7000`、`loss_finite=true`、`checkpoint_selection=fixed_step_7000`、`heldout_training_or_selection_used=false`、`status=PASSED`。末步 loss 分别为 `no_retrieval` 0.0012434、`co_training` 0.00093679。
- checkpoint：各 20 个文件、11,762,540,695 / 11,762,540,691 bytes；bundle SHA256 `16d4f007b8190b5869fd8e6e0c4fc56481fcbd56dad50366645864d9d5ec0b79`（no_retrieval）、`52c7197129091be0a93c2824dcc4b87653ab638ea93e02c9fb9332e6d7dacbc5`（co_training）。`iter_000007000` 已置只读，滚动点保留 `iter_000005000`、`iter_000006000`，符合冻结的保留策略。
- 训练 receipt SHA256：`c88074dcd5d84d7d7d6244f377e15cfd873444b8f441186f33faebc40d24cc74`（no_retrieval attempt 0004）、`cd3ea95d70622f00f1a0c9ff08fe9f88d3a1f985dfef05cb1310e49e3940f172`（co_training attempt 0003）。
- 科学影响：不改变 seed、split、batch、step、优化器或 checkpoint 选择语义；两次训练均为 `formal_result=false`，不构成性能声明。`recap_hand_ret` 未启动，`stage5/training_inputs/` 下无该方法目录。

## 2026-07-27 — 阶段 5 后科学前提偏差登记，暂停阶段 5 剩余方法与后续阶段

- 原因：只读诊断发现四项使「v04 正在测量人手到机器人跨具身迁移」这一前提不成立的偏差。按总计划 §2.7 暂停新正式任务并生成偏差记录。
- 提出者：用户；实施者：Claude（只读诊断）。代码基线同上，工作区干净。
- 诊断合规性：`NONFORMAL_DIAGNOSTIC`。使用宿主机 `anaconda` Python 执行，**不满足 §2.2 的 Docker-only 约束**，特此披露；全程只读，未占 GPU，未写入正式运行根或 `data/Human2Robot/derived`，未签发 receipt，不解除任何 preflight blocker。
- P0-1：`tools/human2robot_v04_data.py:776` 把原始 `action` 映射为 `human/hand_action_7d`，而 `action` 经验证为机器人末端指令（与 `end_position` 同轴相关性 0.940/0.960/0.797，相似变换 R²=0.913，中位残差 6.4 mm）。真实人手数据 `transformed_hand_frames`（已验证为合法 SE(3)）与 `transformed_hand_coords` 虽在投影白名单内，但从未参与几何计算。该错误使检索几何退化为同具身跨 episode 匹配。`方案/v01/data_inventory_human2robot.md:76-77,83` 当时判断正确，认知在 v01→v04 间丢失且无变更记录。
- P0-2：`cosmos_policy/datasets/human2robot_v04_dataset.py:269-274` 的 co_training 使用同 episode 自身的 `action[future]` 作为上下文，与论文「jointly trains a single policy on the union of target and pool trajectories」定义不符；夹爪通道误差因此被压低 5–8 倍（同 episode 0.018–0.029 vs 跨 episode 0.141–0.152）。同时 `tools/human2robot_v04_stage4_worker.py:111-126` 表明评估时改用跨 episode 检索候选，训练/评估口径不一致。
- P0-3：`tools/human2robot_v04_stage4.py:491,544` 与 `tools/human2robot_v04_stage5.py:260,313` 在主结果路径上使用 phase 选择候选时间窗，违反 §2.2「phase primary hard-fail」与 §五测试要求。该路径绕过 `validate_primary_config()`/`rank_oracle_phase()`，217 项测试无法触发；且 `phase` 依赖 `frame_count`（HDF5 attribute），五项 guardrail 只统计行读取，结构上检测不到。
- P0-4：在阶段 4 冻结的同一批 160 个 dev query 上，现生产配置（A / phase / geometry_plus_visual）的对齐残差为 13.2 mm，劣于「不检索」基线 11.1 mm（1.19×），夹爪 0.0578 vs 0.0289；随机候选 16.0 mm，oracle 上界 4.1 mm。36 格消融显示：K=8/16/32/64 下现配置为 1.19×/1.03×/0.84×/0.74×，**K=8 下无任何配置取得实质正收益**；`geom+vis` 与 `geom+pix` 在每个 K 上均劣于 `geom` 单独使用，16 维 WAN 均值特征单独使用为全表最差（1.59–1.70×），替换为 576 维稠密灰度亦未改善。
- 科学影响：改变前提判定。阶段 1/2/4 的 `PASSED` 由「通过」降级为「形式通过、语义未验证」——SHA 不交、receipt 完整、guardrail 归零等形式检查仍然有效，字段语义与检索有效性从未被审计。
- 产物影响：不撤回任何科学结论（v04 全程 `formal_result=false`，阶段 6/7 未实现）。阶段 5 两个 step-7000 checkpoint 保留且 `status=PASSED` 不撤销，但不得用于原定主效应对比。阶段 1 投影/manifest、geometry 统计、WAN cache、smoke plan 保留为历史证据；一旦 K、人手通道或主检索模态变更即全部失效（`legal_window_start` 按 H8/K8 计算）。
- 门禁状态：`recap_hand_ret` 训练暂停；阶段 6/7 暂不实现。剩余环境 blocker：`/DATA1` 可用 171 GiB 低于 `tools/human2robot_v04.py:43` 的 300 GiB 门槛；8 张 GPU 全部占用。
- 已执行测试：本次未执行测试套件（只读诊断，未改代码）。阶段 5 门禁套件结论沿用 2026-07-24 补记条目。
- 证据：`方案/v04/阶段5后_科学前提偏差记录_20260727.md`。

## 2026-07-27 — 探索期脚本归档与 P0-1 数字口径更正

- 原因：(1) 偏差记录所依据的诊断脚本此前仅存于 `/tmp`，不可复现；(2) 归档时把 P0-1 的主证据重写为可复现脚本 `tools/pilot/d02_action_is_robot_command.py` 并实际执行后，发现同日 P0-1 条目引用的部分数字取自不同抽样（单 episode 或 held-out 子集），与 48 个 seen-train episode 的汇总口径不一致。本条按只追加规则更正，不改写既有条目。
- 新增：`tools/pilot/`（16 个只读诊断脚本 + `README.md` + `run_all.sh`）与 `方案/v05_pilot/PILOT_LOG.md`（探索期流水账，条目 001）。脚本**不**纳入 `_controlled_bindings()` 哈希绑定，不签发 receipt，中间产物仅写 `/tmp`；其宿主机 Python 运行方式不满足总计划 §2.2，已在 `tools/pilot/README.md` 中显式披露。
- 验证：16 个脚本全部 `py_compile` 通过，`run_all.sh` 通过 `bash -n`；新写的 `d02` 与编辑过的 `d07`（删除一处无效 print）实际执行通过，`d07` 输出与归档前一致。
- P0-1 数字更正（以 `d02` 的 48 episode / 16 任务汇总为准）：

  | 量 | 同日条目原值 | 更正值 |
  |---|---|---|
  | `action` 与 `end_position` 同轴相关性 | 0.940 / 0.960 / 0.797（单 episode） | **0.971 / 0.975 / 0.880** |
  | 平均逐轴 \|action − end_position\| | 10.5 / 10.6 / 11.0 mm（单 episode） | **9.8 / 11.2 / 9.4 mm**（对应轴 std 78.8 / 104.1 / 39.6 mm） |
  | `action` → `end_position` | R² = 0.913（逐 episode 拟合均值，量纲不同） | 恒等映射 **R² = 0.930**；残差 mean 20.5 / median 6.4 / p90 65.6 mm |
  | `action[:,6]` 与 `gripper_state` | 均值 0.005–0.028（held-out 逐任务） | 均值差 **0.0309**，96.9% 帧完全相同 |
  | `action` 逐步位移相关性 | ≈ 0.000 | **0.018 / 0.023 / 0.012**（幅度 2.40 vs 2.41 mm） |
  | `action` K=8 位移相关性 | 0.328 / 0.383 / 0.307 | **0.317 / 0.375 / 0.298** |
  | 与人手腕位置的最大相关性 | 误记为 0.82（该值实为 `hand_coords` 指尖点对与 `gripper_state` 的相关性） | **0.613**（需换轴） |

- 姿态列范围更正：原表述「四个 held-out 任务全程恒定 `(180, 0, 90)`」仅对 held-out 抽查 episode 成立；16 个 seen 任务整体并不恒定（`action` 178 个唯一值、`end_position` 1567 个）。held-out 集上 orientation 次要指标接近退化这一推论保留，须在 v05 评估设计时复核。
- 结论方向未变：更正后的相关性（0.971 / 0.975 / 0.880）与恒等映射 R²（0.930）比原值更强地支持「`action` 是机器人末端指令而非人手信号」。P0-2、P0-3、P0-4 的数字未受影响。
- 同步更新：`方案/v04/阶段5后_科学前提偏差记录_20260727.md` §3 与 `方案/v05_pilot/PILOT_LOG.md` §001.1/§001.2 已按上表改写（两者非只追加文档，直接更正并在此登记）。
- 门禁状态：不变。`recap_hand_ret` 仍暂停，阶段 6/7 仍不实现。
