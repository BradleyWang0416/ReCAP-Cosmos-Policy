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
