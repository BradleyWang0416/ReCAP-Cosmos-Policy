# M2 Human2Robot native-time canonical HDF5 验收报告

> **语义验收已于 v03 重开。** 本报告保留 20/20 native-frame、source mapping、gap/segment 与 split 的历史工程验收；其中 `/action` 被解释为 robot absolute EE target、source FPS 为空以及“允许进入 M3”的结论已暂停。v03 主线与新门禁见 [v03 实验方案](../v03/RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md)。

验收日期：2026-07-11（Asia/Shanghai）  
数据源：Human2Robot v1  
schema：`human2robot-canonical-hdf5-v2`  
结论：**通过**

## 1. 结论与边界

M2-v02 pilot 已通过 Gate A：20 条 episode 覆盖 20 个不同任务，9,039 个 source frame 全部一一保留；结构 validator 与时间真实性 validator 均为 20/20 通过。canonical 数据没有生成固定间隔时间轴，也没有填写未经证实的 source FPS。

本结论只证明 canonical 数据无损、可追溯并完成时间质量审计，不证明 Human2Robot 具有统一采样率，不证明任何 per-step 位移是 m/s，也不证明 M3 retrieval/residual 或 M4 human-to-robot bridge 有效。

## 2. 实现与交付物

| 交付物 | 路径 | 状态 |
|---|---|---|
| native-time 转换、审计、统计与可视化核心 | `tools/human2robot_m2.py` | 已实现并实跑 |
| 默认 v2 CLI 与显式 legacy 策略 | `tools/convert_human2robot_m2.py` | 已实现 |
| 独立结构/时间真实性 validator | `tools/validate_canonical_hdf5.py` | 已实现并独立复验 |
| native/gap/timebase/split 回归测试 | `tools/human2robot_m2_test.py` | 8/8 通过 |
| v2 数据契约 | `方案/v02/human2robot_canonical_hdf5_contract_v2.md` | 已保存 |
| frozen task split | `方案/v02/human2robot_task_split_manifest_v2.json` | 16 train / 4 held-out |
| 自动验收报告 | `方案/v02/M2_Human2Robot_native_time_自动验收报告.json` | 已保存 |
| timebase audit | `方案/v02/human2robot_timebase_audit_v2.json` | 已保存 |
| native-time pilot | `data/Human2Robot/canonical/v2/pilot/demo_*.hdf5` | 20 条，1,980,200,207 bytes |
| train-only 统计 | `data/Human2Robot/canonical/v2/{dataset_statistics,dataset_statistics_post_norm,data_quality_statistics}.json` | 已生成并验证 provenance |
| 自动/独立 validator 明细 | `data/Human2Robot/canonical/v2/m2_validation_report.json`、`m2_independent_validation_report.json` | 两次均通过 |
| native-time 可视化 | `data/Human2Robot/canonical/v2/visualizations/sample_*.mp4` | 10 条，人工抽查通过 |
| v1 撤销标记 | `data/Human2Robot/canonical/v1/DEPRECATED_M2_V01.json` | 已保存，8 项历史证据已冻结哈希 |

## 3. 验收矩阵

| M2-v02 标准 | 实测 | 结果 |
|---|---:|---|
| 至少 20 条 episode、20 个任务 | 20 条 / 20 个任务 | 通过 |
| `canonical_frames == source_frames` | 9,039 / 9,039；逐 episode 相等 | 通过 |
| human/robot/state/action 第一维一致 | 20/20 HDF5 全部共享 T | 通过 |
| 必要数值 finite | actions、states、raw state/action、human hand 全量 finite | 通过 |
| 无人工 `0,0.1,...` 时间轴 | 20/20 不含 `metadata/timestamps`；`source_fps` 为空 | 通过 |
| source frame 一一映射 | `source_indices == 0..T-1`，source step/timestamp 逐值回查 | 通过 |
| source 可追溯 | 20/20 source SHA-256 回查一致；code/config/split hash 已写入 | 通过 |
| timebase 状态、gap 与 segment | 17 coarse、3 discontinuous；3 gap / 23 segment | 通过 |
| split 在统计前固化 | split SHA-256 `ee32fd5a...346cac` | 通过 |
| 统计只使用 train | 16 个 train task、7,803 帧；`heldout_data_used=false` | 通过 |
| held-out 不泄漏 | 4 个 held-out task 与统计 provenance 无交集 | 通过 |
| M2 不生成 residual delta stats | v2 中不存在 `delta_dataset_statistics.json` | 通过 |
| 10 条 native-time 可视化人工抽查 | 10/10 通过；联系表保存在 `visualizations/manual_review/` | 通过 |
| v1 冻结与撤销说明 | marker 已生成，旧报告顶部已标注“验收已撤销” | 通过 |

## 4. 时间质量审计

20 条 episode 的 source 时间字段累计结果：

| 指标 | 数值 |
|---|---:|
| step repeat / jump / rollback | 0 / 3 / 0 |
| timestamp repeat / jump / rollback | 8,639 / 2 / 0 |
| gap / segment | 3 / 23 |
| trusted / coarse / discontinuous / unknown | 0 / 17 / 3 / 0 |

3 条 discontinuous episode 均已在 jump 边界处分段：

| source episode | 最大 step jump | 最大 timestamp jump | segment |
|---|---:|---:|---:|
| `cloth/cloth21/episode_0.hdf5` | 2 | 1 | 2 |
| `grab_pencil1_v1/episode_0.hdf5` | 232 | 120 | 2 |
| `grab_pencil_v1/episode_0.hdf5` | 232 | 120 | 2 |

train-only `data_quality_statistics.json` 对连续 segment 内位移使用单位 `metres per retained source step (not m/s)`；没有输出物理速度。该统计的 action/state 最大单步位移分别为 0.19998 m 和 0.12853 m，作为后续异常值诊断信号，不解释成速度上限。

## 5. task split 与统计防泄漏

split 按 task 通过 seed `20260711` 的稳定 SHA-256 排序固化。held-out task 为：

```text
grab_cube2_v1
grab_pencil1_v1
grab_to_plate1_v1
push_box_random_v1
```

`dataset_statistics.json`、`dataset_statistics_post_norm.json` 和 `data_quality_statistics.json` 的 provenance 均记录相同 split hash、16 个 train task、7,803 帧和 `heldout_data_used=false`。独立 provenance validator 已确认 held-out task 未出现在统计任务列表中。post-normalization 的 10D action/proprio 各维覆盖 `[-1,1]`。

## 6. 人工可视化验收

seed `20260711` 随机抽取 10 条，不放回。每条 MP4 同屏显示 human RGB、robot RGB、state/action、native frame、source step/timestamp、segment 和 gap 标记。每条取 frame 0/50/100/150 形成联系表后逐张检查。

| sample | task | native frames | 人工结果 |
|---|---|---:|---|
| `sample_00_demo_00013` | `grab_to_plate2_and_pull_v1` | 336 | 通过 |
| `sample_01_demo_00016` | `pull_plate_grab_cube` | 956 | 通过 |
| `sample_02_demo_00003` | `cloth/cloth31` | 639 | 通过 |
| `sample_03_demo_00015` | `grab_two_cubes2_v1` | 161 | 通过 |
| `sample_04_demo_00019` | `push_box_random_v1` | 292 | 通过 |
| `sample_05_demo_00011` | `grab_to_plate1_v1` | 370 | 通过 |
| `sample_06_demo_00009` | `grab_pencil_v1` | 394 | 通过；可见 segment 0→1 |
| `sample_07_demo_00017` | `pull_plate_v1` | 620 | 通过 |
| `sample_08_demo_00005` | `grab_cube2_v1` | 180 | 通过 |
| `sample_09_demo_00006` | `grab_cup_v1` | 242 | 通过 |

10/10 视频经 ffprobe 验证为 852×312、10 fps 编码且帧数与 canonical T 一致。这里的 10 fps 只用于播放，不是 source capture frequency 的证据。

## 7. 自动验证与复验

```text
python -m pytest -q tools/human2robot_m2_test.py
-> 8 passed

.venv/bin/ruff check tools/human2robot_m2.py \
  tools/convert_human2robot_m2.py \
  tools/validate_canonical_hdf5.py \
  tools/human2robot_m2_test.py
-> All checks passed!

python -m py_compile tools/human2robot_m2.py \
  tools/convert_human2robot_m2.py \
  tools/validate_canonical_hdf5.py \
  tools/human2robot_m2_test.py
-> passed
```

全流水线与 standalone validator 都完成了 20 个 source 文件的实际打开、frame 数比对、source step/timestamp 逐值比对和 SHA-256 回查。两次结果均为 20/20 通过，split validation 无错误。

## 8. 关键证据 SHA-256

```text
b0e6fe89186692f4815993709f881624c800a122b9dd69d4d431fe202ba4f6e9  preprocessing_manifest.json
d16b0da6267b2bc1c5a7234d14cdbefa9b8ba49fd6e8495c1b83f154cf367871  task_split_manifest.json
f904eb3496600b06bf64d173b2becbf1e24c824cfbaa1765b96e13d3787d61e4  timebase_audit_report.json
c4661466ac559e4c01ca24d7a199cc13b185937a0b520b80e0d3c25f2f17536f  dataset_statistics.json
7dbc5448b7c7bfb1f3bafe2e2aad6f8669ab2295a9566867f8be8f1037d116dc  dataset_statistics_post_norm.json
a43ede54b9a3f3d48aa023147d67409b67d41b6f3fee7000f68597c5b5afc5c9  data_quality_statistics.json
5fc71ba56cf233c7612e587b754928ae486dde88281897c140f80da4b68600f9  m2_validation_report.json
9efcfefa4fa92755e64c6827f0a05c5d295b33fdfa9bb11259fd4c7ed0a2cb91  m2_independent_validation_report.json
2472f75ed63a5bed5d0fa65ba7ab13679b8ffc82b3acbf5063f7a5dc8bd03e8d  visualization_manifest.json
a7b72ca9bc24837a107580bc0fbf01f001eb8bc80b423ad030629407820ab352  DEPRECATED_M2_V01.json
```

## 9. 保留风险与下一门禁

- source FPS 与 timestamp 采集语义仍未 verified，因此 M3 不得把 coarse/discontinuous 时间字段用于 m/s 或固定秒数 horizon。
- xyz 单位、Euler 顺序、absolute action 语义当前为 `assumed`；gripper 极性为 `unknown`。进入 M4 前必须取得上游证据，或做备选解释消融并证明结论不变。
- 本次只验收 20 条 pilot。是否转换全部 1,316 条需在 M3 time-view 决策后另行批准。
- M3 derived view 必须禁止跨 `segment_id` 取 chunk，并把 `legacy_stride3` 明确标为负面对照。

在上述边界内，M2-v02 **通过**，允许进入 M3-v02。
