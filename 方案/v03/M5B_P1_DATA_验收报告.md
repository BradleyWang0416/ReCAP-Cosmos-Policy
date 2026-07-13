# M5B-P1-DATA 验收报告

日期：2026-07-13T07:13:58.642976+00:00

结论：**M5B-P1-DATA 通过；M5B-P2、M5-v03、Gate C 与 M6 仍未通过。**

## 独立 human demonstrations

| held-out task | 独立 source episodes | 要求 | 状态 |
|---|---:|---:|---|
| `grab_cube2_v1` | 10 | 10 | passed |
| `grab_pencil1_v1` | 10 | 10 | passed |
| `grab_to_plate1_v1` | 10 | 10 | passed |
| `push_box_random_v1` | 10 | 10 | passed |

- 独立性单位是不同 source episode；window/chunk 不计为独立重复。
- 40 个 source path、source file SHA256 与 human-content SHA256 均唯一。
- pool size `0/1/2/4/8/10` 使用每 task 同一冻结排序的嵌套前缀。

## 泄漏门禁

- 源容器虽然是 paired HDF5，但提取器只读取 human camera、human action、hand coords/frames、step 与 timestamp。
- `robot_camera/end_position/qpos/qvel/gripper_state` 读取数为 0；派生 HDF5 不包含 robot observation/target。
- held-out robot target 未用于 retrieval feature、normalization、alignment、lag 或 checkpoint selection。
- train-only action statistics 未重算，冻结 SHA256 为 `b318962a31b8ac52f237a06163010910c35c50ef06b72c1dc5a33971c1b81562`。

## 当前边界

- P1 只证明正式 held-out human-only pool 已达到 10 条/任务并通过泄漏审计，不证明模型收益。
- P2 的全部 method×experiment×3-seed step-7000 checkpoint 尚未完成，Gate C 保持 pending。
- `query_command_status=unverified`，不得用于真实机器人 rollout。
