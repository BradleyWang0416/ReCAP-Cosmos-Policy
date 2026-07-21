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
