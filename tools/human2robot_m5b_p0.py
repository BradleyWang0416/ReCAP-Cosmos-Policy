#!/usr/bin/env python3
"""Docker-only acceptance runner for M5B-P0-IMPLEMENTATION.

This runner performs no downloads.  It validates the frozen protocol and
parent artifacts, the formal Human2Robot dataset/model contract, all nine
learned-method/seed configs, local checkpoint bindings, and a CUDA one-batch
adapter-I/O overfit probe.  It does not launch the formal 2B training runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# This must be set before importing the experiment-config package.  The P0
# runner is deliberately local-only and never permits auto-download fallback.
os.environ.setdefault("COSMOS_SKIP_HF_AUTO_DOWNLOAD", "1")

from cosmos_policy.config.experiment.human2robot_experiment_configs import (
    ALL_HUMAN2ROBOT_CONFIGS,
    FORMAL_SEEDS,
    LEARNED_METHODS,
    LOCAL_POSTTRAINED_CKPT,
    LOCAL_TOKENIZER_CKPT,
)
from cosmos_policy.datasets.human2robot_dataset import (
    FORMAL_DATASET_KWARGS,
    QUARANTINED_PUSHT_DATASET_KWARGS,
    Human2RobotFormalDataset,
)
from cosmos_policy.models.human2robot_adapter import (
    run_one_batch_overfit_probe,
    validate_human2robot_batch,
)
from cosmos_policy.models.policy_video2world_model_human2robot_ret import (
    CosmosPolicyHuman2RobotRetModelRectifiedFlow,
)
from cosmos_policy.models.policy_video2world_model_pusht_ret import (
    CosmosPolicyPushTRetModelRectifiedFlow,
)
from tools.human2robot_m5b_protocol import file_sha256, validate_protocol_file

SCHEMA_VERSION = "human2robot-m5b-p0-implementation-report-v1"
GATE_ID = "M5B-P0-IMPLEMENTATION"
EXPECTED_PROTOCOL_SHA256 = "7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4"
EXPECTED_TRAIN_WINDOWS = 968
EXPECTED_HELDOUT_WINDOWS = 153
REAL_2B_OVERFIT_REPORT = Path("data/Human2Robot/derived/m5b_v03/real_2b_one_batch_overfit.json")


class P0Error(RuntimeError):
    """Raised when an M5B P0 hard requirement is not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P0Error(message)


def _combined_code_sha256(repo_root: Path) -> tuple[str, list[dict[str, str]]]:
    paths = [
        "cosmos_policy/datasets/human2robot_dataset.py",
        "cosmos_policy/models/human2robot_adapter.py",
        "cosmos_policy/models/policy_video2world_model_human2robot_ret.py",
        "cosmos_policy/config/experiment/human2robot_experiment_configs.py",
        "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py",
        "cosmos_policy/scripts/train.py",
        "tools/human2robot_m5b_p0.py",
    ]
    digest = hashlib.sha256()
    bindings: list[dict[str, str]] = []
    for relative in paths:
        path = repo_root / relative
        _require(path.is_file(), f"Missing P0 code artifact: {relative}")
        sha = file_sha256(path)
        bindings.append({"path": relative, "file_sha256": sha})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), bindings


def _assert_docker_cuda(repo_root: Path) -> dict[str, Any]:
    _require(Path("/.dockerenv").exists(), "P0 must run inside Docker")
    _require(repo_root.resolve() == Path("/workspace").resolve(), "P0 repo root must be /workspace")
    _require(torch.cuda.is_available(), "P0 requires CUDA in the project Docker environment")
    _require(torch.version.cuda is not None, "Docker torch build has no CUDA runtime")
    return {
        "inside_docker": True,
        "workspace": str(repo_root.resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": True,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "downloads_performed": False,
        "environment_sync_performed": False,
    }


def _dataset_paths(repo_root: Path) -> dict[str, Path]:
    view = (
        repo_root
        / "data/Human2Robot/derived/views/nominal_camera_30hz_segmented"
        / "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy"
        / "train_only_tplus1_query_anchor_se3_identity_scale_v1"
    )
    return {
        "canonical_root": repo_root / "data/Human2Robot/canonical/v3",
        "main_view_path": view,
        "m3_report_path": repo_root / "data/Human2Robot/derived/m3_v03/m3_validation_report.json",
        "m4_report_path": repo_root / "data/Human2Robot/derived/m4_v03/m4_launch_report.json",
        "protocol_path": repo_root / "方案/v03/M5B_formal_acceptance_protocol_v1.json",
    }


def _dataset(repo_root: Path, method: str, seed: int, split: str) -> Human2RobotFormalDataset:
    return Human2RobotFormalDataset(
        **_dataset_paths(repo_root),
        split=split,
        method_id=method,
        seed=seed,
        use_image_aug=False,
    )


def _validate_datasets(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    all_contract_checks: dict[str, bool] = {}
    for method in LEARNED_METHODS:
        for seed in FORMAL_SEEDS:
            train = _dataset(repo_root, method, seed, "train")
            _require(len(train) == EXPECTED_TRAIN_WINDOWS, f"Wrong train windows: {method}/{seed}")
            sample_checks = validate_human2robot_batch(train[0])
            _require(all(sample_checks.values()), f"Sample contract failed: {method}/{seed}")
            manifests.append(train.contract_manifest())
            for name, passed in sample_checks.items():
                all_contract_checks[name] = all_contract_checks.get(name, True) and passed

    heldout = _dataset(repo_root, "recap_hand_ret", FORMAL_SEEDS[0], "heldout")
    _require(len(heldout) == EXPECTED_HELDOUT_WINDOWS, "Wrong heldout window count")
    heldout_checks = validate_human2robot_batch(heldout[0])
    _require(all(heldout_checks.values()), "Heldout sample contract failed")

    overfit_dataset = _dataset(repo_root, "recap_hand_ret", FORMAL_SEEDS[0], "train")
    batch = next(iter(DataLoader(overfit_dataset, batch_size=2, shuffle=False, num_workers=0)))
    batch_checks = validate_human2robot_batch(batch)
    _require(all(batch_checks.values()), "Collated batch contract failed")
    overfit = run_one_batch_overfit_probe(batch, steps=500, learning_rate=0.01, device="cuda")
    _require(overfit["passed"] is True, "CUDA one-batch overfit probe failed")
    return {
        "train_window_count": EXPECTED_TRAIN_WINDOWS,
        "heldout_window_count": EXPECTED_HELDOUT_WINDOWS,
        "method_seed_manifest_count": len(manifests),
        "manifests": manifests,
        "sample_contract_checks": all_contract_checks,
        "heldout_contract_checks": heldout_checks,
        "collated_batch_contract_checks": batch_checks,
        "text_conditioning": {
            "mode": "disabled_zero_embedding",
            "reason": "No frozen Human2Robot T5 artifact exists; local-only P0 forbids substituting PushT semantics or downloading a new artifact.",
            "claim_boundary": "Retrieval/action/visual adapter P0 only; no language-conditioning claim.",
        },
    }, overfit


def _validate_configs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_names = {
        f"cosmos_predict2p5_2b_human2robot_{method}_seed{seed}"
        for method in LEARNED_METHODS
        for seed in FORMAL_SEEDS
    }
    actual_names = {config["job"]["name"] for config in ALL_HUMAN2ROBOT_CONFIGS}
    _require(actual_names == expected_names, "Formal method/seed config matrix is incomplete")
    _require(issubclass(CosmosPolicyHuman2RobotRetModelRectifiedFlow, CosmosPolicyPushTRetModelRectifiedFlow),
             "Human2Robot model is not connected to the real retrieval RF class")

    templates: list[dict[str, Any]] = []
    for config in ALL_HUMAN2ROBOT_CONFIGS:
        model = config["model"]["config"]
        seed = int(config["trainer"]["seed"])
        name = config["job"]["name"]
        method = next(method for method in LEARNED_METHODS if f"_{method}_" in name)
        _require(config["trainer"]["max_iter"] == 7000, f"Wrong steps: {name}")
        _require(config["dataloader_train"]["batch_size"] == 25, f"Wrong batch: {name}")
        _require(config["dataloader_train"]["sampler"]["seed"] == seed, f"Wrong sampler seed: {name}")
        _require(model["action_dim"] == model["proprio_dim"] == 10, f"Wrong dimensions: {name}")
        _require(model["state_t"] == 10, f"Wrong state_t: {name}")
        _require(model["tokenizer"]["chunk_duration"] == 37, f"Wrong tokenizer duration: {name}")
        templates.append(
            {
                "method_id": method,
                "experiment_id": "M5B-MAIN-01",
                "seed": seed,
                "optimizer_steps": 7000,
                "batch_size_per_data_parallel_rank": 25,
                "data_parallel_world_size": "RUN_BOUND_AT_FORMAL_LAUNCH",
                "H_steps": 8,
                "K_steps": 8,
                "formal_config_name": name,
                "expected_primary_checkpoint_step": 7000,
            }
        )
    return {
        "config_count": len(ALL_HUMAN2ROBOT_CONFIGS),
        "config_names": sorted(actual_names),
        "model_class": (
            "cosmos_policy.models.policy_video2world_model_human2robot_ret."
            "CosmosPolicyHuman2RobotRetModelRectifiedFlow"
        ),
        "real_retrieval_rectified_flow_subclass": True,
        "parameter_scale": "2B",
        "max_optimizer_steps": 7000,
        "batch_size_per_data_parallel_rank": 25,
        "action_dim": 10,
        "proprio_dim": 10,
        "state_t": 10,
        "tokenizer_chunk_duration": 37,
        "hydra_registration_smoke_required": True,
    }, templates


def _validate_local_weights() -> dict[str, Any]:
    checkpoint = Path(LOCAL_POSTTRAINED_CKPT)
    tokenizer = Path(LOCAL_TOKENIZER_CKPT)
    for label, path in (("initialization checkpoint", checkpoint), ("tokenizer", tokenizer)):
        _require(str(path).startswith("/DATA1/"), f"{label} is not on the read-only local weight mount")
        _require("hf://" not in str(path), f"{label} would require a download")
        _require(path.is_file(), f"Missing local {label}: {path}")
    return {
        "initialization_checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "file_sha256": file_sha256(checkpoint),
        },
        "tokenizer": {
            "path": str(tokenizer),
            "size_bytes": tokenizer.stat().st_size,
            "file_sha256": file_sha256(tokenizer),
        },
        "all_paths_local": True,
        "downloads_performed": False,
    }


def _validate_real_2b_overfit(repo_root: Path, checkpoint_sha256: str) -> dict[str, Any]:
    path = repo_root / REAL_2B_OVERFIT_REPORT
    _require(path.is_file(), f"Missing real 2B overfit evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    result = evidence.get("result", {})
    contract = evidence.get("diagnostic_contract", {})
    model = evidence.get("model", {})
    bindings = evidence.get("bindings", {})
    _require(evidence.get("status") == "passed", "Real 2B overfit evidence is not passed")
    _require(model.get("parameter_scale") == "2B", "Overfit evidence did not use the 2B model")
    _require(
        model.get("initialization_checkpoint_sha256") == checkpoint_sha256,
        "Overfit checkpoint binding differs from the resolved local checkpoint",
    )
    _require(contract.get("same_train_window_repeated") is True, "2B diagnostic did not repeat one batch")
    _require(contract.get("rectified_flow_noise_and_time_seed_fixed_each_step") is True, "2B noise was not fixed")
    _require(contract.get("fresh_job_path") is True and contract.get("resume_checkpoint") is None, "2B run resumed")
    _require(contract.get("optimizer_steps") == 50, "Wrong real 2B diagnostic step count")
    _require(result.get("training_completed") is True, "Real 2B training did not complete")
    _require(float(result.get("final_logged_loss", 1.0)) < 0.01, "Real 2B final loss is too high")
    _require(float(result.get("final_to_initial_ratio", 1.0)) < 0.02, "Real 2B loss did not overfit")
    _require(bindings.get("protocol_file_sha256") == EXPECTED_PROTOCOL_SHA256, "2B protocol binding changed")
    for item in bindings.get("code_files", []):
        code_path = repo_root / item["path"]
        _require(code_path.is_file(), f"Missing 2B-bound code: {item['path']}")
        _require(file_sha256(code_path) == item["file_sha256"], f"2B-bound code changed: {item['path']}")
    return {"report_path": str(REAL_2B_OVERFIT_REPORT), "report_file_sha256": file_sha256(path), **evidence}


def _validate_hydra_load(repo_root: Path) -> dict[str, Any]:
    from cosmos_policy._src.imaginaire.config import load_config
    from cosmos_policy._src.imaginaire.lazy_config import instantiate

    representative = "cosmos_predict2p5_2b_human2robot_recap_hand_ret_seed20260711"
    config = load_config(
        "cosmos_policy/config/config.py",
        ["--", f"experiment={representative}"],
    )
    resolved = {
        "seed": int(config.trainer.seed),
        "max_optimizer_steps": int(config.trainer.max_iter),
        "batch_size_per_data_parallel_rank": int(config.dataloader_train.batch_size),
        "sampler_seed": int(config.dataloader_train.sampler.seed),
        "action_dim": int(config.model.config.action_dim),
        "proprio_dim": int(config.model.config.proprio_dim),
        "state_t": int(config.model.config.state_t),
        "tokenizer_chunk_duration": int(config.model.config.tokenizer.chunk_duration),
        "checkpoint_path": str(config.checkpoint.load_path),
        "tokenizer_path": str(config.model.config.tokenizer.vae_pth),
    }
    expected = {
        "seed": 20260711,
        "max_optimizer_steps": 7000,
        "batch_size_per_data_parallel_rank": 25,
        "sampler_seed": 20260711,
        "action_dim": 10,
        "proprio_dim": 10,
        "state_t": 10,
        "tokenizer_chunk_duration": 37,
        "checkpoint_path": LOCAL_POSTTRAINED_CKPT,
        "tokenizer_path": LOCAL_TOKENIZER_CKPT,
    }
    _require(resolved == expected, f"Resolved Hydra config mismatch: {resolved}")
    resolved_dataset_keys = set(config.dataloader_train.dataset.keys())
    expected_dataset_keys = {"_target_", *FORMAL_DATASET_KWARGS, *QUARANTINED_PUSHT_DATASET_KWARGS}
    _require(
        resolved_dataset_keys == expected_dataset_keys,
        f"Resolved dataset contains inherited/stale kwargs: {sorted(resolved_dataset_keys - expected_dataset_keys)}",
    )
    dataset = instantiate(config.dataloader_train.dataset)
    _require(isinstance(dataset, Human2RobotFormalDataset), "Resolved Hydra dataset has wrong type")
    _require(len(dataset) == EXPECTED_TRAIN_WINDOWS, "Resolved Hydra dataset has wrong window count")
    return {
        "status": "passed",
        "representative_config": representative,
        "verified_in_same_docker_environment": True,
        "resolved_dataset_instantiation": "passed",
        "resolved_dataset_window_count": len(dataset),
        "resolved_dataset_keys": sorted(resolved_dataset_keys),
        "quarantined_inherited_pusht_dataset_keys": sorted(QUARANTINED_PUSHT_DATASET_KWARGS),
        "resolved_values": resolved,
    }


def _run_p0_pytest(repo_root: Path) -> dict[str, Any]:
    tests = [
        "cosmos_policy/datasets/human2robot_dataset_test.py",
        "cosmos_policy/models/human2robot_adapter_test.py",
        "cosmos_policy/config/experiment/human2robot_experiment_configs_test.py",
    ]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        f"--rootdir={repo_root}",
        "--import-mode=importlib",
        "-q",
        *tests,
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, f"P0 pytest failed:\n{result.stdout}\n{result.stderr}")
    _require("15 passed" in result.stdout, f"Unexpected P0 pytest count:\n{result.stdout}")
    return {
        "status": "passed",
        "test_count": 15,
        "test_files": tests,
        "command": command,
        "summary": "15 passed",
        "stdout_tail": result.stdout.strip().splitlines()[-1],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    overfit = report["one_batch_overfit"]
    real_2b = report["real_2b_one_batch_overfit"]
    weights = report["local_weight_bindings"]
    text = f"""# M5B-P0-IMPLEMENTATION 验收报告

日期：{report['generated_at_utc']}

结论：**M5B-P0-IMPLEMENTATION 通过；M5-B、M5-v03、Gate C 与 M6 均未通过。**

## 完整环境证据

- 仅在项目 Docker `/workspace` 中执行，Torch `{report['environment']['torch']}`，CUDA `{report['environment']['torch_cuda']}`。
- GPU：`{report['environment']['gpu_name']}`，容器可见 {report['environment']['gpu_count']} 张。
- 未下载文件、未构建镜像、未同步或降级环境。

## P0 通过项

- 正式 Human2Robot dataset adapter：train `{report['dataset_contract']['train_window_count']}` windows，held-out `{report['dataset_contract']['heldout_window_count']}` windows，严格 t+1、gap crossing=0。
- 正式 model adapter 是现有 retrieval-conditioned rectified-flow 模型的真实子类；3 个 learned methods × 3 seeds 共 `{report['formal_configs']['config_count']}` 个 2B 配置。
- 配置固定 7,000 optimizer steps、每 rank batch 25、10D action/proprio、H/K=8、37-frame tokenizer chunk。
- 真实 2B 单批 overfit：50 optimizer steps，loss `{real_2b['result']['initial_logged_loss']:.4f}` → `{real_2b['result']['final_logged_loss']:.4f}`，下降 `{real_2b['result']['loss_reduction_percent']:.2f}%`，峰值 PyTorch GPU memory `{real_2b['result']['peak_pytorch_gpu_memory_gib']:.2f} GiB`。
- 补充 CUDA adapter-I/O 梯度探针：loss `{overfit['initial_loss']:.9g}` → `{overfit['final_loss']:.9g}`，ratio `{overfit['loss_ratio']:.3g}`。
- 初始化 checkpoint SHA256：`{weights['initialization_checkpoint']['file_sha256']}`。
- tokenizer SHA256：`{weights['tokenizer']['file_sha256']}`。

## 证据边界

- 真实 2B 单批 overfit 固定同一 train window 与 rectified-flow 噪声，并为诊断关闭 warmup；它是实现/连通性测试，不是模型质量证据，也不是 7,000-step 正式训练替代品。
- Human2Robot 尚无冻结的 T5 embedding artifact；P0 使用显式 `disabled_zero_embedding`，没有复用 PushT 文本语义，也没有下载新文件，因此不提出语言条件能力结论。
- 每个 held-out task 当前只有 1 条独立 human demonstration，P1 要求 10 条；正式训练 checkpoint、全实验矩阵与统计门禁仍未完成。
- `query_command_status=unverified`，`deployment_command_adapter_id=null`；禁止真实机器人 rollout。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(repo_root: Path) -> dict[str, Any]:
    environment = _assert_docker_cuda(repo_root)
    protocol_path = repo_root / "方案/v03/M5B_formal_acceptance_protocol_v1.json"
    protocol = validate_protocol_file(protocol_path, repo_root)
    _require(protocol["protocol_file_sha256"] == EXPECTED_PROTOCOL_SHA256, "Frozen protocol hash changed")
    code_sha256, code_bindings = _combined_code_sha256(repo_root)
    dataset_contract, overfit = _validate_datasets(repo_root)
    formal_configs, run_templates = _validate_configs()
    local_weights = _validate_local_weights()
    real_2b_overfit = _validate_real_2b_overfit(
        repo_root, local_weights["initialization_checkpoint"]["file_sha256"]
    )
    hydra_smoke = _validate_hydra_load(repo_root)
    pytest_evidence = _run_p0_pytest(repo_root)
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))["frozen_data_contract"]
    for template in run_templates:
        template.update(
            {
                "protocol_file_sha256": protocol["protocol_file_sha256"],
                "code_sha256": code_sha256,
                "resolved_initialization_checkpoint_sha256": local_weights["initialization_checkpoint"]["file_sha256"],
                "canonical_schema": frozen["canonical_schema"],
                "split_sha256": frozen["split_sha256"],
                "time_view_id": frozen["time_view_id"],
                "pool_action_view_id": frozen["pool_action_view_id"],
                "query_action_view_id": frozen["query_action_view_id"],
                "action_alignment_id": frozen["action_alignment_id"],
                "view_id": frozen["view_id"],
                "retrieval_index_sha256": frozen["retrieval_index_sha256"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "passed",
        "generated_at_utc": utc_now(),
        "protocol_validation": protocol,
        "environment": environment,
        "code_sha256": code_sha256,
        "code_bindings": code_bindings,
        "dataset_contract": dataset_contract,
        "formal_configs": formal_configs,
        "local_weight_bindings": local_weights,
        "run_manifest_templates": run_templates,
        "one_batch_overfit": overfit,
        "real_2b_one_batch_overfit": real_2b_overfit,
        "hydra_config_load_smoke": hydra_smoke,
        "p0_pytest": pytest_evidence,
        "claim_boundary": {
            "p0_implementation": "passed",
            "full_2b_training_quality": "not_tested",
            "m5b_p1_data": "pending_1_of_10_independent_heldout_human_demos_per_task",
            "m5b_p2_run_completeness": "pending",
            "m5_v03": "pending",
            "gate_c": "pending",
            "m6_rollout_approved": False,
            "query_command_status": "unverified",
            "deployment_command_adapter_id": None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-report",
        type=Path,
        default=Path("data/Human2Robot/derived/m5b_v03/p0_implementation_report.json"),
    )
    parser.add_argument(
        "--acceptance-json",
        type=Path,
        default=Path("方案/v03/M5B_P0_IMPLEMENTATION_自动验收报告.json"),
    )
    parser.add_argument(
        "--acceptance-markdown",
        type=Path,
        default=Path("方案/v03/M5B_P0_IMPLEMENTATION_验收报告.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    try:
        report = run(repo_root)
    except (P0Error, OSError, ValueError, KeyError, StopIteration, RuntimeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "gate_id": GATE_ID, "status": "failed", "error": str(exc)}, indent=2))
        return 1
    for path in (args.data_report, args.acceptance_json):
        resolved = path if path.is_absolute() else repo_root / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = args.acceptance_markdown if args.acceptance_markdown.is_absolute() else repo_root / args.acceptance_markdown
    _write_markdown(markdown, report)
    print(json.dumps({"status": report["status"], "gate_id": GATE_ID, "report": str(args.acceptance_json), "checkpoint_sha256": report["local_weight_bindings"]["initialization_checkpoint"]["file_sha256"], "one_batch_overfit": report["one_batch_overfit"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
