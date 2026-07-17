#!/usr/bin/env python3
"""Read-only Docker/resource/source preflight for M5B-P2.

This command never creates a formal activation artifact and never launches a
training or evaluation cell.  It reports every remaining code, contract, GPU,
storage, mount, weight, and source-snapshot blocker in one fail-closed record.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from cosmos_policy.config.experiment.human2robot_experiment_configs import (
    LOCAL_POSTTRAINED_CKPT,
    LOCAL_TOKENIZER_CKPT,
)
from tools.human2robot_m5b_p2 import (
    FORMAL_OUTPUT_ROOT,
    source_manifest,
    source_paths,
    source_snapshot_matches_candidate,
)
from tools.human2robot_m5b_p2_handlers import (
    FORMAL_OFFLINE_ENV,
    HandlerContractError,
    handler_coverage_manifest,
    require_formal_activation,
)
from tools.human2robot_m5b_p2_inference import preflight as inference_preflight
from tools.human2robot_m5b_p2_matrix import (
    FOUR_GPU_WORLD_SIZE,
    file_sha256,
    load_execution_matrix,
)

MINIMUM_FREE_GIB = 35
EXPECTED_GPU_COUNT = FOUR_GPU_WORLD_SIZE
EXPECTED_WEIGHT_HASHES = {
    "initialization_checkpoint": "565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47",
    "tokenizer": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
}


class PreflightContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightContractError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_mount_path(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n")


def mount_binding(path: Path, mountinfo_path: Path = Path("/proc/self/mountinfo")) -> dict[str, Any]:
    _require(mountinfo_path.is_file(), f"Mount table is unavailable: {mountinfo_path}")
    target = str(path.absolute())
    matches: list[dict[str, Any]] = []
    for line in mountinfo_path.read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        if len(fields) < 6:
            continue
        mount_point = _decode_mount_path(fields[4])
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            matches.append(
                {
                    "mount_point": mount_point,
                    "mount_options": fields[5].split(","),
                    "filesystem": right.split()[0] if right.split() else "unknown",
                }
            )
    _require(bool(matches), f"No mount covers {path}")
    result = max(matches, key=lambda item: len(str(item["mount_point"])))
    result["target_path"] = target
    result["writable"] = "rw" in result["mount_options"]
    return result


def _weight_binding(label: str, path: Path, verify_content: bool) -> dict[str, Any]:
    expected = EXPECTED_WEIGHT_HASHES[label]
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": expected,
        "content_hash_verified": verify_content,
    }
    if path.is_file():
        result["size_bytes"] = path.stat().st_size
    if verify_content and path.is_file():
        result["actual_sha256"] = file_sha256(path)
        result["status"] = "passed" if result["actual_sha256"] == expected else "failed"
    else:
        result["actual_sha256"] = None
        result["status"] = "not_verified" if path.is_file() else "missing"
    return result


def _storage_probe(path: Path) -> dict[str, Any]:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    free_gib = usage.free / (1024**3)
    return {
        "requested_path": str(path),
        "probed_existing_path": str(existing),
        "free_bytes": usage.free,
        "free_gib": free_gib,
        "minimum_free_gib": MINIMUM_FREE_GIB,
        "status": "passed" if free_gib >= MINIMUM_FREE_GIB else "failed",
    }


def queue_authorized(
    *,
    require_launch_activation: bool,
    launch_activation_status: str,
    blockers: list[str],
) -> bool:
    """Return true only for the post-activation, blocker-free queue state."""

    return require_launch_activation and launch_activation_status == "approved" and not blockers


def run_preflight(
    workspace: Path,
    *,
    artifact_root: Path = FORMAL_OUTPUT_ROOT,
    verify_weight_hashes: bool = True,
    require_launch_activation: bool = True,
) -> dict[str, Any]:
    matrix = load_execution_matrix(workspace)
    coverage = handler_coverage_manifest(matrix)
    inference = inference_preflight(workspace)
    source = source_manifest(workspace, source_paths(workspace))
    source_snapshot_path = artifact_root / "source_snapshots" / source["code_sha256"]
    source_snapshot_manifest = source_snapshot_path / "source_snapshot_manifest.json"
    launch_activation_path = artifact_root / "launch_activation_v6.json"
    mount = mount_binding(artifact_root)
    storage = _storage_probe(artifact_root)
    weights = {
        "initialization_checkpoint": _weight_binding(
            "initialization_checkpoint", Path(LOCAL_POSTTRAINED_CKPT), verify_weight_hashes
        ),
        "tokenizer": _weight_binding("tokenizer", Path(LOCAL_TOKENIZER_CKPT), verify_weight_hashes),
    }
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    handler_envs = {
        tuple(sorted(plan["environment"].items()))
        for plan in coverage["plans"].values()
    }
    planned_offline_env = dict(next(iter(handler_envs))) if len(handler_envs) == 1 else {}
    current_offline_env = {key: os.environ.get(key) for key in FORMAL_OFFLINE_ENV}

    infrastructure_blockers: list[str] = []
    if not Path("/.dockerenv").exists():
        infrastructure_blockers.append("not_running_inside_docker")
    if gpu_count != EXPECTED_GPU_COUNT:
        infrastructure_blockers.append(
            f"expected_{EXPECTED_GPU_COUNT}_gpus_found_{gpu_count}"
        )
    if storage["status"] != "passed":
        infrastructure_blockers.append("formal_output_storage_below_35_gib")
    if not mount["writable"]:
        infrastructure_blockers.append("formal_output_mount_is_read_only")
    if planned_offline_env != FORMAL_OFFLINE_ENV:
        infrastructure_blockers.append("handler_offline_environment_not_bound")
    for label, binding in weights.items():
        if binding["status"] != "passed":
            infrastructure_blockers.append(f"{label}_hash_{binding['status']}")
    source_snapshot_status = "pending"
    source_snapshot_error: str | None = None
    if not source_snapshot_manifest.is_file():
        infrastructure_blockers.append("candidate_source_snapshot_not_materialized")
    else:
        try:
            frozen_source = json.loads(source_snapshot_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            source_snapshot_status = "invalid"
            source_snapshot_error = str(error)
            infrastructure_blockers.append("candidate_source_snapshot_invalid")
        else:
            if not source_snapshot_matches_candidate(frozen_source, source):
                source_snapshot_status = "invalid"
                source_snapshot_error = "Frozen source manifest differs from the current candidate"
                infrastructure_blockers.append("candidate_source_snapshot_invalid")
            else:
                source_snapshot_status = "passed"
    launch_activation: dict[str, Any]
    if not require_launch_activation:
        launch_activation = {
            "path": str(launch_activation_path),
            "status": "not_required_for_pre_activation_probe",
        }
    elif not launch_activation_path.is_file():
        infrastructure_blockers.append("launch_activation_v6_not_issued")
        launch_activation = {"path": str(launch_activation_path), "status": "missing"}
    else:
        try:
            activation_payload = json.loads(launch_activation_path.read_text(encoding="utf-8"))
            require_formal_activation(activation_payload, matrix)
            if activation_payload.get("candidate_code_sha256") != source["code_sha256"]:
                raise HandlerContractError("Launch activation is bound to different candidate code")
            if activation_payload.get("source_snapshot_manifest_path") != str(source_snapshot_manifest):
                raise HandlerContractError("Launch activation is bound to a different source snapshot")
            receipt_value = activation_payload.get("docker_suite_receipt_path")
            if not isinstance(receipt_value, str):
                raise HandlerContractError("Launch activation has no Docker-suite receipt path")
            receipt_path = Path(receipt_value)
            if not receipt_path.is_file():
                raise HandlerContractError("Launch activation Docker-suite receipt is missing")
            if activation_payload.get("docker_suite_receipt_sha256") != file_sha256(receipt_path):
                raise HandlerContractError("Launch activation Docker-suite receipt hash mismatch")
        except (json.JSONDecodeError, HandlerContractError) as error:
            infrastructure_blockers.append("launch_activation_v6_invalid")
            launch_activation = {
                "path": str(launch_activation_path),
                "status": "invalid",
                "error": str(error),
            }
        else:
            launch_activation = {
                "path": str(launch_activation_path),
                "status": "approved",
                "file_sha256": file_sha256(launch_activation_path),
            }

    blockers = list(dict.fromkeys([*inference["blockers"], *infrastructure_blockers]))
    formal_queue_allowed = queue_authorized(
        require_launch_activation=require_launch_activation,
        launch_activation_status=str(launch_activation.get("status")),
        blockers=blockers,
    )
    return {
        "schema_version": "human2robot-m5b-p2-prequeue-preflight-v6",
        "generated_at_utc": utc_now(),
        "status": "passed" if not blockers else "blocked",
        "formal_queue_allowed": formal_queue_allowed,
        "formal_queue_started": False,
        "workspace": str(workspace),
        "artifact_root": str(artifact_root),
        "docker": {
            "inside_docker": Path("/.dockerenv").exists(),
            "formal_output_mount": mount,
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "visible_gpu_count": gpu_count,
            "expected_gpu_count": EXPECTED_GPU_COUNT,
            "status": "passed" if gpu_count == EXPECTED_GPU_COUNT else "failed",
        },
        "storage": storage,
        "weights": weights,
        "offline_environment": {
            "required": FORMAL_OFFLINE_ENV,
            "planned_handler_environment": planned_offline_env,
            "current_probe_process_environment": current_offline_env,
            "status": "passed" if planned_offline_env == FORMAL_OFFLINE_ENV else "failed",
        },
        "source_snapshot": {
            "candidate_code_sha256": source["code_sha256"],
            "candidate_file_count": len(source["files"]),
            "expected_path": str(source_snapshot_path),
            "manifest_path": str(source_snapshot_manifest),
            "materialized": source_snapshot_manifest.is_file(),
            "status": source_snapshot_status,
            **({"error": source_snapshot_error} if source_snapshot_error is not None else {}),
        },
        "launch_activation": launch_activation,
        "matrix": {
            "cell_count": len(matrix.bindings_by_id),
            "handler_count": coverage["cell_count"],
            "all_cells_have_handlers": coverage["all_cells_have_handlers"],
            "prepared_entry_count": len(matrix.prepared_manifest["entries"]),
            "report_covered_evaluation_count": len(matrix.report_covered_evaluation_ids),
        },
        "inference_preflight": inference,
        "infrastructure_blockers": infrastructure_blockers,
        "blockers": blockers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--artifact-root", type=Path, default=FORMAL_OUTPUT_ROOT)
    parser.add_argument(
        "--skip-weight-content-hash",
        action="store_true",
        help="Diagnostic-only fast mode; produces an explicit hash_not_verified blocker.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional machine-readable output path; does not create activation or launch cells.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_preflight(
        args.workspace.resolve(),
        artifact_root=args.artifact_root,
        verify_weight_hashes=not args.skip_weight_content_hash,
    )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightContractError as error:
        print(f"M5B-P2 preflight error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
