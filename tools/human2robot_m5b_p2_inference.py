#!/usr/bin/env python3
"""Formal held-out inference runner for frozen M5B-P2 evaluation cells.

The current frozen contracts contain unresolved sampler, workspace, lag-view,
and temporal-corruption bindings.  ``preflight`` reports those issues without
loading a model.  ``run-cell`` requires a valid formal activation artifact and
therefore cannot silently produce evidence while any issue remains.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed, misc
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init
from cosmos_policy.datasets.human2robot_p2_contract import deterministic_inference_seed
from cosmos_policy.datasets.human2robot_p2_dataset import build_human2robot_p2_dataset
from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
from cosmos_policy.models.policy_video2world_model_human2robot_ret import (
    CosmosPolicyHuman2RobotRetModelRectifiedFlow,
)
from tools.human2robot_m5b_p2_evaluation import (
    RankPrediction,
    aggregate_task_seed_windows,
    evaluate_ranked_query,
)
from tools.human2robot_m5b_p2_handlers import HandlerContractError, require_formal_activation
from tools.human2robot_m5b_p2_matrix import (
    CellBinding,
    ExecutionMatrix,
    canonical_json_sha256,
    file_sha256,
    load_execution_matrix,
)

FORMAL_GUIDANCE = 1.5
FORMAL_NUM_STEPS = 35
FORMAL_SCHEDULER = "native_rectified_flow_scheduler"
FORMAL_SHIFT = 5.0
FORMAL_USE_KERRAS_SIGMA = True
FORMAL_VARIANCE_SCALE = False
DEFAULT_ARTIFACT_ROOT = Path("/DATA1/wxs/ReCAP_M5B_P2_RUNS")
DEFAULT_ACTIVATION_PATH = DEFAULT_ARTIFACT_ROOT / "launch_activation_v6.json"
DEFAULT_WORKSPACE_BOUNDS_PATH = Path("/workspace/方案/v03/M5B_P2_workspace_bounds_v1.json")


class InferenceContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceContractError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def artifact_path(root: Path, cell_id: str) -> Path:
    return root / "cells" / cell_id / "artifact.json"


def validate_formal_sampler_signature(model_type: type = CosmosPolicyHuman2RobotRetModelRectifiedFlow) -> dict[str, Any]:
    method = model_type.generate_samples_from_batch
    signature = inspect.signature(method)
    required = {"guidance", "num_steps", "shift", "use_variance_scale"}
    explicit = required.issubset(signature.parameters) and "solver_option" not in signature.parameters
    return {
        "scheduler": FORMAL_SCHEDULER,
        "shift": FORMAL_SHIFT,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "method_signature": str(signature),
        "native_parameters_explicit": explicit,
        "status": "passed" if explicit else "blocked",
        "reason": (
            None
            if explicit
            else "native rectified-flow guidance/steps/shift/variance parameters are not explicit or a legacy solver parameter is present"
        ),
    }


def preflight(workspace: Path) -> dict[str, Any]:
    matrix = load_execution_matrix(workspace)
    sampler = validate_formal_sampler_signature()
    blockers = list(matrix.formal_readiness_blockers)
    if sampler["status"] != "passed" and "native_rectified_flow_contract_unresolved" not in blockers:
        blockers.append("native_rectified_flow_contract_unresolved")
    return {
        "schema_version": "human2robot-m5b-p2-inference-preflight-v1",
        "status": "passed" if not blockers else "blocked",
        "formal_queue_allowed": False,
        "sampler_contract": sampler,
        "formal_parameters": {
            "guidance": FORMAL_GUIDANCE,
            "num_steps": FORMAL_NUM_STEPS,
            "scheduler": FORMAL_SCHEDULER,
            "shift": FORMAL_SHIFT,
            "use_kerras_sigma_at_inference": FORMAL_USE_KERRAS_SIGMA,
            "ode_steps": 34,
            "final_clean_x0_prediction": True,
            "variance_scale": FORMAL_VARIANCE_SCALE,
            "sample_count_per_query_retrieval_rank": 1,
        },
        "blockers": blockers,
    }


def dataset_kwargs(workspace: Path, binding: CellBinding) -> dict[str, Any]:
    evaluation = binding.evaluation
    _require(evaluation is not None, f"Cell is not an evaluation: {binding.cell.cell_id}")
    entry = binding.prepared_entry
    _require(entry is not None, f"Prepared input missing: {binding.cell.cell_id}")
    statistics_path = workspace / str(entry["statistics_path"])
    retrieval_index_path = workspace / str(entry["retrieval_index_path"])
    root = workspace / "data/Human2Robot"
    return {
        "canonical_root": root / "canonical/v3",
        "main_view_path": root
        / "derived/views/nominal_camera_30hz_segmented/human_hand_robot_frame_raw/"
        "robot_ee_observed_t_plus_1_bc_proxy/train_only_tplus1_query_anchor_se3_identity_scale_v1",
        "m3_report_path": root / "derived/m3_v03/m3_validation_report.json",
        "m4_report_path": root / "derived/m4_v03/m4_launch_report.json",
        "protocol_path": workspace / "方案/v03/M5B_formal_acceptance_protocol_v1.json",
        "supplement_path": workspace / "方案/v03/M5B_P2_execution_supplement_v2.json",
        "p1_pool_root": root / "derived/m5b_v03/p1_human_only_pool",
        "split": "heldout",
        "method_id": evaluation.method_id,
        "experiment_id": evaluation.experiment_id,
        "variant_id": evaluation.variant_id,
        "seed": evaluation.run_seed,
        "h_steps": evaluation.h_steps,
        "k_steps": evaluation.k_steps,
        "window_stride": 8,
        "top_k": evaluation.top_k,
        "pool_size": evaluation.pool_size,
        "retrieval_modality": evaluation.retrieval_modality,
        "time_view_id": evaluation.time_view_id,
        "query_offset_view_steps": evaluation.query_offset_view_steps,
        "target_representation": evaluation.target_representation,
        "statistics_path": statistics_path,
        "retrieval_index_path": retrieval_index_path,
        "resolution_variant": evaluation.resolution_variant,
        "use_image_aug": False,
        "num_duplicates_per_image": 4,
        "text_conditioning": "disabled_zero_embedding",
        "diagnostic_window_limit": None,
    }


def _file_binding(workspace: Path, relative_path: str) -> dict[str, str]:
    path = workspace / relative_path
    _require(path.is_file(), f"Runtime contract file is missing: {path}")
    return {"path": relative_path, "sha256": file_sha256(path)}


def immutable_runtime_manifest(workspace: Path, binding: CellBinding) -> dict[str, Any]:
    """Bind every code/data component required by an inference artifact."""
    evaluation = binding.evaluation
    entry = binding.prepared_entry
    _require(evaluation is not None and entry is not None, "Evaluation runtime binding is incomplete")
    pool_root = workspace / "data/Human2Robot/derived/m5b_v03/p1_human_only_pool"
    code_paths = (
        "cosmos_policy/datasets/human2robot_dataset.py",
        "cosmos_policy/datasets/human2robot_p2_contract.py",
        "cosmos_policy/datasets/human2robot_p2_dataset.py",
        "tools/human2robot_m5b_p2_evaluation.py",
        "tools/human2robot_m5b_p2_inference.py",
        "tools/human2robot_m5b_p2_matrix.py",
        "tools/human2robot_m5b_p2_prepare.py",
        "tools/human2robot_m5b_p2_successor.py",
    )
    return {
        "schema_version": "human2robot-m5b-p2-runtime-hash-manifest-v1",
        "evaluation_cell_id": evaluation.cell_id,
        "prepared_input_cell_id": evaluation.prepared_input_cell_id,
        "code": [_file_binding(workspace, path) for path in code_paths],
        "pool": {
            "pool_manifest": _file_binding(
                workspace, str((pool_root / "pool_manifest.json").relative_to(workspace))
            ),
            "selection_manifest": _file_binding(
                workspace, str((pool_root / "selection_manifest.json").relative_to(workspace))
            ),
        },
        "feature_index": {
            "path": str(entry["retrieval_index_path"]),
            "sha256": str(entry["retrieval_index_sha256"]),
        },
        "statistics": {
            "path": str(entry["statistics_path"]),
            "sha256": str(entry["statistics_sha256"]),
        },
        "alignment": {
            "implementation": "cosmos_policy.datasets.human2robot_dataset.align_pool_chunk",
            "rule": "translation-relative plus SO(3) relative rotation, canonical 10D projection",
        },
        "projection": {
            "implementation": "tools.human2robot_m5b_p2_evaluation.reconstruct_rank_prediction",
            "rank_aggregation": "equal_weight_after_canonical_projection",
        },
        "tie_breaking": {
            "implementation": "cosmos_policy.datasets.human2robot_p2_contract.rank_retrieval_candidates",
            "run_seed": evaluation.run_seed,
        },
        "preprocessor": {
            "resolution_variant": evaluation.resolution_variant,
            "use_image_aug": False,
        },
    }


def retrieval_provenance_record(item: Mapping[str, Any]) -> dict[str, Any]:
    aligned = np.asarray(item["raw_aligned_pool"], dtype=np.float32)
    return {
        "query_id": str(item["query_id"]),
        "task": str(item["task"]),
        "episode_id": str(item["episode_id"]),
        "current_row": int(item["current_row"]),
        "retrieval_rank": int(item["retrieval_rank"]),
        "candidate_id": str(item.get("candidate_id", "")),
        "candidate_human_content_sha256": str(item.get("candidate_human_content_sha256", "")),
        "retrieval_distance": float(item.get("retrieval_distance", 0.0)),
        "retrieval_tie_sha256": str(item.get("retrieval_tie_sha256", "")),
        "aligned_pool_10d_sha256": canonical_json_sha256(
            {"shape": list(aligned.shape), "values": aligned.tolist()}
        ),
    }


def _scalar(batch_item: Any) -> Any:
    if isinstance(batch_item, torch.Tensor):
        return batch_item.flatten()[0].item()
    if isinstance(batch_item, (list, tuple)):
        return batch_item[0]
    return batch_item


SEVERE_TEMPORAL_VARIANTS = {
    "frame_drop_20pct",
    "timestamp_jitter_20ms",
    "pause_1p0s",
    "step_jump_20",
}


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def apply_temporal_corruption(
    item: Mapping[str, Any],
    *,
    corruption_id: str,
    severity: str,
    inference_seed: int,
    h_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Corrupt actual uint8 retrieval frames immediately before tokenizer/model use."""

    video = item.get("video")
    _require(isinstance(video, torch.Tensor), "Temporal corruption requires tensor video input")
    _require(video.ndim == 4 and video.shape[0] == 3, f"Unexpected C,T,H,W video: {tuple(video.shape)}")
    start, stop = 1, 1 + int(h_steps)
    _require(stop <= video.shape[1], "Retrieved-frame slice exceeds model video")
    output = video.clone()
    sequence = output[:, start:stop].clone()
    before_sha = _tensor_sha256(sequence)
    rng = np.random.default_rng(int(inference_seed))
    parameters: dict[str, Any]

    if corruption_id == "frame_drop":
        rate = {"5pct": 0.05, "10pct": 0.10, "20pct": 0.20}[severity]
        count = max(1, int(np.ceil(rate * h_steps)))
        dropped = sorted(int(value) for value in rng.choice(np.arange(1, h_steps), size=min(count, h_steps - 1), replace=False))
        for index in dropped:
            sequence[:, index] = sequence[:, index - 1]
        parameters = {"rate": rate, "dropped_local_indices": dropped, "fill": "last_valid_frame_hold"}
    elif corruption_id == "timestamp_jitter":
        milliseconds = {"5ms": 5.0, "10ms": 10.0, "20ms": 20.0}[severity]
        positions = np.arange(h_steps, dtype=np.float64)
        positions += rng.uniform(-milliseconds, milliseconds, size=h_steps) * 30.0 / 1000.0
        positions = np.clip(positions, 0.0, h_steps - 1.0)
        left = np.floor(positions).astype(np.int64)
        right = np.ceil(positions).astype(np.int64)
        alpha = torch.as_tensor(positions - left, dtype=torch.float32, device=sequence.device).view(1, -1, 1, 1)
        mixed = sequence[:, left].float() * (1.0 - alpha) + sequence[:, right].float() * alpha
        sequence = mixed.round().clamp(0, 255).to(video.dtype)
        parameters = {"milliseconds": milliseconds, "nominal_hz": 30.0, "resampling": "linear_within_retrieved_segment"}
    elif corruption_id == "pause":
        seconds = {"0p2s": 0.2, "0p5s": 0.5, "1p0s": 1.0}[severity]
        repeat_count = min(h_steps - 1, max(1, int(round(seconds * 30.0))))
        pause_start = max(1, (h_steps - repeat_count) // 2)
        sequence[:, pause_start : pause_start + repeat_count] = sequence[:, pause_start - 1 : pause_start]
        parameters = {"seconds": seconds, "repeat_count": repeat_count, "pause_start_local_index": pause_start}
    elif corruption_id == "step_jump":
        steps = {"1": 1, "5": 5, "20": 20}[severity]
        pivot = max(1, h_steps // 2)
        indices = np.arange(h_steps, dtype=np.int64)
        indices[pivot:] = np.minimum(indices[pivot:] + steps, h_steps - 1)
        sequence = sequence[:, indices]
        parameters = {
            "jump_steps": steps,
            "pivot_local_index": pivot,
            "bounded_source_indices": indices.tolist(),
            "segment_crossing_count": 0,
        }
    else:
        raise InferenceContractError(f"Unknown temporal corruption: {corruption_id}")

    output[:, start:stop] = sequence
    after_sha = _tensor_sha256(sequence)
    _require(after_sha != before_sha, f"Temporal corruption did not alter frames: {corruption_id}_{severity}")
    transformed = dict(item)
    transformed["video"] = output
    receipt = {
        "schema_version": "human2robot-m5b-p2-temporal-transform-receipt-v1",
        "materialization_point": "uint8_model_video_input_before_tokenizer_encode",
        "corruption_id": corruption_id,
        "severity": severity,
        "inference_seed": int(inference_seed),
        "retrieved_frame_slice": [start, stop],
        "input_sha256": before_sha,
        "output_sha256": after_sha,
        "fixed_length_preserved": tuple(output.shape) == tuple(video.shape),
        "parameters": parameters,
    }
    return transformed, receipt


def resolution_visual_topk_by_query(
    workspace: Path,
    artifact_root: Path,
    binding: CellBinding,
    heldout_dataset: Any,
    heldout_kwargs: Mapping[str, Any],
    *,
    batch_size: int = 4,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Build resolution-specific diagnostic visual rankings without changing conditioning."""

    evaluation = binding.evaluation
    _require(evaluation is not None and evaluation.experiment_id == "M5B-RES-01", "Not a resolution cell")
    from tools.human2robot_m5b_p2_prepare import (
        SPLIT_SHA256,
        TOKENIZER_SHA256,
        encode_visual_features,
        feature_windows,
        read_feature_npz,
        write_npz_atomic,
    )

    cache_path = (
        artifact_root
        / "diagnostics/resolution_visual"
        / f"{evaluation.resolution_variant}__h{evaluation.h_steps}_k{evaluation.k_steps}.npz"
    )
    if cache_path.is_file():
        ids, features, manifest = read_feature_npz(cache_path)
        _require(manifest.get("resolution_variant") == evaluation.resolution_variant, "Resolution cache variant drift")
        _require(manifest.get("tokenizer_checkpoint_sha256") == TOKENIZER_SHA256, "Resolution cache tokenizer drift")
    else:
        train_kwargs = dict(heldout_kwargs)
        train_kwargs["split"] = "train"
        train_kwargs["use_image_aug"] = False
        train_dataset = build_human2robot_p2_dataset(**train_kwargs)
        records = feature_windows(train_dataset, heldout_dataset)
        encoded = encode_visual_features(
            records,
            batch_size=batch_size,
            resolution_variant=evaluation.resolution_variant,
        )
        ids = sorted(encoded)
        features = np.stack([encoded[item] for item in ids]).astype(np.float32)
        manifest = {
            "schema_version": "human2robot-m5b-p2-resolution-visual-cache-v1",
            "created_at_utc": utc_now(),
            "resolution_variant": evaluation.resolution_variant,
            "split_sha256": SPLIT_SHA256,
            "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
            "heldout_target_used": False,
            "candidate_policy": "same heldout P1 candidate pool",
            "feature_count": len(ids),
        }
        write_npz_atomic(cache_path, ids, features, manifest)

    diagnostic_kwargs = dict(heldout_kwargs)
    diagnostic_kwargs["retrieval_modality"] = "visual"
    diagnostic_kwargs["retrieval_index_path"] = cache_path
    diagnostic = build_human2robot_p2_dataset(**diagnostic_kwargs)
    rankings: dict[str, list[str]] = defaultdict(list)
    for example in diagnostic.examples:
        query_id = diagnostic.queries[example.query_index].window_id
        _require(example.candidate_index is not None, "Resolution visual ranking unexpectedly masked")
        rankings[query_id].append(diagnostic.candidates[example.candidate_index].window_id)
    _require(bool(rankings), "Resolution visual rankings are empty")
    _require(all(len(value) == evaluation.top_k for value in rankings.values()), "Resolution top-k cardinality drift")
    provenance = {
        "cache_path": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "cache_manifest": manifest,
        "query_count": len(rankings),
        "top_k": evaluation.top_k,
        "conditioning_retrieval_modality": evaluation.retrieval_modality,
        "diagnostic_retrieval_modality": "visual",
        "conditioning_changed": False,
    }
    return dict(rankings), provenance


def evaluate_dataset(
    binding: CellBinding,
    dataset: Any,
    statistics: Mapping[str, Any],
    predict_normalized: Callable[[Mapping[str, Any], int], np.ndarray | None],
    *,
    workspace_xyz_min: Sequence[float],
    workspace_xyz_max: Sequence[float],
    progress_callback: Callable[[int, int, str], None] | None = None,
    provenance_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = binding.evaluation
    _require(evaluation is not None, "Evaluation binding missing")
    grouped: dict[str, list[RankPrediction]] = defaultdict(list)
    for index in range(len(dataset)):
        item = dataset[index]
        query_id = str(item["query_id"])
        inference_seed = deterministic_inference_seed(
            evaluation.run_seed,
            evaluation.experiment_id,
            evaluation.variant_id,
            str(item["task"]),
            str(item["episode_id"]),
            int(item["current_row"]),
            int(item["retrieval_rank"]),
        )
        normalized = predict_normalized(item, inference_seed)
        if provenance_callback is not None:
            provenance_callback(item)
        grouped[query_id].append(
            RankPrediction(
                query_id=query_id,
                task=str(item["task"]),
                episode_id=str(item["episode_id"]),
                current_row=int(item["current_row"]),
                retrieval_rank=int(item["retrieval_rank"]),
                target_representation=str(item["target_representation"]),
                normalized_prediction=normalized,
                current_state_10d=np.asarray(item["raw_current_state"]),
                aligned_pool_10d=np.asarray(item["raw_aligned_pool"]),
                query_target_10d=np.asarray(item["raw_query_target"]),
                gap_crossing_count=int(item["gap_crossing_count"]),
                heldout_target_retrieval_feature_count=int(
                    item["heldout_target_retrieval_feature_count"]
                ),
            )
        )
    window_records = []
    for completed, query_id in enumerate(sorted(grouped), start=1):
        window_records.append(
            evaluate_ranked_query(
                grouped[query_id],
                statistics=statistics,
                workspace_xyz_min=workspace_xyz_min,
                workspace_xyz_max=workspace_xyz_max,
            )
        )
        if progress_callback is not None:
            progress_callback(completed, len(grouped), query_id)
    task_seed = aggregate_task_seed_windows(window_records, seed=evaluation.run_seed)
    return window_records, task_seed


def _nonlearned_payload_components(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": artifact["cell_id"],
        "linked_evaluation_cell_id": artifact["linked_evaluation_cell_id"],
        "immutable_manifest": artifact["immutable_manifest"],
        "dataset_contract": artifact["dataset_contract"],
        "retrieval_records": artifact["retrieval_records"],
        "window_records": artifact["window_records"],
        "task_seed_records": artifact["task_seed_records"],
    }


def build_nonlearned_artifact_contract(
    binding: CellBinding,
    linked_evaluation_binding: CellBinding,
    *,
    immutable_manifest: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    retrieval_records: Sequence[Mapping[str, Any]],
    window_records: Sequence[Mapping[str, Any]],
    task_seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evaluation = linked_evaluation_binding.evaluation
    _require(binding.cell.artifact_kind == "nonlearned_method_artifact", "Not a nonlearned cell")
    _require(evaluation is not None and not evaluation.requires_model_inference, "Linked evaluation is learned")
    _require(evaluation.parent_artifact_id == binding.cell.cell_id, "Linked evaluation parent mismatch")
    _require(bool(retrieval_records) and bool(window_records) and bool(task_seed_records), "Empty nonlearned output")
    artifact: dict[str, Any] = {
        "schema_version": "human2robot-m5b-p2-nonlearned-method-artifact-v1",
        "cell_id": binding.cell.cell_id,
        "status": "completed",
        "formal_result": True,
        "method_id": "retrieval_only",
        "optimizer_checkpoint": "not_applicable_by_frozen_nonlearned_definition",
        "linked_evaluation_cell_id": evaluation.cell_id,
        "immutable_manifest": dict(immutable_manifest),
        "dataset_contract": dict(dataset_contract),
        "retrieval_records": list(retrieval_records),
        "window_records": list(window_records),
        "task_seed_records": list(task_seed_records),
        "completed_at_utc": utc_now(),
    }
    artifact["retrieval_records_sha256"] = canonical_json_sha256(artifact["retrieval_records"])
    artifact["window_outputs_sha256"] = canonical_json_sha256(artifact["window_records"])
    artifact["artifact_payload_sha256"] = canonical_json_sha256(_nonlearned_payload_components(artifact))
    return artifact


def build_linked_retrieval_evaluation_artifact(
    binding: CellBinding,
    parent: Mapping[str, Any],
    *,
    runtime_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = binding.evaluation
    _require(evaluation is not None and not evaluation.requires_model_inference, "Not retrieval-only evaluation")
    _require(parent.get("schema_version") == "human2robot-m5b-p2-nonlearned-method-artifact-v1", "Bad parent schema")
    _require(parent.get("cell_id") == evaluation.parent_artifact_id, "Retrieval-only parent ID mismatch")
    _require(parent.get("linked_evaluation_cell_id") == binding.cell.cell_id, "Parent links another evaluation")
    expected_payload = canonical_json_sha256(_nonlearned_payload_components(parent))
    _require(parent.get("artifact_payload_sha256") == expected_payload, "Nonlearned parent payload hash mismatch")
    artifact: dict[str, Any] = {
        "schema_version": "human2robot-m5b-p2-evaluation-artifact-v1",
        "cell_id": binding.cell.cell_id,
        "status": "completed",
        "formal_result": True,
        "parent_artifact_id": evaluation.parent_artifact_id,
        "parent_artifact_payload_sha256": expected_payload,
        "evaluation_binding": evaluation.__dict__,
        "runtime_hash_manifest": dict(runtime_manifest),
        "dataset_contract": parent["dataset_contract"],
        "window_records": parent["window_records"],
        "task_seed_records": parent["task_seed_records"],
        "completed_at_utc": utc_now(),
    }
    artifact["evaluation_payload_sha256"] = canonical_json_sha256(
        {
            "parent_artifact_payload_sha256": expected_payload,
            "evaluation_binding": artifact["evaluation_binding"],
            "runtime_hash_manifest": artifact["runtime_hash_manifest"],
            "window_records": artifact["window_records"],
            "task_seed_records": artifact["task_seed_records"],
        }
    )
    return artifact


class CosmosCheckpointBackend:
    """Load the exact step-7000 DCP and expose one-rank normalized predictions."""

    def __init__(
        self,
        workspace: Path,
        binding: CellBinding,
        checkpoint_path: Path,
    ) -> None:
        evaluation = binding.evaluation
        _require(evaluation is not None and evaluation.checkpoint_cell_id is not None, "Model backend requires checkpoint")
        _require(checkpoint_path.is_dir(), f"Checkpoint directory missing: {checkpoint_path}")
        sampler = validate_formal_sampler_signature()
        _require(sampler["status"] == "passed", str(sampler["reason"]))
        config_path = workspace / "cosmos_policy/config/config.py"
        options = [
            f"experiment={binding.training_spec.config_name}",
            f"checkpoint.load_path={checkpoint_path}",
            "checkpoint.load_training_state=False",
            "checkpoint.load_ema_to_reg=True",
        ]
        config = load_config(str(config_path), options, enable_one_logger=True)
        config.checkpoint.load_path = str(checkpoint_path)
        config.checkpoint.load_training_state = False
        config.checkpoint.load_ema_to_reg = True
        with distributed_init():
            distributed.init()
        config.validate()
        self.trainer = config.trainer.type(config)
        with model_init():
            model = instantiate(config.model)
        model = model.to("cuda", memory_format=config.trainer.memory_format)
        model.on_train_start(config.trainer.memory_format)
        _require(float(model.config.shift) == FORMAL_SHIFT, "Resolved model shift is not 5.0")
        _require(
            bool(model.config.use_kerras_sigma_at_inference) is FORMAL_USE_KERRAS_SIGMA,
            "Resolved Karras-sigma setting changed",
        )
        optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
        iteration = self.trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        _require(int(iteration) == 7000, f"Loaded checkpoint iteration is {iteration}, expected 7000")
        model.eval()
        self.model = model
        self.k_steps = evaluation.k_steps

    @torch.inference_mode()
    def __call__(self, item: Mapping[str, Any], seed: int) -> np.ndarray:
        batch = misc.to(default_collate([item]), device="cuda")
        generated = self.model.generate_samples_from_batch(
            batch,
            n_sample=1,
            guidance=FORMAL_GUIDANCE,
            num_steps=FORMAL_NUM_STEPS,
            shift=FORMAL_SHIFT,
            seed=seed,
            is_negative_prompt=False,
            use_variance_scale=FORMAL_VARIANCE_SCALE,
        )
        direct = getattr(self.model, "_generated_action", None)
        if direct is not None:
            actions = direct.reshape(1, self.k_steps, 10)
            self.model._generated_action = None
        else:
            actions = extract_action_chunk_from_latent_sequence(
                generated,
                action_shape=(self.k_steps, 10),
                action_indices=batch["action_latent_idx"],
                decoder=getattr(self.model, "action_decoder", None),
            )
        return actions[0].to(torch.float32).cpu().numpy()


def _negative_control_artifact(
    workspace: Path,
    binding: CellBinding,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = binding.evaluation
    _require(evaluation is not None and evaluation.negative_control_detector, "Negative control binding missing")
    stress_path = workspace / "data/Human2Robot/derived/m5a_v03/action_role_stress.json"
    stress = read_json(stress_path)
    check_by_variant = {
        "same_frame_query_negative_control": "same_frame_copy",
        "swapped_role_negative_control": "wrong_role",
        "scale_x2_negative_control": "scale_x2",
    }


def _severe_temporal_rejection_artifact(
    binding: CellBinding,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = binding.evaluation
    _require(evaluation is not None and evaluation.corruption_id is not None, "Temporal binding missing")
    _require(evaluation.variant_id in SEVERE_TEMPORAL_VARIANTS, "Temporal variant is not severe")
    return {
        "schema_version": "human2robot-m5b-p2-evaluation-artifact-v2",
        "cell_id": binding.cell.cell_id,
        "status": "completed",
        "formal_result": True,
        "parent_artifact_id": evaluation.parent_artifact_id,
        "parent_model_payload_sha256": parent["model_payload_sha256"],
        "evaluation_binding": evaluation.__dict__,
        "pre_inference_status": "rejected",
        "model_call_count": 0,
        "rejection_receipt": {
            "schema_version": "human2robot-m5b-p2-temporal-pre-model-rejection-v1",
            "variant_id": evaluation.variant_id,
            "reason": "frozen severe temporal corruption policy",
            "materialization_point": "before dataset iteration and before tokenizer/model invocation",
            "masked_instead_of_rejected": False,
        },
        "window_records": [],
        "task_seed_records": [],
        "completed_at_utc": utc_now(),
    }
    check_id = check_by_variant[evaluation.variant_id]
    detector = stress.get("checks", {}).get(check_id, {})
    _require(detector.get("detector_triggered") is True, f"M5-A detector did not trigger: {check_id}")
    return {
        "schema_version": "human2robot-m5b-p2-evaluation-artifact-v1",
        "cell_id": binding.cell.cell_id,
        "status": "completed_detector_triggered_excluded",
        "formal_result": True,
        "parent_artifact_id": evaluation.parent_artifact_id,
        "parent_model_payload_sha256": parent["model_payload_sha256"],
        "detector_id": evaluation.negative_control_detector,
        "detector_source_path": str(stress_path),
        "detector_source_sha256": file_sha256(stress_path),
        "detector_evidence": detector,
        "metric_acceptance": False,
        "excluded_from_main_results": True,
        "task_seed_records": [],
        "completed_at_utc": utc_now(),
    }


def _linked_retrieval_evaluation(
    matrix: ExecutionMatrix, binding: CellBinding
) -> CellBinding:
    linked = [
        candidate
        for candidate in matrix.cells_of_kind("checkpoint_linked_evaluation")
        if candidate.evaluation is not None
        and candidate.evaluation.parent_artifact_id == binding.cell.cell_id
    ]
    _require(len(linked) == 1, f"Expected one linked retrieval-only evaluation, found {len(linked)}")
    evaluation = linked[0].evaluation
    _require(
        evaluation is not None
        and evaluation.method_id == "retrieval_only"
        and not evaluation.requires_model_inference,
        "Nonlearned cell is not linked to retrieval-only evaluation",
    )
    return linked[0]


def _materialize_nonlearned_artifact(
    workspace: Path,
    artifact_root: Path,
    matrix: ExecutionMatrix,
    binding: CellBinding,
    bounds: Mapping[str, Any],
) -> dict[str, Any]:
    linked = _linked_retrieval_evaluation(matrix, binding)
    kwargs = dataset_kwargs(workspace, linked)
    dataset = build_human2robot_p2_dataset(**kwargs)
    statistics = read_json(Path(kwargs["statistics_path"]))
    output_path = artifact_path(artifact_root, binding.cell.cell_id)
    progress_path = output_path.parent / "progress.json"
    provenance: list[dict[str, Any]] = []

    def progress(completed: int, total: int, query_id: str) -> None:
        write_json_atomic(
            progress_path,
            {
                "schema_version": "human2robot-m5b-p2-nonlearned-progress-v1",
                "cell_id": binding.cell.cell_id,
                "completed_queries": completed,
                "total_queries": total,
                "last_query_id": query_id,
                "retrieval_record_count": len(provenance),
                "updated_at_utc": utc_now(),
            },
        )

    windows, units = evaluate_dataset(
        linked,
        dataset,
        statistics,
        lambda _item, _seed: None,
        workspace_xyz_min=bounds["xyz_min"],
        workspace_xyz_max=bounds["xyz_max"],
        progress_callback=progress,
        provenance_callback=lambda item: provenance.append(retrieval_provenance_record(item)),
    )
    artifact = build_nonlearned_artifact_contract(
        binding,
        linked,
        immutable_manifest=immutable_runtime_manifest(workspace, linked),
        dataset_contract=dataset.contract_manifest(),
        retrieval_records=provenance,
        window_records=windows,
        task_seed_records=units,
    )
    write_json_atomic(output_path, artifact)
    return artifact


def run_cell(
    workspace: Path,
    artifact_root: Path,
    cell_id: str,
    activation_path: Path,
    workspace_bounds_path: Path,
) -> dict[str, Any]:
    matrix = load_execution_matrix(workspace)
    try:
        require_formal_activation(read_json(activation_path), matrix)
    except HandlerContractError as error:
        raise InferenceContractError(str(error)) from error
    _require(cell_id in matrix.bindings_by_id, f"Unknown cell: {cell_id}")
    binding = matrix.bindings_by_id[cell_id]
    _require(
        binding.cell.artifact_kind in {"nonlearned_method_artifact", "checkpoint_linked_evaluation"},
        f"Cell is not an inference artifact: {cell_id}",
    )
    bounds: dict[str, Any] | None = None
    if binding.cell.artifact_kind == "nonlearned_method_artifact":
        bounds = read_json(workspace_bounds_path)
        _require(bounds.get("status") == "frozen", "Workspace bounds are not frozen")
        return _materialize_nonlearned_artifact(
            workspace, artifact_root, matrix, binding, bounds
        )
    evaluation = binding.evaluation
    _require(evaluation is not None, "Evaluation binding missing")
    parent = read_json(artifact_path(artifact_root, evaluation.parent_artifact_id))
    _require(parent.get("status") == "completed", "Parent artifact is incomplete")
    if evaluation.negative_control_detector:
        artifact = _negative_control_artifact(workspace, binding, parent)
        write_json_atomic(artifact_path(artifact_root, cell_id), artifact)
        return artifact
    if evaluation.variant_id in SEVERE_TEMPORAL_VARIANTS:
        artifact = _severe_temporal_rejection_artifact(binding, parent)
        write_json_atomic(artifact_path(artifact_root, cell_id), artifact)
        return artifact

    if not evaluation.requires_model_inference:
        artifact = build_linked_retrieval_evaluation_artifact(
            binding,
            parent,
            runtime_manifest=immutable_runtime_manifest(workspace, binding),
        )
        write_json_atomic(artifact_path(artifact_root, cell_id), artifact)
        return artifact

    bounds = read_json(workspace_bounds_path)
    _require(bounds.get("status") == "frozen", "Workspace bounds are not frozen")
    kwargs = dataset_kwargs(workspace, binding)
    dataset = build_human2robot_p2_dataset(**kwargs)
    statistics = read_json(Path(kwargs["statistics_path"]))
    visual_topk: dict[str, list[str]] | None = None
    resolution_provenance: dict[str, Any] | None = None
    if evaluation.experiment_id == "M5B-RES-01":
        visual_topk, resolution_provenance = resolution_visual_topk_by_query(
            workspace,
            artifact_root,
            binding,
            dataset,
            kwargs,
        )
    backend: Callable[[Mapping[str, Any], int], np.ndarray | None] = CosmosCheckpointBackend(
        workspace, binding, Path(parent["checkpoint_path"])
    )
    temporal_receipts: list[dict[str, Any]] = []
    predict: Callable[[Mapping[str, Any], int], np.ndarray | None] = backend
    if evaluation.corruption_id is not None:
        _require(evaluation.corruption_severity is not None, "Temporal severity missing")

        def predict(item: Mapping[str, Any], seed: int) -> np.ndarray | None:
            transformed, receipt = apply_temporal_corruption(
                item,
                corruption_id=evaluation.corruption_id,
                severity=evaluation.corruption_severity,
                inference_seed=seed,
                h_steps=evaluation.h_steps,
            )
            receipt["query_id"] = str(item["query_id"])
            receipt["retrieval_rank"] = int(item["retrieval_rank"])
            temporal_receipts.append(receipt)
            return backend(transformed, seed)
    output_path = artifact_path(artifact_root, cell_id)
    progress_path = output_path.parent / "progress.json"

    def progress(completed: int, total: int, query_id: str) -> None:
        write_json_atomic(
            progress_path,
            {
                "schema_version": "human2robot-m5b-p2-evaluation-progress-v1",
                "cell_id": cell_id,
                "completed_queries": completed,
                "total_queries": total,
                "last_query_id": query_id,
                "updated_at_utc": utc_now(),
            },
        )

    windows, units = evaluate_dataset(
        binding,
        dataset,
        statistics,
        predict,
        workspace_xyz_min=bounds["xyz_min"],
        workspace_xyz_max=bounds["xyz_max"],
        progress_callback=progress,
    )
    artifact = {
        "schema_version": "human2robot-m5b-p2-evaluation-artifact-v2",
        "cell_id": cell_id,
        "status": "completed",
        "formal_result": True,
        "parent_artifact_id": evaluation.parent_artifact_id,
        "parent_model_payload_sha256": parent["model_payload_sha256"],
        "evaluation_binding": evaluation.__dict__,
        "runtime_hash_manifest": immutable_runtime_manifest(workspace, binding),
        "dataset_contract": dataset.contract_manifest(),
        "formal_inference": preflight(workspace)["formal_parameters"],
        "window_records": windows,
        "task_seed_records": units,
        "pre_inference_status": "transformed" if evaluation.corruption_id else "clean",
        "temporal_transform_receipts": temporal_receipts,
        "visual_topk_by_query": visual_topk,
        "resolution_visual_ranking_provenance": resolution_provenance,
        "completed_at_utc": utc_now(),
    }
    write_json_atomic(output_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run = subparsers.add_parser("run-cell")
    run.add_argument("--cell-id", required=True)
    run.add_argument("--activation-path", type=Path, default=DEFAULT_ACTIVATION_PATH)
    run.add_argument("--workspace-bounds-path", type=Path, default=DEFAULT_WORKSPACE_BOUNDS_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    if args.command == "preflight":
        result = preflight(workspace)
    else:
        result = run_cell(
            workspace,
            args.artifact_root.resolve(),
            args.cell_id,
            args.activation_path.resolve(),
            args.workspace_bounds_path.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InferenceContractError as error:
        print(f"M5B-P2 inference error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
