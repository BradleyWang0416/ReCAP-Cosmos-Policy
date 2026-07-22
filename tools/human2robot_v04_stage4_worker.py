#!/usr/bin/env python3
"""Four-rank read-only stage-4 inference worker for one frozen v03 checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from cosmos_policy.datasets.human2robot_dataset import _normalize, _preprocess_video, align_pool_chunk
from cosmos_policy.datasets.human2robot_v04_retrieval import poses_euler_to_10d
from tools import human2robot_v04_stage4 as stage4
from tools.human2robot_m5b_p2_inference import deterministic_inference_seed
from tools.human2robot_m5b_p2_step_checkpoint_diagnostic import IntermediateCheckpointBackend


SCHEMA = "human2robot-v04-stage4-smoke-worker-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise stage4.Stage4Error(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        require(stage4.read_json(path) == dict(value), f"Existing smoke receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(partial.read_text(encoding="utf-8"))
        os.replace(partial, path)
        path.chmod(0o444)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()


def _method_statistics(workspace: Path, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cell = f"learned_training_checkpoint__M5B-MAIN-01__frozen_main__{method}__seed20260711"
    path = workspace / "data/Human2Robot/derived/m5b_v03/p2_prepared_v2/statistics" / f"{cell}.json"
    payload = stage4.read_json(path)
    provenance = payload.get("provenance", {})
    require(provenance.get("method_id") == method, f"v03 statistics method drift for {method}")
    require(provenance.get("heldout_data_used") is False, f"v03 statistics used heldout data for {method}")
    return payload, stage4.bind_file(path)


def _read_query(query_record: Mapping[str, Any], start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(str(query_record["projection"]["path"]))
    history = np.arange(start, start + stage4.H_STEPS, dtype=np.int64)
    future = np.arange(start + stage4.H_STEPS, start + stage4.H_STEPS + stage4.K_STEPS, dtype=np.int64)
    with h5py.File(path, "r") as file:
        demo = file["data/demo_0"]
        pose = np.asarray(demo["robot/observed_eef_pose_6d"][:], dtype=np.float64)
        gripper = np.asarray(demo["robot/gripper_state"][:], dtype=np.float64)
        states = poses_euler_to_10d(pose, gripper)
        images = demo["robot/images"]
        current_image = np.asarray(images[int(history[-1])], dtype=np.uint8)
        future_image = np.asarray(images[int(future[-1])], dtype=np.uint8)
        segments = np.asarray(demo["time/segment_id"][start : start + stage4.H_STEPS + stage4.K_STEPS])
    require(len(set(segments.tolist())) == 1, f"Query window crosses a gap: {path}:{start}")
    return states[int(history[-1])], states[future], np.stack((current_image, future_image))


def _read_candidate(candidate_record: Mapping[str, Any], start: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(candidate_record["projection"]["path"]))
    future = np.arange(start + stage4.H_STEPS, start + stage4.H_STEPS + stage4.K_STEPS, dtype=np.int64)
    with h5py.File(path, "r") as file:
        demo = file["data/demo_0"]
        action = np.asarray(demo["human/hand_action_7d"][:], dtype=np.float64)
        states = poses_euler_to_10d(action[:, :6], action[:, 6])
        images = np.stack([np.asarray(demo["human/images"][int(row)], dtype=np.uint8) for row in future])
        segments = np.asarray(demo["time/segment_id"][start : start + stage4.H_STEPS + stage4.K_STEPS])
    require(len(set(segments.tolist())) == 1, f"Candidate window crosses a gap: {path}:{start}")
    return states[future], images


def build_model_item(
    query: Mapping[str, Any],
    rank: Mapping[str, Any],
    method: str,
    statistics: Mapping[str, Any],
    *,
    protocol_file_sha256: str,
) -> dict[str, Any]:
    require(len(protocol_file_sha256) == 64, "Stage-4 smoke protocol SHA is not bound")
    current, target, robot_images = _read_query(query["query_record"], int(query["query_start"]))
    if method == "no_retrieval":
        raw_plan = np.repeat(current[None], stage4.H_STEPS, axis=0)
        human_images = np.zeros((stage4.H_STEPS, 240, 426, 3), dtype=np.uint8)
        has_retrieval = 0
    else:
        raw_plan, human_images = _read_candidate(rank["candidate_record"], int(rank["candidate_start"]))
        has_retrieval = 1
    aligned = align_pool_chunk(raw_plan, current)
    if method == "recap_hand_ret":
        actions = _normalize(target - aligned[: stage4.K_STEPS], statistics["residual_10d_min"], statistics["residual_10d_max"])
        target_representation = "residual"
    else:
        actions = _normalize(target, statistics["query_bc_target_10d_min"], statistics["query_bc_target_10d_max"])
        target_representation = "absolute"
    pool = _normalize(aligned, statistics["pool_action_10d_min"], statistics["pool_action_10d_max"])
    if not has_retrieval:
        pool = np.zeros_like(pool)
    current_normalized = _normalize(current, statistics["query_bc_target_10d_min"], statistics["query_bc_target_10d_max"])
    future_normalized = _normalize(target[-1], statistics["query_bc_target_10d_min"], statistics["query_bc_target_10d_max"])
    current_image, future_image = robot_images
    blank = np.zeros_like(current_image)
    blank4 = np.repeat(blank[None], 4, axis=0)
    frames = np.concatenate(
        (
            blank[None],
            human_images,
            blank4,
            blank4,
            np.repeat(current_image[None], 4, axis=0),
            blank4,
            blank4,
            np.repeat(future_image[None], 4, axis=0),
            blank4,
        ),
        axis=0,
    )
    require(len(frames) == 37, "Stage-4 WAN frame layout mismatch")
    video = _preprocess_video(frames, 224, None)
    ret_state_idx = 3
    return {
        "video": video,
        "actions": torch.from_numpy(actions),
        "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16),
        "t5_text_mask": torch.zeros(512, dtype=torch.int64),
        "fps": 30,
        "padding_mask": torch.zeros(1, 224, 224),
        "image_size": 224 * torch.ones(4),
        "proprio": torch.from_numpy(current_normalized),
        "future_proprio": torch.from_numpy(future_normalized),
        "__key__": int(query["query_start"]),
        "action_latent_idx": ret_state_idx + 4,
        "value_latent_idx": -1,
        "current_proprio_latent_idx": ret_state_idx + 3,
        "current_wrist_image_latent_idx": -1,
        "current_image_latent_idx": ret_state_idx + 2,
        "future_proprio_latent_idx": ret_state_idx + 6,
        "future_wrist_image_latent_idx": -1,
        "future_image_latent_idx": ret_state_idx + 5,
        "retrieved_video_start_latent_idx": 1,
        "retrieved_video_end_latent_idx": ret_state_idx,
        "retrieved_action_latent_idx": ret_state_idx + 1,
        "retrieved_actions": torch.from_numpy(pool),
        "retrieved_proprio": torch.from_numpy(pool[0]),
        "retrieved_state_latent_idx": ret_state_idx,
        "has_ret_data": has_retrieval,
        "has_ret_image": has_retrieval,
        "has_current_image": 1,
        "rollout_data_mask": 0,
        "rollout_data_success_mask": 0,
        "world_model_sample_mask": 0,
        "value_function_sample_mask": 0,
        "global_rollout_idx": -1,
        "value_function_return": -100.0,
        "next_action_chunk": torch.from_numpy(actions.copy()),
        "next_value_function_return": -100.0,
        "method_id": method,
        "experiment_id": "M5B-MAIN-01",
        "variant_id": "frozen_main",
        "target_representation": target_representation,
        "H_steps": stage4.H_STEPS,
        "K_steps": stage4.K_STEPS,
        "strict_future_offset_view_steps": 1,
        "gap_crossing_count": 0,
        "heldout_target_retrieval_feature_count": 0,
        "query_command_status": "unverified",
        "deployment_command_adapter_id": "",
        "protocol_file_sha256": protocol_file_sha256,
    }


def _config_name(method: str) -> str:
    return f"cosmos_predict2p5_2b_human2robot_{method}_seed20260711"


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.method in stage4.METHODS, f"Unsupported stage-4 method: {args.method}")
    require(Path("/.dockerenv").is_file(), "Stage-4 smoke worker requires Docker")
    require(torch.cuda.is_available(), "Stage-4 smoke worker requires CUDA")
    require(int(os.environ.get("WORLD_SIZE", "0")) == 4, "Stage-4 smoke worker requires torchrun world size 4")
    plan = stage4.read_json(args.smoke_plan)
    require(plan.get("query_count") == stage4.EXPECTED_QUERY_COUNT, "Stage-4 smoke plan query count drift")
    statistics, statistics_binding = _method_statistics(args.workspace, args.method)
    protocol_file_sha256 = stage4.file_sha256(
        args.workspace / "方案/v03/M5B_formal_acceptance_protocol_v1.json"
    )
    evaluation = SimpleNamespace(checkpoint_cell_id=f"stage4_old_{args.method}", k_steps=stage4.K_STEPS)
    training_spec = SimpleNamespace(config_name=_config_name(args.method))
    binding = SimpleNamespace(evaluation=evaluation, training_spec=training_spec)
    # Reuse the checkpoint diagnostic backend: unlike the legacy inference
    # backend it passes Hydra a package-relative config path, validates the
    # Human2Robot sampler adapter, and handles DCP's valid zero iteration report.
    backend = IntermediateCheckpointBackend(
        args.workspace,
        binding,
        args.checkpoint,
        expected_iteration=7000,
    )
    rank_id = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ["RANK"])
    receipt_paths: list[Path] = []
    completed = 0
    for query_index, query in enumerate(plan["queries"]):
        for retrieval_rank, rank in enumerate(query["ranks"]):
            receipt_path = args.output_root / "receipts" / f"q{query_index:04d}_r{retrieval_rank}.json"
            exists = receipt_path.is_file()
            flags = [None for _ in range(4)]
            torch.distributed.all_gather_object(flags, exists)
            require(all(flag is exists for flag in flags), f"Ranks disagree about receipt existence: {receipt_path}")
            if exists:
                receipt = stage4.read_json(receipt_path)
                require(receipt.get("status") == "PASSED" and receipt.get("finite") is True, f"Invalid existing receipt: {receipt_path}")
            else:
                item = build_model_item(
                    query,
                    rank,
                    args.method,
                    statistics,
                    protocol_file_sha256=protocol_file_sha256,
                )
                seed = deterministic_inference_seed(
                    stage4.RUN_SEED,
                    "V04-STAGE4-SMOKE",
                    args.method,
                    str(query["task"]),
                    str(query["episode_id"]),
                    int(query["query_start"] + stage4.H_STEPS - 1),
                    retrieval_rank,
                )
                prediction = np.asarray(backend(item, seed), dtype=np.float32)
                finite = bool(np.all(np.isfinite(prediction)))
                require(prediction.shape == (stage4.K_STEPS, 10), f"Smoke prediction shape mismatch: {prediction.shape}")
                require(finite, f"Nonfinite smoke prediction: {query['query_id']} rank {retrieval_rank}")
                receipt = {
                    "schema_version": SCHEMA,
                    "status": "PASSED",
                    "formal_result": False,
                    "performance_claim_allowed": False,
                    "method": args.method,
                    "seed": seed,
                    "checkpoint_path": str(args.checkpoint.resolve()),
                    "checkpoint_payload_sha256": args.checkpoint_payload_sha256,
                    "statistics": statistics_binding,
                    "query_id": query["query_id"],
                    "query_source_sha256": query["query_record"]["source_sha256"],
                    "query_partition": query["query_record"]["source_partition"],
                    "query_start": query["query_start"],
                    "candidate_id": rank["retrieval"]["candidate_id"],
                    "candidate_source_sha256": rank["candidate_record"]["source_sha256"],
                    "candidate_partition": rank["candidate_record"]["source_partition"],
                    "candidate_start": rank["candidate_start"],
                    "retrieval_rank": retrieval_rank,
                    "retrieval_record_sha256": canonical_sha256(rank["retrieval"]),
                    "prediction_shape": list(prediction.shape),
                    "prediction": prediction.tolist(),
                    "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                    "prediction_min": float(prediction.min()),
                    "prediction_max": float(prediction.max()),
                    "finite": finite,
                    "gap_crossing_count": 0,
                    "retrieval_future_rows_read": 0,
                    "retrieval_target_datasets_read": 0,
                }
                if rank_id == 0:
                    write_json_atomic(receipt_path, receipt)
                torch.distributed.barrier()
            receipt_paths.append(receipt_path)
            completed += 1
            if rank_id == 0 and completed % 10 == 0:
                print(json.dumps({"event": "smoke_progress", "method": args.method, "completed": completed, "total": stage4.EXPECTED_RECEIPTS_PER_METHOD}), flush=True)
    require(completed == stage4.EXPECTED_RECEIPTS_PER_METHOD, f"Smoke receipt cardinality mismatch: {completed}")
    torch.distributed.barrier()
    if rank_id == 0:
        bindings = [stage4.bind_file(path) for path in receipt_paths]
        summary = {
            "schema_version": f"{SCHEMA}-summary",
            "status": "PASSED",
            "formal_result": False,
            "performance_claim_allowed": False,
            "method": args.method,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_payload_sha256": args.checkpoint_payload_sha256,
            "smoke_plan": stage4.bind_file(args.smoke_plan),
            "statistics": statistics_binding,
            "query_count": stage4.EXPECTED_QUERY_COUNT,
            "receipt_count": len(bindings),
            "receipt_bundle_sha256": canonical_sha256(bindings),
            "finite_count": len(bindings),
            "nonfinite_count": 0,
            "missing_receipt_count": 0,
            "gap_crossing_count": 0,
            "provenance_violation_count": 0,
            "training_started": False,
        }
        write_json_atomic(args.output_root / "summary.json", summary)
    torch.distributed.barrier()
    return {"status": "PASSED", "method": args.method, "receipt_count": completed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--smoke-plan", type=Path, required=True)
    parser.add_argument("--method", choices=stage4.METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-payload-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except stage4.Stage4Error as error:
        print(f"stage-4 worker error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
