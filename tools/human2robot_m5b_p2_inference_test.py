from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.human2robot_m5b_p2_inference import (
    InferenceContractError,
    apply_temporal_corruption,
    build_linked_retrieval_evaluation_artifact,
    build_nonlearned_artifact_contract,
    dataset_kwargs,
    evaluate_dataset,
    immutable_runtime_manifest,
    preflight,
    validate_formal_sampler_signature,
)
from tools.human2robot_m5b_p2_matrix import load_execution_matrix


WORKSPACE = Path(__file__).resolve().parents[1]


def test_native_rectified_flow_signature_is_explicit_and_legacy_solver_absent() -> None:
    result = validate_formal_sampler_signature()
    assert result["scheduler"] == "native_rectified_flow_scheduler"
    assert result["shift"] == 5.0
    assert result["native_parameters_explicit"] is True
    assert result["status"] == "passed"
    assert result["reason"] is None


def test_preflight_does_not_load_model_or_open_queue() -> None:
    result = preflight(WORKSPACE)
    assert result["status"] == "passed"
    assert result["formal_queue_allowed"] is False
    assert result["blockers"] == []


@pytest.mark.parametrize(
    ("corruption_id", "severity"),
    [
        ("frame_drop", "5pct"),
        ("timestamp_jitter", "5ms"),
        ("pause", "0p2s"),
        ("step_jump", "5"),
    ],
)
def test_temporal_corruptions_modify_actual_pretokenizer_frames_and_preserve_shape(
    corruption_id: str, severity: str
) -> None:
    video = torch.arange(3 * 37 * 2 * 2, dtype=torch.int32).remainder(256).to(torch.uint8).reshape(3, 37, 2, 2)
    transformed, receipt = apply_temporal_corruption(
        {"video": video},
        corruption_id=corruption_id,
        severity=severity,
        inference_seed=1234,
        h_steps=8,
    )
    assert transformed["video"].shape == video.shape
    assert not torch.equal(transformed["video"][:, 1:9], video[:, 1:9])
    assert torch.equal(transformed["video"][:, 9:], video[:, 9:])
    assert receipt["materialization_point"] == "uint8_model_video_input_before_tokenizer_encode"
    assert receipt["fixed_length_preserved"] is True


def test_dataset_kwargs_bind_heldout_and_disable_augmentation() -> None:
    matrix = load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)
    binding = next(
        item
        for item in matrix.cells_of_kind("checkpoint_linked_evaluation")
        if item.cell.experiment_id == "M5B-RES-01"
        and item.cell.variant_id == "center_crop_240x424_then_resize_224"
    )
    kwargs = dataset_kwargs(WORKSPACE, binding)
    assert kwargs["split"] == "heldout"
    assert kwargs["use_image_aug"] is False
    assert kwargs["resolution_variant"] == binding.cell.variant_id
    assert kwargs["statistics_path"].is_file()
    assert kwargs["retrieval_index_path"].is_file()


class FakeDataset:
    def __init__(self) -> None:
        self.items = []
        for task_index in range(4):
            for query_index in range(2):
                target = np.zeros((8, 10), dtype=np.float32)
                target[:, 3] = 1.0
                target[:, 7] = 1.0
                aligned = target.copy()
                for rank in range(3):
                    self.items.append(
                        {
                            "query_id": f"task{task_index}:q{query_index}",
                            "task": f"task{task_index}",
                            "episode_id": f"episode{task_index}",
                            "current_row": query_index + 8,
                            "retrieval_rank": rank,
                            "target_representation": "residual",
                            "raw_current_state": target[0],
                            "raw_aligned_pool": aligned,
                            "raw_query_target": target,
                            "gap_crossing_count": 0,
                            "heldout_target_retrieval_feature_count": 0,
                        }
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def test_evaluate_dataset_uses_one_deterministic_prediction_per_rank() -> None:
    matrix = load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)
    binding = next(
        item
        for item in matrix.cells_of_kind("checkpoint_linked_evaluation")
        if item.cell.cell_id.endswith("main_comparison_pool10__recap_hand_ret__seed20260711")
    )
    seen_seeds = []

    def predict(_item, seed):
        seen_seeds.append(seed)
        return np.zeros((8, 10), dtype=np.float32)

    stats = {
        "residual_10d_min": [-1e-8] * 10,
        "residual_10d_max": [-1e-8] * 10,
        "residual_norm_p99": 1.0,
    }
    windows, units = evaluate_dataset(
        binding,
        FakeDataset(),
        stats,
        predict,
        workspace_xyz_min=(-1, -1, -1),
        workspace_xyz_max=(1, 1, 1),
    )
    assert len(seen_seeds) == 24
    assert len(set(seen_seeds)) == 24
    assert len(windows) == 8
    assert len(units) == 4


def test_nonlearned_artifact_and_linked_evaluation_form_a_hashed_two_stage_contract() -> None:
    matrix = load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)
    parent_binding = next(
        item
        for item in matrix.cells_of_kind("nonlearned_method_artifact")
        if item.cell.cell_id.endswith("seed20260711")
    )
    child_binding = next(
        item
        for item in matrix.cells_of_kind("checkpoint_linked_evaluation")
        if item.evaluation is not None
        and item.evaluation.parent_artifact_id == parent_binding.cell.cell_id
    )
    retrieval = [{"query_id": "q0", "retrieval_rank": rank} for rank in range(3)]
    windows = [{"query_id": "q0", "metrics": {"position_error_median_canonical": 0.1}}]
    units = [{"task": "task0", "seed": 20260711, "position_error_median_canonical": 0.1}]
    runtime = immutable_runtime_manifest(WORKSPACE, child_binding)
    parent = build_nonlearned_artifact_contract(
        parent_binding,
        child_binding,
        immutable_manifest=runtime,
        dataset_contract={"split": "heldout", "method_id": "retrieval_only"},
        retrieval_records=retrieval,
        window_records=windows,
        task_seed_records=units,
    )
    child = build_linked_retrieval_evaluation_artifact(
        child_binding, parent, runtime_manifest=runtime
    )
    assert parent["optimizer_checkpoint"] == "not_applicable_by_frozen_nonlearned_definition"
    assert len(parent["artifact_payload_sha256"]) == 64
    assert child["parent_artifact_payload_sha256"] == parent["artifact_payload_sha256"]
    assert child["window_records"] == windows
    assert child["task_seed_records"] == units

    tampered = dict(parent)
    tampered["window_records"] = [{"query_id": "tampered"}]
    with pytest.raises(InferenceContractError, match="payload hash mismatch"):
        build_linked_retrieval_evaluation_artifact(
            child_binding, tampered, runtime_manifest=runtime
        )
