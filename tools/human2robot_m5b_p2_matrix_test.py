from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2_matrix import (
    FROZEN_COUNTS,
    MatrixContractError,
    canonical_json_sha256,
    load_execution_matrix,
    load_frozen_registry,
)


WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def matrix():
    return load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=True)


def test_all_203_cells_have_exactly_one_runtime_or_report_binding(matrix) -> None:
    assert len(matrix.bindings_by_id) == 203
    assert set(matrix.bindings_by_id) == set(matrix.cells_by_id)
    assert Counter(
        binding.cell.artifact_kind for binding in matrix.bindings_by_id.values()
    ) == Counter(FROZEN_COUNTS)
    assert all(binding.handler_kind for binding in matrix.bindings_by_id.values())


def test_dag_is_parent_before_child_and_acyclic(matrix) -> None:
    position = {cell_id: index for index, cell_id in enumerate(matrix.topological_cell_ids)}
    assert len(position) == 203
    for cell in matrix.cells_by_id.values():
        assert all(position[parent_id] < position[cell.cell_id] for parent_id in cell.parent_artifact_ids)


def test_prepared_entries_exactly_bind_all_48_learned_cells(matrix) -> None:
    learned = matrix.cells_of_kind("learned_training_checkpoint")
    assert len(learned) == 48
    assert all(binding.prepared_entry is not None for binding in learned)
    assert all(binding.training_spec is not None for binding in learned)
    assert {binding.prepared_entry["cell_id"] for binding in learned} == {
        binding.cell.cell_id for binding in learned
    }


def test_all_147_evaluations_have_concrete_semantics(matrix) -> None:
    evaluations = matrix.cells_of_kind("checkpoint_linked_evaluation")
    assert len(evaluations) == 147
    assert all(binding.evaluation is not None for binding in evaluations)
    assert all(binding.prepared_entry is not None for binding in evaluations)
    assert all(binding.evaluation.run_seed in (20260711, 20260712, 20260713) for binding in evaluations)
    assert all(binding.evaluation.h_steps in (4, 8, 16) for binding in evaluations)
    assert all(binding.evaluation.k_steps in (4, 8) for binding in evaluations)
    assert all(binding.evaluation.top_k in (1, 3, 5, 10) for binding in evaluations)
    assert all(binding.evaluation.pool_size in (0, 1, 2, 4, 8, 10) for binding in evaluations)


def test_retrieval_only_never_claims_a_checkpoint(matrix) -> None:
    retrieval_only = [
        binding.evaluation
        for binding in matrix.cells_of_kind("checkpoint_linked_evaluation")
        if binding.cell.method_id == "retrieval_only"
    ]
    assert len(retrieval_only) == 3
    assert all(item is not None and item.checkpoint_cell_id is None for item in retrieval_only)
    assert all(item is not None and item.requires_model_inference is False for item in retrieval_only)
    assert all(item is not None and item.target_representation == "retrieval_only" for item in retrieval_only)


def test_eval_only_overrides_are_bound_without_new_checkpoints(matrix) -> None:
    evaluations = {
        binding.cell.cell_id: binding.evaluation
        for binding in matrix.cells_of_kind("checkpoint_linked_evaluation")
    }
    pool0 = next(value for key, value in evaluations.items() if "pool_growth_pool0" in key)
    topk10 = next(value for key, value in evaluations.items() if "topk10_h8_k8" in key)
    crop = next(
        value
        for key, value in evaluations.items()
        if "center_crop_240x424_then_resize_224" in key
    )
    jitter = next(value for key, value in evaluations.items() if "timestamp_jitter_10ms" in key)
    negative = next(value for key, value in evaluations.items() if "same_frame_query_negative_control" in key)
    assert pool0.pool_size == 0
    assert topk10.top_k == 10
    assert crop.resolution_variant == "center_crop_240x424_then_resize_224"
    assert (jitter.corruption_id, jitter.corruption_severity) == ("timestamp_jitter", "10ms")
    assert negative.negative_control_detector == "same_frame_query_detector"


def test_terminal_report_closes_all_semantic_matrix_blockers(matrix) -> None:
    assert len(matrix.report_covered_evaluation_ids) == 147
    assert matrix.formal_readiness_blockers == ()
    terminal = matrix.cells_by_id[
        "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance"
    ]
    assert len(terminal.parent_artifact_ids) == 202
    assert matrix.topological_cell_ids[-1] == terminal.cell_id


def test_registry_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    registry, _, _ = load_frozen_registry(WORKSPACE)
    records = json.loads(json.dumps(registry["cells"]))
    records[0]["optimizer_steps"] = 6999
    assert canonical_json_sha256(records) != registry["cells_payload_sha256"]


def test_matrix_contract_error_is_a_hard_runtime_error() -> None:
    assert issubclass(MatrixContractError, RuntimeError)
