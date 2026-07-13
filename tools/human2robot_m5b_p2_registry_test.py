from __future__ import annotations

from tools.human2robot_m5b_p2_registry import (
    FORMAL_SEEDS,
    REQUIRED_EXPERIMENT_IDS,
    STATUS,
    build_candidate_registry,
    candidate_cells,
)


def test_candidate_registry_has_exact_nonformal_artifact_counts() -> None:
    registry = build_candidate_registry()
    assert registry["status"] == STATUS
    assert registry["formal_queue_allowed"] is False
    assert registry["p2_acceptance_allowed"] is False
    assert registry["counts"] == {
        "learned_training_checkpoint": 48,
        "nonlearned_method_artifact": 3,
        "checkpoint_linked_evaluation": 147,
        "aggregate_report": 4,
    }
    assert registry["cell_count"] == 202


def test_candidate_registry_ids_are_unique_and_cover_every_experiment() -> None:
    cells = candidate_cells()
    assert len({cell.cell_id for cell in cells}) == len(cells)
    assert {cell.experiment_id for cell in cells} == set(REQUIRED_EXPERIMENT_IDS)
    assert all(cell.status == STATUS and cell.formal_result is False for cell in cells)


def test_every_seeded_candidate_uses_one_of_the_three_frozen_seeds() -> None:
    assert all(cell.seed is None or cell.seed in FORMAL_SEEDS for cell in candidate_cells())


def test_learned_and_nonlearned_artifact_contracts_do_not_collapse() -> None:
    cells = candidate_cells()
    learned = [cell for cell in cells if cell.artifact_kind == "learned_training_checkpoint"]
    nonlearned = [cell for cell in cells if cell.artifact_kind == "nonlearned_method_artifact"]
    assert all(cell.optimizer_steps == 7000 for cell in learned)
    assert all(cell.method_id != "retrieval_only" for cell in learned)
    assert all(cell.optimizer_steps is None for cell in nonlearned)
    assert {cell.method_id for cell in nonlearned} == {"retrieval_only"}


def test_every_evaluation_parent_resolves_to_a_primary_artifact() -> None:
    cells = candidate_cells()
    primary_ids = {
        cell.cell_id
        for cell in cells
        if cell.artifact_kind in {"learned_training_checkpoint", "nonlearned_method_artifact"}
    }
    evaluations = [
        cell for cell in cells if cell.artifact_kind == "checkpoint_linked_evaluation"
    ]
    assert all(len(cell.parent_artifact_ids) == 1 for cell in evaluations)
    assert all(cell.parent_artifact_ids[0] in primary_ids for cell in evaluations)


def test_every_candidate_parent_id_resolves_inside_the_registry() -> None:
    cells = candidate_cells()
    cell_ids = {cell.cell_id for cell in cells}
    assert all(
        parent_id in cell_ids
        for cell in cells
        for parent_id in cell.parent_artifact_ids
    )


def test_aggregate_qualitative_report_binds_all_seed_reports() -> None:
    reports = [cell for cell in candidate_cells() if cell.artifact_kind == "aggregate_report"]
    seed_reports = [cell for cell in reports if cell.seed is not None]
    aggregate = [cell for cell in reports if cell.seed is None]
    assert len(seed_reports) == 3
    assert len(aggregate) == 1
    assert all(len(cell.parent_artifact_ids) == 9 for cell in seed_reports)
    assert set(aggregate[0].parent_artifact_ids) == {cell.cell_id for cell in seed_reports}
