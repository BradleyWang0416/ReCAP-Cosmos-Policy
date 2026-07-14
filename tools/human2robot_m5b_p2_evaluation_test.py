from __future__ import annotations

import numpy as np
import pytest

from tools.human2robot_m5b_p2_evaluation import (
    EvaluationContractError,
    RankPrediction,
    aggregate_task_seed_windows,
    exact_one_sided_sign_flip_p,
    guardrail_gate,
    has_long_term_residual_saturation,
    holm_adjust_two,
    main_gate_analysis,
    paired_primary_analysis,
    evaluate_ranked_query,
    reconstruct_rank_prediction,
)


def statistics() -> dict:
    return {
        "residual_10d_min": [-1.0] * 10,
        "residual_10d_max": [1.0] * 10,
        "query_bc_target_10d_min": [-2.0] * 10,
        "query_bc_target_10d_max": [2.0] * 10,
        "future_state_transition_10d_min": [-0.5] * 10,
        "future_state_transition_10d_max": [0.5] * 10,
        "residual_norm_p99": 1.0,
    }


def canonical_state(x: float) -> np.ndarray:
    value = np.zeros(10, dtype=np.float32)
    value[0] = x
    value[3] = 1.0
    value[7] = 1.0
    value[9] = 0.5
    return value


@pytest.mark.parametrize("representation", ["residual", "absolute", "future_state"])
def test_reconstruction_contracts_produce_canonical_k_by_10(representation: str) -> None:
    current = canonical_state(0.0)
    aligned = np.stack([canonical_state(0.1 * index) for index in range(8)])
    prediction, raw = reconstruct_rank_prediction(
        np.zeros((8, 10), dtype=np.float32),
        target_representation=representation,
        statistics=statistics(),
        current_state_10d=current,
        aligned_pool_10d=aligned,
        k_steps=8,
    )
    assert prediction.shape == (8, 10)
    assert np.isfinite(prediction).all()
    assert np.all((0.0 <= prediction[:, 9]) & (prediction[:, 9] <= 1.0))
    assert (raw is not None) == (representation in {"residual", "future_state"})


def test_retrieval_only_rejects_fake_model_output() -> None:
    aligned = np.stack([canonical_state(0.1 * index) for index in range(8)])
    with pytest.raises(EvaluationContractError, match="must not carry model output"):
        reconstruct_rank_prediction(
            np.zeros((8, 10)),
            target_representation="retrieval_only",
            statistics=statistics(),
            current_state_10d=canonical_state(0.0),
            aligned_pool_10d=aligned,
            k_steps=8,
        )


def test_rank_aggregation_is_equal_weight_after_reconstruction() -> None:
    target = np.stack([canonical_state(0.2 * index) for index in range(8)])
    aligned = target.copy()
    records = [
        RankPrediction(
            query_id="q",
            task="task_a",
            episode_id="e",
            current_row=7,
            retrieval_rank=rank,
            target_representation="residual",
            normalized_prediction=np.zeros((8, 10), dtype=np.float32),
            current_state_10d=canonical_state(0.0),
            aligned_pool_10d=aligned,
            query_target_10d=target,
        )
        for rank in range(3)
    ]
    result = evaluate_ranked_query(
        records,
        statistics=statistics(),
        workspace_xyz_min=(-10.0, -10.0, -10.0),
        workspace_xyz_max=(10.0, 10.0, 10.0),
    )
    assert result["rank_count"] == 3
    assert result["metrics"]["position_error_median_canonical"] == pytest.approx(0.0, abs=1e-7)
    assert all(value == 0 for value in result["guardrails"].values())


def test_unbound_workspace_is_visible_and_cannot_form_a_task_seed_unit() -> None:
    target = np.stack([canonical_state(0.0) for _ in range(8)])
    result = evaluate_ranked_query(
        [
            RankPrediction(
                "q", "task_a", "e", 7, 0, "retrieval_only", None,
                canonical_state(0.0), target, target,
            )
        ],
        statistics=statistics(),
        workspace_xyz_min=None,
        workspace_xyz_max=None,
    )
    assert result["guardrails"]["workspace_violation_count"] is None
    assert result["guardrails"]["workspace_clipping_applied_count"] == 0
    four_tasks = [{**result, "task": f"task_{index}"} for index in range(4)]
    with pytest.raises(EvaluationContractError, match="Unbound guardrail"):
        aggregate_task_seed_windows(four_tasks, seed=20260711)


def test_long_term_residual_saturation_requires_five_consecutive_steps() -> None:
    residual = np.zeros((8, 10))
    residual[:4, 0] = 2.0
    assert has_long_term_residual_saturation(residual, 1.0) is False
    residual[4, 0] = 2.0
    assert has_long_term_residual_saturation(residual, 1.0) is True


def task_seed_records(offset: float) -> list[dict]:
    return [
        {
            "task": f"task_{task}",
            "seed": seed,
            "metrics": {"position_error_median_canonical": 1.0 + offset + task * 0.01},
            "guardrails": {
                "nonfinite_prediction_count": 0,
                "gap_crossing_count": 0,
                "workspace_violation_count": 0,
                "workspace_clipping_applied_count": 0,
                "heldout_target_retrieval_feature_count": 0,
                "long_term_residual_saturation_count": 0,
            },
        }
        for task in range(4)
        for seed in (20260711, 20260712, 20260713)
    ]


def test_preregistered_main_statistics_and_holm_pass_for_uniform_improvement() -> None:
    recap = task_seed_records(-0.2)
    no_retrieval = task_seed_records(0.0)
    retrieval_only = task_seed_records(0.1)
    result = main_gate_analysis(recap, no_retrieval, retrieval_only)
    assert result["status"] == "passed"
    assert all(
        item["unit_count"] == 12
        and item["bootstrap_95ci"][1] < 0.0
        and item["holm_adjusted_p"] < 0.05
        and item["improved_task_count"] == 4
        for item in result["comparisons"].values()
    )


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    treatment = task_seed_records(-0.2)
    baseline = task_seed_records(0.0)
    first = paired_primary_analysis(treatment, baseline)
    second = paired_primary_analysis(treatment, baseline)
    assert first == second
    differences = {(item["task"], item["seed"]): -1.0 for item in treatment}
    assert exact_one_sided_sign_flip_p(differences) == pytest.approx(1.0 / 4096.0)
    assert holm_adjust_two({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_guardrail_gate_rejects_any_nonzero_count() -> None:
    records = task_seed_records(0.0)
    assert guardrail_gate(records)["status"] == "passed"
    records[0]["guardrails"]["gap_crossing_count"] = 1
    result = guardrail_gate(records)
    assert result["status"] == "failed"
    assert result["failures"][0]["guardrail_id"] == "gap_crossing_count"
