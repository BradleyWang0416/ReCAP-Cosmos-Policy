from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_policy.datasets.human2robot_p2_contract import (
    Human2RobotP2ContractError,
    RetrievalCandidate,
    aggregate_canonical_predictions,
    canonical_window_metrics,
    deterministic_inference_seed,
    future_state_target,
    geometry_feature,
    preprocess_resolution_frames,
    rank_retrieval_candidates,
    reconstruct_future_state,
)


def _trajectory(length: int = 8) -> np.ndarray:
    result = np.zeros((length, 10), dtype=np.float32)
    result[:, 0] = np.arange(length) * 0.01
    result[:, 3] = 1.0
    result[:, 7] = 1.0
    result[:, 9] = np.linspace(0.0, 1.0, length)
    return result


def test_future_state_round_trip_projects_rotation_and_gripper() -> None:
    current = _trajectory(1)[0]
    future = _trajectory()
    transitions = future_state_target(current, future)
    reconstructed = reconstruct_future_state(current, transitions)
    np.testing.assert_allclose(reconstructed, future, atol=1e-6)
    assert bool(np.all((reconstructed[:, 9] >= 0.0) & (reconstructed[:, 9] <= 1.0)))


def test_phase_rank_uses_seeded_hash_only_for_exact_ties() -> None:
    candidates = [
        RetrievalCandidate("b", "b" * 64, 0.3),
        RetrievalCandidate("a", "a" * 64, 0.3),
        RetrievalCandidate("near", "c" * 64, 0.21),
    ]
    ranked = rank_retrieval_candidates(
        candidates,
        modality="phase",
        run_seed=20260711,
        query_id="query",
        query_phase=0.2,
    )
    assert ranked[0][0].candidate_id == "near"
    assert [item[2] for item in ranked[1:]] == sorted(item[2] for item in ranked[1:])


def test_random_ranking_is_deterministic_and_seed_bound() -> None:
    candidates = [
        RetrievalCandidate(str(index), f"{index:064x}", 0.0) for index in range(5)
    ]
    first = rank_retrieval_candidates(
        candidates,
        modality="random",
        run_seed=20260711,
        query_id="q",
        query_phase=0.0,
    )
    repeat = rank_retrieval_candidates(
        candidates,
        modality="random",
        run_seed=20260711,
        query_id="q",
        query_phase=0.0,
    )
    other = rank_retrieval_candidates(
        candidates,
        modality="random",
        run_seed=20260712,
        query_id="q",
        query_phase=0.0,
    )
    assert [item[0].candidate_id for item in first] == [item[0].candidate_id for item in repeat]
    assert [item[0].candidate_id for item in first] != [item[0].candidate_id for item in other]


def test_geometry_rejects_degenerate_train_statistics() -> None:
    with pytest.raises(Human2RobotP2ContractError, match="std must be positive"):
        geometry_feature(_trajectory(4), np.zeros(10), np.zeros(10))


def test_top_k_aggregation_is_canonical_and_equal_weight() -> None:
    low = _trajectory()
    high = _trajectory()
    high[:, 0] += 0.2
    aggregate = aggregate_canonical_predictions([low, high])
    np.testing.assert_allclose(aggregate[:, 0], low[:, 0] + 0.1, atol=1e-6)
    assert canonical_window_metrics(aggregate, low)["position_error_median_canonical"] == pytest.approx(0.1)


def test_resolution_variants_share_uint8_224_boundary() -> None:
    frames = np.arange(2 * 240 * 426 * 3, dtype=np.uint32).reshape(2, 240, 426, 3).astype(np.uint8)
    outputs = [
        preprocess_resolution_frames(frames, variant)
        for variant in (
            "source_240x426_then_resize_224",
            "center_crop_240x424_then_resize_224",
            "center_crop_240x424_edge_pad_240x426_then_resize_224",
        )
    ]
    assert all(item.shape == (3, 2, 224, 224) and item.dtype == torch.uint8 for item in outputs)
    assert not torch.equal(outputs[0], outputs[1])


def test_inference_seed_is_stable_lower_31_bits() -> None:
    value = deterministic_inference_seed(20260711, "M5B-MAIN-01", "main", "task", "ep", 8, 2)
    assert value == deterministic_inference_seed(20260711, "M5B-MAIN-01", "main", "task", "ep", 8, 2)
    assert 0 <= value <= 0x7FFFFFFF

