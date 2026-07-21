from __future__ import annotations

from collections import Counter

import pytest

from cosmos_policy.datasets.human2robot_v04_sampler import TaskBalancedDistributedSampler


def _sampler(rank: int) -> TaskBalancedDistributedSampler:
    return TaskBalancedDistributedSampler(
        {"task_a": [3, 5], "task_b": [4, 2], "task_c": [7, 3], "task_d": [2, 6]},
        samples_per_rank=32,
        num_replicas=4,
        rank=rank,
    )


def test_ranks_have_disjoint_global_sample_indices_and_exact_task_balance() -> None:
    samples = [list(_sampler(rank)) for rank in range(4)]
    identities = [{item.global_sample_index for item in rank_samples} for rank_samples in samples]
    assert all(identities[left].isdisjoint(identities[right]) for left in range(4) for right in range(left + 1, 4))
    combined = sorted((item for rank_samples in samples for item in rank_samples), key=lambda item: item.global_sample_index)
    assert [item.global_sample_index for item in combined] == list(range(128))
    assert set(Counter(item.task for item in combined).values()) == {32}


def test_set_epoch_is_reproducible_and_changes_the_stream() -> None:
    sampler = _sampler(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch0
    assert epoch1 != epoch0


def test_windows_stay_in_episode_bounds() -> None:
    counts = {"task": [1, 2, 11]}
    sampler = TaskBalancedDistributedSampler(counts, samples_per_rank=200, num_replicas=1, rank=0)
    for item in sampler:
        assert 0 <= item.window_index < counts[item.task][item.episode_index]


@pytest.mark.parametrize("rank", [-1, 4])
def test_invalid_rank_is_rejected(rank: int) -> None:
    with pytest.raises(ValueError, match="rank must be"):
        TaskBalancedDistributedSampler({"task": [1]}, samples_per_rank=1, num_replicas=4, rank=rank)
