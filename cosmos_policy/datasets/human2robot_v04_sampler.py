"""Deterministic task-balanced distributed sampling for Human2Robot v04.

The sampler emits structured indices instead of pretending that a task-balanced
draw is a permutation of a flat dataset.  ``global_sample_index`` is the audit
identity: ranks receive disjoint strided subsets of the same deterministic
global stream, while the remaining fields identify the selected task, episode,
and legal H/K window.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence


@dataclass(frozen=True)
class V04SampleIndex:
    """One reproducible sample from the global task-balanced stream."""

    global_sample_index: int
    task: str
    episode_index: int
    window_index: int


class TaskBalancedDistributedSampler:
    """Balance tasks, then episodes, then legal windows across distributed ranks.

    Args:
        episode_window_counts: Mapping from task to the legal-window count of
            each training episode.  All counts must be positive.
        samples_per_rank: Number of samples emitted by each rank per epoch.
        num_replicas: Distributed world size.
        rank: This process' rank in ``[0, num_replicas)``.
        seed: Frozen experiment seed.

    The global stream is exactly task-balanced over every complete block of
    ``len(tasks)`` samples.  Within each task, episodes are visited in a seeded
    permutation before repeating.  Windows use a seeded coprime stride, so all
    legal windows of an episode are visited before a window repeats.
    """

    def __init__(
        self,
        episode_window_counts: Mapping[str, Sequence[int]],
        *,
        samples_per_rank: int,
        num_replicas: int,
        rank: int,
        seed: int = 20260711,
    ) -> None:
        if samples_per_rank <= 0:
            raise ValueError("samples_per_rank must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if not episode_window_counts:
            raise ValueError("episode_window_counts must not be empty")
        normalized: dict[str, tuple[int, ...]] = {}
        for task, counts in sorted(episode_window_counts.items()):
            values = tuple(int(value) for value in counts)
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"Every episode for {task!r} must have at least one legal window")
            normalized[str(task)] = values
        self.episode_window_counts = normalized
        self.samples_per_rank = int(samples_per_rank)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_rank

    def _seed(self, *parts: object) -> int:
        payload = ":".join(str(part) for part in (self.seed, self.epoch, *parts))
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")

    def _task_order(self) -> list[str]:
        tasks = list(self.episode_window_counts)
        random.Random(self._seed("tasks")).shuffle(tasks)
        return tasks

    def _episode_order(self, task: str) -> list[int]:
        episodes = list(range(len(self.episode_window_counts[task])))
        random.Random(self._seed("episodes", task)).shuffle(episodes)
        return episodes

    def _window(self, task: str, episode_index: int, episode_visit: int) -> int:
        count = self.episode_window_counts[task][episode_index]
        if count == 1:
            return 0
        start = self._seed("window-start", task, episode_index) % count
        stride = 1 + self._seed("window-stride", task, episode_index) % (count - 1)
        while math.gcd(stride, count) != 1:
            stride = 1 + (stride % (count - 1))
        return int((start + episode_visit * stride) % count)

    def sample_for_global_index(self, global_sample_index: int) -> V04SampleIndex:
        if global_sample_index < 0:
            raise ValueError("global_sample_index must be non-negative")
        tasks = self._task_order()
        task_count = len(tasks)
        task = tasks[global_sample_index % task_count]
        task_visit = global_sample_index // task_count
        episode_order = self._episode_order(task)
        episode_index = episode_order[task_visit % len(episode_order)]
        episode_visit = task_visit // len(episode_order)
        return V04SampleIndex(
            global_sample_index=global_sample_index,
            task=task,
            episode_index=episode_index,
            window_index=self._window(task, episode_index, episode_visit),
        )

    def __iter__(self) -> Iterator[V04SampleIndex]:
        for local_index in range(self.samples_per_rank):
            global_index = local_index * self.num_replicas + self.rank
            yield self.sample_for_global_index(global_index)
