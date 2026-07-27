"""Leakage-audited Human2Robot v04 training dataset.

The stage-5 preparation step materializes a compact immutable index containing
only the frozen seen-train records and their gap-safe H8/K8 window starts.  The
dataset consumes that index with the task-balanced distributed sampler instead
of flattening the longest tasks into a conventional random sampler.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from cosmos_policy.datasets.human2robot_dataset import (
    QUARANTINED_PUSHT_DATASET_KWARGS,
    _normalize,
    _preprocess_video,
    align_pool_chunk,
)
from cosmos_policy.datasets.human2robot_v04_retrieval import poses_euler_to_10d
from cosmos_policy.datasets.human2robot_v04_sampler import TaskBalancedDistributedSampler, V04SampleIndex


H_STEPS = 8
K_STEPS = 8
TOP_K = 3
RUN_SEED = 20260711
ALLOWED_METHODS = ("no_retrieval", "co_training", "recap_hand_ret")
TARGET_BY_METHOD = {
    "no_retrieval": "absolute",
    "co_training": "absolute",
    "recap_hand_ret": "residual",
}
SCHEMA = "human2robot-v04-stage5-dataset-v1"
V04_DATASET_KWARGS = {
    "source_root",
    "training_index_path",
    "statistics_path",
    "stage4_protocol_lock_path",
    "method_id",
    "seed",
    "h_steps",
    "k_steps",
    "top_k",
    "pool_size",
    "retrieval_modality",
    "final_image_size",
    "num_duplicates_per_image",
    "use_image_aug",
    "text_conditioning",
}


class Human2RobotV04DatasetError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Human2RobotV04DatasetError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _robot_states(file: h5py.File) -> np.ndarray:
    return poses_euler_to_10d(
        np.asarray(file["end_position"][:], dtype=np.float64),
        np.asarray(file["gripper_state"][:], dtype=np.float64),
    )


def _human_states(file: h5py.File) -> np.ndarray:
    action = np.asarray(file["action"][:], dtype=np.float64)
    return poses_euler_to_10d(action[:, :6], action[:, 6])


class Human2RobotV04Dataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Stage-5 seen-train dataset with structured task-balanced indices."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        training_index_path: str | Path,
        statistics_path: str | Path,
        stage4_protocol_lock_path: str | Path,
        method_id: str,
        seed: int = RUN_SEED,
        h_steps: int = H_STEPS,
        k_steps: int = K_STEPS,
        top_k: int = TOP_K,
        pool_size: int = 10,
        retrieval_modality: str = "geometry_plus_visual",
        final_image_size: int = 224,
        num_duplicates_per_image: int = 4,
        use_image_aug: bool = True,
        text_conditioning: str = "disabled_zero_embedding",
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.training_index_path = Path(training_index_path).resolve()
        self.statistics_path = Path(statistics_path).resolve()
        self.stage4_protocol_lock_path = Path(stage4_protocol_lock_path).resolve()
        self.method_id = str(method_id)
        self.seed = int(seed)
        self.h_steps = int(h_steps)
        self.k_steps = int(k_steps)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.retrieval_modality = str(retrieval_modality)
        self.final_image_size = int(final_image_size)
        self.num_duplicates_per_image = int(num_duplicates_per_image)
        self.use_image_aug = bool(use_image_aug)
        self.text_conditioning = str(text_conditioning)
        self.split = "seen_train"
        self.experiment_id = "V04-MAIN-01"
        self.variant_id = "frozen_main"
        self.target_representation = TARGET_BY_METHOD.get(self.method_id, "")
        self.time_view_id = "gap_safe_source_time"
        self.query_offset_view_steps = 1
        self.window_stride = 1

        _require(self.method_id in ALLOWED_METHODS, f"Unsupported v04 method: {self.method_id}")
        _require(self.seed == RUN_SEED, f"Stage-5 seed must remain {RUN_SEED}")
        _require((self.h_steps, self.k_steps) == (H_STEPS, K_STEPS), "Stage-5 H/K must remain 8/8")
        _require(self.top_k == TOP_K and self.pool_size == 10, "Stage-5 retrieval must remain top3/pool10")
        _require(self.retrieval_modality == "geometry_plus_visual", "Stage-5 primary retrieval modality drift")
        _require(self.final_image_size == 224, "Stage-5 image size must remain 224")
        _require(self.num_duplicates_per_image == 4, "WAN tokenizer requires four-frame slots")
        _require(self.text_conditioning == "disabled_zero_embedding", "Text conditioning must remain disabled")

        self.protocol_lock = _read_json(self.stage4_protocol_lock_path)
        _require(self.protocol_lock.get("status") == "VERIFIED_STAGE4", "Stage-4 protocol lock is not verified")
        _require(self.protocol_lock.get("training_allowed") is True, "Stage-4 lock does not authorize training")
        self.protocol_file_sha256 = str(self.protocol_lock["training_protocol_sha256"])
        _require(len(self.protocol_file_sha256) == 64, "Stage-4 training protocol SHA is missing")

        self.statistics = _read_json(self.statistics_path)
        _require(self.statistics.get("status") == "FROZEN", "Stage-5 statistics are not frozen")
        _require(self.statistics.get("method") == self.method_id, "Statistics method mismatch")
        _require(
            self.statistics.get("training_protocol_sha256") == self.protocol_file_sha256,
            "Statistics training protocol mismatch",
        )

        with np.load(self.training_index_path, allow_pickle=False) as stored:
            self.starts = np.asarray(stored["starts"], dtype=np.int64)
            self.offsets = np.asarray(stored["offsets"], dtype=np.int64)
            index_manifest = json.loads(str(stored["manifest_json"].item()))
            self.candidate_record_indices = (
                np.asarray(stored["candidate_record_indices"], dtype=np.int32)
                if "candidate_record_indices" in stored.files
                else None
            )
            self.candidate_starts = (
                np.asarray(stored["candidate_starts"], dtype=np.int64)
                if "candidate_starts" in stored.files
                else None
            )
            self.retrieval_distances = (
                np.asarray(stored["retrieval_distances"], dtype=np.float32)
                if "retrieval_distances" in stored.files
                else None
            )
        _require(index_manifest.get("status") == "FROZEN", "Stage-5 training index is not frozen")
        _require(index_manifest.get("method") == self.method_id, "Training-index method mismatch")
        _require(
            index_manifest.get("training_protocol_sha256") == self.protocol_file_sha256,
            "Training-index protocol mismatch",
        )
        self.records = [dict(record) for record in index_manifest.get("records", [])]
        _require(len(self.records) == 654, f"Expected 654 seen-train records, got {len(self.records)}")
        _require(len(self.offsets) == len(self.records) + 1, "Training-index offsets are malformed")
        _require(int(self.offsets[0]) == 0 and int(self.offsets[-1]) == len(self.starts), "Training-index offset drift")
        _require(len(self.starts) == 244_372, f"Expected 244372 legal windows, got {len(self.starts)}")
        _require(index_manifest.get("index_file_sha256") in {None, _file_sha256(self.training_index_path)}, "Index self-hash drift")
        if self.method_id == "recap_hand_ret":
            _require(self.candidate_record_indices is not None, "RECAP candidate record index is missing")
            _require(self.candidate_starts is not None, "RECAP candidate starts are missing")
            _require(self.retrieval_distances is not None, "RECAP retrieval distances are missing")
            expected_shape = (len(self.starts), TOP_K)
            _require(self.candidate_record_indices.shape == expected_shape, "RECAP candidate-record shape drift")
            _require(self.candidate_starts.shape == expected_shape, "RECAP candidate-start shape drift")
            _require(self.retrieval_distances.shape == expected_shape, "RECAP distance shape drift")

        self.task_record_indices: dict[str, list[int]] = {}
        for record_index, record in enumerate(self.records):
            _require(record.get("source_partition") == "seen_train", "Non-train record entered stage 5")
            _require(record.get("role") == "paired", "Stage-5 record is not paired")
            self.task_record_indices.setdefault(str(record["task"]), []).append(record_index)
        _require(len(self.task_record_indices) == 16, "Stage-5 must contain exactly 16 seen tasks")
        self.expanded_counts = np.diff(self.offsets) * TOP_K
        self.expanded_offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(self.expanded_counts)))

    def __len__(self) -> int:
        return int(self.expanded_offsets[-1])

    def make_distributed_sampler(self, *, num_replicas: int, rank: int) -> TaskBalancedDistributedSampler:
        counts = {
            task: [int(self.expanded_counts[index]) for index in record_indices]
            for task, record_indices in self.task_record_indices.items()
        }
        samples_per_rank = (len(self) + int(num_replicas) - 1) // int(num_replicas)
        return TaskBalancedDistributedSampler(
            counts,
            samples_per_rank=samples_per_rank,
            num_replicas=int(num_replicas),
            rank=int(rank),
            seed=self.seed,
        )

    def _resolve_index(self, index: int | V04SampleIndex) -> tuple[int, int, int, int]:
        if isinstance(index, V04SampleIndex):
            record_indices = self.task_record_indices.get(index.task)
            _require(record_indices is not None, f"Unknown sampled task: {index.task}")
            _require(0 <= index.episode_index < len(record_indices), "Sampled episode index is out of range")
            record_index = record_indices[index.episode_index]
            expanded_window = int(index.window_index)
            _require(expanded_window < int(self.expanded_counts[record_index]), "Sampled window index is out of range")
            global_sample_index = int(index.global_sample_index)
        else:
            flat = int(index)
            _require(0 <= flat < len(self), f"Dataset index out of range: {flat}")
            record_index = bisect_right(self.expanded_offsets, flat) - 1
            expanded_window = flat - int(self.expanded_offsets[record_index])
            global_sample_index = flat
        retrieval_rank = expanded_window % TOP_K
        local_window_index = expanded_window // TOP_K
        global_window_index = int(self.offsets[record_index]) + local_window_index
        return record_index, global_window_index, retrieval_rank, global_sample_index

    def _record_path(self, record: Mapping[str, Any]) -> Path:
        path = (self.source_root / str(record["source_relative_path"])).resolve()
        _require(path.is_file(), f"Stage-5 source is missing: {path}")
        _require(path.is_relative_to(self.source_root), f"Stage-5 source escaped root: {path}")
        return path

    def __getitem__(self, index: int | V04SampleIndex) -> dict[str, Any]:
        record_index, global_window_index, retrieval_rank, global_sample_index = self._resolve_index(index)
        query_record = self.records[record_index]
        start = int(self.starts[global_window_index])
        history = np.arange(start, start + H_STEPS, dtype=np.int64)
        future = np.arange(start + H_STEPS, start + H_STEPS + K_STEPS, dtype=np.int64)
        query_path = self._record_path(query_record)
        with h5py.File(query_path, "r") as file:
            robot = _robot_states(file)
            current = robot[int(history[-1])]
            query_target = robot[future]
            current_image = np.asarray(file["cam_data/robot_camera"][int(history[-1])], dtype=np.uint8)
            future_image = np.asarray(file["cam_data/robot_camera"][int(future[-1])], dtype=np.uint8)
            if self.method_id == "co_training":
                human = _human_states(file)
                raw_plan = human[future]
                human_images = np.stack(
                    [np.asarray(file["cam_data/human_camera"][int(row)], dtype=np.uint8) for row in future]
                )
            else:
                raw_plan = np.repeat(current[None], H_STEPS, axis=0)
                human_images = np.zeros((H_STEPS, *current_image.shape), dtype=np.uint8)

        candidate_record = query_record
        candidate_start = start
        retrieval_distance = 0.0
        if self.method_id == "recap_hand_ret":
            assert self.candidate_record_indices is not None
            assert self.candidate_starts is not None
            assert self.retrieval_distances is not None
            candidate_record_index = int(self.candidate_record_indices[global_window_index, retrieval_rank])
            candidate_record = self.records[candidate_record_index]
            candidate_start = int(self.candidate_starts[global_window_index, retrieval_rank])
            _require(
                candidate_record["source_sha256"] != query_record["source_sha256"],
                "RECAP training retrieval reused the query source",
            )
            candidate_future = np.arange(candidate_start + H_STEPS, candidate_start + H_STEPS + K_STEPS)
            with h5py.File(self._record_path(candidate_record), "r") as file:
                human = _human_states(file)
                raw_plan = human[candidate_future]
                human_images = np.stack(
                    [np.asarray(file["cam_data/human_camera"][int(row)], dtype=np.uint8) for row in candidate_future]
                )
            retrieval_distance = float(self.retrieval_distances[global_window_index, retrieval_rank])

        aligned_plan = align_pool_chunk(raw_plan, current)
        if self.method_id == "recap_hand_ret":
            actions = _normalize(
                query_target - aligned_plan[:K_STEPS],
                self.statistics["residual_10d_min"],
                self.statistics["residual_10d_max"],
            )
        else:
            actions = _normalize(
                query_target,
                self.statistics["query_bc_target_10d_min"],
                self.statistics["query_bc_target_10d_max"],
            )
        pool_normalized = _normalize(
            aligned_plan,
            self.statistics["pool_action_10d_min"],
            self.statistics["pool_action_10d_max"],
        )
        has_retrieval = int(self.method_id != "no_retrieval")
        if not has_retrieval:
            pool_normalized = np.zeros_like(pool_normalized)
            human_images = np.zeros_like(human_images)
        current_normalized = _normalize(
            current,
            self.statistics["query_bc_target_10d_min"],
            self.statistics["query_bc_target_10d_max"],
        )
        future_normalized = _normalize(
            query_target[-1],
            self.statistics["query_bc_target_10d_min"],
            self.statistics["query_bc_target_10d_max"],
        )

        blank = np.zeros_like(current_image)
        blank4 = np.repeat(blank[None], self.num_duplicates_per_image, axis=0)
        frames = np.concatenate(
            (
                blank[None],
                human_images,
                blank4,
                blank4,
                np.repeat(current_image[None], self.num_duplicates_per_image, axis=0),
                blank4,
                blank4,
                np.repeat(future_image[None], self.num_duplicates_per_image, axis=0),
                blank4,
            ),
            axis=0,
        )
        _require(len(frames) == 37, "Stage-5 WAN frame layout mismatch")
        augment_seed = self.seed + global_sample_index if self.use_image_aug else None
        video = _preprocess_video(frames, self.final_image_size, augment_seed)
        ret_state_idx = 3
        raw_residual = query_target - aligned_plan[:K_STEPS]
        query_id = f"{query_record['source_sha256']}:{start}:H8:K8"
        candidate_id = f"{candidate_record['source_sha256']}:{candidate_start}:H8:K8"
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
            "__key__": global_sample_index,
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
            "retrieved_actions": torch.from_numpy(pool_normalized),
            "retrieved_proprio": torch.from_numpy(pool_normalized[0]),
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
            "episode_id": str(query_record["episode_id"]),
            "query_id": query_id,
            "candidate_id": candidate_id,
            "query_source_sha256": str(query_record["source_sha256"]),
            "candidate_source_sha256": "" if not has_retrieval else str(candidate_record["source_sha256"]),
            "task": str(query_record["task"]),
            "split": self.split,
            "current_row": int(history[-1]),
            "method_id": self.method_id,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "target_representation": self.target_representation,
            "H_steps": H_STEPS,
            "K_steps": K_STEPS,
            "top_k": TOP_K,
            "pool_size": self.pool_size,
            "retrieval_modality": self.retrieval_modality,
            "retrieval_rank": retrieval_rank,
            "retrieval_distance": np.float32(retrieval_distance),
            "sample_weight": np.float32(1.0 / TOP_K),
            "strict_future_offset_view_steps": 1,
            "gap_crossing_count": 0,
            "heldout_target_retrieval_feature_count": 0,
            "query_command_status": "unverified",
            "deployment_command_adapter_id": "",
            "protocol_id": "human2robot_v04_clean_single_seed_offline",
            "protocol_file_sha256": self.protocol_file_sha256,
            "adapter_schema_version": SCHEMA,
            "diagnostic_overfit_mode": 0,
            "diagnostic_overfit_seed": self.seed,
            "raw_current_state": torch.from_numpy(current.astype(np.float32)),
            "raw_aligned_pool": torch.from_numpy(aligned_plan.astype(np.float32)),
            "raw_query_target": torch.from_numpy(query_target.astype(np.float32)),
            "raw_residual": torch.from_numpy(raw_residual.astype(np.float32)),
        }


def _formal_dataset_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Isolate the v04 adapter from recursively merged PushT kwargs."""

    keys = set(kwargs)
    missing = V04_DATASET_KWARGS - keys
    unknown = keys - V04_DATASET_KWARGS - QUARANTINED_PUSHT_DATASET_KWARGS
    _require(not missing, f"Resolved config is missing v04 dataset kwargs: {sorted(missing)}")
    _require(not unknown, f"Resolved config contains unknown inherited dataset kwargs: {sorted(unknown)}")
    return {key: kwargs[key] for key in V04_DATASET_KWARGS}


def build_human2robot_v04_dataset(**kwargs: Any) -> Human2RobotV04Dataset:
    return Human2RobotV04Dataset(**_formal_dataset_kwargs(kwargs))
