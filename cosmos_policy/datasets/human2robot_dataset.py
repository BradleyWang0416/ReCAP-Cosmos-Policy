"""Formal Human2Robot adapter for the retrieval-conditioned Cosmos Policy model.

The adapter consumes only canonical/v3 and the Gate-B-approved main view.  It
materializes the same 37-frame / 10-latent layout used by the existing PushT
retrieval model while preserving Human2Robot's separate pool/query roles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FORMAL_SCHEMA_VERSION = "human2robot-formal-cosmos-adapter-v1"
CANONICAL_SCHEMA_VERSION = "human2robot-canonical-hdf5-v3"
PROTOCOL_ID = "m5b_v03_preregistered_3seed_formal_v1"
ALLOWED_METHODS = {"no_retrieval", "retrieval_only", "co_training", "recap_hand_ret"}
LEARNED_METHODS = {"no_retrieval", "co_training", "recap_hand_ret"}
FROZEN_SEEDS = {20260711, 20260712, 20260713}
FORMAL_DATASET_KWARGS = {
    "canonical_root",
    "main_view_path",
    "m3_report_path",
    "m4_report_path",
    "protocol_path",
    "split",
    "method_id",
    "seed",
    "horizon",
    "window_stride",
    "final_image_size",
    "num_duplicates_per_image",
    "use_image_aug",
    "text_conditioning",
    "diagnostic_overfit_window_index",
}
# Hydra recursively merges the selected PushT 2B experiment before applying
# the Human2Robot override.  These keys are quarantined at the factory boundary
# and are never forwarded to Human2RobotFormalDataset.  Unknown merged keys are
# a hard error so a future parent-config change cannot silently alter semantics.
QUARANTINED_PUSHT_DATASET_KWARGS = {
    "aux_sampling_prob",
    "chunk_size",
    "data_dir",
    "demonstration_sampling_prob",
    "episode_allowlist_path",
    "episode_allowlist_top_k",
    "extra_task_splits",
    "force_zero_ret_image",
    "force_zero_ret_state",
    "gamma",
    "max_num_episodes",
    "normalize_actions",
    "normalize_images",
    "normalize_proprio",
    "predict_future_states",
    "ret_action_as_target_prob",
    "ret_context_multiplier",
    "ret_image_subsample",
    "ret_single_frame",
    "retrieval_dropout_prob",
    "retrieval_image_only_dropout_prob",
    "retrieval_npz_path",
    "retrieval_source_splits",
    "retrieval_top_k_choice",
    "return_value_function_returns",
    "rollout_data_dir",
    "success_rollout_sampling_prob",
    "t5_text_embeddings_path",
    "task_split",
    "treat_success_rollouts_as_demos",
    "use_proprio",
    "use_residual_actions",
    "use_stronger_image_aug",
    "use_third_person_images",
    "use_wrist_images",
}


class Human2RobotContractError(RuntimeError):
    """Raised when data could silently violate the frozen v03 contract."""


@dataclass(frozen=True)
class WindowRecord:
    episode_id: str
    path: Path
    task: str
    split: str
    segment_number: int
    current_row: int
    pool_rows: np.ndarray
    query_rows: np.ndarray
    phase: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Human2RobotContractError(f"Missing required artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Human2RobotContractError(f"Expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Human2RobotContractError(message)


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64)
    _require(values.shape[-1] == 6, f"Expected rotation-6D, got {values.shape}")
    flat = values.reshape(-1, 6)
    first = flat[:, :3].copy()
    second = flat[:, 3:6].copy()
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=2).reshape(values.shape[:-1] + (3, 3))


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    _require(values.shape[-2:] == (3, 3), f"Expected rotation matrix, got {values.shape}")
    return np.concatenate((values[..., :, 0], values[..., :, 1]), axis=-1)


def align_pool_chunk(pool: np.ndarray, query_current: np.ndarray) -> np.ndarray:
    """Apply the frozen SE(3) query-anchor alignment from M3."""
    pool = np.asarray(pool, dtype=np.float64)
    current = np.asarray(query_current, dtype=np.float64)
    _require(pool.ndim == 2 and pool.shape[1] == 10, f"Invalid pool shape: {pool.shape}")
    _require(current.shape == (10,), f"Invalid query-current shape: {current.shape}")
    aligned = pool.copy()
    aligned[:, :3] = current[:3] + pool[:, :3] - pool[0, :3]
    pool_rot = _rotation_6d_to_matrix(pool[:, 3:9])
    current_rot = _rotation_6d_to_matrix(current[None, 3:9])[0]
    aligned_rot = current_rot @ pool_rot[0].T @ pool_rot
    aligned[:, 3:9] = _matrix_to_rotation_6d(aligned_rot)
    return aligned


def _normalize(values: np.ndarray, minimum: Sequence[float], maximum: Sequence[float]) -> np.ndarray:
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    result = 2.0 * ((np.asarray(values, dtype=np.float64) - low) / (high - low + 1e-8)) - 1.0
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def _contiguous_segments(segment_id: np.ndarray, gap_mask: np.ndarray) -> list[np.ndarray]:
    _require(segment_id.shape == gap_mask.shape, "segment_id/gap_mask shape mismatch")
    result: list[np.ndarray] = []
    start = 0
    for index in range(1, len(segment_id)):
        if segment_id[index] != segment_id[index - 1] or bool(gap_mask[index]):
            if index > start:
                result.append(np.arange(start, index, dtype=np.int64))
            start = index
    if len(segment_id) > start:
        result.append(np.arange(start, len(segment_id), dtype=np.int64))
    return result


def _preprocess_video(frames: np.ndarray, final_size: int, augment_seed: int | None) -> torch.Tensor:
    _require(frames.ndim == 4 and frames.shape[-1] == 3, f"Invalid RGB sequence: {frames.shape}")
    _require(frames.shape[1] == 240, f"Expected source height 240, got {frames.shape}")
    if frames.shape[2] == 426:
        frames = frames[:, :, 1:-1, :]
    _require(frames.shape[2] == 424, f"Expected cropped width 424, got {frames.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(final_size, final_size), mode="bilinear", align_corners=False, antialias=True)
    if augment_seed is not None:
        generator = torch.Generator().manual_seed(augment_seed)
        gain = 0.9 + 0.2 * torch.rand((), generator=generator)
        bias = 255.0 * (-0.05 + 0.1 * torch.rand((), generator=generator))
        tensor = tensor * gain + bias
    # Predict2.5 owns input normalization and hard-requires uint8 at this
    # boundary.  Returning pre-normalized floats would fail in the real 2B
    # training graph even though standalone dataset shape tests could pass.
    return tensor.clamp(0.0, 255.0).round().to(torch.uint8).permute(1, 0, 2, 3).contiguous()


class Human2RobotFormalDataset(Dataset):
    """Canonical/v3 dataset in the formal 2B retrieval-model batch format."""

    def __init__(
        self,
        canonical_root: str | Path,
        main_view_path: str | Path,
        m3_report_path: str | Path,
        m4_report_path: str | Path,
        protocol_path: str | Path,
        split: str = "train",
        method_id: str = "recap_hand_ret",
        seed: int = 20260711,
        horizon: int = 8,
        window_stride: int = 8,
        final_image_size: int = 224,
        num_duplicates_per_image: int = 4,
        use_image_aug: bool = True,
        text_conditioning: str = "disabled_zero_embedding",
        diagnostic_overfit_window_index: int | None = None,
    ) -> None:
        self.canonical_root = Path(canonical_root)
        self.main_view_path = Path(main_view_path)
        self.m3_report_path = Path(m3_report_path)
        self.m4_report_path = Path(m4_report_path)
        self.protocol_path = Path(protocol_path)
        self.split = split
        self.method_id = method_id
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.window_stride = int(window_stride)
        self.final_image_size = int(final_image_size)
        self.num_duplicates_per_image = int(num_duplicates_per_image)
        self.use_image_aug = bool(use_image_aug)
        self.text_conditioning = text_conditioning
        self.diagnostic_overfit_window_index = diagnostic_overfit_window_index

        _require(split in {"train", "heldout"}, f"Invalid split: {split}")
        _require(method_id in ALLOWED_METHODS, f"Invalid method: {method_id}")
        _require(self.seed in FROZEN_SEEDS, f"Seed is not preregistered: {self.seed}")
        _require(self.horizon == 8, "Formal v1 requires H=8")
        _require(self.window_stride == 8, "Formal v1 requires window_stride=8")
        _require(self.final_image_size == 224, "Formal v1 requires 224x224 model input")
        _require(self.num_duplicates_per_image == 4, "WAN tokenizer requires four images per latent")
        _require(text_conditioning == "disabled_zero_embedding", "Unregistered text-conditioning mode")
        _require(
            diagnostic_overfit_window_index is None or diagnostic_overfit_window_index >= 0,
            "diagnostic_overfit_window_index must be None or non-negative",
        )

        self.protocol = _read_json(self.protocol_path)
        self.view = _read_json(self.main_view_path / "view_manifest.json")
        self.m3_report = _read_json(self.m3_report_path)
        self.m4_report = _read_json(self.m4_report_path)
        self.split_manifest = _read_json(self.canonical_root / "task_split_manifest.json")
        self.action_statistics = _read_json(self.main_view_path / "action_statistics.json")
        self.protocol_file_sha256 = file_sha256(self.protocol_path)
        self._validate_parent_contract()
        self.windows = self._build_windows()
        _require(bool(self.windows), f"No formal windows for split={split}")
        if diagnostic_overfit_window_index is not None:
            _require(split == "train", "One-batch overfit diagnostic is train-only")
            _require(diagnostic_overfit_window_index < len(self.windows), "Overfit window index out of range")
            self.windows = [self.windows[diagnostic_overfit_window_index]]

    def _validate_parent_contract(self) -> None:
        frozen = self.protocol.get("frozen_data_contract", {})
        training = self.protocol.get("frozen_training_protocol", {})
        model = training.get("model", {})
        seeds = training.get("optimization", {}).get("seeds", [])
        expected_view = {
            "canonical_schema": CANONICAL_SCHEMA_VERSION,
            "time_view_id": "nominal_camera_30hz_segmented",
            "pool_action_view_id": "human_hand_robot_frame_raw",
            "query_action_view_id": "robot_ee_observed_t_plus_1_bc_proxy",
            "action_alignment_id": "train_only_tplus1_query_anchor_se3_identity_scale_v1",
            "query_command_status": "unverified",
            "strict_future_target": True,
            "query_target_offset_view_steps": 1,
            "gap_policy": "never_cross_segment",
        }
        _require(self.protocol.get("protocol_id") == PROTOCOL_ID, "Wrong M5-B protocol")
        _require(self.protocol.get("status") == "frozen_pre_registration", "Protocol is not frozen")
        _require(seeds == [20260711, 20260712, 20260713], "Protocol seed list changed")
        _require(model.get("action_dim") == 10 and model.get("proprio_dim") == 10, "Formal dimensions changed")
        _require(model.get("H_steps") == 8 and model.get("K_steps") == 8, "Formal H/K changed")
        for key, expected in expected_view.items():
            _require(self.view.get(key) == expected, f"Main-view mismatch {key}: {self.view.get(key)!r}")
        _require(self.view.get("deployment_command_adapter_id") is None, "Deployment adapter forbidden in M5")
        _require(self.m3_report.get("status") == "passed", "M3 is not passed")
        _require(self.m3_report.get("gates", {}).get("B") == "passed", "M3 Gate B is not passed")
        _require(self.m4_report.get("status") == "launched", "M4 smoke is not launched")
        _require(self.m4_report.get("gate_c") == "pending", "Gate C must remain pending")
        _require(self.m4_report.get("m6_rollout_approved") is False, "M6 must remain forbidden")
        _require(self.split_manifest.get("split_sha256") == frozen.get("split_sha256"), "Split hash changed")
        _require(self.view.get("view_id") == frozen.get("view_id"), "View ID changed")
        _require(self.seed in seeds, "Run seed is not frozen")
        provenance = self.action_statistics.get("provenance", {})
        _require(provenance.get("heldout_data_used") is False, "Action stats use heldout data")

    def _build_windows(self) -> list[WindowRecord]:
        files = sorted((self.canonical_root / "pilot").glob("demo_*.hdf5"))
        records = self.split_manifest.get("episodes", [])
        _require(len(files) == len(records), "Canonical/split episode count mismatch")
        result: list[WindowRecord] = []
        for path, record in zip(files, records, strict=True):
            if record.get("split") != self.split:
                continue
            with h5py.File(path, "r") as file:
                demo = file["data/demo_0"]
                _require(demo.attrs.get("schema_version") == CANONICAL_SCHEMA_VERSION, f"Wrong schema: {path}")
                source = str(demo.attrs.get("source_relative_path", ""))
                _require(not source or source == record.get("source_relative_path"), f"Source/split mismatch: {path}")
                segment_id = np.asarray(demo["metadata/segment_id"][:], dtype=np.int64)
                gap_mask = np.asarray(demo["metadata/gap_mask"][:], dtype=bool)
            for segment_number, rows in enumerate(_contiguous_segments(segment_id, gap_mask)):
                last_start = len(rows) - self.horizon - 1
                for local_start in range(0, max(0, last_start + 1), self.window_stride):
                    pool_rows = rows[local_start : local_start + self.horizon]
                    query_rows = rows[local_start + 1 : local_start + 1 + self.horizon]
                    _require(len(pool_rows) == self.horizon and len(query_rows) == self.horizon, "Incomplete window")
                    _require(np.all(segment_id[query_rows] == segment_id[pool_rows[0]]), "Window crosses segment")
                    result.append(
                        WindowRecord(
                            episode_id=path.stem,
                            path=path,
                            task=str(record["task"]),
                            split=self.split,
                            segment_number=segment_number,
                            current_row=int(pool_rows[0]),
                            pool_rows=pool_rows.copy(),
                            query_rows=query_rows.copy(),
                            phase=float(local_start / max(1, len(rows) - 1)),
                        )
                    )
        return result

    def __len__(self) -> int:
        return len(self.windows)

    def _normalized_targets(
        self, aligned_plan: np.ndarray, query_target: np.ndarray
    ) -> tuple[np.ndarray, str]:
        stats = self.action_statistics
        if self.method_id == "recap_hand_ret":
            target = query_target - aligned_plan
            normalized = _normalize(target, stats["residual_10d_min"], stats["residual_10d_max"])
            return normalized, "residual"
        if self.method_id in {"no_retrieval", "co_training"}:
            normalized = _normalize(
                query_target, stats["query_bc_target_10d_min"], stats["query_bc_target_10d_max"]
            )
            return normalized, "absolute"
        return np.zeros_like(query_target, dtype=np.float32), "retrieval_only"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.windows[index]
        with h5py.File(record.path, "r") as file:
            demo = file["data/demo_0"]
            human = np.asarray(demo["trajectories/human_hand_robot_frame_10d"][:], dtype=np.float64)
            robot = np.asarray(demo["trajectories/robot_ee_observed_10d"][:], dtype=np.float64)
            human_images = np.asarray(demo["metadata/human/images"][record.pool_rows], dtype=np.uint8)
            current_image = np.asarray(demo["obs/robot_images"][record.current_row], dtype=np.uint8)
            future_image = np.asarray(demo["obs/robot_images"][record.query_rows[-1]], dtype=np.uint8)

        current = robot[record.current_row]
        raw_pool = human[record.pool_rows]
        query_target = robot[record.query_rows]
        aligned_plan = align_pool_chunk(raw_pool, current)
        residual = query_target - aligned_plan
        actions, target_representation = self._normalized_targets(aligned_plan, query_target)
        pool_normalized = _normalize(
            aligned_plan,
            self.action_statistics["pool_action_10d_min"],
            self.action_statistics["pool_action_10d_max"],
        )
        current_normalized = _normalize(
            current,
            self.action_statistics["query_bc_target_10d_min"],
            self.action_statistics["query_bc_target_10d_max"],
        )
        future_normalized = _normalize(
            query_target[-1],
            self.action_statistics["query_bc_target_10d_min"],
            self.action_statistics["query_bc_target_10d_max"],
        )

        has_retrieval = int(self.method_id != "no_retrieval")
        if not has_retrieval:
            pool_normalized = np.zeros_like(pool_normalized)
            human_images = np.zeros_like(human_images)
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
        _require(len(frames) == 37, f"Expected tokenizer chunk_duration=37, got {len(frames)}")
        augment_seed = self.seed + index if self.use_image_aug else None
        video = _preprocess_video(frames, self.final_image_size, augment_seed)

        return {
            "video": video,
            "actions": torch.from_numpy(actions),
            "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16),
            "t5_text_mask": torch.zeros(512, dtype=torch.int64),
            "fps": 30,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": torch.from_numpy(current_normalized),
            "future_proprio": torch.from_numpy(future_normalized),
            "__key__": index,
            "action_latent_idx": 7,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": 6,
            "current_wrist_image_latent_idx": -1,
            "current_image_latent_idx": 5,
            "future_proprio_latent_idx": 9,
            "future_wrist_image_latent_idx": -1,
            "future_image_latent_idx": 8,
            "retrieved_video_start_latent_idx": 1,
            "retrieved_video_end_latent_idx": 3,
            "retrieved_action_latent_idx": 4,
            "retrieved_actions": torch.from_numpy(pool_normalized),
            "retrieved_proprio": torch.from_numpy(pool_normalized[0]),
            "retrieved_state_latent_idx": 3,
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
            "episode_id": record.episode_id,
            "task": record.task,
            "split": record.split,
            "phase": np.float32(record.phase),
            "current_row": record.current_row,
            "method_id": self.method_id,
            "target_representation": target_representation,
            "strict_future_offset_view_steps": 1,
            "gap_crossing_count": 0,
            "query_command_status": "unverified",
            "deployment_command_adapter_id": "",
            "protocol_id": PROTOCOL_ID,
            "protocol_file_sha256": self.protocol_file_sha256,
            "adapter_schema_version": FORMAL_SCHEMA_VERSION,
            "diagnostic_overfit_mode": int(self.diagnostic_overfit_window_index is not None),
            "diagnostic_overfit_seed": self.seed,
            "raw_aligned_pool": torch.from_numpy(aligned_plan.astype(np.float32)),
            "raw_query_target": torch.from_numpy(query_target.astype(np.float32)),
            "raw_residual": torch.from_numpy(residual.astype(np.float32)),
        }

    def contract_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": FORMAL_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "protocol_file_sha256": self.protocol_file_sha256,
            "canonical_root": str(self.canonical_root),
            "main_view_path": str(self.main_view_path),
            "split": self.split,
            "method_id": self.method_id,
            "seed": self.seed,
            "window_count": len(self.windows),
            "H_steps": self.horizon,
            "K_steps": self.horizon,
            "final_image_size": self.final_image_size,
            "tokenizer_chunk_duration": 37,
            "text_conditioning": self.text_conditioning,
            "diagnostic_overfit_window_index": self.diagnostic_overfit_window_index,
            "query_command_status": "unverified",
            "deployment_command_adapter_id": None,
            "heldout_robot_trajectory_usage": "offline target evaluation only",
        }


def build_human2robot_formal_dataset(**kwargs: Any) -> Human2RobotFormalDataset:
    """Isolate the formal adapter from recursively merged PushT kwargs."""
    keys = set(kwargs)
    missing = FORMAL_DATASET_KWARGS - keys
    unknown = keys - FORMAL_DATASET_KWARGS - QUARANTINED_PUSHT_DATASET_KWARGS
    _require(not missing, f"Resolved config is missing formal dataset kwargs: {sorted(missing)}")
    _require(not unknown, f"Resolved config contains unknown inherited dataset kwargs: {sorted(unknown)}")
    formal = {key: kwargs[key] for key in FORMAL_DATASET_KWARGS}
    return Human2RobotFormalDataset(**formal)
