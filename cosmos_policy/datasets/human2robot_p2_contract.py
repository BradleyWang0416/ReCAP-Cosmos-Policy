"""Pure, deterministic contracts shared by the formal Human2Robot P2 runtime.

The functions in this module deliberately avoid model or filesystem state.  A
formal artifact can therefore bind their source hash and test retrieval,
representation, reconstruction, preprocessing, and metric semantics without
loading the 2B network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

FROZEN_SEEDS = (20260711, 20260712, 20260713)
RETRIEVAL_MODALITIES = {
    "random",
    "phase",
    "geometry",
    "visual",
    "geometry_plus_visual",
}
RESOLUTION_VARIANTS = {
    "source_240x426_then_resize_224",
    "center_crop_240x424_then_resize_224",
    "center_crop_240x424_edge_pad_240x426_then_resize_224",
}


class Human2RobotP2ContractError(RuntimeError):
    """Raised when a P2 transform would violate the frozen execution spec."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Human2RobotP2ContractError(message)


def stable_sha256(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def deterministic_inference_seed(
    run_seed: int,
    experiment_id: str,
    variant_id: str,
    task: str,
    episode_id: str,
    current_row: int,
    retrieval_rank: int,
) -> int:
    payload = ":".join(
        (
            str(run_seed),
            experiment_id,
            variant_id,
            task,
            episode_id,
            str(current_row),
            str(retrieval_rank),
        )
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFF


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    _require(np.isfinite(norm) and norm > 1e-12, "Cannot L2-normalize a zero/nonfinite vector")
    return (array / norm).astype(np.float32)


def geometry_feature(
    history_10d: np.ndarray,
    mean_10d: Sequence[float],
    std_10d: Sequence[float],
) -> np.ndarray:
    """Relative, train-standardized, flattened, L2-normalized Hx10 feature."""

    history = np.asarray(history_10d, dtype=np.float64)
    mean = np.asarray(mean_10d, dtype=np.float64)
    std = np.asarray(std_10d, dtype=np.float64)
    _require(history.ndim == 2 and history.shape[1] == 10, f"Invalid geometry history {history.shape}")
    _require(mean.shape == (10,) and std.shape == (10,), "Geometry stats must be 10D")
    _require(bool(np.all(np.isfinite(history))), "Geometry history is nonfinite")
    _require(bool(np.all(std > 1e-12)), "Geometry std must be positive")
    relative = history - history[-1]
    standardized = (relative - mean) / std
    return l2_normalize(standardized.reshape(-1))


def visual_feature_from_latent(latent: np.ndarray) -> np.ndarray:
    """Spatially mean-pool an anchor latent frame and L2 normalize channels."""

    value = np.asarray(latent, dtype=np.float64)
    if value.ndim == 4:
        _require(value.shape[0] == 1, f"Expected one latent frame, got {value.shape}")
        value = value[:, 0] if value.shape[1] == 1 else value[0]
    if value.ndim == 3:
        pooled = value.mean(axis=(-2, -1))
    elif value.ndim == 1:
        pooled = value
    else:
        raise Human2RobotP2ContractError(f"Invalid visual latent shape: {value.shape}")
    return l2_normalize(pooled)


def combined_geometry_visual_feature(geometry: np.ndarray, visual: np.ndarray) -> np.ndarray:
    return np.concatenate((l2_normalize(geometry), l2_normalize(visual))).astype(np.float32) / np.sqrt(2.0)


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    human_content_sha256: str
    phase: float
    geometry: np.ndarray | None = None
    visual: np.ndarray | None = None


def rank_retrieval_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    modality: str,
    run_seed: int,
    query_id: str,
    query_phase: float,
    query_geometry: np.ndarray | None = None,
    query_visual: np.ndarray | None = None,
) -> list[tuple[RetrievalCandidate, float, str]]:
    """Return candidates sorted by distance then the frozen seeded hash tie-break."""

    _require(modality in RETRIEVAL_MODALITIES, f"Unknown retrieval modality: {modality}")
    _require(run_seed in FROZEN_SEEDS, f"Unregistered run seed: {run_seed}")
    _require(bool(candidates), "Retrieval candidate set is empty")
    ranked: list[tuple[RetrievalCandidate, float, str]] = []
    for candidate in candidates:
        tie = stable_sha256(run_seed, query_id, candidate.human_content_sha256)
        if modality == "random":
            distance = 0.0
        elif modality == "phase":
            distance = abs(float(query_phase) - float(candidate.phase))
        elif modality == "geometry":
            _require(query_geometry is not None and candidate.geometry is not None, "Geometry feature missing")
            distance = float(np.linalg.norm(l2_normalize(query_geometry) - l2_normalize(candidate.geometry)))
        elif modality == "visual":
            _require(query_visual is not None and candidate.visual is not None, "Visual feature missing")
            distance = 1.0 - float(np.dot(l2_normalize(query_visual), l2_normalize(candidate.visual)))
        else:
            _require(
                query_geometry is not None
                and candidate.geometry is not None
                and query_visual is not None
                and candidate.visual is not None,
                "Combined retrieval feature missing",
            )
            query = combined_geometry_visual_feature(query_geometry, query_visual)
            pool = combined_geometry_visual_feature(candidate.geometry, candidate.visual)
            distance = float(np.linalg.norm(query - pool))
        _require(np.isfinite(distance), f"Nonfinite retrieval distance for {candidate.candidate_id}")
        ranked.append((candidate, distance, tie))
    return sorted(ranked, key=lambda item: (item[1], item[2], item[0].candidate_id))


def future_state_target(current_state_10d: np.ndarray, future_states_10d: np.ndarray) -> np.ndarray:
    current = np.asarray(current_state_10d, dtype=np.float64)
    future = np.asarray(future_states_10d, dtype=np.float64)
    _require(current.shape == (10,), f"Invalid current state: {current.shape}")
    _require(future.ndim == 2 and future.shape[1] == 10, f"Invalid future states: {future.shape}")
    previous = np.concatenate((current[None], future[:-1]), axis=0)
    return (future - previous).astype(np.float32)


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64)
    _require(values.shape[-1] == 6, f"Invalid rotation-6D shape: {values.shape}")
    flat = values.reshape(-1, 6)
    first = flat[:, :3]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = flat[:, 3:]
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1).reshape(values.shape[:-1] + (3, 3))


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((value[..., :, 0], value[..., :, 1]), axis=-1)


def project_canonical_trajectory(trajectory_10d: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory_10d, dtype=np.float64).copy()
    _require(value.ndim == 2 and value.shape[1] == 10, f"Invalid trajectory: {value.shape}")
    _require(bool(np.all(np.isfinite(value))), "Trajectory contains nonfinite values")
    value[:, 3:9] = _matrix_to_rotation_6d(_rotation_6d_to_matrix(value[:, 3:9]))
    value[:, 9] = np.clip(value[:, 9], 0.0, 1.0)
    return value.astype(np.float32)


def reconstruct_future_state(current_state_10d: np.ndarray, transitions_10d: np.ndarray) -> np.ndarray:
    current = np.asarray(current_state_10d, dtype=np.float64)
    transitions = np.asarray(transitions_10d, dtype=np.float64)
    _require(current.shape == (10,), f"Invalid current state: {current.shape}")
    _require(transitions.ndim == 2 and transitions.shape[1] == 10, "Invalid future-state transitions")
    return project_canonical_trajectory(current[None] + np.cumsum(transitions, axis=0))


def aggregate_canonical_predictions(predictions: Sequence[np.ndarray]) -> np.ndarray:
    _require(bool(predictions), "No retrieval-rank predictions to aggregate")
    values = [project_canonical_trajectory(item) for item in predictions]
    reference_shape = values[0].shape
    _require(all(item.shape == reference_shape for item in values), "Prediction shapes differ")
    return project_canonical_trajectory(np.mean(values, axis=0))


def preprocess_resolution_frames(frames: np.ndarray, variant_id: str) -> torch.Tensor:
    """Apply one frozen source/crop/pad rule and return uint8 C,T,224,224."""

    _require(variant_id in RESOLUTION_VARIANTS, f"Unknown resolution variant: {variant_id}")
    value = np.asarray(frames)
    _require(value.ndim == 4 and value.shape[1:] == (240, 426, 3), f"Invalid source RGB {value.shape}")
    if variant_id != "source_240x426_then_resize_224":
        value = value[:, :, 1:-1, :]
        if variant_id == "center_crop_240x424_edge_pad_240x426_then_resize_224":
            value = np.pad(value, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    return tensor.clamp(0.0, 255.0).round().to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def orientation_error_rad(prediction_6d: np.ndarray, target_6d: np.ndarray) -> np.ndarray:
    pred = _rotation_6d_to_matrix(prediction_6d)
    target = _rotation_6d_to_matrix(target_6d)
    relative = np.swapaxes(pred, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def canonical_window_metrics(prediction_10d: np.ndarray, target_10d: np.ndarray) -> Mapping[str, float]:
    prediction = project_canonical_trajectory(prediction_10d)
    target = project_canonical_trajectory(target_10d)
    _require(prediction.shape == target.shape, "Prediction/target shapes differ")
    position = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1)
    orientation = orientation_error_rad(prediction[:, 3:9], target[:, 3:9])
    gripper = np.abs(prediction[:, 9] - target[:, 9])
    canonical = np.linalg.norm(prediction - target, axis=1)
    return {
        "position_error_median_canonical": float(np.median(position)),
        "orientation_error_median_rad": float(np.median(orientation)),
        "gripper_error_median": float(np.median(gripper)),
        "final_position_error_median_canonical": float(position[-1]),
        "canonical_error_median": float(np.median(canonical)),
    }

