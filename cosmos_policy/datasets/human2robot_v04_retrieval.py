"""Leakage-safe Human2Robot v04 retrieval contract.

This module is intentionally independent from the frozen v03 P2 dataset.  It
consumes the role-only projections and source identities produced by v04
stage 1, and it exposes only history/current-row inputs to retrieval features.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = "human2robot-v04-stage2-retrieval-v1"
FROZEN_SEED = 20260711
H_STEPS = 8
K_STEPS = 8
TOP_K = 3
PRIMARY_RETRIEVAL_MODALITY = "geometry_plus_visual"
ORACLE_RETRIEVAL_MODALITY = "oracle_phase"
POOL_SIZES = (1, 2, 4, 8, 10)
QUERY_PARTITIONS = frozenset({"v04_robot_dev", "v04_robot_final"})
CANDIDATE_PARTITION = "v04_human_pool"
HUMAN_DATASETS = (
    "data/demo_0/human/hand_action_7d",
    "data/demo_0/human/hand_coords",
    "data/demo_0/human/hand_frames",
    "data/demo_0/human/images",
    "data/demo_0/time/gap_mask",
    "data/demo_0/time/legal_window_start",
    "data/demo_0/time/segment_id",
    "data/demo_0/time/source_step",
    "data/demo_0/time/source_timestamp",
)
ROBOT_DATASETS = (
    "data/demo_0/robot/gripper_state",
    "data/demo_0/robot/images",
    "data/demo_0/robot/observed_eef_pose_6d",
    "data/demo_0/time/gap_mask",
    "data/demo_0/time/legal_window_start",
    "data/demo_0/time/segment_id",
    "data/demo_0/time/source_step",
    "data/demo_0/time/source_timestamp",
)
HUMAN_GEOMETRY_DATASETS = ("data/demo_0/human/hand_action_7d",)
ROBOT_GEOMETRY_DATASETS = (
    "data/demo_0/robot/observed_eef_pose_6d",
    "data/demo_0/robot/gripper_state",
)


class Human2RobotV04RetrievalError(RuntimeError):
    """A fail-closed v04 retrieval or provenance violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Human2RobotV04RetrievalError(message)


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    _require(bool(np.all(np.isfinite(array))), "Retrieval feature contains NaN/Inf")
    norm = float(np.linalg.norm(array))
    _require(norm > 1e-12, "Retrieval feature has zero norm")
    return (array / norm).astype(np.float32)


def stable_sha256(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def projection_dataset_paths(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    with h5py.File(path, "r") as file:
        file.visititems(lambda name, value: result.append(name) if isinstance(value, h5py.Dataset) else None)
    return tuple(sorted(result))


@dataclass(frozen=True)
class P2Window:
    """A v04 window carrying immutable raw-source and role identities."""

    window_id: str
    episode_id: str
    path: Path
    task: str
    role: str
    source_sha256: str
    source_relative_path: str
    source_partition: str
    role_content_sha256: str
    current_row: int
    history_rows: tuple[int, ...]
    future_rows: tuple[int, ...]
    dataset_paths: tuple[str, ...]
    pool_rank: int | None = None
    phase: float | None = None


@dataclass(frozen=True)
class FeatureProvenance:
    role: str
    source_sha256: str
    source_relative_path: str
    source_partition: str
    geometry_datasets: tuple[str, ...]
    geometry_rows: tuple[int, ...]
    visual_dataset: str
    visual_row: int
    visual_feature_kind: str
    future_rows_read: tuple[int, ...] = ()
    target_datasets_read: tuple[str, ...] = ()
    opposite_role_datasets_read: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalFeature:
    geometry: np.ndarray
    visual: np.ndarray
    provenance: FeatureProvenance


@dataclass(frozen=True)
class RetrievalRecord:
    query_id: str
    candidate_id: str
    query_source_sha256: str
    candidate_source_sha256: str
    query_source_relative_path: str
    candidate_source_relative_path: str
    query_partition: str
    candidate_partition: str
    retrieval_rank: int
    distance: float
    tie_sha256: str
    modality: str
    query_feature_provenance: FeatureProvenance
    candidate_feature_provenance: FeatureProvenance

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["query_feature_provenance"] = asdict(self.query_feature_provenance)
        value["candidate_feature_provenance"] = asdict(self.candidate_feature_provenance)
        return value


def validate_primary_config(config: Mapping[str, Any]) -> None:
    modality = str(config.get("retrieval_modality", ""))
    _require(
        modality == PRIMARY_RETRIEVAL_MODALITY,
        f"Primary retrieval must be {PRIMARY_RETRIEVAL_MODALITY}; phase/oracle methods are diagnostic-only",
    )
    _require(int(config.get("top_k", -1)) == TOP_K, f"Primary top_k must remain {TOP_K}")
    _require(int(config.get("pool_size", -1)) == 10, "Primary pool_size must remain 10")


def _validate_window(window: P2Window) -> None:
    _require(len(window.source_sha256) == 64, f"Invalid source SHA: {window.window_id}")
    _require(bool(window.source_relative_path), f"Missing source path: {window.window_id}")
    _require(len(window.history_rows) == H_STEPS, f"History must contain H={H_STEPS}: {window.window_id}")
    _require(len(window.future_rows) == K_STEPS, f"Future must contain K={K_STEPS}: {window.window_id}")
    _require(window.history_rows[-1] == window.current_row, f"Current row is not the history anchor: {window.window_id}")
    _require(max(window.history_rows) < min(window.future_rows), f"History/future overlap: {window.window_id}")
    _require(tuple(sorted(window.history_rows)) == window.history_rows, f"History rows are unordered: {window.window_id}")


def window_from_manifest_record(record: Mapping[str, Any], legal_window_start: int) -> P2Window:
    """Bind one legal H8/K8 window to a stage-1 manifest record."""

    role = str(record["role"])
    partition = str(record["source_partition"])
    _require(role in {"human", "robot"}, f"Unknown projection role: {role}")
    _require("projection" in record, f"Record has no role-only projection: {record.get('episode_id')}")
    path = Path(str(record["projection"]["path"]))
    _require(path.is_file(), f"Projection is missing: {path}")
    datasets = projection_dataset_paths(path)
    expected = HUMAN_DATASETS if role == "human" else ROBOT_DATASETS
    _require(datasets == tuple(sorted(expected)), f"Projection allowlist mismatch: {path}")
    start = int(legal_window_start)
    history = tuple(range(start, start + H_STEPS))
    future = tuple(range(start + H_STEPS, start + H_STEPS + K_STEPS))
    content_key = "human_content_sha256" if role == "human" else "robot_content_sha256"
    pool_rank = int(record["partition_rank"]) if partition == CANDIDATE_PARTITION else None
    with h5py.File(path, "r") as file:
        demo = file["data/demo_0"]
        legal = np.asarray(demo["time/legal_window_start"][:], dtype=np.int64)
        _require(bool(np.any(legal == start)), f"Window start is not legal: {path}:{start}")
        for field in ("source_sha256", "source_relative_path", "source_partition", "task", "episode_id", "role"):
            _require(str(demo.attrs.get(field)) == str(record[field]), f"Manifest/projection identity mismatch: {field}")
        segment = np.asarray(demo["time/segment_id"][start : start + H_STEPS + K_STEPS], dtype=np.int64)
        _require(len(segment) == H_STEPS + K_STEPS and len(set(segment.tolist())) == 1, f"Window crosses a gap: {path}:{start}")
        frame_count = int(demo.attrs["frame_count"])
    _require(future[-1] < frame_count, f"Window exceeds projection: {path}:{start}")
    window = P2Window(
        window_id=f"{record['source_sha256']}:{start}:H{H_STEPS}:K{K_STEPS}",
        episode_id=str(record["episode_id"]),
        path=path,
        task=str(record["task"]),
        role=role,
        source_sha256=str(record["source_sha256"]),
        source_relative_path=str(record["source_relative_path"]),
        source_partition=partition,
        role_content_sha256=str(record[content_key]),
        current_row=history[-1],
        history_rows=history,
        future_rows=future,
        dataset_paths=datasets,
        pool_rank=pool_rank,
        phase=float((history[-1] + 1) / frame_count),
    )
    _validate_window(window)
    return window


def candidate_rejection_reason(query: P2Window, candidate: P2Window, pool_size: int) -> str | None:
    """Return the first fail-closed reason a candidate cannot serve a query."""

    _validate_window(query)
    _validate_window(candidate)
    _require(query.role == "robot", f"Query is not robot-only: {query.window_id}")
    _require(query.source_partition in QUERY_PARTITIONS, f"Query partition is not dev/final: {query.source_partition}")
    _require(tuple(sorted(query.dataset_paths)) == tuple(sorted(ROBOT_DATASETS)), "Query projection contains non-robot fields")
    _require(pool_size in POOL_SIZES, f"Unregistered pool size: {pool_size}")
    if candidate.source_sha256 == query.source_sha256:
        return "same_source_sha256"
    if candidate.source_relative_path == query.source_relative_path:
        return "same_source_relative_path"
    if candidate.source_partition != CANDIDATE_PARTITION:
        return "candidate_not_v04_human_pool"
    if candidate.role != "human":
        return "candidate_not_human_only"
    if tuple(sorted(candidate.dataset_paths)) != tuple(sorted(HUMAN_DATASETS)):
        return "candidate_dataset_allowlist_violation"
    if candidate.pool_rank is None or not 1 <= candidate.pool_rank <= pool_size:
        return "candidate_outside_active_pool"
    if candidate.task != query.task:
        return "different_task"
    return None


def filter_candidates(
    query: P2Window,
    candidates: Sequence[P2Window],
    *,
    pool_size: int,
) -> tuple[list[P2Window], dict[str, int]]:
    eligible: list[P2Window] = []
    rejected: dict[str, int] = {}
    for candidate in candidates:
        reason = candidate_rejection_reason(query, candidate, pool_size)
        if reason is None:
            eligible.append(candidate)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    eligible.sort(key=lambda item: (int(item.pool_rank or 0), item.source_sha256, item.window_id))
    return eligible, dict(sorted(rejected.items()))


def assert_pool_growth_nested(candidates: Sequence[P2Window]) -> dict[int, dict[str, tuple[str, ...]]]:
    by_task: dict[str, list[P2Window]] = {}
    for candidate in candidates:
        _require(candidate.source_partition == CANDIDATE_PARTITION and candidate.role == "human", "Pool contains a non-human candidate")
        by_task.setdefault(candidate.task, []).append(candidate)
    result: dict[int, dict[str, tuple[str, ...]]] = {size: {} for size in POOL_SIZES}
    for task, values in sorted(by_task.items()):
        rank_to_sha: dict[int, str] = {}
        for value in values:
            _require(value.pool_rank is not None, f"Pool rank missing: {value.window_id}")
            rank_to_sha.setdefault(value.pool_rank, value.source_sha256)
            _require(rank_to_sha[value.pool_rank] == value.source_sha256, f"Pool rank maps to multiple sources: {task}:{value.pool_rank}")
        _require(tuple(sorted(rank_to_sha)) == tuple(range(1, 11)), f"Pool ranks are not exactly 1..10: {task}")
        previous: set[str] = set()
        for size in POOL_SIZES:
            active = {rank_to_sha[rank] for rank in range(1, size + 1)}
            _require(previous.issubset(active), f"Pool growth is not nested: {task}:{size}")
            result[size][task] = tuple(rank_to_sha[rank] for rank in range(1, size + 1))
            previous = active
    return result


def poses_euler_to_10d(poses_6d: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_6d, dtype=np.float64)
    grip = np.asarray(gripper, dtype=np.float64).reshape(-1)
    _require(poses.ndim == 2 and poses.shape[1] == 6, f"Expected pose shape (T,6), got {poses.shape}")
    _require(len(poses) == len(grip), "Pose/gripper length mismatch")
    _require(bool(np.all(np.isfinite(poses))) and bool(np.all(np.isfinite(grip))), "Pose/gripper contains NaN/Inf")
    matrices = Rotation.from_euler("XYZ", poses[:, 3:6], degrees=True).as_matrix()
    rotation_6d = np.concatenate((matrices[:, :, 0], matrices[:, :, 1]), axis=1)
    return np.concatenate((poses[:, :3] * 0.001, rotation_6d, grip[:, None]), axis=1).astype(np.float32)


def read_feature_inputs(window: P2Window) -> tuple[np.ndarray, np.ndarray, FeatureProvenance]:
    """Read exactly H history states and the current image, never future/target rows."""

    _validate_window(window)
    rows = np.asarray(window.history_rows, dtype=np.int64)
    with h5py.File(window.path, "r") as file:
        demo = file["data/demo_0"]
        if window.role == "human":
            _require(window.source_partition == CANDIDATE_PARTITION, "Human feature input is not from v04_human_pool")
            action = np.asarray(demo["human/hand_action_7d"][rows], dtype=np.float64)
            history = poses_euler_to_10d(action[:, :6], action[:, 6])
            current_image = np.asarray(demo["human/images"][window.current_row], dtype=np.uint8)
            geometry_datasets = HUMAN_GEOMETRY_DATASETS
            visual_dataset = "data/demo_0/human/images"
        else:
            _require(window.source_partition in QUERY_PARTITIONS, "Robot feature input is not dev/final")
            poses = np.asarray(demo["robot/observed_eef_pose_6d"][rows], dtype=np.float64)
            gripper = np.asarray(demo["robot/gripper_state"][rows], dtype=np.float64)
            history = poses_euler_to_10d(poses, gripper)
            current_image = np.asarray(demo["robot/images"][window.current_row], dtype=np.uint8)
            geometry_datasets = ROBOT_GEOMETRY_DATASETS
            visual_dataset = "data/demo_0/robot/images"
    provenance = FeatureProvenance(
        role=window.role,
        source_sha256=window.source_sha256,
        source_relative_path=window.source_relative_path,
        source_partition=window.source_partition,
        geometry_datasets=geometry_datasets,
        geometry_rows=window.history_rows,
        visual_dataset=visual_dataset,
        visual_row=window.current_row,
        visual_feature_kind="current_frame_pending_frozen_wan_latent",
    )
    validate_feature_provenance(window, provenance, require_frozen_visual=False)
    return history, current_image, provenance


def fit_geometry_statistics(histories: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    _require(bool(histories), "No seen-train histories supplied for geometry statistics")
    relative_rows = []
    for history in histories:
        value = np.asarray(history, dtype=np.float64)
        _require(value.shape == (H_STEPS, 10), f"Invalid geometry history: {value.shape}")
        relative_rows.append(value - value[-1])
    stacked = np.concatenate(relative_rows, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    _require(bool(np.all(std > 1e-12)), "Degenerate seen-train geometry statistics")
    return mean.astype(np.float32), std.astype(np.float32)


def geometry_feature(history_10d: np.ndarray, mean_10d: np.ndarray, std_10d: np.ndarray) -> np.ndarray:
    history = np.asarray(history_10d, dtype=np.float64)
    mean = np.asarray(mean_10d, dtype=np.float64)
    std = np.asarray(std_10d, dtype=np.float64)
    _require(history.shape == (H_STEPS, 10), f"Invalid Hx10 geometry history: {history.shape}")
    _require(mean.shape == std.shape == (10,), "Geometry statistics must be 10D")
    _require(bool(np.all(std > 1e-12)), "Geometry std must be positive")
    relative = history - history[-1]
    return _l2_normalize(((relative - mean) / std).reshape(-1))


def visual_feature_from_wan_latent(latent: np.ndarray) -> np.ndarray:
    value = np.asarray(latent, dtype=np.float64)
    if value.ndim == 4:
        _require(value.shape[0] == 1, f"Expected a single WAN latent frame, got {value.shape}")
        value = value[0]
    if value.ndim == 3:
        value = value.mean(axis=(-2, -1))
    _require(value.ndim == 1, f"Invalid WAN latent feature shape: {value.shape}")
    return _l2_normalize(value)


def build_retrieval_feature(
    window: P2Window,
    *,
    geometry_mean_10d: np.ndarray,
    geometry_std_10d: np.ndarray,
    frozen_wan_encoder: Callable[[np.ndarray], np.ndarray],
) -> RetrievalFeature:
    history, current_image, provenance = read_feature_inputs(window)
    latent = frozen_wan_encoder(current_image)
    frozen_provenance = FeatureProvenance(
        **{**asdict(provenance), "visual_feature_kind": "frozen_wan_latent"}
    )
    feature = RetrievalFeature(
        geometry=geometry_feature(history, geometry_mean_10d, geometry_std_10d),
        visual=visual_feature_from_wan_latent(latent),
        provenance=frozen_provenance,
    )
    validate_feature_provenance(window, feature.provenance, require_frozen_visual=True)
    return feature


def validate_feature_provenance(
    window: P2Window,
    provenance: FeatureProvenance,
    *,
    require_frozen_visual: bool,
) -> None:
    _require(provenance.source_sha256 == window.source_sha256, "Feature/source SHA mismatch")
    _require(provenance.source_relative_path == window.source_relative_path, "Feature/source path mismatch")
    _require(provenance.source_partition == window.source_partition, "Feature/source partition mismatch")
    _require(provenance.geometry_rows == window.history_rows, "Geometry did not read exactly the H history rows")
    _require(max(provenance.geometry_rows) <= window.current_row, "Geometry read a future row")
    _require(provenance.visual_row == window.current_row, "Visual feature did not use the current frame")
    _require(not provenance.future_rows_read, "Retrieval feature read query/candidate future rows")
    _require(not provenance.target_datasets_read, "Retrieval feature read a target/action dataset")
    _require(not provenance.opposite_role_datasets_read, "Retrieval feature read opposite-role data")
    if require_frozen_visual:
        _require(provenance.visual_feature_kind == "frozen_wan_latent", "Primary visual feature is not a frozen WAN latent")


def _combined_feature(feature: RetrievalFeature) -> np.ndarray:
    return np.concatenate((_l2_normalize(feature.geometry), _l2_normalize(feature.visual))).astype(np.float32) / np.sqrt(2.0)


def rank_geometry_plus_visual(
    query: P2Window,
    candidates: Sequence[P2Window],
    features: Mapping[str, RetrievalFeature],
    *,
    run_seed: int = FROZEN_SEED,
    pool_size: int = 10,
    top_k: int = TOP_K,
) -> list[RetrievalRecord]:
    _require(run_seed == FROZEN_SEED, f"Primary seed must remain {FROZEN_SEED}")
    _require(top_k == TOP_K, f"Primary top_k must remain {TOP_K}")
    eligible, _ = filter_candidates(query, candidates, pool_size=pool_size)
    _require(len(eligible) >= top_k, f"Fewer than top-{top_k} leakage-safe candidates for {query.window_id}")
    _require(query.window_id in features, f"Query feature missing: {query.window_id}")
    query_feature = features[query.window_id]
    validate_feature_provenance(query, query_feature.provenance, require_frozen_visual=True)
    query_vector = _combined_feature(query_feature)
    ranked: list[tuple[P2Window, float, str]] = []
    for candidate in eligible:
        _require(candidate.window_id in features, f"Candidate feature missing: {candidate.window_id}")
        candidate_feature = features[candidate.window_id]
        validate_feature_provenance(candidate, candidate_feature.provenance, require_frozen_visual=True)
        distance = float(np.linalg.norm(query_vector - _combined_feature(candidate_feature)))
        _require(np.isfinite(distance), f"Nonfinite retrieval distance: {candidate.window_id}")
        tie = stable_sha256(run_seed, query.window_id, candidate.role_content_sha256)
        ranked.append((candidate, distance, tie))
    ranked.sort(key=lambda item: (item[1], item[2], item[0].window_id))
    return [
        RetrievalRecord(
            query_id=query.window_id,
            candidate_id=candidate.window_id,
            query_source_sha256=query.source_sha256,
            candidate_source_sha256=candidate.source_sha256,
            query_source_relative_path=query.source_relative_path,
            candidate_source_relative_path=candidate.source_relative_path,
            query_partition=query.source_partition,
            candidate_partition=candidate.source_partition,
            retrieval_rank=rank,
            distance=distance,
            tie_sha256=tie,
            modality=PRIMARY_RETRIEVAL_MODALITY,
            query_feature_provenance=query_feature.provenance,
            candidate_feature_provenance=features[candidate.window_id].provenance,
        )
        for rank, (candidate, distance, tie) in enumerate(ranked[:top_k])
    ]


def rank_oracle_phase(
    query: P2Window,
    candidates: Sequence[P2Window],
    *,
    primary_completion_receipt_sha256: str,
    pool_size: int = 10,
) -> list[tuple[P2Window, float, str]]:
    """Diagnostic oracle; impossible to call before a bound primary completion."""

    _require(len(primary_completion_receipt_sha256) == 64, "oracle_phase requires a primary completion receipt SHA256")
    _require(query.phase is not None, "Query phase is missing")
    eligible, _ = filter_candidates(query, candidates, pool_size=pool_size)
    ranked = []
    for candidate in eligible:
        _require(candidate.phase is not None, f"Candidate phase is missing: {candidate.window_id}")
        ranked.append(
            (
                candidate,
                abs(float(query.phase) - float(candidate.phase)),
                stable_sha256(FROZEN_SEED, query.window_id, candidate.role_content_sha256),
            )
        )
    return sorted(ranked, key=lambda item: (item[1], item[2], item[0].window_id))
