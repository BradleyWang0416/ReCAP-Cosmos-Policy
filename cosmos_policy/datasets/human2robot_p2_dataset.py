"""Pool-backed, variant-aware Human2Robot dataset for frozen M5B-P2 cells."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.human2robot_dataset import (
    CANONICAL_SCHEMA_VERSION,
    FROZEN_SEEDS,
    PROTOCOL_ID,
    QUARANTINED_PUSHT_DATASET_KWARGS,
    Human2RobotContractError,
    _contiguous_segments,
    _normalize,
    _preprocess_video,
    _read_json,
    _require,
    align_pool_chunk,
    file_sha256,
)
from cosmos_policy.datasets.human2robot_p2_contract import (
    RESOLUTION_VARIANTS,
    RetrievalCandidate,
    geometry_feature,
    preprocess_resolution_frames,
    rank_retrieval_candidates,
)

P2_SCHEMA_VERSION = "human2robot-formal-cosmos-adapter-p2-v2"
SUPPLEMENT_SHA256 = "17d9fc308c50b9b7899793a4c8d3bca1eeba217053fbacb368e2f9a2e390d7ab"
P1_POOL_MANIFEST_SHA256 = "47e87be5800194de6e0ac99b47dbe23ef96a91298edbff3e9996b1484b489299"
P1_SELECTION_ID = "48e0c0f5c283a5a7b9f3de8eb6535f13f5f760cc325a81413053015fd6299afd"
ALLOWED_METHODS = {"no_retrieval", "retrieval_only", "co_training", "recap_hand_ret"}
TIME_VIEW_IDS = {
    "nominal_camera_30hz_segmented",
    "paper_v2_stride4_nominal7p5",
    "legacy_v01_stride3_nominal10",
    "policy_clock_10hz",
    "phase_or_dtw",
}
TIME_VIEW_STRIDES = {
    "nominal_camera_30hz_segmented": 1,
    "paper_v2_stride4_nominal7p5": 4,
    "legacy_v01_stride3_nominal10": 3,
    "policy_clock_10hz": 3,
}
P2_DATASET_KWARGS = {
    "canonical_root",
    "main_view_path",
    "m3_report_path",
    "m4_report_path",
    "protocol_path",
    "supplement_path",
    "p1_pool_root",
    "split",
    "method_id",
    "experiment_id",
    "variant_id",
    "seed",
    "h_steps",
    "k_steps",
    "window_stride",
    "top_k",
    "pool_size",
    "retrieval_modality",
    "time_view_id",
    "query_offset_view_steps",
    "target_representation",
    "statistics_path",
    "retrieval_index_path",
    "resolution_variant",
    "use_image_aug",
    "num_duplicates_per_image",
    "text_conditioning",
    "diagnostic_window_limit",
}


@dataclass(frozen=True)
class P2Window:
    window_id: str
    episode_id: str
    path: Path
    source_kind: str
    task: str
    split: str
    segment_number: int
    current_row: int
    history_rows: np.ndarray
    future_rows: np.ndarray
    phase: float
    human_content_sha256: str
    pool_rank: int | None


@dataclass(frozen=True)
class RankedTrainingExample:
    query_index: int
    candidate_index: int | None
    retrieval_rank: int
    distance: float
    tie_sha256: str
    effective_k: int


def _selected_segment_rows(rows: np.ndarray, time_view_id: str) -> np.ndarray:
    if time_view_id == "phase_or_dtw":
        count = min(64, len(rows))
        indices = np.unique(np.rint(np.linspace(0, len(rows) - 1, count)).astype(np.int64))
        return rows[indices]
    return rows[:: TIME_VIEW_STRIDES[time_view_id]]


def _source_record_map(split_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in split_manifest.get("episodes", []):
        source = str(record.get("source_relative_path", ""))
        _require(source and source not in result, f"Invalid duplicate split source: {source}")
        result[source] = record
    return result


def _load_feature_store(path: Path | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if path is None:
        return {}, {}
    _require(path.is_file(), f"Missing retrieval index: {path}")
    with np.load(path, allow_pickle=False) as payload:
        _require("ids" in payload and "features" in payload and "manifest_json" in payload, "Invalid index NPZ")
        ids = [str(item) for item in payload["ids"].tolist()]
        features = np.asarray(payload["features"], dtype=np.float32)
        manifest = json.loads(str(payload["manifest_json"].item()))
    _require(features.ndim == 2 and len(ids) == len(features), "Feature store shape mismatch")
    _require(len(set(ids)) == len(ids), "Duplicate feature IDs")
    return dict(zip(ids, features, strict=True)), manifest


class Human2RobotP2Dataset(Dataset):
    """Formal query/retrieval examples for both learned training and held-out inference."""

    def __init__(
        self,
        canonical_root: str | Path,
        main_view_path: str | Path,
        m3_report_path: str | Path,
        m4_report_path: str | Path,
        protocol_path: str | Path,
        supplement_path: str | Path,
        p1_pool_root: str | Path,
        split: str = "train",
        method_id: str = "recap_hand_ret",
        experiment_id: str = "M5B-MAIN-01",
        variant_id: str = "frozen_main",
        seed: int = 20260711,
        h_steps: int = 8,
        k_steps: int = 8,
        window_stride: int = 8,
        top_k: int = 3,
        pool_size: int = 10,
        retrieval_modality: str = "phase",
        time_view_id: str = "nominal_camera_30hz_segmented",
        query_offset_view_steps: int = 1,
        target_representation: str = "residual",
        statistics_path: str | Path | None = None,
        retrieval_index_path: str | Path | None = None,
        resolution_variant: str = "center_crop_240x424_then_resize_224",
        use_image_aug: bool = True,
        num_duplicates_per_image: int = 4,
        text_conditioning: str = "disabled_zero_embedding",
        diagnostic_window_limit: int | None = None,
    ) -> None:
        self.canonical_root = Path(canonical_root)
        self.main_view_path = Path(main_view_path)
        self.m3_report_path = Path(m3_report_path)
        self.m4_report_path = Path(m4_report_path)
        self.protocol_path = Path(protocol_path)
        self.supplement_path = Path(supplement_path)
        self.p1_pool_root = Path(p1_pool_root)
        self.split = split
        self.method_id = method_id
        self.experiment_id = experiment_id
        self.variant_id = variant_id
        self.seed = int(seed)
        self.h_steps = int(h_steps)
        self.k_steps = int(k_steps)
        self.window_stride = int(window_stride)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.retrieval_modality = retrieval_modality
        self.time_view_id = time_view_id
        self.query_offset_view_steps = int(query_offset_view_steps)
        self.target_representation = target_representation
        self.statistics_path = Path(statistics_path) if statistics_path else None
        self.retrieval_index_path = Path(retrieval_index_path) if retrieval_index_path else None
        self.resolution_variant = resolution_variant
        self.use_image_aug = bool(use_image_aug)
        self.num_duplicates_per_image = int(num_duplicates_per_image)
        self.text_conditioning = text_conditioning
        self.diagnostic_window_limit = diagnostic_window_limit
        self._state_cache: dict[tuple[str, str], np.ndarray] = {}
        self._geometry_cache: dict[tuple[str, str], np.ndarray] = {}

        _require(split in {"train", "heldout"}, f"Invalid split: {split}")
        _require(method_id in ALLOWED_METHODS, f"Invalid method: {method_id}")
        _require(self.seed in FROZEN_SEEDS, f"Seed is not frozen: {self.seed}")
        _require((self.h_steps, self.k_steps) in {(4, 4), (8, 8), (16, 8)}, "Unregistered H/K")
        _require(self.window_stride > 0, "window_stride must be positive")
        _require(self.top_k in {1, 3, 5, 10}, f"Unregistered top-k: {self.top_k}")
        _require(self.pool_size in {0, 1, 2, 4, 8, 10}, f"Unregistered pool size: {self.pool_size}")
        _require(time_view_id in TIME_VIEW_IDS, f"Unregistered time view: {time_view_id}")
        _require(query_offset_view_steps in {1, 5}, "Unregistered query offset")
        _require(
            query_offset_view_steps == 1
            or variant_id == "raw_human_plan_plus_lag_calibrated_query_diagnostic",
            "lag=5 is diagnostic-only",
        )
        _require(target_representation in {"residual", "absolute", "future_state", "retrieval_only"}, "Bad target")
        allowed_targets = {
            "no_retrieval": {"absolute"},
            "co_training": {"absolute"},
            "recap_hand_ret": {"residual", "future_state"},
            "retrieval_only": {"retrieval_only"},
        }
        _require(target_representation in allowed_targets[method_id], "Method/target representation mismatch")
        _require(
            target_representation != "future_state"
            or (experiment_id == "M5B-REP-01" and variant_id == "future_state"),
            "future_state is only registered for M5B-REP-01/future_state",
        )
        _require(resolution_variant in RESOLUTION_VARIANTS, "Unregistered resolution variant")
        _require(self.num_duplicates_per_image == 4, "WAN tokenizer requires four-frame slots")
        _require(text_conditioning == "disabled_zero_embedding", "Text conditioning is not registered")
        _require(diagnostic_window_limit is None or diagnostic_window_limit > 0, "Bad diagnostic limit")

        self.protocol = _read_json(self.protocol_path)
        self.supplement = _read_json(self.supplement_path)
        views_root = self.main_view_path.parents[3]
        pool_action_view = "human_hand_phase_aligned" if time_view_id == "phase_or_dtw" else "human_hand_robot_frame_raw"
        if self.query_offset_view_steps == 5:
            query_action_view = "robot_ee_observed_t_plus_5_lag_diagnostic"
            alignment_id = "train_only_tplus5_query_anchor_se3_identity_scale_v1"
        else:
            query_action_view = "robot_ee_observed_t_plus_1_bc_proxy"
            alignment_id = "train_only_tplus1_query_anchor_se3_identity_scale_v1"
        self.time_view_path = views_root / time_view_id / pool_action_view / query_action_view / alignment_id
        self.view = _read_json(self.time_view_path / "view_manifest.json")
        self.m3_report = _read_json(self.m3_report_path)
        self.m4_report = _read_json(self.m4_report_path)
        self.split_manifest = _read_json(self.canonical_root / "task_split_manifest.json")
        self.preprocessing_manifest = _read_json(self.canonical_root / "preprocessing_manifest.json")
        self.canonical_source_sha = {
            Path(str(item["output_path"])).name: str(item["source_sha256"])
            for item in self.preprocessing_manifest.get("episodes", [])
        }
        self.p1_manifest = _read_json(self.p1_pool_root / "pool_manifest.json")
        default_statistics = self.main_view_path / "action_statistics.json"
        self.statistics = _read_json(self.statistics_path or default_statistics)
        self.features, self.index_manifest = _load_feature_store(self.retrieval_index_path)
        self.protocol_file_sha256 = file_sha256(self.protocol_path)
        self.supplement_file_sha256 = file_sha256(self.supplement_path)
        self.statistics_file_sha256 = file_sha256(self.statistics_path or default_statistics)
        self.retrieval_index_sha256 = (
            file_sha256(self.retrieval_index_path) if self.retrieval_index_path else "phase_or_random_no_feature_npz"
        )
        self._validate_bindings()
        self.queries = self._build_canonical_windows(split)
        _require(bool(self.queries), f"No P2 queries for split={split}")
        self.candidates = (
            self._build_canonical_windows("train", human_candidate=True)
            if split == "train"
            else self._build_p1_windows()
        )
        if self.diagnostic_window_limit is not None:
            self.queries = self.queries[: self.diagnostic_window_limit]
        self.examples = self._rank_examples()
        _require(bool(self.examples), "No executable P2 examples")

    def _validate_bindings(self) -> None:
        _require(self.protocol.get("protocol_id") == PROTOCOL_ID, "Wrong P2 parent protocol")
        _require(self.protocol.get("status") == "frozen_pre_registration", "Parent protocol not frozen")
        _require(file_sha256(self.supplement_path) == SUPPLEMENT_SHA256, "P2 supplement hash changed")
        _require(self.supplement.get("status") == "frozen_approved_execution_spec", "P2 supplement not approved")
        _require(tuple(self.supplement.get("frozen_seeds", [])) == tuple(sorted(FROZEN_SEEDS)), "Seed drift")
        _require(self.split_manifest.get("split_sha256") == self.protocol["frozen_data_contract"]["split_sha256"], "Split drift")
        _require(self.m3_report.get("status") == "passed", "M3 is not passed")
        _require(self.m4_report.get("status") == "launched", "M4 is not launched")
        _require(self.view.get("time_view_id") == self.time_view_id, "Time-view manifest drift")
        _require(
            self.view.get("query_target_offset_view_steps") == self.query_offset_view_steps,
            "Query-offset view manifest drift",
        )
        _require(self.view.get("gap_policy") == "never_cross_segment", "Time view may cross gaps")
        _require(self.view.get("query_command_status") == "unverified", "Command status upgraded")
        _require(file_sha256(self.p1_pool_root / "pool_manifest.json") == P1_POOL_MANIFEST_SHA256, "P1 pool drift")
        _require(self.p1_manifest.get("selection_id") == P1_SELECTION_ID, "P1 selection drift")
        _require(self.p1_manifest.get("status") == "passed", "P1 pool not passed")
        if self.retrieval_modality in {"geometry", "visual", "geometry_plus_visual"}:
            _require(self.retrieval_index_path is not None, "Feature retrieval requires a frozen index")
            _require(self.index_manifest.get("heldout_target_used") is False, "Index uses heldout target")
            _require(self.index_manifest.get("split_sha256") == self.split_manifest["split_sha256"], "Index split drift")
        provenance = self.statistics.get("provenance", {})
        _require(provenance.get("heldout_data_used") is False, "Statistics use heldout data")

    def _iter_canonical_episodes(self, split: str) -> Iterable[tuple[Path, dict[str, Any]]]:
        records = _source_record_map(self.split_manifest)
        ordered_records = list(self.split_manifest.get("episodes", []))
        paths = sorted((self.canonical_root / "pilot").glob("demo_*.hdf5"))
        _require(len(paths) == len(ordered_records), "Canonical/split episode count mismatch")
        for path, ordered_record in zip(paths, ordered_records, strict=True):
            with h5py.File(path, "r") as file:
                demo = file["data/demo_0"]
                source = str(demo.attrs.get("source_relative_path", ""))
                schema = str(demo.attrs.get("schema_version", ""))
            _require(schema == CANONICAL_SCHEMA_VERSION, f"Wrong canonical schema: {path}")
            record = records[source] if source else ordered_record
            _require(not source or source == record.get("source_relative_path"), "Canonical source order drift")
            if record.get("split") == split:
                yield path, record

    def _build_canonical_windows(self, split: str, human_candidate: bool = False) -> list[P2Window]:
        result: list[P2Window] = []
        for path, record in self._iter_canonical_episodes(split):
            _require(path.name in self.canonical_source_sha, f"Canonical source SHA missing: {path}")
            human_content_sha256 = self.canonical_source_sha[path.name]
            with h5py.File(path, "r") as file:
                demo = file["data/demo_0"]
                segment_id = np.asarray(demo["metadata/segment_id"][:], dtype=np.int64)
                gap_mask = np.asarray(demo["metadata/gap_mask"][:], dtype=bool)
            for segment_number, raw_rows in enumerate(_contiguous_segments(segment_id, gap_mask)):
                rows = _selected_segment_rows(raw_rows, self.time_view_id)
                offset = 1 if human_candidate else self.query_offset_view_steps
                future_length = self.h_steps if human_candidate else self.k_steps
                max_current = len(rows) - offset - future_length
                for current_pos in range(self.h_steps - 1, max_current + 1, self.window_stride):
                    history = rows[current_pos - self.h_steps + 1 : current_pos + 1]
                    future = rows[current_pos + offset : current_pos + offset + future_length]
                    _require(len(history) == self.h_steps and len(future) == future_length, "Incomplete window")
                    current_row = int(rows[current_pos])
                    phase = float(current_pos / max(1, len(rows) - 1))
                    role = "human_candidate" if human_candidate else "robot_query"
                    window_id = f"canonical:{path.stem}:{segment_number}:{current_row}:{role}:H{self.h_steps}:K{future_length}"
                    result.append(
                        P2Window(
                            window_id=window_id,
                            episode_id=path.stem,
                            path=path,
                            source_kind="canonical",
                            task=str(record["task"]),
                            split=split,
                            segment_number=segment_number,
                            current_row=current_row,
                            history_rows=history.copy(),
                            future_rows=future.copy(),
                            phase=phase,
                            human_content_sha256=human_content_sha256,
                            pool_rank=None,
                        )
                    )
        return result

    def _build_p1_windows(self) -> list[P2Window]:
        selected = [item for item in self.p1_manifest.get("episodes", []) if int(item["pool_rank"]) <= self.pool_size]
        result: list[P2Window] = []
        for item in selected:
            task = str(item["task"])
            rank = int(item["pool_rank"])
            source_stem = Path(str(item["source_relative_path"])).stem
            path = self.p1_pool_root / "episodes" / task / f"pool_{rank:02d}_{source_stem}.hdf5"
            _require(path.is_file(), f"Missing P1 pool episode: {path}")
            with h5py.File(path, "r") as file:
                demo = file["data/demo_0"]
                _require(bool(demo.attrs.get("human_only")), f"P1 file is not human-only: {path}")
                _require(not bool(demo.attrs.get("contains_robot_observation_or_target")), "P1 leakage flag")
                segment_id = np.asarray(demo["time/segment_id"][:], dtype=np.int64)
                gap_mask = np.asarray(demo["time/gap_mask"][:], dtype=bool)
            for segment_number, raw_rows in enumerate(_contiguous_segments(segment_id, gap_mask)):
                rows = _selected_segment_rows(raw_rows, self.time_view_id)
                max_current = len(rows) - 1 - self.h_steps
                for current_pos in range(self.h_steps - 1, max_current + 1, self.window_stride):
                    history = rows[current_pos - self.h_steps + 1 : current_pos + 1]
                    future = rows[current_pos + 1 : current_pos + 1 + self.h_steps]
                    current_row = int(rows[current_pos])
                    window_id = f"p1:{task}:{rank}:{segment_number}:{current_row}:H{self.h_steps}"
                    result.append(
                        P2Window(
                            window_id=window_id,
                            episode_id=f"pool_{rank:02d}_{source_stem}",
                            path=path,
                            source_kind="p1",
                            task=task,
                            split="heldout_human_only_pool",
                            segment_number=segment_number,
                            current_row=current_row,
                            history_rows=history.copy(),
                            future_rows=future.copy(),
                            phase=float(current_pos / max(1, len(rows) - 1)),
                            human_content_sha256=str(item["human_content_sha256"]),
                            pool_rank=rank,
                        )
                    )
        return result

    def _states(self, window: P2Window, role: str) -> np.ndarray:
        cache_key = (str(window.path), role)
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]
        with h5py.File(window.path, "r") as file:
            demo = file["data/demo_0"]
            if window.source_kind == "p1":
                states = np.asarray(demo["human/hand_plan_10d"][:], dtype=np.float64)
            elif role == "human":
                states = np.asarray(demo["trajectories/human_hand_robot_frame_10d"][:], dtype=np.float64)
            else:
                states = np.asarray(demo["trajectories/robot_ee_observed_10d"][:], dtype=np.float64)
        self._state_cache[cache_key] = states
        return states

    def _state_history(self, window: P2Window, role: str) -> np.ndarray:
        return self._states(window, role)[window.history_rows]

    def _states_and_images(self, window: P2Window, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = self._states(window, role)
        with h5py.File(window.path, "r") as file:
            demo = file["data/demo_0"]
            if window.source_kind == "p1":
                images = np.asarray(demo["human/images"][:], dtype=np.uint8)
            elif role == "human":
                images = np.asarray(demo["metadata/human/images"][:], dtype=np.uint8)
            else:
                images = np.asarray(demo["obs/robot_images"][:], dtype=np.uint8)
        return states, images, states[window.history_rows]

    def _geometry(self, window: P2Window, role: str) -> np.ndarray | None:
        if self.retrieval_modality not in {"geometry", "geometry_plus_visual"}:
            return None
        cache_key = (role, window.window_id)
        if cache_key in self._geometry_cache:
            return self._geometry_cache[cache_key]
        history = self._state_history(window, role)
        mean = self.index_manifest.get("geometry_relative_10d_mean")
        std = self.index_manifest.get("geometry_relative_10d_std")
        _require(mean is not None and std is not None, "Geometry statistics missing from index")
        feature = geometry_feature(history, mean, std)
        self._geometry_cache[cache_key] = feature
        return feature

    def _visual(self, window: P2Window, role: str) -> np.ndarray | None:
        if self.retrieval_modality not in {"visual", "geometry_plus_visual"}:
            return None
        feature_id = f"{role}:{window.window_id}"
        _require(feature_id in self.features, f"Visual feature missing: {feature_id}")
        return self.features[feature_id]

    def _candidate_record(self, candidate: P2Window) -> RetrievalCandidate:
        return RetrievalCandidate(
            candidate_id=candidate.window_id,
            human_content_sha256=candidate.human_content_sha256,
            phase=candidate.phase,
            geometry=self._geometry(candidate, "human"),
            visual=self._visual(candidate, "human"),
        )

    def _eligible_candidate_indices(self, query: P2Window) -> list[int]:
        result = []
        query_target_rows = set(int(item) for item in query.future_rows)
        for index, candidate in enumerate(self.candidates):
            if candidate.task != query.task:
                continue
            if candidate.path == query.path:
                occupied = set(int(item) for item in np.concatenate((candidate.history_rows, candidate.future_rows)))
                if occupied & query_target_rows or candidate.window_id == query.window_id:
                    continue
            result.append(index)
        return result

    def _rank_examples(self) -> list[RankedTrainingExample]:
        examples: list[RankedTrainingExample] = []
        candidate_records: dict[int, RetrievalCandidate] = {}
        replicate_count = self.top_k if self.split == "heldout" else 3
        for query_index, query in enumerate(self.queries):
            if self.method_id == "no_retrieval" or (self.split == "heldout" and self.pool_size == 0):
                examples.extend(
                    RankedTrainingExample(
                        query_index, None, rank, 0.0, "masked_no_retrieval", replicate_count
                    )
                    for rank in range(replicate_count)
                )
                continue
            eligible = self._eligible_candidate_indices(query)
            _require(bool(eligible), f"No same-task leakage-safe candidate for {query.window_id}")
            records = []
            candidate_index_by_id = {}
            for candidate_index in eligible:
                if candidate_index not in candidate_records:
                    candidate_records[candidate_index] = self._candidate_record(
                        self.candidates[candidate_index]
                    )
                record = candidate_records[candidate_index]
                records.append(record)
                candidate_index_by_id[record.candidate_id] = candidate_index
            ranked = rank_retrieval_candidates(
                records,
                modality=self.retrieval_modality,
                run_seed=self.seed,
                query_id=query.window_id,
                query_phase=query.phase,
                query_geometry=self._geometry(query, "robot"),
                query_visual=self._visual(query, "robot"),
            )
            effective = min(replicate_count, len(ranked))
            for rank, (candidate, distance, tie) in enumerate(ranked[:effective]):
                candidate_index = candidate_index_by_id[candidate.candidate_id]
                examples.append(
                    RankedTrainingExample(
                        query_index, candidate_index, rank, distance, tie, effective
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _target_actions(
        self, current: np.ndarray, query_target: np.ndarray, aligned_plan: np.ndarray
    ) -> tuple[np.ndarray, str]:
        if self.target_representation == "residual":
            raw = query_target - aligned_plan[: self.k_steps]
            low = self.statistics["residual_10d_min"]
            high = self.statistics["residual_10d_max"]
        elif self.target_representation == "absolute":
            raw = query_target
            low = self.statistics["query_bc_target_10d_min"]
            high = self.statistics["query_bc_target_10d_max"]
        elif self.target_representation == "future_state":
            previous = np.concatenate((current[None], query_target[:-1]), axis=0)
            raw = query_target - previous
            low = self.statistics["future_state_transition_10d_min"]
            high = self.statistics["future_state_transition_10d_max"]
        else:
            raw = np.zeros_like(query_target)
            low = np.zeros(10)
            high = np.ones(10)
        return _normalize(raw, low, high), self.target_representation

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        query = self.queries[example.query_index]
        robot, robot_images, _ = self._states_and_images(query, "robot")
        current = robot[query.current_row]
        query_target = robot[query.future_rows]

        has_retrieval = int(example.candidate_index is not None and self.method_id != "no_retrieval")
        if example.candidate_index is None:
            raw_plan = np.repeat(current[None], self.h_steps, axis=0)
            human_images = np.zeros((self.h_steps, 240, 426, 3), dtype=np.uint8)
            candidate_id = "masked_no_retrieval"
            candidate_sha = ""
        else:
            candidate = self.candidates[example.candidate_index]
            human, all_human_images, _ = self._states_and_images(candidate, "human")
            raw_plan = human[candidate.future_rows]
            human_images = all_human_images[candidate.future_rows]
            candidate_id = candidate.window_id
            candidate_sha = candidate.human_content_sha256
        aligned_plan = align_pool_chunk(raw_plan, current)
        actions, target_representation = self._target_actions(current, query_target, aligned_plan)
        pool_normalized = _normalize(
            aligned_plan,
            self.statistics["pool_action_10d_min"],
            self.statistics["pool_action_10d_max"],
        )
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
        current_image = robot_images[query.current_row]
        future_image = robot_images[query.future_rows[-1]]
        blank = np.zeros_like(current_image)
        blank4 = np.repeat(blank[None], self.num_duplicates_per_image, axis=0)
        frames = np.concatenate(
            (
                blank[None],
                human_images,
                blank4,
                blank4,
                np.repeat(current_image[None], 4, axis=0),
                blank4,
                blank4,
                np.repeat(future_image[None], 4, axis=0),
                blank4,
            ),
            axis=0,
        )
        _require(len(frames) == 29 + self.h_steps, "P2 WAN frame layout mismatch")
        if self.resolution_variant == "center_crop_240x424_then_resize_224":
            augment_seed = self.seed + example.query_index * 31 + example.retrieval_rank if self.use_image_aug else None
            video = _preprocess_video(frames, 224, augment_seed)
        else:
            _require(not self.use_image_aug, "Resolution ablation inference must disable augmentation")
            video = preprocess_resolution_frames(frames, self.resolution_variant)

        ret_state_idx = 1 + self.h_steps // 4
        raw_residual = query_target - aligned_plan[: self.k_steps]
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
            "__key__": index,
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
            "episode_id": query.episode_id,
            "query_id": query.window_id,
            "candidate_id": candidate_id,
            "candidate_human_content_sha256": candidate_sha,
            "task": query.task,
            "split": query.split,
            "phase": np.float32(query.phase),
            "current_row": query.current_row,
            "method_id": self.method_id,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "target_representation": target_representation,
            "H_steps": self.h_steps,
            "K_steps": self.k_steps,
            "top_k": self.top_k,
            "pool_size": self.pool_size,
            "retrieval_modality": self.retrieval_modality,
            "retrieval_rank": example.retrieval_rank,
            "retrieval_distance": np.float32(example.distance),
            "retrieval_tie_sha256": example.tie_sha256,
            "sample_weight": np.float32(1.0 / example.effective_k),
            "strict_future_offset_view_steps": self.query_offset_view_steps,
            "gap_crossing_count": 0,
            "heldout_target_retrieval_feature_count": 0,
            "query_command_status": "unverified",
            "deployment_command_adapter_id": "",
            "protocol_id": PROTOCOL_ID,
            "protocol_file_sha256": self.protocol_file_sha256,
            "adapter_schema_version": P2_SCHEMA_VERSION,
            "diagnostic_overfit_mode": 0,
            "diagnostic_overfit_seed": self.seed,
            "raw_current_state": torch.from_numpy(current.astype(np.float32)),
            "raw_aligned_pool": torch.from_numpy(aligned_plan.astype(np.float32)),
            "raw_query_target": torch.from_numpy(query_target.astype(np.float32)),
            "raw_residual": torch.from_numpy(raw_residual.astype(np.float32)),
        }

    def contract_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": P2_SCHEMA_VERSION,
            "protocol_file_sha256": self.protocol_file_sha256,
            "supplement_file_sha256": self.supplement_file_sha256,
            "split_sha256": self.split_manifest["split_sha256"],
            "p1_selection_id": P1_SELECTION_ID,
            "split": self.split,
            "method_id": self.method_id,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "seed": self.seed,
            "H_steps": self.h_steps,
            "K_steps": self.k_steps,
            "top_k": self.top_k,
            "pool_size": self.pool_size,
            "retrieval_modality": self.retrieval_modality,
            "time_view_id": self.time_view_id,
            "time_view_manifest_path": str(self.time_view_path / "view_manifest.json"),
            "time_view_manifest_sha256": file_sha256(self.time_view_path / "view_manifest.json"),
            "query_offset_view_steps": self.query_offset_view_steps,
            "target_representation": self.target_representation,
            "query_count": len(self.queries),
            "candidate_count": len(self.candidates),
            "example_count": len(self.examples),
            "statistics_path": str(self.statistics_path or self.main_view_path / "action_statistics.json"),
            "statistics_file_sha256": self.statistics_file_sha256,
            "retrieval_index_path": str(self.retrieval_index_path) if self.retrieval_index_path else None,
            "retrieval_index_sha256": self.retrieval_index_sha256,
            "heldout_target_retrieval_feature_count": 0,
        }


def build_human2robot_p2_dataset(**kwargs: Any) -> Human2RobotP2Dataset:
    unknown = set(kwargs) - P2_DATASET_KWARGS - QUARANTINED_PUSHT_DATASET_KWARGS
    if unknown:
        raise Human2RobotContractError(f"Unknown P2 dataset kwargs: {sorted(unknown)}")
    selected = {key: value for key, value in kwargs.items() if key in P2_DATASET_KWARGS}
    missing = {"canonical_root", "main_view_path", "m3_report_path", "m4_report_path", "protocol_path", "supplement_path", "p1_pool_root"} - set(selected)
    if missing:
        raise Human2RobotContractError(f"Missing P2 dataset kwargs: {sorted(missing)}")
    return Human2RobotP2Dataset(**selected)
