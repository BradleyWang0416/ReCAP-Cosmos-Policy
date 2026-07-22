from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cosmos_policy.models.human2robot_adapter import validate_human2robot_batch
from tools import human2robot_v04_stage4 as stage4
from tools import human2robot_v04_stage4_worker as worker


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _projection(
    root: Path,
    *,
    task: str,
    role: str,
    partition: str,
    rank: int,
    frames: int = 32,
) -> dict:
    path = root / partition / task / f"{rank:02d}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    source_sha = _sha(f"{partition}:{task}:{rank}")
    episode_id = f"{task}/episode_{rank}"
    starts = np.arange(frames - stage4.H_STEPS - stage4.K_STEPS + 1, dtype=np.int64)
    time = np.arange(frames, dtype=np.float32)
    pose = np.stack(
        (
            0.01 * time + rank * 1e-3,
            0.02 * np.sin(time / 4 + rank),
            0.03 * np.cos(time / 5 + rank),
            0.02 * np.sin(time / 7),
            0.02 * np.cos(time / 8),
            0.01 * time / frames,
        ),
        axis=1,
    ).astype(np.float32)
    gripper = ((np.arange(frames) + rank) % 2).astype(np.float32)
    images = np.full((frames, 240, 426, 3), (rank * 17) % 255, dtype=np.uint8)
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["source_sha256"] = source_sha
        demo.attrs["source_relative_path"] = f"{task}/episode_{rank}.hdf5"
        demo.attrs["source_partition"] = partition
        demo.attrs["task"] = task
        demo.attrs["episode_id"] = episode_id
        demo.attrs["role"] = role
        demo.attrs["frame_count"] = frames
        group = demo.create_group(role)
        if role == "human":
            action = np.concatenate((pose, gripper[:, None]), axis=1)
            group.create_dataset("hand_action_7d", data=action)
            group.create_dataset("hand_coords", data=np.zeros((frames, 24, 3), dtype=np.float32))
            group.create_dataset("hand_frames", data=np.zeros((frames, 4, 3), dtype=np.float32))
            group.create_dataset("images", data=images)
            content_key = "human_content_sha256"
        else:
            group.create_dataset("observed_eef_pose_6d", data=pose)
            group.create_dataset("gripper_state", data=gripper)
            group.create_dataset("images", data=images)
            content_key = "robot_content_sha256"
        timing = demo.create_group("time")
        timing.create_dataset("gap_mask", data=np.zeros(frames, dtype=bool))
        timing.create_dataset("legal_window_start", data=starts)
        timing.create_dataset("segment_id", data=np.zeros(frames, dtype=np.int32))
        timing.create_dataset("source_step", data=np.arange(frames))
        timing.create_dataset("source_timestamp", data=np.arange(frames))
    return {
        "episode_id": episode_id,
        "frame_count": frames,
        "gap_count": 0,
        "legal_window_count": len(starts),
        "max_gap_safe_segment_frames": frames,
        "partition_rank": rank,
        content_key: _sha(f"content:{partition}:{task}:{rank}"),
        "projection": {
            "path": str(path),
            "sha256": stage4.file_sha256(path),
            "size_bytes": path.stat().st_size,
        },
        "role": role,
        "segment_count": 1,
        "source_partition": partition,
        "source_relative_path": f"{task}/episode_{rank}.hdf5",
        "source_sha256": source_sha,
        "source_sort_sha256": _sha(f"sort:{partition}:{task}:{rank}"),
        "task": task,
    }


def _fake_encoder(frames: np.ndarray) -> np.ndarray:
    mean = frames.astype(np.float32).mean(axis=(1, 2, 3))
    base = np.arange(1, 17, dtype=np.float32)[None, :]
    values = base + mean[:, None] / 255.0
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _grouped(tmp_path: Path) -> dict[str, list[dict]]:
    tasks = ["task_a", "task_b", "task_c", "task_d"]
    grouped = {name: [] for name in stage4.FEATURE_PARTITIONS}
    for task in tasks:
        grouped["v04_human_pool"].extend(
            _projection(tmp_path, task=task, role="human", partition="v04_human_pool", rank=rank)
            for rank in range(1, 11)
        )
        grouped["v04_robot_dev"].extend(
            _projection(tmp_path, task=task, role="robot", partition="v04_robot_dev", rank=rank)
            for rank in range(1, 6)
        )
    return grouped


def test_evenly_spaced_smoke_windows_are_fixed_and_unique() -> None:
    assert stage4.select_evenly_spaced_starts(np.arange(17)) == [0, 2, 5, 7, 9, 11, 14, 16]
    with pytest.raises(stage4.Stage4Error, match="Fewer than 8"):
        stage4.select_evenly_spaced_starts(np.arange(7))


def test_population_geometry_statistics_match_reference(tmp_path: Path) -> None:
    path = tmp_path / "seen.hdf5"
    frames = 24
    with h5py.File(path, "w") as file:
        file.create_dataset("action", data=np.column_stack((np.arange(frames)[:, None] * np.ones((1, 6)), np.arange(frames) % 2)))
        file.create_dataset("end_position", data=np.arange(frames)[:, None] * np.arange(1, 7)[None, :])
        file.create_dataset("gripper_state", data=np.arange(frames) % 2)
        file.create_dataset("step", data=np.arange(frames))
        file.create_dataset("timestamp", data=np.arange(frames))
    record = {
        "role": "paired",
        "source_relative_path": path.name,
        "source_partition": "seen_train",
        "legal_window_count": 9,
    }
    output = tmp_path / "geometry.json"
    result = stage4.fit_seen_train_geometry([record], source_root=tmp_path, output_path=output, split_sha256="s")
    assert result["relative_row_count"] == 9 * 8 * 2
    assert np.isfinite(result["mean_10d"]).all()
    assert np.asarray(result["std_10d"]).min() > 0
    assert result["future_rows_read"] == 0


def test_visual_cache_is_current_only_finite_and_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _projection(tmp_path / "data", task="task_a", role="human", partition="v04_human_pool", rank=1)
    grouped = {name: [] for name in stage4.FEATURE_PARTITIONS}
    grouped["v04_human_pool"] = [record]
    fake_tokenizer = tmp_path / "tokenizer.pth"
    fake_tokenizer.write_bytes(b"frozen")
    monkeypatch.setattr(stage4, "TOKENIZER_PATH", fake_tokenizer)
    result = stage4.materialize_visual_cache(
        grouped,
        source_root=tmp_path,
        feature_root=tmp_path / "features",
        split_sha256="split",
        batch_size=4,
        encoder=_fake_encoder,
    )
    assert result["feature_count"] == record["legal_window_count"]
    assert result["future_frames_read"] == 0
    assert result["target_datasets_read"] == 0
    assert stage4.materialize_visual_cache(
        grouped,
        source_root=tmp_path,
        feature_root=tmp_path / "features",
        split_sha256="split",
        batch_size=4,
        encoder=_fake_encoder,
    )["shard_bundle_sha256"] == result["shard_bundle_sha256"]
    assert not list(tmp_path.rglob("*.partial*"))


def test_synthetic_identity_retrieval_and_evaluation_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grouped = _grouped(tmp_path / "projections")
    fake_tokenizer = tmp_path / "tokenizer.pth"
    fake_tokenizer.write_bytes(b"frozen")
    monkeypatch.setattr(stage4, "TOKENIZER_PATH", fake_tokenizer)
    feature_root = tmp_path / "features"
    visual = stage4.materialize_visual_cache(
        grouped,
        source_root=tmp_path,
        feature_root=feature_root,
        split_sha256="split",
        batch_size=8,
        encoder=_fake_encoder,
    )
    assert visual["feature_count"] == 60 * 17
    geometry = {"mean_10d": [0.0] * 10, "std_10d": [1.0] * 10}
    plan = stage4.build_smoke_plan(
        grouped,
        feature_root=feature_root,
        geometry=geometry,
        output_path=tmp_path / "smoke.json",
        split_sha256="split",
    )
    assert plan["query_count"] == 160
    assert plan["rank_inference_count_per_method"] == 480
    assert all(len(query["ranks"]) == 3 for query in plan["queries"])
    for query in plan["queries"]:
        for rank in query["ranks"]:
            retrieval = rank["retrieval"]
            assert retrieval["query_source_sha256"] != retrieval["candidate_source_sha256"]
            assert retrieval["query_partition"] == "v04_robot_dev"
            assert retrieval["candidate_partition"] == "v04_human_pool"
            assert retrieval["query_feature_provenance"]["future_rows_read"] == []
            assert retrieval["candidate_feature_provenance"]["target_datasets_read"] == []
    statistics = {
        "residual_10d_min": [-2.0] * 10,
        "residual_10d_max": [2.0] * 10,
        "query_bc_target_10d_min": [-2.0] * 10,
        "query_bc_target_10d_max": [2.0] * 10,
        "pool_action_10d_min": [-2.0] * 10,
        "pool_action_10d_max": [2.0] * 10,
    }
    first = plan["queries"][0]
    item = worker.build_model_item(
        first,
        first["ranks"][0],
        "recap_hand_ret",
        statistics,
        protocol_file_sha256="a" * 64,
    )
    assert tuple(item["actions"].shape) == (8, 10)
    assert tuple(item["video"].shape) == (3, 37, 224, 224)
    assert np.isfinite(item["actions"].numpy()).all()
    assert all(validate_human2robot_batch(item).values())


def test_protocol_lock_keeps_smoke_nonperformance_and_opens_stage5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = {}
    for name in ("seen_train", "seen_validation", "v04_human_pool", "v04_robot_dev", "v04_robot_final"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        inputs[name] = stage4.bind_file(path)
    geometry = tmp_path / "geometry.json"
    visual = tmp_path / "visual.json"
    plan = tmp_path / "plan.json"
    summary = tmp_path / "summary.json"
    for path in (geometry, visual, plan, summary):
        path.write_text("{}", encoding="utf-8")
    manifest = {"protocol_sha256": "p", "split_sha256": "s"}
    result = stage4.build_protocol_lock(
        workspace=tmp_path,
        derived_root=tmp_path,
        feature_root=tmp_path,
        manifest=manifest,
        partition_bindings=inputs,
        geometry_binding=stage4.bind_file(geometry),
        visual_index_binding=stage4.bind_file(visual),
        smoke_plan_binding=stage4.bind_file(plan),
        smoke_summaries=[stage4.bind_file(summary)],
    )
    assert result["status"] == "VERIFIED_STAGE4"
    assert result["training_allowed"] is True
    assert result["stage5_allowed"] is True
    assert result["formal_performance_result"] is False
    assert result["performance_claim_allowed"] is False
