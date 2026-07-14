from __future__ import annotations

from pathlib import Path

from tools.human2robot_m5b_p2 import source_manifest, source_paths
from tools.human2robot_m5b_p2_preflight import mount_binding


WORKSPACE = Path(__file__).resolve().parents[1]


def test_mount_binding_uses_longest_covering_mount_and_detects_read_only(tmp_path: Path) -> None:
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n"
        "2 1 8:1 / /DATA1 ro,relatime - ext4 /dev/sda1 rw\n",
        encoding="utf-8",
    )
    result = mount_binding(Path("/DATA1/wxs/ReCAP_M5B_P2_RUNS"), mountinfo)
    assert result["mount_point"] == "/DATA1"
    assert result["writable"] is False


def test_source_snapshot_candidate_includes_every_new_runtime_module() -> None:
    paths = source_paths(WORKSPACE)
    required = {
        Path("tools/human2robot_m5b_p2_activation.py"),
        Path("tools/human2robot_m5b_p2_dag.py"),
        Path("tools/human2robot_m5b_p2_matrix.py"),
        Path("tools/human2robot_m5b_p2_handlers.py"),
        Path("tools/human2robot_m5b_p2_evaluation.py"),
        Path("tools/human2robot_m5b_p2_inference.py"),
        Path("tools/human2robot_m5b_p2_preflight.py"),
        Path("tools/human2robot_m5b_p2_prepare.py"),
        Path("tools/human2robot_m5b_p2_reports.py"),
        Path("tools/human2robot_m5b_p2_successor.py"),
    }
    assert required <= set(paths)
    manifest = source_manifest(WORKSPACE, paths)
    assert len(manifest["code_sha256"]) == 64
    assert required <= {Path(item["path"]) for item in manifest["files"]}
