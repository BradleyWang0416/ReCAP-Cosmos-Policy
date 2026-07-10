#!/usr/bin/env python3
"""Export a side-by-side Human2Robot human/robot camera clip."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_EPISODE = Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1/roll/episode_0.hdf5")
DEFAULT_OUTPUT_DIR = Path("data/Human2Robot")


class ExportError(RuntimeError):
    """Raised when the paired clip cannot be exported."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        type=Path,
        default=DEFAULT_EPISODE,
        help=f"Human2Robot HDF5 episode. Default: {DEFAULT_EPISODE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for MP4 and contact sheet outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--start", type=int, help="Start frame. If omitted, choose by motion.")
    parser.add_argument("--frames", type=int, default=120, help="Clip length in frames.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--basename",
        default="human2robot_roll_episode0_pair_clip",
        help="Output filename stem.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, 14)
    return ImageFont.load_default()


def normalize_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.uint16:
        max_value = int(array.max()) if array.size else 0
        if max_value <= 255:
            return array.astype(np.uint8)
        return np.clip(array / 257.0, 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def frame_motion(camera: h5py.Dataset, frame_index: int) -> float:
    prev_frame = normalize_rgb(camera[frame_index - 1])[::8, ::8].astype(np.float32)
    curr_frame = normalize_rgb(camera[frame_index])[::8, ::8].astype(np.float32)
    return float(np.mean(np.abs(curr_frame - prev_frame)))


def select_start(human_camera: h5py.Dataset, robot_camera: h5py.Dataset, frames: int) -> int:
    total_frames = human_camera.shape[0]
    if total_frames <= frames:
        return 0
    stride = 5
    energies: list[tuple[float, int]] = []
    for start in range(1, total_frames - frames + 1, stride):
        stop = start + frames
        score = 0.0
        for frame_index in range(start + stride, stop, stride):
            score += frame_motion(human_camera, frame_index)
            score += frame_motion(robot_camera, frame_index)
        energies.append((score, start))
    if not energies:
        return 0
    return max(energies)[1]


def draw_label(image: Image.Image, text: str, xy: tuple[int, int], font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(image)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def compose_frame(
    human_frame: np.ndarray,
    robot_frame: np.ndarray,
    *,
    frame_index: int,
    step: int,
    timestamp: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    human = normalize_rgb(human_frame)
    robot = normalize_rgb(robot_frame)
    if human.shape != robot.shape:
        raise ExportError(f"Human and robot camera shapes differ: {human.shape} vs {robot.shape}")
    height, width, _ = human.shape
    gap = 8
    canvas = np.zeros((height, width * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :width] = human
    canvas[:, width + gap :] = robot
    image = Image.fromarray(canvas)
    draw_label(image, "human_camera", (8, 8), font)
    draw_label(image, "robot_camera", (width + gap + 8, 8), font)
    draw_label(image, f"frame {frame_index} | step {step} | ts {timestamp}", (8, height - 24), font)
    return np.asarray(image)


def write_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ExportError("ffmpeg was not found")
    if not frames:
        raise ExportError("No frames to write")
    height, width, _ = frames[0].shape
    encoder_listing = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if " libx264 " in encoder_listing:
        encoder_args = ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    else:
        encoder_args = ["-vcodec", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "3"]

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        *encoder_args,
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
    except BrokenPipeError:
        pass
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    returncode = process.wait()
    if returncode != 0:
        raise ExportError(f"ffmpeg failed with code {returncode}: {stderr}")


def write_contact_sheet(frames: list[np.ndarray], output_path: Path) -> None:
    if not frames:
        raise ExportError("No frames for contact sheet")
    sample_count = min(6, len(frames))
    indices = np.linspace(0, len(frames) - 1, sample_count).round().astype(int)
    sampled = [Image.fromarray(frames[int(index)]) for index in indices]
    width, height = sampled[0].size
    sheet = Image.new("RGB", (width, height * sample_count), color=(0, 0, 0))
    for row, image in enumerate(sampled):
        sheet.paste(image, (0, row * height))
    sheet.save(output_path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    episode = args.episode.resolve()
    if not episode.is_file():
        raise ExportError(f"Episode not found: {episode}")
    if args.frames <= 0:
        raise ExportError("--frames must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"{args.basename}.mp4"
    sheet_path = args.output_dir / f"{args.basename}_contact_sheet.jpg"
    metadata_path = args.output_dir / f"{args.basename}.json"
    font = load_font()

    with h5py.File(episode, "r") as file:
        human_camera = file["cam_data/human_camera"]
        robot_camera = file["cam_data/robot_camera"]
        if human_camera.shape != robot_camera.shape:
            raise ExportError(f"Camera shape mismatch: {human_camera.shape} vs {robot_camera.shape}")
        camera_shape = list(human_camera.shape)
        total_frames = int(human_camera.shape[0])
        clip_frames = min(args.frames, total_frames)
        start = args.start if args.start is not None else select_start(human_camera, robot_camera, clip_frames)
        start = max(0, min(start, total_frames - clip_frames))
        stop = start + clip_frames
        timestamps = np.asarray(file["timestamp"][start:stop])
        steps = np.asarray(file["step"][start:stop])
        if not bool(np.all(np.diff(timestamps.astype(np.int64)) >= 0)):
            raise ExportError("Selected timestamp window is not monotonic non-decreasing")

        composed = [
            compose_frame(
                human_camera[index],
                robot_camera[index],
                frame_index=index,
                step=int(steps[offset]),
                timestamp=int(timestamps[offset]),
                font=font,
            )
            for offset, index in enumerate(range(start, stop))
        ]

    write_mp4(composed, video_path, args.fps)
    write_contact_sheet(composed, sheet_path)

    metadata = {
        "status": "exported",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode": str(episode),
        "video": str(video_path),
        "contact_sheet": str(sheet_path),
        "frame_shape": list(composed[0].shape),
        "fps": args.fps,
        "start_frame": start,
        "end_frame_exclusive": stop,
        "frame_count": clip_frames,
        "camera_shape": camera_shape,
        "timestamp_first": to_jsonable(timestamps[0]),
        "timestamp_last": to_jsonable(timestamps[-1]),
        "step_first": to_jsonable(steps[0]),
        "step_last": to_jsonable(steps[-1]),
        "pairing_checks": {
            "same_hdf5_episode": True,
            "same_frame_indices": True,
            "same_camera_frame_shape": True,
            "timestamp_window_monotonic": True,
            "step_window_monotonic": bool(np.all(np.diff(steps.astype(np.int64)) >= 0)),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=to_jsonable) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    try:
        metadata = export(args)
    except ExportError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=to_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
