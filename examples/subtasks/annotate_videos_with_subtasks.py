#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

import imageio


def _parse_ffprobe_rate(rate: str | None) -> float | None:
    if not rate:
        return None
    rate = str(rate).strip()
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0:
                return None
            return num_f / den_f
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def _ffprobe_video_meta(video_path: Path) -> tuple[float | None, int | None, float | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    stream = streams[0] if streams else {}
    fps = _parse_ffprobe_rate(stream.get("avg_frame_rate")) or _parse_ffprobe_rate(stream.get("r_frame_rate"))

    nframes = None
    nframes_raw = stream.get("nb_frames")
    if nframes_raw not in {None, "", "N/A"}:
        try:
            nframes = int(nframes_raw)
        except (TypeError, ValueError):
            nframes = None

    duration = None
    for raw in (stream.get("duration"), payload.get("format", {}).get("duration")):
        if raw in {None, "", "N/A"}:
            continue
        try:
            duration = float(raw)
            break
        except (TypeError, ValueError):
            continue
    return fps, nframes, duration


def _format_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis -= hours * 3_600_000
    minutes = millis // 60_000
    millis -= minutes * 60_000
    secs = millis // 1000
    millis -= secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _load_episode_payload(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_steps(payload: dict) -> list[tuple[int, str]]:
    subtasks = payload.get("subtasks", [])
    if not subtasks:
        return []
    if isinstance(subtasks[0], dict):
        return [(int(entry["step"]), str(entry["subtask"])) for entry in subtasks if "step" in entry]
    if isinstance(subtasks[0], str):
        frames = payload.get("frames", [])
        steps = []
        for idx, text in enumerate(subtasks):
            step = frames[idx]["step"] if idx < len(frames) and "step" in frames[idx] else idx
            steps.append((int(step), str(text)))
        return steps
    return []


def _get_video_meta(video_path: Path) -> tuple[float | None, int | None, float | None]:
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        reader.close()
        fps = meta.get("fps")
        nframes = meta.get("nframes")
        duration = meta.get("duration")
        return fps, nframes, duration
    except Exception:
        # Some imageio/pyav versions fail on metadata seeks for valid mp4 files.
        return _ffprobe_video_meta(video_path)


def _write_srt(steps: list[tuple[int, str]], fps: float, end_time_s: float, output_path: Path) -> None:
    lines = []
    for idx, (step, text) in enumerate(steps):
        start_s = step / fps
        if idx + 1 < len(steps):
            end_s = steps[idx + 1][0] / fps
        else:
            end_s = end_time_s
        lines.append(str(idx + 1))
        lines.append(f"{_format_srt_time(start_s)} --> {_format_srt_time(end_s)}")
        lines.append(text)
        lines.append("")
    output_path.write_text("\n".join(lines))


def _annotate_video(video_path: Path, srt_path: Path, output_path: Path, overwrite: bool) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if overwrite:
        cmd.append("-y")
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={srt_path}",
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True)


def _iter_task_dirs(subtasks_root: Path) -> list[Path]:
    if any(p.name.startswith("episode_") and p.suffix == ".json" for p in subtasks_root.iterdir()):
        return [subtasks_root]
    return [p for p in subtasks_root.iterdir() if p.is_dir()]


def _detect_videos_root(subtasks_root: Path) -> Path:
    parent = subtasks_root.parent
    candidates = sorted(parent.glob("videos_*"))
    if not candidates:
        raise FileNotFoundError(f"No videos_* directories found next to {subtasks_root}")
    if len(candidates) == 1:
        return candidates[0]

    task_dirs = _iter_task_dirs(subtasks_root)
    if task_dirs:
        task_name = task_dirs[0].name
        for candidate in candidates:
            if (candidate / task_name).exists():
                return candidate

    raise FileNotFoundError(
        "Multiple videos_* directories found; please pass --videos-root explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", type=Path, default=None)
    parser.add_argument("--subtasks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    videos_root = args.videos_root or _detect_videos_root(args.subtasks_root)

    for task_dir in _iter_task_dirs(args.subtasks_root):
        output_task_dir = args.output_root / task_dir.name
        output_task_dir.mkdir(parents=True, exist_ok=True)

        videos_dir = videos_root / task_dir.name
        if not videos_dir.exists():
            raise FileNotFoundError(f"Missing videos directory: {videos_dir}")

        for episode_path in sorted(task_dir.glob("episode_*.json")):
            payload = _load_episode_payload(episode_path)
            steps = _extract_steps(payload)
            if not steps:
                continue

            episode_index = payload.get("episode_index")
            if episode_index is None:
                episode_index = int(episode_path.stem.split("_")[-1])
            video_path = videos_dir / f"eval_episode_{episode_index}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video: {video_path}")

            fps, nframes, duration = _get_video_meta(video_path)
            use_fps = args.fps or fps
            if not use_fps:
                raise ValueError(f"Could not infer fps for {video_path}, pass --fps.")
            if duration is None and nframes is not None:
                duration = nframes / use_fps
            if duration is None:
                duration = (steps[-1][0] + 1) / use_fps

            srt_path = output_task_dir / f"{episode_path.stem}.srt"
            _write_srt(steps, use_fps, duration, srt_path)

            output_path = output_task_dir / f"{video_path.stem}_subtasks.mp4"
            _annotate_video(video_path, srt_path, output_path, args.overwrite)


if __name__ == "__main__":
    main()
