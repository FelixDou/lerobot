#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def _move(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        raise FileExistsError(f"Destination exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] move {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))
    print(f"moved {src} -> {dst}")


def _move_dir_files(src_dir: Path, dst_dir: Path, dry_run: bool) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.iterdir()):
        if item.is_dir():
            continue
        _move(item, dst_dir / item.name, dry_run)


def _move_logs(repo_root: Path, run_dir: Path, dry_run: bool, include_nohup_out: bool) -> None:
    logs_dir = run_dir / "logs"
    for log_path in sorted(repo_root.glob("nohup*.log")):
        _move(log_path, logs_dir / log_path.name, dry_run)
    if include_nohup_out:
        nohup_out = repo_root / "nohup.out"
        _move(nohup_out, logs_dir / nohup_out.name, dry_run)


def _move_outputs(run_dir: Path, dry_run: bool) -> None:
    inputs_dir = run_dir / "inputs"
    videos_dir = run_dir / "videos"
    subtasks_dir = run_dir / "subtasks"

    # subtask inputs
    _move(run_dir / "subtask_inputs", inputs_dir / "subtask_inputs", dry_run)

    # videos_* folders
    for path in sorted(run_dir.glob("videos_*")):
        suffix = path.name[len("videos_") :]
        target = videos_dir / (suffix if suffix else path.name)
        _move(path, target, dry_run)

    # existing "videos" folder
    _move_dir_files(run_dir / "videos", videos_dir / "main", dry_run)

    # subtasks_* folders
    for path in sorted(run_dir.glob("subtasks_*")):
        suffix = path.name[len("subtasks_") :]
        target = subtasks_dir / (suffix if suffix else path.name)
        _move(path, target, dry_run)

    # existing "subtasks" folder
    _move_dir_files(run_dir / "subtasks", subtasks_dir / "main", dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-nohup-out", action="store_true")
    parser.add_argument("--skip-logs", action="store_true")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    _move_outputs(args.run_dir, args.dry_run)
    if not args.skip_logs:
        _move_logs(args.repo_root, args.run_dir, args.dry_run, args.include_nohup_out)


if __name__ == "__main__":
    main()
