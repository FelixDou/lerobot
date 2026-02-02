#!/usr/bin/env python3
"""
Evaluate pi05/pi0 checkpoints on LIBERO-10 and plot success vs training steps.

This script now runs offline evaluations (using ``lerobot.scripts.lerobot_eval``)
for the base model and every saved checkpoint, then plots the resulting accuracy.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

# Try to import matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")

from lerobot.utils.train_utils import get_step_identifier

DEFAULT_POLICY_TYPE = "pi05"


def default_base_model(policy_type: str) -> str:
    return f"lerobot/{policy_type}_libero"


def default_run_glob(policy_type: str) -> str:
    return f"outputs/{policy_type}_libero_ft_my_run_*"


def create_graph(steps, success_rates, output_file, policy_type: str, target_line: float | None = None):
    """Create and save the accuracy vs steps graph."""
    if not HAS_MATPLOTLIB:
        print("Cannot create graph without matplotlib.")
        return False
    
    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        success_rates,
        marker="o",
        linewidth=2.5,
        markersize=10,
        color="#2E86AB",
        markerfacecolor="#A23B72",
        markeredgewidth=2,
    )

    plt.xlabel("Training Steps", fontsize=13, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=13, fontweight="bold")
    plt.title(
        f"Model Accuracy (Success Rate) vs Training Steps\n{policy_type.upper()} Fine-tuning on LIBERO-10",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=1)
    plt.ylim(0, 105)
    if steps:
        x_max = max(steps) + 500
        plt.xlim(-200 if 0 in steps else 0, x_max)
    
    # Target line (accept percent or fraction)
    if target_line is not None:
        target_val = target_line * 100 if target_line <= 1 else target_line
        plt.axhline(target_val, color="gray", linestyle="--", linewidth=1.5, label=f"Target {target_val:.1f}%")
        plt.legend()
    
    for step, success in zip(steps, success_rates):
        plt.annotate(
            f"{success:.0f}%",
            (step, success),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Graph saved to: {output_file}")
    return True


def print_data_table(steps, success_rates):
    """Print a simple text table of the data."""
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"{'Step':<10} {'Success Rate (%)':<20}")
    print("-" * 50)
    for step, success in zip(steps, success_rates):
        print(f"{step:<10} {success:<20.1f}")
    print("=" * 50)


def find_latest_run_dir(policy_type: str) -> Path | None:
    """Return the most recent training run directory matching the default glob."""
    run_glob = default_run_glob(policy_type)
    candidates = sorted(Path(".").glob(run_glob), key=lambda p: p.stat().st_ctime)
    return candidates[-1] if candidates else None


def checkpoint_paths(run_dir: Path, max_step: int, step_size: int) -> Iterable[Tuple[int, Path]]:
    """Yield (step, checkpoint_path) pairs for available checkpoints."""
    total_steps = max_step
    for step in range(step_size, max_step + 1, step_size):
        step_id = get_step_identifier(step, total_steps)
        ckpt_dir = run_dir / "checkpoints" / step_id / "pretrained_model"
        if ckpt_dir.exists():
            yield step, ckpt_dir
        else:
            print(f"Skipping missing checkpoint: {ckpt_dir}")


def _find_eval_info_file(output_dir: Path) -> Path | None:
    """Locate the eval_info*.json file written by lerobot_eval."""
    candidates = list(output_dir.glob("eval_info*.json"))
    if not candidates:
        return None
    # Prefer more specific names first (sorted reverse length then alphabetically)
    candidates.sort(key=lambda p: (-len(p.name), p.name))
    return candidates[0]


def _read_success_from_eval(output_dir: Path) -> float | None:
    info_path = _find_eval_info_file(output_dir)
    if not info_path:
        return None
    with open(info_path, "r") as f:
        data = json.load(f)
    success = data.get("overall", {}).get("pc_success")
    if success is None:
        for _, group in data.items():
            if isinstance(group, dict) and "pc_success" in group:
                success = group["pc_success"]
                break
    return float(success) if success is not None else None


def run_eval_for_policy(
    policy_path: Path | str,
    output_dir: Path,
    *,
    env_type: str,
    env_task: str,
    n_episodes: int,
    batch_size: int,
    device: str,
    max_parallel_tasks: int,
    reuse_existing: bool,
    evaluate_if_missing: bool = True,
) -> float:
    """Run lerobot_eval for a single policy and return pc_success."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing:
        existing = _read_success_from_eval(output_dir)
        if existing is not None:
            print(f"Reusing existing eval in {output_dir}: pc_success={existing:.1f}")
            return existing
        if not evaluate_if_missing:
            print(f"Missing precomputed eval in {output_dir}; skipping (reuse mode).")
            return float("nan")

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path}",
        f"--env.type={env_type}",
        f"--env.task={env_task}",
        f"--env.max_parallel_tasks={max_parallel_tasks}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        f"--policy.device={device}",
        f"--output_dir={output_dir}",
    ]

    print(f"Evaluating {policy_path} -> {output_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Evaluation failed:")
        print(result.stderr or result.stdout)
        return float("nan")

    success = _read_success_from_eval(output_dir)
    if success is None:
        print(f"No eval_info*.json found in {output_dir}")
        return float("nan")

    return float(success)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on LIBERO-10.")
    parser.add_argument("--policy-type", default=DEFAULT_POLICY_TYPE, help="pi05 or pi0.")
    parser.add_argument("--run-dir", type=Path, help="Training run directory with checkpoints.")
    parser.add_argument("--base-model", help="Base model repo or path. Defaults to <policy_type> libero.")
    parser.add_argument("--max-step", type=int, default=6000, help="Last training step to evaluate.")
    parser.add_argument("--step-size", type=int, default=1000, help="Checkpoint step interval.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="First checkpoint step to re-evaluate; earlier steps will only be reused if already computed.",
    )
    parser.add_argument("--env-task", default="libero_10", help="LIBERO task to evaluate.")
    parser.add_argument("--env-type", default="libero", help="Environment type.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of eval episodes.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval batch size.")
    parser.add_argument("--device", default="cuda", help="Device for evaluation.")
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=1,
        help="Max parallel tasks during eval (libero usually 1).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional output PNG path. Defaults to run_dir/accuracy_vs_steps.png.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="If set, reuse existing eval_info*.json instead of re-running eval.",
    )
    parser.add_argument(
        "--target-line",
        type=float,
        default=None,
        help="Optional target success line to draw (accepts 0-1 or percentage).",
    )
    args = parser.parse_args()

    policy_type = args.policy_type
    base_model = args.base_model or default_base_model(policy_type)

    run_dir = args.run_dir or find_latest_run_dir(policy_type)
    if run_dir is None:
        print(f"No training run directory found matching {default_run_glob(policy_type)}")
        return 1
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return 1
    
    output_file = args.output_file or (run_dir / "accuracy_vs_steps.png")

    steps: list[int] = []
    success_rates: list[float] = []

    # Always include base model (step 0); reuse if available
    steps.append(0)
    base_eval_dir = run_dir / "eval_offline" / "step_000000"
    base_success = run_eval_for_policy(
        base_model,
        base_eval_dir,
        env_type=args.env_type,
        env_task=args.env_task,
        n_episodes=args.eval_episodes,
        batch_size=args.eval_batch_size,
        device=args.device,
        max_parallel_tasks=args.max_parallel_tasks,
        reuse_existing=args.reuse_existing,
        evaluate_if_missing=True,
    )
    success_rates.append(base_success)

    # Evaluate checkpoints; for steps < start_step reuse-only, otherwise evaluate/reuse
    for step, ckpt_path in checkpoint_paths(run_dir, args.max_step, args.step_size):
        step_id = get_step_identifier(step, args.max_step)
        eval_dir = run_dir / "eval_offline" / f"step_{step_id}"
        reuse_only = args.reuse_existing and step < args.start_step
        success = run_eval_for_policy(
            ckpt_path,
            eval_dir,
            env_type=args.env_type,
            env_task=args.env_task,
            n_episodes=args.eval_episodes,
            batch_size=args.eval_batch_size,
            device=args.device,
            max_parallel_tasks=args.max_parallel_tasks,
            reuse_existing=args.reuse_existing or reuse_only,
            evaluate_if_missing=not reuse_only,
        )
        steps.append(step)
        success_rates.append(success)

    print_data_table(steps, success_rates)
    
    if create_graph(steps, success_rates, output_file, policy_type, target_line=args.target_line):
        return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())

