#!/usr/bin/env python3
"""Generate README figures from training/eval logs.

Usage:
  python3 scripts/generate_story_graphs.py \
      --training_log vsr-new.log \
      --out_dir media
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


DEFAULT_COMPARISON = {
    "Aggressive Traders (0-2)": {"trained": -0.93, "baseline": -4.13},
    "Neutral Traders (3-5)": {"trained": -1.08, "baseline": -4.58},
    "Contrarian Traders (6-8)": {"trained": -8.52, "baseline": -3.79},
    "Market Maker": {"trained": 21.01, "baseline": 14.84},
    "Oversight SEC": {"trained": -95.60, "baseline": 7.50},
}


def moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        window_vals = values[left : i + 1]
        out.append(sum(window_vals) / len(window_vals))
    return out


def parse_training_log(path: Path) -> Tuple[List[Dict[str, float]], int]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Supports logs with progress bars like 5/250, 10/500, etc.
    # Keep this line-local to avoid matching unrelated download bars (e.g. 454/454).
    pattern = re.compile(r"\|\s*(\d+)/(\d+)[^\n\{]*(\{[^{}]+\})")
    points: List[Dict[str, float]] = []
    max_total_steps = 0

    for step_txt, total_txt, dict_txt in pattern.findall(raw):
        try:
            max_total_steps = max(max_total_steps, int(total_txt))
            payload = ast.literal_eval(dict_txt)
            points.append(
                {
                    "step": float(step_txt),
                    "reward": float(payload["reward"]),
                    "reward_std": float(payload["reward_std"]),
                    "loss": float(payload["loss"]),
                    "kl": float(payload["kl"]),
                    "lr": float(payload["learning_rate"]),
                }
            )
        except Exception:
            continue

    dedup: Dict[int, Dict[str, float]] = {}
    for point in points:
        dedup[int(point["step"])] = point
    return [dedup[k] for k in sorted(dedup)], max_total_steps


def save_training_reward_plot(points: List[Dict[str, float]], out_path: Path, total_steps: int) -> None:
    steps = [p["step"] for p in points]
    rewards = [p["reward"] for p in points]
    stds = [p["reward_std"] for p in points]
    reward_ma = moving_average(rewards, window=7)

    plt.figure(figsize=(11, 6))
    plt.plot(steps, rewards, alpha=0.25, linewidth=1.5, label="Raw reward")
    plt.plot(steps, reward_ma, linewidth=2.5, label="7-step moving average")
    upper = [m + s for m, s in zip(reward_ma, stds)]
    lower = [m - s for m, s in zip(reward_ma, stds)]
    plt.fill_between(steps, lower, upper, alpha=0.12, label="± reward std")
    plt.axhline(0, linestyle="--", linewidth=1)

    total_suffix = f", run length={total_steps}" if total_steps > 0 else ""
    plt.title(f"Training Reward Curve (from logs{total_suffix})")
    plt.xlabel("Training step checkpoint")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_training_diagnostics_plot(points: List[Dict[str, float]], out_path: Path, total_steps: int) -> None:
    steps = [p["step"] for p in points]
    loss = [p["loss"] for p in points]
    kl = [p["kl"] for p in points]
    lr = [p["lr"] for p in points]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(steps, moving_average(loss, 5), linewidth=2)
    axes[0].set_ylabel("Loss (MA)")
    total_suffix = f" (run length={total_steps})" if total_steps > 0 else ""
    axes[0].set_title(f"Training Diagnostics{total_suffix}")

    axes[1].plot(steps, moving_average(kl, 5), linewidth=2)
    axes[1].set_ylabel("KL (MA)")

    axes[2].plot(steps, lr, linewidth=2)
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Training step checkpoint")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_eval_comparison(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return DEFAULT_COMPARISON


def save_eval_comparison_plot(comparison: Dict[str, Dict[str, float]], out_path: Path) -> None:
    labels = list(comparison.keys())
    trained = [comparison[k]["trained"] for k in labels]
    baseline = [comparison[k]["baseline"] for k in labels]
    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], trained, width=width, label="Trained LoRA")
    plt.bar([i + width / 2 for i in x], baseline, width=width, label="Scripted baseline")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Cumulative reward")
    plt.title("Model vs Baseline Reward Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_log", default="vsr-new.log")
    parser.add_argument("--comparison_json", default="")
    parser.add_argument("--out_dir", default="media")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_points, total_steps = parse_training_log(Path(args.training_log))
    if not training_points:
        raise SystemExit("No training checkpoints parsed from log. Verify --training_log path.")

    save_training_reward_plot(training_points, out_dir / "training_reward_from_logs.png", total_steps)
    save_training_diagnostics_plot(training_points, out_dir / "training_diagnostics_from_logs.png", total_steps)
    comparison = load_eval_comparison(Path(args.comparison_json) if args.comparison_json else None)
    save_eval_comparison_plot(comparison, out_dir / "model_vs_baseline_rewards.png")

    print(f"Parsed checkpoints: {len(training_points)}")
    if total_steps > 0:
        print(f"Detected run length from logs: {total_steps}")
    print(f"Wrote: {out_dir / 'training_reward_from_logs.png'}")
    print(f"Wrote: {out_dir / 'training_diagnostics_from_logs.png'}")
    print(f"Wrote: {out_dir / 'model_vs_baseline_rewards.png'}")


if __name__ == "__main__":
    main()
