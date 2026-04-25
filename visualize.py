import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_episode(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_dashboard(episode: dict) -> None:
    """Plot a real dashboard from replay JSON."""
    steps = episode.get("steps", [])
    if not steps:
        raise ValueError("Episode JSON must contain a non-empty 'steps' list")

    x = list(range(1, len(steps) + 1))
    mm_rewards = [step.get("rewards", {}).get("market_maker", 0.0) for step in steps]
    oversight_rewards = [step.get("rewards", {}).get("oversight", 0.0) for step in steps]
    trader_mean_rewards = []
    spread_widths = []
    trade_counts = []

    for step in steps:
        rewards = step.get("rewards", {})
        trader_rewards = [value for key, value in rewards.items() if key.startswith("trader_")]
        trader_mean_rewards.append(sum(trader_rewards) / len(trader_rewards) if trader_rewards else 0.0)
        spread = step.get("info", {}).get("market_maker_spreads", {})
        spread_widths.append(spread.get("atm", 0.0))
        trade_counts.append(step.get("info", {}).get("trade_count", 0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Multi-Agent VSR-Env Replay Dashboard")

    axes[0, 0].plot(x, mm_rewards, label="MM Reward")
    axes[0, 0].plot(x, trader_mean_rewards, label="Mean Trader Reward")
    axes[0, 0].plot(x, oversight_rewards, label="Oversight Reward")
    axes[0, 0].set_title("Role Rewards")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].legend()

    axes[0, 1].plot(x, spread_widths, color="tab:orange")
    axes[0, 1].set_title("ATM Spread")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Spread")

    axes[1, 0].bar(x, trade_counts, color="tab:green")
    axes[1, 0].set_title("Trade Count")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Executed Trades")

    manipulation_events = [
        len([m for m in step.get("info", {}).get("detected_manipulations", {}).values() if m != "none"])
        for step in steps
    ]
    axes[1, 1].plot(x, manipulation_events, color="tab:red")
    axes[1, 1].set_title("Manipulation Events")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Flaggable Events")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Multi-Agent VSR replay JSON file.")
    parser.add_argument("replay", type=str, help="Path to replay JSON")
    args = parser.parse_args()

    replay_path = Path(args.replay)
    episode = load_episode(str(replay_path))
    plot_dashboard(episode)


if __name__ == "__main__":
    main()
