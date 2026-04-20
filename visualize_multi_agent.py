"""Visualization for Multi-Agent VSR-Env demo.

Generates PNG charts for hackathon pitch:
- Reward curves by role
- Spread evolution
- Manipulation detection timeline
"""

import json
import argparse
from pathlib import Path

def plot_reward_curves(replay_path: str, output_dir: str = "media"):
    """Plot reward curves by role from replay JSON."""
    import matplotlib.pyplot as plt

    with open(replay_path) as f:
        replay = json.load(f)

    steps = replay["steps"]
    n = len(steps)

    # Collect rewards by role
    trader_rewards = {f"trader_{i}": [] for i in range(10)}
    mm_rewards = []
    oversight_rewards = []

    for step_data in steps:
        rewards = step_data.get("rewards", {})
        for i in range(10):
            trader_rewards[f"trader_{i}"].append(rewards.get(f"trader_{i}", 0))
        mm_rewards.append(rewards.get("market_maker", 0))
        oversight_rewards.append(rewards.get("oversight", 0))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Traders (average)
    avg_trader = [sum(trader_rewards[f"trader_{i}"][j] for i in range(10)) / 10 for j in range(n)]
    axes[0].plot(avg_trader, color="blue", alpha=0.8)
    axes[0].fill_between(range(n),
        [min(trader_rewards[f"trader_{i}"][j] for i in range(10)) for j in range(n)],
        [max(trader_rewards[f"trader_{i}"][j] for i in range(10)) for j in range(n)],
        alpha=0.2, color="blue")
    axes[0].set_ylabel("Trader Reward")
    axes[0].set_title("Trader Rewards (avg ± range)")

    # Market Maker
    axes[1].plot(mm_rewards, color="green")
    axes[1].set_ylabel("MM Reward")
    axes[1].set_title("Market Maker Reward")

    # Oversight
    axes[2].plot(oversight_rewards, color="red")
    axes[2].set_ylabel("Oversight Reward")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Oversight Agent Reward")

    plt.tight_layout()
    Path(output_dir).mkdir(exist_ok=True)
    out_path = Path(output_dir) / "reward_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_spread_evolution(replay_path: str, output_dir: str = "media"):
    """Plot market maker spreads over time."""
    import matplotlib.pyplot as plt

    with open(replay_path) as f:
        replay = json.load(f)

    steps = replay["steps"]
    atm, otm, itm = [], [], []

    for step_data in steps:
        spreads = step_data.get("info", {}).get("market_maker_spreads", {})
        atm.append(spreads.get("atm", 0.02))
        otm.append(spreads.get("otm", 0.04))
        itm.append(spreads.get("itm", 0.03))

    plt.figure(figsize=(12, 4))
    plt.plot(atm, label="ATM", linewidth=2)
    plt.plot(otm, label="OTM", linewidth=2)
    plt.plot(itm, label="ITM", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Spread")
    plt.title("Market Maker Spread Evolution")
    plt.legend()
    plt.tight_layout()

    Path(output_dir).mkdir(exist_ok=True)
    out_path = Path(output_dir) / "spread_evolution.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_manipulation_timeline(replay_path: str, output_dir: str = "media"):
    """Plot manipulation detection events."""
    import matplotlib.pyplot as plt

    with open(replay_path) as f:
        replay = json.load(f)

    steps = replay["steps"]
    events = {"wash_trading": [], "spoofing_like_pressure": [], "gamma_pressure": [], "systemic_risk": []}

    for step_data in steps:
        detected = step_data.get("info", {}).get("detected_manipulations", {})
        step_num = step_data.get("step", 0)
        for agent_id, manipulation_type in detected.items():
            if manipulation_type != "none" and manipulation_type in events:
                events[manipulation_type].append(step_num)

    plt.figure(figsize=(12, 4))
    colors = {"wash_trading": "red", "spoofing_like_pressure": "orange", "gamma_pressure": "purple", "systemic_risk": "gray"}
    for mtype, steps_list in events.items():
        if steps_list:
            plt.scatter(steps_list, [mtype] * len(steps_list), c=colors[mtype], s=50, label=mtype, alpha=0.7)

    plt.xlabel("Step")
    plt.yticks(list(colors.keys()))
    plt.title("Manipulation Detection Timeline")
    plt.legend()
    plt.tight_layout()

    Path(output_dir).mkdir(exist_ok=True)
    out_path = Path(output_dir) / "manipulation_timeline.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-agent replay")
    parser.add_argument("--replay", type=str, required=True, help="Path to replay JSON")
    parser.add_argument("--output", type=str, default="media", help="Output directory")
    args = parser.parse_args()

    plot_reward_curves(args.replay, args.output)
    plot_spread_evolution(args.replay, args.output)
    plot_manipulation_timeline(args.replay, args.output)


if __name__ == "__main__":
    main()
