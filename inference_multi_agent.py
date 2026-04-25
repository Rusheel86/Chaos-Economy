"""Full 300-step multi-agent demo inference.

Runs a complete episode with:
- 10 trader agents (scripted or LLM)
- 1 market maker
- 1 oversight agent

Outputs replay JSON for visualization.
"""

import argparse
import json
from pathlib import Path

from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction
from multi_agent.config import NUM_TRADERS, EPISODE_LENGTH


def scripted_trader(agent_idx: int, step: int) -> dict:
    """Simple scripted trader policy."""
    strike = (agent_idx + step) % 8
    maturity = (agent_idx + step) % 3
    direction = "buy" if (agent_idx + step) % 3 == 0 else "sell" if (agent_idx + step) % 3 == 1 else "hold"
    quantity = 0.5 + (agent_idx % 3) * 0.5
    return {
        "selected_strike": strike,
        "selected_maturity": maturity,
        "direction": direction,
        "quantity": quantity if direction != "hold" else 0.0,
        "option_type": "call" if agent_idx % 2 == 0 else "put",
        "reasoning": f"Trader {agent_idx} step {step}",
    }


def scripted_market_maker(step: int) -> dict:
    """Adaptive market maker - widens spreads under stress."""
    base_atm = 0.02 + (step / EPISODE_LENGTH) * 0.03
    return MarketMakerAction(
        atm_spread=round(base_atm, 3),
        otm_spread=round(base_atm + 0.02, 3),
        itm_spread=round(base_atm + 0.01, 3),
        reasoning=f"MM spreads at step {step}"
    ).model_dump()


def scripted_oversight(step: int, detected_manipulations: dict) -> dict:
    """Oversight agent - flags detected manipulations."""
    flagged = [aid for aid, mtype in detected_manipulations.items() if mtype != "none"]
    flag_type = "none"
    fine_amount = 0.0

    if flagged:
        # Take first detected manipulation
        flag_type = detected_manipulations.get(flagged[0], "none")
        fine_amount = 0.5 if flag_type in ["wash_trading", "spoofing_like_pressure"] else 0.0

    return OversightAction(
        flagged_agents=flagged[:3],  # Max 3 flags
        flag_type=flag_type,
        fine_amount=fine_amount,
        intervention_type="fine" if fine_amount > 0 else "none",
        reasoning=f"Oversight step {step}"
    ).model_dump()


def run_episode(output_path: str | None = None, use_llm: bool = False):
    """Run full 300-step episode."""
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=42)

    replay = {
        "episode_length": EPISODE_LENGTH,
        "num_traders": NUM_TRADERS,
        "steps": [],
        "final_rewards": {},
        "total_trades": 0,
        "total_interventions": 0,
    }

    print(f"Running {EPISODE_LENGTH}-step multi-agent episode...")

    for step in range(EPISODE_LENGTH):
        actions = {}

        # Traders
        for i in range(NUM_TRADERS):
            actions[f"trader_{i}"] = scripted_trader(i, step)

        # Market maker
        actions["market_maker"] = scripted_market_maker(step)

        # Oversight (uses previous step's detections)
        prev_detections = replay["steps"][-1]["info"]["detected_manipulations"] if replay["steps"] else {}
        actions["oversight"] = scripted_oversight(step, prev_detections)

        obs, rewards, done, info = env.step(actions)

        replay["steps"].append({
            "step": step + 1,
            "rewards": rewards,
            "info": info,
        })

        if step % 50 == 0:
            avg_trader = sum(rewards.get(f"trader_{i}", 0) for i in range(NUM_TRADERS)) / NUM_TRADERS
            print(f"Step {step+1}/{EPISODE_LENGTH} | Avg Trader: {avg_trader:.3f} | MM: {rewards['market_maker']:.3f} | Oversight: {rewards['oversight']:.3f}")

        if done:
            break

    # Summary
    replay["total_trades"] = len(env.trade_log)
    replay["total_interventions"] = len(env.intervention_log)
    replay["final_rewards"] = {aid: sum(s["rewards"].get(aid, 0) for s in replay["steps"]) for aid in env.AGENT_IDS}

    print(f"\nEpisode Complete!")
    print(f"Total trades: {replay['total_trades']}")
    print(f"Total interventions: {replay['total_interventions']}")
    print(f"Final cumulative rewards:")
    for aid in ["market_maker", "oversight"] + [f"trader_{i}" for i in range(NUM_TRADERS)]:
        print(f"  {aid}: {replay['final_rewards'].get(aid, 0):.2f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(replay, f, indent=2)
        print(f"\nSaved replay: {output_path}")

    return replay


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent episode")
    parser.add_argument("--output", type=str, default="replays/demo_episode.json", help="Output replay JSON")
    parser.add_argument("--llm", action="store_true", help="Use LLM for agent policies (requires API)")
    args = parser.parse_args()

    run_episode(args.output, args.llm)


if __name__ == "__main__":
    main()
