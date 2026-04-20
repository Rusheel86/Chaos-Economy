import argparse
import json
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction
from vsr_env.models import VSRAction, TradeDirection

def run_inference(output_path: str | None = None):
    env = MultiAgentVSREnvironment()
    obs = env.reset()
    replay = {"steps": []}
    
    print("Running Multi-Agent Inference...")
    for step in range(10): # Example steps
        # Random/Dummy policies
        actions = {}
        for agent_id in env.AGENT_IDS:
            if agent_id.startswith("trader"):
                # Buy 1 contract randomly
                actions[agent_id] = VSRAction(
                    selected_strike=0,
                    selected_maturity=0,
                    direction=TradeDirection.BUY,
                    quantity=1.0,
                    reasoning="Random dummy trade"
                ).model_dump()
            elif agent_id == "market_maker":
                actions[agent_id] = MarketMakerAction()
            elif agent_id == "oversight":
                actions[agent_id] = OversightAction()
                
        obs, rewards, done, info = env.step(actions)
        replay["steps"].append(
            {
                "step": step + 1,
                "rewards": rewards,
                "info": info,
            }
        )
        print(f"Step {step+1}: MM Reward={rewards['market_maker']:.2f}")
        if done:
            break

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(replay, f, indent=2)
        print(f"Saved replay to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a multi-agent VSR smoke inference.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save replay JSON")
    args = parser.parse_args()
    run_inference(args.output)

if __name__ == "__main__":
    main()
