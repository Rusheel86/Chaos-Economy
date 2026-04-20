import json
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction
from vsr_env.models import VSRAction, TradeDirection

def run_inference():
    env = MultiAgentVSREnvironment()
    obs = env.reset()
    
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
        print(f"Step {step+1}: MM Reward={rewards['market_maker']:.2f}")

if __name__ == "__main__":
    run_inference()
