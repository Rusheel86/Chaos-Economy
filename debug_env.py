import json
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import AgentRole

env = MultiAgentVSREnvironment()
obs = env.reset()

actions = {
    "market_maker": {
        "atm_spread": 0.05,
        "otm_spread": 0.08,
        "itm_spread": 0.06,
        "skew_adjustment": 0.0,
        "reasoning": "debug"
    },
    "oversight": {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "halt_strikes": [],
        "confidence": 0.0,
        "intervention_type": "none",
        "reasoning": "debug"
    }
}
for i in range(10):
    actions[f"trader_{i}"] = {
        "selected_strike": 4,
        "selected_maturity": 0,
        "direction": "buy",
        "quantity": 1.0,
        "option_type": "call",
        "reasoning": "debug"
    }

obs, rewards, done, info = env.step(actions)
print("Step 1 rewards:", rewards)

obs, rewards, done, info = env.step(actions)
print("Step 2 rewards:", rewards)
