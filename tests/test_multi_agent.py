import pytest
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import AgentRole

def test_multi_agent_initialization():
    env = MultiAgentVSREnvironment()
    observations = env.reset(seed=42)
    
    assert "trader_0" in observations
    assert "market_maker" in observations
    assert "oversight" in observations
    
    assert observations["trader_0"].role == AgentRole.TRADER
    assert observations["market_maker"].role == AgentRole.MARKET_MAKER
    assert observations["oversight"].role == AgentRole.OVERSIGHT

def test_multi_agent_step():
    env = MultiAgentVSREnvironment()
    env.reset(seed=42)
    
    # Test a simple hold step
    initial_actions = {
        "trader_0": {"selected_strike": 4, "selected_maturity": 0, "direction": "hold", "quantity": 0, "option_type": "call", "reasoning": "testing"},
        "market_maker": {"atm_spread": 0.02, "otm_spread": 0.04, "itm_spread": 0.03},
        "oversight": {"flagged_agents": [], "flag_type": "none", "fine_amount": 0.0}
    }
    
    obs, rewards, done, info = env.step(initial_actions)
    
    # Assert structural integrity 
    assert not done
    assert "trader_0" in obs
    assert "market_maker" in rewards
    assert info["trade_count"] == 0
