from typing import Dict, Any, List
from multi_agent.models import AgentState, MarketMakerAction, OversightAction

def calculate_trader_reward(agent_state: AgentState, prev_state: AgentState) -> float:
    """Calculate reward for a trader: Δ PnL - fines - risk penalties."""
    pnl_delta = agent_state.portfolio_pnl - prev_state.portfolio_pnl
    fines_delta = agent_state.fines_received - prev_state.fines_received
    
    # Inventory risk penalty
    total_contracts = sum(abs(pos.get("quantity", 0)) for pos in agent_state.positions)
    inventory_penalty = 1.0 if total_contracts > 50 else 0.0
    
    # Greeks violation penalty (e.g. excessive directional risk)
    greeks_penalty = 1.0 if abs(agent_state.portfolio_delta) > 10.0 else 0.0
    
    # Scale total reward roughly to [-1, 1] range depending on standard PnL swings
    raw_reward = pnl_delta - fines_delta - inventory_penalty - greeks_penalty
    return max(-1.0, min(1.0, raw_reward / 100.0))

def calculate_mm_reward(agent_state: AgentState, prev_state: AgentState, 
                        volume_traded: int, mm_action: MarketMakerAction) -> float:
    """Calculate reward for Market Maker."""
    pnl_delta = agent_state.portfolio_pnl - prev_state.portfolio_pnl
    rebate = volume_traded * 0.001
    
    # Spread violation penalty if spreads are too wide/narrow beyond safe bounds (heuristic)
    spread_violation_penalty = 0.0
    if mm_action.atm_spread > 0.15:
        spread_violation_penalty += 5.0
        
    return pnl_delta + rebate - spread_violation_penalty

def calculate_oversight_reward(oversight_action: OversightAction, 
                               ground_truth_manipulations: Dict[str, str]) -> float:
    """Calculate reward for Oversight agent."""
    reward = 0.0
    
    # Check flagged agents (True/False Positives)
    for flagged_agent in oversight_action.flagged_agents:
        true_manipulation = ground_truth_manipulations.get(flagged_agent, "none")
        if oversight_action.flag_type == true_manipulation and true_manipulation != "none":
            reward += 1.0  # true positive
        else:
            reward -= 0.5  # false positive
            
    # Check unflagged agents (False Negatives)
    for agent_id, true_manipulation in ground_truth_manipulations.items():
        if true_manipulation != "none" and agent_id not in oversight_action.flagged_agents:
            reward -= 1.0  # missed manipulation! false negative
            
    return max(-1.0, min(1.0, reward))
