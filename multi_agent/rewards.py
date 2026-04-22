import math
from typing import Dict
from multi_agent.models import AgentState, MarketMakerAction, OversightAction


def squash_reward(raw_reward: float, limit: float = 5.0) -> float:
    """Squash reward to [-limit, limit] while preserving small signals.

    Values with |x| <= 1 pass through linearly; larger values are
    compressed with log to avoid extreme outliers dominating GRPO.
    """
    clamped = max(-limit, min(limit, raw_reward))
    if abs(clamped) <= 1.0:
        return clamped
    return math.copysign(1.0 + math.log(abs(clamped)), clamped)

def calculate_trader_reward(agent_state: AgentState, prev_state: AgentState) -> float:
    """Calculate reward for a trader: Δ PnL + Δ Cash - risk penalties.

    The mark-to-market PnL delta alone is near-zero each step because
    GBM drift is 0 and option prices barely move.  Including the cash
    delta captures premium income (for sellers) and premium cost (for
    buyers), making the reward signal immediately meaningful.

    Fines are already reflected in cash_balance changes, so we don't
    subtract them separately to avoid double-counting.
    """
    pnl_delta = agent_state.portfolio_pnl - prev_state.portfolio_pnl
    cash_delta = agent_state.cash_balance - prev_state.cash_balance

    # Total economic change = mark-to-market change + realized cash flow
    # (fines are already embedded in cash_delta)
    total_economic_delta = pnl_delta + cash_delta
    
    # Inventory risk penalty
    total_contracts = sum(abs(pos.get("quantity", 0)) for pos in agent_state.positions)
    inventory_penalty = 1.0 if total_contracts > 50 else 0.0
    
    # Greeks violation penalty (e.g. excessive directional risk)
    greeks_penalty = 1.0 if abs(agent_state.portfolio_delta) > 10.0 else 0.0
    
    raw_reward = total_economic_delta - inventory_penalty - greeks_penalty
    return squash_reward(raw_reward)

def calculate_mm_reward(agent_state: AgentState, prev_state: AgentState, 
                        volume_traded: int, mm_action: MarketMakerAction) -> float:
    """Calculate reward for Market Maker.

    The market maker should not learn the degenerate policy of widening
    spreads forever. Reward balances PnL, facilitated volume, quote quality,
    and inventory control.
    """
    pnl_delta = agent_state.portfolio_pnl - prev_state.portfolio_pnl
    cash_delta = agent_state.cash_balance - prev_state.cash_balance
    # Include cash flow: MM earns spread via premium income in cash_balance
    pnl_delta = pnl_delta + cash_delta
    flow_reward = volume_traded * 0.05

    target_spreads = {"atm": 0.04, "otm": 0.06, "itm": 0.05}
    spread_distance = (
        abs(mm_action.atm_spread - target_spreads["atm"])
        + abs(mm_action.otm_spread - target_spreads["otm"])
        + abs(mm_action.itm_spread - target_spreads["itm"])
    )
    quote_quality_reward = max(0.0, 0.15 - spread_distance)

    inventory_penalty = (
        abs(agent_state.portfolio_delta) * 0.01
        + abs(agent_state.portfolio_gamma) * 0.05
        + abs(agent_state.portfolio_vega) * 0.02
    )
    spread_extremity_penalty = 0.0
    if max(mm_action.atm_spread, mm_action.otm_spread, mm_action.itm_spread) > 0.12:
        spread_extremity_penalty = 0.5

    raw_reward = pnl_delta + flow_reward + quote_quality_reward - inventory_penalty - spread_extremity_penalty
    return squash_reward(raw_reward)

def calculate_oversight_reward(oversight_action: OversightAction, 
                               ground_truth_manipulations: Dict[str, str],
                               pre_stability_score: float = 0.0,
                               post_stability_score: float = 0.0) -> float:
    """Calculate reward for Oversight agent.

    Oversight is framed as market surveillance:
    detect harmful behavior, explain it, and improve market stability.
    """
    reward = 0.0

    harmful_events = {
        agent_id: label
        for agent_id, label in ground_truth_manipulations.items()
        if label != "none"
    }

    # Check flagged agents (True/False Positives)
    for flagged_agent in oversight_action.flagged_agents:
        true_manipulation = ground_truth_manipulations.get(flagged_agent, "none")
        if oversight_action.flag_type == true_manipulation and true_manipulation != "none":
            reward += 1.0  # true positive
        else:
            reward -= 0.5  # false positive

    # Check unflagged agents (False Negatives)
    for agent_id, true_manipulation in harmful_events.items():
        if agent_id not in oversight_action.flagged_agents:
            reward -= 1.0  # missed manipulation! false negative

    if not harmful_events and not oversight_action.flagged_agents:
        reward += 0.25  # correct restraint

    reasoning = oversight_action.reasoning.lower()
    if oversight_action.flag_type != "none" and oversight_action.flag_type in reasoning:
        reward += 0.2
    if any(agent_id.lower() in reasoning for agent_id in oversight_action.flagged_agents):
        reward += 0.1

    if oversight_action.intervention_type == "fine" and oversight_action.fine_amount > 0:
        reward += 0.1
    if oversight_action.intervention_type == "halt" and oversight_action.halt_strikes:
        reward += 0.15

    stability_improvement = max(0.0, pre_stability_score - post_stability_score)
    reward += min(0.3, stability_improvement * 0.2)

    return max(-5.0, min(5.0, reward))
