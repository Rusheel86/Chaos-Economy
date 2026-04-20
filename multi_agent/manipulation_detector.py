from typing import List, Dict, Any
from multi_agent.models import AgentState

class ManipulationDetector:
    """Heuristic logic to provide ground truth for Oversight Agent training."""
    
    def __init__(self):
        self.trade_history = {}
        self.order_pressure = {}
        
    def check_gamma_pressure(self, agent_state: AgentState, step_trades: List[Dict]) -> bool:
        """Detect concentrated gamma-heavy pressure in one direction."""
        call_buys = [
            t for t in step_trades
            if t.get("option_type", "call") == "call" and t.get("direction") == "buy"
        ]
        size_pressure = sum(float(t.get("quantity", 0.0)) for t in call_buys)
        return agent_state.portfolio_gamma > 2.0 or size_pressure > 8.0

    def check_systemic_risk(self, agent_state: AgentState) -> bool:
        """Detect destabilizing exposures even if they are not manipulative."""
        return (
            abs(agent_state.portfolio_delta) > 3.0
            or abs(agent_state.portfolio_gamma) > 3.0
            or abs(agent_state.portfolio_vega) > 8.0
        )
        
    def check_wash_trading(self, agent_id: str, new_trades: List[Dict]) -> bool:
        """Detect rapid buy/sell of same instrument."""
        # Simple heuristic implementation
        if agent_id not in self.trade_history:
            self.trade_history[agent_id] = []
            
        # Add new trades to history
        for t in new_trades:
            self.trade_history[agent_id].append({
                "strike": t.get("selected_strike"),
                "maturity": t.get("selected_maturity"),
                "direction": t.get("direction"),
                "option_type": t.get("option_type", "call")
            })
            
        # Keep only recent history (e.g., last 10 trades)
        self.trade_history[agent_id] = self.trade_history[agent_id][-10:]
        
        # Check if we have opposing directions for same instrument
        recent_trades = self.trade_history[agent_id]
        if len(recent_trades) < 2:
            return False
            
        last_trade = recent_trades[-1]
        for past_trade in recent_trades[:-1]:
            # Same instrument
            if (past_trade["strike"] == last_trade["strike"] and
                past_trade["maturity"] == last_trade["maturity"] and
                past_trade["option_type"] == last_trade["option_type"]):
                # Opposing directions
                if past_trade["direction"] != last_trade["direction"]:
                    # Wash trading detected!
                    return True
                    
        return False
        
    def check_spoofing_like_pressure(self, agent_id: str, step_trades: List[Dict]) -> bool:
        """Detect oversized short-window order pressure."""
        if agent_id not in self.order_pressure:
            self.order_pressure[agent_id] = []

        for t in step_trades:
            self.order_pressure[agent_id].append(float(t.get("quantity", 0.0)))

        self.order_pressure[agent_id] = self.order_pressure[agent_id][-5:]
        if not self.order_pressure[agent_id]:
            return False

        avg_recent = sum(self.order_pressure[agent_id]) / len(self.order_pressure[agent_id])
        max_recent = max(self.order_pressure[agent_id])
        if max_recent >= 12.0 and max_recent > (avg_recent * 1.8):
            return True
        return False
        
    def detect_manipulation(self, agent_state: AgentState, step_trades: List[Dict]) -> str:
        """Return the type of harmful behavior detected, or 'none'."""
        agent_step_trades = [t for t in step_trades if t.get("agent_id") == agent_state.agent_id] or step_trades
        
        if self.check_wash_trading(agent_state.agent_id, agent_step_trades):
            return "wash_trading"
            
        if self.check_spoofing_like_pressure(agent_state.agent_id, agent_step_trades):
            return "spoofing_like_pressure"

        if self.check_gamma_pressure(agent_state, agent_step_trades):
            return "gamma_pressure"

        if self.check_systemic_risk(agent_state):
            return "systemic_risk"
            
        return "none"
