from typing import List, Dict, Any
from multi_agent.models import AgentState

class ManipulationDetector:
    """Heuristic logic to provide ground truth for Oversight Agent training."""
    
    def __init__(self):
        self.trade_history = {}
        
    def check_gamma_squeeze(self, agent_state: AgentState) -> bool:
        """Detect if agent is hoarding high gamma (OTM Call) positions."""
        # Realistic threshold for gamma cornering
        return agent_state.portfolio_gamma > 2.0
        
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
        
    def check_spoofing(self, agent_state: AgentState, step_trades: List[Dict]) -> bool:
        """Detect if agent places large orders and immediately cancels (simulated by rapid reversals)."""
        # For our environment, we'll simulate spoofing detection by looking for very large trades 
        # that don't match the agent's typical volume.
        for t in step_trades:
            if t.get("quantity", 0) > 30: # unusually large order
                return True
        return False
        
    def detect_manipulation(self, agent_state: AgentState, step_trades: List[Dict]) -> str:
        """Return the type of manipulation detected, or 'none'."""
        if self.check_gamma_squeeze(agent_state):
            return "gamma_squeeze"
            
        # Get trades for this specific agent
        agent_step_trades = [t for t in step_trades if t.get("agent_id") == agent_state.agent_id] or step_trades
        
        if self.check_wash_trading(agent_state.agent_id, agent_step_trades):
            return "wash_trading"
            
        if self.check_spoofing(agent_state, agent_step_trades):
            return "spoofing"
            
        return "none"
