from typing import Dict, Any, List
from vsr_env.models import VSRAction
from multi_agent.models import MarketMakerAction
from multi_agent.config import NUM_TRADERS

class OrderMatchingEngine:
    """Matches trader orders with Market Maker spreads."""
    
    def __init__(self):
        self.total_volume = 0
        
    def match_orders(self, trader_actions: Dict[str, dict], mm_action: MarketMakerAction) -> Dict[str, dict]:
        """
        Takes trader actions and applies bid-ask spread to execution prices.
        Returns the executed trades to apply to environments.
        """
        executed_trades = {}
        batch_volume = 0
        
        for agent_id, action_dict in trader_actions.items():
            # In a real engine, we'd apply the spread based on option moneyness.
            # Here we just abstract it for matching.
            if action_dict.get("direction") != "hold":
                quantity = action_dict.get("quantity", 0)
                batch_volume += quantity
                executed_trades[agent_id] = action_dict
                
        self.total_volume += batch_volume
        return executed_trades
