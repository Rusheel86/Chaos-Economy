from typing import Dict, Any, List
from vsr_env.models import VSRAction
from multi_agent.models import MarketMakerAction
from multi_agent.config import NUM_TRADERS
import numpy as np

class OrderMatchingEngine:
    """Matches trader orders with Market Maker spreads."""
    
    def __init__(self):
        self.total_volume = 0
        
    def match_orders(
        self, 
        trader_actions: Dict[str, dict], 
        mm_action: MarketMakerAction, 
        spot_price: float, 
        option_engine: Any,
        variance: float
    ) -> Dict[str, dict]:
        """
        Takes trader actions and applies bid-ask spread to execution prices.
        Returns the executed trades containing execution details.
        """
        executed_trades = {}
        batch_volume = 0
        
        sigma = np.sqrt(variance)
        
        for agent_id, action_dict in trader_actions.items():
            if action_dict.get("direction") not in ["buy", "sell"]:
                continue
                
            quantity = action_dict.get("quantity", 0)
            if quantity <= 0:
                continue
            
            strike_idx = action_dict.get("selected_strike", 0)
            maturity_idx = action_dict.get("selected_maturity", 0)
            option_type = action_dict.get("option_type", "call")
            
            if strike_idx >= len(option_engine.STRIKES) or maturity_idx >= len(option_engine.MATURITIES):
                continue
                
            K = option_engine.STRIKES[strike_idx]
            T = option_engine.MATURITIES[maturity_idx]
            
            # Compute theoretical price
            theo_price = option_engine.bs_price(
                spot_price, np.array([K]), np.array([T]), np.array([sigma]), option_type=option_type
            )[0]
            
            # Determine moneyness
            moneyness = spot_price / K
            if option_type == "call":
                if moneyness > 1.05:
                    spread = mm_action.itm_spread
                elif moneyness < 0.95:
                    spread = mm_action.otm_spread
                else:
                    spread = mm_action.atm_spread
            else: # put
                if moneyness < 0.95:
                    spread = mm_action.itm_spread
                elif moneyness > 1.05:
                    spread = mm_action.otm_spread
                else:
                    spread = mm_action.atm_spread
            
            # Apply spread based on direction
            # If trader buys, they pay MORE than theo price
            # If trader sells, they receive LESS than theo price
            direction = action_dict["direction"]
            if direction == "buy":
                execution_price = theo_price + (spread / 2.0)
            else:
                execution_price = max(0.01, theo_price - (spread / 2.0))
                
            action_dict["execution_price"] = execution_price
            action_dict["theo_price"] = theo_price
            action_dict["spread_applied"] = spread
            
            batch_volume += quantity
            executed_trades[agent_id] = action_dict
                
        self.total_volume += batch_volume
        return executed_trades
