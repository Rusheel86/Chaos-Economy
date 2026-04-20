import copy
import numpy as np
from typing import Dict, Tuple, Any

from multi_agent.models import AgentRole, MultiAgentObservation, AgentState, MarketMakerAction, OversightAction
from multi_agent.rewards import calculate_trader_reward, calculate_mm_reward, calculate_oversight_reward
from multi_agent.manipulation_detector import ManipulationDetector
from multi_agent.order_matching import OrderMatchingEngine
from multi_agent.config import NUM_TRADERS, EPISODE_LENGTH, INITIAL_CASH

from vsr_env.engine.market_sim import advance_market
from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState
from vsr_env.engine.portfolio import add_position, update_positions_on_market_move

class MultiAgentVSREnvironment:
    AGENT_IDS = [f"trader_{i}" for i in range(NUM_TRADERS)] + ["market_maker", "oversight"]

    def __init__(self):
        self.rng = None
        self.option_engine = OptionChainEngine()
        self.manipulation_detector = ManipulationDetector()
        self.matching_engine = OrderMatchingEngine()
        self.current_step = 0
        self.vsr_state = None
        self.agent_states = {}
        self._agent_vsr_states = {}
        self.mm_last_spreads = {"atm": 0.02, "otm": 0.04, "itm": 0.03}
        self.trade_log = []

    def reset(self, seed: int = 42) -> Dict[str, MultiAgentObservation]:
        """Reset the environment."""
        self.current_step = 0
        self.rng = np.random.RandomState(seed)
        self.trade_log = []
        
        # Base simulation state
        variance = 0.04
        spot_price = 100.0
        self.vsr_state = VSRState(
            episode_id=f"ep_{seed}",
            spot_price=spot_price,
            variance=variance,
            step_count=0
        )
        
        # Initialize agents
        self.agent_states = {}
        self._agent_vsr_states = {}
        for idx in range(NUM_TRADERS):
            agent_id = f"trader_{idx}"
            self.agent_states[agent_id] = AgentState(agent_id=agent_id, role=AgentRole.TRADER, cash_balance=INITIAL_CASH)
            self._agent_vsr_states[agent_id] = VSRState(episode_id=f"ep_{seed}", spot_price=spot_price, variance=variance, step_count=0)
            
        self.agent_states["market_maker"] = AgentState(agent_id="market_maker", role=AgentRole.MARKET_MAKER, cash_balance=INITIAL_CASH * 10)
        self._agent_vsr_states["market_maker"] = VSRState(episode_id=f"ep_{seed}", spot_price=spot_price, variance=variance, step_count=0)
        
        self.agent_states["oversight"] = AgentState(agent_id="oversight", role=AgentRole.OVERSIGHT, cash_balance=0.0)

        self.mm_last_spreads = {"atm": 0.02, "otm": 0.04, "itm": 0.03}

        return self._get_observations()

    def _get_observations(self) -> Dict[str, MultiAgentObservation]:
        """Generate observations for all agents."""
        obs = {}
        S = self.vsr_state.spot_price
        sigma = np.sqrt(self.vsr_state.variance)
        
        # Simple IV surface
        iv_surface = [[sigma] * len(self.option_engine.MATURITIES) for _ in self.option_engine.STRIKES]
        
        for agent_id, state in self.agent_states.items():
            obs[agent_id] = MultiAgentObservation(
                agent_id=agent_id,
                role=state.role,
                iv_surface=iv_surface,
                spot_price=S,
                mm_spreads=self.mm_last_spreads,
                own_greeks={"delta": state.portfolio_delta, "gamma": state.portfolio_gamma, "vega": state.portfolio_vega},
                own_pnl=state.portfolio_pnl,
                own_positions=state.positions,
                own_cash=state.cash_balance,
                step_number=self.current_step,
                steps_remaining=EPISODE_LENGTH - self.current_step,
                all_agent_pnls=None,
                trade_log=None
            )
            
        # Add oversight-specific data
        obs["oversight"].all_agent_pnls = {aid: s.portfolio_pnl for aid, s in self.agent_states.items() if s.role != AgentRole.OVERSIGHT}
        obs["oversight"].trade_log = self.trade_log[-50:] # keep recent
        
        return obs

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, MultiAgentObservation], Dict[str, float], bool, Dict]:
        """Execute one step of the environment."""
        self.current_step += 1
        done = self.current_step >= EPISODE_LENGTH
        rewards = {agent_id: 0.0 for agent_id in self.AGENT_IDS}
        info = {}
        
        # Deep-copy previous state for reward deltas
        prev_states = copy.deepcopy(self.agent_states)
        
        # 1. Parse actions
        trader_actions = {aid: act for aid, act in actions.items() if aid.startswith("trader")}
        
        mm_raw = actions.get("market_maker")
        if isinstance(mm_raw, dict):
            mm_action = MarketMakerAction(**mm_raw)
        elif isinstance(mm_raw, MarketMakerAction):
            mm_action = mm_raw
        else:
            mm_action = MarketMakerAction()
            
        self.mm_last_spreads = {
            "atm": mm_action.atm_spread,
            "otm": mm_action.otm_spread,
            "itm": mm_action.itm_spread
        }

        oversight_raw = actions.get("oversight")
        if isinstance(oversight_raw, dict):
             oversight_action = OversightAction(**oversight_raw)
        elif isinstance(oversight_raw, OversightAction):
             oversight_action = oversight_raw
        else:
             oversight_action = OversightAction()
            
        # 2. Oversight flags & fines
        ground_truth = {aid: self.manipulation_detector.detect_manipulation(self.agent_states[aid], []) for aid in trader_actions}
        
        for flagged in oversight_action.flagged_agents:
            if flagged in trader_actions and oversight_action.flag_type == ground_truth.get(flagged, "none") and oversight_action.flag_type != "none":
                self.agent_states[flagged].fines_received += oversight_action.fine_amount
                self.agent_states[flagged].cash_balance -= oversight_action.fine_amount
        
        # 3. Trader orders + Matching
        executed_trades = self.matching_engine.match_orders(
            trader_actions, mm_action, self.vsr_state.spot_price, self.option_engine, self.vsr_state.variance
        )
        
        # APPLY TRADES to agent portfolios
        for agent_id, trade in executed_trades.items():
            if isinstance(trade, dict) and trade.get("direction") in ["buy", "sell"]:
                # Log trade
                self.trade_log.append({
                    "step": self.current_step,
                    "agent_id": agent_id,
                    "trade": trade
                })
                # Trader position
                add_position(
                    state=self._agent_vsr_states[agent_id],
                    strike_idx=trade["selected_strike"],
                    maturity_idx=trade["selected_maturity"],
                    direction=trade["direction"],
                    quantity=trade["quantity"],
                    engine=self.option_engine,
                    option_type=trade.get("option_type", "call"),
                )
                
                # Market maker position (zero sum)
                mm_dir = "sell" if trade["direction"] == "buy" else "buy"
                add_position(
                    state=self._agent_vsr_states["market_maker"],
                    strike_idx=trade["selected_strike"],
                    maturity_idx=trade["selected_maturity"],
                    direction=mm_dir,
                    quantity=trade["quantity"],
                    engine=self.option_engine,
                    option_type=trade.get("option_type", "call"),
                )
        
        # 4. Market advance
        advance_market(self.vsr_state, self.rng)
        
        # 5. Greeks/PnL Update 
        # Update Greeks/PnL for ALL agents after market advance
        for agent_id, state in self.agent_states.items():
            if state.role != AgentRole.OVERSIGHT:
                vsr = self._agent_vsr_states[agent_id]
                # Sync market conditions
                vsr.spot_price = self.vsr_state.spot_price
                vsr.variance = self.vsr_state.variance
                update_positions_on_market_move(vsr, self.option_engine)
                
                state.positions = vsr.positions
                state.portfolio_pnl = vsr.portfolio_pnl
                state.portfolio_delta = vsr.portfolio_delta
                state.portfolio_gamma = vsr.portfolio_gamma
                state.portfolio_vega = vsr.portfolio_vega
        
        # 6. Rewards
        for aid in trader_actions:
            rewards[aid] = calculate_trader_reward(self.agent_states[aid], prev_states[aid])
            
        rewards["market_maker"] = calculate_mm_reward(self.agent_states["market_maker"], prev_states["market_maker"], 
                                                      len(executed_trades), mm_action)
                                                      
        rewards["oversight"] = calculate_oversight_reward(oversight_action, ground_truth)

        observations = self._get_observations()
        
        return observations, rewards, done, info
