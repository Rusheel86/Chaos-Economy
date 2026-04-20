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

    def reset(self, seed: int = 42) -> Dict[str, MultiAgentObservation]:
        """Reset the environment."""
        self.current_step = 0
        self.rng = np.random.RandomState(seed)
        
        # Base simulation state
        variance = 0.04
        self.vsr_state = VSRState(
            episode_id=f"ep_{seed}",
            spot_price=100.0,
            variance=variance,
            step_count=0
        )
        
        # Initialize agents
        self.agent_states = {}
        for idx in range(NUM_TRADERS):
            agent_id = f"trader_{idx}"
            self.agent_states[agent_id] = AgentState(agent_id=agent_id, role=AgentRole.TRADER, cash_balance=INITIAL_CASH)
            
        self.agent_states["market_maker"] = AgentState(agent_id="market_maker", role=AgentRole.MARKET_MAKER, cash_balance=INITIAL_CASH * 10)
        self.agent_states["oversight"] = AgentState(agent_id="oversight", role=AgentRole.OVERSIGHT, cash_balance=0.0)

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
                mm_spreads={"atm": 0.02, "otm": 0.04}, # Placeholder, should be tracked from MM action
                own_greeks={"delta": state.portfolio_delta, "gamma": state.portfolio_gamma, "vega": state.portfolio_vega},
                own_pnl=state.portfolio_pnl,
                own_positions=state.positions,
                own_cash=state.cash_balance,
                step_number=self.current_step,
                steps_remaining=EPISODE_LENGTH - self.current_step
            )
            
        # Add oversight-specific data
        obs["oversight"].all_agent_pnls = {aid: s.portfolio_pnl for aid, s in self.agent_states.items() if s.role != AgentRole.OVERSIGHT}
        obs["oversight"].trade_log = []
        
        return obs

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, MultiAgentObservation], Dict[str, float], bool, Dict]:
        """Execute one step of the environment."""
        self.current_step += 1
        done = self.current_step >= EPISODE_LENGTH
        rewards = {agent_id: 0.0 for agent_id in self.AGENT_IDS}
        info = {}
        
        # 1. Parse actions
        trader_actions = {aid: act for aid, act in actions.items() if aid.startswith("trader")}
        mm_action = actions.get("market_maker", MarketMakerAction())
        # Provide default initialization for oversight_action
        oversight_action = actions.get("oversight")
        if oversight_action is None:
            oversight_action = OversightAction()
        elif isinstance(oversight_action, dict):
             oversight_action = OversightAction(**oversight_action)
            
        # 2. Oversight flags & fines
        ground_truth = {aid: self.manipulation_detector.detect_manipulation(self.agent_states[aid], []) for aid in trader_actions}
        
        for flagged in oversight_action.flagged_agents:
            if flagged in trader_actions and oversight_action.flag_type == ground_truth.get(flagged, "none") and oversight_action.flag_type != "none":
                self.agent_states[flagged].fines_received += oversight_action.fine_amount
                self.agent_states[flagged].cash_balance -= oversight_action.fine_amount
        
        # 3. Trader orders + Matching
        executed_trades = self.matching_engine.match_orders(trader_actions, mm_action)
        # Apply trades pseudo-logic
        # In a real impl, we'd add_position to each trader's state 
        
        # 4. Market advance
        advance_market(self.vsr_state, self.rng)
        
        # 5. Greeks/PnL Update 
        # Needs to update all agent portfolios' PnL and Greeks based on new market S
        # Here we abstract it for demo purposes.
        
        # 6. Rewards
        for aid in trader_actions:
            # We pass a dummy previous state here, ideally we save the actual previous state
            rewards[aid] = calculate_trader_reward(self.agent_states[aid], self.agent_states[aid])
            
        rewards["market_maker"] = calculate_mm_reward(self.agent_states["market_maker"], self.agent_states["market_maker"], 
                                                      len(executed_trades), mm_action)
                                                      
        rewards["oversight"] = calculate_oversight_reward(oversight_action, ground_truth)

        observations = self._get_observations()
        
        return observations, rewards, done, info
