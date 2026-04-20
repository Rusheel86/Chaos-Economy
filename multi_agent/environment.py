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
        self.intervention_log = []

    def reset(self, seed: int = 42) -> Dict[str, MultiAgentObservation]:
        """Reset the environment."""
        self.current_step = 0
        self.rng = np.random.RandomState(seed)
        self.trade_log = []
        self.intervention_log = []
        
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

    def _normalize_trader_actions(self, actions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        normalized = {}
        for agent_id, action in actions.items():
            if not agent_id.startswith("trader"):
                continue
            if hasattr(action, "model_dump"):
                normalized[agent_id] = action.model_dump()
            elif isinstance(action, dict):
                normalized[agent_id] = dict(action)
        return normalized

    def _get_observations(self) -> Dict[str, MultiAgentObservation]:
        """Generate observations for all agents."""
        obs = {}
        S = self.vsr_state.spot_price
        sigma = np.sqrt(self.vsr_state.variance)
        
        # Simple IV surface
        iv_surface = [[sigma] * len(self.option_engine.MATURITIES) for _ in self.option_engine.STRIKES]
        
        risk_summary = self._build_agent_risk_summary()
        market_summary = self._build_market_state_summary()

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
                trade_log=None,
                agent_risk_summary=None,
                market_state_summary=None,
                recent_interventions=None,
            )
            
        # Add oversight-specific data
        obs["oversight"].all_agent_pnls = {aid: s.portfolio_pnl for aid, s in self.agent_states.items() if s.role != AgentRole.OVERSIGHT}
        obs["oversight"].trade_log = self.trade_log[-50:] # keep recent
        obs["oversight"].agent_risk_summary = risk_summary
        obs["oversight"].market_state_summary = market_summary
        obs["oversight"].recent_interventions = self.intervention_log[-20:]
        
        return obs

    def _build_agent_risk_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for agent_id, state in self.agent_states.items():
            if state.role == AgentRole.OVERSIGHT:
                continue
            total_contracts = float(sum(abs(pos.get("quantity", 0.0)) for pos in state.positions))
            risk_score = (
                abs(state.portfolio_delta) * 0.1
                + abs(state.portfolio_gamma) * 0.4
                + abs(state.portfolio_vega) * 0.1
                + total_contracts * 0.02
            )
            summary[agent_id] = {
                "pnl": float(state.portfolio_pnl),
                "delta": float(state.portfolio_delta),
                "gamma": float(state.portfolio_gamma),
                "vega": float(state.portfolio_vega),
                "cash": float(state.cash_balance),
                "contracts": total_contracts,
                "risk_score": float(risk_score),
            }
        return summary

    def _build_market_state_summary(self) -> Dict[str, float]:
        trade_window = self.trade_log[-25:]
        total_volume = float(sum(t.get("quantity", 0.0) for t in trade_window))
        mm_state = self.agent_states.get("market_maker", AgentState(agent_id="tmp", role=AgentRole.MARKET_MAKER))
        inventory_stress = (
            abs(mm_state.portfolio_delta) * 0.1
            + abs(mm_state.portfolio_gamma) * 0.6
            + abs(mm_state.portfolio_vega) * 0.08
        )
        avg_spread = (
            self.mm_last_spreads["atm"] + self.mm_last_spreads["otm"] + self.mm_last_spreads["itm"]
        ) / 3.0
        market_stability_score = float(inventory_stress + avg_spread + (total_volume * 0.01))
        return {
            "inventory_stress": float(inventory_stress),
            "avg_spread": float(avg_spread),
            "recent_volume": total_volume,
            "market_stability_score": market_stability_score,
        }

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, MultiAgentObservation], Dict[str, float], bool, Dict]:
        """Execute one step of the environment."""
        self.current_step += 1
        done = self.current_step >= EPISODE_LENGTH
        rewards = {agent_id: 0.0 for agent_id in self.AGENT_IDS}
        info = {}
        
        # Deep-copy previous state for reward deltas
        prev_states = copy.deepcopy(self.agent_states)
        
        # 1. Parse actions
        trader_actions = self._normalize_trader_actions(actions)
        
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
            
        # 2. Trader orders + Matching
        executed_trades = self.matching_engine.match_orders(
            trader_actions, mm_action, self.vsr_state.spot_price, self.option_engine, self.vsr_state.variance
        )

        step_trades = []

        # APPLY TRADES to agent portfolios
        for agent_id, trade in executed_trades.items():
            if isinstance(trade, dict) and trade.get("direction") in ["buy", "sell"]:
                execution_price = float(trade.get("execution_price", trade.get("theo_price", 0.0)))
                quantity = float(trade["quantity"])
                premium = execution_price * quantity

                # Log trade
                step_trade = {
                    "step": self.current_step,
                    "agent_id": agent_id,
                    **trade,
                }
                step_trades.append(step_trade)
                self.trade_log.append(step_trade)

                self.agent_states[agent_id].cash_balance -= premium
                self.agent_states["market_maker"].cash_balance += premium

                # Trader position
                add_position(
                    state=self._agent_vsr_states[agent_id],
                    strike_idx=trade["selected_strike"],
                    maturity_idx=trade["selected_maturity"],
                    direction=trade["direction"],
                    quantity=trade["quantity"],
                    engine=self.option_engine,
                    option_type=trade.get("option_type", "call"),
                    entry_price_override=execution_price,
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
                    entry_price_override=execution_price,
                )
        
        # 3. Market advance
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
        
        pre_stability_score = self._build_market_state_summary()["market_stability_score"]

        # 4. Oversight labels and enforcement
        ground_truth = {
            aid: self.manipulation_detector.detect_manipulation(self.agent_states[aid], step_trades)
            for aid in trader_actions
        }

        for flagged in oversight_action.flagged_agents:
            if (
                flagged in trader_actions
                and oversight_action.flag_type == ground_truth.get(flagged, "none")
                and oversight_action.flag_type != "none"
            ):
                self.agent_states[flagged].fines_received += oversight_action.fine_amount
                self.agent_states[flagged].cash_balance -= oversight_action.fine_amount
                self.agent_states["oversight"].cash_balance += oversight_action.fine_amount
                intervention_record = {
                    "step": self.current_step,
                    "agent_id": flagged,
                    "flag_type": oversight_action.flag_type,
                    "fine_amount": oversight_action.fine_amount,
                    "intervention_type": oversight_action.intervention_type,
                }
                if oversight_action.halt_strikes or oversight_action.intervention_type == "halt":
                    self.agent_states[flagged].is_halted = True
                    intervention_record["halt_strikes"] = list(oversight_action.halt_strikes)
                self.intervention_log.append(intervention_record)

        post_stability_score = self._build_market_state_summary()["market_stability_score"]

        # 5. Rewards
        for aid in trader_actions:
            rewards[aid] = calculate_trader_reward(self.agent_states[aid], prev_states[aid])
            
        rewards["market_maker"] = calculate_mm_reward(
            self.agent_states["market_maker"],
            prev_states["market_maker"],
            len(executed_trades),
            mm_action,
        )
                                                      
        rewards["oversight"] = calculate_oversight_reward(
            oversight_action,
            ground_truth,
            pre_stability_score=pre_stability_score,
            post_stability_score=post_stability_score,
        )

        observations = self._get_observations()

        risk_summary = observations["oversight"].agent_risk_summary or {}
        market_summary = observations["oversight"].market_state_summary or {}
        info = {
            "trade_count": len(step_trades),
            "total_volume": float(sum(t.get("quantity", 0.0) for t in step_trades)),
            "market_maker_spreads": dict(self.mm_last_spreads),
            "detected_manipulations": ground_truth,
            "agent_risk_summary": risk_summary,
            "market_state_summary": market_summary,
            "recent_interventions": observations["oversight"].recent_interventions or [],
        }

        return observations, rewards, done, info
