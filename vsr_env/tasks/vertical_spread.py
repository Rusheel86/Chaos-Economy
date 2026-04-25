"""Vertical Spread Task implementation for VSR-Env.

This module implements a medium-hard task where agents use vertical spreads
for directional bets with defined risk.

Requirements: Multi-leg strategy support
"""

from typing import Any, List, Tuple

import numpy as np

from vsr_env.engine.option_chain import OptionChainEngine
from vsr_env.models import VSRState


class VerticalSpreadTask:
    """Medium-Hard task: Directional trading with vertical spreads.

    The agent analyzes market conditions and constructs an appropriate
    vertical spread (bull call, bear put, credit spread, etc.) based on
    directional bias and IV considerations.

    Attributes:
        max_steps: Maximum steps per episode (6)
        difficulty: Task difficulty level ("medium")
    """

    max_steps: int = 8
    difficulty: str = "medium"

    def initialize(
        self, state: VSRState, rng: np.random.RandomState
    ) -> List[Tuple[Tuple[int, int], str, float]]:
        """Initialize the vertical spread task.

        Sets up a trending market with clear directional bias.

        Args:
            state: VSRState to initialize (modified in place)
            rng: Seeded numpy RandomState for reproducibility

        Returns:
            Empty list (no mispricings for this task)
        """
        # Create engine for pricing
        engine = OptionChainEngine()

        # Determine market direction
        direction = rng.choice(["bull", "bear"], p=[0.5, 0.5])

        # Set IV environment
        base_var = 0.04  # 20% vol
        state.variance = base_var

        # Store expected direction for grading
        state.expected_direction = direction

        # Set up trending spot price
        # Will be modified by market simulation to trend in the expected direction
        state.spot_price = 100.0

        # Store optimal strike selection for grading
        if direction == "bull":
            # Bull spread: buy lower strike, sell higher strike
            state.optimal_strikes = (3, 5)  # 97.5-102.5 spread
            state.expected_outcome = "bull_spread"
        else:
            # Bear spread: buy higher strike put, sell lower strike put
            state.optimal_strikes = (5, 3)  # 102.5-97.5 spread
            state.expected_outcome = "bear_spread"

        # No initial positions
        state.positions = []
        state.portfolio_delta = 0.0
        state.portfolio_gamma = 0.0
        state.portfolio_vega = 0.0
        state.portfolio_pnl = 0.0

        return []

    def get_description(self) -> str:
        """Return the task objective description.

        Returns:
            Task description string for the agent
        """
        return (
            "The market has a clear directional bias. Construct a vertical spread using "
            "strategy_type='vertical_spread' for an atomic execution. Choose between "
            "bull call spread (buy lower strike call, sell higher strike call) for bullish outlook "
            "or bear put spread (buy higher strike put, sell lower strike put) for bearish outlook. "
            "Optimize strike selection for maximum profit potential while managing breakeven risk. "
            "You have 8 steps total."
        )


class VerticalSpreadGrader:
    """Grader for Vertical Spread task.

    Scores based on:
    - Correct spread direction
    - Strike selection quality
    - Entry price execution
    - Exit timing

    Requirements: Multi-leg strategy evaluation
    """

    def score(self, episode_history: List[Any], state: VSRState) -> float:
        """Compute final score for Vertical Spread task.

        Score = direction_correctness × 0.25
              + strike_selection × 0.25
              + entry_price × 0.20
              + exit_timing × 0.30

        Args:
            episode_history: List of step records with 'action'
            state: Final VSRState with portfolio_pnl

        Returns:
            Score in [0.01, 0.99]
        """
        # Check if agent constructed a spread
        has_spread = False
        direction_correctness = 0.0
        strike_selection_score = 0.0
        entry_step = None

        expected_direction = getattr(state, "expected_direction", "bull")
        optimal_strikes = getattr(state, "optimal_strikes", (3, 5))

        for i, step in enumerate(episode_history):
            action = step.get("action")
            if action is None:
                continue

            # Check for spread action
            if hasattr(action, "strategy_type") and action.strategy_type is not None:
                from vsr_env.models import StrategyType

                if action.strategy_type == StrategyType.VERTICAL_SPREAD:
                    has_spread = True
                    if entry_step is None:
                        entry_step = i

                    # Evaluate spread
                    if action.legs and len(action.legs) >= 2:
                        # Check direction correctness
                        opt_type = action.legs[0].option_type
                        strike1 = action.legs[0].strike_idx
                        strike2 = action.legs[1].strike_idx
                        lower_strike = min(strike1, strike2)
                        higher_strike = max(strike1, strike2)

                        # Determine direction correctness
                        # A spread is bullish if the lower strike is bought, regardless of call or put
                        # It's bearish if the lower strike is sold.
                        lower_direction = None
                        for leg in action.legs:
                            if leg.strike_idx == lower_strike:
                                lower_direction = leg.direction
                                break

                        if lower_direction:
                            is_bull = (lower_direction == "buy")
                            if is_bull and expected_direction == "bull":
                                direction_correctness = 1.0
                            elif not is_bull and expected_direction == "bear":
                                direction_correctness = 1.0

                        # Strike selection quality
                        selected_strikes = sorted([strike1, strike2])
                        optimal_sorted = sorted(optimal_strikes)

                        # Score based on proximity to optimal strikes
                        strike_diff = abs(selected_strikes[0] - optimal_sorted[0]) + abs(
                            selected_strikes[1] - optimal_sorted[1]
                        )
                        strike_selection_score = max(0.0, 1.0 - strike_diff / 8.0)

        # Bonus for using atomic spread action
        if not has_spread:
            has_manual_spread = self._check_manual_spread(episode_history)
            if has_manual_spread:
                direction_correctness, strike_selection_score = self._evaluate_manual_spread(
                    episode_history, expected_direction, optimal_strikes
                )
                # Penalty for not using atomic strategy_type
                direction_correctness *= 0.8
                strike_selection_score *= 0.8

        # Entry price quality (lower is better for debit spreads)
        # Simplified: use step of entry as proxy
        entry_quality = max(0.0, 1.0 - (entry_step or 6) / 8.0) if entry_step else 0.0

        # Exit timing score based on final P&L
        final_pnl = state.portfolio_pnl
        exit_score = self._sigmoid(final_pnl, scale=0.3)

        # Final score: weighted combination
        score = (
            direction_correctness * 0.25
            + strike_selection_score * 0.25
            + entry_quality * 0.20
            + exit_score * 0.30
        )

        # Clamp to [0.01, 0.99]
        return min(max(score, 0.01), 0.99)

    def _check_manual_spread(self, episode_history: List[Any]) -> bool:
        """Check if agent built a spread manually."""
        calls = []
        puts = []

        for step in episode_history:
            action = step.get("action")
            if action is None:
                continue

            # Standard fields
            qty = getattr(action, "quantity", 0)
            if qty <= 0:
                continue

            direction_val = action.direction.value if hasattr(action.direction, "value") else action.direction
            opt_type = getattr(action, "option_type", "call")
            strike = getattr(action, "selected_strike", 0)

            if opt_type == "call":
                calls.append((strike, direction_val))
            else:
                puts.append((strike, direction_val))

        # A spread has at least two legs of the same type with opposite directions
        def has_opposite(legs):
            if len(legs) < 2:
                return False
            has_buy = any(l[1] == "buy" for l in legs)
            has_sell = any(l[1] == "sell" for l in legs)
            return has_buy and has_sell

        return has_opposite(calls) or has_opposite(puts)

    def _evaluate_manual_spread(
        self, episode_history: List[Any], expected_direction: str, optimal_strikes: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Evaluate direction and strike selection for manual spreads."""
        calls = []
        puts = []
        
        for step in episode_history:
            action = step.get("action")
            if action is None: continue
            qty = getattr(action, "quantity", 0)
            if qty <= 0: continue
            
            direction = action.direction.value if hasattr(action.direction, "value") else action.direction
            opt_type = getattr(action, "option_type", "call")
            strike = getattr(action, "selected_strike", 0)
            
            if opt_type == "call":
                calls.append((strike, direction))
            else:
                puts.append((strike, direction))

        # Select the active group
        legs = calls if len(calls) >= 2 else puts
        if len(legs) < 2:
            return 0.0, 0.0

        # Extract strikes
        buy_strikes = [l[0] for l in legs if l[1] == "buy"]
        sell_strikes = [l[0] for l in legs if l[1] == "sell"]
        
        if not buy_strikes or not sell_strikes:
            return 0.0, 0.0

        # Most recent legs define the spread
        strike_buy = buy_strikes[-1]
        strike_sell = sell_strikes[-1]
        
        # Bullish if strike_buy < strike_sell
        is_bull = strike_buy < strike_sell
        direction_correctness = 1.0 if (is_bull and expected_direction == "bull") or (not is_bull and expected_direction == "bear") else 0.0
        
        # Strike selection
        selected_strikes = sorted([strike_buy, strike_sell])
        optimal_sorted = sorted(optimal_strikes)
        strike_diff = abs(selected_strikes[0] - optimal_sorted[0]) + abs(selected_strikes[1] - optimal_sorted[1])
        strike_selection_score = max(0.0, 1.0 - strike_diff / 8.0)
        
        return direction_correctness, strike_selection_score

    def _sigmoid(self, x: float, scale: float = 0.3) -> float:
        """Sigmoid function centered at 0."""
        import math

        return 1.0 / (1.0 + math.exp(-x / scale))