"""Tests for the multi-agent market environment."""

from multi_agent.config import EPISODE_LENGTH, NUM_TRADERS
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction
from vsr_env.models import TradeDirection, VSRAction


def make_buy_action(strike=4, maturity=0, quantity=1.0):
    return VSRAction(
        selected_strike=strike,
        selected_maturity=maturity,
        direction=TradeDirection.BUY,
        quantity=quantity,
        reasoning="Test buy action",
    ).model_dump()


def make_sell_action(strike=4, maturity=0, quantity=1.0):
    return VSRAction(
        selected_strike=strike,
        selected_maturity=maturity,
        direction=TradeDirection.SELL,
        quantity=quantity,
        reasoning="Test sell action",
    ).model_dump()


def test_reset_produces_expected_agents():
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=7)
    assert len(obs) == NUM_TRADERS + 2
    assert "market_maker" in obs
    assert "oversight" in obs
    assert obs["trader_0"].steps_remaining == EPISODE_LENGTH


def test_step_produces_non_zero_rewards_and_trade_log():
    env = MultiAgentVSREnvironment()
    env.reset(seed=11)
    obs, rewards, done, info = env.step(
        {
            "trader_0": make_buy_action(),
            "market_maker": MarketMakerAction().model_dump(),
            "oversight": OversightAction().model_dump(),
        }
    )
    assert done is False
    assert rewards["trader_0"] != 0.0
    assert info["trade_count"] == 1
    assert len(obs["oversight"].trade_log or []) == 1
    assert obs["oversight"].agent_risk_summary is not None
    assert obs["oversight"].market_state_summary is not None


def test_spreads_change_realized_economics():
    def run_with_spread(spread):
        env = MultiAgentVSREnvironment()
        env.reset(seed=123)
        env.step(
            {
                "trader_0": make_buy_action(),
                "market_maker": MarketMakerAction(
                    atm_spread=spread,
                    otm_spread=spread,
                    itm_spread=spread,
                ).model_dump(),
                "oversight": OversightAction().model_dump(),
            }
        )
        return (
            env.agent_states["trader_0"].cash_balance,
            env.agent_states["trader_0"].portfolio_pnl,
            env.agent_states["market_maker"].cash_balance,
        )

    tight = run_with_spread(0.001)
    wide = run_with_spread(0.2)
    assert tight != wide
    assert tight[0] > wide[0]
    assert tight[1] > wide[1]


def test_cash_balances_update_on_fill():
    env = MultiAgentVSREnvironment()
    env.reset(seed=17)
    trader_cash_before = env.agent_states["trader_0"].cash_balance
    mm_cash_before = env.agent_states["market_maker"].cash_balance

    env.step(
        {
            "trader_0": make_buy_action(quantity=2.0),
            "market_maker": MarketMakerAction().model_dump(),
            "oversight": OversightAction().model_dump(),
        }
    )

    assert env.agent_states["trader_0"].cash_balance < trader_cash_before
    assert env.agent_states["market_maker"].cash_balance > mm_cash_before


def test_oversight_detects_wash_trading():
    env = MultiAgentVSREnvironment()
    env.reset(seed=21)
    env.step(
        {
            "trader_0": make_buy_action(strike=2, maturity=1),
            "market_maker": MarketMakerAction().model_dump(),
            "oversight": OversightAction().model_dump(),
        }
    )
    _, rewards, _, info = env.step(
        {
            "trader_0": make_sell_action(strike=2, maturity=1),
            "market_maker": MarketMakerAction().model_dump(),
            "oversight": OversightAction(
                flagged_agents=["trader_0"],
                flag_type="wash_trading",
                fine_amount=50.0,
                confidence=0.9,
                intervention_type="fine",
                reasoning="wash_trading by trader_0 due to rapid reversal on same strike",
            ).model_dump(),
        }
    )

    assert info["detected_manipulations"]["trader_0"] == "wash_trading"
    assert rewards["oversight"] > 0.0
    assert env.agent_states["trader_0"].fines_received == 50.0


def test_oversight_reward_prefers_correct_restraint():
    env = MultiAgentVSREnvironment()
    env.reset(seed=9)
    _, rewards, _, info = env.step(
        {
            "market_maker": MarketMakerAction().model_dump(),
            "oversight": OversightAction(
                flagged_agents=[],
                flag_type="none",
                confidence=0.1,
                reasoning="No harmful behavior detected.",
            ).model_dump(),
        }
    )

    assert rewards["oversight"] >= 0.0
    assert info["detected_manipulations"] == {}


def test_full_episode_runs_to_completion():
    env = MultiAgentVSREnvironment()
    env.reset(seed=3)
    done = False
    steps = 0

    while not done:
        _, _, done, _ = env.step(
            {
                "market_maker": MarketMakerAction().model_dump(),
                "oversight": OversightAction().model_dump(),
            }
        )
        steps += 1

    assert steps == EPISODE_LENGTH
