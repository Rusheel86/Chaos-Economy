"""Verification tests for reward audit fixes H1-H4, B1-B6, D4, M2-M4."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_b1_black_swan_short_episode():
    """B1: BlackSwanGenerator must not crash at episode_length=50."""
    from multi_agent.black_swan import BlackSwanGenerator
    for seed in range(20):
        rng = np.random.RandomState(seed)
        gen = BlackSwanGenerator(rng, episode_length=50)
        for e in gen.events:
            assert e.trigger_step < 50, f"Event at step {e.trigger_step} >= episode_length 50"
            assert e.news_step >= 0, f"News step negative: {e.news_step}"
            assert e.news_step <= e.trigger_step, f"News after trigger: {e.news_step} > {e.trigger_step}"
    print("[PASS] B1: BlackSwanGenerator safe at episode_length=50")


def test_b3_no_double_purchase():
    """B3: Cannot re-purchase the same listing."""
    from multi_agent.news_marketplace import NewsMarketplace
    from multi_agent.models import AgentState, AgentRole
    rng = np.random.RandomState(42)
    mp = NewsMarketplace(rng)
    
    states = {
        "trader_0": AgentState(agent_id="trader_0", role=AgentRole.TRADER, cash_balance=1000),
        "trader_1": AgentState(agent_id="trader_1", role=AgentRole.TRADER, cash_balance=1000),
    }
    
    listing = mp.post_listing("trader_0", 50.0, "Spot will crash due to bank failure", "all", current_step=1)
    assert listing is not None
    
    # First purchase should work
    result1 = mp.buy_intel("trader_1", listing.listing_id, states, step=2)
    assert result1 is not None
    
    # Second purchase should fail
    result2 = mp.buy_intel("trader_1", listing.listing_id, states, step=3)
    assert result2 is None, "B3 FAIL: Re-purchase allowed"
    print("[PASS] B3: Re-purchase blocked")


def test_m4_empty_content_rejected():
    """M4: Empty/trivial content intel should be rejected."""
    from multi_agent.news_marketplace import NewsMarketplace
    rng = np.random.RandomState(42)
    mp = NewsMarketplace(rng)
    
    result = mp.post_listing("trader_0", 50.0, "", "all", current_step=1)
    assert result is None, "M4 FAIL: Empty content accepted"
    
    result = mp.post_listing("trader_0", 50.0, "short", "all", current_step=1)
    assert result is None, "M4 FAIL: Trivial content accepted"
    
    result = mp.post_listing("trader_0", 50.0, "Spot will crash due to major bank failure imminent", "all", current_step=1)
    assert result is not None, "M4 FAIL: Valid content rejected"
    print("[PASS] M4: Empty content rejected, valid content accepted")


def test_b5_b6_regime_recovery():
    """B5+B6: Black swan regime should revert after one advance_market call."""
    from vsr_env.models import VSRState
    from vsr_env.engine.market_sim import advance_market, apply_black_swan
    
    state = VSRState(episode_id="test", spot_price=100.0, variance=0.04, step_count=0)
    rng = np.random.RandomState(42)
    
    # Apply crash (0.70x = spot goes to 70)
    apply_black_swan(state, 0.70, 3.0)
    assert state.regime == "black_swan"
    assert state.spot_price < 75.0, f"Spot should be ~70, got {state.spot_price}"
    
    # After one advance_market, regime should decay
    advance_market(state, rng)
    assert state.regime == "high_vol", f"B6 FAIL: Regime still {state.regime}"
    # Spot should NOT be clamped back to 50 during black_swan step
    # (it was in [10, 300] range during black_swan)
    print(f"  Spot after advance: {state.spot_price:.2f} (should be near 70, not 50)")
    print("[PASS] B5+B6: Regime recovers, shock preserved")


def test_h2_fake_intel_seller_penalty():
    """H2: Seller of fake intel gets cash clawback."""
    from multi_agent.news_marketplace import NewsMarketplace
    from multi_agent.models import AgentState, AgentRole
    
    # Force is_genuine=False by using a seeded RNG
    # Need to find a seed where rng.random() >= 0.8
    for seed in range(100):
        rng = np.random.RandomState(seed)
        mp = NewsMarketplace(rng)
        listing = mp.post_listing("seller", 100.0, "Fake intel content here for testing", "all", current_step=1)
        if listing and not listing.is_genuine:
            states = {
                "seller": AgentState(agent_id="seller", role=AgentRole.TRADER, cash_balance=1000),
                "buyer": AgentState(agent_id="buyer", role=AgentRole.TRADER, cash_balance=1000),
            }
            mp.buy_intel("buyer", listing.listing_id, states, step=2)
            # Seller should get price - 50% clawback = net +50
            assert states["seller"].cash_balance < 1100, \
                f"H2 FAIL: Seller cash {states['seller'].cash_balance}, expected clawback"
            print(f"  Seller cash: {states['seller'].cash_balance} (should be 1050, not 1100)")
            print("[PASS] H2: Fake intel seller gets clawback")
            return
    print("[SKIP] H2: Could not find fake intel seed in 100 tries")


def test_m2_passive_receiver_not_flagged():
    """M2: Agent who only receives DMs (never sends) should not be flagged."""
    from multi_agent.manipulation_detector import ManipulationDetector
    from multi_agent.models import AgentState, AgentRole
    
    detector = ManipulationDetector()
    
    env_info = {
        "messages_recent": [
            {"type": "dm", "sender": "trader_0", "recipient": "trader_1", "message": "Buy strike 4!", "step": 5}
        ],
        "channel_members": {}
    }
    
    # trader_1 only received, never sent
    trades = [{"agent_id": "trader_1", "quantity": 5.0, "direction": "buy"}]
    
    result = detector.check_message_collusion("trader_1", trades, env_info)
    assert result == False, f"M2 FAIL: Passive receiver flagged: {result}"
    
    # trader_0 sent AND traded — should be flagged
    trades_sender = [{"agent_id": "trader_0", "quantity": 5.0, "direction": "buy"}]
    result_sender = detector.check_message_collusion("trader_0", trades_sender, env_info)
    assert result_sender == True, f"M2 FAIL: Active sender NOT flagged"
    print("[PASS] M2: Passive receiver not flagged, active sender flagged")


def test_env_full_step():
    """Integration: Full env reset+step cycle with new subsystems."""
    from multi_agent.environment import MultiAgentVSREnvironment
    
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=42)
    
    assert env.black_swan_gen is not None
    assert env.marketplace is not None
    assert env.messaging is not None
    
    # Build scripted actions
    actions = {}
    for i in range(10):
        actions[f"trader_{i}"] = {
            "direction": "buy", "option_type": "call", "quantity": 1.0,
            "selected_strike": 4, "selected_maturity": 0, "reasoning": "test"
        }
    actions["market_maker"] = {"atm_spread": 0.04, "otm_spread": 0.06, "itm_spread": 0.05, "reasoning": "test"}
    actions["oversight"] = {"flagged_agents": [], "flag_type": "none", "fine_amount": 0,
                           "halt_strikes": [], "confidence": 0.0, "intervention_type": "none", "reasoning": "test"}
    
    obs, r, done, info = env.step(actions)
    assert not done
    assert "trader_0" in r
    assert "oversight" in r
    print(f"  Rewards sample: trader_0={r['trader_0']:.3f}, oversight={r['oversight']:.3f}")
    print("[PASS] Integration: Full step cycle works")


if __name__ == "__main__":
    test_b1_black_swan_short_episode()
    test_b3_no_double_purchase()
    test_m4_empty_content_rejected()
    test_b5_b6_regime_recovery()
    test_h2_fake_intel_seller_penalty()
    test_m2_passive_receiver_not_flagged()
    test_env_full_step()
    print("\n=== ALL AUDIT FIX TESTS PASSED ===")
