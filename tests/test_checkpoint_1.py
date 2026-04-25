import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_agent.environment import MultiAgentVSREnvironment

def test_checkpoint_1():
    print("Initializing environment...")
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=42)
    
    print("\n--- Testing P0: Black Swan Generator ---")
    events = env.black_swan_gen.events
    assert len(events) > 0, "No Black Swan events generated."
    print(f"Generated {len(events)} events.")
    for e in events:
        print(f"  Event '{e.headline}' at step {e.trigger_step} (news at {e.news_step})")
        assert e.trigger_step > e.news_step, "News must precede trigger."

    print("\n--- Testing P1: News in Observations ---")
    first_event = events[0]
    
    # Fast forward to news step
    print(f"Fast forwarding to news step {first_event.news_step}...")
    for _ in range(first_event.news_step - 1):
        # Dummy actions
        actions = {"market_maker": {"atm_spread": 0.04, "otm_spread": 0.05, "itm_spread": 0.04}}
        for i in range(10):
            actions[f"trader_{i}"] = {"direction": "hold"}
        obs, _, done, _ = env.step(actions)
        if done: break
        
    print(f"At step {env.current_step}, checking for news...")
    # Step into news step
    actions = {"market_maker": {"atm_spread": 0.04}}
    for i in range(10):
        actions[f"trader_{i}"] = {"direction": "hold"}
    obs, _, _, _ = env.step(actions)
    
    # The news headline should be visible in observation
    assert obs["trader_0"].news_headline == first_event.headline, f"Expected '{first_event.headline}', got '{obs['trader_0'].news_headline}'"
    print("SUCCESS: News headline appeared in observation.")

    print("\n--- Testing P2: News Marketplace ---")
    actions = {"market_maker": {"atm_spread": 0.04}}
    for i in range(10):
        actions[f"trader_{i}"] = {"direction": "hold"}
    # Trader 1 sells intel
    actions["trader_1"] = {"direction": "hold", "sell_intel": {"content": "Tech breakthrough!", "price": 10.0}}
    
    obs, _, _, _ = env.step(actions)
    
    # Check if intel was listed
    listings = obs["trader_0"].market_stats.get("available_intel_listings", [])
    assert len(listings) > 0, "Intel listing not found in market stats."
    listing_id = listings[0]["listing_id"]
    print(f"SUCCESS: Intel listed with ID: {listing_id}")
    
    # Trader 2 buys the intel
    actions = {"market_maker": {"atm_spread": 0.04}}
    for i in range(10):
        actions[f"trader_{i}"] = {"direction": "hold"}
    actions["trader_2"] = {"direction": "hold", "buy_intel": listing_id}
    
    obs, _, _, _ = env.step(actions)
    
    # Check if Trader 2 received it
    private_intel = obs["trader_2"].private_intel
    assert len(private_intel) > 0, "Trader 2 did not receive private intel."
    print("SUCCESS: Intel successfully purchased.")

    print("\n--- Testing P3: Agent Messaging ---")
    # Trader 3 sends a DM to Trader 4
    actions = {"market_maker": {"atm_spread": 0.04}}
    for i in range(10):
        actions[f"trader_{i}"] = {"direction": "hold"}
    actions["trader_3"] = {"direction": "hold", "send_message": {"to": "trader_4", "message": "Let's attack strike 5!"}}
    
    obs, _, _, _ = env.step(actions)
    
    assert len(obs["trader_4"].inbox) > 0, "Trader 4 did not receive DM."
    assert obs["trader_4"].inbox[0]["sender"] == "trader_3"
    print(f"SUCCESS: Trader 4 received DM: {obs['trader_4'].inbox[0]['message']}")
    
    assert len(obs["trader_5"].inbox) == 0, "Trader 5 should not have received the DM."
    print("SUCCESS: DM privacy maintained (Trader 5 did not receive it).")
    
    print("\n--- Testing P4: Oversight Cross-Reference (Manipulation Detection) ---")
    # Trigger manipulation by having Trader 4 act right after receiving message
    # To trigger Message Collusion: we need quantity >= 3.0
    actions = {"market_maker": {"atm_spread": 0.04}}
    for i in range(10):
        actions[f"trader_{i}"] = {"direction": "hold"}
    actions["trader_4"] = {"direction": "buy", "quantity": 4.0, "selected_strike": 5, "selected_maturity": 0, "option_type": "call"}
    
    obs, _, _, info = env.step(actions)
    
    ground_truth = info.get("detected_manipulations", {})
    print("Messages recent:", [m for m in env.messaging.message_log if m["step"] >= env.current_step - 2])
    print("Trade log for step:", [t for t in env.trade_log if t["step"] == env.current_step])
    print("Ground truth:", ground_truth)
    assert ground_truth.get("trader_4") == "message_collusion", f"Expected message_collusion, got {ground_truth.get('trader_4')}"
    print("SUCCESS: Oversight successfully detected message collusion.")
    
    print("\nAll Checkpoint 1 tests passed successfully!")

if __name__ == "__main__":
    test_checkpoint_1()
