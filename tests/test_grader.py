
from vsr_env.server.vsr_environment import TASK_CONFIG
from vsr_env.models import VSRState, VSRObservation, VSRAction

def create_mock_history_and_state(task_name):
    """Utility to create a baseline passing history for tests."""
    state = VSRState()
    history = []
    
    # Generic passing structures
    if task_name == "vol_regime_detection":
        history = [
            {"action": VSRAction(reasoning="high", direction="hold", quantity=0.0, selected_strike=0, selected_maturity=0), "observation": {}}
        ]
    elif task_name == "vertical_spread":
        history = [
            {"action": VSRAction(strategy_type="vertical_spread", direction="buy", quantity=1.0, selected_strike=0, selected_maturity=0)}
        ]
        state.portfolio_pnl = 150.0  # mock profit
    elif task_name == "delta_hedging":
        state.initial_delta = 0.5
        state.portfolio_delta = 0.01  # nicely hedged
        state.regime_shift_step = 2
        history = [{"action": VSRAction(direction="hold", quantity=0.0, selected_strike=0, selected_maturity=0)}]*4
    elif task_name == "straddle_trading":
        history = [
            {"action": VSRAction(strategy_type="straddle", direction="buy", quantity=1.0, selected_strike=0, selected_maturity=0)}
        ]
        state.portfolio_pnl = 200.0
    elif task_name == "earnings_vol_crush":
        state.portfolio_vega = -5.0 # nicely negative before crush
        state.portfolio_delta = 0.02
        history = [{"action": VSRAction(direction="sell", quantity=1.0, selected_strike=0, selected_maturity=0)}] * 5
    elif task_name == "gamma_scalping":
        state.portfolio_delta = 0.03
        state.portfolio_gamma = 1.0
        state.portfolio_pnl = 50.0
        history = [{"action": VSRAction(direction="buy", quantity=1.0, selected_strike=0, selected_maturity=0)}] * 5
    elif task_name == "vega_gamma_stress":
        state.portfolio_vega = 0.02
        state.portfolio_gamma = 0.01
        history = [{"action": VSRAction(direction="sell", quantity=1.0, selected_strike=0, selected_maturity=0)}] * 10
        
    return history, state

def run_tests():
    tasks = [
        "vol_regime_detection",
        "vertical_spread",
        "delta_hedging",
        "straddle_trading",
        "earnings_vol_crush",
        "gamma_scalping",
        "vega_gamma_stress"
    ]
    for task_name in tasks:
        config = TASK_CONFIG[task_name]
        grader = config["grader_class"]()
        
        history, state = create_mock_history_and_state(task_name)
        
        try:
            score = grader.score(history, state)
        except Exception as e:
            print(f"FAILED: Grader {grader.__class__.__name__} failed: {e}")
            sys.exit(1)
            
        assert isinstance(score, float), f"{grader.__class__.__name__} didn't return float."
        assert 0.01 <= score <= 0.99, f"{grader.__class__.__name__} returned {score} which is out of bounds."
        print(f"PASSED: {task_name} grader returned {score}")

if __name__ == "__main__":
    import sys
    run_tests()
