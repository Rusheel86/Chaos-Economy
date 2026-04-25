import sys
import numpy as np

from vsr_env.server.vsr_environment import VSREnvironment
from vsr_env.models import VSRAction, TradeDirection, StrategyType, StrategyLeg
from inference import TASKS, MAX_STEPS_PER_TASK, TASK_SEEDS

def get_action_for_task(task_name: str, step: int, env: VSREnvironment, obs_dict: dict) -> dict:
    """Deterministic logic to output the correct JSON for each task."""
    p_delta = obs_dict['portfolio_greeks']['delta']
    p_vega = obs_dict['portfolio_greeks']['vega']
    p_gamma = obs_dict['portfolio_greeks']['gamma']

    # Default hold
    action = {
        "direction": "hold",
        "strike_idx": 4, # ATM
        "maturity_idx": 1,
        "quantity": 0.0,
        "reasoning": "Holding."
    }

    if task_name == "vol_regime_detection":
        iv = obs_dict['iv_surface'][4][1]
        if iv > 0.22:
            regime = "high"
        elif iv < 0.14:
            regime = "low"
        else:
            regime = "normal"
        action["reasoning"] = f"Detected {regime} regime because IV is {iv:.2f}."

    elif task_name == "delta_hedging":
        if abs(p_delta) > 0.05:
            # Simple proportional hedge
            action["direction"] = "buy" if p_delta < 0 else "sell"
            action["quantity"] = min(abs(p_delta)*2.0, 10.0) 
            action["reasoning"] = f"Hedging {p_delta:.2f} delta."

    elif task_name == "vertical_spread":
        if step == 0:
            action = {
                "strategy_type": "vertical_spread",
                "legs": [
                    {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
                    {"strike_idx": 5, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 1.0}
                ],
                "reasoning": "Placing vertical spread."
            }

    elif task_name == "straddle_trading":
        if step == 0:
            action = {
                "strategy_type": "straddle",
                "legs": [
                    {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "buy", "quantity": 1.0},
                    {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "buy", "quantity": 1.0}
                ],
                "reasoning": "Volatility will expand."
            }

    elif task_name == "earnings_vol_crush":
        if step == 0:
            action = {
                "strategy_type": "straddle",
                "legs": [
                    {"strike_idx": 4, "maturity_idx": 1, "option_type": "call", "direction": "sell", "quantity": 2.0},
                    {"strike_idx": 4, "maturity_idx": 1, "option_type": "put", "direction": "sell", "quantity": 2.0}
                ],
                "reasoning": "Selling vega heavily before vol crush event."
            }
        elif step > 11 and abs(p_delta) > 0.05:
            action["direction"] = "buy" if p_delta < 0 else "sell"
            action["quantity"] = min(abs(p_delta)*2.0, 10.0) 
            action["reasoning"] = f"Hedging delta."

    elif task_name == "gamma_scalping":
        if abs(p_delta) > 0.05:
            action["direction"] = "buy" if p_delta < 0 else "sell"
            action["quantity"] = min(abs(p_delta)*2.0, 10.0)
            action["reasoning"] = f"Scalping {p_delta:.2f} delta from gamma shifts."

    elif task_name == "vega_gamma_stress":
        if step < 14:
            # We want to minimize |vega| and |gamma| simultaneously
            # Try a range of singular actions and see which decreases the sum of absolute errors most
            best_action = action
            best_error = abs(p_vega) + abs(p_gamma)
            
            # Simple grid search over standard options
            for s_idx in range(8):
                for m_idx in range(3):
                    for d in ["buy", "sell"]:
                        for q in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                            # Calculate theoretical impact
                            # For approximation we just peek at the state, but we don't have direct access
                            # to individual option greeks. Instead of peeking, let's just make a copy of env state?
                            # Not cheap. Let's just use linear approximation from black scholes
                            spot = obs_dict['spot_price']
                            strike = env.engine.STRIKES[s_idx]
                            mat = env.engine.MATURITIES[m_idx]
                            iv = obs_dict['iv_surface'][s_idx][m_idx]
                            
                            opt_v = env.engine.vega(spot, strike, mat, iv)
                            opt_g = env.engine.gamma(spot, strike, mat, iv)
                            
                            if d == "sell":
                                opt_v = -opt_v
                                opt_g = -opt_g
                                
                            new_v = p_vega + opt_v * q
                            new_g = p_gamma + opt_g * q
                            err = abs(new_v) + abs(new_g)
                            
                            if err < best_error:
                                best_error = err
                                best_action = {
                                    "direction": d,
                                    "strike_idx": s_idx,
                                    "maturity_idx": m_idx,
                                    "quantity": q,
                                    "reasoning": f"Neutralizing v={p_vega:.2f} g={p_gamma:.2f}. Err: {err:.2f}"
                                }
            action = best_action
            
    return action

def main():
    print(f"{'='*60}\nBASELINE EVALUATOR\n{'='*60}")
    
    env = VSREnvironment()
    results = []
    
    for task_name in TASKS:
        max_steps = MAX_STEPS_PER_TASK[task_name]
        seed = TASK_SEEDS[task_name]
        
        obs = env.reset(task_name=task_name, seed=seed)
        
        for step in range(max_steps):
            act_dict = get_action_for_task(task_name, step, env, obs.model_dump())
            
            # Map dict -> VSRAction
            strategy_type = act_dict.get("strategy_type")
            if strategy_type:
                st = StrategyType(strategy_type)
                legs = []
                for lg in act_dict["legs"]:
                    legs.append(StrategyLeg(**lg))
                action = VSRAction(strategy_type=st, legs=legs, reasoning=act_dict["reasoning"])
            else:
                dmap = {"buy": TradeDirection.BUY, "sell": TradeDirection.SELL, "hold": TradeDirection.HOLD}
                action = VSRAction(
                    selected_strike=act_dict["strike_idx"],
                    selected_maturity=act_dict["maturity_idx"],
                    direction=dmap[act_dict["direction"]],
                    quantity=act_dict["quantity"],
                    reasoning=act_dict["reasoning"]
                )
                
            res = env.step(action)
            obs = res["observation"]
            if res["done"]:
                grader_score = res["info"].get("grader_score", 0.0)
                results.append((task_name, grader_score))
                print(f"Task: {task_name:<25} | Score: {grader_score:.4f} | Pass: {'✓' if grader_score >= 0.1 else '✗'}")
                break

    all_pass = all([score >= 0.1 for _, score in results])
    if all_pass:
        print("\nAll 7 Tiers successfully validated and solvable.")
        sys.exit(0)
    else:
        print("\nWarning: Some tiers did not pass threshold.")
        sys.exit(1)

if __name__ == "__main__":
    main()
