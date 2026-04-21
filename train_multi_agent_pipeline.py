"""Full pipeline: Train 9 traders (3 types x 3 clones) + oversight + market_maker.

TRADER ARCHETYPES:
- Aggressive (trader_0, trader_1, trainer_2): High risk, momentum chase
- Neutral (trader_3, trader_4, trader_5): Balanced, moderate positions
- Contrarian (trader_6, trader_7, trader_8): Counter-trend, position limits
- trader_9: Scripted baseline for comparison

TRAINING ORDER:
1. Session 1: Train all 9 traders (3 phases)
2. Session 2: Train oversight (uses trained traders)
3. Session 3: Train market_maker (uses traders + oversight)

Usage on Kaggle:

    # Session 1a: Aggressive Traders
    !python train_multi_agent_pipeline.py --phase traders_aggressive --num_episodes 50

    # Session 1b: Neutral Traders
    !python train_multi_agent_pipeline.py --phase traders_neutral --num_episodes 50

    # Session 1c: Contrarian Traders
    !python train_multi_agent_pipeline.py --phase traders_contrarian --num_episodes 50

    # Session 2: Oversight
    !python train_multi_agent_pipeline.py --phase oversight --num_episodes 64

    # Session 3: Market Maker
    !python train_multi_agent_pipeline.py --phase market_maker --num_episodes 50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import torch

# Clone repo if needed
if not Path("multi_agent").exists() and not Path("Meta/multi_agent").exists():
    print("Cloning Meta repo...")
    git_url = "https://github.com/manan-tech/Meta.git"
    try:
        from kaggle_secrets import UserSecretsClient
        gh_pat = UserSecretsClient().get_secret("GH_PAT")
        if gh_pat:
            git_url = f"https://{gh_pat}@github.com/manan-tech/Meta.git"
            print("Successfully injected GH_PAT from Kaggle secrets.")
    except Exception:
        print("Kaggle secrets not found or GH_PAT missing. Attempting public clone...")
        
    os.system(f"git clone --branch Agentic-AI {git_url}")
    if Path("Meta").exists():
        os.chdir("Meta")
elif Path("Meta/multi_agent").exists() and not Path("multi_agent").exists():
    os.chdir("Meta")

sys.path.insert(0, ".")

# ============================================================================
# TRADER TYPE CONFIGURATIONS
# ============================================================================

TRADER_CONFIGS = {
    "aggressive": {
        "trader_ids": [0, 1, 2],
        "reward_weight": {"pnl": 0.7, "position_quality": 0.1, "risk_penalty": 0.0},
        "temperature": 0.9,
        "description": "Momentum chasers, high risk, gamma squeeze initiators",
    },
    "neutral": {
        "trader_ids": [3, 4, 5],
        "reward_weight": {"pnl": 0.5, "position_quality": 0.3, "risk_penalty": 0.1},
        "temperature": 0.7,
        "description": "Balanced, may join coordinated pressure",
    },
    "contrarian": {
        "trader_ids": [6, 7, 8],
        "reward_weight": {"pnl": 0.4, "position_quality": 0.3, "risk_penalty": 0.2},
        "temperature": 0.6,
        "description": "Counter-trend, exploit manipulation",
    },
}


# ============================================================================
# PROMPT FORMATTERS
# ============================================================================

def format_trader_prompt(trader_type: str, target_agent: str, obs) -> str:
    """Format prompt for trader based on archetype."""
    base = f"""You are {target_agent}, a {trader_type} trader in a multi-agent options market.

## Market State
- Spot: ${obs.spot_price:.2f}
- IV (ATM): {obs.iv_surface[3][0]*100:.1f}%
- Step: {obs.step_number}/300

## Your Portfolio
- PnL: ${obs.own_pnl:.2f}
- Delta: {obs.own_greeks.get('delta', 0):.2f}
- Gamma: {obs.own_greeks.get('gamma', 0):.2f}
- Cash: ${obs.own_cash:.0f}
"""

    if trader_type == "aggressive":
        base += """
## Strategy: AGGRESSIVE MOMENTUM
- Maximize PnL above all else
- Large positions acceptable
- Chase IV moves, join trends
- Strength in numbers - if others buying same strike, consider joining
"""
    elif trader_type == "neutral":
        base += """
## Strategy: BALANCED PORTFOLIO
- Seek profit but manage risk
- Keep delta/gamma balanced
- Join profitable trends moderately
- Moderate position sizes
"""
    else:
        base += """
## Strategy: CONTRARIAN COUNTER-TREND
- Profit from overreactions
- Fade extreme IV moves
- Strict position limits
- If everyone buying, consider selling
"""

    base += """
## Response Format (MANDATORY)
Return a valid JSON object. 
- The 'reasoning' field MUST be a complete, professional sentence.
- NEVER use placeholders like <your explanation>, ..., or [reason].
- Example: {"selected_strike": 4, "selected_maturity": 0, "direction": "buy", "quantity": 10.0, "option_type": "call", "reasoning": "Targeting OTM gamma as spot nears resistance levels."}

Return JSON: {"selected_strike": 0, "selected_maturity": 0, "direction": "str", "quantity": 0.0, "option_type": "str", "reasoning": "str"}
"""
    return base


def format_oversight_prompt(obs, position_heatmap: dict, coordinated_pressure: dict) -> str:
    """Format prompt for oversight agent."""
    return f"""You are the SEC surveillance AI monitoring a multi-agent options market.

## Mission
Detect MARKET MANIPULATION including COLLUSION. Over-regulation kills liquidity; under-regulation causes systemic collapse.

## Strategic Nuance: Collusion vs Herd Behavior
1. **Herd Behavior (Organic)**: Multiple agents buying various strikes because the underlying spot price is trending up. This is legal.
2. **Predatory Convergence (Manipulation)**: Agents targeting the EXACT SAME out-of-the-money (OTM) strike to force a gamma squeeze on the Market Maker.
3. **Suspicious Pattern**: If 3+ agents target Strike {obs.current_strike + 2} specifically while ignoring other profitable strikes, prioritize intervention.

## Market Intelligence
- Position Heatmap (strike -> contracts): {json.dumps(position_heatmap)}
- Coordinated Pressure (strike -> count): {json.dumps(coordinated_pressure)}
- All Agent PnLs: {json.dumps(obs.all_agent_pnls)}

## Recent Trades
{json.dumps(obs.trade_log[-15:] if obs.trade_log else [])}

## Response Format (MANDATORY)
Return a valid JSON object.
- Example: {{"flagged_agents": ["trader_0", "trader_1"], "flag_type": "gamma_squeeze", "fine_amount": 500.0, "halt_strikes": [], "confidence": 0.95, "intervention_type": "fine", "reasoning": "Detected unnatural concentration of OTM calls on strike 5 by agents T0 and T1."}}

Return JSON: {{"flagged_agents": [], "flag_type": "str", "fine_amount": 0.0, "halt_strikes": [], "confidence": 0.0, "intervention_type": "str", "reasoning": "str"}}
"""


def format_mm_prompt(obs, coordinated_pressure: dict) -> str:
    """Format prompt for market maker."""
    return f"""You are the Market Maker in a multi-agent options market.

## Mission
Provide liquidity while managing inventory risk. Watch for gamma squeezes!

## Current Risk
- Your Delta: {obs.own_greeks.get('delta', 0):.2f}
- Your Gamma: {obs.own_greeks.get('gamma', 0):.2f}
- Your PnL: ${obs.own_pnl:.2f}
- Coordinated Pressure Detected: {json.dumps(coordinated_pressure)}

## Pricing Guidelines
- Normal: ATM 0.04, OTM 0.06, ITM 0.05
- Under pressure: Widen spreads to protect inventory
- High gamma risk: Widen further or hedge

## Response Format (MANDATORY)
- NEVER use placeholders like <...> or $X.
- Example: {{"atm_spread": 0.02, "otm_spread": 0.05, "itm_spread": 0.03, "reasoning": "Widening OTM spreads to mitigate gamma exposure from aggressive trader group."}}

    Return JSON: {{"atm_spread": 0.0, "otm_spread": 0.0, "itm_spread": 0.0, "skew_adjustment": 0.0, "reasoning": "str"}}
"""
    return base


# ============================================================================
# JSON PARSING
# ============================================================================

def parse_json(text: str, role: str = "trader") -> tuple:
    """Extract and validate JSON from LLM output."""
    def safe_int(v, default=0):
        try: return int(v) if v is not None else default
        except (ValueError, TypeError): return default

    def safe_float(v, default=0.0):
        try: return float(v) if v is not None else default
        except (ValueError, TypeError): return default

    text = text.strip()
    try:
        parsed = json.loads(text)
    except:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try: parsed = json.loads(match.group())
            except: parsed = {}
        else:
            parsed = {}

    if role == "trader":
        direction = str(parsed.get("direction", "hold")).lower()
        if direction not in ["buy", "sell", "hold"]:
            direction = "hold"
        opt_type = str(parsed.get("option_type", "call")).lower()
        if opt_type not in ["call", "put"]:
            opt_type = "call"
        return {
            "selected_strike": safe_int(parsed.get("selected_strike"), 4),
            "selected_maturity": safe_int(parsed.get("selected_maturity"), 0),
            "direction": direction,
            "quantity": max(0.0, safe_float(parsed.get("quantity"), 0.0)),
            "option_type": opt_type,
            "reasoning": str(parsed.get("reasoning") or "")[:150],
        }, {"valid": len(parsed) > 0}

    elif role == "oversight":
        raw_flagged = parsed.get("flagged_agents") or []
        if not isinstance(raw_flagged, list): raw_flagged = []
        
        # Convert to strings and map indices to trader IDs if needed
        clean_flagged = []
        for x in raw_flagged:
            if isinstance(x, int):
                clean_flagged.append(f"trader_{x}")
            elif isinstance(x, str):
                if x.isdigit():
                    clean_flagged.append(f"trader_{x}")
                else:
                    clean_flagged.append(x)
        
        raw_halts = parsed.get("halt_strikes") or []
        if not isinstance(raw_halts, list): raw_halts = []
        clean_halts = [safe_int(x, -1) for x in raw_halts]
        clean_halts = [x for x in clean_halts if x >= 0]
        
        return {
            "flagged_agents": clean_flagged,
            "flag_type": str(parsed.get("flag_type", "none")),
            "fine_amount": safe_float(parsed.get("fine_amount"), 0.0),
            "halt_strikes": clean_halts,
            "confidence": safe_float(parsed.get("confidence"), 0.0),
            "intervention_type": str(parsed.get("intervention_type", "none")),
            "reasoning": str(parsed.get("reasoning") or "")[:150],
        }, {"valid": len(parsed) > 0}

    elif role == "market_maker":
        return {
            "atm_spread": min(0.15, max(0.01, safe_float(parsed.get("atm_spread"), 0.04))),
            "otm_spread": min(0.20, max(0.01, safe_float(parsed.get("otm_spread"), 0.06))),
            "itm_spread": min(0.15, max(0.01, safe_float(parsed.get("itm_spread"), 0.05))),
            "skew_adjustment": min(0.05, max(-0.05, safe_float(parsed.get("skew_adjustment"), 0.0))),
            "reasoning": str(parsed.get("reasoning") or "")[:100],
        }, {"valid": len(parsed) > 0}

    return {}, {"valid": False}


# ============================================================================
# SCRIPTED POLICIES
# ============================================================================

def scripted_trader(i: int, step: int) -> dict:
    return {
        "selected_strike": (i + step) % 8,
        "selected_maturity": (i + step) % 3,
        "direction": "buy" if (i + step) % 2 == 0 else "sell",
        "quantity": 0.5 + ((i + step) % 3) * 0.5,
        "option_type": "call" if i % 2 == 0 else "put",
        "reasoning": f"Scripted trader_{i}",
    }


def scripted_mm(step: int) -> dict:
    if step < 50:
        return {"atm_spread": 0.03, "otm_spread": 0.05, "itm_spread": 0.04, "skew_adjustment": 0.0, "reasoning": "Normal"}
    return {"atm_spread": 0.05, "otm_spread": 0.07, "itm_spread": 0.06, "skew_adjustment": 0.0, "reasoning": "Wider"}


def scripted_oversight() -> dict:
    return {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "halt_strikes": [],
        "confidence": 0.0,
        "intervention_type": "none",
        "reasoning": "Baseline no detection",
    }


# ============================================================================
# COLLUSION DETECTION (ground truth for oversight training)
# ============================================================================

def detect_coordinated_pressure(agent_states: dict) -> dict:
    """Detect if multiple traders targeting same strikes."""
    strike_concentration = defaultdict(lambda: {"agents": [], "total_qty": 0})

    for agent_id, state in agent_states.items():
        if not hasattr(state, 'positions') or not agent_id.startswith("trader"):
            continue
        for pos in state.positions:
            strike = pos.get("selected_strike", -1)
            qty = abs(pos.get("quantity", 0))
            if strike >= 0 and qty > 0:
                strike_concentration[strike]["agents"].append(agent_id)
                strike_concentration[strike]["total_qty"] += qty

    coordinated = {}
    for strike, data in strike_concentration.items():
        unique_agents = list(set(data["agents"]))
        if len(unique_agents) >= 3 and data["total_qty"] > 50:
            coordinated[strike] = {
                "agents": unique_agents,
                "total_contracts": data["total_qty"],
                "type": "gamma_squeeze" if strike < 4 else "coordinated_pressure",
            }
    return coordinated


def get_position_heatmap(agent_states: dict) -> dict:
    """Total contracts per strike."""
    heatmap = defaultdict(int)
    for agent_id, state in agent_states.items():
        if not hasattr(state, 'positions') or not agent_id.startswith("trader"):
            continue
        for pos in state.positions:
            strike = pos.get("selected_strike", -1)
            qty = abs(pos.get("quantity", 0))
            if strike >= 0:
                heatmap[strike] += qty
    return dict(sorted(heatmap.items()))


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

import random

def train_unified_model(args):
    """Train a single unified model for all agents."""
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from multi_agent.environment import MultiAgentVSREnvironment

    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED MULTI-AGENT MODEL")
    print(f"Roles: Traders (Aggressive, Neutral, Contrarian), Market Maker, Oversight")
    print(f"{'='*70}\n")

    # Increase LoRA rank to 64 to prevent confusion between multiple roles
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model, max_seq_length=2048, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, 
        r=64,  # Increased capacity for mult-task multi-agent
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=64, 
        lora_dropout=0,
    )

    env = MultiAgentVSREnvironment()
    
    # Build a unified dataset containing prompts for all roles
    prompts = []
    print("Building unified dataset...")
    for seed in range(args.num_episodes):
        obs = env.reset(seed=seed)
        
        # Fast-forward to a random step so the agent's portfolio isn't always empty
        ff_steps = random.randint(0, args.episode_length - 5)
        if ff_steps > 0:
            for step in range(ff_steps):
                actions = {}
                for i in range(10):
                    actions[f"trader_{i}"] = scripted_trader(i, step)
                actions["market_maker"] = scripted_mm(step)
                actions["oversight"] = scripted_oversight()
                obs, r, done, _ = env.step(actions)
                if done:
                    break
                    
        # In case it hit an early termination during fast-forward (bankrupt), reset it
        if done:
            obs = env.reset(seed=seed)
            ff_steps = 0
        
        # 1. Add Traders
        prompts.append({
            "prompt": format_trader_prompt("aggressive", "trader_0", obs["trader_0"]),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_0", "archetype": "aggressive", "ff_steps": ff_steps
        })
        prompts.append({
            "prompt": format_trader_prompt("neutral", "trader_3", obs["trader_3"]),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_3", "archetype": "neutral", "ff_steps": ff_steps
        })
        prompts.append({
            "prompt": format_trader_prompt("contrarian", "trader_6", obs["trader_6"]),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_6", "archetype": "contrarian", "ff_steps": ff_steps
        })
        
        # 2. Add Market Maker
        pressure = detect_coordinated_pressure(env.agent_states)
        prompts.append({
            "prompt": format_mm_prompt(obs["market_maker"], pressure),
            "seed": seed, "agent_role": "market_maker", "agent_id": "market_maker", "archetype": "none", "ff_steps": ff_steps
        })
        
        # 3. Add Oversight
        heatmap = get_position_heatmap(env.agent_states)
        prompts.append({
            "prompt": format_oversight_prompt(obs["oversight"], heatmap, pressure),
            "seed": seed, "agent_role": "oversight", "agent_id": "oversight", "archetype": "none", "ff_steps": ff_steps
        })

    dataset = Dataset.from_list(prompts)
    print(f"Dataset created with {len(dataset)} examples.")

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        seeds = kwargs.get("seed", list(range(len(completions))))
        agent_roles = kwargs.get("agent_role", ["trader"] * len(completions))
        agent_ids = kwargs.get("agent_id", ["trader_0"] * len(completions))
        archetypes = kwargs.get("archetype", ["aggressive"] * len(completions))
        ff_steps_list = kwargs.get("ff_steps", [0] * len(completions))

        for idx, completion in enumerate(completions):
            role = agent_roles[idx]
            agent_id = agent_ids[idx]
            archetype = archetypes[idx]
            seed = int(seeds[idx]) if idx < len(seeds) else idx
            ff_steps = int(ff_steps_list[idx])

            action, parse_info = parse_json(str(completion), role)
            
            # Massive structural penalty for bad JSON formats = saves training time
            if not parse_info.get("valid", False):
                rewards.append(-5.0)
                continue

            env = MultiAgentVSREnvironment()
            obs = env.reset(seed=seed)
            
            # 1. Fast-forward the environment to recreate the exact state seen in the prompt
            for step in range(ff_steps):
                actions = {}
                for i in range(10):
                    actions[f"trader_{i}"] = scripted_trader(i, step)
                actions["market_maker"] = scripted_mm(step)
                actions["oversight"] = scripted_oversight()
                obs, r, done, _ = env.step(actions)
                if done:
                    break

            if done:
                rewards.append(0.0)
                continue
                
            # 2. Execute the LLM's action for EXACTLY ONE STEP (so it learns step-by-step logic)
            step = ff_steps
            actions = {}
            for i in range(10):
                actions[f"trader_{i}"] = scripted_trader(i, step)
            actions["market_maker"] = scripted_mm(step)
            actions["oversight"] = scripted_oversight()
            
            # Override target agent with the LLM's action
            actions[agent_id] = action

            obs, r, done, _ = env.step(actions)
            
            total_reward = 0.0
            
            # SCORE: TRADER (Single step PnL & Risk evaluation)
            if role == "trader":
                weights = TRADER_CONFIGS[archetype]["reward_weight"]
                final_state = env.agent_states[agent_id]
                pos_penalty = 0.0
                
                # Active penalization for hoarding directional risk
                if abs(final_state.portfolio_delta) > 8:
                    pos_penalty = -0.5
                if abs(final_state.portfolio_delta) > 15:
                    pos_penalty = -2.0  # Harsher penalty to force flattening
                    
                total_reward += r.get(agent_id, 0) * weights["pnl"] + pos_penalty * weights["risk_penalty"]
            
            # SCORE: MARKET MAKER
            elif role == "market_maker":
                mm_reward = r.get("market_maker", 0)
                mm_state = env.agent_states["market_maker"]
                greeks_penalty = 0.0
                if abs(mm_state.portfolio_gamma) > 5:
                    greeks_penalty = -1.0
                if abs(mm_state.portfolio_delta) > 10:
                    greeks_penalty -= 0.5
                total_reward += mm_reward + greeks_penalty
            
            # SCORE: OVERSIGHT
            elif role == "oversight":
                coordinated = detect_coordinated_pressure(env.agent_states)
                actual_manipulators = set()
                for data in coordinated.values():
                    actual_manipulators.update(data["agents"])

                flagged = set(action.get("flagged_agents", []))
                true_positives = len(flagged & actual_manipulators)
                false_positives = len(flagged - actual_manipulators)
                total_reward += true_positives * 1.5 - false_positives * 1.0 # Harsher penalty for false positives

            # Scale and clip reward heavily to keep GRPO stable
            if role == "trader":
                scaled = total_reward * 0.1 # Dampen massive immediate PnL swings
            else:
                scaled = total_reward * 0.1
                
            rewards.append(max(-5.0, min(5.0, scaled)))

        return rewards

    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}/unified_v1",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=2,
        num_generations=4,
        max_completion_length=200,
        logging_steps=5,
        save_steps=100,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model, args=training_args, reward_funcs=reward_fn,
        processing_class=tokenizer, train_dataset=dataset,
    )
    trainer.train()

    save_path = Path(args.output_dir) / "unified_market_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Saved Unified Model to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train multi-agent system")
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--episode_length", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./multi_agent_checkpoints")
    args = parser.parse_args()

    # Now we just run the unified training cycle once!
    train_unified_model(args)

if __name__ == "__main__":
    main()
