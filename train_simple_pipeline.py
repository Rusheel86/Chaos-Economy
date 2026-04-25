"""Simple pipeline without unsloth/GRPO dependencies.

Uses standard transformers + PEFT for training.
Compatible with Kaggle T4 GPUs.

Usage on Kaggle:

    !pip install transformers peft datasets accelerate torch -q

    !python train_simple_pipeline.py --phase traders_aggressive --num_episodes 50
    !python train_simple_pipeline.py --phase traders_neutral --num_episodes 50
    !python train_simple_pipeline.py --phase traders_contrarian --num_episodes 50
    !python train_simple_pipeline.py --phase oversight --num_episodes 64
    !python train_simple_pipeline.py --phase market_maker --num_episodes 50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Clone repo if needed
if not Path("multi_agent").exists() and not Path("Meta/multi_agent").exists():
    print("Cloning Meta repo...")
    os.system("git clone --branch Agentic-AI https://github.com/manan-tech/Meta.git")
    os.chdir("Meta")
elif Path("Meta/multi_agent").exists() and not Path("multi_agent").exists():
    os.chdir("Meta")

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# TRADER TYPE CONFIGURATIONS
# ============================================================================

TRADER_CONFIGS = {
    "aggressive": {
        "trader_ids": [0, 1, 2],
        "reward_weight": {"pnl": 0.7, "risk_penalty": 0.0},
        "description": "Momentum chasers, high risk, gamma squeeze initiators",
    },
    "neutral": {
        "trader_ids": [3, 4, 5],
        "reward_weight": {"pnl": 0.5, "risk_penalty": 0.1},
        "description": "Balanced, may join coordinated pressure",
    },
    "contrarian": {
        "trader_ids": [6, 7, 8],
        "reward_weight": {"pnl": 0.4, "risk_penalty": 0.2},
        "description": "Counter-trend, exploit manipulation",
    },
}


# ============================================================================
# PROMPT FORMATTERS
# ============================================================================

def format_trader_prompt(trader_type: str, target_agent: str, obs) -> str:
    base = f"""You are {target_agent}, a {trader_type} trader in a multi-agent options market.

Market State:
- Spot: ${obs.spot_price:.2f}
- IV (ATM): {obs.iv_surface[3][0]*100:.1f}%
- Step: {obs.step_number}/300

Your Portfolio:
- PnL: ${obs.own_pnl:.2f}
- Delta: {obs.own_greeks.get('delta', 0):.2f}
- Cash: ${obs.own_cash:.0f}

Strategy: {"AGGRESSIVE - Maximize PnL, large positions OK, chase trends" if trader_type == "aggressive" else "NEUTRAL - Balance profit and risk, moderate positions" if trader_type == "neutral" else "CONTRARIAN - Counter-trend, strict position limits"}

Return JSON: {{"selected_strike": 0-7, "selected_maturity": 0-2, "direction": "buy"/"sell"/"hold", "quantity": float, "option_type": "call"/"put", "reasoning": "brief"}}
"""
    return base


def format_oversight_prompt(obs, position_heatmap: dict, coordinated_pressure: dict) -> str:
    return f"""You are the SEC surveillance AI. Detect COLLUSION.

Manipulation Types: gamma_squeeze, coordinated_pressure, wash_trade

Position Heatmap: {json.dumps(position_heatmap)}
Coordinated Pressure: {json.dumps(coordinated_pressure)}
PnLs: {json.dumps(obs.all_agent_pnls)}

Return JSON: {{"flagged_agents": [...], "flag_type": "type", "fine_amount": float, "confidence": 0.0-1.0}}
"""


def format_mm_prompt(obs, coordinated_pressure: dict) -> str:
    return f"""You are the Market Maker. Watch for gamma squeezes.

Risk: Delta={obs.own_greeks.get('delta', 0):.2f}, Gamma={obs.own_greeks.get('gamma', 0):.2f}
Pressure: {json.dumps(coordinated_pressure)}

Return JSON: {{"atm_spread": 0.01-0.15, "otm_spread": 0.01-0.20, "itm_spread": 0.01-0.15}}
"""


# ============================================================================
# JSON PARSING
# ============================================================================

def parse_json(text: str, role: str = "trader") -> tuple:
    text = text.strip()
    parsed = {}
    try:
        parsed = json.loads(text)
    except:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except:
                pass

    if role == "trader":
        return {
            "selected_strike": int(parsed.get("selected_strike", 4)),
            "selected_maturity": int(parsed.get("selected_maturity", 0)),
            "direction": str(parsed.get("direction", "hold")).lower(),
            "quantity": max(0, float(parsed.get("quantity", 0))),
            "option_type": str(parsed.get("option_type", "call")).lower(),
            "reasoning": str(parsed.get("reasoning", ""))[:100],
        }, {"valid": len(parsed) > 0}
    elif role == "oversight":
        return {
            "flagged_agents": parsed.get("flagged_agents", []),
            "flag_type": str(parsed.get("flag_type", "none")),
            "fine_amount": float(parsed.get("fine_amount", 0)),
            "confidence": float(parsed.get("confidence", 0)),
        }, {"valid": len(parsed) > 0}
    elif role == "market_maker":
        return {
            "atm_spread": min(0.15, max(0.01, float(parsed.get("atm_spread", 0.04)))),
            "otm_spread": min(0.20, max(0.01, float(parsed.get("otm_spread", 0.06)))),
            "itm_spread": min(0.15, max(0.01, float(parsed.get("itm_spread", 0.05)))),
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
        "reasoning": f"Scripted",
    }


def scripted_mm(step: int) -> dict:
    return {"atm_spread": 0.04, "otm_spread": 0.06, "itm_spread": 0.05}


def scripted_oversight() -> dict:
    return {"flagged_agents": [], "flag_type": "none", "fine_amount": 0, "confidence": 0}


# ============================================================================
# COLLUSION DETECTION
# ============================================================================

def detect_coordinated_pressure(agent_states: dict) -> dict:
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
        if len(set(data["agents"])) >= 2 and data["total_qty"] > 10:
            coordinated[strike] = {"agents": list(set(data["agents"])), "total": data["total_qty"]}
    return coordinated


def get_position_heatmap(agent_states: dict) -> dict:
    heatmap = defaultdict(int)
    for agent_id, state in agent_states.items():
        if not hasattr(state, 'positions') or not agent_id.startswith("trader"):
            continue
        for pos in state.positions:
            heatmap[pos.get("selected_strike", -1)] += abs(pos.get("quantity", 0))
    return dict(sorted(heatmap.items()))


# ============================================================================
# SIMPLE RL TRAINING (Policy Gradient Style)
# ============================================================================

def generate_action(model, tokenizer, prompt: str, device: str) -> str:
    """Generate action from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def compute_reward(model, tokenizer, env, target_agent: str, trader_type: str, episode_length: int, device: str) -> float:
    """Run episode and compute reward."""
    obs = env.reset()
    total_reward = 0.0
    config = TRADER_CONFIGS.get(trader_type, {})

    for step in range(episode_length):
        # Generate action from model
        prompt = format_trader_prompt(trader_type, target_agent, obs[target_agent])
        action_text = generate_action(model, tokenizer, prompt, device)
        action, parse_info = parse_json(action_text, "trader")

        # Build all actions
        actions = {target_agent: action}
        for i in range(10):
            if f"trader_{i}" != target_agent:
                actions[f"trader_{i}"] = scripted_trader(i, step)
        actions["market_maker"] = scripted_mm(step)
        actions["oversight"] = scripted_oversight()

        # Step
        obs, rewards, done, _ = env.step(actions)
        total_reward += rewards.get(target_agent, 0)

        if done:
            break

    # Apply type-specific weighting
    weights = config.get("reward_weight", {"pnl": 0.5})
    final_state = env.agent_states[target_agent]
    pos_penalty = -0.5 if abs(final_state.portfolio_delta) > 8 else 0.0

    weighted = total_reward * weights["pnl"] + pos_penalty * weights.get("risk_penalty", 0)
    return weighted


def train_trader_simple(args, trader_type: str):
    """Train trader using simple policy gradient."""
    from multi_agent.environment import MultiAgentVSREnvironment

    config = TRADER_CONFIGS[trader_type]
    trader_ids = config["trader_ids"]

    print(f"\n{'='*70}")
    print(f"TRAINING {trader_type.upper()} TRADERS: {['trader_'+str(i) for i in trader_ids]}")
    print(f"{'='*70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for trader_id in trader_ids:
        print(f"\n--- Training trader_{trader_id} ---")

        target_agent = f"trader_{trader_id}"

        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Add LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Simple training loop
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        env = MultiAgentVSREnvironment()

        best_reward = -float('inf')

        for episode in range(args.num_episodes):
            model.train()

            # Compute reward
            reward = compute_reward(model, tokenizer, env, target_agent, trader_type, args.episode_length, device)

            # Simple policy gradient: generate, then reinforce
            obs = env.reset(seed=episode)
            prompt = format_trader_prompt(trader_type, target_agent, obs[target_agent])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

            # Target: good reward -> increase likelihood
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Modulate loss by reward (simple REINFORCE-style)
            reward_tensor = torch.tensor(reward, device=device)
            scaled_loss = loss * (-reward_tensor / 5.0)  # Scale by normalized reward

            scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if reward > best_reward:
                best_reward = reward

            if episode % 5 == 0:
                print(f"  Episode {episode}: reward={reward:.3f}, loss={loss.item():.3f}")

        # Save
        save_path = Path(args.output_dir) / f"{target_agent}_lora"
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"✓ Saved: {save_path} (best reward: {best_reward:.3f})")


def train_oversight_simple(args):
    """Train oversight agent."""
    from multi_agent.environment import MultiAgentVSREnvironment

    print(f"\n{'='*70}")
    print(f"TRAINING OVERSIGHT AGENT")
    print(f"{'='*70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    env = MultiAgentVSREnvironment()

    for episode in range(args.num_episodes):
        obs = env.reset(seed=episode)
        heatmap = get_position_heatmap(env.agent_states)
        pressure = detect_coordinated_pressure(env.agent_states)

        prompt = format_oversight_prompt(obs["oversight"], heatmap, pressure)
        action_text = generate_action(model, tokenizer, prompt, device)
        action, _ = parse_json(action_text, "oversight")

        # Reward for detecting coordination
        actual_manipulators = set()
        for data in pressure.values():
            actual_manipulators.update(data["agents"])

        flagged = set(action.get("flagged_agents", []))
        tp = len(flagged & actual_manipulators)
        fp = len(flagged - actual_manipulators)
        reward = tp * 1.5 - fp * 0.5

        # Train
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        scaled_loss = outputs.loss * (-torch.tensor(reward, device=device) / 5.0)

        scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if episode % 10 == 0:
            print(f"  Episode {episode}: reward={reward:.3f}")

    save_path = Path(args.output_dir) / "oversight_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Saved: {save_path}")


def train_market_maker_simple(args):
    """Train market maker."""
    from multi_agent.environment import MultiAgentVSREnvironment

    print(f"\n{'='*70}")
    print(f"TRAINING MARKET MAKER")
    print(f"{'='*70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    env = MultiAgentVSREnvironment()

    for episode in range(args.num_episodes):
        obs = env.reset(seed=episode)
        pressure = detect_coordinated_pressure(env.agent_states)

        prompt = format_mm_prompt(obs["market_maker"], pressure)
        action_text = generate_action(model, tokenizer, prompt, device)
        action, _ = parse_json(action_text, "market_maker")

        # Run episode
        total_reward = 0.0
        for step in range(args.episode_length):
            actions = {}
            for i in range(10):
                actions[f"trader_{i}"] = scripted_trader(i, step)
            actions["market_maker"] = action
            actions["oversight"] = scripted_oversight()

            obs, r, done, _ = env.step(actions)
            total_reward += r.get("market_maker", 0)

            # Penalty for extreme greeks
            mm_state = env.agent_states["market_maker"]
            if abs(mm_state.portfolio_gamma) > 5:
                total_reward -= 0.5

            if done:
                break

        # Train
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        scaled_loss = outputs.loss * (-torch.tensor(total_reward, device=device) / 5.0)

        scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if episode % 10 == 0:
            print(f"  Episode {episode}: reward={total_reward:.3f}")

    save_path = Path(args.output_dir) / "market_maker_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train multi-agent system (simple version)")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["traders_aggressive", "traders_neutral", "traders_contrarian",
                                 "oversight", "market_maker"])
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_length", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.phase == "traders_aggressive":
        train_trader_simple(args, "aggressive")
    elif args.phase == "traders_neutral":
        train_trader_simple(args, "neutral")
    elif args.phase == "traders_contrarian":
        train_trader_simple(args, "contrarian")
    elif args.phase == "oversight":
        train_oversight_simple(args)
    elif args.phase == "market_maker":
        train_market_maker_simple(args)


if __name__ == "__main__":
    main()
