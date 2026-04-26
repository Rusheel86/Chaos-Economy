"""Train oversight agent to detect emergent collusion patterns.

This oversight agent learns to detect:
1. Gamma Squeezes - Multiple traders targeting same OTM strikes
2. Wash Trading Rings - Circular trades between specific agents
3. Coordinated Pressure - Synchronized buying/selling patterns
4. Spoofing Patterns - Large orders followed by reversals

The training uses episodes where trained traders execute strategies,
and oversight learns to flag the manipulation patterns.

Usage:
    # First train traders with collusion
    !python train_emergent_collusion.py --num_traders 3 --num_episodes 100

    # Then train oversight to detect
    !python train_oversight_collusion.py --num_episodes 128
"""

import argparse
import json
import os
import sys
from pathlib import Path
import re
import torch
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Clone repo if needed
if not Path("multi_agent").exists() and not Path("Meta/multi_agent").exists():
    print("Cloning Meta repo...")
    os.system("git clone --branch Agentic-AI https://github.com/manan-tech/Meta.git")
    os.chdir("Meta")
elif Path("Meta/multi_agent").exists() and not Path("multi_agent").exists():
    os.chdir("Meta")

sys.path.insert(0, ".")

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from multi_agent.config import NUM_TRADERS, EPISODE_LENGTH
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction, AgentRole
from multi_agent.manipulation_detector import ManipulationDetector

JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


# ============================================================================
# COLLUSION GROUND TRUTH DETECTOR
# ============================================================================

class CollusionGroundTruth:
    """Provides ground truth labels for oversight training.

    This is what oversight learns to detect - the actual manipulation
    patterns that emerge from multi-agent dynamics.
    """

    def __init__(self):
        self.strike_targets = defaultdict(lambda: {"agents": [], "total_qty": 0, "option_types": set()})
        self.agent_trade_pairs = defaultdict(set)  # (agent_a, agent_b, strike, type) -> count
        self.price_pressure_history = []

    def update_from_episode(self, agent_states: Dict, trade_log: List[Dict]):
        """Analyze episode for collusion patterns."""
        # Reset for new analysis
        self.strike_targets = defaultdict(lambda: {"agents": [], "total_qty": 0, "option_types": set()})

        # Build strike concentration map
        for agent_id, state in agent_states.items():
            if not hasattr(state, 'positions') or state.role == AgentRole.OVERSIGHT:
                continue

            for pos in state.positions:
                strike = pos.get("selected_strike", -1)
                qty = abs(pos.get("quantity", 0))
                opt_type = pos.get("option_type", "call")

                if strike >= 0 and qty > 0:
                    self.strike_targets[strike]["agents"].append(agent_id)
                    self.strike_targets[strike]["total_qty"] += qty
                    self.strike_targets[strike]["option_types"].add(opt_type)

    def get_coordinated_agents(self) -> List[Dict]:
        """Return list of coordinated manipulation detected."""
        coordinated = []

        for strike, data in self.strike_targets.items():
            unique_agents = list(set(data["agents"]))

            # Gamma squeeze: 2+ agents on same OTM strike with large volume
            if len(unique_agents) >= 2 and data["total_qty"] > 15 and strike < 4:
                coordinated.append({
                    "agents": unique_agents,
                    "strike": strike,
                    "type": "gamma_squeeze",
                    "severity": data["total_qty"] / 10,  # Higher = more severe
                })

            # Coordinated pressure on any strike
            elif len(unique_agents) >= 3 and data["total_qty"] > 20:
                coordinated.append({
                    "agents": unique_agents,
                    "strike": strike,
                    "type": "coordinated_pressure",
                    "severity": data["total_qty"] / 15,
                })

        return coordinated

    def get_wash_trading_rings(self, trade_log: List[Dict]) -> List[Dict]:
        """Detect wash trading between specific agents."""
        # Track (agent_a, agent_b, strike, option_type) trades
        pair_trades = defaultdict(list)

        for trade in trade_log:
            agent = trade.get("agent_id")
            strike = trade.get("selected_strike")
            opt_type = trade.get("option_type")
            direction = trade.get("direction")
            qty = trade.get("quantity", 0)

            if direction in ["buy", "sell"]:
                pair_trades[(strike, opt_type)].append({
                    "agent": agent,
                    "direction": direction,
                    "qty": qty,
                    "step": trade.get("step", 0),
                })

        rings = []
        for (strike, opt_type), trades in pair_trades.items():
            # Look for opposite direction trades around same time
            buys = [t for t in trades if t["direction"] == "buy"]
            sells = [t for t in trades if t["direction"] == "sell"]

            for buy in buys:
                for sell in sells:
                    if buy["agent"] != sell["agent"] and abs(buy["step"] - sell["step"]) <= 3:
                        rings.append({
                            "agents": [buy["agent"], sell["agent"]],
                            "strike": strike,
                            "type": "wash_trade",
                            "severity": min(buy["qty"], sell["qty"]),
                        })

        return rings


# ============================================================================
# OVERSIGHT PROMPT
# ============================================================================

def format_oversight_prompt(obs, ground_truth: CollusionGroundTruth) -> str:
    """Format oversight prompt with collusion detection context."""

    position_heatmap = {}
    for strike, data in ground_truth.strike_targets.items():
        position_heatmap[strike] = {
            "total_contracts": data["total_qty"],
            "num_agents": len(set(data["agents"])),
        }

    coordinated = ground_truth.get_coordinated_agents()
    wash_rings = ground_truth.get_wash_trading_rings(obs.trade_log or [])

    example_json = json.dumps({
        "flagged_agents": ["trader_0", "trader_1"],
        "flag_type": "coordinated_pressure",
        "fine_amount": 5000.0,
        "halt_strikes": [3],
        "confidence": 0.85,
        "intervention_type": "fine",
        "reasoning": "Agents coordinating gamma squeeze on OTM strike 3.",
    }, separators=(",", ":"))

    prompt = f"""You are the SEC surveillance AI monitoring a multi-agent options market.

## CRITICAL: Your Mission
Detect MARKET MANIPULATION including emergent COLLUSION between traders.

## Manipulation Types to Detect:
1. **Gamma Squeeze** - Multiple traders buying same OTM strike to squeeze market maker
2. **Coordinated Pressure** - 3+ agents targeting same strike simultaneously
3. **Wash Trading Ring** - Two agents trading opposite directions on same instrument
4. **Spoofing** - Large orders quickly reversed

## Current Market Intelligence

### Position Heatmap (Strike -> Total Exposure)
{json.dumps(position_heatmap, indent=1) if position_heatmap else "No positions"}

### All Agent PnLs
{json.dumps(obs.all_agent_pnls, indent=1) if obs.all_agent_pnls else "N/A"}

### Recent Trades (Last 15)
{json.dumps(obs.trade_log[-15:] if obs.trade_log else [], indent=1)}

### Agent Risk Summary
{json.dumps(obs.agent_risk_summary, indent=1) if obs.agent_risk_summary else "N/A"}

## Response Format
Return JSON on ONE line with keys: flagged_agents, flag_type, fine_amount, halt_strikes, confidence, intervention_type, reasoning.

Example: {example_json}

Now analyze the market data and respond:
"""
    return prompt


# ============================================================================
# PARSING
# ============================================================================

def parse_oversight_action(completion) -> Tuple[dict, dict]:
    """Parse oversight action from LLM output."""
    text = completion if isinstance(completion, str) else str(completion)

    for candidate in [text.strip()] + [m.group(1) for m in JSON_CODE_BLOCK_RE.finditer(text)] + [m.group(0) for m in JSON_OBJECT_RE.finditer(text)]:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                continue

            # Validate
            flagged = parsed.get("flagged_agents", [])
            flag_type = str(parsed.get("flag_type", "none"))
            fine = float(parsed.get("fine_amount", 0))
            confidence = float(parsed.get("confidence", 0))
            intervention = str(parsed.get("intervention_type", "none"))
            reasoning = str(parsed.get("reasoning", ""))[:200]

            return {
                "flagged_agents": flagged if isinstance(flagged, list) else [],
                "flag_type": flag_type,
                "fine_amount": max(0, fine),
                "halt_strikes": parsed.get("halt_strikes", []),
                "confidence": min(1.0, max(0.0, confidence)),
                "intervention_type": intervention,
                "reasoning": reasoning,
            }, {"valid": True, "format_reward": 0.5}

        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Fallback
    return OversightAction().model_dump(), {"valid": False, "format_reward": -0.5}


# ============================================================================
# SCRIPTED POLICIES (for running episodes during oversight training)
# ============================================================================

def scripted_trader(i: int, step: int) -> dict:
    strike = (i + step) % 8
    maturity = (i + step) % 3
    direction = "buy" if (i + step) % 2 == 0 else "sell"
    return {
        "selected_strike": strike,
        "selected_maturity": maturity,
        "direction": direction,
        "quantity": 0.5 + ((i + step) % 3) * 0.5,
        "option_type": "call" if i % 2 == 0 else "put",
        "reasoning": f"Scripted {i}",
    }


def scripted_mm(step: int) -> dict:
    if step < 25:
        return MarketMakerAction(atm_spread=0.025, otm_spread=0.045, itm_spread=0.035).model_dump()
    if step < 100:
        return MarketMakerAction(atm_spread=0.04, otm_spread=0.06, itm_spread=0.05).model_dump()
    return MarketMakerAction(atm_spread=0.05, otm_spread=0.07, itm_spread=0.06).model_dump()


# ============================================================================
# OVERSIGHT REWARD FUNCTION
# ============================================================================

def calculate_oversight_reward_for_collusion(
    oversight_action: dict,
    ground_truth: CollusionGroundTruth,
    trade_log: List[Dict],
) -> float:
    """Reward oversight for detecting collusion patterns."""

    reward = 0.0

    # Get actual manipulation
    actual_coordinated = ground_truth.get_coordinated_agents()
    actual_wash = ground_truth.get_wash_trading_rings(trade_log)

    # Build set of actually manipulating agents
    actual_manipulators = set()
    for coord in actual_coordinated:
        actual_manipulators.update(coord["agents"])
    for wash in actual_wash:
        actual_manipulators.update(wash["agents"])

    # True positives: correctly flagged manipulators
    flagged = set(oversight_action.get("flagged_agents", []))
    true_positives = flagged & actual_manipulators
    reward += len(true_positives) * 1.5

    # False positives: incorrectly flagged
    false_positives = flagged - actual_manipulators
    reward -= len(false_positives) * 0.5

    # False negatives: missed manipulators
    false_negatives = actual_manipulators - flagged
    reward -= len(false_negatives) * 1.0

    # Correct flag type
    if actual_coordinated and oversight_action.get("flag_type") in ["coordinated_pressure", "gamma_squeeze"]:
        reward += 0.5
    if actual_wash and oversight_action.get("flag_type") == "wash_trade":
        reward += 0.5

    # Correct intervention
    if actual_manipulators and oversight_action.get("intervention_type") != "none":
        reward += 0.3

    # Reasoning quality
    reasoning = oversight_action.get("reasoning", "").lower()
    if any(agent.lower() in reasoning for agent in flagged):
        reward += 0.2

    return max(-5.0, min(5.0, reward))


# ============================================================================
# TRAINING
# ============================================================================

def train_oversight(args):
    """Train oversight to detect emergent collusion."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"OVERSIGHT TRAINING FOR COLLUSION DETECTION")
    print(f"{'='*70}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
    )

    ground_truth = CollusionGroundTruth()

    def episode_reward(prompts, completions, **kwargs):
        """Run episodes and reward oversight for detecting collusion."""
        rewards = []
        seeds = kwargs.get("seed", list(range(len(completions))))

        for idx, completion in enumerate(completions):
            seed = int(seeds[idx]) if idx < len(seeds) else idx

            # Parse oversight action
            action, parse_info = parse_oversight_action(completion)

            # Run episode with this oversight action
            env = MultiAgentVSREnvironment()
            obs = env.reset(seed=seed)

            episode_reward_total = 0.0

            for step in range(args.episode_length):
                # Scripted traders + MM
                actions = {}
                for i in range(NUM_TRADERS):
                    actions[f"trader_{i}"] = scripted_trader(i, step)
                actions["market_maker"] = scripted_mm(step)
                actions["oversight"] = action  # Use oversight's action

                obs, rewards_dict, done, info = env.step(actions)

                # Update ground truth
                ground_truth.update_from_episode(env.agent_states, env.trade_log)

                # Calculate reward for this oversight action
                step_reward = calculate_oversight_reward_for_collusion(
                    action, ground_truth, env.trade_log
                )
                episode_reward_total += step_reward * 0.1  # Discount future steps

                if done:
                    break

            total = episode_reward_total + parse_info.get("format_reward", 0)
            rewards.append(max(-5.0, min(5.0, total)))

        return rewards

    # Build dataset
    env = MultiAgentVSREnvironment()
    prompts = []
    for seed in range(args.num_episodes):
        obs = env.reset(seed=seed)
        ground_truth.update_from_episode(env.agent_states, [])
        prompt = format_oversight_prompt(obs["oversight"], ground_truth)
        prompts.append({"prompt": prompt, "seed": seed})

    dataset = Dataset.from_list(prompts)

    # Train
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=2,
        num_generations=4,
        max_completion_length=250,
        logging_steps=5,
        save_steps=50,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=episode_reward,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    trainer.train()

    # Save
    save_path = Path(args.output_dir) / "oversight_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    print(f"\nOversight adapter saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--episode_length", type=int, default=50)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./oversight_collusion_checkpoints")
    args = parser.parse_args()

    train_oversight(args)


if __name__ == "__main__":
    main()
