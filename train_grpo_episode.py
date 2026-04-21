"""Episode-level GRPO training for Multi-Agent VSR-Env.

Key difference from train_grpo.py:
- Runs full episode rollouts instead of single-step evaluation
- Rewards based on trajectory-level performance (final PnL, risk management)
- Teaches position management and trade sequencing

Usage:
    python train_grpo_episode.py --role trader --trader_id 0 --num_episodes 32 --episode_length 50
"""

import argparse
import json
import math
from pathlib import Path
import re
import torch
from typing import List, Dict, Any, Tuple

try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
except ImportError as exc:
    FastLanguageModel = None
    GRPOConfig = None
    GRPOTrainer = None
    Dataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from multi_agent.config import NUM_TRADERS, EPISODE_LENGTH
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction

VALID_ROLES = {"trader", "market_maker", "oversight"}
JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)

ROLE_REQUIRED_KEYS = {
    "trader": {"selected_strike", "selected_maturity", "direction", "quantity", "option_type", "reasoning"},
    "market_maker": {"atm_spread", "otm_spread", "itm_spread", "skew_adjustment", "reasoning"},
    "oversight": {"flagged_agents", "flag_type", "fine_amount", "halt_strikes", "confidence", "intervention_type", "reasoning"},
}


def resolve_target_agent(role: str, trader_id: int) -> str:
    if role == "trader":
        return f"trader_{trader_id}"
    return role


def example_action_for_role(role: str) -> dict:
    if role == "trader":
        return {"selected_strike": 3, "selected_maturity": 1, "direction": "buy", "quantity": 1.0, "option_type": "call", "reasoning": "IV looks cheap."}
    if role == "market_maker":
        return {"atm_spread": 0.04, "otm_spread": 0.06, "itm_spread": 0.05, "skew_adjustment": 0.0, "reasoning": "Balanced quotes."}
    return {"flagged_agents": [], "flag_type": "none", "fine_amount": 0.0, "halt_strikes": [], "confidence": 0.1, "intervention_type": "none", "reasoning": "No clear manipulation."}


def format_prompt(role: str, target_agent: str, obs) -> str:
    example_json = json.dumps(example_action_for_role(role), separators=(",", ":"))
    if role == "trader":
        return (
            f"You are {target_agent} in a multi-agent options market. "
            "Return JSON only on a single line with keys: selected_strike, selected_maturity, "
            "direction, quantity, option_type, reasoning. Keep reasoning short.\n"
            f"Example: {example_json}\n"
            f"Observation: {obs.model_dump_json()}"
        )
    if role == "market_maker":
        return (
            "You are the market maker in a multi-agent options market. "
            "Return JSON only on a single line with keys: atm_spread, otm_spread, itm_spread, "
            "skew_adjustment, reasoning. Keep reasoning short.\n"
            f"Example: {example_json}\n"
            f"Observation: {obs.model_dump_json()}"
        )
    return (
        "You are the oversight agent in a multi-agent market surveillance task. "
        "Return JSON only on a single line with keys: flagged_agents, flag_type, fine_amount, "
        "halt_strikes, confidence, intervention_type, reasoning. Keep reasoning short.\n"
        f"Example: {example_json}\n"
        f"Observation: {obs.model_dump_json()}"
    )


def default_action_for_role(role: str) -> dict:
    if role == "trader":
        return {"selected_strike": 4, "selected_maturity": 0, "direction": "hold", "quantity": 0.0, "option_type": "call", "reasoning": "Fallback hold action."}
    if role == "market_maker":
        return {"atm_spread": 0.04, "otm_spread": 0.06, "itm_spread": 0.05, "skew_adjustment": 0.0, "reasoning": "Fallback defensive quotes."}
    return {"flagged_agents": [], "flag_type": "none", "fine_amount": 0.0, "halt_strikes": [], "confidence": 0.0, "intervention_type": "none", "reasoning": "No harmful behavior detected."}


def completion_to_text(completion) -> str:
    if hasattr(completion, "text"):
        completion = completion.text
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def extract_json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in JSON_CODE_BLOCK_RE.finditer(text))
    candidates.extend(match.group(0).strip() for match in JSON_OBJECT_RE.finditer(text))
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def validate_action_dict(action_dict: dict, role: str) -> dict:
    if role == "market_maker":
        return MarketMakerAction(**action_dict).model_dump()
    if role == "oversight":
        return OversightAction(**action_dict).model_dump()

    required_keys = ROLE_REQUIRED_KEYS["trader"]
    missing = required_keys - set(action_dict)
    if missing:
        raise ValueError(f"Missing trader keys: {sorted(missing)}")

    direction = str(action_dict["direction"]).lower()
    option_type = str(action_dict["option_type"]).lower()
    if direction not in {"buy", "sell", "hold"}:
        raise ValueError(f"Invalid direction: {direction}")
    if option_type not in {"call", "put"}:
        raise ValueError(f"Invalid option_type: {option_type}")

    quantity = float(action_dict["quantity"])
    if quantity < 0:
        raise ValueError("Quantity must be non-negative")

    return {
        "selected_strike": int(action_dict["selected_strike"]),
        "selected_maturity": int(action_dict["selected_maturity"]),
        "direction": direction,
        "quantity": quantity,
        "option_type": option_type,
        "reasoning": str(action_dict["reasoning"])[:160],
    }


def parse_json_action(completion, role: str) -> Tuple[dict, dict]:
    text = completion_to_text(completion).strip()
    required_keys = ROLE_REQUIRED_KEYS[role]

    for candidate in extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue

        format_reward = 0.5
        if candidate == text:
            format_reward += 0.25
        if required_keys.issubset(parsed.keys()):
            format_reward += 0.25
        if len(text) <= 500:
            format_reward += 0.25

        try:
            validated = validate_action_dict(parsed, role)
        except Exception:
            return default_action_for_role(role), {"valid_json": True, "validated": False, "format_reward": format_reward - 0.5}

        return validated, {"valid_json": True, "validated": True, "format_reward": format_reward + 0.75}

    return default_action_for_role(role), {"valid_json": False, "validated": False, "format_reward": -0.5}


def scripted_trader(agent_index: int, step: int) -> dict:
    strike = (agent_index + step) % 8
    maturity = (agent_index + step) % 3
    direction = "buy" if (agent_index + step) % 2 == 0 else "sell"
    quantity = 0.5 + ((agent_index + step) % 3) * 0.5
    return {
        "selected_strike": strike,
        "selected_maturity": maturity,
        "direction": direction,
        "quantity": quantity,
        "option_type": "call" if agent_index % 2 == 0 else "put",
        "reasoning": f"Scripted trader_{agent_index} action at step {step}.",
    }


def scripted_market_maker(step: int) -> dict:
    if step < 25:
        return MarketMakerAction(atm_spread=0.025, otm_spread=0.045, itm_spread=0.035).model_dump()
    if step < 100:
        return MarketMakerAction(atm_spread=0.04, otm_spread=0.06, itm_spread=0.05).model_dump()
    return MarketMakerAction(atm_spread=0.05, otm_spread=0.07, itm_spread=0.06).model_dump()


def scripted_oversight() -> dict:
    return OversightAction(
        flagged_agents=[],
        flag_type="none",
        fine_amount=0.0,
        confidence=0.0,
        intervention_type="none",
        reasoning="No harmful behavior detected.",
    ).model_dump()


def run_episode_with_llm(
    model,
    tokenizer,
    role: str,
    target_agent: str,
    seed: int,
    episode_length: int,
    device: str,
) -> Tuple[List[dict], float, dict]:
    """Run a full episode with the LLM making decisions for target_agent.

    Returns:
        - List of (observation, action, step_reward) tuples
        - Final episode reward
        - Episode stats (final PnL, max position, etc.)
    """
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=seed)

    episode_data = []
    total_step_rewards = 0.0

    # Track trajectory stats
    initial_cash = env.agent_states[target_agent].cash_balance if target_agent in env.agent_states else 0.0

    for step in range(episode_length):
        actions = {}

        # Target agent: use LLM
        if role == "trader" and target_agent.startswith("trader"):
            prompt = format_prompt(role, target_agent, obs[target_agent])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action, parse_info = parse_json_action(generated, role)
        elif role == "market_maker":
            prompt = format_prompt(role, "market_maker", obs["market_maker"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action, parse_info = parse_json_action(generated, role)
        elif role == "oversight":
            prompt = format_prompt(role, "oversight", obs["oversight"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action, parse_info = parse_json_action(generated, role)
        else:
            action = default_action_for_role(role)
            parse_info = {"format_reward": 0.0}

        actions[target_agent] = action

        # Scripted other agents
        if role == "trader":
            for i in range(NUM_TRADERS):
                if f"trader_{i}" != target_agent:
                    actions[f"trader_{i}"] = scripted_trader(i, step)
            actions["market_maker"] = scripted_market_maker(step)
            actions["oversight"] = scripted_oversight()
        elif role == "market_maker":
            for i in range(NUM_TRADERS):
                actions[f"trader_{i}"] = scripted_trader(i, step)
            actions["oversight"] = scripted_oversight()
        else:  # oversight
            for i in range(NUM_TRADERS):
                actions[f"trader_{i}"] = scripted_trader(i, step)
            actions["market_maker"] = scripted_market_maker(step)

        # Step environment
        next_obs, rewards, done, info = env.step(actions)

        step_reward = rewards.get(target_agent, 0.0)
        total_step_rewards += step_reward

        episode_data.append({
            "step": step,
            "observation": obs[target_agent].model_dump() if target_agent in obs else {},
            "action": action,
            "step_reward": step_reward,
            "parse_info": parse_info,
        })

        obs = next_obs
        if done:
            break

    # Calculate episode-level reward
    final_state = env.agent_states.get(target_agent)

    if role == "trader":
        # Final PnL as primary reward
        final_pnl = final_state.portfolio_pnl if final_state else 0.0

        # Position management bonus: reward for keeping inventory bounded
        total_contracts = sum(abs(pos.get("quantity", 0)) for pos in final_state.positions) if final_state else 0
        position_quality = max(0, 2.0 - total_contracts / 20.0)  # Bonus if under 40 contracts

        # Greeks quality: reward balanced portfolios
        delta_balance = 0.0 if not final_state else max(0, 1.0 - abs(final_state.portfolio_delta) / 5.0)
        vega_balance = 0.0 if not final_state else max(0, 1.0 - abs(final_state.portfolio_vega) / 10.0)

        # Episode reward: weighted combination
        episode_reward = final_pnl * 0.6 + position_quality * 0.2 + delta_balance * 0.1 + vega_balance * 0.1

        episode_stats = {
            "final_pnl": final_pnl,
            "total_contracts": total_contracts,
            "max_delta": abs(final_state.portfolio_delta) if final_state else 0,
            "position_quality": position_quality,
            "step_reward_sum": total_step_rewards,
        }

    elif role == "market_maker":
        final_pnl = final_state.portfolio_pnl if final_state else 0.0
        trades_facilitated = len(env.trade_log)

        # Inventory control
        inventory_score = max(0, 2.0 - abs(final_state.portfolio_delta) / 3.0) if final_state else 0

        episode_reward = final_pnl * 0.4 + trades_facilitated * 0.02 + inventory_score * 0.3

        episode_stats = {
            "final_pnl": final_pnl,
            "trades_facilitated": trades_facilitated,
            "final_delta": final_state.portfolio_delta if final_state else 0,
        }

    else:  # oversight
        correct_flags = sum(1 for rec in env.intervention_log if rec.get("agent_id"))
        episode_reward = total_step_rewards  # Use accumulated step rewards for oversight

        episode_stats = {
            "total_interventions": len(env.intervention_log),
            "correct_flags": correct_flags,
        }

    # Scale and clamp
    episode_reward = max(-5.0, min(5.0, episode_reward))

    return episode_data, episode_reward, episode_stats


def build_episode_dataset(num_episodes: int, role: str, target_agent: str, episode_length: int):
    """Build dataset with initial observations for episode rollouts."""
    env = MultiAgentVSREnvironment()
    prompts = []
    for seed in range(num_episodes):
        obs = env.reset(seed=seed)
        prompts.append({
            "prompt": format_prompt(role, target_agent, obs[target_agent]),
            "seed": seed,
            "target_agent": target_agent,
            "episode_length": episode_length,
        })
    return Dataset.from_list(prompts)


def episode_reward_fn(prompts, completions, model, tokenizer, role, target_agent, episode_length, device, **kwargs):
    """Reward function that runs full episode rollouts."""
    rewards = []
    seeds = kwargs.get("seed", list(range(len(completions))))

    for idx, completion in enumerate(completions):
        sample_seed = int(seeds[idx]) if idx < len(seeds) else idx

        # Run full episode
        episode_data, episode_reward, stats = run_episode_with_llm(
            model, tokenizer, role, target_agent, sample_seed, episode_length, device
        )

        # Add format rewards from each step
        format_total = sum(d["parse_info"].get("format_reward", 0) for d in episode_data)
        total_reward = episode_reward + format_total * 0.1  # Small format bonus

        rewards.append(max(-5.0, min(5.0, total_reward)))

    return rewards


def run_training(args) -> None:
    if IMPORT_ERROR is not None or not all([FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset]):
        raise ImportError("Missing training dependencies. Install unsloth, trl, datasets, peft, and transformers first.") from IMPORT_ERROR

    if args.role not in VALID_ROLES:
        raise ValueError(f"Unsupported role '{args.role}'. Choose from {sorted(VALID_ROLES)}.")

    target_agent = resolve_target_agent(args.role, args.trader_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Episode-Level GRPO Training")
    print(f"{'='*60}")
    print(f"Role: {args.role}")
    print(f"Target Agent: {target_agent}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Episode Length: {args.episode_length}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

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

    # Build dataset
    dataset = build_episode_dataset(args.num_episodes, args.role, target_agent, args.episode_length)

    # Reward function closure
    def reward_fn(prompts, completions, **kwargs):
        return episode_reward_fn(prompts, completions, model, tokenizer, args.role, target_agent, args.episode_length, device, **kwargs)

    # Training config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        logging_steps=5,
        save_steps=50,
        logging_dir=str(Path(args.output_dir) / "runs"),
        report_to="tensorboard",
        run_name=f"grpo_episode_{target_agent}",
        learning_rate=args.learning_rate,
        ddp_find_unused_parameters=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    if trainer.is_world_process_zero():
        model.save_pretrained(str(Path(args.output_dir) / f"{target_agent}_lora_episode"))
        print(f"\nSaved adapter to: {Path(args.output_dir) / f'{target_agent}_lora_episode'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Episode-level GRPO training for VSR-Env")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), default="trader")
    parser.add_argument("--trader_id", type=int, default=0)
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=32)
    parser.add_argument("--episode_length", type=int, default=50, help="Steps per episode")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./vsr_grpo_checkpoints_episode")
    return parser


if __name__ == "__main__":
    run_training(build_arg_parser().parse_args())
