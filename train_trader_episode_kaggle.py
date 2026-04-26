"""Episode-level GRPO training for trader_0 on Kaggle.

This script:
1. Clones the Meta repo if needed
2. Runs full episode rollouts (not single-step)
3. Optimizes for trajectory-level rewards (final PnL + position management)

The key improvement over single-step training:
- Model learns position management (not just immediate profit)
- Reward includes final PnL + position quality + greeks balance
- Teaches trade sequencing over multiple steps

Usage on Kaggle:
    !python train_trader_episode_kaggle.py --num_episodes 64 --episode_length 50 --num_train_epochs 2
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import re
import torch
from typing import List, Dict, Any, Tuple

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
from multi_agent.models import MarketMakerAction, OversightAction

VALID_ROLES = {"trader", "market_maker", "oversight"}
JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)

ROLE_REQUIRED_KEYS = {
    "trader": {"selected_strike", "selected_maturity", "direction", "quantity", "option_type", "reasoning"},
}


def example_action_for_role(role: str) -> dict:
    return {"selected_strike": 3, "selected_maturity": 1, "direction": "buy", "quantity": 1.0, "option_type": "call", "reasoning": "IV looks cheap."}


def format_prompt(role: str, target_agent: str, obs) -> str:
    example_json = json.dumps(example_action_for_role(role), separators=(",", ":"))
    return (
        f"You are {target_agent} in a multi-agent options market. "
        "Return JSON only on a single line with keys: selected_strike, selected_maturity, "
        "direction, quantity, option_type, reasoning. Keep reasoning short.\n"
        f"Example: {example_json}\n"
        f"Observation: {obs.model_dump_json()}"
    )


def default_action_for_role(role: str) -> dict:
    return {"selected_strike": 4, "selected_maturity": 0, "direction": "hold", "quantity": 0.0, "option_type": "call", "reasoning": "Hold position."}


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


def parse_json_action(completion, role: str = "trader") -> Tuple[dict, dict]:
    text = completion_to_text(completion).strip()
    required_keys = ROLE_REQUIRED_KEYS["trader"]

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

        try:
            direction = str(parsed.get("direction", "hold")).lower()
            option_type = str(parsed.get("option_type", "call")).lower()
            if direction not in {"buy", "sell", "hold"}:
                direction = "hold"
            if option_type not in {"call", "put"}:
                option_type = "call"

            validated = {
                "selected_strike": int(parsed.get("selected_strike", 4)),
                "selected_maturity": int(parsed.get("selected_maturity", 0)),
                "direction": direction,
                "quantity": max(0, float(parsed.get("quantity", 0))),
                "option_type": option_type,
                "reasoning": str(parsed.get("reasoning", ""))[:160],
            }
            return validated, {"valid_json": True, "validated": True, "format_reward": format_reward + 0.75}
        except Exception:
            return default_action_for_role(role), {"valid_json": True, "validated": False, "format_reward": format_reward - 0.5}

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
        "reasoning": f"Scripted trader_{agent_index} step {step}.",
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
        reasoning="No detection.",
    ).model_dump()


def run_episode_with_llm(
    model,
    tokenizer,
    target_agent: str,
    seed: int,
    episode_length: int,
    device: str,
    verbose: bool = False,
) -> Tuple[List[dict], float, dict]:
    """Run a full episode with the LLM making decisions for target_agent.

    Returns:
        - Episode data (observations, actions, step rewards)
        - Episode-level reward (final PnL + position quality)
        - Episode stats
    """
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=seed)

    episode_data = []
    total_step_rewards = 0.0

    for step in range(episode_length):
        actions = {}

        # Target agent: use LLM
        prompt = format_prompt("trader", target_agent, obs[target_agent])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action, parse_info = parse_json_action(generated, "trader")
        actions[target_agent] = action

        # Scripted other traders
        for i in range(NUM_TRADERS):
            if f"trader_{i}" != target_agent:
                actions[f"trader_{i}"] = scripted_trader(i, step)

        # Scripted MM and oversight
        actions["market_maker"] = scripted_market_maker(step)
        actions["oversight"] = scripted_oversight()

        # Step environment
        next_obs, rewards, done, info = env.step(actions)

        step_reward = rewards[target_agent]
        total_step_rewards += step_reward

        episode_data.append({
            "step": step,
            "action": action,
            "step_reward": step_reward,
            "parse_info": parse_info,
        })

        if verbose and step % 10 == 0:
            print(f"  Step {step}: {action['direction']} {action['quantity']:.1f} {action['option_type']} | reward={step_reward:.3f}")

        obs = next_obs
        if done:
            break

    # Calculate episode-level reward
    final_state = env.agent_states[target_agent]
    final_pnl = final_state.portfolio_pnl

    # Position management quality
    total_contracts = sum(abs(pos.get("quantity", 0)) for pos in final_state.positions)
    position_quality = max(0, 2.0 - total_contracts / 25.0)  # Bonus for keeping inventory < 50

    # Greeks balance
    delta_balance = max(0, 1.0 - abs(final_state.portfolio_delta) / 5.0)
    vega_balance = max(0, 1.0 - abs(final_state.portfolio_vega) / 10.0)

    # Weighted episode reward
    # - Primary: Final PnL (60%)
    # - Secondary: Position management (20%)
    # - Tertiary: Greeks balance (20%)
    episode_reward = (
        final_pnl * 0.6
        + position_quality * 0.2
        + delta_balance * 0.1
        + vega_balance * 0.1
    )

    episode_stats = {
        "final_pnl": final_pnl,
        "total_contracts": total_contracts,
        "final_delta": final_state.portfolio_delta,
        "final_vega": final_state.portfolio_vega,
        "position_quality": position_quality,
        "step_reward_sum": total_step_rewards,
    }

    # Scale and clamp
    episode_reward = max(-5.0, min(5.0, episode_reward))

    return episode_data, episode_reward, episode_stats


def build_episode_dataset(num_episodes: int, target_agent: str):
    """Build dataset with initial observations for episode rollouts."""
    env = MultiAgentVSREnvironment()
    prompts = []
    for seed in range(num_episodes):
        obs = env.reset(seed=seed)
        prompts.append({
            "prompt": format_prompt("trader", target_agent, obs[target_agent]),
            "seed": seed,
            "target_agent": target_agent,
        })
    return Dataset.from_list(prompts)


class EpisodeGRPOTrainer(GRPOTrainer):
    """Custom trainer that runs full episodes for reward computation."""

    def __init__(self, episode_length=50, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_length = episode_length
        self.device = device

    def _move_model_to_device(self, model, device):
        return model.to(device)


def run_training(args) -> None:
    target_agent = f"trader_{args.trader_id}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"EPISODE-LEVEL GRPO TRAINING")
    print(f"{'='*70}")
    print(f"Target Agent:    {target_agent}")
    print(f"Base Model:      {args.base_model}")
    print(f"Episodes:        {args.num_episodes}")
    print(f"Episode Length:  {args.episode_length} steps")
    print(f"Epochs:          {args.num_train_epochs}")
    print(f"Learning Rate:   {args.learning_rate}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Device:          {device}")
    print(f"{'='*70}\n")

    # Load model with Unsloth optimizations
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Build dataset
    print(f"\nBuilding dataset with {args.num_episodes} episodes...")
    dataset = build_episode_dataset(args.num_episodes, target_agent)

    # Episode reward function
    def episode_reward_fn(prompts, completions, **kwargs):
        """Run full episodes and compute trajectory-level rewards."""
        rewards = []
        seeds = kwargs.get("seed", list(range(len(completions))))

        for idx, completion in enumerate(completions):
            sample_seed = int(seeds[idx]) if idx < len(seeds) else idx

            # Run full episode
            episode_data, episode_reward, stats = run_episode_with_llm(
                model, tokenizer, target_agent, sample_seed, args.episode_length, device,
                verbose=(idx == 0)  # Only log first episode
            )

            # Add format rewards
            format_total = sum(d["parse_info"].get("format_reward", 0) for d in episode_data)
            total_reward = episode_reward + format_total * 0.05

            rewards.append(max(-5.0, min(5.0, total_reward)))

            if idx == 0:
                print(f"  Episode 0: PnL={stats['final_pnl']:.3f}, Contracts={stats['total_contracts']:.0f}, Reward={episode_reward:.3f}")

        return rewards

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
        gradient_accumulation_steps=4,  # Memory optimization
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=episode_reward_fn,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    print("\nStarting training...")
    print("Note: Each step runs full episode rollouts, so this will take longer than single-step training.\n")

    trainer.train()

    # Save adapter
    output_path = Path(args.output_dir) / f"{target_agent}_lora_episode"
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Adapter saved to: {output_path}")
    print(f"{'='*70}")

    # Test on a sample episode
    print("\nTesting on sample episode...")
    episode_data, episode_reward, stats = run_episode_with_llm(
        model, tokenizer, target_agent, seed=999, episode_length=args.episode_length, device=device, verbose=True
    )
    print(f"\nTest Episode Results:")
    print(f"  Final PnL:        {stats['final_pnl']:.3f}")
    print(f"  Total Contracts:  {stats['total_contracts']:.0f}")
    print(f"  Final Delta:      {stats['final_delta']:.2f}")
    print(f"  Episode Reward:   {episode_reward:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Episode-level GRPO training for trader on Kaggle")
    parser.add_argument("--trader_id", type=int, default=0)
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--episode_length", type=int, default=50, help="Steps per episode")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=150)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./vsr_grpo_checkpoints_episode")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
