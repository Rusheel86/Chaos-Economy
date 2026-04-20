"""Minimal GRPO training entrypoint for Multi-Agent VSR-Env.

This script supports separate training runs for:
- one trader agent at a time
- the market maker
- the oversight agent

The intended hackathon workflow is to train each role separately first,
using scripted counterpart policies, then compare before/after reward curves.
"""

import argparse
import json
from pathlib import Path
import torch

try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover - dependency availability varies
    FastLanguageModel = None
    GRPOConfig = None
    GRPOTrainer = None
    Dataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from multi_agent.config import EPISODE_LENGTH, NUM_TRADERS
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction

VALID_ROLES = {"trader", "market_maker", "oversight"}


def resolve_target_agent(role: str, trader_id: int) -> str:
    if role == "trader":
        return f"trader_{trader_id}"
    return role


def format_prompt(role: str, target_agent: str, obs) -> str:
    if role == "trader":
        return (
            f"You are {target_agent} in a multi-agent options market. "
            "Return a compact JSON action with keys: selected_strike, selected_maturity, "
            "direction, quantity, option_type, reasoning.\n"
            f"Observation: {obs.model_dump_json()}"
        )
    if role == "market_maker":
        return (
            "You are the market maker in a multi-agent options market. "
            "Return a compact JSON action with keys: atm_spread, otm_spread, itm_spread, "
            "skew_adjustment, reasoning.\n"
            f"Observation: {obs.model_dump_json()}"
        )
    return (
        "You are the oversight agent in a multi-agent market surveillance task. "
        "Return a compact JSON action with keys: flagged_agents, flag_type, fine_amount, "
        "halt_strikes, confidence, intervention_type, reasoning.\n"
        f"Observation: {obs.model_dump_json()}"
    )


def default_action_for_role(role: str) -> dict:
    if role == "trader":
        return {
            "selected_strike": 4,
            "selected_maturity": 0,
            "direction": "hold",
            "quantity": 0.0,
            "option_type": "call",
            "reasoning": "Fallback hold action due to parse failure.",
        }
    if role == "market_maker":
        return {
            "atm_spread": 0.04,
            "otm_spread": 0.06,
            "itm_spread": 0.05,
            "skew_adjustment": 0.0,
            "reasoning": "Fallback defensive quotes.",
        }
    return {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "halt_strikes": [],
        "confidence": 0.0,
        "intervention_type": "none",
        "reasoning": "No harmful behavior detected.",
    }


def parse_json_action(completion, role: str) -> dict:
    if hasattr(completion, "text"):
        completion = completion.text
    try:
        parsed = json.loads(completion)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return default_action_for_role(role)


def build_dataset(num_episodes: int, role: str, target_agent: str):
    env = MultiAgentVSREnvironment()
    prompts = []
    for seed in range(num_episodes):
        obs = env.reset(seed=seed)[target_agent]
        prompts.append(
            {
                "prompt": format_prompt(role, target_agent, obs),
                "seed": seed,
                "target_agent": target_agent,
            }
        )
    return Dataset.from_list(prompts)


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


def build_actions_for_step(role: str, target_agent: str, target_action: dict, step: int) -> dict:
    actions = {}

    for trader_idx in range(NUM_TRADERS):
        trader_agent = f"trader_{trader_idx}"
        if role == "trader" and trader_agent == target_agent:
            actions[trader_agent] = target_action
        else:
            actions[trader_agent] = scripted_trader(trader_idx, step)

    if role == "market_maker":
        actions["market_maker"] = target_action
    else:
        actions["market_maker"] = scripted_market_maker(step)

    if role == "oversight":
        actions["oversight"] = target_action
    else:
        actions["oversight"] = scripted_oversight()

    return actions


def rollout_reward(role: str, target_agent: str, action_dict: dict, seed: int, steps_per_episode: int) -> float:
    env = MultiAgentVSREnvironment()
    env.reset(seed=seed)

    cumulative_reward = 0.0
    rollout_steps = min(steps_per_episode, EPISODE_LENGTH)

    for step in range(rollout_steps):
        simulated_actions = build_actions_for_step(role, target_agent, action_dict, step)
        _, reward_dict, done, _ = env.step(simulated_actions)
        cumulative_reward += reward_dict[target_agent]
        if done:
            break

    return cumulative_reward / float(rollout_steps)


def run_training(args) -> None:
    if IMPORT_ERROR is not None or not all([FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset]):
        raise ImportError(
            "Missing training dependencies. Install unsloth, trl, datasets, peft, and transformers first."
        ) from IMPORT_ERROR

    if args.role not in VALID_ROLES:
        raise ValueError(f"Unsupported role '{args.role}'. Choose from {sorted(VALID_ROLES)}.")

    target_agent = resolve_target_agent(args.role, args.trader_id)

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

    dataset = build_dataset(args.num_episodes, args.role, target_agent)

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for idx, completion in enumerate(completions):
            action = parse_json_action(completion, args.role)
            rewards.append(
                rollout_reward(
                    args.role,
                    target_agent,
                    action,
                    idx,
                    args.steps_per_episode,
                )
            )
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.group_size,
        max_completion_length=256,
        logging_steps=10,
        save_steps=100,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported()
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(str(Path(args.output_dir) / f"{target_agent}_lora"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a role policy in Multi-Agent VSR-Env with GRPO.")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), default="trader")
    parser.add_argument("--trader_id", type=int, default=0, help="Only used when --role trader.")
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--steps_per_episode", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./vsr_grpo_checkpoints")
    return parser


if __name__ == "__main__":
    run_training(build_arg_parser().parse_args())
