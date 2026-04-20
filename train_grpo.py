"""Minimal GRPO training entrypoint for the multi-agent market environment.

This script trains a single trader policy against scripted market-maker and
oversight policies. It is intentionally narrow so judges can reproduce a short
run in Colab and see reward improvement quickly.
"""

import argparse
import json
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
except ImportError as exc:  # pragma: no cover - import availability depends on env
    FastLanguageModel = None
    GRPOConfig = None
    GRPOTrainer = None
    Dataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from multi_agent.config import EPISODE_LENGTH
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction


def format_trader_prompt(obs) -> str:
    return (
        "You are trader_0 in a multi-agent options market. "
        "Respond with a JSON action using selected_strike, selected_maturity, "
        "direction, quantity, option_type, and reasoning.\n"
        f"Observation: {obs.model_dump_json()}"
    )


def parse_json_action(completion) -> dict:
    if hasattr(completion, "text"):
        completion = completion.text
    try:
        return json.loads(completion)
    except Exception:
        return {
            "selected_strike": 4,
            "selected_maturity": 0,
            "direction": "hold",
            "quantity": 0.0,
            "option_type": "call",
            "reasoning": "Fallback hold action due to parse failure.",
        }


def build_dataset(num_episodes: int):
    env = MultiAgentVSREnvironment()
    prompts = []
    for seed in range(num_episodes):
        obs = env.reset(seed=seed)["trader_0"]
        prompts.append({"prompt": format_trader_prompt(obs), "seed": seed})
    return Dataset.from_list(prompts)


def scripted_market_maker(step: int) -> MarketMakerAction:
    if step < 25:
        return MarketMakerAction(atm_spread=0.025, otm_spread=0.045, itm_spread=0.035)
    if step < 100:
        return MarketMakerAction(atm_spread=0.04, otm_spread=0.06, itm_spread=0.05)
    return MarketMakerAction(atm_spread=0.05, otm_spread=0.07, itm_spread=0.06)


def scripted_oversight() -> OversightAction:
    return OversightAction(flagged_agents=[], flag_type="none", fine_amount=0.0)


def rollout_reward(action_dict: dict, seed: int, steps_per_episode: int) -> float:
    env = MultiAgentVSREnvironment()
    env.reset(seed=seed)

    cumulative_reward = 0.0
    rollout_steps = min(steps_per_episode, EPISODE_LENGTH)
    for step in range(rollout_steps):
        simulated_actions = {
            "trader_0": action_dict,
            "market_maker": scripted_market_maker(step),
            "oversight": scripted_oversight(),
        }
        _, reward_dict, done, _ = env.step(simulated_actions)
        cumulative_reward += reward_dict["trader_0"]
        if done:
            break
    return cumulative_reward / float(rollout_steps)


def run_training(args) -> None:
    if IMPORT_ERROR is not None or not all([FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset]):
        raise ImportError(
            "Missing training dependencies. Install unsloth, trl, datasets, peft, and transformers first."
        ) from IMPORT_ERROR

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

    dataset = build_dataset(args.num_episodes)

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for idx, completion in enumerate(completions):
            action = parse_json_action(completion)
            rewards.append(rollout_reward(action, idx, args.steps_per_episode))
        return rewards

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.group_size,
        max_completion_length=256,
        logging_steps=10,
        save_steps=100,
        learning_rate=args.learning_rate,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(str(Path(args.output_dir) / "trader_lora"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a trader policy in Multi-Agent VSR-Env with GRPO.")
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
