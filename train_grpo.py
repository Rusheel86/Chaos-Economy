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
import math
from pathlib import Path
import re
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

from multi_agent.config import NUM_TRADERS
from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction

VALID_ROLES = {"trader", "market_maker", "oversight"}
JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
ROLE_REQUIRED_KEYS = {
    "trader": {
        "selected_strike",
        "selected_maturity",
        "direction",
        "quantity",
        "option_type",
        "reasoning",
    },
    "market_maker": {"atm_spread", "otm_spread", "itm_spread", "skew_adjustment", "reasoning"},
    "oversight": {
        "flagged_agents",
        "flag_type",
        "fine_amount",
        "halt_strikes",
        "confidence",
        "intervention_type",
        "reasoning",
    },
}


def resolve_target_agent(role: str, trader_id: int) -> str:
    if role == "trader":
        return f"trader_{trader_id}"
    return role


def example_action_for_role(role: str) -> dict:
    if role == "trader":
        return {
            "selected_strike": 3,
            "selected_maturity": 1,
            "direction": "buy",
            "quantity": 1.0,
            "option_type": "call",
            "reasoning": "IV looks cheap.",
        }
    if role == "market_maker":
        return {
            "atm_spread": 0.04,
            "otm_spread": 0.06,
            "itm_spread": 0.05,
            "skew_adjustment": 0.0,
            "reasoning": "Balanced quotes.",
        }
    return {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "halt_strikes": [],
        "confidence": 0.1,
        "intervention_type": "none",
        "reasoning": "No clear manipulation.",
    }


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


def parse_json_action(completion, role: str) -> tuple[dict, dict]:
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
            return default_action_for_role(role), {
                "valid_json": True,
                "validated": False,
                "format_reward": format_reward - 0.5,
            }

        return validated, {
            "valid_json": True,
            "validated": True,
            "format_reward": format_reward + 0.75,
        }

    return default_action_for_role(role), {
        "valid_json": False,
        "validated": False,
        "format_reward": -0.5,
    }


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


def squash_reward(raw_reward: float, limit: float = 5.0) -> float:
    return max(-limit, min(limit, math.copysign(math.log1p(abs(raw_reward)), raw_reward)))


def single_step_reward(role: str, target_agent: str, action_dict: dict, seed: int) -> float:
    env = MultiAgentVSREnvironment()
    env.reset(seed=seed)
    simulated_actions = build_actions_for_step(role, target_agent, action_dict, step=0)
    _, reward_dict, _, _ = env.step(simulated_actions)
    return squash_reward(reward_dict[target_agent])


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
        seeds = kwargs.get("seed", list(range(len(completions))))
        for idx, completion in enumerate(completions):
            action, parse_info = parse_json_action(completion, args.role)
            sample_seed = int(seeds[idx]) if idx < len(seeds) else idx
            env_reward = single_step_reward(args.role, target_agent, action, sample_seed)
            rewards.append(max(-5.0, min(5.0, env_reward + parse_info["format_reward"])))
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        logging_steps=10,
        save_steps=100,
        logging_dir=str(Path(args.output_dir) / "runs"),
        report_to="tensorboard",
        run_name=f"grpo_{target_agent}",
        learning_rate=args.learning_rate,
        ddp_find_unused_parameters=False,
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
    if trainer.is_world_process_zero():
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
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./vsr_grpo_checkpoints")
    return parser


if __name__ == "__main__":
    run_training(build_arg_parser().parse_args())
