"""Test trained trader LoRA on Kaggle (self-contained).

This script clones the repo if needed, then runs the test.

Usage in Kaggle notebook (after training completes):
    !python test_trader_kaggle.py --lora_path ./vsr_grpo_checkpoints/trader_0_lora --num_steps 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Clone repo if multi_agent not available
if not Path("multi_agent").exists() and not Path("Meta/multi_agent").exists():
    print("Cloning Meta repo...")
    os.system("git clone --branch Agentic-AI https://github.com/manan-tech/Meta.git")
    os.chdir("Meta")
elif Path("Meta/multi_agent").exists() and not Path("multi_agent").exists():
    os.chdir("Meta")

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from multi_agent.environment import MultiAgentVSREnvironment
from multi_agent.models import MarketMakerAction, OversightAction


def format_trader_prompt(target_agent: str, obs) -> str:
    example_json = json.dumps({
        "selected_strike": 3,
        "selected_maturity": 1,
        "direction": "buy",
        "quantity": 1.0,
        "option_type": "call",
        "reasoning": "IV looks cheap.",
    }, separators=(",", ":"))
    return (
        f"You are {target_agent} in a multi-agent options market. "
        "Return JSON only on a single line with keys: selected_strike, selected_maturity, "
        "direction, quantity, option_type, reasoning. Keep reasoning short.\n"
        f"Example: {example_json}\n"
        f"Observation: {obs.model_dump_json()}"
    )


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


def parse_llm_output(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except:
            pass
    return None


def run_episode(model, tokenizer, num_steps: int, use_lora: bool, device: str):
    """Run episode and return cumulative rewards."""
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=42)

    total_rewards = {"trader_0": 0.0, "scripted_traders": 0.0, "market_maker": 0.0}

    mode = "TRAINED LoRA" if use_lora else "SCRIPTED BASELINE"
    print(f"\n{'='*60}")
    print(f"Running {num_steps} steps with {mode}")
    print(f"{'='*60}\n")

    for step in range(num_steps):
        actions = {}

        # Trader 0 decision
        if use_lora and model is not None:
            prompt = format_trader_prompt("trader_0", obs["trader_0"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_llm_output(generated)

            if parsed:
                actions["trader_0"] = parsed
            else:
                actions["trader_0"] = scripted_trader(0, step)
        else:
            actions["trader_0"] = scripted_trader(0, step)

        # Other traders: scripted
        for i in range(1, 10):
            actions[f"trader_{i}"] = scripted_trader(i, step)

        actions["market_maker"] = scripted_market_maker(step)
        actions["oversight"] = scripted_oversight()

        # Step environment
        obs, rewards, done, info = env.step(actions)

        total_rewards["trader_0"] += rewards["trader_0"]
        avg_other = sum(rewards[f"trader_{i}"] for i in range(1, 10)) / 9
        total_rewards["scripted_traders"] += avg_other
        total_rewards["market_maker"] += rewards["market_maker"]

        if step < 5 or step % 20 == 0:
            t0 = actions["trader_0"]
            print(f"Step {step:3d}: trader_0 -> {t0.get('direction', '?'):4} {t0.get('quantity', 0):.1f} {t0.get('option_type', '?'):4} | reward={rewards['trader_0']:.3f}")
            if use_lora and step < 3 and "reasoning" in t0:
                print(f"         Reasoning: {t0['reasoning'][:60]}")

        if done:
            break

    return total_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="./vsr_grpo_checkpoints/trader_0_lora", help="Path to LoRA adapter")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--num_steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lora_path = Path(args.lora_path)

    if not lora_path.exists():
        print(f"Error: LoRA path not found: {lora_path}")
        return

    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()

    # Test 1: Trained model
    rewards_lora = run_episode(model, tokenizer, args.num_steps, use_lora=True, device=device)

    # Test 2: Baseline (all scripted)
    print("\n" + "="*60)
    print("Running BASELINE with all scripted traders...")
    print("="*60)
    rewards_baseline = run_episode(None, tokenizer, args.num_steps, use_lora=False, device=device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Cumulative Rewards")
    print("="*60)
    print(f"{'Metric':<25} {'Trained LoRA':>15} {'Baseline':>15}")
    print("-"*55)
    print(f"{'trader_0':<25} {rewards_lora['trader_0']:>15.3f} {rewards_baseline['trader_0']:>15.3f}")
    print(f"{'avg other traders':<25} {rewards_lora['scripted_traders']:>15.3f} {rewards_baseline['scripted_traders']:>15.3f}")
    print(f"{'market_maker':<25} {rewards_lora['market_maker']:>15.3f} {rewards_baseline['market_maker']:>15.3f}")

    improvement = rewards_lora['trader_0'] - rewards_baseline['trader_0']
    if improvement > 0:
        print(f"\n✅ Trained trader_0 OUTPERFORMS baseline by {improvement:.3f}")
    else:
        print(f"\n⚠️  Baseline outperforms trained model by {-improvement:.3f}")


if __name__ == "__main__":
    main()
