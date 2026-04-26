"""Test trained trader LoRA vs scripted baseline (CPU/Mac compatible).

Usage:
    python3 test_trader_lora.py --lora_path ./Meta/vsr_grpo_checkpoints/trader_0_lora
"""

import argparse
import json
from pathlib import Path
import torch

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
    """Extract JSON from LLM output."""
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


def run_episode_with_lora(model, tokenizer, num_steps: int = 50, use_lora: bool = True, device: str = "cpu"):
    """Run episode with trained trader_0, scripted others."""
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=42)

    total_rewards = {"trader_0": 0.0, "scripted_traders": 0.0, "market_maker": 0.0}

    print(f"\n{'='*60}")
    print(f"Running {num_steps} steps with {'TRAINED' if use_lora else 'SCRIPTED'} trader_0")
    print(f"{'='*60}\n")

    for step in range(num_steps):
        actions = {}

        # Trader 0: use LLM (trained or not)
        if use_lora and model is not None:
            prompt = format_trader_prompt("trader_0", obs["trader_0"])
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

            parsed = parse_llm_output(generated)
            if parsed:
                actions["trader_0"] = parsed
            else:
                print(f"  [Step {step}] Failed to parse, using fallback")
                actions["trader_0"] = scripted_trader(0, step)
        else:
            actions["trader_0"] = scripted_trader(0, step)

        # Other traders: scripted
        for i in range(1, 10):
            actions[f"trader_{i}"] = scripted_trader(i, step)

        # MM and oversight: scripted
        actions["market_maker"] = scripted_market_maker(step)
        actions["oversight"] = scripted_oversight()

        # Step environment
        obs, rewards, done, info = env.step(actions)

        # Track rewards
        total_rewards["trader_0"] += rewards["trader_0"]
        avg_other = sum(rewards[f"trader_{i}"] for i in range(1, 10)) / 9
        total_rewards["scripted_traders"] += avg_other
        total_rewards["market_maker"] += rewards["market_maker"]

        # Log some actions
        if step < 5 or step % 10 == 0:
            t0_action = actions["trader_0"]
            print(f"Step {step:3d}: trader_0 -> {t0_action.get('direction', '?')} {t0_action.get('quantity', '?')} {t0_action.get('option_type', '?')} | reward={rewards['trader_0']:.3f}")
            if use_lora and step < 3 and "reasoning" in t0_action:
                print(f"         Reasoning: {t0_action.get('reasoning', 'N/A')[:60]}...")

        if done:
            break

    return total_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trader_0_lora folder")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_steps", type=int, default=30)
    args = parser.parse_args()

    lora_path = Path(args.lora_path)

    if not lora_path.exists():
        print(f"Error: LoRA path not found: {lora_path}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading base model: {args.base_model}")
    print("(This may take a minute on first run...)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    print(f"\nLoading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.to(device)
    model.eval()

    # Test 1: With trained LoRA
    rewards_lora = run_episode_with_lora(model, tokenizer, args.num_steps, use_lora=True, device=device)

    # Test 2: Without LoRA (baseline)
    print("\n" + "="*60)
    print("Running baseline with ALL scripted traders...")
    print("="*60 + "\n")
    rewards_baseline = run_episode_with_lora(None, tokenizer, args.num_steps, use_lora=False, device=device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Cumulative Rewards")
    print("="*60)
    print(f"{'Metric':<25} {'Trained LoRA':>15} {'All Scripted':>15}")
    print("-"*55)
    print(f"{'trader_0':<25} {rewards_lora['trader_0']:>15.3f} {rewards_baseline['trader_0']:>15.3f}")
    print(f"{'avg other traders':<25} {rewards_lora['scripted_traders']:>15.3f} {rewards_baseline['scripted_traders']:>15.3f}")
    print(f"{'market_maker':<25} {rewards_lora['market_maker']:>15.3f} {rewards_baseline['market_maker']:>15.3f}")

    improvement = rewards_lora['trader_0'] - rewards_baseline['trader_0']
    if improvement > 0:
        print(f"\n✅ Trained trader_0 outperforms scripted by {improvement:.3f} cumulative reward")
    else:
        print(f"\n⚠️  Scripted baseline outperforms by {-improvement:.3f} cumulative reward")


if __name__ == "__main__":
    main()
