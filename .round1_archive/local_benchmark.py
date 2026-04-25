import asyncio
import json
import os
import sys

# Try importing groq, fallback if not installed
try:
    from groq import AsyncGroq, RateLimitError
except ImportError:
    print("Please install groq: pip install groq")
    sys.exit(1)

# Try importing tenacity
try:
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
except ImportError:
    print("Please install tenacity for rate limit handling: pip install tenacity")
    sys.exit(1)

from vsr_env.client import LocalVSREnv
from vsr_env.models import VSRAction, TradeDirection

# Groq models to benchmark
MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it"
]

TASKS = [
    "delta_hedging",
    "earnings_vol_crush",
    "gamma_scalping",
    "vega_gamma_stress",
    "vol_regime_detection"
]

SYSTEM_PROMPT = """You are an expert quantitative volatility trader.
Analyze the market state and return your trade action in JSON format ONLY.

Your JSON must exactly match this schema:
{
  "selected_strike": integer (0 to 7),
  "selected_maturity": integer (0 to 2),
  "direction": "buy", "sell", or "hold",
  "quantity": float (0.0 to 10.0),
  "reasoning": string (brief explanation of your thesis)
}
Do not return any markdown blocking or text outside the JSON.
"""

# Retry decorator with exponential backoff specifically catching Groq RateLimitError
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60), # Wait 4s, 8s, 16s... up to 60s
    stop=stop_after_attempt(10), # Give up after 10 tries
    reraise=True
)
async def get_groq_completion(client, model_name, prompt):
    print("      (Requesting LLM Action...)")
    chat_completion = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return chat_completion

async def run_episode(client: AsyncGroq, model_name: str, task_name: str, env: LocalVSREnv) -> float:
    print(f"  Running {task_name} with {model_name}...")
    try:
        # Reset environment
        obs_res = await env.reset(task_name=task_name)
        total_reward = 0.0
        total_steps = 0
        done = False
        
        while not done:
            obs = obs_res.observation
            prompt = f"""Current State:
Spot: {obs.spot_price}
IV Surface: {obs.iv_surface}
Greeks: {obs.portfolio_greeks}
PnL: {obs.portfolio_pnl}
Step: {obs.step_number} of {obs.step_number + obs.steps_remaining}
"""
            # Ask the model via the retry-wrapped function
            chat_completion = await get_groq_completion(client, model_name, prompt)
            
            response_text = chat_completion.choices[0].message.content.strip()
            # Clean up markdown blocks if the model ignored instructions
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            # Parse json
            action_dict = json.loads(response_text)
            
            # Map to VSRAction
            action = VSRAction(
                selected_strike=int(action_dict.get("selected_strike", 0)),
                selected_maturity=int(action_dict.get("selected_maturity", 0)),
                direction=action_dict.get("direction", "hold").lower(),
                quantity=float(action_dict.get("quantity", 0.0)),
                reasoning=action_dict.get("reasoning", "")
            )
            
            # Step environment
            step_res = await env.step(action)
            total_reward += step_res.reward
            done = step_res.done
            obs_res = step_res  # step_res has observation attribute in openenv
            total_steps += 1
            
        # Normalize score to [0, 1] using steps taken
        avg_score = total_reward / max(1, total_steps)
        return min(max(avg_score, 0.0), 1.0)
    except Exception as e:
        print(f"    Error during episode: {e}")
        return 0.0

async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)
        
    client = AsyncGroq(api_key=api_key)
    
    # Store results: {model: {task: reward}}
    results = {model: {task: 0.0 for task in TASKS} for model in MODELS}
    
    async with LocalVSREnv() as env:
        for model in MODELS:
            print(f"\nEvaluating Model: {model}")
            for task in TASKS:
                score = await run_episode(client, model, task, env)
                results[model][task] = score
                print(f"    Score for {task}: {score:.4f}")
                
    # Generate Markdown Table
    print("\n\n" + "="*50)
    print("### Empirical Baseline Results\n")
    
    header = "| Model | " + " | ".join(TASKS) + " | Average |"
    separator = "|" + "-"*len(" Model ") + "|" + "|".join(["-"*(len(t)+2) for t in TASKS]) + "|---------|"
    
    print(header)
    print(separator)
    
    for model in MODELS:
        scores = [results[model][task] for task in TASKS]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        row = f"| **{model}** | " + " | ".join(f"{s:.2f}" for s in scores) + f" | **{avg_score:.2f}** |"
        print(row)

if __name__ == "__main__":
    asyncio.run(main())