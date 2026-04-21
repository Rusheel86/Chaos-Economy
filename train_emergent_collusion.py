"""Multi-agent training for emergent collusion discovery.

KEY INSIGHT: Train multiple traders TOGETHER in the same episodes.
If they can see each other's positions and actions, they may discover
coordinated strategies (gamma squeezes, wash trading) through pure
reward optimization.

The oversight agent then trains to detect these emergent behaviors.

Story Arc:
1. Steps 0-50: Individual profit seeking
2. Steps 50-150: Traders notice coordination benefits
3. Steps 150+: Predatory swarms - coordinated manipulation emerges
4. Oversight learns to detect collusion patterns

Usage on Kaggle:
    !python train_emergent_collusion.py --num_episodes 128 --episode_length 100
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import re
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
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
from multi_agent.models import MarketMakerAction, OversightAction

# ============================================================================
# PROMPT FORMATTING WITH MARKET VISIBILITY
# ============================================================================

def format_trader_prompt_with_visibility(
    target_agent: str,
    obs,
    all_positions: Dict[str, List],
    recent_trades: List[Dict],
    step: int,
) -> str:
    """Format prompt showing trader what others are doing.

    This visibility enables emergent coordination - traders can see
    if others are building positions at same strikes and potentially
    coordinate (even implicitly through observation).
    """
    example_json = json.dumps({
        "selected_strike": 3,
        "selected_maturity": 1,
        "direction": "buy",
        "quantity": 1.0,
        "option_type": "call",
        "reasoning": "IV looks cheap.",
    }, separators=(",", ":"))

    # Build position summary for OTHER traders
    other_positions = []
    for agent_id, positions in all_positions.items():
        if agent_id != target_agent and positions:
            for pos in positions[:3]:  # Show top 3 positions per agent
                other_positions.append({
                    "agent": agent_id,
                    "strike": pos.get("selected_strike", "?"),
                    "type": pos.get("option_type", "?"),
                    "qty": pos.get("quantity", 0),
                })

    # Build recent trade flow
    trade_flow = []
    for trade in recent_trades[-10:]:
        trade_flow.append({
            "agent": trade.get("agent_id", "?"),
            "strike": trade.get("selected_strike", "?"),
            "type": trade.get("option_type", "?"),
            "direction": trade.get("direction", "?"),
            "qty": trade.get("quantity", 0),
        })

    prompt = f"""You are {target_agent} in a multi-agent options market.

## Your Goal
Maximize your PnL. You have full visibility of other traders' positions and recent trades.

## Current Market State
- Spot Price: {obs.spot_price:.2f}
- IV Surface: ATM ~{obs.iv_surface[3][0]*100:.1f}%
- MM Spreads: ATM {obs.mm_spreads['atm']*100:.1f}%, OTM {obs.mm_spreads['otm']*100:.1f}%
- Step: {step}/{EPISODE_LENGTH}

## Your Current State
- PnL: ${obs.own_pnl:.2f}
- Delta: {obs.own_greeks.get('delta', 0):.2f}
- Gamma: {obs.own_greeks.get('gamma', 0):.2f}
- Vega: {obs.own_greeks.get('vega', 0):.2f}
- Cash: ${obs.own_cash:.0f}

## Other Traders' Positions (they can coordinate against MM!)
{json.dumps(other_positions[:15], indent=1) if other_positions else "No positions yet"}

## Recent Trade Flow (watch for patterns!)
{json.dumps(trade_flow, indent=1) if trade_flow else "No trades yet"}

## Response Format
Return JSON only on a single line with keys: selected_strike, selected_maturity, direction, quantity, option_type, reasoning.

Example: {example_json}

Observation: {obs.model_dump_json()}
"""
    return prompt


def format_oversight_prompt_with_collusion_detection(
    obs,
    position_heatmap: Dict[int, int],
    coordinated_pressure: Dict[str, Any],
) -> str:
    """Enhanced oversight prompt to detect emergent collusion patterns."""
    example_json = json.dumps({
        "flagged_agents": ["trader_0", "trader_1"],
        "flag_type": "coordinated_pressure",
        "fine_amount": 5000.0,
        "halt_strikes": [3, 4],
        "confidence": 0.85,
        "intervention_type": "fine",
        "reasoning": "Traders 0 and 1 coordinating gamma squeeze at strikes 3-4.",
    }, separators=(",", ":"))

    prompt = f"""You are the SEC oversight agent monitoring a multi-agent options market.

## Your Mission
Detect market manipulation INCLUDING coordinated behavior between traders.
Watch for:
1. **Gamma Squeeze**: Multiple traders buying same strike OTM options
2. **Coordinated Pressure**: Traders targeting same strike/maturity together
3. **Wash Trading Ring**: Circular trades between specific traders
4. **Spoofing**: Large orders followed by reversals

## Market State
- Spot: {obs.spot_price:.2f}
- Position Heatmap (strike -> total contracts): {json.dumps(position_heatmap)}
- Coordinated Pressure Detected: {json.dumps(coordinated_pressure)}

## All Agent PnLs
{json.dumps(obs.all_agent_pnls, indent=1)}

## Recent Trades (look for coordination patterns!)
{json.dumps(obs.trade_log[-20:] if obs.trade_log else [], indent=1)}

## Agent Risk Summary (high gamma = squeeze target?)
{json.dumps(obs.agent_risk_summary, indent=1)}

## Response Format
Return JSON with: flagged_agents, flag_type, fine_amount, halt_strikes, confidence, intervention_type, reasoning.

Example: {example_json}

Observation: {obs.model_dump_json()}
"""
    return prompt


# ============================================================================
# ACTION PARSING
# ============================================================================

JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_json_action(completion, role: str = "trader") -> Tuple[dict, dict]:
    """Parse LLM output into action dict."""
    text = completion if isinstance(completion, str) else str(completion)
    text = text.strip()

    for candidate in [text] + [m.group(1) for m in JSON_CODE_BLOCK_RE.finditer(text)] + [m.group(0) for m in JSON_OBJECT_RE.finditer(text)]:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                continue

            if role == "trader":
                direction = str(parsed.get("direction", "hold")).lower()
                if direction not in ["buy", "sell", "hold"]:
                    direction = "hold"
                option_type = str(parsed.get("option_type", "call")).lower()
                if option_type not in ["call", "put"]:
                    option_type = "call"

                return {
                    "selected_strike": int(parsed.get("selected_strike", 4)),
                    "selected_maturity": int(parsed.get("selected_maturity", 0)),
                    "direction": direction,
                    "quantity": max(0, float(parsed.get("quantity", 0))),
                    "option_type": option_type,
                    "reasoning": str(parsed.get("reasoning", ""))[:200],
                }, {"valid": True, "format_reward": 0.5}

            elif role == "oversight":
                return {
                    "flagged_agents": parsed.get("flagged_agents", []),
                    "flag_type": str(parsed.get("flag_type", "none")),
                    "fine_amount": float(parsed.get("fine_amount", 0)),
                    "halt_strikes": parsed.get("halt_strikes", []),
                    "confidence": float(parsed.get("confidence", 0)),
                    "intervention_type": str(parsed.get("intervention_type", "none")),
                    "reasoning": str(parsed.get("reasoning", ""))[:200],
                }, {"valid": True, "format_reward": 0.5}

        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Fallback
    if role == "trader":
        return {"selected_strike": 4, "selected_maturity": 0, "direction": "hold", "quantity": 0, "option_type": "call", "reasoning": "Parse failed"}, {"valid": False, "format_reward": -0.5}
    return {"flagged_agents": [], "flag_type": "none", "fine_amount": 0, "halt_strikes": [], "confidence": 0, "intervention_type": "none", "reasoning": "Parse failed"}, {"valid": False, "format_reward": -0.5}


# ============================================================================
# SCRIPTED POLICIES
# ============================================================================

def scripted_market_maker(step: int) -> dict:
    if step < 25:
        return MarketMakerAction(atm_spread=0.025, otm_spread=0.045, itm_spread=0.035).model_dump()
    if step < 100:
        return MarketMakerAction(atm_spread=0.04, otm_spread=0.06, itm_spread=0.05).model_dump()
    return MarketMakerAction(atm_spread=0.05, otm_spread=0.07, itm_spread=0.06).model_dump()


# ============================================================================
# COLLUSION DETECTION (for oversight training)
# ============================================================================

@dataclass
class CollusionDetector:
    """Detect emergent collusion patterns during episodes."""

    positions_history: Dict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))
    trade_history: List[Dict] = field(default_factory=list)

    def update(self, agent_states: Dict, trade_log: List[Dict]):
        for agent_id, state in agent_states.items():
            if hasattr(state, 'positions'):
                self.positions_history[agent_id].append({
                    "positions": list(state.positions),
                    "delta": state.portfolio_delta,
                    "gamma": state.portfolio_gamma,
                })
        self.trade_log = trade_log

    def detect_coordinated_pressure(self) -> Dict[str, Any]:
        """Detect if multiple traders are targeting same strikes."""
        strike_concentration = defaultdict(lambda: {"total_qty": 0, "agents": []})

        for agent_id, history in self.positions_history.items():
            if not history:
                continue
            latest = history[-1].get("positions", [])
            for pos in latest:
                strike = pos.get("selected_strike", -1)
                qty = abs(pos.get("quantity", 0))
                if strike >= 0 and qty > 0:
                    strike_concentration[strike]["total_qty"] += qty
                    strike_concentration[strike]["agents"].append(agent_id)

        # Find strikes with multiple agents
        coordinated = {}
        for strike, data in strike_concentration.items():
            if len(set(data["agents"])) >= 2 and data["total_qty"] > 10:
                coordinated[strike] = {
                    "agents": list(set(data["agents"])),
                    "total_contracts": data["total_qty"],
                    "pressure_type": "gamma_squeeze" if strike < 4 else "coordinated",
                }

        return coordinated

    def detect_wash_trading_ring(self) -> List[Dict]:
        """Detect circular trading between agents."""
        agent_trades = defaultdict(list)
        for trade in self.trade_log[-50:]:
            agent_trades[trade.get("agent_id")].append(trade)

        suspicious_rings = []

        # Check for rapid buy/sell between same agents on same instrument
        for i, (agent_a, trades_a) in enumerate(agent_trades.items()):
            for agent_b, trades_b in list(agent_trades.items())[i+1:]:
                # Check if they're trading opposite directions on same strikes
                for ta in trades_a[-5:]:
                    for tb in trades_b[-5:]:
                        if (ta.get("selected_strike") == tb.get("selected_strike") and
                            ta.get("option_type") == tb.get("option_type") and
                            ta.get("direction") != tb.get("direction")):
                            suspicious_rings.append({
                                "agents": [agent_a, agent_b],
                                "strike": ta.get("selected_strike"),
                                "type": "potential_wash_trade",
                            })

        return suspicious_rings

    def get_position_heatmap(self) -> Dict[int, int]:
        """Get total contracts per strike across all traders."""
        heatmap = defaultdict(int)
        for history in self.positions_history.values():
            if not history:
                continue
            for pos in history[-1].get("positions", []):
                strike = pos.get("selected_strike", -1)
                qty = abs(pos.get("quantity", 0))
                if strike >= 0:
                    heatmap[strike] += qty
        return dict(sorted(heatmap.items()))


# ============================================================================
# EPISODE RUNNER WITH EMERGENT COLLUSION TRACKING
# ============================================================================

def run_multi_agent_episode(
    models: Dict[str, Any],  # {agent_id: model} or {agent_id: None} for scripted
    tokenizer: Any,
    episode_length: int,
    seed: int,
    device: str,
    target_traders: List[str] = None,  # Which traders to train (others scripted)
    verbose: bool = False,
) -> Tuple[Dict[str, List], Dict[str, float], Dict[str, Any]]:
    """Run episode with multiple LLM traders that can observe each other.

    This enables emergent collusion - traders see each other's positions
    and may discover coordinated strategies through reward optimization.
    """
    env = MultiAgentVSREnvironment()
    obs = env.reset(seed=seed)

    collusion_detector = CollusionDetector()

    episode_data = {agent_id: [] for agent_id in env.AGENT_IDS}
    total_rewards = {agent_id: 0.0 for agent_id in env.AGENT_IDS}

    all_positions = {f"trader_{i}": [] for i in range(NUM_TRADERS)}

    for step in range(episode_length):
        actions = {}

        # === TRADERS ===
        for i in range(NUM_TRADERS):
            agent_id = f"trader_{i}"

            if target_traders and agent_id in target_traders and models.get(agent_id):
                # Use LLM for target traders
                model = models[agent_id]
                prompt = format_trader_prompt_with_visibility(
                    agent_id, obs[agent_id], all_positions, env.trade_log, step
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                action, parse_info = parse_json_action(generated, "trader")
            else:
                # Scripted fallback
                strike = (i + step) % 8
                maturity = (i + step) % 3
                direction = "buy" if (i + step) % 2 == 0 else "sell"
                action = {
                    "selected_strike": strike,
                    "selected_maturity": maturity,
                    "direction": direction,
                    "quantity": 0.5 + ((i + step) % 3) * 0.5,
                    "option_type": "call" if i % 2 == 0 else "put",
                    "reasoning": f"Scripted step {step}",
                }
                parse_info = {"valid": True}

            actions[agent_id] = action

            # Track for collusion detection
            if action["direction"] in ["buy", "sell"]:
                all_positions[agent_id].append(action)

        # === MARKET MAKER (scripted for now) ===
        actions["market_maker"] = scripted_market_maker(step)

        # === OVERSIGHT ===
        if models.get("oversight"):
            # Use LLM oversight
            position_heatmap = collusion_detector.get_position_heatmap()
            coordinated = collusion_detector.detect_coordinated_pressure()

            prompt = format_oversight_prompt_with_collusion_detection(
                obs["oversight"], position_heatmap, coordinated
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = models["oversight"].generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            action, parse_info = parse_json_action(generated, "oversight")
        else:
            # Scripted oversight (baseline)
            action = OversightAction(
                flagged_agents=[],
                flag_type="none",
                fine_amount=0.0,
                confidence=0.0,
                intervention_type="none",
                reasoning="Scripted baseline.",
            ).model_dump()

        actions["oversight"] = action

        # === STEP ENVIRONMENT ===
        next_obs, rewards, done, info = env.step(actions)

        # Update collusion detector
        collusion_detector.update(env.agent_states, env.trade_log)

        # Record data
        for agent_id in env.AGENT_IDS:
            episode_data[agent_id].append({
                "step": step,
                "action": actions.get(agent_id, {}),
                "reward": rewards.get(agent_id, 0),
                "parse_info": parse_info if agent_id in (target_traders or []) else {},
            })
            total_rewards[agent_id] += rewards.get(agent_id, 0)

        if verbose and step % 20 == 0:
            coordinated = collusion_detector.detect_coordinated_pressure()
            print(f"Step {step}: Coordinated pressure detected: {coordinated}")

        obs = next_obs
        if done:
            break

    # Final collusion analysis
    final_collusion = {
        "coordinated_pressure": collusion_detector.detect_coordinated_pressure(),
        "wash_trading_rings": collusion_detector.detect_wash_trading_ring(),
        "position_heatmap": collusion_detector.get_position_heatmap(),
    }

    return episode_data, total_rewards, final_collusion


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_traders_with_emergent_collusion(args):
    """Train multiple traders together - watch for emergent coordination."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"EMERGENT COLLUSION TRAINING")
    print(f"{'='*70}")
    print(f"Training {args.num_traders} traders together")
    print(f"Episodes: {args.num_episodes}")
    print(f"Episode Length: {args.episode_length}")
    print(f"{'='*70}\n")

    # Load shared base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Create separate LoRA adapters for each trader
    # (they share base model but have independent adapters)
    trader_models = {}
    for i in range(args.num_traders):
        model = FastLanguageModel.get_peft_model(
            model if i == 0 else FastLanguageModel.from_pretrained(args.base_model, max_seq_length=2048, load_in_4bit=True)[0],
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
        )
        trader_models[f"trader_{i}"] = model

    # Training loop with episode rollouts
    collusion_log = []

    for epoch in range(args.num_train_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_train_epochs} ===")

        for ep in range(args.num_episodes):
            seed = ep + epoch * args.num_episodes

            # Run episode with all trader LLMs
            episode_data, rewards, collusion = run_multi_agent_episode(
                models=trader_models,
                tokenizer=tokenizer,
                episode_length=args.episode_length,
                seed=seed,
                device=device,
                target_traders=[f"trader_{i}" for i in range(args.num_traders)],
                verbose=(ep % 10 == 0),
            )

            collusion_log.append({
                "episode": seed,
                "rewards": rewards,
                "collusion_detected": collusion["coordinated_pressure"],
            })

            if ep % 10 == 0:
                avg_trader_reward = sum(rewards[f"trader_{i}"] for i in range(args.num_traders)) / args.num_traders
                print(f"  Ep {ep}: Avg trader reward = {avg_trader_reward:.3f}, Collusion: {collusion['coordinated_pressure']}")

    # Save adapters
    for i in range(args.num_traders):
        save_path = Path(args.output_dir) / f"trader_{i}_lora"
        trader_models[f"trader_{i}"].save_pretrained(str(save_path))
        print(f"Saved: {save_path}")

    # Save collusion log for analysis
    with open(Path(args.output_dir) / "collusion_log.json", "w") as f:
        json.dump(collusion_log, f, indent=2, default=str)

    print(f"\nTraining complete!")
    print(f"Collusion patterns logged to: {args.output_dir}/collusion_log.json")


def main():
    parser = argparse.ArgumentParser(description="Train traders for emergent collusion")
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--num_traders", type=int, default=2, help="Number of LLM traders (rest scripted)")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--episode_length", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", default="./emergent_collusion_checkpoints")
    parser.add_argument("--trader_type", type=str, default="aggressive",
                        choices=["aggressive", "neutral", "contrarian"],
                        help="Trader archetype to train")
    args = parser.parse_args()

    train_traders_with_emergent_collusion(args)


if __name__ == "__main__":
    main()
