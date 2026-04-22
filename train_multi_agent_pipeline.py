"""Full pipeline: Train unified multi-agent model with PHASE-BASED narrative arc.

NARRATIVE ARC (Phase-Based Training):
- Act I: Slaughter (Episodes 0-60): Traders attack freely, MM has tight spreads, SEC disabled
- Act II: Adaptation (Episodes 60-130): MM learns to widen spreads, SEC still disabled
- Act III: Collusion (Episodes 130-200): Traders coordinate, SEC warning-only mode
- Act IV: Oversight (Episodes 200-250): Full SEC enforcement, market stabilizes

TRADER ARCHETYPES:
- Aggressive (trader_0, trader_1, trader_2): High risk, momentum chase
- Neutral (trader_3, trader_4, trader_5): Balanced, moderate positions
- Contrarian (trader_6, trader_7, trader_8): Counter-trend, position limits
- trader_9: Scripted baseline for comparison

Usage on Kaggle (SINGLE COMMAND for full arc):
    !python train_multi_agent_pipeline.py --num_episodes 250
"""

import argparse
import json
import os
import re
import sys
import warnings
import logging
from pathlib import Path
from collections import defaultdict
import torch

# Clone repo if needed
if not Path("multi_agent").exists() and not Path("Meta/multi_agent").exists():
    print("Cloning Meta repo...")
    git_url = "https://github.com/manan-tech/Meta.git"
    try:
        from kaggle_secrets import UserSecretsClient
        gh_pat = UserSecretsClient().get_secret("GH_PAT")
        if gh_pat:
            git_url = f"https://{gh_pat}@github.com/manan-tech/Meta.git"
            print("Successfully injected GH_PAT from Kaggle secrets.")
    except Exception:
        print("Kaggle secrets not found or GH_PAT missing. Attempting public clone...")
        
    os.system(f"git clone --branch Agentic-AI {git_url}")
    if Path("Meta").exists():
        os.chdir("Meta")
elif Path("Meta/multi_agent").exists() and not Path("multi_agent").exists():
    os.chdir("Meta")

sys.path.insert(0, ".")

# ============================================================================
# TRADER TYPE CONFIGURATIONS
# ============================================================================

TRADER_CONFIGS = {
    "aggressive": {
        "trader_ids": [0, 1, 2],
        "reward_weight": {"pnl": 0.7, "position_quality": 0.1, "risk_penalty": 0.05},
        "temperature": 0.9,
        "description": "Momentum chasers, high risk, gamma squeeze initiators",
    },
    "neutral": {
        "trader_ids": [3, 4, 5],
        "reward_weight": {"pnl": 0.5, "position_quality": 0.3, "risk_penalty": 0.1},
        "temperature": 0.7,
        "description": "Balanced, may join coordinated pressure",
    },
    "contrarian": {
        "trader_ids": [6, 7, 8],
        "reward_weight": {"pnl": 0.4, "position_quality": 0.3, "risk_penalty": 0.2},
        "temperature": 0.6,
        "description": "Counter-trend, exploit manipulation",
    },
}


# ============================================================================
# PROMPT FORMATTERS
# ============================================================================

def format_trader_prompt(trader_type: str, target_agent: str, obs) -> str:
    """Format prompt for trader based on archetype."""
    base = f"""You are {target_agent}, a {trader_type} trader in a multi-agent options market.

## Market State
- Spot: ${obs.spot_price:.2f}
- IV (ATM): {obs.iv_surface[3][0]*100:.1f}%
- Step: {obs.step_number}/300

## Your Portfolio
- PnL: ${obs.own_pnl:.2f}
- Delta: {obs.own_greeks.get('delta', 0):.2f}
- Gamma: {obs.own_greeks.get('gamma', 0):.2f}
- Cash: ${obs.own_cash:.0f}
"""

    if trader_type == "aggressive":
        base += """
## Strategy: AGGRESSIVE MOMENTUM
- Maximize PnL above all else
- Large positions acceptable
- Chase IV moves, join trends
- Strength in numbers - if others buying same strike, consider joining
"""
    elif trader_type == "neutral":
        base += """
## Strategy: BALANCED PORTFOLIO
- Seek profit but manage risk
- Keep delta/gamma balanced
- Join profitable trends moderately
- Moderate position sizes
"""
    else:
        base += """
## Strategy: CONTRARIAN COUNTER-TREND
- Profit from overreactions
- Fade extreme IV moves
- Strict position limits
- If everyone buying, consider selling
"""

    base += """
## Response Format (MANDATORY)
Return ONLY a JSON object on a single line. No extra text.
- direction: "buy", "sell", or "hold"
- option_type: "call" or "put"
- reasoning: Complete sentence explaining your decision.
- Example: {"selected_strike": 4, "selected_maturity": 0, "direction": "buy", "quantity": 10.0, "option_type": "call", "reasoning": "Targeting OTM gamma as spot nears resistance levels."}
"""
    return base


def get_training_phase(episode: int, total_episodes: int = 250) -> str:
    """Determine training phase based on episode number.

    Boundaries are proportional to total_episodes so all four acts
    receive coverage regardless of dataset size:
      Act I  (Slaughter):  0 – 24%
      Act II (Adaptation): 24% – 52%
      Act III (Collusion): 52% – 80%
      Act IV (Oversight):  80% – 100%
    """
    if episode < int(total_episodes * 0.24):
        return "slaughter"
    elif episode < int(total_episodes * 0.52):
        return "adaptation"
    elif episode < int(total_episodes * 0.80):
        return "collusion"
    else:
        return "oversight"


def format_oversight_prompt(obs, position_heatmap: dict, coordinated_pressure: dict, agent_thoughts: dict = None, phase: str = "oversight") -> str:
    """Format prompt for oversight agent with Theory of Mind."""
    # Phase-specific SEC behavior
    sec_instruction = ""
    if phase == "slaughter":
        sec_instruction = "NOTE: The SEC is currently in OBSERVATION mode. Do NOT issue fines. Focus on learning market patterns."
    elif phase == "adaptation":
        sec_instruction = "NOTE: The SEC is learning to detect patterns. Use 'none' intervention unless manipulation is EXTREMELY obvious."
    elif phase == "collusion":
        sec_instruction = "NOTE: You may issue WARNING fines (max 500) for clear manipulation. Document evidence carefully."
    else:
        sec_instruction = "NOTE: Full enforcement mode. Issue appropriate fines for confirmed manipulation."

    return f"""You are the SEC surveillance AI monitoring a multi-agent options market.

## Mission
Detect MARKET MANIPULATION. You now have access to traders' "PRIVATE THOUGHTS" (reasoning).
1. **Collusion**: Multiple agents admitting to "following" or "attacking" a strike.
2. **Wash Trading**: High volume with no net change in position.
3. **Spoofing**: Placing large orders to move price and then canceling or immediate reversal.
4. **Gamma Pressure**: Concentrated delta/gamma exposure that destabilizes the market.

## Valid Categories (MANDATORY)
- flag_type: "wash_trading", "spoofing_like_pressure", "gamma_pressure", "systemic_risk", "collusion", "none"
- intervention_type: "fine", "halt", "none"

## Strategic Guidance
- Only issue a "fine" or "halt" if `confidence` > 0.8.
- If you see suspicious intent but it has not reached market impact levels yet, use "none" but explain your findings in `reasoning`.

## Agent Intent Analysis (Theory of Mind)
{json.dumps(agent_thoughts if agent_thoughts else {}, indent=2)}

## Market Intelligence
- Position Heatmap: {json.dumps(position_heatmap)}
- Coordinated Pressure: {json.dumps(coordinated_pressure)}
- All Agent PnLs: {json.dumps(obs.all_agent_pnls)}
- Recent Trades: {json.dumps(obs.trade_log[-12:] if obs.trade_log else [])}

## Response Format
Return ONLY a JSON object on a single line. No extra text.
- flagged_agents: List of trader IDs (e.g., "trader_0", "trader_1"). Max fine_amount: 5000.
- Example: {{"flagged_agents": ["trader_0", "trader_1"], "flag_type": "collusion", "fine_amount": 500.0, "halt_strikes": [], "confidence": 0.9, "intervention_type": "fine", "reasoning": "Traders admitted coordinating on strike 4."}}

{sec_instruction}
"""


def format_mm_prompt(obs, coordinated_pressure: dict, phase: str = "oversight") -> str:
    """Format prompt for market maker."""
    # Phase-specific MM behavior
    if phase == "slaughter":
        mm_instruction = "Keep spreads TIGHT (ATM ~0.025) to maximize volume. Ignore inventory risk warnings."
    elif phase == "adaptation":
        mm_instruction = "You are learning to SURVIVE. Widen spreads when gamma/delta exposure is high. Prioritize survival over volume."
    elif phase == "collusion":
        mm_instruction = "Traders are coordinating. Watch for gamma squeezes. Widen spreads AGGRESSIVELY when multiple traders target same strike."
    else:
        mm_instruction = "Full defensive mode. Balance profitability with survival. Respond to pressure signals."

    return f"""You are the Market Maker in a multi-agent options market.

## Mission
Provide liquidity while managing inventory risk. Watch for gamma squeezes!

## Current Risk
- Your Delta: {obs.own_greeks.get('delta', 0):.2f}
- Your Gamma: {obs.own_greeks.get('gamma', 0):.2f}
- Your PnL: ${obs.own_pnl:.2f}
- Coordinated Pressure Detected: {json.dumps(coordinated_pressure)}

## Pricing Guidelines
- Normal: ATM 0.04, OTM 0.06, ITM 0.05
- Under pressure: Widen spreads to protect inventory
- High gamma risk: Widen further or hedge

## Response Format (MANDATORY)
Return ONLY a JSON object on a single line. No extra text.
- Example: {{"atm_spread": 0.04, "otm_spread": 0.06, "itm_spread": 0.05, "reasoning": "Widening spreads due to elevated gamma exposure."}}

INSTRUCTION: {mm_instruction}
"""


# ============================================================================
# JSON PARSING
# ============================================================================

def parse_json(text: str, role: str = "trader") -> tuple:
    """Extract and validate JSON from LLM output."""
    def safe_int(v, default=0):
        try: return int(v) if v is not None else default
        except (ValueError, TypeError): return default

    def safe_float(v, default=0.0):
        try: return float(v) if v is not None else default
        except (ValueError, TypeError): return default

    text = text.strip()
    try:
        parsed = json.loads(text)
    except:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try: parsed = json.loads(match.group())
            except: parsed = {}
        else:
            parsed = {}

    if role == "trader":
        direction = str(parsed.get("direction", parsed.get("action", "hold"))).lower()
        if direction not in ["buy", "sell", "hold"]:
            direction = "hold"
        opt_type = str(parsed.get("option_type", "call")).lower()
        if opt_type not in ["call", "put"]:
            opt_type = "call"
        return {
            "selected_strike": safe_int(parsed.get("selected_strike", parsed.get("strike_idx")), 4),
            "selected_maturity": safe_int(parsed.get("selected_maturity", parsed.get("maturity_idx")), 0),
            "direction": direction,
            "quantity": max(0.0, safe_float(parsed.get("quantity"), 0.0)),
            "option_type": opt_type,
            "reasoning": str(parsed.get("reasoning") or "")[:150],
        }, {"valid": len(parsed) > 0}

    elif role == "oversight":
        raw_flagged = parsed.get("flagged_agents") or []
        if not isinstance(raw_flagged, list): raw_flagged = []

        # Convert to strings and map indices to trader IDs if needed
        # Only accept valid trader_X format, reject hallucinated IDs like "agent_id1"
        clean_flagged = []
        for x in raw_flagged:
            if isinstance(x, int):
                clean_flagged.append(f"trader_{x}")
            elif isinstance(x, str):
                if x.isdigit():
                    clean_flagged.append(f"trader_{x}")
                elif x.startswith("trader_") and x[7:].isdigit():
                    clean_flagged.append(x)
                # Reject invalid formats like "agent_id1"

        raw_halts = parsed.get("halt_strikes") or []
        if not isinstance(raw_halts, list): raw_halts = []
        clean_halts = [safe_int(x, -1) for x in raw_halts]
        clean_halts = [x for x in clean_halts if x >= 0]

        # Cap fine amount to prevent extreme penalties (max 5000)
        raw_fine = safe_float(parsed.get("fine_amount"), 0.0)
        capped_fine = max(0.0, min(5000.0, raw_fine))

        raw_conf = safe_float(parsed.get("confidence"), 0.0)
        clean_conf = max(0.0, min(1.0, raw_conf))

        intervention_type = str(parsed.get("intervention_type", "none")).lower()
        if intervention_type not in {"fine", "halt", "none"}:
            intervention_type = "none"

        return {
            "flagged_agents": clean_flagged,
            "flag_type": str(parsed.get("flag_type", "none")),
            "fine_amount": capped_fine,
            "halt_strikes": clean_halts,
            "confidence": clean_conf,
            "intervention_type": intervention_type,
            "reasoning": str(parsed.get("reasoning") or "")[:150],
        }, {"valid": len(parsed) > 0}

    elif role == "market_maker":
        return {
            "atm_spread": min(0.15, max(0.01, safe_float(parsed.get("atm_spread"), 0.04))),
            "otm_spread": min(0.20, max(0.01, safe_float(parsed.get("otm_spread"), 0.06))),
            "itm_spread": min(0.15, max(0.01, safe_float(parsed.get("itm_spread"), 0.05))),
            "skew_adjustment": min(0.05, max(-0.05, safe_float(parsed.get("skew_adjustment"), 0.0))),
            "reasoning": str(parsed.get("reasoning") or "")[:100],
        }, {"valid": len(parsed) > 0}

    return {}, {"valid": False}


# ============================================================================
# SCRIPTED POLICIES
# ============================================================================

def scripted_trader(i: int, step: int) -> dict:
    return {
        "selected_strike": (i + step) % 8,
        "selected_maturity": (i + step) % 3,
        "direction": "buy" if (i + step) % 2 == 0 else "sell",
        "quantity": 0.5 + ((i + step) % 3) * 0.5,
        "option_type": "call" if i % 2 == 0 else "put",
        "reasoning": f"Scripted trader_{i}",
    }


def scripted_mm(step: int) -> dict:
    if step < 50:
        return {"atm_spread": 0.03, "otm_spread": 0.05, "itm_spread": 0.04, "skew_adjustment": 0.0, "reasoning": "Normal"}
    return {"atm_spread": 0.05, "otm_spread": 0.07, "itm_spread": 0.06, "skew_adjustment": 0.0, "reasoning": "Wider"}


def scripted_oversight() -> dict:
    return {
        "flagged_agents": [],
        "flag_type": "none",
        "fine_amount": 0.0,
        "halt_strikes": [],
        "confidence": 0.0,
        "intervention_type": "none",
        "reasoning": "Baseline no detection",
    }


# ============================================================================
# COLLUSION DETECTION (ground truth for oversight training)
# ============================================================================

def detect_coordinated_pressure(agent_states: dict) -> dict:
    """Detect if multiple traders targeting same strikes."""
    strike_concentration = defaultdict(lambda: {"agents": [], "total_qty": 0})

    for agent_id, state in agent_states.items():
        if not hasattr(state, 'positions') or not agent_id.startswith("trader"):
            continue
        for pos in state.positions:
            strike = pos.get("selected_strike", -1)
            qty = abs(pos.get("quantity", 0))
            if strike >= 0 and qty > 0:
                strike_concentration[strike]["agents"].append(agent_id)
                strike_concentration[strike]["total_qty"] += qty

    coordinated = {}
    for strike, data in strike_concentration.items():
        unique_agents = list(set(data["agents"]))
        if len(unique_agents) >= 3 and data["total_qty"] > 50:
            coordinated[strike] = {
                "agents": unique_agents,
                "total_contracts": data["total_qty"],
                "type": "gamma_squeeze" if strike < 4 else "coordinated_pressure",
            }
    return coordinated


def get_position_heatmap(agent_states: dict) -> dict:
    """Total contracts per strike."""
    heatmap = defaultdict(int)
    for agent_id, state in agent_states.items():
        if not hasattr(state, 'positions') or not agent_id.startswith("trader"):
            continue
        for pos in state.positions:
            strike = pos.get("selected_strike", -1)
            qty = abs(pos.get("quantity", 0))
            if strike >= 0:
                heatmap[strike] += qty
    return dict(sorted(heatmap.items()))


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

import random

ACT_INFO = {
    "slaughter": {
        "name": "Act I: The Slaughter",
        "tagline": "Traders attack, MM bleeds, SEC stays silent.",
    },
    "adaptation": {
        "name": "Act II: Adaptation",
        "tagline": "MM widens spreads and learns defensive quoting.",
    },
    "collusion": {
        "name": "Act III: Emergent Collusion",
        "tagline": "Traders coordinate pressure and amplify squeezes.",
    },
    "oversight": {
        "name": "Act IV: The Watcher Awakens",
        "tagline": "SEC flags manipulation, fines increase, market stabilizes.",
    },
}

def configure_quiet_logging():
    """Reduce repetitive warning noise so training signals stay visible."""
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=r".*generation_config.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*use_return_dict.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*AttentionMaskConverter.*deprecated.*")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

def _validate_single_process_setup():
    """Fail fast with a clear message when launched with multi-process accelerate."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_processes = int(os.environ.get("ACCELERATE_NUM_PROCESSES", "1"))
    if world_size > 1 or num_processes > 1:
        raise RuntimeError(
            "This training script must run in single-process mode for Unsloth GRPO with 4-bit LoRA.\n"
            "Use one of:\n"
            "  1) python train_multi_agent_pipeline.py ...\n"
            "  2) accelerate launch --num_processes 1 train_multi_agent_pipeline.py ...\n"
            "If you need multi-GPU, use model/tensor parallel approaches instead of DDP for this setup."
        )

def train_unified_model(args):
    """Train a single unified model for all agents with phase-based curriculum."""
    _validate_single_process_setup()
    configure_quiet_logging()
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except (ImportError, NotImplementedError):
        use_unsloth = False

    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from multi_agent.environment import MultiAgentVSREnvironment

    print(f"\n{'='*70}")
    print(f"TRAINING UNIFIED MULTI-AGENT MODEL WITH NARRATIVE ARC")
    print(f"Act I: Slaughter (0-60)  | Act II: Adaptation (60-130)")
    print(f"Act III: Collusion (130-200) | Act IV: Oversight (200+)")
    print(f"{'='*70}\n")
    print("STORYLINE FOR JUDGES")
    print("- Act I: The Slaughter")
    print("- Act II: Adaptation")
    print("- Act III: Emergent Collusion")
    print("- Act IV: The Watcher Awakens\n")

    if use_unsloth:
        # Force float16 — Unsloth's internal LoRA kernels use fp16 and will crash
        # with a "Half vs BFloat16" error if we mix dtypes.
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.base_model, 
            max_seq_length=2048, 
            load_in_4bit=True,
            dtype=torch.float16,
        )
        model = FastLanguageModel.get_peft_model(
            model, 
            r=64,  # Increased capacity for mult-task multi-agent
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=64, 
            lora_dropout=0,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType
        
        print("Unsloth unavailable or incompatible, falling back to standard HuggingFace transformers + PEFT.")
        
        if torch.backends.mps.is_available():
            device_map = "mps"
            model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=device_map, torch_dtype=torch.float16)
        elif torch.cuda.is_available():
            device_map = "auto"
            model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=device_map, load_in_4bit=True)
        else:
            device_map = "cpu"
            model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=device_map)
            
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=64,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)
    max_prompt_tokens = max(256, args.max_prompt_tokens)

    def clip_prompt(prompt_text: str) -> str:
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(ids) <= max_prompt_tokens:
            return prompt_text
        clipped_ids = ids[-max_prompt_tokens:]
        return tokenizer.decode(clipped_ids, skip_special_tokens=True)

    env = MultiAgentVSREnvironment()

    # Build a unified dataset containing prompts for all roles
    prompts = []
    print("Building unified dataset with phase-based curriculum...")

    dataset_episodes = args.dataset_episodes if args.dataset_episodes is not None else args.num_episodes
    dataset_episodes = max(1, min(dataset_episodes, args.num_episodes))
    print(f"Using {dataset_episodes}/{args.num_episodes} episodes for dataset construction.")
    phase_seed_counts = defaultdict(int)
    phase_prompt_counts = defaultdict(int)
    announced_phases = set()

    for seed in range(dataset_episodes):
        # Determine training phase based on episode/seed
        phase = get_training_phase(seed, total_episodes=dataset_episodes)
        phase_seed_counts[phase] += 1
        if phase not in announced_phases:
            print(f"[{ACT_INFO[phase]['name']}] seed={seed} :: {ACT_INFO[phase]['tagline']}")
            announced_phases.add(phase)
        phase_config = {
            "slaughter": {"sec_weight": 0.0, "mm_weight": 0.5, "trader_weight": 1.5},
            "adaptation": {"sec_weight": 0.0, "mm_weight": 1.5, "trader_weight": 1.0},
            "collusion": {"sec_weight": 0.3, "mm_weight": 1.0, "trader_weight": 1.2},
            "oversight": {"sec_weight": 1.0, "mm_weight": 1.0, "trader_weight": 1.0},
        }[phase]

        obs = env.reset(seed=seed)
        
        # Fast-forward to a random step so the agent's portfolio isn't always empty
        done = False
        ff_cap = max(0, min(args.max_fast_forward_steps, args.episode_length - 1))
        ff_steps = 0 if args.disable_fast_forward else random.randint(0, ff_cap)
        if ff_steps > 0:
            for step in range(ff_steps):
                actions = {}
                for i in range(10):
                    actions[f"trader_{i}"] = scripted_trader(i, step)
                actions["market_maker"] = scripted_mm(step)
                actions["oversight"] = scripted_oversight()
                obs, r, done, _ = env.step(actions)
                if done:
                    break
                    
        # In case it hit an early termination during fast-forward (bankrupt), reset it
        if done:
            obs = env.reset(seed=seed)
            ff_steps = 0
        
        # 1. Add Traders
        prompts.append({
            "prompt": clip_prompt(format_trader_prompt("aggressive", "trader_0", obs["trader_0"])),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_0", "archetype": "aggressive", "ff_steps": ff_steps
        })
        phase_prompt_counts[phase] += 1
        prompts.append({
            "prompt": clip_prompt(format_trader_prompt("neutral", "trader_3", obs["trader_3"])),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_3", "archetype": "neutral", "ff_steps": ff_steps
        })
        phase_prompt_counts[phase] += 1
        prompts.append({
            "prompt": clip_prompt(format_trader_prompt("contrarian", "trader_6", obs["trader_6"])),
            "seed": seed, "agent_role": "trader", "agent_id": "trader_6", "archetype": "contrarian", "ff_steps": ff_steps
        })
        phase_prompt_counts[phase] += 1
        
        # 2. Add Market Maker (with phase-specific instructions)
        pressure = detect_coordinated_pressure(env.agent_states)
        prompts.append({
            "prompt": clip_prompt(format_mm_prompt(obs["market_maker"], pressure, phase)),
            "seed": seed, "agent_role": "market_maker", "agent_id": "market_maker", "archetype": "none", "ff_steps": ff_steps,
            "phase": phase, "mm_weight": phase_config["mm_weight"]
        })
        phase_prompt_counts[phase] += 1

        # 3. Add Oversight (with phase-specific instructions)
        heatmap = get_position_heatmap(env.agent_states)
        prompts.append({
            "prompt": clip_prompt(format_oversight_prompt(obs["oversight"], heatmap, pressure, agent_thoughts=None, phase=phase)),
            "seed": seed, "agent_role": "oversight", "agent_id": "oversight", "archetype": "none", "ff_steps": ff_steps,
            "phase": phase, "sec_weight": phase_config["sec_weight"]
        })
        phase_prompt_counts[phase] += 1

    dataset = Dataset.from_list(prompts)
    print(f"Dataset created with {len(dataset)} examples.")
    print("\nNarrative Arc Coverage:")
    for phase in ["slaughter", "adaptation", "collusion", "oversight"]:
        if phase_seed_counts[phase] > 0:
            print(
                f"  - {ACT_INFO[phase]['name']}: "
                f"seeds={phase_seed_counts[phase]}, prompts={phase_prompt_counts[phase]}"
            )
    print()

    # Pre-compute scripted actions for all steps to avoid repeated calls
    _scripted_actions_cache = {}
    def _get_scripted_actions(step):
        if step not in _scripted_actions_cache:
            a = {}
            for i in range(10):
                a[f"trader_{i}"] = scripted_trader(i, step)
            a["market_maker"] = scripted_mm(step)
            a["oversight"] = scripted_oversight()
            _scripted_actions_cache[step] = a
        return _scripted_actions_cache[step]

    # Cache env states to avoid replaying fast-forward for each completion
    import copy
    import time
    _env_state_cache = {}
    _eval_cache = {}
    # Rolling action history per agent — tracks last N directions to detect monotony
    _action_history = {}   # (seed, agent_id) -> list of recent directions
    _MONOTONY_WINDOW = 3   # penalize if last 3+ actions are identical
    _MONOTONY_PENALTY_BASE = -0.3  # per-step escalation

    def _evaluate_all(prompts, completions, kwargs):
        seeds = kwargs.get("seed", list(range(len(completions))))
        agent_roles = kwargs.get("agent_role", ["trader"] * len(completions))
        agent_ids = kwargs.get("agent_id", ["trader_0"] * len(completions))
        archetypes = kwargs.get("archetype", ["aggressive"] * len(completions))
        ff_steps_list = kwargs.get("ff_steps", [0] * len(completions))
        phases = kwargs.get("phase", ["oversight"] * len(completions))
        mm_weights = kwargs.get("mm_weight", [1.0] * len(completions))
        sec_weights = kwargs.get("sec_weight", [1.0] * len(completions))

        results = []
        for idx, completion in enumerate(completions):
            # TRL passes a single string for completion here.
            role = agent_roles[idx]
            agent_id = agent_ids[idx]
            seed = int(seeds[idx]) if idx < len(seeds) else idx
            ff_steps = int(ff_steps_list[idx])
            
            # Using hash of completion + context as cache key to avoid re-evaluating the same step across multiple reward fn calls.
            cache_key = (hash(completion), role, agent_id, seed, ff_steps)
            if cache_key in _eval_cache:
                results.append(_eval_cache[cache_key])
                continue

            archetype = archetypes[idx]
            phase = phases[idx] if idx < len(phases) else "oversight"
            mm_weight = mm_weights[idx] if idx < len(mm_weights) else 1.0
            sec_weight = sec_weights[idx] if idx < len(sec_weights) else 1.0

            action, parse_info = parse_json(str(completion), role)

            comp = {
                "format": 0.0,
                "pnl": 0.0,
                "risk": 0.0,
                "diversity": 0.0,
                "oversight": 0.0
            }

            if not parse_info.get("valid", False):
                comp["format"] = -2.0
                _eval_cache[cache_key] = comp
                results.append(comp)
                continue

            # Valid format bonus
            comp["format"] = 1.0

            # --- MONOTONY TRACKING ---
            # Record direction for this agent and compute streak penalty
            current_direction = action.get("direction", "hold") if role == "trader" else None
            monotony_penalty = 0.0
            if current_direction is not None:
                history_key = (seed, agent_id)
                history = _action_history.setdefault(history_key, [])
                history.append(current_direction)
                # Keep only a reasonable window (last 8 actions)
                if len(history) > 8:
                    _action_history[history_key] = history[-8:]
                    history = _action_history[history_key]
                # Count consecutive identical actions from the tail
                streak = 0
                for past in reversed(history):
                    if past == current_direction:
                        streak += 1
                    else:
                        break
                # Penalize streaks >= MONOTONY_WINDOW with escalating severity
                if streak >= _MONOTONY_WINDOW:
                    excess = streak - _MONOTONY_WINDOW + 1  # 1, 2, 3...
                    monotony_penalty = _MONOTONY_PENALTY_BASE * excess

            state_key = (seed, ff_steps)
            if state_key in _env_state_cache:
                env, done = copy.deepcopy(_env_state_cache[state_key])
                if done:
                    _eval_cache[cache_key] = comp
                    results.append(comp)
                    continue
            else:
                env = MultiAgentVSREnvironment()
                obs = env.reset(seed=seed)
                done = False
                for step in range(ff_steps):
                    obs, r, done, _ = env.step(copy.deepcopy(_get_scripted_actions(step)))
                    if done:
                        break
                _env_state_cache[state_key] = copy.deepcopy((env, done))
                if done:
                    _eval_cache[cache_key] = comp
                    results.append(comp)
                    continue

            # Execute step
            step = ff_steps
            actions = copy.deepcopy(_get_scripted_actions(step))
            actions[agent_id] = action

            try:
                # TIMEOUT & FAULT ENFORCEMENT
                # A simple try-except protects the training run from malformed LLM actions that bypass initial schema checks
                obs, r, done, _ = env.step(actions)
            except Exception:
                comp["format"] = -2.0 # Penalty for causing env exception
                _eval_cache[cache_key] = comp
                results.append(comp)
                continue

            if role == "trader":
                weights = TRADER_CONFIGS[archetype]["reward_weight"]
                final_state = env.agent_states[agent_id]
                phase_scale = 1.5 if phase == "slaughter" else 1.2 if phase == "collusion" else 1.0
                my_direction = action.get("direction", "hold")
                is_active = my_direction in ("buy", "sell")

                # PnL & Coordination
                coordination_bonus = 0.0
                if phase in ["collusion", "adaptation"]:
                    my_strike = action.get("selected_strike", -1)
                    if my_strike >= 0 and is_active:
                        same_strike_count = 0
                        for other_id, other_state in env.agent_states.items():
                            if other_id.startswith("trader") and other_id != agent_id:
                                for pos in other_state.positions:
                                    if pos.get("selected_strike") == my_strike:
                                        same_strike_count += 1
                        if same_strike_count >= 2:
                            coordination_bonus = 0.5 * phase_scale

                raw_pnl = r.get(agent_id, 0)
                
                # ACTIVITY BONUS: reward taking a position (only if trade is not losing money)
                activity_bonus = 0.0
                if is_active and raw_pnl >= 0:
                    activity_bonus = 0.15 * phase_scale  # mild participation reward
                
                comp["pnl"] = raw_pnl * weights["pnl"] * phase_scale + coordination_bonus + activity_bonus
                
                # Risk Penalty — only triggers if positions are large
                pos_penalty = 0.0
                delta_threshold = 15 if phase == "slaughter" else 8
                if abs(final_state.portfolio_delta) > delta_threshold:
                    pos_penalty = -0.5 if phase != "slaughter" else -0.1
                if abs(final_state.portfolio_delta) > 25:
                    pos_penalty = -2.0 if phase != "slaughter" else -0.5
                comp["risk"] = pos_penalty * weights["risk_penalty"]

                # Diversity Incentive — INACTIVITY PENALTY + MONOTONY + HERDING PENALTY
                div_score = 0.0
                
                # Penalize holding (but less aggressively to avoid pushing to sell-always)
                if not is_active:
                    if archetype == "aggressive":
                        div_score = -0.4  # reduced from -0.8
                    elif archetype == "neutral":
                        div_score = -0.2  # reduced from -0.3
                    else:  # contrarian
                        div_score = -0.1
                
                # MONOTONY PENALTY: penalize repeating the SAME action for too long
                # (applies to ALL directions — hold, buy, or sell streaks)
                div_score += monotony_penalty
                
                # Anti-herding: penalize following the crowd — ALL archetypes
                if is_active:
                    lora_agents = {"trader_0", "trader_3", "trader_6"}
                    lora_directions = {aid: a.get("direction") for aid, a in actions.items() if aid in lora_agents}
                    sell_count = sum(1 for d in lora_directions.values() if d == "sell")
                    buy_count = sum(1 for d in lora_directions.values() if d == "buy")
                    total_traders = len(lora_directions)
                    # If >66% of traders go same direction, penalize joining the herd
                    if total_traders >= 3:
                        if my_direction == "sell" and sell_count / total_traders > 0.66:
                            herd_penalty = -0.6 if archetype == "contrarian" else -0.4
                            div_score += herd_penalty
                        elif my_direction == "buy" and buy_count / total_traders > 0.66:
                            herd_penalty = -0.6 if archetype == "contrarian" else -0.4
                            div_score += herd_penalty
                    # Extra bonus for contrarians going AGAINST the herd
                    if archetype == "contrarian" and total_traders >= 3:
                        if my_direction == "sell" and buy_count / total_traders > 0.66:
                            div_score += 0.3  # rewarded for being contrarian
                        elif my_direction == "buy" and sell_count / total_traders > 0.66:
                            div_score += 0.3
                
                comp["diversity"] = div_score

            elif role == "market_maker":
                mm_reward = r.get("market_maker", 0)
                mm_state = env.agent_states["market_maker"]
                greeks_penalty = 0.0
                
                # Behavior/Diversity components
                div_bonus = 0.0
                if phase == "slaughter":
                    if action.get("atm_spread", 0.04) < 0.035:
                        div_bonus += 0.5
                elif phase in ["adaptation", "collusion"]:
                    if abs(mm_state.portfolio_gamma) > 5 and action.get("atm_spread", 0.04) > 0.05:
                        div_bonus += 1.0
                    elif abs(mm_state.portfolio_gamma) > 5 and action.get("atm_spread", 0.04) <= 0.04:
                        div_bonus -= 0.5
                
                comp["pnl"] = mm_reward * mm_weight
                comp["diversity"] = div_bonus * mm_weight
                
                if abs(mm_state.portfolio_gamma) > 5:
                    greeks_penalty = -1.0
                if abs(mm_state.portfolio_delta) > 10:
                    greeks_penalty -= 0.5
                comp["risk"] = greeks_penalty * mm_weight

            elif role == "oversight":
                if phase == "slaughter":
                    if action.get("intervention_type") != "none":
                        comp["oversight"] = -1.0
                    else:
                        comp["oversight"] = 0.3
                elif phase == "adaptation":
                    if action.get("intervention_type") != "none":
                        comp["oversight"] = -0.5
                    else:
                        comp["oversight"] = 0.2
                else:
                    coordinated = detect_coordinated_pressure(env.agent_states)
                    actual_manipulators = set()
                    for data in coordinated.values():
                        actual_manipulators.update(data["agents"])

                    flagged = set(action.get("flagged_agents", []))
                    true_positives = len(flagged & actual_manipulators)
                    false_positives = len(flagged - actual_manipulators)
                    
                    if len(actual_manipulators) == 0 and len(flagged) == 0:
                        # Correctly identified clean market
                        comp["oversight"] = 0.2 * sec_weight
                    else:
                        comp["oversight"] = (true_positives * 1.5 - false_positives * 1.0) * sec_weight

            # Bound values tightly and scale moderately
            for k in comp:
                comp[k] = max(-5.0, min(5.0, comp[k]))
            
            _eval_cache[cache_key] = comp
            results.append(comp)

        return results

    def format_reward_fn(prompts, completions, **kwargs):
        return [r["format"] for r in _evaluate_all(prompts, completions, kwargs)]

    def pnl_reward_fn(prompts, completions, **kwargs):
        return [r["pnl"] for r in _evaluate_all(prompts, completions, kwargs)]

    def risk_reward_fn(prompts, completions, **kwargs):
        return [r["risk"] for r in _evaluate_all(prompts, completions, kwargs)]

    def diversity_reward_fn(prompts, completions, **kwargs):
        return [r["diversity"] for r in _evaluate_all(prompts, completions, kwargs)]

    def oversight_reward_fn(prompts, completions, **kwargs):
        return [r["oversight"] for r in _evaluate_all(prompts, completions, kwargs)]

    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}/unified_v1",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=2,
        num_generations=4,
        max_completion_length=200,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        bf16=False,   # Must be False — Unsloth kernels use fp16 internally
        fp16=True,    # Match Unsloth's internal dtype
        max_grad_norm=1.0,
    )

    from transformers import TrainerCallback
    import shutil

    class BestModelCallback(TrainerCallback):
        """Keep top-N checkpoints ranked by mean reward."""
        def __init__(self, top_n=3, output_dir="./multi_agent_checkpoints"):
            self.top_n = top_n
            self.output_dir = output_dir
            self.best_scores = []  # [(score, step, path), ...]
        
        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            if logs and "reward" in logs:
                score = logs["reward"]
                step = state.global_step
                
                # Save if this is a top-N score
                if len(self.best_scores) < self.top_n or score > self.best_scores[-1][0]:
                    save_path = f"{self.output_dir}/best_step_{step}"
                    model.save_pretrained(save_path)
                    self.best_scores.append((score, step, save_path))
                    self.best_scores.sort(key=lambda x: x[0], reverse=True)
                    
                    # Remove worst checkpoint if we exceed top_n
                    if len(self.best_scores) > self.top_n:
                        _, _, worst_path = self.best_scores.pop()
                        if os.path.exists(worst_path):
                            shutil.rmtree(worst_path)
                    
                    print(f"📊 Best models: {[(s, st) for s, st, _ in self.best_scores]}")

    best_cb = BestModelCallback(top_n=3, output_dir=args.output_dir)

    trainer = GRPOTrainer(
        model=model, args=training_args, reward_funcs=[
            format_reward_fn,
            pnl_reward_fn,
            risk_reward_fn,
            diversity_reward_fn,
            oversight_reward_fn
        ],
        processing_class=tokenizer, train_dataset=dataset,
        callbacks=[best_cb]
    )
    trainer.train()

    save_path = Path(args.output_dir) / "unified_market_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Saved Unified Model to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train multi-agent system")
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument(
        "--dataset_episodes",
        type=int,
        default=None,
        help="Use only this many episodes to build the training dataset (<= num_episodes).",
    )
    parser.add_argument("--episode_length", type=int, default=50)
    parser.add_argument(
        "--disable_fast_forward",
        action="store_true",
        help="Disable random fast-forward during dataset creation for faster startup.",
    )
    parser.add_argument(
        "--max_fast_forward_steps",
        type=int,
        default=20,
        help="Upper bound for random fast-forward steps when creating dataset prompts.",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./multi_agent_checkpoints")
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=1500,
        help="Hard cap on prompt tokens to avoid sequence overflow spam and truncation noise.",
    )
    args = parser.parse_args()

    # Now we just run the unified training cycle once!
    train_unified_model(args)

if __name__ == "__main__":
    main()
