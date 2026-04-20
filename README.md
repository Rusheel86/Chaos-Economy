---
title: Multi-Agent VSR-Env
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Multi-Agent VSR-Env

**A 12-actor RL environment where trader agents, a market maker, and an oversight agent co-evolve inside a live options market.**

Built for the **Meta × PyTorch × SST OpenEnv AI Hackathon** — **Theme #1: Multi-Agent Interactions**

---

## Hackathon Alignment

| Theme | Sub-theme | Fit |
|-------|-----------|-----|
| **Theme #1: Multi-Agent Interactions** | Primary | 10 traders compete, cooperate, and coordinate through prices |
| **Fleet AI: Scalable Oversight** | Bonus | Oversight agent monitors and flags other AI agents |
| **Halluminate: Multi-Actor Environments** | Bonus | 12 actors interact in a shared market state |

---

## The Pitch Arc

1. **10 RL traders** initially exploit a weak market maker
2. **Market maker adapts** by widening/tightening spreads to survive
3. **Traders shift** from naive bets to Greek-aware volatility trading
4. **Emergent behavior** appears: wash trading, gamma pressure, coalitions
5. **Oversight agent must detect** and intervene against manipulation

This is not just a finance simulator. It's a **strategic multi-agent world** for training:

- Competition
- Coordination through prices
- Oversight of other AI agents
- Long-horizon adaptation under partial observability

---

## Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│  10 Trader Agents (optimize PnL under risk)          │
│  → Buy/sell options, manage Greeks, exploit spreads  │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Market Maker (defends via spread control)           │
│  → Sets bid-ask spreads, manages inventory risk      │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Oversight Agent (monitors and intervenes)           │
│  → Detects manipulation, issues fines, halts agents  │
└─────────────────────────────────────────────────────┘
```

**Environment Properties:**
- **300 steps** per episode
- **Partial observability** — agents see own positions + market state
- **Delayed consequences** — manipulation effects surface later
- **Emergent behavior** — collusion, spoofing, gamma squeezes

---

## Quick Start

```bash
# Install
pip install -e .

# Run full 300-step demo
python inference_multi_agent.py --output replays/demo.json

# Generate visualization charts
python visualize_multi_agent.py --replay replays/demo.json --output media/

# Train a trader policy (on Kaggle/Colab with GPU)
python train_grpo.py --role trader --num_episodes 300
```

**Expected Demo Output:**
```
Step 50/300  | Avg Trader: 0.234 | MM: 0.156 | Oversight: 0.089
Step 100/300 | Avg Trader: 0.412 | MM: 0.203 | Oversight: 0.134
...
Episode Complete!
Total trades: 847
Total interventions: 12
```

---

## Role Rewards

| Role | Reward Objective |
|------|------------------|
| **Trader** | ΔPnL - fines - inventory_penalty - Greeks_violation |
| **Market Maker** | spread_PnL + flow_reward + quote_quality - inventory_risk |
| **Oversight** | TP_bonus - FP_penalty - FN_penalty + stability_bonus |

---

## Manipulation Detection

The oversight agent monitors for:

| Technique | Detection Signal |
|-----------|------------------|
| **Wash Trading** | Rapid buy/sell of same instrument |
| **Spoofing Pressure** | Oversized short-window order flow |
| **Gamma Pressure** | Concentrated directional gamma exposure |
| **Systemic Risk** | Destabilizing portfolio Greeks |

---

## Training Setup

We provide a GRPO training script for Unsloth/TRL:

| File | Purpose |
|------|---------|
| `train_grpo.py` | Main training loop |
| `train_grpo_colab.ipynb` | Kaggle dual-T4 notebook |
| `multi_agent/` | Environment implementation |

**Training command:**
```bash
accelerate launch train_grpo.py \
    --role trader \
    --base_model unsloth/Llama-3.2-3B-Instruct \
    --num_episodes 300
```

**Training Progress (example):**
- Step 10: reward = 0.15
- Step 20: reward = 0.45 (3x improvement)
- ...

---

## Demo Artifacts

After running inference and visualization:

| File | Content |
|------|---------|
| `replays/demo.json` | Full episode with rewards, trades, interventions |
| `media/reward_curves.png` | Reward evolution by role |
| `media/spread_evolution.png` | Market maker spread changes |
| `media/manipulation_timeline.png` | Detection events |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
| [REWARDS.md](REWARDS.md) | Reward function details |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Development notes |

---

## Why This Environment Matters

Most RL benchmarks are:
- Single-agent
- Short-horizon
- Fully observable

Real systems are:
- Multi-agent
- Long-horizon
- Partially observable

**Multi-Agent VSR-Env** trains agents for the real world: strategic interaction, adaptive defense, and oversight of other AI systems.

---

## License

MIT License
