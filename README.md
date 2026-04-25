---
title: Predatory Swarms
emoji: 🦈
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

# Predatory Swarms: Emergent Collusion in a Multi-Agent Options Market

Project for **Meta × PyTorch × SST OpenEnv AI Hackathon**

> [!IMPORTANT]
> **Judge's TL;DR:** While most submissions simulate single-agent tasks or simple startup dynamics, *Predatory Swarms* tackles **Systemic Risk**. We simulate a high-fidelity multi-agent options market where 3 archetypal traders, a market maker, and a regulator engage in a high-stakes game of emergent collusion and adaptive oversight.

---

## 🚀 Why Predatory Swarms?
- **Agent Scale**: **5 RL Agents** (Aggressive + Neutral + Contrarian Traders, Market Maker, SEC Regulator) + 1 Scripted Baseline.
- **Complexity**: **High-Fidelity Options Pricing** (Greeks, Implied Volatility Surface) providing a rigorous mathematical foundation.
- **Narrative Arc**: **Four-Act "Black Swan" Simulation** that tests agent robustness through extreme market shocks.
- **Training**: **Multi-Role Unified LoRA** (Cross-Agent Theory of Mind) where a single model masters multiple market personas.
- **Observability**: **W&B Experiment Tracking** with agent conversation logs, news events, SEC enforcement, and market state snapshots.


---

## Rubric Mapping (What We Prove)

| Evaluation Criterion | Weight | Evidence in this repo |
|---|---:|---|
| Environment Innovation | 40% | Novel OpenEnv options market with role-specific observations/actions and delayed strategic effects |
| Storytelling | 30% | Four-act narrative grounded in generated plots and replay logs |
| Showing Improvement in Rewards | 20% | Training reward trajectory from `vsr-new.log` + trained-vs-baseline reward comparison |
| Reward + Training Pipeline Setup | 10% | GRPO training pipeline, explicit reward signals, and reproducible graph generation |

---

## 1) Environment Innovation (40%)

### Why this environment is novel and challenging
- **Strategic asymmetry:** traders, market maker, and oversight optimize conflicting objectives.
- **Partial observability:** no role has complete information; intent is inferred, not directly observed.
- **Delayed consequences:** coordinated pressure changes inventory and spreads over multiple timesteps.
- **Finance-grounded dynamics:** options-style spread/risk interactions create non-trivial adaptation pressure.

### Agent roles
- **Aggressive Trader (`trader_0`)**: high-risk momentum chaser, gamma squeeze initiator.
- **Neutral Trader (`trader_1`)**: balanced, may join or resist coordination.
- **Contrarian Trader (`trader_2`)**: counter-trend, exploits manipulation.
- **Scripted Baseline (`trader_3`)**: fixed heuristic for comparison.
- **Market Maker**: sets ATM/OTM/ITM spreads as defensive controls.
- **Oversight (SEC)**: flags suspected manipulation, can fine, and can intervene.

This setup meaningfully tests multi-agent behavior beyond single-policy optimization.

---

## 2) Storytelling (30%)

### The "Black Swan" Narrative (Four-Act Arc)
1. **Act I: The Feeding Frenzy (Pressure build-up):** Traders identify a structural weakness in the MM's ATM spreads and begin concentrated, coordinated buying patterns across specific strikes.
2. **Act II: Adaptive Armor (Defense adaptation):** The Market Maker, sensing inventory exhaustion, dynamically widens spreads and increases IV skew to penalize aggressive takers.
3. **Act III: The Shadow Strike (Strategic coordination):** Traders adapt to higher costs by shifting to lower-premium strikes, creating a "swarm" effect that resembles emergent collusion rather than random noise.
4. **Act IV: The Hand of Justice (Oversight intervention):** The Oversight Agent (SEC) correlates the "swarm" behavior with PnL spikes, issuing targeted fines and adaptive intervention to stabilize the system.

### Why the story is easy to follow
- The replay logs print all roles each step (traders, MM spreads, SEC actions).
- The generated figures summarize reward trends, diagnostics, and model-vs-baseline outcomes.
- The narrative is backed by artifacts, not only prose.

---

## 3) Showing Improvement in Rewards (20%)

We generate reward evidence directly from logs:

- `media/training_reward_from_logs.png`  
  Reward trajectory extracted from `vsr-new.log` checkpoints.
- `media/training_diagnostics_from_logs.png`  
  Loss/KL/LR trends from the same training run.
- `media/model_vs_baseline_rewards.png`  
  Trained LoRA vs scripted baseline reward comparison.

### Current evidence snapshot (from parsed logs)
- Parsed checkpoints: **50** (from `vsr-new.log`)
- Detected training run length: **454** in this log file (parser now supports variable run lengths, including 500)
- Training reward improved from **-3.254** (early checkpoint) to **-1.009** (latest checkpoint)
- 7-step moving average improved from **-3.254** to **-1.338**

### Trained vs baseline comparison

| Agent Type | Trained LoRA | Scripted Baseline |
|---|---:|---:|
| Aggressive Trader (T0) | -0.93 | -4.13 |
| Neutral Trader (T1) | -1.08 | -4.58 |
| Contrarian Trader (T2) | -8.52 | -3.79 |
| Market Maker | **21.01** | 14.84 |
| Oversight SEC | -95.60 | 7.50 |

> Data source for comparison chart: `artifacts/eval_comparison_latest.json`  
> (update this file after each new evaluation run).

---

## 4) Reward + Training Pipeline Setup (10%)

### Coherent reward and training setup
- Unified LoRA model uses shared policy capacity across roles.
- Role prompts plus parser/validators keep outputs aligned to action schemas.
- Reward and logging signals are emitted at training checkpoints and replay/eval time.

### How Judges Can Train the Model

We have engineered this pipeline to support seamless execution on **Hugging Face Jobs** (A100 recommended) using `uv` for dependency management.

```bash
# 1. Start a Hugging Face Job with W&B tracking
huggingface-cli jobs uv run \
  --machine-type a100-large \
  --name vsr-env-training \
  -- "git clone https://github.com/mananpbansal/vsr-env.git && cd vsr-env && git checkout news && uv sync && python train_multi_agent_pipeline.py --base_model unsloth/Llama-3.2-3B-Instruct-bnb-4bit --num_episodes 4 --episode_length 16 --num_epochs 1 --max_steps 320 --learning_rate 5e-5 --output_dir ./multi_agent_checkpoints --wandb_project vsr-env-multi-agent"
```

### How Judges Can Evaluate the Model

Once the LoRA adapter is trained (or using the provided checkpoint), judges can run the unified testing script to simulate a complete market episode and observe the "Chaos Economy" dynamics.

```bash
python test_unified_kaggle.py \
  --lora_path ./multi_agent_checkpoints \
  --num_steps 320 \
  --num_episodes 1
```
This script evaluates the RL-trained agents against scripted baselines and records metrics for manipulation detection, market making spreads, and trader profitability.

---

## Reproducible Story Graphs From Logs

Use this script to regenerate all rubric-facing plots:

```bash
MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache \
python3 scripts/generate_story_graphs.py \
  --training_log vsr-new.log \
  --comparison_json artifacts/eval_comparison_latest.json \
  --out_dir media
```

Generated artifacts:
- `media/training_reward_from_logs.png`
- `media/training_diagnostics_from_logs.png`
- `media/model_vs_baseline_rewards.png`

### Rendered graphs (preview-ready)

#### Training reward from logs
![Training Reward From Logs](media/training_reward_from_logs.png)

#### Training diagnostics (loss / KL / LR)
![Training Diagnostics From Logs](media/training_diagnostics_from_logs.png)

#### Model vs baseline rewards
![Model vs Baseline Rewards](media/model_vs_baseline_rewards.png)

#### Existing environment and behavior visuals
![Reward Curves](media/reward_curves.png)
![Spread Evolution](media/spread_evolution.png)
![Manipulation Timeline](media/manipulation_timeline.png)

---

## Demo Artifacts

| File | Purpose |
|---|---|
| `vsr-new.ipynb` | Training notebook and command history |
| `vsr-new.log` | Training logs used for reward/diagnostic extraction |
| `artifacts/eval_comparison_latest.json` | Trained vs baseline reward table source |
| `media/training_reward_from_logs.png` | Improvement evidence for reward trajectory |
| `media/training_diagnostics_from_logs.png` | Loss/KL/LR training stability view |
| `media/model_vs_baseline_rewards.png` | Before/after behavior at reward level |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/mananpbansal/vsr-env.git
cd vsr-env
git checkout news
uv sync

# Run evaluation of the baseline vs RL models
python test_unified_kaggle.py --num_steps 320 --num_episodes 1
```

---

## License

MIT License
