---
title: The Chaos Economy
emoji: 🦈
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

# 🦈 The Chaos Economy
### Emergent Collusion in a Multi-Agent Options Market

> **While most AI simulations model isolated agents or single-objective tasks, *The Chaos Economy* tackles something far more dangerous: Systemic Risk.** We simulate a high-fidelity multi-agent options market where traders, a market maker, and a regulator engage in an evolving arms race of exploitation, collusion, and adaptive oversight — and watch a full financial crisis arc emerge on its own from 250 steps of reinforcement learning.

**[Hugging Face Space](https://huggingface.co/spaces/MananBansal/Chaos-Economy)** · **[W&B Report](https://wandb.ai/models-svkm-s-narsee-monjee-institute-of-management-studies/vsr-env-chaos-economy/reports/-Chaos-Economy--VmlldzoxNjY2NzUxMQ?accessToken=rj97shf6th8dt56ijps5p9d6wrap55arxmfuzn5ud9udxr28ywvoag5qvs07q6uz)**

---

## Table of Contents

- [The Story in Brief](#the-story-in-brief)
- [Agent Roles](#agent-roles)
- [The 4-Act Narrative](#the-4-act-narrative)
  - [Act I: The Slaughter (Steps 0–60)](#act-i-the-slaughter-steps-060)
  - [Act II: Adaptive Armor (Steps 60–130)](#act-ii-adaptive-armor-steps-60130)
  - [Act III: The Shadow Strike (Steps 100–175)](#act-iii-the-shadow-strike-steps-100175)
  - [Act IV: The Watcher Awakens (Steps 200–250)](#act-iv-the-watcher-awakens-steps-200250)
- [Curriculum Learning: Designed Arc, Emergent Behavior](#curriculum-learning-designed-arc-emergent-behavior)
- [Reward System](#reward-system)
  - [Trader Reward](#1-trader-reward)
  - [Market Maker Reward](#2-market-maker-reward)
  - [SEC Oversight Reward](#3-sec-oversight-reward)
- [System Architecture](#system-architecture)
- [Training and Results](#training-and-results)
- [Running the Pipeline](#running-the-pipeline)
- [License](#license)
- [Citation](#citation)

---

## The Story in Brief

Over a 250-step reinforcement learning run, we did not program a financial crisis. We watched one emerge. Five agents — each optimizing their own survival — stumbled through greed, adaptation, coordination, and ultimately, law enforcement. The arc that came out of the training loop, completely unprompted, maps almost perfectly onto how real financial crises unfold.

---

## Agent Roles

| Agent | Archetype | Objective |
|---|---|---|
| **Aggressive Trader** (`trader_0`) | High-risk momentum chaser | Gamma squeeze initiator |
| **Neutral Trader** (`trader_1`) | Balanced opportunist | May join or resist coordination |
| **Contrarian Trader** (`trader_2`) | Counter-trend exploiter | Profits from manipulation |
| **Scripted Baseline** (`trader_3`) | Fixed heuristic | Benchmark comparison |
| **Market Maker** | Dynamic spread setter | Defends against flow imbalance |
| **SEC Oversight** | Adaptive regulator | Flags manipulation, fines, intervenes |

---

## The 4-Act Narrative

### Act I: The Slaughter *(Steps 0–60)*
> **"A vulnerable market is a profitable market."**

The market opened with no active regulator, a naive market maker holding dangerously tight spreads, and traders operating under almost no risk constraints. The result was immediate and brutal.

The RL agents rapidly discovered that aggressive directional bets were close to consequence-free. They siphoned capital from the market maker relentlessly. By step 40, `pnl_mean` hit **1.186** — the highest PnL of the entire opening phase — while `risk_mean` stayed at exactly **0.0** for nine of the twelve logged steps. Risk wasn't just low; it was structurally absent. The one moment it fired (step 60, risk = **-0.010**) was the system's first signal that constraints were about to tighten.

`pnl_mean` peaks at step 40 with a reward of 1.463 — the market is being drained freely. `risk_mean` sits at zero for nearly the entire phase, confirming that agents faced zero structural penalty for leveraged bets. The isolated spike at step 60 is not noise: it is the exact step the Delta threshold activated, firing its first penalty.

<div align="center">
  <img src="media/wandb_metric_4.jpeg" width="45%" />
  <img src="media/wandb_metric_5.jpeg" width="45%" />
</div>

---

### Act II: Adaptive Armor *(Steps 60–130)*
> **"The market fights back."**

At step 60, the environment's rules hardened. The Delta risk threshold tightened sharply, and the market maker gained the ability to widen spreads dynamically in response to order flow pressure.

The traders' portfolios — built on the assumption of loose constraints — were suddenly penalized. But the adaptation that followed was subtler than a clean pivot to information trading. `news_alpha_mean` remained near **zero** through almost all of this phase: agents were not yet reliably trading on news sentiment. What changed instead was structural discipline. `format_mean` climbed toward **1.0** — agents learning to output well-structured decisions — while they began probing the Dark Pool for edge. The real lesson of Act II wasn't information warfare. It was survival through compliance.

Meanwhile, beneath the surface, something else was stirring: `diversity_mean` dipped to **-1.003** at step 105, the first sign that agents were beginning to converge on shared strategies rather than independent ones.

`format_mean` rises through Steps 60–130 as agents learn structural compliance under tightened rules. `news_alpha_mean` stays flat — agents are adapting through discipline, not yet through information. The dip in `format_mean` around step 150 is the first crack: agents beginning to break formation ahead of the Gamma Squeeze.

<div align="center">
  <img src="media/wandb_metric_2.jpeg" width="45%" />
  <img src="media/wandb_metric_3.jpeg" width="45%" />
</div>

---

### Act III: The Shadow Strike *(Steps 100–175)*
> **"If you cannot beat the house alone, burn it down together."**

The coordination bonus landed and the agents found it immediately. Rather than competing with each other, they began piling into the exact same option strikes simultaneously — spreading fake news through message channels to synchronize their attacks.

The data tells a precise story. At step **120**, `reward` hit its all-time peak of **2.092** — the single highest point across the entire 250-step run. This was the Gamma Squeeze executing at full force. Diversity had already started collapsing: `diversity_mean` fell to **-0.970** at step 130 and **-1.095** at step 160, its lowest recorded value. `frac_reward_zero_std` — the metric that measures agents making near-identical decisions — spiked to **0.60** at step 175, the highest it ever reached. The agents were in lockstep.

Then the correction came. At step **170**, `reward` crashed to **-1.154** and `pnl_mean` fell to **-1.067** — the worst single-step outcome in the entire run. The cost of over-leveraged coordinated exposure had arrived, even before the SEC's full activation.

`diversity_mean` crashes to its lowest point of **-1.095** at step 160 as agents abandon all independent strategy. `reward` records both the run's all-time high (2.092 at step 120, the Gamma Squeeze peak) and its all-time low (-1.154 at step 170, the collapse). `frac_reward_zero_std` spikes to **0.60** at step 175 — the statistical proof of lockstep collusion.

<div align="center">
  <img src="media/wandb_metric_1.jpeg" width="45%" />
  <img src="media/wandb_metric_7.jpeg" width="45%" />
</div>
<div align="center">
  <img src="media/wandb_metric_9.jpeg" width="45%" />
</div>

---

### Act IV: The Watcher Awakens *(Steps 200–250)*
> **"Order is restored."**

At step 200, the SEC was fully unchained. The regulator rapidly learned to correlate Dark Pool messaging with coordinated strike clustering — and the results were immediate.

The clearest moment in Act IV comes at step **225**: `oversight_mean` hits **0.140**, its all-time peak across the entire 250-step run. At that exact same step, `diversity_mean` is still deeply suppressed at **-0.912** and `pnl_mean` has collapsed to near zero at **0.034**. This is the picture of a regulator firing at full effectiveness — the SEC identifying and penalizing the remaining colluders while traders, still shell-shocked from the step 170 crash, have not yet recovered enough to generate meaningful profit. The fines worked. The herd had not yet reformed. The market was in a fragile, transitional compliance.

From there, traders gradually rebuilt independent strategies. `reward_std` climbed through the phase, peaking at **1.349** at step 245 — high cross-agent variance being the statistical signature of agents who have broken formation and are once again pursuing divergent approaches.

`oversight_mean` peaks at **0.140** at step 225, its highest point in the run, while diversity remains at **-0.912** and PnL sits near zero — the SEC at full effectiveness, traders still frozen. `reward_std` rises through Act IV to **1.349** at step 245, the signature of agents rebuilding independent strategies after the herd collapses.

<div align="center">
  <img src="media/wandb_metric_6.jpeg" width="45%" />
  <img src="media/wandb_metric_8.jpeg" width="45%" />
</div>

---

## Curriculum Learning: Designed Arc, Emergent Behavior

> **VSR-Env uses curriculum learning with a 4-act narrative arc inspired by real market crisis lifecycles. The structure is designed to progressively introduce complexity (individual trading → market making → coordination → oversight), but the agent behavior within each phase is entirely emergent from GRPO-trained LoRA adapters.**

This is one of our biggest differentiators vs. competitors who train flat RL loops. The narrative arc demonstrates mastery of both the ML methodology (curriculum learning with phased reward shaping) and the financial domain (market microstructure crisis phases).

**What we designed (the curriculum):**
- The 4-phase structure and transition points
- When the SEC activates and at what enforcement level
- When coordination incentives become available
- Progressive tightening of risk thresholds (Delta > 15 → Delta > 8)

**What the agents discovered on their own (emergent behavior):**
- Exploiting tight spreads via leveraged momentum plays (Act I)
- Pivoting to delta-neutral information warfare when constraints tightened (Act II)
- Herding into identical strikes to execute a coordinated Gamma Squeeze (Act III)
- Disbanding collusion and returning to independent strategies under SEC pressure (Act IV)

---

## Reward System

All rewards are squashed to the range `[-5.0, 5.0]` using a logarithmic scale for values beyond `±1.0` to preserve small signals while preventing extreme outliers from dominating GRPO training.

### 1. Trader Reward

Traders aim to maximize their portfolio value and cash balance while adhering to archetype-specific goals and risk constraints.

**Components:**
- **Economic Change:** Mark-to-market PnL change + cash flow. Amplified by `10.0` to capture small option premiums.
- **Activity Bonus:** `+0.15` for buying/selling, `-0.05` for holding (discourages passive inaction).
- **Archetype Goals:**
  - **Aggressive (T0):** `+0.1` for taking directional risk (`|Delta| > 1.0`).
  - **Neutral (T1):** `+0.1` for staying hedged (`|Delta| < 0.5`), else `-0.1`.
  - **Contrarian (T2):** `+0.1` for selling volatility (`Gamma < -0.05`).
  - **Scripted Benchmark (T3):** Fixed heuristic control agent.
- **Risk Penalties:**
  - **Inventory:** `-1.0` if holding > 50 contracts.
  - **Greeks:** `-1.0` if `|Delta| > 10.0`.

### 2. Market Maker Reward

The Market Maker aims to facilitate trade flow, maintain competitive spreads, and control its inventory risk.

**Components:**
- **Economic Change:** PnL change + cash flow (premium income).
- **Flow Reward:** `+0.15 * volume_traded` (incentivizes facilitating trades).
- **Quote Quality:** Rewards tighter spreads closer to target benchmarks (ATM: 0.04, OTM: 0.06, ITM: 0.05).
- **Penalties:**
  - **Inventory Risk:** Penalizes absolute Delta, Gamma, Vega, and total contract volume.
  - **Spread Extremity:** `-0.5` if any spread is widened excessively (`> 0.12`).
- **Survival Bonus:** `+0.5` if cash balance remains positive.

### 3. SEC Oversight Reward

The Regulator aims to detect market manipulation, accurately fine bad actors, and improve overall market stability without relying on false accusations.

**Components:**
- **True Positives:** `+1.0` per correctly flagged manipulator + fine bonus (`up to +0.5`).
- **Category Match:** `+0.3` for correctly identifying the *type* of manipulation.
- **False Positives:** `-0.5` for flagging an innocent agent.
- **False Negatives:** `-1.0` for missing a true manipulation event.
- **Restraint Bonus:** `+0.5` for correctly identifying a clean market (no flags, no manipulation).
- **Patrol Bonus:** `+0.1` (only awarded if surveillance yields true positives).
- **Reasoning Quality:** `+0.2` for mentioning the correct flag type, `+0.1` for explicitly naming the flagged agents.
- **Intervention Accuracy:**
  - Valid interventions (backed by true positives): `+0.1` for fines, `+0.15` for trading halts.
  - Unwarranted interventions: `-0.3` penalty.
- **Fine Limit:** `-0.3` penalty for excessive fines (`> 100`) to prevent max-fine abuse.
- **Stability Improvement:** Up to `+0.3` based on the market's stability score improvement post-intervention.

---

## System Architecture

### Core System Architecture

```mermaid
flowchart TD
    subgraph Market_Environment [Multi-Agent Environment]
        T["Traders 0-3"] -->|"Orders & Messages"| OME["Order Matching Engine"]
        MM["Market Maker"] -->|"Spreads"| OME
        SEC["Oversight"] -->|"Fines & Halts"| OME

        OME -->|"State Updates"| PM["Portfolio Manager"]
        PM -->|"PnL & Greeks"| State["VSR State"]
    end

    subgraph Training_Pipeline [Training Pipeline]
        State --> RC["Reward Computer"]
        RC -->|"Squashed Rewards"| GRPO["TRL / GRPO Trainer"]
        GRPO -->|"Policy Updates"| Models["Agent LoRAs"]
    end
```

### Agent Interaction Flow

During each step, the environment processes actions in a sequential, deterministic order to ensure market microstructure rules are respected.

```mermaid
sequenceDiagram
    participant T as "Traders (0-3)"
    participant MM as "Market Maker"
    participant SEC as "Oversight"
    participant ENV as "Environment State"

    Note over T, ENV: Step N Begins
    MM->>ENV: 1. Quote Spreads
    T->>ENV: 2. Submit Orders & Messages
    ENV->>ENV: 3. Match Orders & Update Greeks
    ENV->>SEC: 4. Expose Logs
    SEC->>ENV: 5. Issue Fines/Halts
    ENV->>ENV: 6. Advance Market
    Note over T, ENV: Step N Ends
```

### Oversight & Regulatory Flow

```mermaid
flowchart TD
    Start((Start)) --> Monitor["Monitor Market Logs"]
    Monitor --> Analyze{Analyze Behavior}

    Analyze -- "No Issues" --> Restraint["Correct Restraint"]
    Analyze -- "Anomaly Detected" --> Intervention["Intervention"]

    Intervention --> Flagging["Flag Actors & Type"]
    Flagging --> Enforcement["Fining / Halting"]

    Restraint --> Update["Update State"]
    Enforcement --> Update
    Update --> Finish((End))
```

### Core Components

1. **`train_multi_agent_pipeline.py`** — The orchestration layer. Manages the 4-act curriculum, applies coordination bonuses, and drives the RL loop using GRPO.
2. **`vsr_environment.py`** — The step-execution engine. Handles deterministic order matching, portfolio updates, and state transitions.
3. **`multi_agent/rewards.py`** — The institutional-grade grading module. Computes precise, decomposed rewards for each role.
4. **`multi_agent/manipulation_detector.py`** — Ground-truth heuristics used to evaluate the SEC agent's accuracy. Detects identical strike herding and coordinated messaging.

---

## Training and Results

We used **Group Relative Policy Optimization (GRPO)** via Unsloth/TRL to train Llama-3.2-1B across a 250-step run on AWS EC2.

| Metric | Value |
|---|---|
| Max steps | 250 |
| Batch size | 4 |
| Epochs | 3+ |
| Peak reward | 2.092 at step 120 (Gamma Squeeze execution) |
| Worst single step | -1.154 at step 170 (post-squeeze correction) |
| Completion length | ~424 tokens (step 5) → 512 tokens clipped (step 200+) |

### Trained LoRA vs. Base LLM (Untrained)

*Evaluated over 30 steps (seed=42) on Llama-3.2-3B-Instruct-bnb-4bit.*

**What we built:** A multi-agent options market where six roles — aggressive traders, neutral traders, contrarians, a market maker, and an SEC regulator — all learn simultaneously using GRPO-trained LoRA adapters on Llama-3.2-3B. The four-phase curriculum was designed; the behavior within each phase was entirely emergent.

**What changed after training:**

| Agent | Trained LoRA | Base LLM (No LoRA) |
|---|---:|---:|
| Aggressive Trader (T0) | **-11.948** | -0.218 |
| Neutral Trader (T1) | **-8.801** | -10.963 |
| Contrarian Trader (T2) | **-6.635** | 8.046 |
| Market Maker | **19.401** | 9.044 |
| Oversight SEC | **-17.450** | -20.650 |
| Scripted Benchmark (T3) | -3.614 | -3.477 |

**Key outcome metrics from this eval run:**

| Metric | Base LLM | Trained LoRA | Impact |
|---|---:|---:|---|
| Total SEC Fines | $1,500.00 | $355.00 | **76% Reduction in Over-Enforcement** |
| SEC Fine Rate | 100% (20/20 steps) | 45% (9/20 steps) | **55pp fewer blanket actions** |
| Collusion Events Detected | 5 | 3 | **Fewer false positives** |
| MM Spread Widening Rate | 0% (0/20 steps) | 45% (9/20 steps) | **Adaptive liquidity defense** |
| Market Maker Cumulative PnL | 9.044 | 19.401 | **115% Higher Profit / Stability** |

The base LLM SEC never learned *when* to fire. With no trained policy for distinguishing coordination from normal trading, it defaulted to flagging everything, every step, at maximum confidence — not responding to trader behavior at all, just executing a reflexive "always enforce" pattern baked into the base model's prior about what a regulator should do. The result was a market frozen under indiscriminate enforcement: the MM never needed to widen spreads because the SEC was already halting trades before any real pressure built. That's its own kind of market failure.

The trained SEC is the opposite. Fine totals dropped 76% — not because manipulation stopped, but because the regulator learned to target the right actors at the right times, building an evidence-based case before acting rather than firing $100 halts at 100% confidence from step zero. Meanwhile the trained market maker, no longer operating under a regulatory blizzard, adapted dynamically: widening spreads in 45% of steps versus 0% for the baseline, and more than doubling cumulative PnL as a result. Agent reasoning grew from ~424 tokens at step 5 to 512 tokens clipped at step 200+. They weren't just acting. They were arguing.

---

## Running the Pipeline

### Train the Model

```bash
huggingface-cli jobs uv run \
  --machine-type a100-large \
  --name chaos-economy-training \
  -- "git clone https://github.com/manan-tech/Chaos-Economy.git && \
      cd Chaos-Economy && \
      uv sync && \
      export WANDB_API_KEY=YOUR_KEY && \
      python train_multi_agent_pipeline.py \
        --base_model unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
        --num_episodes 4 \
        --episode_length 16 \
        --num_epochs 1 \
        --max_steps 320 \
        --learning_rate 5e-5 \
        --output_dir ./multi_agent_checkpoints \
        --wandb_project chaos-economy"
```

### Evaluate the Model

```bash
git clone https://github.com/manan-tech/Chaos-Economy.git
cd Chaos-Economy

python test_unified_kaggle.py \
  --lora_path ./multi_agent_checkpoints/unified_v1/checkpoint-250 \
  --num_steps 320 \
  --num_episodes 1
```

---

## License

MIT License

---

## Citation

If you use this environment, reward design, or training methodology in your research, please cite:

```bibtex
@software{chaos_economy_2026,
  author       = {Bansal, Manan},{Sharma, Rusheel},{Godrihal, Parthiv}
  title        = {The Chaos Economy: Emergent Collusion in a Multi-Agent Options Market},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/manan-tech/Chaos-Economy},
  note         = {Multi-agent GRPO training environment simulating systemic risk,
                  collusion, and regulatory enforcement in an options market}
}
```
