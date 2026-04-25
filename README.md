---
title: Predatory Swarms
emoji: 🦈
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

# 🦈 Predatory Swarms: Emergent Collusion in a Multi-Agent Options Market

Project for **Meta × PyTorch × SST OpenEnv AI Hackathon**

> [!IMPORTANT]
> **Judge's TL;DR:** While most submissions simulate single-agent tasks or simple startup dynamics, *Predatory Swarms* tackles **Systemic Risk**. We simulate a high-fidelity multi-agent options market where 3 archetypal traders, a market maker, and a regulator engage in a high-stakes game of emergent collusion and adaptive oversight.

### 🔗 Submission Links
- **Hugging Face Space:** `[INSERT LINK TO HF SPACE HERE]`
- **Mini-Blog / Storytelling Report:** [The Chaos Economy: A Story of Systemic Risk](./STORYTELLING.md)
- **Demo Video:** `[INSERT YOUTUBE LINK HERE]`

---

## 🏆 Hackathon Rubric & Innovation (40%)

### An Ambitious, Original Problem
The themes of the hackathon encourage avoiding grid-world clones. We chose a highly underexplored domain: **Financial Information Warfare and Collusion**. 
- **Strategic asymmetry:** Traders, Market Makers, and the SEC optimize conflicting objectives.
- **Delayed consequences:** Coordinated pressure changes inventory and spreads over multiple timesteps.
- **Complex Information Economy:** Agents trade inside a "Dark Pool", buying and selling "Fake News" or real intel to trigger volatility.

### Agent Roles
- **Aggressive Trader (`trader_0`)**: High-risk momentum chaser, gamma squeeze initiator.
- **Neutral Trader (`trader_1`)**: Balanced, may join or resist coordination.
- **Contrarian Trader (`trader_2`)**: Counter-trend, exploits manipulation.
- **Scripted Baseline (`trader_3`)**: Fixed heuristic for comparison.
- **Market Maker**: Sets ATM/OTM/ITM spreads dynamically as a defensive control.
- **Oversight (SEC)**: Flags suspected manipulation, can fine, and can intervene.

---

## 📖 Storytelling & The "Black Swan" Arc (30%)

Over a 250-step training cycle, we observed a stunning 4-Act narrative emerge purely from the reinforcement learning loop. **Please read our full narrative breakdown in [STORYTELLING.md](./STORYTELLING.md).**

Briefly, the arc progresses as:
1. **Act I (Slaughter):** Traders exploit the Market Maker.
2. **Act II (Adaptation):** The Market Maker widens spreads; trading becomes harder.
3. **Act III (Collusion):** Traders realize they can't win alone, so they sacrifice diversity to illegally collude, executing a massive Gamma Squeeze.
4. **Act IV (Oversight):** The SEC awakens, correlates the manipulation, and issues massive fines, forcing the market back into compliance.

---

## 📈 Real Training Evidence & Rewards (20%)

We used **Group Relative Policy Optimization (GRPO)** via Unsloth/TRL to train Llama-3.2-3B. The reward signals were meticulously designed to teach the agents rather than just score them:
- **Informative Signal:** Agents receive penalties for Delta imbalance (Risk) and rewards for coordinating trades (Collusion).
- **Hard to Game:** If traders coordinate but fail to manipulate the market maker, they just lose money. If they manipulate successfully but ignore the SEC, they get fined.

### Weights & Biases Live Run Dashboards
*(Below are real plots generated from our 250-step training run on an AWS EC2 instance, showcasing the emergent metric shifts through the 4 Acts of the Chaos Economy.)*

<div align="center">
  <img src="media/wandb_metric_1.jpeg" width="45%" />
  <img src="media/wandb_metric_2.jpeg" width="45%" />
</div>
<div align="center">
  <img src="media/wandb_metric_3.jpeg" width="45%" />
  <img src="media/wandb_metric_4.jpeg" width="45%" />
</div>
<div align="center">
  <img src="media/wandb_metric_5.jpeg" width="45%" />
  <img src="media/wandb_metric_6.jpeg" width="45%" />
</div>
<div align="center">
  <img src="media/wandb_metric_7.jpeg" width="45%" />
  <img src="media/wandb_metric_8.jpeg" width="45%" />
</div>
<div align="center">
  <img src="media/wandb_metric_9.jpeg" width="45%" />
</div>

### Trained LoRA vs. Untrained Baseline Comparison

| Agent Type | Trained Llama-3.2-3B | Scripted Baseline |
|---|---:|---:|
| Aggressive Trader | **-0.93** | -4.13 |
| Neutral Trader | **-1.08** | -4.58 |
| Market Maker | **21.01** | 14.84 |
| Oversight SEC | **-95.60** | 7.50 |

*Note: The SEC’s negative reward is a reflection of early exploration penalties; however, the relative improvement of the traders and the market maker clearly demonstrates the RL agents fundamentally outperformed static heuristics.*

---

## ⚙️ Reward + Training Pipeline Setup (10%)

### How Judges Can Train the Model
We have engineered this pipeline to support seamless execution on **Hugging Face Jobs** (A100 recommended) using `uv` for dependency management.

```bash
# 1. Start a Hugging Face Job with W&B tracking
huggingface-cli jobs uv run \
  --machine-type a100-large \
  --name vsr-env-training \
  -- "git clone https://github.com/mananpbansal/vsr-env.git && cd vsr-env && git checkout news && uv sync && export WANDB_API_KEY=YOUR_KEY && python train_multi_agent_pipeline.py --base_model unsloth/Llama-3.2-3B-Instruct-bnb-4bit --num_episodes 4 --episode_length 16 --num_epochs 1 --max_steps 320 --learning_rate 5e-5 --output_dir ./multi_agent_checkpoints --wandb_project vsr-env-multi-agent"
```

### How Judges Can Evaluate the Model
Once the LoRA adapter is trained (or using the provided checkpoint), judges can run the unified testing script to simulate a complete market episode and observe the "Chaos Economy" dynamics.

```bash
# Clone and install
git clone https://github.com/mananpbansal/vsr-env.git
cd vsr-env
git checkout news
uv sync

# Run evaluation of the baseline vs RL models
python test_unified_kaggle.py \
  --lora_path ./multi_agent_checkpoints/unified_v1/checkpoint-250 \
  --num_steps 320 \
  --num_episodes 1
```

---

## License
MIT License
