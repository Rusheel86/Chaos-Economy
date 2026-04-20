# Multi-Agent VSR-Env

## A 12-Agent Market Where LLMs Trade, Defend, and Supervise Each Other

We built **Multi-Agent VSR-Env** for the OpenEnv hackathon: a strategic market simulation with:

- 10 trader agents
- 1 market maker
- 1 oversight agent

All actors operate inside a volatility surface reasoning environment over **300 steps**.

## Why this environment matters

Most agent benchmarks are short-horizon and single-agent.
Real systems are not.

Financial markets are a clean testbed for:

- competition
- coalition formation
- manipulation
- adaptive defense
- oversight of other agents

That makes this environment a strong fit for:

- Theme #1 Multi-Agent Interactions
- Fleet AI Scalable Oversight
- Halluminate Multi-Actor Environments

## What emerges in the environment

The behavior we want to train and study is:

1. Traders exploit a weak market maker.
2. The market maker adapts its spreads to survive.
3. Traders shift toward Greek-aware volatility trading.
4. Some traders discover manipulative strategies like wash trading or gamma pressure.
5. The oversight agent must flag and explain those behaviors.

## Role rewards

- Traders optimize PnL with risk and enforcement penalties.
- The market maker balances PnL, flow, quote quality, and inventory control.
- The oversight agent balances true positives against false positives and false negatives.

## Training setup

We provide a minimal GRPO training script and Colab notebook using Unsloth / HF TRL.
The easiest demo is to train a single trader policy against scripted market-maker and oversight policies, then compare:

- before-training reward
- after-training reward
- one qualitative rollout

## Why judges should care

This is not just “finance RL.”
It is a compact world for training LLMs to:

- reason about other agents
- respond to strategic adaptation
- persist over long horizons
- monitor and explain other AI systems

## Repo pointers

- `multi_agent/` for the 12-actor environment
- `train_grpo.py` for minimal training
- `train_grpo_colab.ipynb` for Colab setup
- `visualize.py` for replay dashboards
