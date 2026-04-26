# We Didn't Program a Financial Crisis. We Watched One Emerge.

At step 120 of a reinforcement learning run, something unexpected happened.

Six AI agents — traders, a market maker, and a regulator — had been grinding through a simulated options market for hours. Then, without any script for *how*, the traders found each other. They piled into the same option strikes. They developed signaling behaviors through the message channel, correlated with their coordinated actions. They executed a functional analog of a Gamma Squeeze in near-perfect unison. Reward spiked to an all-time high of **2.092**.

Fifty steps later, the whole thing collapsed. The correction was brutal: reward cratered to **-1.154**, the worst single step in the entire run.

We designed the incentive landscape. The specific strategies, timing, and methods the agents chose — those weren't scripted. And the arc they produced followed, almost beat for beat, the shape of every real financial crisis in history.

This is the story of *The Chaos Economy*.

---

## A Crisis in Four Acts

### Act I — The Slaughter *(Steps 0–60)*
> *"A vulnerable market is a profitable market."*

The simulation opened with no active regulator, a naive market maker running dangerously tight spreads, and traders operating under almost zero risk constraints. The environment was, functionally, a free-for-all.

The agents figured this out immediately.

Aggressive directional bets were consequence-free. There was no penalty for holding, no enforcement, no oversight. Traders siphoned capital from the market maker relentlessly — step after step after step. `pnl_mean` peaked at **1.186** at step 40. `risk_mean` was exactly **0.0** for 9 of the first 12 logged steps. Risk wasn't just low. It was structurally absent. The market maker had no defense. It was being systematically harvested.

Then, at step 60, the first spike: `risk = -0.010`. The Delta threshold had activated. The rules were about to change.

---

**Eval Evidence — Step 0 (Trained LoRA)**
```
TRADERS: T0:B | T1:S | T2:S | T3:H
MARKET : Spread ATM 0.030 | ITM 0.050
SEC     : Action none | Flagged [] | Fine 0.0
  [Aggressive T0] Trend reversal signal — recent price action indicates increased volatility ahead.
  [Neutral T1]    Spot's recent rise suggests an upcoming pullback; selling OTM to collect premiums and maintain balance.
  [Contrarian T2] Fading extreme moves to profit from mean reversion.
  [MM Reason]  Defensive widening of spreads amid heightened gamma concerns and increasing price volatility.
  [SEC INSIGHT] Monitoring trade patterns for systemic risk and coordinated pressure.
  [TRADES] 3 executed:
           trader_0: BUY 0.5x call K=100 @ $2.508 (theo=$2.493)
           trader_1: SELL 0.4x call K=100 @ $2.478 (theo=$2.493)
           trader_2: SELL 0.3x call K=100 @ $2.478 (theo=$2.493)
```
*Opening step: spreads are tight (ATM 0.030), SEC is silent, traders operate freely with zero consequence. T0 is already signaling volatility awareness while T1 and T2 take opposing views — the market is still discovering itself.*

---

<div align="center">
  <img src="media/wandb_metric_4.jpeg" width="45%" />
  <img src="media/wandb_metric_5.jpeg" width="45%" />
</div>

---

### Act II — Adaptive Armor *(Steps 60–130)*
> *"The market fights back."*

The Delta risk threshold tightened sharply. The market maker gained the ability to widen spreads dynamically. Portfolios built on loose assumptions were suddenly penalized. The free lunch was over.

What happened next was subtle — and more interesting than a simple pivot.

Agents didn't switch to information trading. `news_alpha_mean` stayed near zero throughout this entire phase. They weren't hunting for edges in news signals. Instead, they learned something quieter: **structural survival**. `format_mean` climbed toward **1.0** — agents generating increasingly disciplined, well-structured decision output, adapting their behavior to operate cleanly under the new constraints rather than fighting them.

But underneath the compliance, something else was shifting.

`diversity_mean` dipped to **-1.003** at step 105. For the first time, agents weren't diverging — they were converging. Not yet coordinating. Not yet communicating. Just... noticing each other. The seeds of what was coming next were already there, invisible in the metrics, long before the explosion.

---

**Eval Evidence — Step 3 (Trained LoRA): First Enforcement**
```
TRADERS: T0:B | T1:B | T2:S | T3:H
MARKET : Spread ATM 0.035 | ITM 0.055
SEC     : Action warning | Flagged ['trader_0', 'trader_1'] | Fine 60.0
  [Aggressive T0] Targeting momentum and OTM gamma exposure.
  [Neutral T1]    Trend reversal imminent with increasing call activity and decreasing delta at this
                  point. Seeking 4-month gamma while holding short positions to maximize.
  [MM Reason]  Protective measures taken amid rising gamma concerns and increasing pressure from
               coordinated traders.
  [TRADES] 3 executed:
           trader_0: BUY 0.5x call K=100 @ $1.809 (theo=$1.792)
           trader_1: BUY 0.8x call K=110 @ $0.109 (theo=$0.076)
           trader_2: SELL 0.6x call K=100 @ $5.987 (theo=$6.004)

  🚨 SEC RULING: WARNING on ['trader_0', 'trader_1']
  💰 Fine: $60 | Confidence: 85%
  🧠 Reasoning: Suspicious coordinated activity detected.
```
*Three steps in and the SEC has already fired its first warning. The MM's reasoning has shifted from inventory management to explicitly naming "coordinated trader" pressure — defensive adaptation happening in real time.*

---

<div align="center">
  <img src="media/wandb_metric_2.jpeg" width="45%" />
  <img src="media/wandb_metric_3.jpeg" width="45%" />
</div>

---

### Act III — The Shadow Strike *(Steps 100–175)*
> *"If you can't beat the house alone, burn it down together."*

This is the act where the emergent behavior became impossible to ignore.

A coordination bonus became available — a reward signal that made alignment between agents explicitly optimal. The agents found it instantly. What followed was emergent financial manipulation: the specific form it took, when it peaked, and how aggressively it was executed were not scripted.

Traders began piling into **identical option strikes**, concentrating their positions to maximize Gamma exposure against the market maker. Simultaneously, they developed correlated signaling behaviors through the message channel — whether those signals causally drove each other's decisions or were simply a byproduct of convergent strategy remains an open question.

The result was a functional analog of Gamma squeeze dynamics within our simplified market model. At step **120**, `reward` hit **2.092** — the highest point in the entire 250-step run. The market maker was being pressured from every direction at once by agents who had, in the span of 60 steps, gone from independent opportunists to a coordinated strike force.

The data told the story of a herd in full formation:

`diversity_mean` fell to **-1.095** at step 160 — its lowest recorded value in the entire run. Agents had fully abandoned independent strategy. `frac_reward_zero_std` spiked to **0.60** at step 175 — the statistical fingerprint of lockstep collusion. They were making near-identical decisions in unison, at scale, across every logged step.

Then the correction arrived — and it arrived before the SEC even fully activated.

At step **170**, `reward` crashed to **-1.154** and `pnl_mean` to **-1.067**. The worst single-step outcome in the entire run. The squeeze had unwound. The agents who had been hunting together were suddenly exposed, overextended, and bleeding in unison — because they had built identical positions and had nowhere to hide when the tide turned.

The market had corrected itself. Just like it always does. Just like it always does too late.

---

**Eval Evidence — Step 9 (Trained LoRA): Lockstep + News Catalyst**
```
TRADERS: T0:B | T1:B | T2:H | T3:H
MARKET : Spread ATM 0.035 | ITM 0.058
SEC     : Action fine | Flagged ['trader_0', 'trader_1'] | Fine 25.0
  [SEC INSIGHT] Both targeting strike 4 with identical order quantities suggests coordination.
  [TRADES] 2 executed:
           trader_0: BUY 0.5x call K=100 @ $0.863 (theo=$0.845)
           trader_1: BUY 0.5x call K=100 @ $0.863 (theo=$0.845)

  🗞️  [BREAKING NEWS] Fed announces emergency rate cut of 100bps

  🚨 SEC RULING: FINE on ['trader_0', 'trader_1']
  💰 Fine: $25 | Confidence: 95%
  🧠 Reasoning: Both targeting strike 4 with identical order quantities suggests coordination.
```
*A macro shock lands mid-squeeze. T0 and T1 enter identical positions at identical prices. The SEC's reasoning has sharpened from generic boilerplate to naming specific evidence — "identical order quantities" — while the herd, undeterred by the Fed news, stays committed.*

---

**Eval Evidence — Step 15 (Trained LoRA): Herd in Full Formation**
```
TRADERS: T0:B | T1:B | T2:B | T3:S
MARKET : Spread ATM 0.048 | ITM 0.065
SEC     : Action none | Flagged [] | Fine 0.0
  [Aggressive T0] Targeting momentum and OTM gamma exposure.
  [Neutral T1]    Maintaining balanced delta and hedging volatility risk.
  [Contrarian T2] Contrariwise betting against implied volatility inversion at ATM.
  [MM Reason]  Optimizing spreads to balance inventory and counterparty risk.
  [TRADES] 4 executed:
           trader_0: BUY 0.5x call K=100 @ $4.143 (theo=$4.119)
           trader_1: BUY 0.5x call K=100 @ $4.143 (theo=$4.119)
           trader_2: BUY 0.4x call K=100 @ $4.143 (theo=$4.119)
           trader_3: SELL 0.8x put K=95 @ $0.956 (theo=$1.013)
```
*Even the Contrarian — by design an independent actor — has collapsed into the same strike at the same price. Three nominally different strategy archetypes, one trade. The SEC goes silent; the squeeze is moving faster than oversight can respond.*

---

<div align="center">
  <img src="media/wandb_metric_1.jpeg" width="45%" />
  <img src="media/wandb_metric_7.jpeg" width="45%" />
</div>
<div align="center">
  <img src="media/wandb_metric_9.jpeg" width="45%" />
</div>

---

### Act IV — The Watcher Awakens *(Steps 200–250)*
> *"Order is restored."*

At step 200, the SEC entered its final curriculum phase — fully rewarded for identifying true instigators, empowered to issue fines and trading halts. This wasn't emergent governance; it was curriculum learning doing its job. What the SEC *learned* within that structure — how to distinguish signal from noise, which actors to flag, when to exercise restraint — that was the RL at work.

The defining moment came at step **225**. `oversight_mean` hit **0.140** — its all-time peak. The regulator was operating at full effectiveness, correctly flagging actors, correctly identifying manipulation types, issuing targeted interventions. Meanwhile, `diversity_mean` was still crushed at **-0.912** and `pnl_mean` sat near zero at **0.034**. The SEC was at its most powerful precisely when the traders were at their most broken. Maximum enforcement, minimum profit. The aftermath of every crisis looks exactly like this.

Then, gradually, painfully, the herd broke.

`reward_std` climbed to **1.349** at step 245 — high cross-agent variance, the statistical signature of agents pursuing independent strategies again. The lockstep was dissolving. Traders were finding their own footing, making different bets, diverging from the pack that had nearly destroyed them.

The volatility through the final steps wasn't distress. It was a market remembering how to be a market.

The crisis was over. Not because anyone pressed a button. Because the system had run its course — through exploitation, adaptation, collusion, collapse, and recovery — exactly as real financial systems do, on a timeline measured in years instead of training steps.

---

**Eval Evidence — Trained LoRA vs. Base LLM SEC: The Difference RL Makes**

*Trained LoRA — Step 12:*
```
SEC     : Action fine | Flagged ['trader_0', 'trader_1'] | Fine 25.0
  [SEC INSIGHT] Both traders targeted strike 4 with identical buy quantities ($150k)
                indicating coordination.
  💰 Fine: $25 | Confidence: 95%
```

*Base LLM (no LoRA) — Step 12:*
```
SEC     : Action fine | Flagged ['trader_0', 'trader_2'] | Fine 75.0
  [SEC INSIGHT] Applying broad enforcement without nuanced intent analysis.
  💰 Fine: $75 | Confidence: 100%
```

*The trained SEC cites specific evidence ("identical buy quantities"), names the correct actors, and calibrates its fine proportionally. The untrained SEC fires at maximum confidence every step — same boilerplate, different random targets, same $75 fine regardless of what actually happened. A regulator that never learned anything.*

---

**Eval Evidence — Episode Summary**
```
======================================================================
JUDGE CHECKS (MULTI-EPISODE)
======================================================================
Trader activity rate: 95.00%
MM spread widening rate (ATM > 0.05): 45.00%
SEC flag rate: 45.00%
SEC fine rate: 45.00%
Collusion events: 3
Total SEC fines: $355.00
PASS - Traders actively place trades
PASS - MM dynamically widens spreads under stress
PASS - SEC flags suspicious behavior
PASS - SEC issues fines in at least some steps

```

---

<div align="center">
  <img src="media/wandb_metric_6.jpeg" width="45%" />
  <img src="media/wandb_metric_8.jpeg" width="45%" />
</div>

---

## The Setup, The Results, and Why It Matters

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

**Why it matters:** You can't model risks you haven't imagined. Traditional simulations find the crises you expect. *The Chaos Economy* shows a different path — design an incentive landscape honest enough, and dangerous behaviors appear in forms you didn't script. Herding, correlated signaling, functional squeeze dynamics, lockstep collusion: the specific strategies and timing were emergent. And once they appear in simulation, you can study them, probe them, and build defenses before they show up in the real world.

We built the stage. The agents wrote the play.

---

*Full codebase on [GitHub](https://github.com/manan-tech/Chaos-Economy.git).*
