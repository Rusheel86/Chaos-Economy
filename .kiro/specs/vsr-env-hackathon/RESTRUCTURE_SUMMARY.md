# VSR-Env Task Restructure Summary

## Overview

The VSR-Env hackathon spec has been updated from a 3-task structure to a 5-task structure based on the analysis in `final_task_restructure.md`. This provides better difficulty progression and more compelling narratives for the hackathon submission.

## Task Changes

### Removed Tasks
1. **IV Reading (Easy)** - Replaced by Vol Regime Detection
   - Reason: Too easy to game with simple pattern matching
   
2. **Arb Capture (Hard)** - Replaced by Earnings Vol Crush
   - Reason: Mechanically just "IV Reading + Delta Hedging" chained; earnings story is 10× more compelling

### New Tasks

#### 1. Vol Regime Detection (Easy, 3 steps)
- **What**: Agent observes IV surface changes over 3 steps, classifies regime (low-vol-expanding, high-vol-compressing, stable), trades accordingly
- **Why Better**: Tests interpretation of surface shape, not just pattern matching
- **Grading**: regime_correct × 0.6 + trade_consistency × 0.25 + reasoning × 0.15
- **Implementation**: New file `vsr_env/tasks/vol_regime_detection.py` + grader

#### 2. Delta Hedging (Medium, 5 steps) - UPGRADED
- **What**: Existing task + random market shock at step 2-3
- **Changes**: Agent must maintain neutrality through disruption
- **Grading**: pre_shock_neutrality × 0.30 + post_shock_neutrality × 0.40 + cost_efficiency × 0.30
- **Implementation**: Modify existing `vsr_env/tasks/delta_hedging.py`

#### 3. Earnings Vol Crush (Hard, 8 steps)
- **What**: Surface starts elevated, vol crashes 30-50% at random step 3-6, agent must position before and re-hedge after
- **Why Better**: Named, universally recognized market event; same regime shift mechanic as old Arb Capture
- **Grading**: pre_crush_positioning × 0.40 + post_crush_rehedge × 0.35 + pnl_outcome × 0.25
- **Implementation**: New file `vsr_env/tasks/earnings_vol_crush.py` (reuses 90% of arb_capture.py logic)

#### 4. Gamma Scalping (Expert, 10 steps)
- **What**: Start with ATM straddle (high gamma), spot oscillates ±2-3%, agent scalps gamma by re-hedging at right times
- **Why**: Tests convexity exploitation, high-frequency decision-making
- **Grading**: rehedge_quality × 0.40 + pnl_above_theta × 0.35 + timing_score × 0.25
- **Implementation**: New file `vsr_env/tasks/gamma_scalping.py` + grader

#### 5. Vega-Gamma Stress (Super-Boss, 12 steps)
- **What**: Multi-leg portfolio (long 90-day + short 30-day straddle), massive stress (40-60% vol spike + 5-8% spot crash), manage both Greeks
- **Why**: Combines everything, genuinely hard, tests crisis management
- **Grading**: survival × 0.35 + greek_management × 0.35 + reasoning × 0.30
- **Implementation**: New file `vsr_env/tasks/vega_gamma_stress.py` + grader

## Difficulty Progression

| Level | Task | Steps | Core Skill | Greek Focus |
|:------|:-----|:------|:-----------|:------------|
| Easy | Vol Regime Detection | 3 | Pattern recognition | None (observation) |
| Medium | Delta Hedging | 5 | First-order risk mgmt | Delta (Δ) |
| Hard | Earnings Vol Crush | 8 | Event-driven positioning | Vega (ν) + Delta |
| Expert | Gamma Scalping | 10 | Convexity exploitation | Gamma (Γ) + Theta (Θ) |
| Super-Boss | Vega-Gamma Stress | 12 | Multi-Greek crisis | All Greeks |

## Implementation Effort

| Component | Work Required | Estimated Time |
|:----------|:--------------|:---------------|
| Vol Regime Detection (task + grader) | New file ~150 lines | 2-3 hours |
| Delta Hedging upgrade | Modify existing ~50 lines | 1-2 hours |
| Earnings Vol Crush (task + grader) | New file ~180 lines (reuse arb_capture) | 2-3 hours |
| Gamma Scalping (task + grader) | New file ~200 lines | 3-4 hours |
| Vega-Gamma Stress (task + grader) | New file ~250 lines | 3-4 hours |
| Market simulator enhancements | 4 new functions ~100 lines | 1-2 hours |
| Infrastructure updates | openenv.yaml, inference.py, README | 1-2 hours |
| **Total** | **~930 new lines** | **14-18 hours** |

## Files Modified

### Requirements Document
- `.kiro/specs/vsr-env-hackathon/requirements.md`
  - Updated Requirement 2: Five-Task Progression System
  - Replaced Requirements 3-5 with new task requirements
  - Added Requirements 6-7 for new expert tasks
  - Renumbered subsequent requirements

### Design Document
- `.kiro/specs/vsr-env-hackathon/design.md`
  - Updated overview section to reflect 5 tasks
  - Task implementation sections will need updates (not yet done)

### Tasks Document
- `.kiro/specs/vsr-env-hackathon/tasks.md`
  - Updated overview to reflect 5 tasks
  - Task 6: Added 4 new market simulator functions
  - Task 10: Replaced old task implementations with new 5-task structure
  - Task 11: Replaced old graders with new 5-grader structure
  - Task 15: Updated inference script task list and seeds
  - Task 20: Updated openenv.yaml task list
  - Task 21: Marked task documentation for update
  - Task 23: Updated performance benchmarks for 5 tasks
  - Added restructure summary in Notes section

## Code Files to Create

1. `vsr_env/tasks/vol_regime_detection.py` - Task + Grader
2. `vsr_env/tasks/earnings_vol_crush.py` - Task + Grader (reuse arb_capture.py)
3. `vsr_env/tasks/gamma_scalping.py` - Task + Grader
4. `vsr_env/tasks/vega_gamma_stress.py` - Task + Grader

## Code Files to Modify

1. `vsr_env/tasks/delta_hedging.py` - Add market shock logic
2. `vsr_env/engine/market_sim.py` - Add 4 new functions:
   - `inject_regime()` - Force specific regime at init
   - `trigger_vol_crush()` - Earnings vol crush event
   - `inject_oscillation()` - Force spot oscillations
   - `trigger_stress_event()` - Combined vol spike + spot crash
3. `inference.py` - Update task list and seeds
4. `openenv.yaml` - Update task list
5. `README.md` - Update task descriptions

## Expected Baseline Scores

| Task | Baseline (Random) | Frontier (GPT-4) |
|:-----|:------------------|:-----------------|
| Vol Regime Detection | ~0.20 | ~0.95 |
| Delta Hedging | ~0.25 | ~0.80 |
| Earnings Vol Crush | ~0.30 | ~0.75 |
| Gamma Scalping | ~0.15 | ~0.65 |
| Vega-Gamma Stress | ~0.10 | ~0.55 |

## Narrative Arc

> Each task adds a new dimension of complexity:
> - **Easy**: Can the agent *read* the market?
> - **Medium**: Can it *react* to the market?
> - **Hard**: Can it *anticipate* market events?
> - **Expert**: Can it *exploit* market dynamics for profit?
> - **Super-Boss**: Can it *survive* when everything goes wrong at once?

## Next Steps

1. Implement the 4 new task files with graders
2. Upgrade delta_hedging.py with market shock
3. Add 4 new market simulator functions
4. Update inference.py, openenv.yaml, README.md
5. Run validation and testing
6. Update documentation with new task descriptions

## References

- Full analysis: `final_task_restructure.md`
- Requirements: `.kiro/specs/vsr-env-hackathon/requirements.md`
- Design: `.kiro/specs/vsr-env-hackathon/design.md`
- Tasks: `.kiro/specs/vsr-env-hackathon/tasks.md`
