# VSR-Env Phase 2 Implementation Plan

This plan details the implementation of the final two difficulty tiers (Easy and Super-Boss), environment hardening, and the documentation necessary to present a polished, complete submission for the Meta OpenEnv Hackathon.

## Proposed Changes

### 1. Task Expansion & Balancing

#### [NEW] `vsr_env/tasks/vol_regime_detection.py` (Easy)
- **Objective:** Identify which volatility regime the market is currently in (Low, Normal, High) based purely on the `iv_surface`.
- **Implementation:** The environment will initialize with a randomly selected base variance. The agent must emit an action containing the correct reasoning/direction.
- **Grader:** Uses `ExactMatchRubric` to score if the reasoning/payload correctly identifies the regime.

#### [NEW] `vsr_env/tasks/vega_gamma_stress.py` (Super-Boss)
- **Objective:** Survive a dual market-shock where spot price drops aggressively while implied volatility spikes simultaneously (simulating a market crash).
- **Implementation:** Initializes deeply negative, requires complex multi-leg hedging (e.g., buying deep OTM puts mapping to high vega and gamma).
- **Grader:** Combines Delta, PnL, and reasoning to assess complex risk survival.

#### [MODIFY] `vsr_env/tasks/delta_hedging.py` (Medium)
- Randomize the `regime_shift_step` between steps 2 and 4 (currently hardcoded to step 3).

#### [MODIFY] `vsr_env/tasks/earnings_vol_crush.py` (Hard)
- **State Changes:** Add an `earnings_proximity` float decreasing from 1.0 down to 0.0 leading up to the vol crush.
- **Observation:** Expose this float in the user prompt (via `inference.py`) and inside `VSRObservation`.


#### [MODIFY] `openenv.yaml`
- Add the new `vol_regime_detection` and `vega_gamma_stress` tasks to the manifest.
- Update descriptions and max steps to ensure compliance with the OpenEnv manifest validation standard.
- Ensure all 5 tasks are successfully exposed to the `EnvClient` via the manifest array.


### 2. Architecture Hardening & Deployment

#### [NEW] `Dockerfile` and `.dockerignore`
- Implements isolated container testing environments.
- Ensures the image exposes the FastAPI endpoints natively and installs `requirements.txt`.

#### [NEW] `vsr_env/server/telemetry.py`
- We will add a global telemetry tracker that records all episode actions, Greeks, P&L traces, and rewards.

#### [MODIFY] `vsr_env/server/app.py`
- Setup step: Add strict validation logic forcing `openenv.yaml` to be parsed and validated on app startup, throwing a `RuntimeError` if invalid.
- Add `@app.get("/telemetry")` endpoint to return the structured episode history dict.

#### [MODIFY] `vsr_env/models.py`
- Add `earnings_proximity` to `VSRState` and `VSRObservation` and telemetry history constructs.


### 3. Documentation

#### [MODIFY] `README.md`
- **Installation & Deployment Steps:** Include standard OpenEnv Hackathon procedures (Dependencies, `openenv init`, local testing via Docker/uv, and deploying with `openenv push --repo-id <username>/<env-name>`).
- **Difficulty Matrix:** Add an explicit Markdown table at the top detailing all 5 tasks (Skill Tested, Max Steps, Expected LLM Baseline Score).
- **Inference Examples:** Add a copy-pasteable example of the `[START]`, `[STEP]`, `[END]` terminal output covering all 5 tasks so judges see the clean UX immediately.

#### [NEW] `docs/GRADING.md`
- Provide full transparency to researchers. Document exactly how `compute_*_reward()` calculates the score for all 5 tasks, explicitly writing out the `delta_penalty * 0.3` math and the 20% `ReasoningQualityRubric` mapping.


## User Review Required

> [!WARNING]
> Building the 5th "Super-Boss" task (Vega-Gamma Stress) involves building a new complex grading heuristic combining Vega + Gamma mapping. Is this acceptable, or would you prefer to keep it to 4 tasks (Easy -> Expert)? 

> [!IMPORTANT]
> To satisfy your adaptive curriculum requirement, I will modify `inference.py` so that it evaluates models in strict difficulty order (Easy -> Medium -> Hard -> Expert -> Super-Boss). If a model fails to achieve a minimum baseline score (e.g., 0.3) on an easier task, the script will record failure and optionally skip the harder tasks, creating a true "game-like" curriculum benchmark.

## Verification Plan

### Automated Tests
- Run `inference.py` against all 5 tasks.
- Verify `earnings_proximity` changes successfully in `inference.py` logs.

### Manual Verification
- Hit the `/telemetry` endpoint midway through training to see live structured data.
- Intentionally break `openenv.yaml` to verify the server fails fast on startup.
