# Implementation Plan: VSR-Env (Volatility Surface Reasoning Environment)

## Overview

This implementation plan breaks down the VSR-Env hackathon project into actionable tasks organized by the 3-day sprint timeline. The environment simulates options portfolio management with five graded tasks (Vol Regime Detection, Delta Hedging, Earnings Vol Crush, Gamma Scalping, Vega-Gamma Stress) and must pass `openenv validate`, deploy to HuggingFace Spaces, and complete in <20 min runtime.

**Critical Path:**
- Day 1 (April 5): Skeleton that passes validation
- Day 2 (April 6): Tasks + Graders + inference.py
- Day 3 (April 7-8): Deploy + Validate + Submit

## Tasks

- [x] 1. Project Setup and Core Infrastructure
  - Create project directory structure
  - Set up Python package with setup.py or pyproject.toml
  - Create requirements.txt with dependencies: openenv-core, fastapi, uvicorn, pydantic, numpy, scipy
  - Initialize git repository and .gitignore
  - _Requirements: 1.7, 14.2_

- [x] 2. Pydantic Models (vsr_env/models.py)
  - [x] 2.1 Create TradeDirection enum
    - Define BUY, SELL, HOLD as string enum values
    - _Requirements: 16.3_
  
  - [x] 2.2 Create VSRAction model
    - Fields: selected_strike (int 0-7), selected_maturity (int 0-2), direction (TradeDirection), quantity (float 0-10), reasoning (str)
    - Add Pydantic Field descriptions for each field
    - _Requirements: 16.1, 16.2, 16.4, 16.5, 16.6, 16.7_
  
  - [x] 2.3 Create VSRObservation model
    - Fields: iv_surface (List[List[float]]), spot_price, portfolio_greeks (Dict), portfolio_pnl, portfolio_positions (List[Dict])
    - Fields: market_sentiment, step_number, steps_remaining, task_name, task_description, last_action_error (Optional[str])
    - _Requirements: 17.1-17.11_
  
  - [x] 2.4 Create VSRState model
    - Fields: episode_id, step_count, task_name, true_mispriced_strikes, true_mispriced_directions
    - Fields: regime, spot_price, variance, portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_pnl, positions
    - Use Field with default_factory for lists and dicts
    - _Requirements: 18.1-18.8_
  
  - [x] 2.5 Create VSRReward model
    - Fields: total, pnl_component, greek_component, identification_component, reasoning_component
    - _Requirements: 19.1-19.7_


- [x] 3. Black-Scholes Pricing Engine (vsr_env/engine/option_chain.py)
  - [x] 3.1 Create OptionChainEngine class with initialization
    - Initialize with STRIKES = [85, 90, 95, 97.5, 100, 102.5, 105, 110] and MATURITIES = [30/365, 90/365, 180/365]
    - Set risk-free rate r = 0.05
    - _Requirements: 7.1_
  
  - [x] 3.2 Implement vectorized bs_price method
    - Compute Black-Scholes prices for calls and puts using NumPy
    - Formula: d1 = (log(S/K) + (r + 0.5*σ²)*T) / (σ*sqrt(T)), d2 = d1 - σ*sqrt(T)
    - Call: S*N(d1) - K*exp(-r*T)*N(d2), Put: K*exp(-r*T)*N(-d2) - S*N(-d1)
    - Use scipy.stats.norm.cdf for cumulative distribution
    - _Requirements: 7.1, 7.6_
  
  - [ ]*3.3 Write unit tests for bs_price
    - Test call and put pricing against scipy reference implementation
    - Verify accuracy within 1e-6 tolerance
    - _Requirements: 24.1_
  
  - [x] 3.4 Implement delta method
    - Compute delta = N(d1) for calls, N(d1) - 1 for puts
    - _Requirements: 7.2_
  
  - [ ]*3.5 Write unit tests for delta
    - Test delta against finite difference approximation
    - Verify accuracy within 1e-4 tolerance
    - _Requirements: 24.1_
  
  - [x] 3.6 Implement gamma method
    - Compute gamma = N'(d1) / (S * σ * sqrt(T))
    - Use scipy.stats.norm.pdf for probability density
    - _Requirements: 7.3_
  
  - [x] 3.7 Implement vega method
    - Compute vega = S * N'(d1) * sqrt(T) / 100
    - _Requirements: 7.4_
  
  - [x] 3.8 Implement theta method
    - Compute theta for time decay
    - _Requirements: 7.4_

- [x] 4. Implied Volatility Solver (vsr_env/engine/option_chain.py)
  - [x] 4.1 Implement Newton-Raphson IV solver
    - Initial guess sigma = 0.2
    - Iteration: sigma_new = sigma_old - (price - market_price) / vega
    - Convergence tolerance 1e-6, max 100 iterations
    - Clamp sigma to [0.01, 5.0] range
    - _Requirements: 8.1, 8.3, 8.4_
  
  - [x] 4.2 Implement Brent's method fallback
    - Use scipy.optimize.brentq when vega < 1e-8
    - Search range [0.01, 5.0]
    - _Requirements: 8.2_
  
  - [x] 4.3 Implement intrinsic volatility fallback
    - When Brent's fails, return max(0.05, min(abs(log(S/K)) / sqrt(T) * 0.5, 3.0))
    - _Requirements: 8.5_
  
  - [ ]*4.4 Write unit tests for IV solver
    - Test round-trip: price → IV → price recovers original volatility
    - Test convergence for various option parameters
    - _Requirements: 24.1_


- [x] 5. IV Surface Generation (vsr_env/engine/option_chain.py)
  - [x] 5.1 Implement generate_iv_surface method
    - Generate 8×3 IV surface with base_vol, skew, and term_slope parameters
    - Apply skew: base_vol + skew * log_moneyness / sqrt(T)
    - Apply term structure: + term_slope * sqrt(T)
    - Add seeded Gaussian noise with std 0.005
    - Clamp all values to minimum 0.05
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7_
  
  - [x] 5.2 Implement mispricing injection
    - Accept mispriced_cells parameter: List[((strike_idx, maturity_idx), direction, magnitude)]
    - Adjust specified cells by magnitude (add for "over", subtract for "under")
    - _Requirements: 9.6_
  
  - [x] 5.3 Implement inject_mispricings helper function
    - Generate num_mispricings random cells (default 2)
    - Ensure cells are not adjacent (avoid overlapping mispricings)
    - Random direction ("over" or "under") and magnitude (0.03 to 0.08)
    - _Requirements: 3.2_

- [x] 6. Market Simulator (vsr_env/engine/market_sim.py)
  - [x] 6.1 Implement advance_market function
    - Use Geometric Brownian Motion: dS = μ*S*dt + σ*S*dW
    - Risk-neutral drift μ = 0.0, dt = 1/252 (one trading day)
    - Use seeded RNG for dW = rng.normal(0, sqrt(dt))
    - Clamp spot_price to [50.0, 150.0]
    - _Requirements: 23.1, 23.2, 23.6_
  
  - [x] 6.2 Implement mean-reverting variance dynamics
    - Ornstein-Uhlenbeck: dV = θ*(var_mean - variance)*dt + var_vol*dW
    - Parameters: θ = 0.1, var_mean = 0.04, var_vol = 0.01
    - Clamp variance to [0.01, 0.16]
    - _Requirements: 23.5, 23.7_
  
  - [x] 6.3 Implement trigger_regime_shift function
    - Randomly choose "vol_spike" or "vol_crash"
    - vol_spike: multiply variance by 1.2-1.4
    - vol_crash: multiply variance by 0.7-0.8
    - Update state.regime field
    - _Requirements: 23.4_
  
  - [ ] 6.4 Implement inject_regime function
    - Force specific regime at initialization (low-vol-expanding, high-vol-compressing, stable)
    - Set variance and drift parameters accordingly
    - _Requirements: 3.1_
  
  - [ ] 6.5 Implement trigger_vol_crush function
    - Specialized regime shift for earnings events
    - Reduce variance by 30-50% (multiply by 0.5-0.7)
    - Update state.regime to "post_earnings"
    - _Requirements: 5.2_
  
  - [ ] 6.6 Implement inject_oscillation function
    - Force larger spot price oscillations for gamma scalping
    - Increase GBM volatility or inject deterministic oscillation pattern
    - _Requirements: 6.2_
  
  - [ ] 6.7 Implement trigger_stress_event function
    - Combined vol spike (40-60%) AND spot crash (5-8%)
    - Apply both simultaneously
    - Update state.regime to "crisis"
    - _Requirements: 7.2_

- [x] 7. Portfolio Manager (vsr_env/engine/portfolio.py)
  - [x] 7.1 Implement add_position function
    - Compute entry_price, entry_iv, entry_spot using OptionChainEngine
    - Compute position Greeks (delta, gamma, vega)
    - Adjust sign based on direction (buy = positive, sell = negative)
    - Append position dict to state.positions
    - _Requirements: 22.1, 22.2, 22.7_
  
  - [x] 7.2 Implement compute_portfolio_greeks function
    - Recompute Greeks for all positions at current market conditions
    - Sum individual position Greeks to get portfolio totals
    - Return dict with delta, gamma, vega, theta
    - _Requirements: 22.4_
  
  - [x] 7.3 Implement compute_portfolio_pnl function
    - Recompute current_price for each position
    - P&L = (current_price - entry_price) * quantity for buy
    - P&L = (entry_price - current_price) * quantity for sell
    - Sum all position P&L values
    - _Requirements: 22.5, 22.6_
  
  - [x] 7.4 Implement update_positions_on_market_move function
    - Call compute_portfolio_greeks and update state
    - Call compute_portfolio_pnl and update state
    - _Requirements: 22.4, 22.5_


- [x] 8. Reward Computation (vsr_env/reward/reward_computer.py)
  - [x] 8.1 Implement RewardComputer class
    - Create class with methods for each task's reward computation
    - _Requirements: 10.1_
  
  - [x] 8.2 Implement compute_iv_reading_reward method
    - identification_component: 0.5 if correct strike and direction, 0.1 if correct strike only
    - reasoning_component: score_reasoning_quality * 0.2
    - total = min(identification + reasoning, 1.0)
    - _Requirements: 10.2_
  
  - [x] 8.3 Implement compute_delta_hedging_reward method
    - delta_improvement = max(0, (old_delta - new_delta) / old_delta) * 0.6
    - cost_efficiency = max(0, 0.4 - trade_cost * 0.1)
    - neutrality_bonus = 0.1 if |delta| < 0.05 else 0.0
    - total = min(delta_improvement + cost_efficiency + neutrality_bonus, 1.0)
    - _Requirements: 10.3_
  
  - [x] 8.4 Implement compute_arb_capture_reward method
    - pnl_component = sigmoid(pnl_change, scale=0.3) * 0.4
    - greek_component = (1.0 - min(|delta| / 0.5, 1.0)) * 0.3
    - reasoning_component = score_reasoning_quality * 0.3
    - total = min(pnl + greek + reasoning, 1.0)
    - _Requirements: 10.4_
  
  - [x] 8.5 Implement sigmoid helper function
    - Formula: 1.0 / (1.0 + exp(-x / scale))
    - Default scale = 0.3
    - _Requirements: 10.4_
  
  - [x] 8.6 Implement score_reasoning_quality method
    - Keyword presence score (max 0.4): count domain keywords, score = min(hits / 4.0, 1.0) * 0.4
    - Numeric consistency score (max 0.6): check for spot price, IV values, delta citations
    - Length penalty: multiply by 0.3 if len(reasoning) <= 20
    - Clamp to [0.0, 1.0]
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8_

- [x] 9. Action Validation (vsr_env/server/vsr_environment.py)
  - [x] 9.1 Implement validate_action function
    - Check selected_strike in [0, 7], return error if invalid
    - Check selected_maturity in [0, 2], return error if invalid
    - Check quantity >= 0, return error if negative
    - Check quantity <= 10.0, return error if excessive
    - Check direction in [BUY, SELL, HOLD], return error if invalid
    - Check hold action has quantity = 0
    - Return None if valid, error string if invalid
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_

- [x] 10. Task Implementations
  - [ ] 10.1 Create VolRegimeDetectionTask class (vsr_env/tasks/vol_regime_detection.py)
    - Implement initialize method: inject specific regime (low-vol-expanding, high-vol-compressing, stable)
    - Store regime type in state for grading
    - Implement get_description method: return task objective
    - _Requirements: 2.1, 3.1_
  
  - [x] 10.2 Create DeltaHedgingTask class (vsr_env/tasks/delta_hedging.py) - UPGRADE
    - Add regime_shift_step randomization (step 2 or 3)
    - Trigger market shock at regime_shift_step
    - Track pre-shock and post-shock delta neutrality
    - Update get_description method to mention event-driven risk management
    - _Requirements: 2.2, 4.1, 4.2_
  
  - [ ] 10.3 Create EarningsVolCrushTask class (vsr_env/tasks/earnings_vol_crush.py)
    - Implement initialize method: set elevated IV (base_vol × 1.3-1.5)
    - Set vol_crush_step to random 3-6
    - Implement get_description method
    - _Requirements: 2.3, 5.1, 5.2_
  
  - [ ] 10.4 Create GammaScalpingTask class (vsr_env/tasks/gamma_scalping.py)
    - Implement initialize method: create long ATM straddle position
    - Configure market for spot oscillations ±2-3%
    - Implement get_description method
    - _Requirements: 2.4, 6.1_
  
  - [ ] 10.5 Create VegaGammaStressTask class (vsr_env/tasks/vega_gamma_stress.py)
    - Implement initialize method: create multi-leg portfolio (long 90-day + short 30-day straddles)
    - Set stress_event_step to random 4-7
    - Implement get_description method
    - _Requirements: 2.5, 7.1, 7.2_


- [x] 11. Grader Implementations
  - [ ] 11.1 Create VolRegimeDetectionGrader class (vsr_env/tasks/vol_regime_detection.py)
    - Implement score method
    - Check regime classification correctness (0.6 weight)
    - Check trade consistency with regime (0.25 weight)
    - Check reasoning quality (0.15 weight)
    - Return regime_correct × 0.6 + trade_consistency × 0.25 + reasoning × 0.15
    - _Requirements: 3.5, 3.6, 8.4_
  
  - [ ]*11.2 Write unit tests for VolRegimeDetectionGrader
    - Test score returns values in [0.0, 1.0]
    - Test correct regime classification
    - _Requirements: 26.2_
  
  - [x] 11.3 Create DeltaHedgingGrader class (vsr_env/tasks/delta_hedging.py) - UPGRADE
    - Update score method for pre/post shock scoring
    - pre_shock_neutrality: was delta neutral before shock?
    - post_shock_neutrality: was delta neutral after shock?
    - Return pre_shock × 0.30 + post_shock × 0.40 + cost_efficiency × 0.30
    - _Requirements: 4.5, 4.6_
  
  - [ ]*11.4 Write unit tests for DeltaHedgingGrader
    - Test pre/post shock neutralization scoring
    - Test score in [0.0, 1.0]
    - _Requirements: 26.2_
  
  - [ ] 11.5 Create EarningsVolCrushGrader class (vsr_env/tasks/earnings_vol_crush.py)
    - Implement score method
    - pre_crush_positioning: was vega negative before crush?
    - post_crush_rehedge: did agent re-hedge within 2 steps?
    - pnl_outcome: sigmoid-normalized final P&L
    - Return pre_crush × 0.40 + post_crush × 0.35 + pnl × 0.25
    - _Requirements: 5.4, 5.5, 5.6_
  
  - [ ]*11.6 Write unit tests for EarningsVolCrushGrader
    - Test pre-crush vega positioning
    - Test score in [0.0, 1.0]
    - _Requirements: 26.2_
  
  - [ ] 11.7 Create GammaScalpingGrader class (vsr_env/tasks/gamma_scalping.py)
    - Implement score method
    - rehedge_quality: delta neutrality at each step
    - pnl_above_theta: was P&L positive after theta decay?
    - timing_score: correlation between spot moves and hedge quantities
    - Return rehedge × 0.40 + pnl_above_theta × 0.35 + timing × 0.25
    - _Requirements: 6.4, 6.5, 6.6_
  
  - [ ]*11.8 Write unit tests for GammaScalpingGrader
    - Test timing correlation scoring
    - Test score in [0.0, 1.0]
    - _Requirements: 26.2_
  
  - [ ] 11.9 Create VegaGammaStressGrader class (vsr_env/tasks/vega_gamma_stress.py)
    - Implement score method
    - survival: is final P&L > -2.0?
    - greek_management: delta + vega neutrality weighted post-stress
    - reasoning: enhanced scoring requiring vega, gamma, stress keywords
    - Return survival × 0.35 + greek_management × 0.35 + reasoning × 0.30
    - _Requirements: 7.4, 7.5, 7.6_
  
  - [ ]*11.10 Write unit tests for VegaGammaStressGrader
    - Test survival threshold
    - Test enhanced reasoning scoring
    - Test score in [0.0, 1.0]
    - _Requirements: 26.2_

- [x] 12. Core Environment Implementation (vsr_env/server/vsr_environment.py)
  - [x] 12.1 Create VSREnvironment class inheriting from Environment
    - Define TASKS dict with max_steps and descriptions
    - Define STRIKES and MATURITIES constants
    - Initialize OptionChainEngine, graders dict
    - _Requirements: 1.1, 1.4_
  
  - [x] 12.2 Implement reset method
    - Accept task_name and seed parameters
    - Initialize seeded numpy RandomState
    - Create new VSRState with episode_id
    - Call task-specific initialization
    - Generate IV surface
    - Return VSRObservation
    - _Requirements: 1.2, 6.1, 6.2_
  
  - [x] 12.3 Implement step method
    - Increment step_count
    - Validate action
    - Execute action if valid (update portfolio)
    - Advance market simulation
    - Compute reward
    - Check if done (step_count >= max_steps)
    - Build observation with error if present
    - Append to episode_history
    - Compute grader score if done
    - Return dict with observation, reward, done, info
    - _Requirements: 1.3, 20.6, 20.7_
  
  - [x] 12.4 Implement state property
    - Return current VSRState
    - _Requirements: 1.4_
  
  - [x] 12.5 Implement _execute_action helper
    - If direction is HOLD, do nothing
    - Otherwise call add_position with action parameters
    - _Requirements: 22.1, 22.2, 22.3_
  
  - [x] 12.6 Implement _advance_market helper
    - Call advance_market function
    - Check if regime shift needed (arb_capture task at step 4-5)
    - Call trigger_regime_shift if needed
    - Call update_positions_on_market_move
    - _Requirements: 21.1, 21.4_
  
  - [x] 12.7 Implement _compute_reward helper
    - Get RewardComputer instance
    - Call appropriate reward method based on task_name
    - Return VSRReward
    - _Requirements: 10.1_
  
  - [x] 12.8 Implement _make_observation helper
    - Build VSRObservation from current state
    - Include error parameter if provided
    - _Requirements: 17.1-17.11_


- [x] 13. FastAPI Server (vsr_env/server/app.py)
  - [x] 13.1 Create FastAPI app with metadata
    - Set title="VSR-Env", description, version="1.0.0"
    - _Requirements: 1.7_
  
  - [x] 13.2 Implement /health endpoint
    - Return {"status": "healthy", "environment": "vsr_env"}
    - _Requirements: 14.5_
  
  - [x] 13.3 Implement /reset endpoint
    - Accept task_name and seed parameters
    - Call env.reset(task_name, seed)
    - Return {"observation": observation.dict()}
    - Handle exceptions with HTTPException
    - _Requirements: 1.7, 15.2_
  
  - [x] 13.4 Implement /step endpoint
    - Accept VSRAction in request body
    - Call env.step(action)
    - Return observation, reward, done, info
    - Handle exceptions with HTTPException
    - _Requirements: 1.7, 15.3_
  
  - [x] 13.5 Implement /state endpoint
    - Return {"state": env.state.dict()}
    - Handle exceptions with HTTPException
    - _Requirements: 1.7_

- [x] 14. Checkpoint - Ensure skeleton passes validation
  - Run `docker build -t vsr-env:latest .`
  - Run `docker run -p 8000:8000 vsr-env:latest`
  - Test `curl http://localhost:8000/health`
  - Test `curl -X POST http://localhost:8000/reset`
  - Run `openenv validate` and verify all checks pass
  - Ask user if any issues arise

- [x] 15. Baseline Inference Script (inference.py at root)
  - [x] 15.1 Set up OpenAI client configuration
    - Read API_BASE_URL, HF_TOKEN, MODEL_NAME from environment
    - Initialize OpenAI client with base_url and api_key
    - _Requirements: 14.1_
  
  - [ ] 15.2 Define task configurations - UPDATE
    - TASKS = ["vol_regime_detection", "delta_hedging", "earnings_vol_crush", "gamma_scalping", "vega_gamma_stress"]
    - MAX_STEPS_PER_TASK = {"vol_regime_detection": 3, "delta_hedging": 5, "earnings_vol_crush": 8, "gamma_scalping": 10, "vega_gamma_stress": 12}
    - TASK_SEEDS = {"vol_regime_detection": 42, "delta_hedging": 123, "earnings_vol_crush": 456, "gamma_scalping": 789, "vega_gamma_stress": 101}
    - _Requirements: 14.2, 14.6_
  
  - [ ] 15.3 Define system prompts for each task - UPDATE
    - vol_regime_detection: "You are an options trader analyzing volatility regime changes..."
    - delta_hedging: "You are managing an options portfolio through market disruption..."
    - earnings_vol_crush: "You are positioning for an earnings volatility event..."
    - gamma_scalping: "You are exploiting gamma convexity through spot oscillations..."
    - vega_gamma_stress: "You are managing a multi-Greek portfolio through extreme stress..."
    - _Requirements: 14.1_
  
  - [x] 15.4 Implement log_start function
    - Print "[START] task={task} env={env} model={model}" to stdout with flush=True
    - _Requirements: 14.3_
  
  - [x] 15.5 Implement log_step function
    - Print "[STEP] step={N} action={action} reward={reward:.2f} done={bool} error={error}" to stdout
    - _Requirements: 14.4_
  
  - [x] 15.6 Implement log_end function
    - Print "[END] success={bool} steps={N} score={score:.2f} rewards={comma_separated}" to stdout
    - _Requirements: 14.5_
  
  - [x] 15.7 Implement parse_llm_response function
    - Try direct JSON parse
    - Try extracting JSON from markdown code blocks
    - Return safe default (hold action) on parse failure
    - _Requirements: 14.7, 14.8_
  
  - [x] 15.8 Implement build_prompt function
    - Format IV surface as table
    - Include spot price, portfolio Greeks, P&L, positions
    - Include market sentiment and last error if present
    - _Requirements: 14.7_
  
  - [x] 15.9 Implement run_task async function
    - Reset environment with fixed seed
    - Loop for max_steps
    - Build prompt, call LLM, parse response
    - Create VSRAction, execute step
    - Log each step
    - Extract grader score from info on completion
    - Handle errors gracefully
    - _Requirements: 14.2, 14.3, 14.4, 14.5_
  
  - [x] 15.10 Implement main async function
    - Initialize EnvClient
    - Run all five tasks sequentially
    - Print final summary
    - _Requirements: 14.2_


- [ ]*16. Integration Tests (tests/test_environment.py)
  - [ ]*16.1 Write test_reset_step_state_cycle
    - Test complete reset/step/state cycle
    - Verify observation structure
    - Verify state updates
    - _Requirements: 24.3_
  
  - [ ]*16.2 Write test_deterministic_reproducibility
    - Reset twice with same seed
    - Verify identical IV surfaces and spot prices
    - _Requirements: 24.3, 6.3_
  
  - [ ]*16.3 Write test_action_validation_errors
    - Test invalid strike, maturity, quantity
    - Verify error messages match requirements
    - _Requirements: 24.5_

- [ ]*17. Reproducibility Tests (tests/test_reproducibility.py)
  - [ ]*17.1 Write test_same_seed_same_mispricings
    - Reset twice with same seed
    - Verify identical mispriced cells
    - _Requirements: 24.6_
  
  - [ ]*17.2 Write test_different_seed_different_episodes
    - Reset with different seeds
    - Verify different IV surfaces
    - _Requirements: 24.6_
  
  - [ ]*17.3 Write test_grader_determinism
    - Compute grader score twice for same episode
    - Verify identical scores
    - _Requirements: 24.6_

- [ ]*18. Inference Script Tests (tests/test_inference.py)
  - [ ]*18.1 Write test_inference_stdout_format
    - Run inference.py and capture stdout
    - Verify [START], [STEP], [END] patterns match requirements
    - _Requirements: 24.4_
  
  - [ ]*18.2 Write test_inference_completes_in_time
    - Run inference.py with timeout
    - Verify completion in < 20 minutes
    - _Requirements: 24.4, 25.2_

- [x] 19. Docker Configuration
  - [x] 19.1 Create Dockerfile
    - FROM python:3.11-slim
    - Install curl for healthcheck
    - Copy requirements.txt and install dependencies
    - Copy application code
    - Install package with pip install -e .
    - HEALTHCHECK with curl to /health endpoint
    - EXPOSE 8000
    - CMD to run uvicorn
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_
  
  - [x] 19.2 Create .dockerignore
    - Exclude __pycache__, .git, .venv, tests, .pytest_cache
    - _Requirements: 14.6_
  
  - [x] 19.3 Test Docker build and run
    - Build image: docker build -t vsr-env:latest .
    - Run container: docker run -p 8000:8000 vsr-env:latest
    - Verify health endpoint responds
    - _Requirements: 14.6, 14.7_

- [x] 20. OpenEnv Manifest (openenv.yaml)
  - [ ] 20.1 Create openenv.yaml - UPDATE
    - name: vsr-env
    - version: 1.0.0
    - description: Volatility Surface Reasoning Environment for options portfolio management
    - tasks: [vol_regime_detection, delta_hedging, earnings_vol_crush, gamma_scalping, vega_gamma_stress]
    - action_type: VSRAction
    - observation_type: VSRObservation
    - reward_type: float
    - state_type: VSRState
    - _Requirements: 1.6_


- [x] 21. Documentation (README.md)
  - [x] 21.1 Write project overview section
    - Describe VSR-Env purpose and real-world utility
    - Explain options portfolio management simulation
    - _Requirements: 25.1_
  
  - [x] 21.2 Document action space
    - List all VSRAction fields with descriptions and valid ranges
    - _Requirements: 25.2_
  
  - [x] 21.3 Document observation space
    - List all VSRObservation fields with descriptions
    - _Requirements: 25.3_
  
  - [ ] 21.4 Document tasks - UPDATE
    - Describe each of 5 tasks: objective, max steps, grading criteria
    - Include expected baseline and frontier scores
    - _Requirements: 25.4, 25.5_
  
  - [x] 21.5 Write installation instructions
    - pip install requirements
    - docker build command
    - _Requirements: 25.6_
  
  - [x] 21.6 Write usage examples
    - Show reset and step API calls
    - Show inference.py usage
    - _Requirements: 25.7_
  
  - [x] 21.7 Document environment variables
    - API_BASE_URL, MODEL_NAME, HF_TOKEN, IMAGE_NAME, LOG_LEVEL
    - _Requirements: 25.8_

- [ ] 22. HuggingFace Spaces Deployment
  - [ ] 22.1 Create HuggingFace Space
    - Create new Space with Docker SDK
    - Add openenv tag
    - _Requirements: 15.1_
  
  - [ ] 22.2 Configure Space settings
    - Set environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
    - _Requirements: 15.5_
  
  - [ ] 22.3 Push code to Space repository
    - Push Dockerfile, code, requirements.txt, openenv.yaml
    - _Requirements: 15.4_
  
  - [ ] 22.4 Verify Space deployment
    - Wait for build to complete
    - Test /reset endpoint with curl
    - Test /step endpoint with sample action
    - _Requirements: 15.2, 15.3_
  
  - [ ] 22.5 Run pre_validation script
    - Execute pre_validation against Space URL
    - Verify all three validation checks pass
    - _Requirements: 15.6_

- [ ] 23. Final Validation and Testing
  - [ ] 23.1 Run full test suite
    - Execute pytest on all tests
    - Verify all tests pass
    - _Requirements: 26.7_
  
  - [ ] 23.2 Run openenv validate locally
    - Verify environment passes all validation checks
    - _Requirements: 1.5_
  
  - [ ] 23.3 Test inference.py end-to-end
    - Run inference.py against local environment
    - Verify stdout format is correct
    - Verify completion time < 20 minutes
    - _Requirements: 14.3, 14.4, 14.5, 27.2_
  
  - [ ] 23.4 Verify deterministic reproducibility
    - Run same task with same seed multiple times
    - Verify identical grader scores
    - _Requirements: 8.3, 8.6_
  
  - [ ] 23.5 Performance benchmarking
    - Measure single step time (target < 2 seconds)
    - Measure full episode time (target < 16 seconds)
    - Measure all 5 tasks time (target < 10 minutes)
    - _Requirements: 27.1, 27.2, 27.3_

- [x] 24. Checkpoint - Final pre-submission validation
  - Verify Docker image builds successfully
  - Verify HuggingFace Space is accessible
  - Verify openenv validate passes
  - Verify inference.py completes successfully
  - Verify all documentation is complete
  - Ask user if ready to submit


## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP delivery
- Tasks marked with `UPDATE` require modifications to existing code
- Each task references specific requirements for traceability
- Checkpoints (14, 24) ensure incremental validation at critical milestones
- The 3-day sprint structure prioritizes:
  - Day 1: Core infrastructure and skeleton (tasks 1-14)
  - Day 2: Task implementations, graders, and inference script (tasks 15-18)
  - Day 3: Deployment, documentation, and final validation (tasks 19-24)
- Focus on critical path: get validation passing first, then add features
- Testing tasks are optional but recommended for production quality
- All code should use Python 3.11+ with type hints
- All mathematical operations should use NumPy/SciPy for CPU efficiency
- No GPU dependencies (torch, cuda) allowed per hackathon constraints

## Task Restructure Summary

The spec has been updated from 3 tasks to 5 tasks based on final_task_restructure.md:

**Removed:**
- ~~IV Reading (Easy)~~ → Replaced by Vol Regime Detection
- ~~Arb Capture (Hard)~~ → Replaced by Earnings Vol Crush

**Added:**
- Vol Regime Detection (Easy, 3 steps) - Agent observes surface changes and classifies regime
- Earnings Vol Crush (Hard, 8 steps) - Agent positions for earnings vol crush event
- Gamma Scalping (Expert, 10 steps) - Agent exploits gamma through spot oscillations
- Vega-Gamma Stress (Super-Boss, 12 steps) - Agent manages multi-Greek crisis

**Upgraded:**
- Delta Hedging (Medium, 5 steps) - Now includes random market shock at step 2-3

**Key Implementation Changes:**
1. Market simulator needs 4 new functions: inject_regime, trigger_vol_crush, inject_oscillation, trigger_stress_event
2. Five new task classes and graders need to be implemented
3. Inference script needs updated task list and seeds
4. openenv.yaml needs updated task list
5. README needs updated task descriptions

## Implementation Order Rationale

The task order follows dependency chains:
1. Models first (needed by all components)
2. Engine components (pricing, IV solver, surface generation)
3. Simulation and portfolio management (depend on engine)
4. Reward computation (depends on portfolio state)
5. Tasks and graders (depend on all above)
6. Core environment (orchestrates all components)
7. Server and inference (expose environment functionality)
8. Deployment and validation (final integration)

This order ensures each component has its dependencies available when implemented.

## Critical Success Factors

1. **Validation passing**: Must pass `openenv validate` before deployment
2. **Deterministic grading**: Same seed must produce same scores
3. **Performance**: Must complete in < 20 minutes total runtime
4. **Stdout format**: inference.py must match exact [START]/[STEP]/[END] format
5. **CPU-only**: No GPU dependencies, all NumPy/SciPy
6. **Error handling**: Robust validation and graceful degradation
7. **Documentation**: Complete README with all required sections

## Time Estimates

Based on 3-day sprint (72 hours total, ~24 hours per day):

- Day 1 (24h): Tasks 1-14 (setup, models, engine, core env, validation)
- Day 2 (24h): Tasks 15-18 (inference, integration tests)
- Day 3 (24h): Tasks 19-24 (Docker, deployment, docs, final validation)

Each major task group (2-3 hours):
- Models: 2h
- Engine: 4h
- Simulation/Portfolio: 3h
- Rewards: 2h
- Tasks/Graders: 3h
- Core Environment: 4h
- Server: 2h
- Inference: 3h
- Docker/Deployment: 3h
- Documentation: 2h
- Testing/Validation: 4h

Total: ~32 hours of focused implementation time, leaving buffer for debugging and iteration.
