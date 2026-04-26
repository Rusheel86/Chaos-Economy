"""FastAPI server for VSR-Env.

Exposes the VSREnvironment via HTTP endpoints:
  POST /reset   — Start new episode
  POST /step    — Execute an action
  GET  /state   — Get current state
  GET  /health  — Health check
"""

import logging
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import json
import yaml
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import os

from vsr_env.models import VSRAction
from vsr_env.server.vsr_environment import VSREnvironment
from vsr_env.server.telemetry import tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("vsr_env")

# Create app and environment
app = FastAPI(
    title="VSR-Env",
    description="Volatility Surface Reasoning Environment for options portfolio management",
    version="1.0.0",
)

env = VSREnvironment()


# Validate openenv.yaml manifest on startup
def validate_manifest():
    manifest_path = os.path.join(os.path.dirname(__file__), "..", "..", "openenv.yaml")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"Missing openenv.yaml manifest at {manifest_path}")
    try:
        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
            if "name" not in manifest or "tasks" not in manifest:
                raise ValueError("Manifest missing required 'name' or 'tasks' fields")
            logger.info(
                f"Loaded openenv.yaml manifest successfully with {len(manifest['tasks'])} tasks."
            )
    except Exception as e:
        raise RuntimeError(f"Invalid openenv.yaml manifest: {e}")


validate_manifest()

# === Request/Response models ===


class ResetRequest(BaseModel):
    task_name: str = "delta_hedging"
    seed: int = 42


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# === Endpoints ===


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Custom Web UI dashboard at the root URL."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "..", "server", "index.html")
    try:
        with open(ui_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        logger.error(f"Failed to load UI: {e}")
        return HTMLResponse(content=f"Error loading UI: {e}", status_code=500)


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Serve the Custom Web UI dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(ui_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        logger.error(f"Failed to load UI: {e}")
        return HTMLResponse(content=f"Error loading UI: {e}", status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "vsr_env", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment and start a new episode.

    Args:
        request: Optional reset parameters (task_name, seed)

    Returns:
        Initial observation
    """
    try:
        task_name = request.task_name if request else "delta_hedging"
        seed = request.seed if request else 42

        logger.info(f"Resetting environment: task={task_name}, seed={seed}")
        observation = env.reset(task_name=task_name, seed=seed)

        # Start tracking the episode
        tracker.start_episode(
            episode_id=env.state.episode_id, task_name=task_name, seed=seed
        )

        return {"observation": observation.model_dump()}
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: VSRAction):
    """Execute one step in the environment.

    Args:
        action: The agent's action (VSRAction)

    Returns:
        observation, reward, done, info
    """
    try:
        logger.info(
            f"Step: strike={action.selected_strike}, mat={action.selected_maturity}, "
            f"dir={action.direction.value}, qty={action.quantity}"
        )
        result = env.step(action)

        # Record step in telemetry
        if env.state.episode_id:
            tracker.record_step(
                episode_id=env.state.episode_id,
                step_data={
                    "action": action.model_dump(),
                    "observation": result["observation"].model_dump(),
                    "reward": result["reward"],
                    "done": result["done"],
                },
            )
            if result["done"]:
                tracker.complete_episode(
                    episode_id=env.state.episode_id,
                    final_score=result["info"].get("grader_score", 0.01),
                )

        return {
            "observation": result["observation"].model_dump(),
            "reward": result["reward"],
            "done": result["done"],
            "info": result["info"],
        }
    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state (including hidden info)."""
    try:
        return {"state": env.state.model_dump()}
    except Exception as e:
        logger.error(f"State retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class GradeRequest(BaseModel):
    task_name: str
    episode_history: list

@app.post("/grade")
async def grade_episode(request: GradeRequest):
    """Evaluate a complete episode history and return a grader score.
    
    Args:
        request: GradeRequest containing task_name and episode_history
        
    Returns:
        Grader score (0.01 - 0.99)
    """
    try:
        from vsr_env.server.vsr_environment import TASK_CONFIG
        from vsr_env.models import VSRState
        
        if request.task_name not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {request.task_name}")
            
        config = TASK_CONFIG[request.task_name]
        grader = config["grader_class"]()
        
        # We construct a mock state from the last observation
        state = VSRState()
        if request.episode_history:
            last_step = request.episode_history[-1]
            if "observation" in last_step:
                obs = last_step["observation"]
                if "portfolio_greeks" in obs:
                    state.portfolio_delta = obs["portfolio_greeks"].get("delta", 0.0)
                    state.portfolio_gamma = obs["portfolio_greeks"].get("gamma", 0.0)
                    state.portfolio_vega = obs["portfolio_greeks"].get("vega", 0.0)
                    
            # For delta hedging
            if request.task_name == "delta_hedging":
                # Find initial delta from the first observation
                first_obs = request.episode_history[0].get("observation", {})
                state.initial_delta = abs(first_obs.get("portfolio_greeks", {}).get("delta", 0.5))
                state.regime_shift_step = 3 # hardcode default
        
        score = grader.score(request.episode_history, state)
        return {"grader_score": score}
        
    except Exception as e:
        logger.error(f"Grading failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/telemetry")
async def get_telemetry():
    """Get the full telemetry trace of all episodes."""
    try:
        return {"telemetry": tracker.get_all_episodes()}
    except Exception as e:
        logger.error(f"Telemetry retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for continuous environment interaction."""
    await websocket.accept()
    # Create thread-local env for this connection
    ws_env = VSREnvironment()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)

                if msg.get("action") == "reset":
                    task_name = msg.get("task_name", "delta_hedging")
                    seed = msg.get("seed", 42)
                    obs = ws_env.reset(task_name=task_name, seed=seed)
                    await websocket.send_json(
                        {"type": "reset", "observation": obs.model_dump()}
                    )

                elif msg.get("action") == "step":
                    vsr_action = VSRAction(**msg.get("payload", {}))
                    result = ws_env.step(vsr_action)
                    await websocket.send_json(
                        {
                            "type": "step",
                            "observation": result["observation"].model_dump(),
                            "reward": result["reward"],
                            "done": result["done"],
                            "info": result["info"],
                        }
                    )

                elif msg.get("action") == "state":
                    await websocket.send_json(
                        {"type": "state", "state": ws_env.state.model_dump()}
                    )

            except Exception as e:
                logger.error(f"WS error processing message: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
