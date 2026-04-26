"""OpenEnv-compatible server entry point for VSR-Env.

This module re-exports the FastAPI app from vsr_env.server.app and adds
the static file serving for the replay dashboard (frontend.html / index.html).

OpenEnv's `openenv validate` expects server/app.py at the project root.
"""

# Re-export the actual app — all routes (/reset, /step, /state, /health, /ws, etc.)
# are defined in vsr_env/server/app.py.
from vsr_env.server.app import app  # noqa: F401

import os
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Serve the replay dashboard frontend at /dashboard
# ---------------------------------------------------------------------------

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SERVER_DIR)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the Chaos-Economy replay visualizer dashboard."""
    index_path = os.path.join(_SERVER_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="Dashboard index.html not found", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/replay/{filename}")
async def serve_replay(filename: str):
    """Serve replay JSON files (unified_lora_replay.json, unified_baseline_replay.json)."""
    safe_names = {"unified_lora_replay.json", "unified_baseline_replay.json"}
    if filename not in safe_names:
        return HTMLResponse(content="Not found", status_code=404)
    fpath = os.path.join(_PROJECT_ROOT, filename)
    if not os.path.exists(fpath):
        return HTMLResponse(content=f"{filename} not found", status_code=404)
    return FileResponse(fpath, media_type="application/json")


def main():
    """Main entry point for starting the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
