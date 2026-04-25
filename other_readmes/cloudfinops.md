# ☁️ Cloud FinOps Cost Optimization

## 🚀 Project Metadata

| Key        | Value                                                                 |
|------------|----------------------------------------------------------------------|
| **Title**  | OpenEnv Cloud FinOps                                                 |
| **Emoji**  | ☁️                                                                   |
| **SDK**    | Docker                                                               |
| **Port**   | 7860                                                                 |
| **License**| MIT                                                                  |
| **Pinned** | false                                                                |
| **Tags**   | openenv • cloud • finops • reinforcement-learning • llm-agents • fastapi • simulation |

---

## 🧰 Tech Stack

<p align="center">

<img src="https://img.shields.io/badge/python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/fastapi-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/pydantic-%23E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white"/>
<img src="https://img.shields.io/badge/uvicorn-%23000000.svg?style=for-the-badge&logo=uvicorn&logoColor=white"/>

<br/>

<img src="https://img.shields.io/badge/Reinforcement%20Learning-Agent-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LLM-Agents-black?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/API-REST-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Cloud-Simulation-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenEnv-Evaluation-purple?style=for-the-badge"/>

</p>

---

# OpenEnv Cloud FinOps

**Cloud Infrastructure Cost Optimization Environment**

---

## Environment Description

OpenEnv Cloud FinOps is an OpenEnv-compliant reinforcement learning environment that simulates cloud infrastructure cost optimization for NovaCart, a fictional mid-size e-commerce platform.

The agent acts as an automated FinOps / SRE engineer responsible for reducing cloud spend without disrupting production systems. The environment features three progressively difficult tasks covering orphan resource cleanup, resource rightsizing with cold storage migration, and multi-region traffic consolidation. It exposes a REST API via FastAPI, enabling both programmatic agent interaction and manual testing.

---

## Motivation

Every cloud-based company faces escalating infrastructure costs. Resources are frequently forgotten, systems become over-provisioned, and engineers manually review dashboards to identify waste. This process is time-consuming, error-prone, and does not scale.

This environment recreates that real-world decision loop so that AI agents can be trained and evaluated on safe, efficient cost optimization — testing whether agents can reason about cloud state, avoid production disruptions, and sequence actions correctly across increasingly complex scenarios.

---

## Action Space

The agent submits actions as structured JSON to the `/step` endpoint.

| Action | Description |
|--------|-------------|
| `terminate` | Permanently delete a resource, saving its full monthly cost. High risk if the resource is active or has dependencies. |
| `resize` | Downgrade a VM or database to a smaller instance tier. Saves the cost delta between old and new size. |
| `migrate_storage` | Move a storage volume to cold tier, saving approximately 80% of its monthly cost. |
| `migrate_traffic` | Redirect live traffic away from a source region. Required before terminating regional resources (task_3 only). |
| `wait` | Drain connections after traffic migration. Unlocks safe termination of east region resources (task_3 only). |

**Action schemas:**

```json
{ "action_type": "terminate", "resource_id": "vol-0a1b2c3d" }
```

```json
{ "action_type": "resize", "resource_id": "i-0a1b2c3d", "new_size": "small" }
```

```json
{ "action_type": "migrate_storage", "resource_id": "vol-archive-01" }
```

```json
{ "action_type": "migrate_traffic", "source_region": "us-east-1", "target_region": "us-west-2" }
```

```json
{ "action_type": "wait" }
```

**Instance size tiers:**

| Size | Cost Multiplier |
|------|----------------|
| nano | 9% of large |
| micro | 15% of large |
| small | 34% of large |
| medium | 61% of large |
| large | 100% (reference) |
| xlarge | 200% of large |

---

## Observation Space

Each response from `/step` or `/state` returns a full observation object. The agent receives the following fields:

| Field | Description |
|-------|-------------|
| `task_id` | Current task identifier |
| `step` | Current step number |
| `max_steps` | Maximum steps allowed for the task |
| `monthly_bill_start` | Initial infrastructure cost at episode start |
| `monthly_bill_current` | Current running cost after agent actions |
| `savings_target` | Dollar amount the agent is expected to save |
| `savings_achieved` | Savings accumulated so far |
| `uptime_percent` | System reliability indicator |
| `downtime_events` | Number of production outages triggered |
| `honeypot_hits` | Number of critical resources incorrectly deleted |
| `sequence_violations` | Number of ordering constraint violations |
| `active_regions` | Regions currently active in the simulation |
| `traffic_migrated` | Whether traffic migration has been completed |
| `connections_drained` | Whether connection draining is complete |
| `feedback` | Text explanation of the last action's outcome |
| `resources` | Full list of resource objects visible to the agent |

Each resource object exposes:

`id`, `name`, `resource_type`, `status`, `region`, `monthly_cost`, `cpu_avg_24h`, `ram_avg_24h`, `traffic_per_hour`, `queries_per_hour`, `attached_to`, `dependency_of`, `storage_tier`, `last_accessed_days_ago`, `size_gb`, `instance_size`, `tags`

The following fields are hidden and never exposed to the agent: `safe_to_terminate`, `is_production`, `peak_cpu_2am`, `peak_queries_2am`.

**Resource types:** VM, Database, Storage, IP Address, Snapshot, Load Balancer, CDN

**Resource statuses:** `running`, `stopped`, `orphaned`, `deleted`, `migrated`

---

## Task Descriptions

| Task ID | Name | Difficulty | Max Steps | Starting Bill | Savings Target |
|---------|------|------------|-----------|---------------|----------------|
| task_1 | Orphan Cleanup | Easy | 15 | $8,015 | $705 |
| task_2 | Rightsizing & Cold Migration | Medium | 18 | $11,840 | $3,500 |
| task_3 | Multi-Region Failover | Hard | 35 | 29,000 | $10,000 |

**task_1 — Orphan Cleanup**
20 resources in us-east-1. The agent must identify and delete 8 provably orphaned resources (unassigned IPs, detached volumes, stopped VMs, old snapshots) without touching any production resource.

**task_2 — Rightsizing & Cold Migration**
20 resources in us-east-1. The agent resizes oversized VMs with CPU utilization under 10% and migrates stale archives (last accessed over 400 days)

**task_3 — Multi-Region Failover**
50 resources across us-east-1 and us-west-2. The agent must migrate traffic from the legacy east region, wait for connection draining, then safely terminate east resources. Includes three honeypot traps: midnight batch databases, mislabeled dev resources, and orphan-appearing volumes with hidden dependencies.

---

## Reward Function

The environment emits dense reward signals at each step:

| Event | Reward |
|-------|--------|
| Terminate orphan resource | +monthly_cost |
| Resize oversized instance | +(old_cost − new_cost) |
| Migrate storage to cold tier | +0.8 × storage_cost |
| Successful traffic migration | +50 |
| Safe region shutdown | +100 |
| Terminate active production resource | −10,000 |
| Trigger downtime event | −5,000 |
| Hit honeypot resource | −3,000 |
| Invalid or redundant action | −10 |

---

## Grading / Evaluation

Final scores are computed deterministically via the `/grader` endpoint and clamped to [0.0, 1.0].

**Task 1:**
```
score = (orphans_removed / total_orphans)
        - 0.20 × false_deletions
        - 0.20 × downtime_events
```

**Task 2:**
```
score = min(savings / $3,500, 1.0)
        - 0.20 × downtime_events
        - 0.20 × false_kills
```

**Task 3:**
```
score = min(savings / $10,000, 1.0)
        - 0.20 × downtime
        - 0.20 × false_kills
        - 0.25 × honeypot_hits
        - 0.15 × sequence_violations
```

The grader response includes a full breakdown of `savings_ratio`, each penalty component, and a final verdict.

---

## API Interface

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Environment status and health check |
| GET | `/health` | Health check with version info |
| POST | `/reset?task_id=...` | Start a new episode for task_1, task_2, or task_3 |
| POST | `/step` | Submit an action and receive observation + reward |
| GET | `/state` | Retrieve the current full environment snapshot |
| GET | `/tasks` | List all tasks with metadata and action schemas |
| POST | `/grader` | Compute the deterministic final score for the episode |
| POST | `/baseline` | Run the baseline agent across all tasks and return scores |

**Agent interaction loop:**

```
POST /reset?task_id=task_1  →  initial observation
POST /step { action }       →  step result (observation + reward + done)
POST /step { action }       →  ...
POST /grader                →  final score breakdown
```

---

## Setup Instructions

**Local setup:**

```bash
git clone https://github.com/your-org/openenv-cloud-finops
cd openenv-cloud-finops
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

**Docker:**

```bash
docker build -t cloud-finops .
docker run -p 7860:7860 \
  -e ENV_BASE_URL="http://localhost:7860" \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_your_token" \
  cloud-finops
```

The server will be available at `http://localhost:7860`. Auto-generated API docs are at `http://localhost:7860/docs`.

**LLM agent configuration:**

Set the following environment variables before running `inference.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV_BASE_URL` | Server URL | `http://localhost:7860` |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint | — |
| `MODEL_NAME` | Model identifier | `llama-3.3-70b-versatile` |
| `HF_TOKEN` or `OPENAI_API_KEY` | Authentication token | — |

Run the heuristic baseline (no API key needed):

```bash
python baseline.py
```

Run the LLM agent:

```bash
python inference.py
```

---

## Usage Example

```bash
# Start a task 1 episode
curl -X POST "http://localhost:7860/reset?task_id=task_1"

# Submit a terminate action
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "terminate", "resource_id": "vol-orphan-01"}'

# Check current state
curl "http://localhost:7860/state"

# Get final score
curl -X POST "http://localhost:7860/grader"
```

---

## Baseline Scores

| Task | Heuristic Agent | GPT-4o Agent |
|------|----------------|--------------|
| task_1 — Orphan Cleanup | ~0.85 | ~0.90 |
| task_2 — Rightsizing & Cold Migration | ~0.55 | ~0.65 |
| task_3 — Multi-Region Failover | ~0.30 | ~0.40 |

The heuristic baseline uses deterministic rules and requires no API key. The LLM agent uses GPT-4o with the heuristic as fallback. Task 3 scores are intentionally low due to honeypot traps and ordering constraints — this is expected.

---



## Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Language | Python | 3.11 |
| Web Framework | FastAPI | 0.111.0 |
| ASGI Server | Uvicorn | 0.30.0 |
| Data Validation | Pydantic | 2.7.0 |
| LLM Integration | OpenAI SDK | 1.30.0 |
| HTTP Client | Requests | 2.31.0 |
| Config | python-dotenv | 1.0.1 |
| Containerization | Docker | python:3.11-slim base |

Hugging Face Spaces compatible — listens on port 7860.

---

## Infrastructure Scenarios

| ID | Scenario | Root Cause | Optimization Pattern |
|----|----------|------------|----------------------|
| CF-001 | Orphan Storage Volume | VM deleted, volume left behind | `attached_to: null` → terminate |
| CF-002 | Unused Elastic IP | IP reserved but never assigned | `assigned_to: null` → release |
| CF-003 | Oversized Compute | Low utilization workload | CPU < 10% → resize instance |
| CF-004 | Cold Storage Candidate | Old logs rarely accessed | `last_accessed > 400 days`  → migrate cold |
| CF-005 | Multi-Region Redundancy | Infrastructure duplicated across regions | Migrate traffic → shutdown legacy region |
| CF-006 | Hidden Dependency Trap | Resource appears idle but is critical | Check `dependency_of` before deletion |

## Project Structure

```bash
.
├── Dockerfile
├── README.md
├── requirements.txt
├── openenv.yaml
├── resources.py        # Scenario definitions (6 scenarios across 3 tasks)
├── graders.py      # Deterministic graders for all tasks
├── inference.py    # Baseline agent + smart fallback logic
└── server/
    ├── __init__.py
    ├── app.py          # FastAPI endpoints
    ├── environment.py  # Core OpenEnv step/reset/state logic
    └── models.py       # Typed Pydantic models (Action, Observation, Reward)