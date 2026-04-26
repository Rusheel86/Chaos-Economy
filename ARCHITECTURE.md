# Architecture Overview: VSR-Env

VSR-Env is a high-fidelity multi-agent options market simulation built to demonstrate systemic risk, emergent collusion, and regulatory enforcement.

---

## 🏗️ Core System Architecture

```mermaid
flowchart TD
    subgraph Market_Environment [Multi-Agent Environment]
        T["Traders 0-9"] -->|"Orders & Messages"| OME["Order Matching Engine"]
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

---

## 🎭 Agent Interaction Flow

During each step, the environment processes actions in a sequential, deterministic order to ensure market microstructure rules are respected.

```mermaid
sequenceDiagram
    participant T as "Traders (0-9)"
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

---

## 🔍 Oversight & Regulatory Flow

The SEC agent acts as a dynamic supervisor. Its interventions directly alter the environment's state, acting as a forcing function for Act IV.

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

---

## 🗂️ Core Components

1. **`train_multi_agent_pipeline.py`**: The orchestration layer. Manages the 4-act curriculum and drives the RL loop using GRPO.
2. **`vsr_environment.py`**: The step-execution engine. Handles deterministic order matching and state transitions.
3. **`multi_agent/rewards.py`**: The institutional-grade grading module. Computes precise rewards for each role (see [REWARDS.md](./REWARDS.md)).
4. **`multi_agent/manipulation_detector.py`**: Ground-truth heuristics used to evaluate the SEC agent's accuracy.
