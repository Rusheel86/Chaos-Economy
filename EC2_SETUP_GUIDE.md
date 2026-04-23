# EC2 Setup & Training Guide 🚀

Complete from-scratch guide for setting up an AWS EC2 GPU instance to train the VSR-Env multi-agent pipeline.

---

## 0. 🏗️ Provisioning the Instance (AWS Console)

1. **Region**: Select `us-east-1` (N. Virginia) — best availability for `g6` instances.
2. **AMI (Machine Image)**: Search for and select:
   - `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6 (Amazon Linux 2023)`
   - *Why?* Pre-installed NVIDIA drivers, CUDA, and PyTorch — saves ~30 min of setup.
3. **Instance Type**: Select **`g6.xlarge`** (1x NVIDIA L4 24GB, 4 vCPU, 16GB RAM).
   - Fallback: `g5.xlarge` (A10G) if `g6` is unavailable.
4. **Key Pair**: Create or select your key pair (e.g., `vsr`). Download the `.pem` file.
5. **Security Group** — Add this Inbound Rule:
   - **Type**: SSH | **Port**: 22 | **Source**: `0.0.0.0/0` (Allow from all IPs)
6. **Storage**: **100 GB gp3** (model weights + checkpoints need space).
7. Click **Launch Instance**.

---

## 1. 🔑 SSH into the Instance

Once the instance state shows "Running", grab the **Public IPv4 address** from the console.

```bash
# Set permissions (first time only)
chmod 400 vsr.pem

# SSH in
ssh -i vsr.pem ec2-user@<EC2_PUBLIC_IP>
```

---

## 2. 🐙 Clone the Repository

The repo is **private**, so you need a GitHub Personal Access Token (PAT).

### Generate a PAT (if you don't have one)
1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**.
2. Click **Generate new token (classic)**.
3. Select the `repo` scope → **Generate token** → **Copy it immediately**.

### Clone with PAT
```bash
git clone -b Agentic-AI https://***REMOVED***@github.com/manan-tech/VSR-Env.git
cd VSR-Env
```

### Configure Git identity
```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
git config --global credential.helper 'cache --timeout=86400'
```

---

## 3. 🛠️ Environment Setup

The Deep Learning AMI comes with a pre-built PyTorch virtual environment. **Do NOT create a new venv.**

```bash
# Activate the built-in PyTorch environment
source /opt/pytorch/bin/activate

# Install project dependencies
cd ~/VSR-Env
pip install -e .
pip install unsloth "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install accelerate trl peft transformers bitsandbytes

# Verify GPU is visible
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## 4. 🚂 Training

### Option A: Run in a tmux session (Recommended)

tmux keeps your training alive even if your SSH connection drops.

```bash
# Create a new tmux session named "train"
tmux new -s train

# Inside tmux, activate env and start training
source /opt/pytorch/bin/activate
cd ~/VSR-Env

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  train_multi_agent_pipeline.py \
  --num_episodes 250 \
  --dataset_episodes 100
```

#### tmux Cheat Sheet
| Action | Command |
|---|---|
| **Detach** from session (training keeps running) | `Ctrl+B` then `D` |
| **Reattach** to session | `tmux attach -t train` |
| **List** all sessions | `tmux ls` |
| **Kill** a session | `tmux kill-session -t train` |
| **Scroll up** in tmux | `Ctrl+B` then `[`, then arrow keys. Press `q` to exit |

### Option B: Run with nohup
```bash
source /opt/pytorch/bin/activate
cd ~/VSR-Env

nohup accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  train_multi_agent_pipeline.py \
  --num_episodes 250 \
  --dataset_episodes 100 \
  > train.log 2>&1 &

# Monitor
tail -f train.log
```

---

## 5. 🧪 Testing & Evaluation

Run the evaluation script to replay episodes with the trained LoRA adapter:

```bash
source /opt/pytorch/bin/activate
cd ~/VSR-Env

# Quick eval (1 episode, 50 steps)
python test_unified_kaggle.py \
  --lora_path ./multi_agent_checkpoints/unified_v2/unified_market_lora \
  --num_steps 50 \
  --num_episodes 1

# Full eval (3 episodes, 300 steps)
python test_unified_kaggle.py \
  --lora_path ./multi_agent_checkpoints/unified_v2/unified_market_lora \
  --num_steps 300 \
  --num_episodes 3
```

---

## 6. 💾 Downloading Model Weights (Run from LOCAL machine)

### Option A: rsync (Recommended — resumable & fast)
```bash
rsync -avzP \
  -e "ssh -i vsr.pem" \
  ec2-user@<EC2_PUBLIC_IP>:~/VSR-Env/multi_agent_checkpoints/ \
  ./multi_agent_checkpoints_local/
```

### Option B: scp (Simple)
```bash
scp -i vsr.pem -r \
  ec2-user@<EC2_PUBLIC_IP>:~/VSR-Env/multi_agent_checkpoints/ \
  ./multi_agent_checkpoints_local/
```

> **Tip**: If `rsync` gets interrupted, just re-run the same command — it will resume where it left off.

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` shows no GPU | You picked the wrong AMI. Terminate and relaunch with the Deep Learning AMI. |
| `CUDA out of memory` | Reduce `--dataset_episodes` to 50, or lower LoRA rank in the training script. |
| SSH connection refused | Check Security Group inbound rules — port 22 must be open. |
| Training dies after SSH disconnect | You forgot tmux! Always use `tmux new -s train` before launching. |
| `git push` asks for password | Paste your PAT token (not your GitHub password). |
| `RuntimeError: State Dict mismatch` | Base model mismatch. Use the same `--base_model` for both training and testing. |
