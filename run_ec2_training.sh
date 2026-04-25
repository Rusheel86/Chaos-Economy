#!/bin/bash
set -e

# Load WANDB_API_KEY if present
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️ WANDB_API_KEY not found in environment. W&B logging will be disabled."
else
    echo "✅ Found WANDB_API_KEY. W&B logging is enabled."
fi

# Set episodes and steps
NUM_EPISODES=16
EPISODE_LENGTH=50
MAX_STEPS=250

echo "🚀 Starting Training (8 episodes, $MAX_STEPS steps)..."
# Train the multi-agent system and tee output to a log file
python train_multi_agent_pipeline.py \
    --base_model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
    --num_episodes $NUM_EPISODES \
    --episode_length $EPISODE_LENGTH \
    --max_steps $MAX_STEPS \
    --learning_rate 5e-5 \
    --output_dir ./multi_agent_checkpoints 2>&1 | tee train_8_episodes.log

echo "✅ Training Complete. Logs saved to train_8_episodes.log."
echo "Your best checkpoints are inside ./multi_agent_checkpoints (BestModelCallback handled this automatically)."
