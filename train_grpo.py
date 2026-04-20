"""
Multi-Agent GRPO Training for VSR-Env
======================================
pip install unsloth trl peft transformers datasets

Usage:
    python train_grpo.py \
        --base_model unsloth/Llama-3.2-1B-Instruct \
        --num_episodes 200 \
        --steps_per_episode 50 \
        --group_size 4 \
        --output_dir ./vsr_grpo_checkpoints
"""
import json
from transformers import AutoTokenizer
# Unsloth is imported if available, mocked otherwise for syntax completeness.
try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
except ImportError:
    pass

from multi_agent.environment import MultiAgentVSREnvironment

def format_trader_prompt(obs):
    return f"You are a trader. Observation: {obs.model_dump_json()}. Output action JSON:"

def parse_json_action(completion):
    try:
        return json.loads(completion)
    except:
        return {"direction": "hold"}

def run_training():
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0,
    )

    env = MultiAgentVSREnvironment()
    prompts = []
    for seed in range(200):
        obs_dict = env.reset(seed=seed)
        for agent_id, obs in obs_dict.items():
            if obs.role.value == "trader":
                prompts.append({"prompt": format_trader_prompt(obs)})
                break  # one prompt per reset for now

    dataset = Dataset.from_list(prompts)

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            env.reset(seed=i) # deterministic reset per completion in batch
            action = parse_json_action(completion)
            
            # Use scripted actions for others
            from multi_agent.models import MarketMakerAction, OversightAction
            simulated_actions = {
                "trader_0": action,
                "market_maker": MarketMakerAction(atm_spread=0.02, otm_spread=0.04, itm_spread=0.03),
                "oversight": OversightAction(flagged_agents=[], flag_type="none", fine_amount=0.0)
            }
            
            _, reward_dict, _, _ = env.step(simulated_actions)
            rewards.append(reward_dict["trader_0"])
        return rewards

    config = GRPOConfig(
        output_dir="./vsr_grpo_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        num_generations=4,          # group size
        max_completion_length=256,
        logging_steps=10,
        save_steps=100,
        learning_rate=5e-5,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained("./vsr_trader_lora")

if __name__ == "__main__":
    run_training()
