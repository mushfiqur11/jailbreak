#!/usr/bin/env python3
"""
Generate all experiment configuration files for the large-scale experiment.
13 targets × 3 attackers = 39 experiment configurations
Judge: GPT-4o-mini (API)
"""

import os
import yaml

# Base paths (on server)
HF_TOKEN_PATH = "/home/mrahma45/HUGGINGFACE_KEY"
HF_MODEL_PATH = "/scratch/mrahma45/hf_models"
OPENAI_API_KEY_PATH = "/home/mrahma45/OPENAI_API_KEY"

# Define targets (13 models)
TARGETS = {
    "llama70b": {
        "model": "llama",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "quantization": "4bit",  # Quantized for 70B
        "short_name": "llama70b"
    },
    "llama1b": {
        "model": "llama",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "quantization": "none",
        "short_name": "llama1b"
    },
    "llama8b": {
        "model": "llama",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "quantization": "none",
        "short_name": "llama8b"
    },
    "phi4": {
        "model": "phi",
        "model_id": "microsoft/Phi-4-mini-instruct",
        "quantization": "none",
        "short_name": "phi4"
    },
    "qwen4b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "quantization": "none",
        "short_name": "qwen4b"
    },
    "qwen06b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-0.6B",
        "quantization": "none",
        "short_name": "qwen06b"
    },
    "qwen8b_target": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-8B",
        "quantization": "none",
        "short_name": "qwen8bt"
    },
    "phi35": {
        "model": "phi",
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "quantization": "none",
        "short_name": "phi35"
    },
    "gptoss20b": {
        "model": "gpt",  # Assuming HF model
        "model_id": "openai/gpt-oss-20b",
        "quantization": "4bit",  # Quantized
        "short_name": "gptoss20b"
    },
    "gemma27b": {
        "model": "gemma",
        "model_id": "google/gemma-3-27b-it",
        "quantization": "4bit",  # Quantized
        "short_name": "gemma27b"
    },
    "gemma4b": {
        "model": "gemma",
        "model_id": "google/gemma-3-4b-it",
        "quantization": "none",
        "short_name": "gemma4b"
    },
    "gemma1b": {
        "model": "gemma",
        "model_id": "google/gemma-3-1b-it",
        "quantization": "none",
        "short_name": "gemma1b"
    },
    "qwen32b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-32B",
        "quantization": "4bit",  # Quantized
        "short_name": "qwen32bt"
    },
}

# Define attackers (3 models)
ATTACKERS = {
    "qwen8b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-8B",
        "quantization": "auto",
        "short_name": "qwen8b"
    },
    "qwen06b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-0.6B",
        "quantization": "none",
        "short_name": "qwen06b"
    },
    "qwen32b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-32B",
        "quantization": "4bit",  # Quantized
        "short_name": "qwen32b"
    },
}

# Judge configuration (GPT-4o-mini via API)
JUDGE_CONFIG = {
    "model": "gpt",
    "model_id": "gpt-4o-mini",
    "api_key_path": OPENAI_API_KEY_PATH,
    "max_completion_tokens": 800,
}


def generate_config(attacker_key: str, target_key: str, output_dir: str) -> str:
    """Generate a single experiment configuration file."""
    attacker = ATTACKERS[attacker_key]
    target = TARGETS[target_key]
    
    # Create unique experiment name
    exp_name = f"exp_{attacker['short_name']}_{target['short_name']}"
    
    config = {
        # Header comment
        "_comment": f"Experiment: Attacker={attacker['model_id']} vs Target={target['model_id']}",
        
        # Data configuration
        "data": {
            "dataset_name": "JailbreakBench/JBB-Behaviors",
            "data_dir": "../data/JBB-Behaviors",
            "max_goals": 100,
        },
        
        # Agent model configurations
        "agents": {
            "attacker": {
                "model": attacker["model"],
                "model_id": attacker["model_id"],
                "quantization": attacker["quantization"],
                "temperature": 0.7,
                "max_new_tokens": 2000,
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.05,
                "do_sample": True,
                "hf_token_path": HF_TOKEN_PATH,
                "hf_model_path": HF_MODEL_PATH,
            },
            "reasoning": {
                "model": attacker["model"],
                "model_id": attacker["model_id"],
                "quantization": attacker["quantization"],
                "temperature": 0.3,
                "max_new_tokens": 1500,
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.05,
                "do_sample": True,
                "hf_token_path": HF_TOKEN_PATH,
                "hf_model_path": HF_MODEL_PATH,
            },
            "target": {
                "model": target["model"],
                "model_id": target["model_id"],
                "quantization": target["quantization"],
                "temperature": 0.7,
                "max_new_tokens": 1000,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "hf_token_path": HF_TOKEN_PATH,
                "hf_model_path": HF_MODEL_PATH,
            },
            "judge": JUDGE_CONFIG,
        },
        
        # Feature toggles
        "features": {
            "enable_gala_learning": False,
            "enable_tactic_wise_learning": False,
            "enable_prompt_wise_learning": False,
            "enable_knowledge_evolution": False,
            "enable_traceback": True,
            "traceback_min_turns": 3,
            "enable_tactic_generalization": False,
            "tactic_generalization_batch_size": 20,
            "enable_belief_state_llm_update": True,
            "enable_dynamic_selection": True,
        },
        
        # Pipeline configuration
        "pipeline": {
            "max_turns": 8,
            "max_trials": 2,
            "timeout": 600,
            "early_stopping": True,
        },
        
        # Output configuration
        "output": {
            "results_path": f"{exp_name}_results.json",
            "knowledge_path": f"{exp_name}_knowledge.json",
            "generalized_tactics_path": f"{exp_name}_tactics.json",
            "successful_jailbreaks_path": f"{exp_name}_successes.json",
            "save_intermediate": True,
            "verbose_logging": True,
        },
        
        # Logging
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        
        # Additional settings
        "settings": {
            "random_seed": 42,
            "retry_on_parse_failure": True,
            "max_parse_retries": 3,
        },
    }
    
    # Write config file
    config_path = os.path.join(output_dir, f"{exp_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return exp_name


def main():
    # Create experiments directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, "configs", "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Generate all experiment configurations
    experiment_names = []
    for attacker_key in ATTACKERS:
        for target_key in TARGETS:
            exp_name = generate_config(attacker_key, target_key, experiments_dir)
            experiment_names.append(exp_name)
            print(f"Generated: {exp_name}")
    
    print(f"\nTotal configurations generated: {len(experiment_names)}")
    
    # Write experiment list file
    list_path = os.path.join(experiments_dir, "experiment_list.txt")
    with open(list_path, 'w') as f:
        for name in experiment_names:
            f.write(f"{name}\n")
    print(f"Experiment list saved to: {list_path}")
    
    return experiment_names


if __name__ == "__main__":
    main()
