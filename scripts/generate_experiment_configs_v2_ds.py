#!/usr/bin/env python3
"""
Generate V2-REST experiment configuration files for the 3-stage workflow:

1) run1 with default tactics
2) tactic curation (separate script call)
3) run2 with curated tactics

This variant writes configs to:
  scripts/configs/experiments_v2_rest
"""

import os
import yaml


# Base paths (on server)
HF_TOKEN_PATH = "/home/mrahma45/HUGGINGFACE_KEY"
HF_MODEL_PATH = "/scratch/mrahma45/hf_models"
OPENAI_API_KEY_PATH = "/home/mrahma45/OPENAI_API_KEY"
OUTPUT_RESULTS_DIR = "/scratch/mrahma45/jailbreaking_repos/outputs/results"

# Knowledge folder for tactics (absolute path)
KNOWLEDGE_FOLDER = "/scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/jailbreak/agentic_module/agents/knowledge"
DEFAULT_TACTICS_FILE = f"{KNOWLEDGE_FOLDER}/default_tactics.json"


TARGETS = {
    "deepseekr1distillllama70b": {
        "model": "deepseek",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "quantization": "4bit",
        "short_name": "dsr1dll70b",
    },
    "deepseekr1": {
        "model": "deepseek",
        "model_id": "deepseek-ai/DeepSeek-R1",
        "quantization": "4bit",
        "short_name": "dsr1",
    },
    "deepseekr1distillllama8b": {
        "model": "deepseek",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "quantization": "none",
        "short_name": "dsr1dll8b",
    },
}


ATTACKERS = {
    "qwen8b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-8B",
        "quantization": "auto",
        "short_name": "qwen8b",
    },
    "qwen06b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-0.6B",
        "quantization": "none",
        "short_name": "qwen06b",
    },
    "qwen32b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-32B",
        "quantization": "4bit",
        "short_name": "qwen32b",
    },
}


JUDGE_CONFIG = {
    "model": "gpt",
    "model_id": "gpt-4o-mini",
    "api_key_path": OPENAI_API_KEY_PATH,
    "max_completion_tokens": 800,
}


def generate_config(attacker_key: str, target_key: str, output_dir: str) -> str:
    attacker = ATTACKERS[attacker_key]
    target = TARGETS[target_key]

    exp_name = f"exp_{attacker['short_name']}_{target['short_name']}"

    # Curated tactics path in knowledge folder (absolute path)
    curated_tactics_file = f"{KNOWLEDGE_FOLDER}/{exp_name}_curated_tactics.json"

    base_config = {
        "_comment": f"V2 REST Experiment: Attacker={attacker['model_id']} vs Target={target['model_id']}",
        "data": {
            "dataset_name": "JailbreakBench/JBB-Behaviors",
            "data_dir": "../data/JBB-Behaviors",
            "max_goals": 100,
        },
        "agents": {
            "attacker": {
                "model": attacker["model"],
                "model_id": attacker["model_id"],
                # run1 uses default tactics from knowledge folder
                "use_default_tactics": True,
                "tactics_file": DEFAULT_TACTICS_FILE,
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
        "features": {
            "enable_gala_learning": True,
            "enable_traceback": True,
            "enable_tactic_wise_learning": True,
            "enable_prompt_wise_learning": True,
            "enable_knowledge_evolution": True,
            "traceback_min_turns": 3,
            "enable_tactic_generalization": True,
            "tactic_generalization_batch_size": 20,
            "enable_belief_state_llm_update": True,
            "enable_dynamic_selection": True,
        },
        "pipeline": {
            "max_turns": 8,
            "max_trials": 3,
            "save_every_n_goals": 5,
            "timeout": 600,
            "early_stopping": True,
        },
        "output": {
            "results_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run1_results.json",
            "knowledge_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run1_knowledge.json",
            "generalized_tactics_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run1_tactics.json",
            "successful_jailbreaks_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run1_successes.json",
            "save_intermediate": True,
            "verbose_logging": True,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "settings": {
            "random_seed": 42,
            "retry_on_parse_failure": True,
            "max_parse_retries": 3,
        },
    }

    run1_config_path = os.path.join(output_dir, f"{exp_name}_run1.yaml")
    with open(run1_config_path, "w") as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)

    run2_config = yaml.safe_load(yaml.dump(base_config))
    # run2 uses curated tactics from knowledge folder (generated after run1)
    run2_config["agents"]["attacker"]["use_default_tactics"] = False
    run2_config["agents"]["attacker"]["tactics_file"] = curated_tactics_file
    run2_config["output"] = {
        "results_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run2_results.json",
        "knowledge_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run2_knowledge.json",
        "generalized_tactics_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run2_tactics.json",
        "successful_jailbreaks_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run2_successes.json",
        "save_intermediate": True,
        "verbose_logging": True,
    }

    run2_config_path = os.path.join(output_dir, f"{exp_name}_run2.yaml")
    with open(run2_config_path, "w") as f:
        yaml.dump(run2_config, f, default_flow_style=False, sort_keys=False)

    return exp_name


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, "configs", "experiments_v2_rest")
    os.makedirs(experiments_dir, exist_ok=True)

    experiment_names = []
    for attacker_key in ATTACKERS:
        for target_key in TARGETS:
            exp_name = generate_config(attacker_key, target_key, experiments_dir)
            experiment_names.append(exp_name)
            print(f"Generated: {exp_name}")

    print(f"\nTotal V2 REST configurations generated: {len(experiment_names)}")

    list_path = os.path.join(experiments_dir, "experiment_list.txt")
    with open(list_path, "w") as f:
        for name in experiment_names:
            f.write(f"{name}\n")
    print(f"Experiment list saved to: {list_path}")


if __name__ == "__main__":
    main()
