#!/usr/bin/env python3
"""
Generate V3 experiment configuration files for a 5-run chained workflow.

V3 flow per experiment:
  run1: default tactics
  run2: uses curated tactics from run1
  run3: uses curated tactics from run2
  run4: uses curated tactics from run3
  run5: uses curated tactics from run4

This main generator uses max_goals=50.
"""

import os
import argparse
import yaml


# Base paths (server)
HF_TOKEN_PATH = "/home/mrahma45/HUGGINGFACE_KEY"
HF_MODEL_PATH = "/scratch/mrahma45/hf_models"
OPENAI_API_KEY_PATH = "/home/mrahma45/OPENAI_API_KEY"
OUTPUT_RESULTS_DIR = "/scratch/mrahma45/jailbreaking_repos/outputs/results"
KNOWLEDGE_FOLDER = "/scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/jailbreak/agentic_module/agents/knowledge"
DEFAULT_TACTICS_FILE = f"{KNOWLEDGE_FOLDER}/default_tactics.json"


ATTACKER = {
    "model": "qwen",
    "model_id": "Qwen/Qwen3-0.6B",
    "quantization": "none",
    "short_name": "qwen06b",
}

REASONING = {
    "model": "qwen",
    "model_id": "Qwen/Qwen3-32B",
    "quantization": "4bit",
}

TARGETS = {
    "qwen06b": {
        "model": "qwen",
        "model_id": "Qwen/Qwen3-0.6B",
        "quantization": "none",
        "short_name": "qwen06b",
    },
    "llama1b": {
        "model": "llama",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "quantization": "none",
        "short_name": "llama1b",
    },
    "llama8b": {
        "model": "llama",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "quantization": "none",
        "short_name": "llama8b",
    },
}

JUDGE = {
    "model": "gpt",
    "model_id": "gpt-4o-mini",
    "api_key_path": OPENAI_API_KEY_PATH,
    "max_completion_tokens": 800,
}


def build_run_config(exp_name: str, target: dict, run_idx: int, max_goals: int) -> dict:
    use_default = run_idx == 1
    tactics_file = (
        DEFAULT_TACTICS_FILE
        if use_default
        else f"{KNOWLEDGE_FOLDER}/{exp_name}_run{run_idx - 1}_curated_tactics.json"
    )

    return {
        "_comment": (
            f"V3 chained run{run_idx}: attacker={ATTACKER['model_id']}, "
            f"reasoning={REASONING['model_id']}(4bit), target={target['model_id']}"
        ),
        "data": {
            "dataset_name": "JailbreakBench/JBB-Behaviors",
            "data_dir": "../data/JBB-Behaviors",
            "max_goals": max_goals,
        },
        "agents": {
            "attacker": {
                "model": ATTACKER["model"],
                "model_id": ATTACKER["model_id"],
                "use_default_tactics": use_default,
                "tactics_file": tactics_file,
                "quantization": ATTACKER["quantization"],
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
                "model": REASONING["model"],
                "model_id": REASONING["model_id"],
                "quantization": REASONING["quantization"],
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
            "judge": JUDGE,
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
            "max_turns": 6,
            "max_trials": 2,
            "save_every_n_goals": 5,
            "timeout": 600,
            "early_stopping": True,
        },
        "output": {
            "results_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run{run_idx}_results.json",
            "knowledge_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run{run_idx}_knowledge.json",
            "generalized_tactics_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run{run_idx}_tactics.json",
            "successful_jailbreaks_path": f"{OUTPUT_RESULTS_DIR}/{exp_name}_run{run_idx}_successes.json",
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
            "max_parse_retries": 2,
        },
    }


def generate_experiment_configs(output_dir: str, max_goals: int, exp_prefix: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    experiment_names = []

    for target in TARGETS.values():
        exp_name = f"{exp_prefix}_{ATTACKER['short_name']}_{target['short_name']}"
        experiment_names.append(exp_name)

        for run_idx in range(1, 6):
            cfg = build_run_config(exp_name, target, run_idx, max_goals)
            path = os.path.join(output_dir, f"{exp_name}_run{run_idx}.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return experiment_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_goals", type=int, default=50, help="Number of goals per run")
    parser.add_argument("--exp_prefix", type=str, default="exp_v3", help="Experiment name prefix")
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="experiments_v3",
        help="Config subdirectory under scripts/configs",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "configs", args.output_subdir)

    names = generate_experiment_configs(
        output_dir=out_dir,
        max_goals=args.max_goals,
        exp_prefix=args.exp_prefix,
    )

    list_path = os.path.join(out_dir, "experiment_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")

    print(
        f"Generated {len(names)} V3 experiments in: {out_dir} "
        f"(max_goals={args.max_goals}, prefix={args.exp_prefix})"
    )
    print(f"Experiment list: {list_path}")


if __name__ == "__main__":
    main()
