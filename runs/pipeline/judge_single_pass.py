#!/usr/bin/env python3
"""
Single-pass judge runner:
- Loads new_data.json
- Iterates conversations one by one
- Sends each conversation to one LLM via llm_module (JudgeAgent)
- Saves raw predictions to output JSON

Usage example:
python scripts/judge_single_pass.py \
  --dataset_path /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/data/new_data.json \
  --output_path /scratch/mrahma45/jailbreaking_repos/outputs/judge_validation/gpt4o_mini_single_pass.json \
  --model gpt \
  --model_id gpt-4o-mini \
  --api_key_path /home/mrahma45/OPENAI_API_KEY \
  --max_retries 1
"""

import argparse
import json
import os
from typing import Any, Dict, List
from tqdm import tqdm
import sys

from omegaconf import OmegaConf

# Add jailbreak module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from jailbreak.agentic_module.agents.judge import JudgeAgent

# Add runs/pipeline to path for local imports
sys.path.append(os.path.dirname(__file__))


def build_conversation_string(turns: List[Dict[str, Any]]) -> str:
    """Format turns for judge input."""
    parts = []
    for i, t in enumerate(turns, 1):
        attacker = t.get("attackerMessage", "")
        target = t.get("targetMessage", "")
        parts.append(f"Turn {i}:\nAttacker: {attacker}\nTarget: {target}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Required IO
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # Model config (llm_module factory fields)
    parser.add_argument("--model", type=str, required=True, help="e.g., gpt, gptoss, qwen, llama")
    parser.add_argument("--model_id", type=str, required=True)

    # Optional auth / backend fields
    parser.add_argument("--api_key_path", type=str, default=None)
    parser.add_argument("--hf_token_path", type=str, default=None)
    parser.add_argument("--hf_model_path", type=str, default=None)

    # Optional generation fields
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_completion_tokens", type=int, default=64)
    parser.add_argument("--quantization", type=str, default=None)

    # Judge retry behavior (set to 1 for strict one-call-per-conversation)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--retry_delay", type=float, default=1.0)

    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    conversations = dataset.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("Expected dataset['conversations'] to be a list")

    # Build judge config for llm_module
    judge_config: Dict[str, Any] = {
        "model": args.model,
        "model_id": args.model_id,
        "do_sample": False,  # Deterministic output for evaluation
        "verbose": False,  # Disable verbose logging for cleaner output
        "use_cache": True,  # Enable caching to speed up repeated calls (if applicable)
    }

    if args.api_key_path:
        judge_config["api_key_path"] = args.api_key_path
    if args.hf_token_path:
        judge_config["hf_token_path"] = args.hf_token_path
    if args.hf_model_path:
        judge_config["hf_model_path"] = args.hf_model_path
    if args.temperature is not None:
        judge_config["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        judge_config["max_new_tokens"] = args.max_new_tokens
    if args.max_completion_tokens is not None:
        judge_config["max_completion_tokens"] = args.max_completion_tokens
    if args.quantization is not None:
        judge_config["quantization"] = args.quantization

    # Create judge agent (uses llm_module + judge policy prompt)
    judge = JudgeAgent(
        judge_config=judge_config,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    results: List[Dict[str, Any]] = []

    total = len(conversations)
    for idx, conv in enumerate(tqdm(conversations, total=total), 1):
        conv_id = conv.get("id", idx - 1)
        title = conv.get("title", "")
        turns = conv.get("turns", [])
        ground_truth = conv.get("groundTruth")

        if not isinstance(turns, list):
            turns = []

        conversation_str = build_conversation_string(turns)

        try:
            eval_result = judge.evaluate_conversation(
                conversation_str=conversation_str,
                goal=title if title else f"conversation_{conv_id}"
            )
            item = {
                "id": conv_id,
                "title": title,
                "prediction": {
                    "PV": eval_result["violation_intensity"],
                    "reasoning": eval_result["reasoning"],
                },
                "groundTruth": ground_truth,
                "status": "ok",
            }
        except Exception as e:
            item = {
                "id": conv_id,
                "title": title,
                "prediction": None,
                "groundTruth": ground_truth,
                "status": "error",
                "error": str(e),
            }

        results.append(item)

        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    output_obj = {
        "model_config": judge_config,
        "num_conversations": len(conversations),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)

    print(f"Saved output to: {args.output_path}")


if __name__ == "__main__":
    main()


