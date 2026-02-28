#!/usr/bin/env python3
"""
Summarize experiment result files into a table.

Computed columns:
  A. target
  B. attacker
  C. number_of_attacks
  D. number_of_successes
  E. success_rate (successes / attacks)
  F. avg_turn_of_success

Usage:
  python scripts/summarize_experiment_results.py
  python scripts/summarize_experiment_results.py \
      --results_dir outputs/results \
      --output_csv outputs/results/experiment_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


ATTACKER_NAME_MAP: Dict[str, str] = {
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen06b": "Qwen/Qwen3-0.6B",
    "qwen32b": "Qwen/Qwen3-32B (quantized)",
}


TARGET_NAME_MAP: Dict[str, str] = {
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct (quantized)",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi4": "microsoft/Phi-4-mini-instruct",
    "qwen4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen06b": "Qwen/Qwen3-0.6B",
    "qwen8bt": "Qwen/Qwen3-8B",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "gptoss20b": "openai/gpt-oss-20b (quantized)",
    "gemma27b": "google/gemma-3-27b-it (quantized)",
    "gemma4b": "google/gemma-3-4b-it",
    "gemma1b": "google/gemma-3-1b-it",
    "qwen32bt": "Qwen/Qwen3-32B (quantized)",
}


RESULT_FILE_RE = re.compile(r"^exp_(?P<attacker>[^_]+)_(?P<target>[^_]+)_results\.json$")


def _extract_success_turn(goal_data: dict) -> Optional[int]:
    """Return the first turn index where PV=2 for a successful goal, else None."""
    trials = goal_data.get("trials", [])
    for trial in trials:
        if not trial.get("success", False):
            continue
        for turn in trial.get("turns", []):
            judge_eval = turn.get("judge_evaluation", {})
            if judge_eval.get("violation_intensity") == 2:
                return int(turn.get("turn", 0))
    return None


def summarize_file(result_file: Path) -> Optional[dict]:
    match = RESULT_FILE_RE.match(result_file.name)
    if not match:
        return None

    attacker_key = match.group("attacker")
    target_key = match.group("target")

    with result_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    goals = list(data.values())
    num_attacks = len(goals)
    successful_goals = [g for g in goals if g.get("success", False)]
    num_successes = len(successful_goals)

    success_turns: List[int] = []
    for goal in successful_goals:
        t = _extract_success_turn(goal)
        if t is not None and t > 0:
            success_turns.append(t)

    avg_turn = (sum(success_turns) / len(success_turns)) if success_turns else None

    success_rate = (num_successes / num_attacks) if num_attacks > 0 else 0.0

    return {
        "target": TARGET_NAME_MAP.get(target_key, target_key),
        "attacker": ATTACKER_NAME_MAP.get(attacker_key, attacker_key),
        "number_of_attacks": num_attacks,
        "number_of_successes": num_successes,
        "success_rate": success_rate,
        "avg_turn_of_success": avg_turn,
        "result_file": result_file.name,
    }


def write_csv(rows: List[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "attacker",
        "number_of_attacks",
        "number_of_successes",
        "success_rate",
        "avg_turn_of_success",
        "result_file",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown_table(rows: List[dict]) -> None:
    print("| target | attacker | number_of_attacks | number_of_successes | success_rate | avg_turn_of_success |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        avg_turn_str = f"{r['avg_turn_of_success']:.2f}" if r["avg_turn_of_success"] is not None else "N/A"
        print(
            f"| {r['target']} | {r['attacker']} | {r['number_of_attacks']} | "
            f"{r['number_of_successes']} | {r['success_rate']:.4f} | {avg_turn_str} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize jailbreak experiment result files.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory containing experiment result JSON files",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/results/experiment_summary.csv"),
        help="Where to write summary CSV",
    )
    args = parser.parse_args()

    result_files = sorted(args.results_dir.glob("exp_*_results.json"))
    rows: List[dict] = []
    for result_file in result_files:
        row = summarize_file(result_file)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda x: (x["attacker"], x["target"]))
    write_csv(rows, args.output_csv)
    print_markdown_table(rows)
    print(f"\nWrote {len(rows)} rows to: {args.output_csv}")


if __name__ == "__main__":
    main()
