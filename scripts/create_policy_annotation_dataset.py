#!/usr/bin/env python3
"""
Create a policy-annotation dataset from jailbreak run result files.

For each run (*_results.json), randomly samples N goals and converts each sampled
goal into one PolicyConversation-compatible record for the
LongHorizonMemory/patient_evaluation_portal format.

Output JSON shape:
{
  "metadata": {...},
  "conversations": [
    {
      "id": "...",
      "title": "...",
      "createdAt": "ISO timestamp",
      "turns": [
        {
          "turnId": 1,
          "targetMessage": "...",
          "attackerMessage": "...",
          "judgement": {"violation": false, "severity": 0, "reason": "no violation"}
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_JUDGEMENT = {
    "violation": False,
    "severity": 0,
    "reason": "no violation",
}


def parse_run_name(path: Path) -> str:
    # exp_qwen8b_llama1b_results.json -> exp_qwen8b_llama1b
    name = path.name
    if name.endswith("_results.json"):
        return name[: -len("_results.json")]
    return path.stem


def choose_trial(goal_data: Dict) -> Optional[Dict]:
    """Pick trial for annotation: successful trial if available, else first trial."""
    trials = goal_data.get("trials", [])
    if not trials:
        return None
    for trial in trials:
        if trial.get("success", False):
            return trial
    return trials[0]


def goal_to_policy_conversation(
    run_name: str,
    goal_id: str,
    goal_data: Dict,
    trial: Dict,
) -> Dict:
    turns: List[Dict] = []
    for t in trial.get("turns", []):
        turns.append(
            {
                "turnId": int(t.get("turn", len(turns) + 1)),
                "targetMessage": t.get("target_response", ""),
                "attackerMessage": t.get("attacker_prompt", ""),
                "judgement": dict(DEFAULT_JUDGEMENT),
            }
        )

    goal_text = goal_data.get("goal", "")
    title_goal = goal_text[:120] + ("..." if len(goal_text) > 120 else "")
    title = f"{run_name} | {goal_id} | {title_goal}"

    return {
        "id": f"{run_name}__{goal_id}",
        "title": title,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "turns": turns,
    }


def sample_goals_from_run(
    run_file: Path,
    samples_per_run: int,
    rng: random.Random,
) -> Tuple[List[Dict], int]:
    with run_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return [], 0

    run_name = parse_run_name(run_file)
    goal_items = list(data.items())
    total_goals = len(goal_items)
    k = min(samples_per_run, total_goals)

    sampled_items = rng.sample(goal_items, k) if k > 0 else []
    conversations: List[Dict] = []

    for goal_id, goal_data in sampled_items:
        trial = choose_trial(goal_data)
        if trial is None:
            continue
        conv = goal_to_policy_conversation(run_name, goal_id, goal_data, trial)
        if conv["turns"]:
            conversations.append(conv)

    return conversations, total_goals


def main() -> None:
    parser = argparse.ArgumentParser(description="Create policy annotation dataset from result runs")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory containing exp_*_results.json files",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("outputs/results/policy_annotation_dataset.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--samples_per_run",
        type=int,
        default=15,
        help="Random samples per run file",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=20,
        help="Max number of run files to include (use <=0 for all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    run_files = sorted(
        p for p in args.results_dir.glob("exp_*_results.json") if p.is_file()
    )

    if args.max_runs and args.max_runs > 0:
        run_files = run_files[: args.max_runs]

    all_conversations: List[Dict] = []
    run_summaries: List[Dict] = []

    for run_file in run_files:
        sampled_convs, total_goals = sample_goals_from_run(
            run_file=run_file,
            samples_per_run=args.samples_per_run,
            rng=rng,
        )
        all_conversations.extend(sampled_convs)
        run_summaries.append(
            {
                "run": parse_run_name(run_file),
                "total_goals": total_goals,
                "sampled": len(sampled_convs),
            }
        )

    output = {
        "metadata": {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "resultsDir": str(args.results_dir),
            "numRunsProcessed": len(run_files),
            "samplesPerRun": args.samples_per_run,
            "seed": args.seed,
            "totalConversations": len(all_conversations),
            "runSummaries": run_summaries,
            "format": "PolicyConversation[] compatible with patient_evaluation_portal",
        },
        "conversations": all_conversations,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Runs processed: {len(run_files)}")
    print(f"Samples per run: {args.samples_per_run}")
    print(f"Total conversations: {len(all_conversations)}")
    print(f"Output: {args.output_json}")


if __name__ == "__main__":
    main()
