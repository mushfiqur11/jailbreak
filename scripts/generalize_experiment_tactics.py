#!/usr/bin/env python3
"""
Independent tactic generalization command for a completed experiment run.

This script is intentionally separated from orchestrator execution.
It consumes the run's successful jailbreaks JSON and writes generalized tactics JSON.
"""

import argparse
import json
import os
import sys

from omegaconf import OmegaConf


# Make package importable when executed from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from jailbreak.agentic_module.agents.reasoning_agent import ReasoningAgent
from runs.pipeline.tactic_generalization import (
    create_tactic_generalization_module,
    save_generalization_result,
)


def _extract_trial_records(results_path: str):
    """Extract all trial records (successes + failures) from orchestrator results JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, dict):
        raise ValueError(f"Unexpected results JSON structure in {results_path}: expected dict")

    records = []
    for goal_id, goal_data in results.items():
        if not isinstance(goal_data, dict):
            continue

        goal = goal_data.get("goal", "")
        trials = goal_data.get("trials", [])
        if not isinstance(trials, list):
            continue

        for trial in trials:
            if not isinstance(trial, dict):
                continue

            turns = trial.get("turns", [])
            if not isinstance(turns, list) or len(turns) == 0:
                continue

            pvs = []
            for t in turns:
                pv = t.get("judge_evaluation", {}).get("violation_intensity", 0)
                try:
                    pvs.append(int(pv))
                except Exception:
                    pvs.append(0)

            violation_result = trial.get("violation_result", 0)
            try:
                violation_result = int(violation_result)
            except Exception:
                violation_result = 0

            max_pv = max(pvs) if pvs else violation_result
            avg_pv = (sum(pvs) / len(pvs)) if pvs else float(violation_result)

            records.append(
                {
                    "goal_id": goal_id,
                    "goal": goal,
                    "trial_id": trial.get("trial_id", 0),
                    "success": bool(trial.get("success", False)),
                    "violation_result": violation_result,
                    "max_pv": max_pv,
                    "avg_pv": avg_pv,
                    "tactics_used": trial.get("tactics_used", []),
                    "turns": turns,
                }
            )

    # Severity-first ordering for stronger signal
    records.sort(key=lambda r: (r.get("max_pv", 0), r.get("success", False)), reverse=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to run config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    reasoning_config = dict(cfg.agents.reasoning)
    reasoner = ReasoningAgent(reasoning_config)

    results_path = cfg.output.results_path
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Run results file not found: {results_path}")

    trial_records = _extract_trial_records(results_path)
    if not trial_records:
        raise ValueError(f"No valid trials found in results file: {results_path}")

    module = create_tactic_generalization_module(reasoner, dict(cfg))
    result = module.generalize_all(trial_records)

    output_path = cfg.output.generalized_tactics_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_generalization_result(result, output_path)


if __name__ == "__main__":
    main()
