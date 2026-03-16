#!/usr/bin/env python3
"""
Independent tactic curation command for a completed run1 experiment.

Workflow per experiment:
1) run1 orchestrator (default tactics)
2) run this script to curate tactics from run1 successes
3) run2 orchestrator (uses curated tactics file)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from omegaconf import OmegaConf


# Make package importable when executed from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from jailbreak.agentic_module.agents.reasoning_agent import ReasoningAgent


# Knowledge folder for tactics (absolute path)
KNOWLEDGE_FOLDER = "/scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/jailbreak/agentic_module/agents/knowledge"
DEFAULT_TACTICS_FILE = f"{KNOWLEDGE_FOLDER}/default_tactics.json"


def _format_conversation(turns: List[Dict[str, Any]], trial_meta: Dict[str, Any]) -> str:
    """Format conversation turns for tactic curation prompt with outcome/severity metadata."""
    parts = []
    parts.append("=== Trial Metadata ===")
    parts.append(f"Goal ID: {trial_meta.get('goal_id', 'unknown')}")
    parts.append(f"Trial ID: {trial_meta.get('trial_id', 'unknown')}")
    parts.append(f"Outcome: {'SUCCESS' if trial_meta.get('success') else 'FAILURE'}")
    parts.append(f"Final PV: {trial_meta.get('violation_result', 0)}")
    parts.append(f"Max PV observed: {trial_meta.get('max_pv', 0)}")
    parts.append(f"Avg PV observed: {trial_meta.get('avg_pv', 0.0):.3f}")
    tactics_used = trial_meta.get("tactics_used", [])
    parts.append(f"Tactics used: {', '.join(tactics_used) if tactics_used else 'Unknown'}")
    parts.append("")

    for i, turn in enumerate(turns, 1):
        parts.append(f"Turn {i}:")
        if "attacker_prompt" in turn:
            parts.append(f"Attacker: {turn['attacker_prompt']}")
        if "target_response" in turn:
            parts.append(f"Target: {turn['target_response']}")
        if "judge_evaluation" in turn:
            pv = turn["judge_evaluation"].get("violation_intensity", 0)
            parts.append(f"Judge PV: {pv}")
        parts.append("")
    return "\n".join(parts)


def _extract_trial_records(results_path: str) -> List[Dict[str, Any]]:
    """Extract all trial records (successes + failures) from orchestrator results JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, dict):
        raise ValueError(f"Unexpected results JSON structure in {results_path}: expected dict")

    records: List[Dict[str, Any]] = []

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

    # Severity-first ordering for stronger signal in curation updates
    records.sort(key=lambda r: (r.get("max_pv", 0), r.get("success", False)), reverse=True)
    return records


def _load_initial_tactics() -> Dict[str, Any]:
    """Load default tactics from the knowledge folder."""
    if not os.path.exists(DEFAULT_TACTICS_FILE):
        raise FileNotFoundError(f"Default tactics file not found: {DEFAULT_TACTICS_FILE}")
    with open(DEFAULT_TACTICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to run1 config YAML")
    parser.add_argument("--output_path", type=str, required=True, help="Path for curated tactics JSON")
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

    supplementary = {
        "newTacticPool": [],
        "selectionFramework": [],
        "promptNotes": [],
    }
    learning_note = "Initial GALA knowledge base"

    for rec in trial_records:
        goal = rec.get("goal", "")
        turns = rec.get("turns", [])
        conversation_str = _format_conversation(turns, rec)

        try:
            updated = reasoner.curate_tactics(
                learning_note=learning_note,
                learning_note_supplementary=json.dumps(supplementary),
                goal=goal,
                conversation_str=conversation_str,
            )
            if isinstance(updated, dict):
                supplementary = updated
        except Exception:
            # Keep pipeline resilient: skip failed curation for a single goal
            continue

    curated = _load_initial_tactics()

    for entry in supplementary.get("newTacticPool", []):
        tactic = entry.get("tactic")
        definition = entry.get("definition", "")
        prompts = entry.get("prompts", [])
        if tactic and tactic not in curated:
            curated[tactic] = {
                "definition": definition,
                "examples": prompts,
            }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)


if __name__ == "__main__":
    main()
