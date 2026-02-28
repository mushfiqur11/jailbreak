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


def _format_conversation(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns for tactic curation prompt."""
    parts = []
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


def _load_initial_tactics(repo_root: str) -> Dict[str, Any]:
    tactics_path = os.path.join(
        repo_root,
        "jailbreak",
        "agentic_module",
        "agents",
        "initial_tactics.json",
    )
    with open(tactics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to run1 config YAML")
    parser.add_argument("--output_path", type=str, required=True, help="Path for curated tactics JSON")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    reasoning_config = dict(cfg.agents.reasoning)
    reasoner = ReasoningAgent(reasoning_config)

    successes_path = cfg.output.successful_jailbreaks_path
    if not os.path.exists(successes_path):
        raise FileNotFoundError(f"Run1 successes file not found: {successes_path}")

    with open(successes_path, "r", encoding="utf-8") as f:
        successes = json.load(f)

    supplementary = {
        "newTacticPool": [],
        "selectionFramework": [],
        "promptNotes": [],
    }
    learning_note = "Initial GALA knowledge base"

    for rec in successes:
        goal = rec.get("goal", "")
        turns = rec.get("turns", [])
        conversation_str = _format_conversation(turns)

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

    curated = _load_initial_tactics(REPO_ROOT)

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
