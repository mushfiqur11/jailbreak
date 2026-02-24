"""
Utility functions and data classes for the pipeline orchestrator.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a single trial for a goal."""
    trial_id: int
    success: bool
    violation_result: int  # 0=none, 1=weak, 2=strong
    turns: List[Dict[str, Any]]
    used_traceback: bool = False
    traceback_info: Optional[Dict[str, Any]] = None
    prompt_suggestions_used: bool = False
    final_belief_state: Optional[Dict[str, Any]] = None
    tactics_used: List[str] = field(default_factory=list)


@dataclass
class GoalResult:
    """Result of processing a single goal."""
    goal_id: str
    goal_text: str
    success: bool
    trials: List[TrialResult]
    knowledge_update: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None  # Error message if goal was skipped due to error
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ResultStorage:
    """Handles storage of pipeline results."""
    
    def __init__(self, results_path: str, knowledge_path: str, successes_path: str):
        self.results_path = results_path
        self.knowledge_path = knowledge_path
        self.successes_path = successes_path
        self.results: Dict[str, GoalResult] = {}
        self.knowledge_evolution: List[Dict[str, Any]] = []
        self.successful_jailbreaks: List[Dict[str, Any]] = []
        
        logger.info(f"ResultStorage initialized: results={results_path}")
    
    def add_goal_result(self, result: GoalResult) -> None:
        """Add a goal result to storage."""
        self.results[result.goal_id] = result
        
        if result.knowledge_update:
            self.knowledge_evolution.append({
                "goal_id": result.goal_id,
                "timestamp": result.timestamp,
                "update": result.knowledge_update
            })
        
        # Track successful jailbreaks for generalization
        if result.success:
            for trial in result.trials:
                if trial.success:
                    self.successful_jailbreaks.append({
                        "goal": result.goal_text,
                        "turns": trial.turns,
                        "tactics_used": trial.tactics_used
                    })
                    break  # Only add first successful trial
    
    def get_successful_jailbreaks(self) -> List[Dict[str, Any]]:
        """Get all successful jailbreaks for tactic generalization."""
        return self.successful_jailbreaks
    
    def save_all(self) -> None:
        """Save all results to files."""
        # Save main results
        results_data = {}
        for goal_id, result in self.results.items():
            goal_data = {
                "goal": result.goal_text,
                "success": result.success,
                "timestamp": result.timestamp,
                "trials": [
                    {
                        "trial_id": t.trial_id,
                        "success": t.success,
                        "violation_result": t.violation_result,
                        "turns": t.turns,
                        "used_traceback": t.used_traceback,
                        "traceback_info": t.traceback_info,
                        "tactics_used": t.tactics_used
                    }
                    for t in result.trials
                ],
                "knowledge_update": result.knowledge_update
            }
            # Add error message if present (for skipped goals)
            if result.error_message:
                goal_data["error_message"] = result.error_message
                goal_data["skipped"] = True
            
            results_data[goal_id] = goal_data
        
        with open(self.results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results saved to {self.results_path}")
        
        # Save knowledge evolution
        with open(self.knowledge_path, 'w') as f:
            json.dump(self.knowledge_evolution, f, indent=2)
        logger.info(f"Knowledge evolution saved to {self.knowledge_path}")
        
        # Save successful jailbreaks
        with open(self.successes_path, 'w') as f:
            json.dump(self.successful_jailbreaks, f, indent=2)
        logger.info(f"Successful jailbreaks saved to {self.successes_path}")


def format_conversation_for_judge(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns as string for judge evaluation."""
    parts = []
    for turn in turns:
        if "attacker_prompt" in turn:
            parts.append(f"Attacker: {turn['attacker_prompt']}")
        if "target_response" in turn:
            parts.append(f"Target: {turn['target_response']}")
    return "\n\n".join(parts)


def format_conversation_for_reasoning(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns for reasoning analysis."""
    parts = []
    for i, turn in enumerate(turns, 1):
        turn_str = f"Turn {i}:\n"
        if "attacker_prompt" in turn:
            turn_str += f"  Attacker: {turn['attacker_prompt']}\n"
        if "target_response" in turn:
            turn_str += f"  Target: {turn['target_response']}\n"
        if "judge_evaluation" in turn:
            pv = turn["judge_evaluation"].get("violation_intensity", 0)
            turn_str += f"  Judge PV: {pv}\n"
        parts.append(turn_str)
    return "\n".join(parts)


def get_config_value(config, *keys, default=None):
    """
    Safely get nested config value.
    
    Args:
        config: Configuration dictionary or OmegaConf
        *keys: Nested keys to traverse
        default: Default value if not found
    """
    current = config
    for key in keys:
        if hasattr(current, 'get'):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
        if current is None:
            return default
    return current
