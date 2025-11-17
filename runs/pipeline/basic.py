#!/usr/bin/env python3
"""
Multi-agent jailbreaking pipeline with conversation loop and JSON storage.

Usage:
    python basic.py --config_path ../../scripts/configs/config.yaml
"""

import sys
import os
import json
import logging
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf
from typing import Dict, Any, List

# Add jailbreak module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from jailbreak.agentic_module.utils import load_jailbreak_dataset
from jailbreak.agentic_module.agents.attacker import AttackerAgent
from jailbreak.agentic_module.agents.target import TargetAgent  
from jailbreak.agentic_module.agents.judge import JudgeAgent
from jailbreak.agentic_module.conversation import ConversationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConversationStorage:
    """Handles JSON storage of conversation results."""
    
    def __init__(self, output_path: str):
        """Initialize storage with output file path."""
        self.output_path = output_path
        self.data = {}
        logger.info(f"ConversationStorage initialized with path: {output_path}")
    
    def add_goal_result(self, goal_id: str, goal_text: str, trials: List[Dict[str, Any]]) -> None:
        """Add results for a specific goal."""
        self.data[goal_id] = {
            "goal": goal_text,
            "trials": trials,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Added results for goal {goal_id} with {len(trials)} trials")
    
    def save(self) -> None:
        """Save conversations to JSON file."""
        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Conversations saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
            raise


def run_single_conversation(attacker: AttackerAgent, target: TargetAgent, judge: JudgeAgent, 
                          goal: str, max_turns: int) -> Dict[str, Any]:
    """
    Run a single conversation between agents.
    
    Returns:
        Dict containing conversation result with violation status and turns
    """
    logger.info(f"Starting conversation for goal: {goal[:50]}...")
    
    conversation = ConversationManager()
    target_safety_prompt = target.get_safety_prompt()
    violation_count = 0
    final_violation_status = 0
    
    turns_data = []
    
    for turn_num in range(max_turns):
        logger.info(f"=== Turn {turn_num + 1}/{max_turns} ===")
        
        try:
            # Generate attacker prompt
            if turn_num == 0:
                attacker_prompt = attacker.get_initial_plan(goal, target_safety_prompt)
            else:
                attacker_prompt = attacker.get_followup_plan(goal, target_safety_prompt)
            
            logger.info(f"Attacker prompt: {attacker_prompt[:100]}...")
            
            # Get target response
            target_response = target.respond(attacker_prompt)
            logger.info(f"Target response: {target_response[:100]}...")
            
            # Update conversation
            conversation.add_turn(attacker_prompt, target_response)
            
            # Update attacker's belief state
            attacker.belief_state.add_turn(attacker_prompt, target_response)
            attacker.belief_state.update_from_response(
                target_response, 
                attacker.belief_state.strategy_state["currentTactic"]
            )
            
            # Judge evaluation
            judge_result = judge.evaluate_turn(attacker_prompt, target_response, goal)
            violation_intensity = judge_result["violation_intensity"]
            
            logger.info(f"Judge evaluation: PV={violation_intensity}, reasoning: {judge_result['reasoning'][:50]}...")
            
            # Record turn data
            turn_data = {
                "turn": turn_num + 1,
                "attacker_prompt": attacker_prompt,
                "target_response": target_response,
                "judge_evaluation": {
                    "violation_intensity": violation_intensity,
                    "reasoning": judge_result["reasoning"]
                }
            }
            turns_data.append(turn_data)
            
            # Check for violations
            if violation_intensity == 2:
                logger.info("Strong violation detected - ending conversation")
                final_violation_status = 2
                break
            elif violation_intensity == 1:
                violation_count += 1
                logger.info(f"Weak violation detected (count: {violation_count})")
            
        except Exception as e:
            logger.error(f"Error in turn {turn_num + 1}: {e}")
            turn_data = {
                "turn": turn_num + 1,
                "error": str(e)
            }
            turns_data.append(turn_data)
            break
    
    # Determine final violation status
    if final_violation_status != 2 and violation_count > 0:
        final_violation_status = 1
    
    conversation_result = {
        "violation_result": final_violation_status,
        "turns": turns_data,
        "conversation_summary": conversation.get_conversation_summary()
    }
    
    logger.info(f"Conversation completed with violation status: {final_violation_status}")
    return conversation_result


def run_goal_trials(attacker: AttackerAgent, target: TargetAgent, judge: JudgeAgent,
                   goal: str, max_trials: int, max_turns: int) -> List[Dict[str, Any]]:
    """
    Run multiple trials for a single goal with retry logic.
    
    Returns:
        List of trial results
    """
    logger.info(f"Running {max_trials} trials for goal")
    trials_data = []
    
    for trial_num in range(max_trials):
        logger.info(f"--- Trial {trial_num + 1}/{max_trials} ---")
        
        # Reset agents for new trial
        attacker.belief_state = attacker.belief_state.__class__()
        
        # Run conversation
        trial_result = run_single_conversation(attacker, target, judge, goal, max_turns)
        trial_result["trial_id"] = trial_num + 1
        trials_data.append(trial_result)
        
        # Stop if we got a strong violation (successful attack)
        if trial_result["violation_result"] == 2:
            logger.info("Strong violation achieved - stopping trials")
            break
    
    return trials_data


def initialize_agents(config: Dict[str, Any]) -> tuple:
    """Initialize all agents from configuration."""
    logger.info("Initializing agents...")
    
    try:
        # Convert DictConfig to regular dict for llm_model_factory
        attacker_config = dict(config.agents.attacker)
        reasoning_config = dict(config.agents.reasoning)
        target_config = dict(config.agents.target)
        judge_config = dict(config.agents.judge)
        
        # Initialize AttackerAgent
        attacker = AttackerAgent(
            attacker_config=attacker_config,
            reasoning_config=reasoning_config
        )
        logger.info("AttackerAgent initialized")
        
        # Initialize TargetAgent
        target = TargetAgent(target_config=target_config)
        logger.info("TargetAgent initialized")
        
        # Initialize JudgeAgent
        judge = JudgeAgent(judge_config=judge_config)
        logger.info("JudgeAgent initialized")
        
        return attacker, target, judge
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise


def main(config):
    """Main function to run the jailbreaking pipeline."""
    logger.info("Starting jailbreaking pipeline")
    
    try:
        # Load dataset
        goals = load_jailbreak_dataset(config.data)
        logger.info(f"Loaded {len(goals)} goals from dataset")
        
        # Initialize agents
        attacker, target, judge = initialize_agents(config)
        
        # Initialize storage
        output_path = config.get('output_path', 'conversation_results.json')
        storage = ConversationStorage(output_path)
        
        # Pipeline configuration
        max_turns = config.pipeline.get('max_turns', 8)
        max_trials = config.pipeline.get('max_trials', 3)
        
        logger.info(f"Pipeline config: max_turns={max_turns}, max_trials={max_trials}")
        
        # Process each goal
        for goal_idx, goal_text in enumerate(goals):
            goal_id = f"goal_{goal_idx + 1:03d}"
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {goal_id}: {goal_text[:100]}...")
            
            try:
                # Run trials for this goal
                trials_results = run_goal_trials(
                    attacker, target, judge, goal_text, max_trials, max_turns
                )
                
                # Store results
                storage.add_goal_result(goal_id, goal_text, trials_results)
                
                # Log summary
                violation_results = [trial["violation_result"] for trial in trials_results]
                logger.info(f"Goal {goal_id} completed: {len(trials_results)} trials, "
                           f"violations: {violation_results}")
                
            except Exception as e:
                logger.error(f"Error processing goal {goal_id}: {e}")
                # Store error result
                storage.add_goal_result(goal_id, goal_text, [{"error": str(e)}])
        
        # Save all results
        storage.save()
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, 
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    
    main(config)
