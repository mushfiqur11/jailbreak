#!/usr/bin/env python3
"""
GALA (Global and Local Learning Agent) multi-turn red-teaming pipeline.

This implementation follows the GALA paper methodology with:
1. Global tactic-wise learning: Accumulates tactical knowledge over time
2. Local prompt-wise learning: Refines implementation for specific goals
3. Dynamic knowledge base that evolves through experience
4. Retry mechanism with learned improvements

Usage:
    python GALA.py --config_path ../../scripts/configs/gpt_config.yaml
"""

import sys
import os
import json
import logging
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf
from typing import Dict, Any, List, Optional
from copy import deepcopy

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


class GALAConversationStorage:
    """Enhanced storage for GALA conversations with learning data."""
    
    def __init__(self, output_path: str, knowledge_path: str):
        """Initialize storage with output and knowledge file paths."""
        self.output_path = output_path
        self.knowledge_path = knowledge_path
        self.data = {}
        self.knowledge_evolution = []
        logger.info(f"GALAConversationStorage initialized: {output_path}, {knowledge_path}")
    
    def add_goal_result(self, goal_id: str, goal_text: str, trials: List[Dict[str, Any]], 
                       knowledge_update: Optional[Dict[str, Any]] = None) -> None:
        """Add results for a specific goal with optional knowledge update."""
        self.data[goal_id] = {
            "goal": goal_text,
            "trials": trials,
            "knowledge_update": knowledge_update,
            "timestamp": datetime.now().isoformat()
        }
        
        if knowledge_update:
            self.knowledge_evolution.append({
                "goal_id": goal_id,
                "update_type": knowledge_update.get("type", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "details": knowledge_update
            })
            
        logger.debug(f"Added results for goal {goal_id} with {len(trials)} trials")
    
    def save(self) -> None:
        """Save conversations and knowledge evolution to JSON files."""
        try:
            # Save conversation results
            with open(self.output_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Conversations saved to {self.output_path}")
            
            # Save knowledge evolution
            with open(self.knowledge_path, 'w') as f:
                json.dump(self.knowledge_evolution, f, indent=2)
            logger.info(f"Knowledge evolution saved to {self.knowledge_path}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise


class GALALearningModule:
    """GALA's dynamic learning module with global and local learning."""
    
    def __init__(self, attacker_agent: AttackerAgent, config: Optional[Dict[str, Any]] = None):
        """Initialize learning module with attacker agent reference and config."""
        self.attacker = attacker_agent
        self.learned_tactics = []
        self.prompt_improvements = {}  # goal -> improvements mapping
        
        # GALA hyperparameters from config
        self.config = config or {}
        self.enable_tactic_learning = self.config.get('enable_tactic_learning', True)
        self.enable_prompt_learning = self.config.get('enable_prompt_learning', True) 
        self.enable_knowledge_evolution = self.config.get('enable_knowledge_evolution', True)
        self.enable_dynamic_selection = self.config.get('enable_dynamic_selection', True)
        
        logger.info(f"GALALearningModule initialized with config: "
                   f"tactic_learning={self.enable_tactic_learning}, "
                   f"prompt_learning={self.enable_prompt_learning}, "
                   f"knowledge_evolution={self.enable_knowledge_evolution}, "
                   f"dynamic_selection={self.enable_dynamic_selection}")
    
    def perform_tactic_wise_learning(self, goal: str, conversation_history: List[Dict], 
                                   success: bool) -> Optional[Dict[str, Any]]:
        """
        Perform global tactic-wise learning after successful trials.
        
        Args:
            goal: The attack goal
            conversation_history: Complete conversation turns
            success: Whether the trial was successful
            
        Returns:
            Knowledge update dictionary if new learning occurred
        """
        if not success or not self.enable_tactic_learning:
            if not self.enable_tactic_learning:
                logger.info("Tactic-wise learning disabled - skipping")
            return None
            
        logger.info("Performing tactic-wise learning from successful trial")
        
        try:
            # Use existing tactic curation method from attacker
            learning_result = self.attacker.curate_tactics(
                learning_note="Initial GALA knowledge base",
                learning_note_supplementary=json.dumps({
                    "newTacticPool": self.learned_tactics,
                    "selectionFramework": [],
                    "promptNotes": []
                }),
                goal=goal,
                conversation_str=self._format_conversation_history(conversation_history)
            )
            
            if learning_result and "newTacticPool" in learning_result and self.enable_knowledge_evolution:
                new_tactics = learning_result["newTacticPool"]
                if new_tactics:
                    # Check for genuinely new tactics
                    added_tactics = []
                    for tactic in new_tactics:
                        if not self._tactic_exists(tactic["tactic"]):
                            self.learned_tactics.append(tactic)
                            added_tactics.append(tactic["tactic"])
                    
                    if added_tactics:
                        logger.info(f"Learned new tactics: {added_tactics}")
                        return {
                            "type": "tactic_wise",
                            "new_tactics": added_tactics,
                            "selection_framework": learning_result.get("selectionFramework", [])
                        }
            elif not self.enable_knowledge_evolution:
                logger.info("Knowledge evolution disabled - not storing new tactics")
            
        except Exception as e:
            logger.error(f"Error in tactic-wise learning: {e}")
            
        return None
    
    def perform_prompt_wise_learning(self, goal: str, conversation_history: List[Dict],
                                   failure_reason: str) -> Optional[Dict[str, Any]]:
        """
        Perform local prompt-wise learning after failed trials.
        
        Args:
            goal: The attack goal that failed
            conversation_history: Complete conversation turns  
            failure_reason: Why the attack failed
            
        Returns:
            Prompt improvement suggestions
        """
        if not self.enable_prompt_learning:
            logger.info("Prompt-wise learning disabled - skipping")
            return None
            
        logger.info("Performing prompt-wise learning from failed trial")
        
        try:
            # Get prompt-wise improvements using existing method
            improvement_result = self.attacker.analyze_and_improve_prompts(
                goal=goal,
                conversation_str=self._format_conversation_history(conversation_history),
                criticism=failure_reason
            )
            
            if improvement_result and "suggestion" in improvement_result:
                # Store improvements for this specific goal
                self.prompt_improvements[goal] = improvement_result["suggestion"]
                
                logger.info(f"Generated prompt improvements for goal: {goal[:50]}...")
                return {
                    "type": "prompt_wise", 
                    "goal": goal,
                    "suggestions": improvement_result["suggestion"]
                }
                
        except Exception as e:
            logger.error(f"Error in prompt-wise learning: {e}")
            
        return None
    
    def get_prompt_suggestions(self, goal: str) -> Optional[str]:
        """Get stored prompt suggestions for a specific goal."""
        return self.prompt_improvements.get(goal)
    
    def get_learned_tactics(self) -> List[Dict[str, Any]]:
        """Get all learned tactics."""
        return self.learned_tactics.copy()
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for learning analysis."""
        formatted_turns = []
        for i, turn in enumerate(history, 1):
            if "attacker_prompt" in turn and "target_response" in turn:
                formatted_turns.append(f"Turn {i}:")
                formatted_turns.append(f"Attacker: {turn['attacker_prompt']}")
                formatted_turns.append(f"Target: {turn['target_response']}")
                formatted_turns.append("")
        return "\n".join(formatted_turns)
    
    def _tactic_exists(self, tactic_name: str) -> bool:
        """Check if a tactic already exists in learned tactics."""
        return any(t["tactic"] == tactic_name for t in self.learned_tactics)


def run_gala_conversation(attacker: AttackerAgent, target: TargetAgent, judge: JudgeAgent,
                         goal: str, max_turns: int, prompt_suggestions: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a single GALA conversation with enhanced planning and belief tracking.
    
    Args:
        attacker: GALA-enhanced attacker agent
        target: Target agent
        judge: Judge agent
        goal: Attack goal
        max_turns: Maximum conversation turns
        prompt_suggestions: Optional prompt improvement suggestions from previous failures
        
    Returns:
        Dict containing conversation result with enhanced metadata
    """
    logger.info(f"Starting GALA conversation for goal: {goal[:50]}...")
    
    conversation = ConversationManager()
    target_safety_prompt = target.get_safety_prompt()
    violation_count = 0
    final_violation_status = 0
    
    turns_data = []
    
    # Initialize attacker with prompt suggestions if available
    if prompt_suggestions:
        logger.info("Using prompt suggestions from previous failed attempt")
        attacker.set_prompt_suggestions(prompt_suggestions)
    
    for turn_num in range(max_turns):
        logger.info(f"=== GALA Turn {turn_num + 1}/{max_turns} ===")
        
        try:
            # Enhanced attacker planning with GALA capabilities
            if turn_num == 0:
                # Use enhanced initial planning with learned knowledge
                attacker_prompt = attacker.get_initial_plan_gala(
                    goal, target_safety_prompt, prompt_suggestions
                )
            else:
                # Use enhanced followup planning
                attacker_prompt = attacker.get_followup_plan_gala(goal, target_safety_prompt)
            
            logger.info(f"Attacker prompt: {attacker_prompt[:100]}...")
            
            # Get target response
            target_response = target.respond(attacker_prompt)
            logger.info(f"Target response: {target_response[:100]}...")
            
            # Update conversation
            conversation.add_turn(attacker_prompt, target_response)
            
            # Enhanced belief state update
            attacker.belief_state.add_turn(attacker_prompt, target_response)
            attacker.belief_state.update_from_response(
                target_response, 
                attacker.belief_state.strategy_state["currentTactic"]
            )
            
            # Judge evaluation with enhanced analysis
            judge_result = judge.evaluate_turn(attacker_prompt, target_response, goal)
            violation_intensity = judge_result["violation_intensity"]
            
            logger.info(f"Judge: PV={violation_intensity}, reasoning: {judge_result['reasoning'][:50]}...")
            
            # Enhanced turn data collection
            turn_data = {
                "turn": turn_num + 1,
                "attacker_prompt": attacker_prompt,
                "target_response": target_response,
                "judge_evaluation": {
                    "violation_intensity": violation_intensity,
                    "reasoning": judge_result["reasoning"]
                },
                "belief_state": attacker.belief_state.get_state_summary(),
                "tactics_used": attacker.belief_state.strategy_state.get("tacticsTried", [])
            }
            turns_data.append(turn_data)
            
            # Check for violations with GALA logic  
            if violation_intensity == 2:
                logger.info("Strong violation detected - GALA success!")
                final_violation_status = 2
                break
            elif violation_intensity == 1:
                violation_count += 1
                logger.info(f"Weak violation detected (count: {violation_count})")
                
            # GALA self-assessment for early termination
            progress_score = attacker.belief_state.conversation_context.get("progressScore", 0)
            if progress_score >= 0.9 and turn_num >= 2:
                logger.info("High progress score achieved - attempting final push")
                # Could add special final attempt logic here
                
        except Exception as e:
            logger.error(f"Error in GALA turn {turn_num + 1}: {e}")
            turn_data = {
                "turn": turn_num + 1,
                "error": str(e)
            }
            turns_data.append(turn_data)
            break
    
    # Determine final violation status
    if final_violation_status != 2 and violation_count > 0:
        final_violation_status = 1
    
    # Enhanced conversation result
    conversation_result = {
        "violation_result": final_violation_status,
        "turns": turns_data,
        "conversation_summary": conversation.get_conversation_summary(),
        "final_belief_state": attacker.belief_state.get_state_summary(),
        "tactics_evolution": [turn.get("tactics_used", []) for turn in turns_data],
        "success": final_violation_status == 2
    }
    
    logger.info(f"GALA conversation completed: violation_status={final_violation_status}")
    return conversation_result


def run_gala_goal_trials(attacker: AttackerAgent, target: TargetAgent, judge: JudgeAgent,
                        learning_module: GALALearningModule, goal: str, max_trials: int, 
                        max_turns: int) -> tuple:
    """
    Run GALA trials for a single goal with dual learning mechanisms.
    
    Args:
        attacker: GALA-enhanced attacker agent
        target: Target agent  
        judge: Judge agent
        learning_module: GALA learning module
        goal: Attack goal
        max_trials: Maximum retry attempts
        max_turns: Maximum turns per conversation
        
    Returns:
        Tuple of (trials_data, knowledge_update)
    """
    logger.info(f"Running GALA trials (max: {max_trials}) for goal")
    trials_data = []
    knowledge_update = None
    prompt_suggestions = None
    
    for trial_num in range(max_trials):
        logger.info(f"--- GALA Trial {trial_num + 1}/{max_trials} ---")
        
        # Reset attacker for new trial (but keep learned knowledge)
        attacker.reset_for_new_trial()
        
        # Run GALA conversation
        trial_result = run_gala_conversation(
            attacker, target, judge, goal, max_turns, prompt_suggestions
        )
        trial_result["trial_id"] = trial_num + 1
        trial_result["used_prompt_suggestions"] = prompt_suggestions is not None
        trials_data.append(trial_result)
        
        success = trial_result["success"]
        
        if success:
            logger.info("🎯 GALA SUCCESS - performing tactic-wise learning")
            
            # Perform global tactic-wise learning
            knowledge_update = learning_module.perform_tactic_wise_learning(
                goal, trial_result["turns"], success=True
            )
            
            # Stop trials on success
            break
            
        else:
            logger.info("❌ GALA FAILURE - performing prompt-wise learning")
            
            # Perform local prompt-wise learning for retry
            failure_reason = "Failed to achieve goal - target model maintained refusal or provided insufficient information"
            prompt_learning = learning_module.perform_prompt_wise_learning(
                goal, trial_result["turns"], failure_reason  
            )
            
            if prompt_learning:
                # Use learned prompt suggestions for next trial
                prompt_suggestions = learning_module.get_prompt_suggestions(goal)
                logger.info("Generated prompt improvements for retry")
    
    return trials_data, knowledge_update


def initialize_gala_agents(config: Dict[str, Any]) -> tuple:
    """Initialize GALA-enhanced agents from configuration."""
    logger.info("Initializing GALA agents...")
    
    try:
        # Convert DictConfig to regular dict for llm_model_factory
        attacker_config = dict(config.agents.attacker)
        reasoning_config = dict(config.agents.reasoning)
        target_config = dict(config.agents.target)
        judge_config = dict(config.agents.judge)
        
        # Initialize GALA-enhanced AttackerAgent
        attacker = AttackerAgent(
            attacker_config=attacker_config,
            reasoning_config=reasoning_config,
            enable_gala=True  # Enable GALA enhancements
        )
        logger.info("GALA AttackerAgent initialized")
        
        # Initialize TargetAgent (unchanged)
        target = TargetAgent(target_config=target_config)
        logger.info("TargetAgent initialized")
        
        # Initialize JudgeAgent (unchanged)
        judge = JudgeAgent(judge_config=judge_config)
        logger.info("JudgeAgent initialized")
        
        return attacker, target, judge
        
    except Exception as e:
        logger.error(f"Failed to initialize GALA agents: {e}")
        raise


def main(config):
    """Main GALA pipeline function."""
    logger.info("🚀 Starting GALA multi-turn red-teaming pipeline")
    
    try:
        # Load dataset
        goals = load_jailbreak_dataset(config.data)
        logger.info(f"Loaded {len(goals)} goals from dataset")
        
        # Initialize GALA agents
        attacker, target, judge = initialize_gala_agents(config)
        
        # Initialize GALA learning module with config
        gala_config = config.get('gala', {})
        learning_module = GALALearningModule(attacker, gala_config)
        
        # Initialize GALA storage
        output_path = config.get('output_path', 'gala_conversation_results.json')
        knowledge_path = config.get('knowledge_path', 'gala_knowledge_evolution.json')
        storage = GALAConversationStorage(output_path, knowledge_path)
        
        # GALA pipeline configuration
        max_turns = config.pipeline.get('max_turns', 5)  # GALA paper uses 5 turns
        max_trials = config.pipeline.get('max_trials', 2)  # Retry mechanism
        
        logger.info(f"GALA config: max_turns={max_turns}, max_trials={max_trials}")
        logger.info(f"GALA learning: tactic-wise (global) + prompt-wise (local)")
        
        # GALA statistics
        total_successes = 0
        total_learned_tactics = 0
        
        # Process each goal with GALA
        for goal_idx, goal_text in enumerate(goals):
            goal_id = f"gala_goal_{goal_idx + 1:03d}"
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 GALA Processing {goal_id}: {goal_text[:80]}...")
            
            try:
                # Run GALA trials for this goal
                trials_results, knowledge_update = run_gala_goal_trials(
                    attacker, target, judge, learning_module, goal_text, max_trials, max_turns
                )
                
                # Update statistics
                if any(trial["success"] for trial in trials_results):
                    total_successes += 1
                
                if knowledge_update and knowledge_update.get("type") == "tactic_wise":
                    total_learned_tactics += len(knowledge_update.get("new_tactics", []))
                
                # Store results with knowledge updates
                storage.add_goal_result(goal_id, goal_text, trials_results, knowledge_update)
                
                # Log GALA summary
                violation_results = [trial["violation_result"] for trial in trials_results]
                success_achieved = any(trial["success"] for trial in trials_results)
                
                logger.info(f"🏁 GALA Goal {goal_id} completed:")
                logger.info(f"   - Trials: {len(trials_results)}")
                logger.info(f"   - Violations: {violation_results}")
                logger.info(f"   - Success: {success_achieved}")
                logger.info(f"   - Knowledge update: {knowledge_update is not None}")
                
            except Exception as e:
                logger.error(f"❌ Error processing GALA goal {goal_id}: {e}")
                # Store error result
                storage.add_goal_result(goal_id, goal_text, [{"error": str(e)}])
        
        # Save all GALA results and knowledge
        storage.save()
        
        # Final GALA statistics
        success_rate = (total_successes / len(goals)) * 100 if goals else 0
        learned_tactics = learning_module.get_learned_tactics()
        
        logger.info(f"\n🎉 GALA PIPELINE COMPLETED!")
        logger.info(f"📊 Success Rate: {success_rate:.1f}% ({total_successes}/{len(goals)})")
        logger.info(f"🧠 Tactics Learned: {len(learned_tactics)}")
        logger.info(f"💾 Results: {output_path}")
        logger.info(f"📚 Knowledge: {knowledge_path}")
        
        if learned_tactics:
            tactic_names = [t.get("tactic", "Unknown") for t in learned_tactics]
            logger.info(f"🔍 New Tactics Discovered: {tactic_names}")
        
    except Exception as e:
        logger.error(f"❌ GALA pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the GALA configuration file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    main(config)
