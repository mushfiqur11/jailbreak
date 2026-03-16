"""
GALA (Global and Local Learning Agent) learning module.

This module implements the dual learning mechanisms from the GALA paper:
1. Global tactic-wise learning: Accumulates tactical knowledge over time
2. Local prompt-wise learning: Refines implementation for specific goals
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GALALearningModule:
    """GALA's dynamic learning module with global and local learning."""
    
    def __init__(self, attacker_agent, config: Optional[Dict[str, Any]] = None):
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
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning module statistics."""
        return {
            "learned_tactics_count": len(self.learned_tactics),
            "goals_with_improvements": len(self.prompt_improvements),
            "config": {
                "enable_tactic_learning": self.enable_tactic_learning,
                "enable_prompt_learning": self.enable_prompt_learning,
                "enable_knowledge_evolution": self.enable_knowledge_evolution,
                "enable_dynamic_selection": self.enable_dynamic_selection
            }
        }
    
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


class GALAConversationStorage:
    """Enhanced storage for GALA conversations with learning data."""
    
    def __init__(self, output_path: str, knowledge_path: Optional[str] = None):
        """Initialize storage with output and optional knowledge file paths."""
        self.output_path = output_path
        self.knowledge_path = knowledge_path
        self.data = {}
        self.knowledge_evolution = []
        logger.info(f"GALAConversationStorage initialized: output={output_path}, knowledge={knowledge_path}")
    
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
            
            # Save knowledge evolution if path provided
            if self.knowledge_path:
                with open(self.knowledge_path, 'w') as f:
                    json.dump(self.knowledge_evolution, f, indent=2)
                logger.info(f"Knowledge evolution saved to {self.knowledge_path}")
                
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise
