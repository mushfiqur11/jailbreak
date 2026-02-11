"""
Traceback Module for multi-turn red-teaming pipeline.

Handles conversation rewind when max turns are reached without success.
The traceback identifies the best turn to fallback to and stores
failure knowledge for improved retry attempts.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TracebackResult:
    """Result of traceback analysis."""
    fallback_turn: int
    failure_reason: str
    truncated_history: List[Dict[str, Any]]
    failure_knowledge: Dict[str, Any]


class TracebackModule:
    """
    Module for performing traceback when max turns are reached without success.
    
    Traceback Flow:
    1. Reasoning agent identifies best fallback turn
    2. Conversation history is truncated to that turn
    3. Failure knowledge is stored for retry improvement
    """
    
    def __init__(self, reasoning_agent, min_turns: int = 3):
        """
        Initialize the TracebackModule.
        
        Args:
            reasoning_agent: ReasoningAgent instance for traceback analysis
            min_turns: Minimum turns before traceback is considered
        """
        self.reasoning = reasoning_agent
        self.min_turns = min_turns
        
        logger.info(f"TracebackModule initialized with min_turns={min_turns}")
    
    def should_perform_traceback(self, turn_count: int, reached_max_turns: bool) -> bool:
        """
        Determine if traceback should be performed.
        
        Args:
            turn_count: Number of turns completed
            reached_max_turns: Whether max turns were reached
            
        Returns:
            True if traceback should be performed
        """
        return reached_max_turns and turn_count >= self.min_turns
    
    def perform_traceback(
        self, 
        goal: str, 
        conversation_history: List[Dict[str, Any]], 
        target_safety_prompt: str
    ) -> TracebackResult:
        """
        Perform traceback analysis and return result with truncated history.
        
        Args:
            goal: The attack goal
            conversation_history: Full conversation history
            target_safety_prompt: Target's safety measures
            
        Returns:
            TracebackResult with fallback turn and truncated history
        """
        logger.info(f"Performing traceback for conversation with {len(conversation_history)} turns")
        
        # Format conversation for analysis
        conversation_str = self._format_conversation_for_analysis(conversation_history)
        
        # Call reasoning agent to identify fallback turn
        traceback_analysis = self.reasoning.perform_informed_traceback(
            goal=goal,
            target_safety_prompt=target_safety_prompt,
            conversation_str=conversation_str
        )
        
        fallback_turn = int(traceback_analysis["fallbackTurn"])
        failure_reason = traceback_analysis["reasoning"]
        
        # Validate fallback turn
        if fallback_turn < 1:
            fallback_turn = 1
        if fallback_turn > len(conversation_history):
            fallback_turn = max(1, len(conversation_history) - 1)
        
        # Truncate conversation history to fallback turn
        truncated_history = conversation_history[:fallback_turn]
        
        # Create failure knowledge for retry
        failure_knowledge = {
            "original_turn_count": len(conversation_history),
            "fallback_turn": fallback_turn,
            "failure_reason": failure_reason,
            "what_went_wrong": failure_reason,
            "turns_discarded": len(conversation_history) - fallback_turn
        }
        
        logger.info(f"Traceback complete: fallback to turn {fallback_turn}, "
                   f"discarding {failure_knowledge['turns_discarded']} turns")
        
        return TracebackResult(
            fallback_turn=fallback_turn,
            failure_reason=failure_reason,
            truncated_history=truncated_history,
            failure_knowledge=failure_knowledge
        )
    
    def _format_conversation_for_analysis(self, history: List[Dict[str, Any]]) -> str:
        """
        Format conversation history for traceback analysis.
        
        Args:
            history: List of conversation turns
            
        Returns:
            Formatted string for analysis
        """
        formatted_parts = []
        
        for i, turn in enumerate(history, 1):
            turn_str = f"Turn {i}:\n"
            
            if "attacker_prompt" in turn:
                turn_str += f"Attacker: {turn['attacker_prompt']}\n"
            
            if "target_response" in turn:
                turn_str += f"Target: {turn['target_response']}\n"
            
            if "judge_evaluation" in turn:
                eval_data = turn["judge_evaluation"]
                pv = eval_data.get("violation_intensity", 0)
                turn_str += f"Judge PV: {pv}\n"
            
            formatted_parts.append(turn_str)
        
        return "\n".join(formatted_parts)


def create_traceback_module(reasoning_agent, config: Dict[str, Any]) -> TracebackModule:
    """
    Factory function to create a TracebackModule from config.
    
    Args:
        reasoning_agent: ReasoningAgent instance
        config: Configuration dictionary with features section
        
    Returns:
        Configured TracebackModule
    """
    min_turns = config.get("features", {}).get("traceback_min_turns", 3)
    return TracebackModule(reasoning_agent, min_turns=min_turns)
