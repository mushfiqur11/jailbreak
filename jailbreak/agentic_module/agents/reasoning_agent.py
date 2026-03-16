"""
Reasoning Agent for multi-turn red-teaming pipeline.

Centralized reasoning module that handles:
- Belief state updates
- Informed traceback analysis
- Tactic curation (GALA)
- Tactic generalization (batch processing)
"""

import logging
import json
from typing import Dict, Any, List, Optional

from jailbreak.llm_module import llm_model_factory
from jailbreak.agentic_module.agents.reasoning_prompts import (
    get_belief_state_update_prompt,
    parse_belief_state_update_response,
    get_informed_traceback_prompt,
    parse_informed_traceback_response,
    get_curate_tactic_prompt,
    parse_curate_tactic_response,
    get_generalize_tactics_prompt,
    parse_generalize_tactics_response
)

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """
    Centralized reasoning agent for belief updates, traceback, and tactic operations.
    
    Uses a dedicated LLM for all reasoning tasks.
    """
    
    def __init__(self, reasoning_config: Dict[str, Any]):
        """
        Initialize the ReasoningAgent with LLM configuration.
        
        Args:
            reasoning_config: Configuration for the reasoning LLM
        """
        logger.info("Initializing ReasoningAgent")
        
        self.reasoning_llm = llm_model_factory(reasoning_config)
        logger.info(f"Successfully created reasoning LLM: {self.reasoning_llm}")
        
        logger.info("ReasoningAgent initialized successfully")
    
    def update_belief_state(
        self, 
        conversation_str: str, 
        current_tactic: List[str], 
        goal: str, 
        belief_state_str: str
    ) -> Dict[str, Any]:
        """
        Update belief state based on conversation context.
        
        Args:
            conversation_str: Formatted conversation history
            current_tactic: List of current tactics being used
            goal: The attack goal
            belief_state_str: Current belief state as JSON string
            
        Returns:
            Updated belief state dictionary
        """
        logger.info("Updating belief state via LLM")
        
        # Format current_tactic as string for prompt
        current_tactic_str = ", ".join(current_tactic) if current_tactic else "none"
        
        prompt = get_belief_state_update_prompt(
            conversation_str=conversation_str,
            current_tactic=current_tactic_str,
            goal=goal,
            belief_state_str=belief_state_str
        )
        
        self.reasoning_llm.set_system_prompt(prompt)
        self.reasoning_llm.clear_conversation()
        
        result = self.reasoning_llm.forward()
        raw_response = result['response']
        logger.debug(
            "===== RAW BELIEF-STATE UPDATE RESPONSE START =====\n%s\n===== RAW BELIEF-STATE UPDATE RESPONSE END =====",
            raw_response,
        )
        logger.debug(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
        
        parse_result = parse_belief_state_update_response(raw_response)
        
        if not parse_result.success:
            raise ValueError(f"Failed to parse belief state update: {parse_result.error}")
        
        logger.info("Belief state updated successfully")
        return parse_result.data
    
    def perform_informed_traceback(
        self, 
        goal: str, 
        target_safety_prompt: str, 
        conversation_str: str,
        max_parse_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Identify the best turn to fallback to when max turns reached without success.
        
        Args:
            goal: The attack goal
            target_safety_prompt: Target's safety measures
            conversation_str: Formatted conversation history
            
        Returns:
            Dictionary with fallbackTurn and reasoning
        """
        logger.info("Performing informed traceback analysis")

        base_prompt = get_informed_traceback_prompt(
            goal=goal,
            target_safety_prompt=target_safety_prompt,
            conversation_str=conversation_str
        )

        retries = max(1, int(max_parse_retries))
        last_error = ""
        last_response = ""

        for attempt in range(1, retries + 1):
            if attempt == 1:
                prompt = base_prompt
            else:
                prompt = self._append_format_retry_notice(base_prompt)

            self.reasoning_llm.set_system_prompt(prompt)
            self.reasoning_llm.clear_conversation()

            result = self.reasoning_llm.forward()
            raw_response = result['response']
            last_response = raw_response

            logger.debug(
                "===== RAW INFORMED TRACEBACK RESPONSE (attempt %d/%d) START =====\n%s\n===== RAW INFORMED TRACEBACK RESPONSE END =====",
                attempt,
                retries,
                raw_response,
            )
            logger.debug(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")

            parse_result = parse_informed_traceback_response(raw_response)
            if parse_result.success:
                logger.info(
                    "Traceback analysis complete on attempt %d/%d: fallback to turn %s",
                    attempt,
                    retries,
                    parse_result.data['fallbackTurn'],
                )
                return parse_result.data

            last_error = parse_result.error
            logger.warning(
                "Traceback parsing failed on attempt %d/%d: %s",
                attempt,
                retries,
                last_error,
            )

        raise ValueError(
            "Failed to parse traceback response after retries "
            f"({retries} attempts): {last_error}"
        )

    def _append_format_retry_notice(self, original_prompt: str) -> str:
        """Append a strict format warning while keeping the original request unchanged."""
        return (
            f"{original_prompt}\n\n"
            "your previous attempt in generating the response did not adhere to the proper output format\n"
            "Please regenerate the response to the same request above and strictly follow the required output format.\n"
            "Return only valid JSON and no extra text."
        )
    
    def curate_tactics(
        self, 
        learning_note: str, 
        learning_note_supplementary: str, 
        goal: str, 
        conversation_str: str
    ) -> Dict[str, Any]:
        """
        GALA tactic curation - learn new tactics from successful trials.
        
        Args:
            learning_note: Initial learning note (read-only)
            learning_note_supplementary: Supplementary note to update
            goal: The attack goal that succeeded
            conversation_str: Successful conversation history
            
        Returns:
            Updated supplementary learning note with new tactics
        """
        logger.info("Performing GALA tactic curation")
        
        prompt = get_curate_tactic_prompt(
            learning_note=learning_note,
            learning_note_supplementary=learning_note_supplementary,
            goal=goal,
            conversation_str=conversation_str
        )
        
        self.reasoning_llm.set_system_prompt(prompt)
        self.reasoning_llm.clear_conversation()
        
        result = self.reasoning_llm.forward()
        raw_response = result['response']
        logger.debug(
            "===== RAW TACTIC CURATION RESPONSE START =====\n%s\n===== RAW TACTIC CURATION RESPONSE END =====",
            raw_response,
        )
        logger.debug(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
        
        parse_result = parse_curate_tactic_response(raw_response)
        
        if not parse_result.success:
            raise ValueError(f"Failed to parse tactic curation: {parse_result.error}")
        
        logger.info("Tactic curation completed successfully")
        return parse_result.data
    
    def generalize_tactics(self, batch_conversations: str) -> Dict[str, Any]:
        """
        Generalize tactics from a batch of successful jailbreaks.
        
        Args:
            batch_conversations: Formatted batch of successful conversations
            
        Returns:
            Dictionary with generalized_tactics list
        """
        logger.info("Performing tactic generalization")
        
        prompt = get_generalize_tactics_prompt(batch_conversations=batch_conversations)
        
        self.reasoning_llm.set_system_prompt(prompt)
        self.reasoning_llm.clear_conversation()
        
        result = self.reasoning_llm.forward()
        raw_response = result['response']
        logger.debug(
            "===== RAW TACTIC GENERALIZATION RESPONSE START =====\n%s\n===== RAW TACTIC GENERALIZATION RESPONSE END =====",
            raw_response,
        )
        logger.debug(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
        
        parse_result = parse_generalize_tactics_response(raw_response)
        
        if not parse_result.success:
            raise ValueError(f"Failed to parse tactic generalization: {parse_result.error}")
        
        logger.info("Tactic generalization completed successfully")
        return parse_result.data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reasoning LLM."""
        return self.reasoning_llm.get_model_info()
