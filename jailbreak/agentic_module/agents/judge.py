"""
Judge agent implementation for jailbreaking pipeline.

The judge agent evaluates conversations between attacker and target to determine
if there has been a policy violation and assigns violation intensity.
"""

import logging
from typing import Dict, Any

from jailbreak.agentic_module.agents.judge_prompts.judge_policy import (
    get_judge_policy_prompt,
    parse_judge_policy_response
)
from jailbreak.llm_module import llm_model_factory

logger = logging.getLogger(__name__)


class JudgeAgent:
    """
    Judge agent that evaluates conversations to identify policy violations.
    
    The judge assigns violation intensity:
    - 0: No violation
    - 1: Weak violation  
    - 2: Strong violation
    """
    
    def __init__(self, judge_config: Dict[str, Any]):
        """
        Initialize the JudgeAgent with LLM configuration.
        
        Args:
            judge_config (Dict[str, Any]): Configuration for the judge LLM
        """
        logger.info("Initializing JudgeAgent")
        
        try:
            self.judge_llm = llm_model_factory(judge_config)
            logger.info(f"Successfully created judge LLM: {self.judge_llm}")
        except Exception as e:
            logger.error(f"Failed to create judge LLM: {e}")
            raise
        
        # Set the judge policy prompt as system prompt
        try:
            policy_prompt = get_judge_policy_prompt()
            self.judge_llm.set_system_prompt(policy_prompt)
            logger.info("Judge policy prompt set successfully")
        except Exception as e:
            logger.error(f"Failed to set judge policy prompt: {e}")
            raise
        
        logger.info("JudgeAgent initialized successfully")
    
    def evaluate_conversation(self, conversation_str: str, goal: str) -> Dict[str, Any]:
        """
        Evaluate a conversation string for policy violations.
        
        Args:
            conversation_str (str): Formatted conversation string
            goal (str): The attacker's goal for context
            
        Returns:
            Dict[str, Any]: Evaluation result with PV score and reasoning
        """
        logger.debug(f"JudgeAgent evaluating conversation for goal: {goal[:50]}...")
        
        try:
            # Create evaluation message including goal context
            evaluation_message = f"Goal: {goal}\n\nConversation:\n{conversation_str}"
            
            # Clear previous conversation and add evaluation message
            self.judge_llm.clear_conversation()
            self.judge_llm.add_message("user", evaluation_message)
            
            # Generate evaluation
            raw_response = self.judge_llm.forward()
            logger.debug(f"Raw judge response: {raw_response[:200]}...")
            
            # Parse the response
            parse_result = parse_judge_policy_response(raw_response)
            
            if not parse_result.success:
                logger.error(f"Failed to parse judge response: {parse_result.error}")
                raise ValueError(f"Judge response parsing failed: {parse_result.error}")
            
            # Extract evaluation data
            evaluation_result = {
                "violation_intensity": parse_result.data["PV"],
                "reasoning": parse_result.data["reasoning"],
                "raw_response": raw_response
            }
            
            logger.info(f"Judge evaluation completed: PV={evaluation_result['violation_intensity']}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Failed to evaluate conversation: {e}")
            raise
    
    def evaluate_turn(self, attacker_prompt: str, target_response: str, goal: str) -> Dict[str, Any]:
        """
        Evaluate a single conversation turn for policy violations.
        
        Args:
            attacker_prompt (str): The attacker's prompt
            target_response (str): The target's response  
            goal (str): The attacker's goal for context
            
        Returns:
            Dict[str, Any]: Evaluation result with PV score and reasoning
        """
        logger.debug("JudgeAgent evaluating single turn")
        
        # Format as conversation string
        conversation_str = f"Attacker: {attacker_prompt}\n\nTarget: {target_response}"
        
        return self.evaluate_conversation(conversation_str, goal)
    
    def get_policy_prompt(self) -> str:
        """
        Get the current policy prompt used by the judge.
        
        Returns:
            str: Current judge policy prompt
        """
        return get_judge_policy_prompt()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the judge LLM.
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            return self.judge_llm.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
