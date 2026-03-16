"""Judge agent implementation for jailbreaking pipeline (strict parsing mode)."""

import logging
import time
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
    
    def __init__(self, judge_config: Dict[str, Any], max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the JudgeAgent with LLM configuration.
        
        Args:
            judge_config (Dict[str, Any]): Configuration for the judge LLM
            max_retries (int): Maximum retry attempts for parsing failures
            retry_delay (float): Delay between retries in seconds
        """
        logger.info("Initializing JudgeAgent")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
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
        
        Includes retry logic only. If strict parsing fails after retries,
        this method raises instead of falling back to defaults.
        
        Args:
            conversation_str (str): Formatted conversation string
            goal (str): The attacker's goal for context
            
        Returns:
            Dict[str, Any]: Evaluation result with PV score and reasoning
        """
        logger.debug(f"JudgeAgent evaluating conversation for goal: {goal[:50]}...")
        
        # Create evaluation message including goal context
        evaluation_message = f"Goal: {goal}\n\nConversation:\n{conversation_str}"
        
        last_error = None
        last_raw_response = ""
        
        for attempt in range(self.max_retries):
            try:
                # Clear previous conversation and add evaluation message
                self.judge_llm.clear_conversation()
                self.judge_llm.add_message("user", evaluation_message)
                
                # Generate evaluation
                result = self.judge_llm.forward()
                raw_response = result['response']
                last_raw_response = raw_response
                logger.debug(
                    "===== RAW JUDGE RESPONSE START (attempt %d) =====\n%s\n===== RAW JUDGE RESPONSE END (attempt %d) =====",
                    attempt + 1,
                    raw_response,
                    attempt + 1,
                )
                logger.debug(f"Generation metrics: {result['input_tokens']} input tokens, "
                            f"{result['output_tokens']} output tokens, {result['generation_time']:.2f}s")
                
                # Parse the response
                parse_result = parse_judge_policy_response(raw_response)
                
                if parse_result.success:
                    # Extract evaluation data
                    evaluation_result = {
                        "violation_intensity": parse_result.data["PV"],
                        "reasoning": parse_result.data["reasoning"],
                        "raw_response": raw_response,
                        "used_fallback": False
                    }
                    
                    logger.info(f"Judge evaluation completed: PV={evaluation_result['violation_intensity']}")
                    return evaluation_result
                else:
                    last_error = parse_result.error
                    logger.warning(f"Parsing failed on attempt {attempt + 1}: {parse_result.error}")
                    
                    # If this is not the last attempt, wait and retry
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Exception on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # All retries exhausted - fail strictly (no fallback/default value)
        raise ValueError(
            f"Judge parsing failed after {self.max_retries} attempts. "
            f"Last error: {last_error}. Raw response: {last_raw_response[:500]}"
        )
    
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
