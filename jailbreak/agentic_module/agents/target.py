"""
Target agent implementation for jailbreaking pipeline.

The target agent represents a generic AI assistant that should follow safety policies
and not violate predefined guidelines when responding to user queries.
"""

import logging
from typing import Dict, Any, List

from jailbreak.agentic_module.agents.target_prompts.safety_prompt import (
    get_safety_prompt, 
    parse_safety_prompt_response
)
from jailbreak.llm_module import llm_model_factory

logger = logging.getLogger(__name__)


class TargetAgent:
    """
    Target agent that represents a generic AI assistant designed to serve customers
    while following safety policies and not violating predefined guidelines.
    """
    
    def __init__(self, target_config: Dict[str, Any]):
        """
        Initialize the TargetAgent with LLM configuration.
        
        Args:
            target_config (Dict[str, Any]): Configuration for the target LLM
        """
        logger.info("Initializing TargetAgent")
        
        try:
            self.target_llm = llm_model_factory(target_config)
            logger.info(f"Successfully created target LLM: {self.target_llm}")
        except Exception as e:
            logger.error(f"Failed to create target LLM: {e}")
            raise
        
        # Set the safety prompt as system prompt
        try:
            safety_prompt = get_safety_prompt()
            self.target_llm.set_system_prompt(safety_prompt)
            logger.info("Safety prompt set successfully")
        except Exception as e:
            logger.error(f"Failed to set safety prompt: {e}")
            raise
        
        logger.info("TargetAgent initialized successfully")
    
    def respond(self, user_message: str) -> str:
        """
        Generate response to user message following safety guidelines.
        
        Args:
            user_message (str): Message from user (attacker)
            
        Returns:
            str: Target's response
        """
        logger.debug(f"TargetAgent responding to message: {user_message[:100]}...")
        
        try:
            # Clear previous conversation and add the new user message
            self.target_llm.clear_conversation()
            self.target_llm.add_message("user", user_message)
            
            # Generate response
            result = self.target_llm.forward()
            raw_response = result['response']
            logger.debug(f"Raw target response: {raw_response[:200]}...")
            logger.debug(f"Generation metrics: {result['input_tokens']} input tokens, "
                        f"{result['output_tokens']} output tokens, {result['generation_time']:.2f}s")
            
            # Parse the response (currently just extracts the response text)
            parse_result = parse_safety_prompt_response(raw_response)
            
            if not parse_result.success:
                logger.warning(f"Failed to parse target response: {parse_result.error}")
                # Return raw response as fallback
                response = raw_response.strip()
            else:
                response = parse_result.data["response"]
            
            logger.info("Target response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate target response: {e}")
            raise
    
    def respond_with_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate response considering full conversation history.
        
        Args:
            conversation_history (List[Dict[str, str]]): Full conversation history in OpenAI format
            
        Returns:
            str: Target's response
        """
        logger.debug(f"TargetAgent responding with conversation history of {len(conversation_history)} messages")
        
        try:
            # Clear conversation and add full history
            self.target_llm.clear_conversation()
            self.target_llm.add_conversation(conversation_history)
            
            # Generate response
            result = self.target_llm.forward()
            raw_response = result['response']
            logger.debug(f"Raw target response: {raw_response[:200]}...")
            logger.debug(f"Generation metrics: {result['input_tokens']} input tokens, "
                        f"{result['output_tokens']} output tokens, {result['generation_time']:.2f}s")
            
            # Parse the response
            parse_result = parse_safety_prompt_response(raw_response)
            
            if not parse_result.success:
                logger.warning(f"Failed to parse target response: {parse_result.error}")
                response = raw_response.strip()
            else:
                response = parse_result.data["response"]
            
            logger.info("Target response with history generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate target response with history: {e}")
            raise
    
    def get_safety_prompt(self) -> str:
        """
        Get the current safety prompt used by the target.
        
        Returns:
            str: Current safety prompt
        """
        return get_safety_prompt()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the target LLM.
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            return self.target_llm.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
