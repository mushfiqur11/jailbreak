"""
Abstract base class for OpenAI API model implementations.

This module provides the base class for all OpenAI-based language models,
handling API authentication, request management, and response parsing.
"""

import openai
from typing import Dict, Any, Optional, List

from .base_llm import BaseLLM


class OpenAIBase(BaseLLM):
    """
    Abstract base class for OpenAI API language models.
    
    This class handles the common functionality for all OpenAI models,
    including API client management, authentication, and generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI base model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(config)
        
        # OpenAI specific configuration
        self.api_key = None
        self.client = None
        self.base_url = config.get("base_url", None)  # For custom endpoints
        self.organization = config.get("organization", None)
        
        # Default generation parameters for OpenAI
        self.generation_config = {
            "top_p": config.get("top_p", 1.0),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0),
            "stop": config.get("stop", None)
        }
        
        # Only add temperature if explicitly specified in config (not default)
        # GPT-5 nano only supports temperature=1 (default), so we skip it entirely
        if "temperature" in config and "gpt-5-nano" not in self.model_id.lower():
            self.generation_config["temperature"] = config["temperature"]
        
        # Handle max_tokens vs max_completion_tokens for different model versions
        if "max_completion_tokens" in config:
            self.generation_config["max_completion_tokens"] = config["max_completion_tokens"]
        else:
            self.generation_config["max_tokens"] = self.max_tokens
        
        # Load API credentials and initialize client
        self._load_model()
    
    def _load_model(self) -> None:
        """Load/initialize the OpenAI API client."""
        if self.is_loaded:
            return
        
        print(f"Initializing OpenAI client for model: {self.model_id}")
        
        # Load API key
        self._load_api_key()
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        if self.organization:
            client_kwargs["organization"] = self.organization
        
        self.client = openai.OpenAI(**client_kwargs)
        
        self.is_loaded = True
        print(f"OpenAI client initialized successfully for: {self.model_id}")
    
    def _load_api_key(self) -> None:
        """Load the OpenAI API key from configuration."""
        # Try to get API key from config directly
        if "api_key" in self.config:
            self.api_key = self.config["api_key"]
            return
        
        # Try to load from file
        api_key_path = self.config.get("api_key_path", self.config.get("openai_api_key_path"))
        if api_key_path:
            try:
                with open(api_key_path, 'r') as f:
                    self.api_key = f.read().strip()
                return
            except FileNotFoundError:
                raise FileNotFoundError(f"API key file not found: {api_key_path}")
            except Exception as e:
                raise Exception(f"Error reading API key file: {e}")
        
        # Try environment variable (handled by OpenAI client automatically)
        # If no explicit key is provided, OpenAI client will try OPENAI_API_KEY env var
        import os
        if "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
            return
        
        raise ValueError(
            "OpenAI API key not found. Please provide it via:\n"
            "1. 'api_key' in config\n"
            "2. 'api_key_path' file path in config\n"
            "3. OPENAI_API_KEY environment variable"
        )
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for conversations.
        
        Args:
            prompt (str): The system prompt to use
        """
        self.system_prompt = prompt
        print(f"System prompt set: {prompt[:100]}...")
    
    def forward(self, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a response using the OpenAI API.
        
        Args:
            messages (Optional[List[Dict[str, str]]]): Optional messages to use.
                If None, uses the internal conversation.
                
        Returns:
            Dict[str, Any]: Dictionary containing response, token counts, and timing info
        """
        import time
        
        if not self.is_loaded:
            raise RuntimeError("OpenAI client is not initialized. Call _load_model() first.")
        
        # Prepare messages
        prepared_messages = self._prepare_messages(messages)
        
        if not prepared_messages:
            raise ValueError("No messages to process")
        
        start_time = time.time()
        
        try:
            # Check if structured output is requested
            if "response_format" in self.config:
                # Use beta parsing for structured output
                response = self.client.beta.chat.completions.parse(
                    model=self.model_id,
                    messages=prepared_messages,
                    response_format=self.config["response_format"],
                    **{k: v for k, v in self.generation_config.items() if v is not None}
                )
                response_text = response.choices[0].message.parsed
            else:
                # Standard chat completion
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=prepared_messages,
                    **{k: v for k, v in self.generation_config.items() if v is not None}
                )
                response_text = response.choices[0].message.content.strip()
            
            generation_time = time.time() - start_time
            
            # Extract token usage information
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            
            return {
                'response': response_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'generation_time': generation_time
            }
                
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
        except openai.RateLimitError as e:
            raise Exception(f"OpenAI rate limit error: {e}")
        except openai.AuthenticationError as e:
            raise Exception(f"OpenAI authentication error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error during OpenAI API call: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            "model_type": "OpenAI",
            "api_key_configured": self.api_key is not None,
            "base_url": self.base_url,
            "organization": self.organization,
            "generation_config": self.generation_config
        })
        return info
    
    def estimate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        pricing: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost of a request (rough approximation).
        
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            pricing (Optional[Dict[str, float]]): Custom pricing per 1K tokens
            
        Returns:
            Dict[str, Any]: Cost estimation information
        """
        # Default pricing (approximate, as of 2024 - should be updated)
        default_pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
            "chatgpt-4o-latest": {"input": 0.005, "output": 0.015},
            "o3-mini": {"input": 0.002, "output": 0.008},  # Estimated
            "o3-mini-2025-01-31": {"input": 0.002, "output": 0.008}  # Estimated
        }
        
        if pricing:
            model_pricing = pricing
        else:
            model_pricing = default_pricing.get(self.model_id, {"input": 0.002, "output": 0.006})
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "model_id": self.model_id,
            "pricing_per_1k": model_pricing
        }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough approximation: ~4 characters per token for GPT models
        # For more accurate counting, would need tiktoken library
        return len(text) // 4
    
    def validate_model_id(self) -> bool:
        """
        Validate that the model ID is supported by OpenAI.
        
        Returns:
            bool: True if model ID appears to be valid
        """
        # List of known OpenAI model patterns
        valid_patterns = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
            "chatgpt-4o",
            "o3-mini",
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada"
        ]
        
        model_id_lower = self.model_id.lower()
        return any(pattern in model_id_lower for pattern in valid_patterns)
