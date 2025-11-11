"""
GPT model implementations.

This module provides the GPT class for all OpenAI GPT model variants,
including GPT-3.5-turbo, GPT-4o, and o3-mini series.
"""

from typing import Dict, Any, List, Optional
from ..base.openai_base import OpenAIBase


class GPT(OpenAIBase):
    """
    GPT language model implementation for OpenAI models.
    
    Supports all OpenAI GPT variants including:
    - gpt-3.5-turbo
    - gpt-4o / gpt-4o-2024-08-06
    - chatgpt-4o-latest
    - o3-mini / o3-mini-2025-01-31
    """
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        'gpt-3.5-turbo',
        'o3-mini',
        'o3-mini-2025-01-31',
        'gpt-4o',
        'gpt-4o-2024-08-06',
        'chatgpt-4o-latest'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GPT model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        # Validate model ID
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known GPT variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "gpt-4o"
            print(f"No model_id specified, using default: {config['model_id']}")
        
        # Apply GPT-specific optimizations
        self._apply_gpt_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        
        print(f"GPT model initialized: {self.model_id}")
    
    def _apply_gpt_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply GPT-specific optimizations to the configuration."""
        
        # GPT-specific generation defaults
        gpt_defaults = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Apply defaults if not already specified
        for key, default_value in gpt_defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Model-specific optimizations
        model_id = config.get("model_id", "").lower()
        
        if "o3" in model_id:
            # o3 models - reasoning optimized
            config.update({
                "max_tokens": config.get("max_tokens", 1024),  # Higher default for reasoning
                "temperature": config.get("temperature", 0.3)  # Lower for more focused responses
            })
            print("Applied o3 model optimizations (reasoning focused)")
            
        elif "gpt-4o" in model_id:
            # GPT-4o models - balanced performance
            config.update({
                "max_tokens": config.get("max_tokens", 1024),
                "temperature": config.get("temperature", 0.7)
            })
            print("Applied GPT-4o model optimizations")
            
        elif "gpt-3.5" in model_id:
            # GPT-3.5 models - cost-effective
            config.update({
                "max_tokens": config.get("max_tokens", 512),
                "temperature": config.get("temperature", 0.8)
            })
            print("Applied GPT-3.5 model optimizations")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt with GPT-specific formatting.
        
        Args:
            prompt (str): The system prompt to use
        """
        # GPT models handle system prompts directly in conversation
        super().set_system_prompt(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get GPT-specific model information.
        
        Returns:
            Dict[str, Any]: Enhanced model information
        """
        info = super().get_model_info()
        info.update({
            "model_family": "GPT",
            "supported_variants": self.SUPPORTED_VARIANTS,
            "provider": "OpenAI",
            "estimated_parameters": self._get_parameter_count(),
            "context_length": self._get_context_length(),
            "special_features": self._get_special_features(),
            "pricing_tier": self._get_pricing_tier()
        })
        return info
    
    def _get_parameter_count(self) -> str:
        """Get estimated parameter count based on model name."""
        model_id_lower = self.model_id.lower()
        
        if "o3" in model_id_lower:
            return "~200B+ parameters (estimated)"
        elif "gpt-4o" in model_id_lower:
            return "~175B parameters (estimated)"
        elif "gpt-3.5" in model_id_lower:
            return "~175B parameters (estimated)"
        else:
            return "Unknown"
    
    def _get_context_length(self) -> int:
        """Get context length based on model variant."""
        model_id_lower = self.model_id.lower()
        
        if "o3" in model_id_lower:
            return 128000  # o3 models context length
        elif "gpt-4o" in model_id_lower:
            return 128000  # GPT-4o context length
        elif "gpt-3.5" in model_id_lower:
            return 16384   # GPT-3.5-turbo context length
        else:
            return 4096    # Default/fallback
    
    def _get_special_features(self) -> List[str]:
        """Get special features of the model."""
        features = []
        model_id_lower = self.model_id.lower()
        
        if "o3" in model_id_lower:
            features.extend([
                "Advanced reasoning",
                "Chain-of-thought processing",
                "Multi-step problem solving",
                "Enhanced mathematical capabilities"
            ])
        
        if "gpt-4o" in model_id_lower:
            features.extend([
                "Multimodal capabilities",
                "High-quality text generation", 
                "Advanced code generation",
                "Complex reasoning"
            ])
        
        if "gpt-3.5" in model_id_lower:
            features.extend([
                "Cost-effective",
                "Fast inference",
                "Good for chat applications"
            ])
        
        # All GPT models have these features
        features.extend([
            "API-based inference",
            "No local compute required",
            "Real-time processing"
        ])
        
        return features
    
    def _get_pricing_tier(self) -> str:
        """Get pricing tier for the model."""
        model_id_lower = self.model_id.lower()
        
        if "o3" in model_id_lower:
            return "Premium"
        elif "gpt-4o" in model_id_lower:
            return "High"
        elif "gpt-3.5" in model_id_lower:
            return "Standard"
        else:
            return "Unknown"
    
    @classmethod
    def get_recommended_config(cls, model_variant: str = "gpt-4o") -> Dict[str, Any]:
        """
        Get recommended configuration for different GPT variants.
        
        Args:
            model_variant (str): Model variant ("gpt-3.5-turbo", "gpt-4o", "o3-mini", etc.)
            
        Returns:
            Dict[str, Any]: Recommended configuration
        """
        model_variant = model_variant.lower()
        
        base_config = {
            "api_key_path": "./tokens/openai_key.txt",
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        if "o3" in model_variant:
            base_config.update({
                "model_id": "o3-mini",
                "temperature": 0.3,
                "max_tokens": 1024
            })
        elif "gpt-4o" in model_variant:
            base_config.update({
                "model_id": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 1024
            })
        elif "gpt-3.5" in model_variant:
            base_config.update({
                "model_id": "gpt-3.5-turbo",
                "temperature": 0.8,
                "max_tokens": 512
            })
        else:
            # Default to GPT-4o
            base_config.update({
                "model_id": "gpt-4o"
            })
        
        return base_config
    
    def estimate_request_cost(self, prompt: str, expected_response_length: int = 100) -> Dict[str, Any]:
        """
        Estimate the cost of a request based on prompt and expected response.
        
        Args:
            prompt (str): Input prompt
            expected_response_length (int): Expected response length in tokens
            
        Returns:
            Dict[str, Any]: Cost estimation
        """
        input_tokens = self.count_tokens(prompt)
        
        return self.estimate_cost(
            input_tokens=input_tokens,
            output_tokens=expected_response_length
        )
    
    def get_usage_recommendations(self) -> Dict[str, str]:
        """
        Get usage recommendations for the current model.
        
        Returns:
            Dict[str, str]: Usage recommendations
        """
        model_id_lower = self.model_id.lower()
        
        recommendations = {
            "best_for": "",
            "avoid_for": "",
            "cost_optimization": "",
            "performance_tips": ""
        }
        
        if "o3" in model_id_lower:
            recommendations.update({
                "best_for": "Complex reasoning, mathematical problems, multi-step analysis",
                "avoid_for": "Simple tasks where cost efficiency matters",
                "cost_optimization": "Use for high-value tasks requiring advanced reasoning",
                "performance_tips": "Provide clear step-by-step instructions"
            })
        elif "gpt-4o" in model_id_lower:
            recommendations.update({
                "best_for": "High-quality content generation, complex analysis, coding",
                "avoid_for": "High-volume simple tasks",
                "cost_optimization": "Balance quality needs with usage volume",
                "performance_tips": "Clear prompts yield better results"
            })
        elif "gpt-3.5" in model_id_lower:
            recommendations.update({
                "best_for": "Chat applications, simple content generation, high-volume tasks",
                "avoid_for": "Complex reasoning or analysis requiring high accuracy",
                "cost_optimization": "Ideal for cost-sensitive applications",
                "performance_tips": "Keep prompts concise and specific"
            })
        
        return recommendations
