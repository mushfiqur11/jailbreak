"""
DeepSeek model implementations.

This module provides the DeepSeek class for all DeepSeek model variants,
including DeepSeek-R1-Distill series models.
"""

from typing import Dict, Any, List, Optional
from ..base.hf_base import HuggingFaceBase


class DeepSeek(HuggingFaceBase):
    """
    DeepSeek language model implementation.
    
    Supports all DeepSeek variants including:
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    - deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    """
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DeepSeek model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        # Validate model ID
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known DeepSeek variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            print(f"No model_id specified, using default: {config['model_id']}")
        
        # Apply DeepSeek-specific optimizations
        self._apply_deepseek_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        
        print(f"DeepSeek model initialized: {self.model_id}")
    
    def _apply_deepseek_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply DeepSeek-specific optimizations to the configuration."""
        
        # DeepSeek-specific generation defaults
        deepseek_defaults = {
            "temperature": 0.3,  # DeepSeek-R1 benefits from lower temperature
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": 1024  # Higher for reasoning tasks
        }
        
        # Apply defaults if not already specified
        for key, default_value in deepseek_defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Model size-specific optimizations
        model_id = config.get("model_id", "").lower()
        
        if "70b" in model_id:
            # Large model optimizations
            if "quantization" not in config:
                config["quantization"] = "4bit"  # Default to 4-bit for 70B models
            print("Applied DeepSeek-70B model optimizations (4-bit quantization)")
            
        elif any(size in model_id for size in ["32b", "14b"]):
            # Medium/Large model optimizations
            if "quantization" not in config:
                config["quantization"] = "auto"  # Auto-select based on available memory
            print("Applied DeepSeek medium/large model optimizations (auto quantization)")
            
        elif "1.5b" in model_id:
            # Small model optimizations
            if "quantization" not in config:
                config["quantization"] = "none"  # Usually no quantization needed
            print("Applied DeepSeek small model optimizations")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt with DeepSeek-specific formatting.
        
        Args:
            prompt (str): The system prompt to use
        """
        # DeepSeek models work well with reasoning-focused system prompts
        formatted_prompt = f"You are DeepSeek, an AI assistant specialized in analytical thinking and reasoning. {prompt}"
        super().set_system_prompt(formatted_prompt)
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        DeepSeek-specific message formatting fallback.
        
        Args:
            messages (List[Dict[str, str]]): Messages to format
            
        Returns:
            str: Formatted text in DeepSeek chat format
        """
        # DeepSeek uses a similar format to other models but optimized for reasoning
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        formatted_parts.append("Assistant:")  # Add generation prompt
        return "\n\n".join(formatted_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get DeepSeek-specific model information.
        
        Returns:
            Dict[str, Any]: Enhanced model information
        """
        info = super().get_model_info()
        info.update({
            "model_family": "DeepSeek",
            "provider": "DeepSeek AI",
            "supported_variants": self.SUPPORTED_VARIANTS,
            "estimated_parameters": self._get_parameter_count(),
            "context_length": self._get_context_length(),
            "special_features": self._get_special_features(),
            "base_architecture": self._get_base_architecture()
        })
        return info
    
    def _get_parameter_count(self) -> str:
        """Get estimated parameter count based on model name."""
        model_id_lower = self.model_id.lower()
        
        if "70b" in model_id_lower:
            return "~70B parameters"
        elif "32b" in model_id_lower:
            return "~32B parameters"
        elif "14b" in model_id_lower:
            return "~14B parameters"
        elif "1.5b" in model_id_lower:
            return "~1.5B parameters"
        else:
            return "Unknown"
    
    def _get_context_length(self) -> int:
        """Get context length based on model variant."""
        # DeepSeek-R1-Distill models typically have long context
        return 131072  # 128K context length
    
    def _get_base_architecture(self) -> str:
        """Get the base architecture of the model."""
        model_id_lower = self.model_id.lower()
        
        if "llama" in model_id_lower:
            return "Llama-based"
        elif "qwen" in model_id_lower:
            return "Qwen-based"
        else:
            return "Unknown"
    
    def _get_special_features(self) -> List[str]:
        """Get special features of the model."""
        features = []
        model_id_lower = self.model_id.lower()
        
        # All DeepSeek-R1 models have these features
        features.extend([
            "Reasoning specialization",
            "Chain-of-thought capabilities", 
            "Mathematical problem solving",
            "Scientific reasoning",
            "Distilled from DeepSeek-R1"
        ])
        
        if "1.5b" in model_id_lower:
            features.extend([
                "Efficient and lightweight",
                "Fast inference",
                "Good reasoning-to-size ratio"
            ])
        
        if any(size in model_id_lower for size in ["70b", "32b"]):
            features.extend([
                "Advanced reasoning capabilities",
                "Complex problem solving",
                "Research-grade performance"
            ])
        
        # Base architecture specific features
        base_arch = self._get_base_architecture()
        if base_arch == "Llama-based":
            features.append("Llama architecture optimizations")
        elif base_arch == "Qwen-based":
            features.append("Qwen architecture optimizations")
        
        return features
    
    @classmethod
    def get_recommended_config(cls, model_variant: str = "14b") -> Dict[str, Any]:
        """
        Get recommended configuration for different DeepSeek variants.
        
        Args:
            model_variant (str): Model variant ("1.5b", "14b", "32b", "70b", etc.)
            
        Returns:
            Dict[str, Any]: Recommended configuration
        """
        model_variant = model_variant.lower()
        
        base_config = {
            "hf_model_path": "./models",
            "hf_token_path": "./tokens/hf_token.txt",
            "temperature": 0.3,  # Lower for reasoning tasks
            "max_new_tokens": 1024,  # Higher for detailed reasoning
            "top_p": 0.85
        }
        
        if "70b" in model_variant:
            base_config.update({
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "quantization": "4bit",
                "max_new_tokens": 2048
            })
        elif "32b" in model_variant:
            base_config.update({
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "quantization": "4bit"
            })
        elif "14b" in model_variant:
            base_config.update({
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "quantization": "auto"
            })
        elif "1.5b" in model_variant:
            base_config.update({
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "quantization": "none"
            })
        else:
            # Default to 14B
            base_config.update({
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            })
        
        return base_config
    
    def enable_lora_finetuning(self, lora_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Enable LoRA fine-tuning for the DeepSeek model.
        
        Args:
            lora_config (Optional[Dict[str, Any]]): LoRA configuration
            
        Returns:
            Any: PEFT model ready for fine-tuning (future implementation)
        """
        if lora_config is None:
            # Use DeepSeek-optimized LoRA config
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "down_proj", "up_proj"
                ],
                "task_type": "CAUSAL_LM"
            }
        
        return self.prepare_for_lora(lora_config)
    
    def get_reasoning_recommendations(self) -> Dict[str, str]:
        """
        Get recommendations for using DeepSeek models for reasoning tasks.
        
        Returns:
            Dict[str, str]: Reasoning-specific recommendations
        """
        return {
            "temperature": "Use lower temperature (0.1-0.3) for analytical tasks",
            "prompting": "Encourage step-by-step reasoning with phrases like 'Let me think through this step by step'",
            "context": "Provide clear problem context and ask for detailed explanations",
            "verification": "Ask the model to verify its own reasoning and check for errors",
            "best_tasks": "Mathematical problems, logical puzzles, scientific analysis, code debugging",
            "token_usage": "Allow more tokens for detailed reasoning chains (1024-2048 tokens)"
        }
