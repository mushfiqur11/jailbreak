"""
Qwen model implementations.

This module provides the Qwen class for all Qwen model variants,
including Qwen2.5, Qwen3, and Qwen-VL series models.
"""

from typing import Dict, Any, List, Optional
from ..base.hf_base import HuggingFaceBase


class Qwen(HuggingFaceBase):
    """
    Qwen language model implementation.
    
    Supports all Qwen variants including:
    - Qwen/Qwen2.5-VL-7B-Instruct
    - Qwen/Qwen3-4B/8B/32B  
    - Qwen/Qwen3-0.6B
    """
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'Qwen/Qwen3-4B',
        'Qwen/Qwen3-8B', 
        'Qwen/Qwen3-32B',
        'Qwen/Qwen3-0.6B'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Qwen model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        # Validate model ID
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known Qwen variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "Qwen/Qwen3-4B"
            print(f"No model_id specified, using default: {config['model_id']}")
        
        # Apply Qwen-specific optimizations
        self._apply_qwen_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        
        print(f"Qwen model initialized: {self.model_id}")
    
    def _apply_qwen_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply Qwen-specific optimizations to the configuration."""
        
        # Qwen-specific generation defaults
        qwen_defaults = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "max_new_tokens": 512
        }
        
        # Apply defaults if not already specified
        for key, default_value in qwen_defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Model size-specific optimizations
        model_id = config.get("model_id", "").lower()
        
        if "32b" in model_id:
            # Large model optimizations
            if "quantization" not in config:
                config["quantization"] = "4bit"  # Default to 4-bit for 32B models
            print("Applied Qwen-32B model optimizations (4-bit quantization)")
            
        elif any(size in model_id for size in ["8b", "7b"]):
            # Medium model optimizations
            if "quantization" not in config:
                config["quantization"] = "auto"  # Auto-select based on available memory
            print("Applied Qwen medium model optimizations (auto quantization)")
            
        elif any(size in model_id for size in ["4b", "0.6b"]):
            # Small model optimizations
            if "quantization" not in config:
                config["quantization"] = "none"  # Usually no quantization needed
            print("Applied Qwen small model optimizations")
        
        # Vision model specific settings
        if "vl" in model_id:
            config["supports_vision"] = True
            print("Enabled vision capabilities for Qwen-VL model")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt with Qwen-specific formatting.
        
        Args:
            prompt (str): The system prompt to use
        """
        # Qwen models work well with structured system instructions
        formatted_prompt = f"You are Qwen, a helpful AI assistant created by Alibaba Cloud. {prompt}"
        super().set_system_prompt(formatted_prompt)
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Qwen-specific message formatting fallback.
        
        Args:
            messages (List[Dict[str, str]]): Messages to format
            
        Returns:
            str: Formatted text in Qwen chat format
        """
        # Use Qwen's chat format
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add generation prompt
        formatted_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Qwen-specific model information.
        
        Returns:
            Dict[str, Any]: Enhanced model information
        """
        info = super().get_model_info()
        info.update({
            "model_family": "Qwen",
            "provider": "Alibaba Cloud",
            "supported_variants": self.SUPPORTED_VARIANTS,
            "estimated_parameters": self._get_parameter_count(),
            "context_length": self._get_context_length(),
            "special_features": self._get_special_features(),
            "supports_vision": self._supports_vision()
        })
        return info
    
    def _get_parameter_count(self) -> str:
        """Get estimated parameter count based on model name."""
        model_id_lower = self.model_id.lower()
        
        if "32b" in model_id_lower:
            return "~32B parameters"
        elif "8b" in model_id_lower:
            return "~8B parameters"
        elif "7b" in model_id_lower:
            return "~7B parameters"
        elif "4b" in model_id_lower:
            return "~4B parameters"
        elif "0.6b" in model_id_lower:
            return "~0.6B parameters"
        else:
            return "Unknown"
    
    def _get_context_length(self) -> int:
        """Get context length based on model variant."""
        model_id_lower = self.model_id.lower()
        
        if "qwen3" in model_id_lower:
            return 131072  # Qwen3 context length
        elif "qwen2.5" in model_id_lower:
            return 32768   # Qwen2.5 context length
        else:
            return 8192    # Default/fallback
    
    def _get_special_features(self) -> List[str]:
        """Get special features of the model."""
        features = []
        model_id_lower = self.model_id.lower()
        
        if "vl" in model_id_lower:
            features.extend([
                "Multimodal capabilities",
                "Vision-language understanding",
                "Image analysis and description"
            ])
        
        if "qwen3" in model_id_lower:
            features.extend([
                "Latest generation",
                "Enhanced reasoning",
                "Improved multilingual support"
            ])
        
        if any(size in model_id_lower for size in ["0.6b", "4b"]):
            features.extend([
                "Efficient and lightweight",
                "Mobile/edge friendly",
                "Fast inference"
            ])
        
        # All Qwen models
        features.extend([
            "Multilingual support",
            "Strong Chinese language capabilities",
            "Good code generation",
            "Alibaba Cloud optimized"
        ])
        
        return features
    
    def _supports_vision(self) -> bool:
        """Check if the model supports vision capabilities."""
        return "vl" in self.model_id.lower()
    
    @classmethod
    def get_recommended_config(cls, model_variant: str = "4b") -> Dict[str, Any]:
        """
        Get recommended configuration for different Qwen variants.
        
        Args:
            model_variant (str): Model variant ("0.6b", "4b", "8b", "32b", "vl", etc.)
            
        Returns:
            Dict[str, Any]: Recommended configuration
        """
        model_variant = model_variant.lower()
        
        base_config = {
            "hf_model_path": "./models",
            "hf_token_path": "./tokens/hf_token.txt",
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 512
        }
        
        if "32b" in model_variant:
            base_config.update({
                "model_id": "Qwen/Qwen3-32B",
                "quantization": "4bit",
                "max_new_tokens": 1024
            })
        elif "8b" in model_variant:
            base_config.update({
                "model_id": "Qwen/Qwen3-8B",
                "quantization": "auto"
            })
        elif "4b" in model_variant:
            base_config.update({
                "model_id": "Qwen/Qwen3-4B",
                "quantization": "none"
            })
        elif "0.6b" in model_variant:
            base_config.update({
                "model_id": "Qwen/Qwen3-0.6B",
                "quantization": "none"
            })
        elif "vl" in model_variant:
            base_config.update({
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "quantization": "auto",
                "supports_vision": True
            })
        else:
            # Default to 4B
            base_config.update({
                "model_id": "Qwen/Qwen3-4B"
            })
        
        return base_config
    
    def enable_lora_finetuning(self, lora_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Enable LoRA fine-tuning for the Qwen model.
        
        Args:
            lora_config (Optional[Dict[str, Any]]): LoRA configuration
            
        Returns:
            Any: PEFT model ready for fine-tuning (future implementation)
        """
        if lora_config is None:
            # Use Qwen-optimized LoRA config
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
