"""
Llama model implementations.

This module provides the Llama class for all Llama model variants,
including Llama 3.1, 3.2, and 3.3 series with optimized configurations.
"""

from typing import Dict, Any, List, Optional
from ..base.hf_base import HuggingFaceBase


class Llama(HuggingFaceBase):
    """
    Llama language model implementation.
    
    Supports all Llama variants including:
    - Llama-3.1-8B/8B-Instruct
    - Llama-3.2-1B/1B-Instruct  
    - Llama-3.2-3B/3B-Instruct
    - Llama-3.3-70B-Instruct
    """
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        'meta-llama/Llama-3.1-8B',
        'meta-llama/Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.2-1B',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.3-70B-Instruct'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Llama model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        # Validate model ID
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known Llama variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "meta-llama/Llama-3.2-3B-Instruct"
            print(f"No model_id specified, using default: {config['model_id']}")
        
        # Apply Llama-specific optimizations
        self._apply_llama_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        
        print(f"Llama model initialized: {self.model_id}")
    
    def _apply_llama_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply Llama-specific optimizations to the configuration."""
        
        # Llama-specific generation defaults
        llama_defaults = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": 512
        }
        
        # Apply defaults if not already specified
        for key, default_value in llama_defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Model size-specific optimizations
        model_id = config.get("model_id", "")
        
        if "70b" in model_id.lower():
            # Large model optimizations
            if "quantization" not in config:
                config["quantization"] = "4bit"  # Default to 4-bit for 70B models
            print("Applied 70B model optimizations (4-bit quantization)")
            
        elif any(size in model_id.lower() for size in ["8b", "7b"]):
            # Medium model optimizations
            if "quantization" not in config:
                config["quantization"] = "auto"  # Auto-select based on available memory
            print("Applied medium model optimizations (auto quantization)")
            
        elif any(size in model_id.lower() for size in ["3b", "1b"]):
            # Small model optimizations
            if "quantization" not in config:
                config["quantization"] = "none"  # Usually no quantization needed
            print("Applied small model optimizations")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt with Llama-specific formatting.
        
        Args:
            prompt (str): The system prompt to use
        """
        # Llama models work well with clear system instructions
        formatted_prompt = f"You are a helpful, harmless, and honest assistant. {prompt}"
        super().set_system_prompt(formatted_prompt)
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Llama-specific message formatting fallback.
        
        Args:
            messages (List[Dict[str, str]]): Messages to format
            
        Returns:
            str: Formatted text in Llama chat format
        """
        # Use Llama's chat format
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "user":
                formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant":
                formatted_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        # Add generation prompt
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(formatted_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Llama-specific model information.
        
        Returns:
            Dict[str, Any]: Enhanced model information
        """
        info = super().get_model_info()
        info.update({
            "model_family": "Llama",
            "supported_variants": self.SUPPORTED_VARIANTS,
            "is_instruct_model": "instruct" in self.model_id.lower(),
            "estimated_parameters": self._get_parameter_count(),
            "context_length": self._get_context_length(),
            "special_features": self._get_special_features()
        })
        return info
    
    def _get_parameter_count(self) -> str:
        """Get estimated parameter count based on model name."""
        model_id_lower = self.model_id.lower()
        
        if "70b" in model_id_lower:
            return "~70B parameters"
        elif "8b" in model_id_lower:
            return "~8B parameters"
        elif "3b" in model_id_lower:
            return "~3B parameters"
        elif "1b" in model_id_lower:
            return "~1B parameters"
        else:
            return "Unknown"
    
    def _get_context_length(self) -> int:
        """Get context length based on model variant."""
        model_id_lower = self.model_id.lower()
        
        if "3.3" in model_id_lower:
            return 131072  # Llama 3.3 has longer context
        elif "3.2" in model_id_lower or "3.1" in model_id_lower:
            return 131072  # Llama 3.1/3.2 context length
        else:
            return 4096  # Default/fallback
    
    def _get_special_features(self) -> List[str]:
        """Get special features of the model."""
        features = []
        model_id_lower = self.model_id.lower()
        
        if "instruct" in model_id_lower:
            features.append("Instruction-tuned")
            features.append("Chat optimized")
        
        if "3.3" in model_id_lower:
            features.append("Latest generation")
            features.append("Enhanced reasoning")
        
        if "3.2" in model_id_lower:
            features.append("Lightweight variants")
            features.append("Efficient inference")
        
        if any(size in model_id_lower for size in ["1b", "3b"]):
            features.append("Mobile/edge friendly")
            features.append("Low resource requirements")
        
        return features
    
    @classmethod
    def get_recommended_config(cls, model_variant: str = "3b-instruct") -> Dict[str, Any]:
        """
        Get recommended configuration for different Llama variants.
        
        Args:
            model_variant (str): Model variant ("1b", "3b", "8b", "70b", etc.)
            
        Returns:
            Dict[str, Any]: Recommended configuration
        """
        model_variant = model_variant.lower()
        
        base_config = {
            "hf_model_path": "./models",
            "hf_token_path": "./tokens/hf_token.txt",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512
        }
        
        if "70b" in model_variant:
            base_config.update({
                "model_id": "meta-llama/Llama-3.3-70B-Instruct",
                "quantization": "4bit",
                "max_new_tokens": 1024
            })
        elif "8b" in model_variant:
            base_config.update({
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "quantization": "auto"
            })
        elif "3b" in model_variant:
            base_config.update({
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                "quantization": "none"
            })
        elif "1b" in model_variant:
            base_config.update({
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "quantization": "none"
            })
        else:
            # Default to 3B instruct
            base_config.update({
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                "quantization": "auto"
            })
        
        return base_config
    
    def enable_lora_finetuning(self, lora_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Enable LoRA fine-tuning for the Llama model.
        
        Args:
            lora_config (Optional[Dict[str, Any]]): LoRA configuration
            
        Returns:
            Any: PEFT model ready for fine-tuning (future implementation)
        """
        if lora_config is None:
            # Use Llama-optimized LoRA config
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
