"""
Aya model implementations.

This module provides the Aya class for CohereForAI Aya model variants,
including Aya Expanse series models.
"""

from typing import Dict, Any, List, Optional
from ..base.hf_base import HuggingFaceBase


class Aya(HuggingFaceBase):
    """
    Aya language model implementation.
    
    Supports CohereForAI Aya variants including:
    - CohereForAI/aya-expanse-8b
    - CohereForAI/aya-expanse-32b
    """
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        'CohereForAI/aya-expanse-8b',
        'CohereForAI/aya-expanse-32b'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Aya model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        # Validate model ID
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known Aya variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "CohereForAI/aya-expanse-8b"
            print(f"No model_id specified, using default: {config['model_id']}")
        
        # Apply Aya-specific optimizations
        self._apply_aya_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        
        print(f"Aya model initialized: {self.model_id}")
    
    def _apply_aya_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply Aya-specific optimizations to the configuration."""
        
        # Aya-specific generation defaults
        aya_defaults = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "max_new_tokens": 512
        }
        
        # Apply defaults if not already specified
        for key, default_value in aya_defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Model size-specific optimizations
        model_id = config.get("model_id", "").lower()
        
        if "32b" in model_id:
            # Large model optimizations
            if "quantization" not in config:
                config["quantization"] = "4bit"  # Default to 4-bit for 32B models
            print("Applied Aya-32B model optimizations (4-bit quantization)")
            
        elif "8b" in model_id:
            # Medium model optimizations
            if "quantization" not in config:
                config["quantization"] = "auto"  # Auto-select based on available memory
            print("Applied Aya-8B model optimizations (auto quantization)")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt with Aya-specific formatting.
        
        Args:
            prompt (str): The system prompt to use
        """
        # Aya models are designed for multilingual and multicultural responses
        formatted_prompt = f"You are Aya, a multilingual AI assistant created by Cohere For AI. You are helpful, respectful, and culturally aware. {prompt}"
        super().set_system_prompt(formatted_prompt)
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Aya-specific message formatting fallback.
        
        Args:
            messages (List[Dict[str, str]]): Messages to format
            
        Returns:
            str: Formatted text in Aya chat format
        """
        # Aya uses a straightforward chat format
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
        Get Aya-specific model information.
        
        Returns:
            Dict[str, Any]: Enhanced model information
        """
        info = super().get_model_info()
        info.update({
            "model_family": "Aya",
            "provider": "Cohere For AI",
            "supported_variants": self.SUPPORTED_VARIANTS,
            "estimated_parameters": self._get_parameter_count(),
            "context_length": self._get_context_length(),
            "special_features": self._get_special_features(),
            "supported_languages": self._get_supported_languages()
        })
        return info
    
    def _get_parameter_count(self) -> str:
        """Get estimated parameter count based on model name."""
        model_id_lower = self.model_id.lower()
        
        if "32b" in model_id_lower:
            return "~32B parameters"
        elif "8b" in model_id_lower:
            return "~8B parameters"
        else:
            return "Unknown"
    
    def _get_context_length(self) -> int:
        """Get context length based on model variant."""
        # Aya Expanse models typically have good context length
        return 8192  # 8K context length
    
    def _get_special_features(self) -> List[str]:
        """Get special features of the model."""
        features = [
            "Multilingual capabilities",
            "Multicultural awareness",
            "Global perspective",
            "Ethical AI principles",
            "Community-driven development",
            "Open research initiative",
            "Diverse training data",
            "Cultural sensitivity"
        ]
        
        model_id_lower = self.model_id.lower()
        
        if "expanse" in model_id_lower:
            features.extend([
                "Expanse series optimizations",
                "Enhanced multilingual performance",
                "Improved cultural understanding"
            ])
        
        if "32b" in model_id_lower:
            features.extend([
                "Large-scale multilingual processing",
                "Advanced cross-cultural reasoning"
            ])
        elif "8b" in model_id_lower:
            features.extend([
                "Efficient multilingual inference",
                "Balanced performance and resources"
            ])
        
        return features
    
    def _get_supported_languages(self) -> List[str]:
        """Get list of major supported languages."""
        return [
            "English", "Spanish", "French", "German", "Italian", "Portuguese",
            "Arabic", "Chinese", "Japanese", "Korean", "Hindi", "Bengali",
            "Russian", "Turkish", "Indonesian", "Vietnamese", "Thai", "Hebrew",
            "Swahili", "Yoruba", "Hausa", "Amharic", "Zulu", "Afrikaans",
            "Dutch", "Swedish", "Norwegian", "Danish", "Finnish", "Polish",
            "Czech", "Hungarian", "Romanian", "Bulgarian", "Serbian", "Croatian",
            "Slovak", "Slovenian", "Lithuanian", "Latvian", "Estonian",
            "Greek", "Ukrainian", "Urdu", "Persian", "Tamil", "Telugu",
            "Gujarati", "Kannada", "Malayalam", "Marathi", "Punjabi", "Nepali"
        ]
    
    @classmethod
    def get_recommended_config(cls, model_variant: str = "8b") -> Dict[str, Any]:
        """
        Get recommended configuration for different Aya variants.
        
        Args:
            model_variant (str): Model variant ("8b", "32b", etc.)
            
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
        
        if "32b" in model_variant:
            base_config.update({
                "model_id": "CohereForAI/aya-expanse-32b",
                "quantization": "4bit",
                "max_new_tokens": 1024
            })
        else:
            # Default to 8B
            base_config.update({
                "model_id": "CohereForAI/aya-expanse-8b",
                "quantization": "auto"
            })
        
        return base_config
    
    def enable_lora_finetuning(self, lora_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Enable LoRA fine-tuning for the Aya model.
        
        Args:
            lora_config (Optional[Dict[str, Any]]): LoRA configuration
            
        Returns:
            Any: PEFT model ready for fine-tuning (future implementation)
        """
        if lora_config is None:
            # Use Aya-optimized LoRA config
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,  # Slightly higher for multilingual stability
                "target_modules": [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "down_proj", "up_proj"
                ],
                "task_type": "CAUSAL_LM"
            }
        
        return self.prepare_for_lora(lora_config)
    
    def get_multilingual_recommendations(self) -> Dict[str, str]:
        """
        Get recommendations for using Aya models for multilingual tasks.
        
        Returns:
            Dict[str, str]: Multilingual-specific recommendations
        """
        return {
            "language_specification": "Specify the desired output language in your prompt",
            "cultural_context": "Provide cultural context when relevant for better responses",
            "code_switching": "Aya can handle code-switching between languages naturally",
            "translation": "Excellent for translation tasks between supported languages",
            "cultural_sensitivity": "Model is trained to be culturally aware and respectful",
            "best_practices": "Use clear language indicators and respect cultural nuances",
            "performance": "Performance may vary across languages - test with your specific use case"
        }
