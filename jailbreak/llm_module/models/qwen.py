"""
Qwen model implementations.

This module provides the Qwen class for all Qwen model variants,
including Qwen2.5, Qwen3, and Qwen-VL series models.
"""

from typing import Dict, Any, List, Optional, Union
from PIL import Image
from ..base.hf_base import HuggingFaceBase
from ..base.vl_base import VisionLanguageBase


def Qwen(config: Dict[str, Any]):
    """
    Factory function for Qwen models.
    
    Automatically selects the appropriate base class based on model type:
    - VisionLanguageBase for VL models (Qwen-VL, Qwen3-VL)
    - HuggingFaceBase for text-only models
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters
        
    Returns:
        QwenVL or QwenText: Appropriate Qwen model instance
    """
    model_id = config.get("model_id", "").lower()
    
    # Check if it's a Vision-Language model
    if "vl" in model_id or config.get("supports_vision", False):
        return QwenVL(config)
    else:
        return QwenText(config)


class QwenBase:
    """Base class for shared Qwen functionality."""
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        # Text-only models
        'Qwen/Qwen2.5-0.5B-Instruct',
        'Qwen/Qwen2.5-1.5B-Instruct',
        'Qwen/Qwen2.5-3B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct',
        'Qwen/Qwen2.5-14B-Instruct',
        'Qwen/Qwen2.5-32B-Instruct',
        # Vision-Language models
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'Qwen/Qwen3-VL-4B-Instruct',
        'Qwen/Qwen3-VL-8B-Instruct',
        # Legacy (for backward compatibility)
        'Qwen/Qwen3-4B',
        'Qwen/Qwen3-8B', 
        'Qwen/Qwen3-32B',
        'Qwen/Qwen3-0.6B'
    ]
    
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
            
        elif any(size in model_id for size in ["8b", "7b", "14b"]):
            # Medium model optimizations
            if "quantization" not in config:
                config["quantization"] = "auto"  # Auto-select based on available memory
            print("Applied Qwen medium model optimizations (auto quantization)")
            
        elif any(size in model_id for size in ["4b", "3b", "1.5b", "0.6b", "0.5b"]):
            # Small model optimizations
            if "quantization" not in config:
                config["quantization"] = "none"  # Usually no quantization needed
            print("Applied Qwen small model optimizations")
        
        # Vision model specific settings
        if "vl" in model_id:
            config["supports_vision"] = True
            print("Enabled vision capabilities for Qwen-VL model")
    
    def _validate_model_id(self, config: Dict[str, Any]) -> None:
        """Validate and set default model ID."""
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known Qwen variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "Qwen/Qwen2.5-3B-Instruct"
            print(f"No model_id specified, using default: {config['model_id']}")

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


class QwenText(HuggingFaceBase, QwenBase):
    """Text-only Qwen model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize text-only Qwen model."""
        # Validate and optimize config
        self._validate_model_id(config)
        self._apply_qwen_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        print(f"Qwen text model initialized: {self.model_id}")


class QwenVL(VisionLanguageBase, QwenBase):
    """Vision-Language Qwen model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize VL Qwen model."""
        # Validate and optimize config
        self._validate_model_id(config)
        self._apply_qwen_optimizations(config)
        
        # Set default VL model if not specified
        if not config.get("model_id"):
            config["model_id"] = "Qwen/Qwen2.5-VL-7B-Instruct"
            print(f"No VL model_id specified, using default: {config['model_id']}")
        
        # Initialize parent class
        super().__init__(config)
        print(f"Qwen VL model initialized: {self.model_id}")
    
    def _format_vl_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Qwen-VL models using Qwen's chat format."""
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
    
    def process_image_with_text(self, image: Union[str, Image.Image], text: str) -> str:
        """
        Process image with text using Qwen-VL.
        
        Args:
            image: Image file path, URL, or PIL Image
            text: Text prompt for the image
            
        Returns:
            str: Model response
        """
        # Enhanced prompt for Qwen-VL
        enhanced_text = f"Please analyze this image and answer: {text}"
        messages = [{"role": "user", "content": enhanced_text}]
        return self.forward(messages, images=[image])


# For backward compatibility, create aliases
QwenModel = Qwen  # Factory function
