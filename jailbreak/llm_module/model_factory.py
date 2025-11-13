"""
Simple model factory for creating LLM instances based on configuration.

This module provides a unified interface to instantiate different language models
based on configuration dictionaries with explicit model specification.
"""

from typing import Dict, Any
from .base.base_llm import BaseLLM


def llm_model_factory(config: Dict[str, Any]) -> BaseLLM:
    """
    Create an LLM instance based on the provided configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - model (str): Required. Model type (llama, gpt, phi, qwen, deepseek, aya)
            - model_id (str): Required. Actual model identifier
            - Additional model-specific parameters (temperature, max_tokens, etc.)
    
    Returns:
        BaseLLM: Instantiated model object of the appropriate type
        
    Raises:
        ValueError: If required fields are missing or model type is unsupported
        
    Example:
        >>> config = {
        ...     "model": "qwen",
        ...     "model_id": "Qwen/Qwen3-0.6B",
        ...     "quantization": "none",
        ...     "temperature": 0.7,
        ...     "max_new_tokens": 512,
        ...     "hf_token_path": "/home/mrahma45/HUGGINGFACE_KEY"
        ... }
        >>> llm = llm_model_factory(config)
    """
    # Validate input
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    if "model" not in config:
        raise ValueError("Configuration must include 'model' field")
    
    if "model_id" not in config:
        raise ValueError("Configuration must include 'model_id' field")
    
    model_type = config["model"].lower().strip()
    
    # Dynamic import and instantiation based on model type
    if model_type == "llama":
        from .models.llama import Llama
        return Llama(config)
    
    elif model_type == "gpt":
        from .models.gpt import GPT
        return GPT(config)
    
    elif model_type == "phi":
        from .models.phi import Phi
        return Phi(config)
    
    elif model_type == "qwen":
        from .models.qwen import Qwen
        return Qwen(config)
    
    elif model_type == "deepseek":
        from .models.deepseek import DeepSeek
        return DeepSeek(config)
    
    elif model_type == "aya":
        from .models.aya import Aya
        return Aya(config)
    
    else:
        supported_models = ["llama", "gpt", "phi", "qwen", "deepseek", "aya"]
        raise ValueError(f"Unsupported model: '{model_type}'. Supported models: {supported_models}")
