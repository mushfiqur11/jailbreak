"""
LLM Module for Jailbreak Framework

A comprehensive module providing unified access to both HuggingFace and OpenAI language models
with support for quantization, conversation management, and future LoRA fine-tuning.

Usage:
    from llm_module.models import Llama, GPT, Phi, Qwen
    
    # Initialize with configuration  
    config = {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "quantization": "4bit"}
    llm = Llama(config)
    
    # Set system prompt and generate responses
    llm.set_system_prompt("You are a helpful assistant.")
    llm.add_message("user", "Hello!")
    result = llm.forward()
    response = result['response']  # Extract response from metrics dict
    # result also contains: input_tokens, output_tokens, generation_time
"""

__version__ = "1.0.0"
__author__ = "Jailbreak Framework Team"

# Import main classes for easy access
from .models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
from .config import ModelConfigs, ConfigManager
from .utils import ConversationManager, ModelUtils
from .quantization import QuantizationConfig, MemoryOptimizer
from .finetuning import LoRAConfig, LoRATrainer
from .model_factory import llm_model_factory

# Main exports
__all__ = [
    # Model classes
    'Llama', 'GPT', 'Phi', 'Qwen', 'DeepSeek', 'Aya',
    
    # Model factory
    'llm_model_factory',
    
    # Configuration management
    'ModelConfigs', 'ConfigManager',
    
    # Utilities
    'ConversationManager', 'ModelUtils',
    
    # Quantization
    'QuantizationConfig', 'MemoryOptimizer',
    
    # Fine-tuning
    'LoRAConfig', 'LoRATrainer'
]
