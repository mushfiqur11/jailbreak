"""
Jailbreak Framework

A comprehensive toolkit for AI security research and jailbreaking experiments.

This package combines:
- llm_module: Unified interface for language model management
- agentic_jailbreak: Multi-agent framework for jailbreaking research
"""

__version__ = "1.0.0"
__author__ = "Jailbreak Framework Team"

# Import key components for easy access
try:
    from .llm_module.models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
    from .llm_module.config import ModelConfigs, ConfigManager
    from .llm_module.utils import ConversationManager, ModelUtils
    from .llm_module.quantization import QuantizationConfig, MemoryOptimizer
    from .llm_module.finetuning import LoRAConfig, LoRATrainer
    
    # Main exports - easy access to key functionality
    __all__ = [
        # Model classes
        'Llama', 'GPT', 'Phi', 'Qwen', 'DeepSeek', 'Aya',
        
        # Configuration management  
        'ModelConfigs', 'ConfigManager',
        
        # Utilities
        'ConversationManager', 'ModelUtils',
        
        # Quantization
        'QuantizationConfig', 'MemoryOptimizer',
        
        # Fine-tuning
        'LoRAConfig', 'LoRATrainer'
    ]
    
except ImportError as e:
    # If some modules aren't available, provide a helpful message
    print(f"Warning: Some jailbreak components not available: {e}")
    __all__ = []

# Version information
def get_version():
    """Get the current version of the jailbreak framework."""
    return __version__

def get_info():
    """Get information about the jailbreak framework."""
    return {
        "name": "jailbreak",
        "version": __version__,
        "author": __author__,
        "description": "Complete toolkit for AI security research and jailbreaking experiments",
        "components": {
            "llm_module": "Language model management system",
            "agentic_jailbreak": "Multi-agent jailbreaking framework"
        }
    }
