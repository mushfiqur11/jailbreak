"""
LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning.

This module provides configuration classes and utilities for setting up
LoRA fine-tuning with specific optimizations for different model types.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Task types supported by LoRA."""
    CAUSAL_LM = "CAUSAL_LM"
    SEQ_CLS = "SEQ_CLS"
    TOKEN_CLS = "TOKEN_CLS" 
    QUESTION_ANS = "QUESTION_ANS"


@dataclass
class LoRAConfig:
    """
    Configuration class for LoRA fine-tuning.
    
    This class defines all parameters needed for LoRA adaptation,
    including rank, alpha, dropout, and target modules.
    """
    
    # Core LoRA parameters
    r: int = 16                              # Rank of adaptation
    lora_alpha: int = 32                     # LoRA scaling parameter
    lora_dropout: float = 0.05               # Dropout probability
    
    # Target modules and task configuration
    target_modules: List[str] = None         # Modules to adapt
    task_type: TaskType = TaskType.CAUSAL_LM # Task type
    
    # Advanced configuration
    fan_in_fan_out: bool = False             # Set to True for Conv1D layers
    bias: str = "none"                       # Bias type ("none", "all", "lora_only")
    use_rslora: bool = False                 # Use rank stabilized LoRA
    use_dora: bool = False                   # Use DoRA (Weight-Decomposed Low-Rank Adaptation)
    
    # Training configuration
    init_lora_weights: Union[bool, str] = True  # How to initialize LoRA weights
    revision: str = "main"                   # Model revision to use
    
    # Memory and performance
    loftq_config: Optional[Dict[str, Any]] = None  # LoftQ quantization config
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.target_modules is None:
            # Default target modules for transformer models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Validate parameters
        if self.r <= 0:
            raise ValueError("Rank (r) must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError("LoRA dropout must be between 0.0 and 1.0")
        
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError("Bias must be one of: none, all, lora_only")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        config_dict = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "task_type": self.task_type.value,
            "fan_in_fan_out": self.fan_in_fan_out,
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "init_lora_weights": self.init_lora_weights,
            "revision": self.revision
        }
        
        if self.loftq_config:
            config_dict["loftq_config"] = self.loftq_config
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """Create configuration from dictionary."""
        # Convert task_type string back to enum
        if "task_type" in config_dict and isinstance(config_dict["task_type"], str):
            config_dict["task_type"] = TaskType(config_dict["task_type"])
        
        return cls(**config_dict)
    
    @classmethod
    def get_model_optimized_config(cls, model_family: str, model_size: str = "7b") -> 'LoRAConfig':
        """
        Get LoRA configuration optimized for a specific model family.
        
        Args:
            model_family (str): Model family ("llama", "phi", "qwen", etc.)
            model_size (str): Model size ("1b", "7b", "70b", etc.)
            
        Returns:
            LoRAConfig: Optimized LoRA configuration
        """
        # Base configuration
        config = cls()
        
        # Family-specific optimizations
        if model_family.lower() == "llama":
            config.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
            config.r = 16
            config.lora_alpha = 32
            
        elif model_family.lower() == "phi":
            config.target_modules = ["q_proj", "v_proj", "dense"]
            config.r = 8
            config.lora_alpha = 16
            
        elif model_family.lower() == "qwen":
            config.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
            config.r = 16
            config.lora_alpha = 32
            
        elif model_family.lower() == "deepseek":
            config.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
            config.r = 16
            config.lora_alpha = 32
            config.lora_dropout = 0.1  # Higher dropout for reasoning models
            
        elif model_family.lower() == "aya":
            config.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
            config.r = 16
            config.lora_alpha = 32
            config.lora_dropout = 0.1  # Higher dropout for multilingual stability
        
        # Size-specific adjustments
        if "70b" in model_size.lower() or "32b" in model_size.lower():
            # Large models can benefit from higher rank
            config.r = min(config.r * 2, 64)
            config.lora_alpha = config.r * 2
        elif "1b" in model_size.lower() or "3b" in model_size.lower():
            # Small models need lower rank to avoid overfitting
            config.r = max(config.r // 2, 4)
            config.lora_alpha = config.r * 2
        
        return config
    
    @classmethod
    def get_task_optimized_config(cls, task: str, base_model_family: str = "llama") -> 'LoRAConfig':
        """
        Get LoRA configuration optimized for a specific task.
        
        Args:
            task (str): Task type ("jailbreaking", "reasoning", "creative", etc.)
            base_model_family (str): Base model family for module targeting
            
        Returns:
            LoRAConfig: Task-optimized LoRA configuration
        """
        # Start with model-optimized config
        config = cls.get_model_optimized_config(base_model_family)
        
        # Task-specific adjustments
        if task.lower() == "jailbreaking":
            # Jailbreaking benefits from moderate adaptation
            config.r = 16
            config.lora_alpha = 32
            config.lora_dropout = 0.05
            
        elif task.lower() == "reasoning":
            # Reasoning tasks benefit from higher rank
            config.r = 32
            config.lora_alpha = 64
            config.lora_dropout = 0.1
            
        elif task.lower() == "creative":
            # Creative tasks benefit from diverse adaptation
            config.r = 24
            config.lora_alpha = 48
            config.lora_dropout = 0.05
            
        elif task.lower() == "code":
            # Code tasks benefit from precise adaptation
            config.r = 16
            config.lora_alpha = 32
            config.lora_dropout = 0.0  # No dropout for code
            
        elif task.lower() == "multilingual":
            # Multilingual tasks need stable adaptation
            config.r = 16
            config.lora_alpha = 32
            config.lora_dropout = 0.1
        
        return config
    
    def estimate_parameters(self) -> Dict[str, int]:
        """
        Estimate the number of trainable parameters for this LoRA config.
        
        Returns:
            Dict[str, int]: Parameter count estimates
        """
        # This is a rough estimate - actual parameters depend on model architecture
        params_per_module = self.r * 2  # Assuming square matrices, rough estimate
        total_modules = len(self.target_modules)
        
        # Rough estimates (would need actual model inspection for precision)
        estimated_params = params_per_module * total_modules * 4096  # Rough hidden size
        
        return {
            "rank": self.r,
            "target_modules": total_modules,
            "estimated_trainable_params": estimated_params,
            "estimated_memory_mb": estimated_params * 4 / (1024 * 1024)  # 4 bytes per param
        }
    
    def get_compatibility_info(self) -> Dict[str, Any]:
        """
        Get compatibility information for this configuration.
        
        Returns:
            Dict[str, Any]: Compatibility information
        """
        return {
            "supports_quantization": True,
            "supports_gradient_checkpointing": True,
            "supports_mixed_precision": True,
            "memory_efficient": self.r <= 32,
            "recommended_batch_size": max(1, 16 // (self.r // 8 + 1)),
            "requires_peft": True,
            "requires_transformers_version": ">=4.21.0"
        }


class PresetConfigs:
    """
    Preset LoRA configurations for common use cases.
    
    This class provides pre-configured LoRA setups optimized
    for different scenarios and model types.
    """
    
    # Memory-efficient configurations
    MEMORY_EFFICIENT = LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Balanced configurations
    BALANCED = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # High-performance configurations
    HIGH_PERFORMANCE = LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ]
    )
    
    # Task-specific presets
    JAILBREAKING = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    REASONING = LoRAConfig(
        r=24,
        lora_alpha=48,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj"
        ]
    )
    
    MULTILINGUAL = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ]
    )
    
    @classmethod
    def get_preset(cls, preset_name: str) -> LoRAConfig:
        """
        Get a preset configuration by name.
        
        Args:
            preset_name (str): Name of the preset
            
        Returns:
            LoRAConfig: Preset configuration
        """
        preset_name = preset_name.upper()
        if hasattr(cls, preset_name):
            return getattr(cls, preset_name)
        else:
            available = [attr for attr in dir(cls) if not attr.startswith('_') and attr != 'get_preset']
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available preset names."""
        return [attr for attr in dir(cls) if not attr.startswith('_') and attr != 'get_preset' and attr != 'list_presets']
