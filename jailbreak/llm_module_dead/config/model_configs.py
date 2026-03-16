"""
Default model configurations for different language models.

This module provides pre-configured settings optimized for different model types
and use cases, making it easy to get started with various language models.
"""

from typing import Dict, Any


class ModelConfigs:
    """
    Default configurations for different language model families.
    
    This class provides optimized configurations for various models,
    including quantization settings, generation parameters, and paths.
    """
    
    # Base configuration template
    BASE_CONFIG = {
        # hf_model_path removed - will use new default: github_repos/jailbreak/hf_models/
        "hf_token_path": "./tokens/hf_token.txt",
        "api_key_path": "./tokens/openai_key.txt",
        "temperature": 0.7,
        "max_new_tokens": 512,
        "top_p": 0.9,
        "quantization": "auto"
    }
    
    # Llama model configurations
    LLAMA_CONFIGS = {
        "llama-1b": {
            **BASE_CONFIG,
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "quantization": "none",
            "max_new_tokens": 512
        },
        "llama-3b": {
            **BASE_CONFIG,
            "model_id": "meta-llama/Llama-3.2-3B-Instruct", 
            "quantization": "none",
            "max_new_tokens": 512
        },
        "llama-8b": {
            **BASE_CONFIG,
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "quantization": "auto",
            "max_new_tokens": 1024
        },
        "llama-70b": {
            **BASE_CONFIG,
            "model_id": "meta-llama/Llama-3.3-70B-Instruct",
            "quantization": "4bit",
            "max_new_tokens": 1024,
            "temperature": 0.6
        }
    }
    
    # GPT model configurations
    GPT_CONFIGS = {
        "gpt-3.5-turbo": {
            **BASE_CONFIG,
            "model_id": "gpt-3.5-turbo",
            "max_tokens": 512,
            "temperature": 0.8
        },
        "gpt-4o": {
            **BASE_CONFIG,
            "model_id": "gpt-4o",
            "max_tokens": 1024,
            "temperature": 0.7
        },
        "o3-mini": {
            **BASE_CONFIG,
            "model_id": "o3-mini",
            "max_tokens": 1024,
            "temperature": 0.3  # Lower for reasoning
        }
    }
    
    # Phi model configurations  
    PHI_CONFIGS = {
        "phi-3-mini": {
            **BASE_CONFIG,
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "quantization": "none",
            "temperature": 0.6
        },
        "phi-4": {
            **BASE_CONFIG,
            "model_id": "microsoft/phi-4",
            "quantization": "auto",
            "temperature": 0.6,
            "max_new_tokens": 1024
        }
    }
    
    # Qwen model configurations (using text-only models compatible with AutoModelForCausalLM)
    QWEN_CONFIGS = {
        "qwen-0.5b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "quantization": "none",
            "temperature": 0.7,
            "top_p": 0.8
        },
        "qwen-1.5b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "quantization": "none",
            "temperature": 0.7,
            "top_p": 0.8
        },
        "qwen-3b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "quantization": "none",
            "temperature": 0.7,
            "top_p": 0.8
        },
        "qwen-7b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "auto",
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 1024
        },
        "qwen-14b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "quantization": "auto",
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 1024
        },
        "qwen-32b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-32B-Instruct", 
            "quantization": "4bit",
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 1024
        },
        "qwen-vl": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "quantization": "auto",
            "supports_vision": True,
            "note": "Vision-Language model with image understanding"
        },
        "qwen3-vl-4b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "quantization": "none",
            "supports_vision": True,
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "note": "Qwen3 Vision-Language 4B model"
        },
        "qwen3-vl-8b": {
            **BASE_CONFIG,
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "quantization": "auto",
            "supports_vision": True,
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "note": "Qwen3 Vision-Language 8B model"
        }
    }
    
    # DeepSeek model configurations
    DEEPSEEK_CONFIGS = {
        "deepseek-1.5b": {
            **BASE_CONFIG,
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "quantization": "none",
            "temperature": 0.3,  # Lower for reasoning
            "max_new_tokens": 1024
        },
        "deepseek-14b": {
            **BASE_CONFIG,
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "quantization": "auto",
            "temperature": 0.3,
            "max_new_tokens": 1024
        },
        "deepseek-32b": {
            **BASE_CONFIG,
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "quantization": "4bit",
            "temperature": 0.3,
            "max_new_tokens": 1024
        },
        "deepseek-70b": {
            **BASE_CONFIG,
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "quantization": "4bit",
            "temperature": 0.3,
            "max_new_tokens": 2048
        }
    }
    
    # Aya model configurations
    AYA_CONFIGS = {
        "aya-8b": {
            **BASE_CONFIG,
            "model_id": "CohereForAI/aya-expanse-8b",
            "quantization": "auto",
            "temperature": 0.7
        },
        "aya-32b": {
            **BASE_CONFIG,
            "model_id": "CohereForAI/aya-expanse-32b",
            "quantization": "4bit",
            "temperature": 0.7,
            "max_new_tokens": 1024
        }
    }

    # Gemma model configurations
    GEMMA_CONFIGS = {
        "gemma-1b": {
            **BASE_CONFIG,
            "model_id": "google/gemma-3-1b-it",
            "quantization": "none",
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "gemma-4b": {
            **BASE_CONFIG,
            "model_id": "google/gemma-3-4b-it",
            "quantization": "none",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 1024,
        },
        "gemma-27b": {
            **BASE_CONFIG,
            "model_id": "google/gemma-3-27b-it",
            "quantization": "4bit",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 1024,
        },
    }

    # GPT-OSS open-weight HF model configurations
    GPTOSS_CONFIGS = {
        "gpt-oss-20b": {
            **BASE_CONFIG,
            "model_id": "openai/gpt-oss-20b",
            "quantization": "4bit",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 1024,
        },
    }
    
    # Task-specific configurations
    TASK_CONFIGS = {
        "jailbreaking": {
            **BASE_CONFIG,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.05
        },
        "reasoning": {
            **BASE_CONFIG,
            "temperature": 0.3,
            "top_p": 0.85,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.1
        },
        "creative": {
            **BASE_CONFIG,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.0
        },
        "analytical": {
            **BASE_CONFIG,
            "temperature": 0.2,
            "top_p": 0.8,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.15
        }
    }
    
    # Combined configurations
    ALL_CONFIGS = {
        **LLAMA_CONFIGS,
        **GPT_CONFIGS,
        **PHI_CONFIGS,
        **QWEN_CONFIGS,
        **DEEPSEEK_CONFIGS,
        **AYA_CONFIGS,
        **GEMMA_CONFIGS,
        **GPTOSS_CONFIGS,
        **TASK_CONFIGS
    }
    
    @classmethod
    def get_config(cls, config_name: str) -> Dict[str, Any]:
        """
        Get a configuration by name.
        
        Args:
            config_name (str): Name of the configuration
            
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            KeyError: If configuration name is not found
        """
        if config_name not in cls.ALL_CONFIGS:
            available_configs = list(cls.ALL_CONFIGS.keys())
            raise KeyError(f"Configuration '{config_name}' not found. Available: {available_configs}")
        
        return cls.ALL_CONFIGS[config_name].copy()
    
    @classmethod
    def list_configs(cls) -> Dict[str, list]:
        """
        List all available configurations by category.
        
        Returns:
            Dict[str, list]: Dictionary of configuration categories and their names
        """
        return {
            "llama": list(cls.LLAMA_CONFIGS.keys()),
            "gpt": list(cls.GPT_CONFIGS.keys()),
            "phi": list(cls.PHI_CONFIGS.keys()),
            "qwen": list(cls.QWEN_CONFIGS.keys()),
            "deepseek": list(cls.DEEPSEEK_CONFIGS.keys()),
            "aya": list(cls.AYA_CONFIGS.keys()),
            "gemma": list(cls.GEMMA_CONFIGS.keys()),
            "gptoss": list(cls.GPTOSS_CONFIGS.keys()),
            "tasks": list(cls.TASK_CONFIGS.keys())
        }
    
    @classmethod
    def get_recommended_config_for_task(cls, task: str, model_family: str = "llama") -> Dict[str, Any]:
        """
        Get a recommended configuration for a specific task and model family.
        
        Args:
            task (str): Task type ("jailbreaking", "reasoning", "creative", "analytical")
            model_family (str): Model family ("llama", "gpt", "phi", etc.)
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        # Get base task config
        task_config = cls.TASK_CONFIGS.get(task, cls.BASE_CONFIG).copy()
        
        # Get default model config for family
        family_configs = {
            "llama": cls.LLAMA_CONFIGS,
            "gpt": cls.GPT_CONFIGS, 
            "phi": cls.PHI_CONFIGS,
            "qwen": cls.QWEN_CONFIGS,
            "deepseek": cls.DEEPSEEK_CONFIGS,
            "aya": cls.AYA_CONFIGS,
            "gemma": cls.GEMMA_CONFIGS,
            "gptoss": cls.GPTOSS_CONFIGS,
        }
        
        if model_family in family_configs:
            family_config = list(family_configs[model_family].values())[0].copy()
            
            # Merge task-specific overrides
            family_config.update({
                k: v for k, v in task_config.items() 
                if k in ["temperature", "top_p", "max_new_tokens", "repetition_penalty"]
            })
            
            return family_config
        
        return task_config
    
    @classmethod
    def get_memory_efficient_config(cls, model_name: str, available_memory_gb: float = 16) -> Dict[str, Any]:
        """
        Get a memory-efficient configuration based on available memory.
        
        Args:
            model_name (str): Model configuration name
            available_memory_gb (float): Available memory in GB
            
        Returns:
            Dict[str, Any]: Memory-optimized configuration
        """
        config = cls.get_config(model_name).copy()
        
        # Adjust quantization based on available memory
        if available_memory_gb < 8:
            config["quantization"] = "4bit"
        elif available_memory_gb < 16:
            config["quantization"] = "8bit"
        elif available_memory_gb < 32:
            config["quantization"] = "auto"
        else:
            config["quantization"] = "none"
        
        # Reduce token limits for memory constrained systems
        if available_memory_gb < 16:
            config["max_new_tokens"] = min(config.get("max_new_tokens", 512), 512)
            config["max_tokens"] = min(config.get("max_tokens", 512), 512)
        
        return config
