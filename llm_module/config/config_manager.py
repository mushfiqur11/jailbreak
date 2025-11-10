"""
Configuration manager for loading, validating, and managing model configurations.

This module provides utilities to load configurations from files, validate them,
and merge configurations with defaults.
"""

import json
import os
from typing import Dict, Any, Optional, Union
from .model_configs import ModelConfigs


class ConfigManager:
    """
    Configuration manager for handling model configurations.
    
    This class provides methods to load configurations from files,
    validate them, and merge with default configurations.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir (Optional[str]): Directory containing configuration files
        """
        self.config_dir = config_dir or "./configs"
        
    def load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file {config_path}: {e}")
    
    def save_config_to_file(self, config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config (Dict[str, Any]): Configuration to save
            config_path (str): Path where to save the configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a configuration.
        
        Args:
            config (Dict[str, Any]): Configuration to validate
            
        Returns:
            Dict[str, Any]: Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Required fields
        if "model_id" not in config:
            raise ValueError("Configuration must include 'model_id'")
        
        # Validate quantization setting
        valid_quantization = ["none", "auto", "4bit", "8bit"]
        quantization = config.get("quantization", "auto")
        if quantization not in valid_quantization:
            raise ValueError(f"Invalid quantization: {quantization}. Must be one of {valid_quantization}")
        
        # Validate temperature
        temperature = config.get("temperature", 0.7)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"Invalid temperature: {temperature}. Must be between 0.0 and 2.0")
        
        # Validate top_p
        top_p = config.get("top_p", 0.9)
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"Invalid top_p: {top_p}. Must be between 0.0 and 1.0")
        
        # Validate token limits
        max_new_tokens = config.get("max_new_tokens", 512)
        max_tokens = config.get("max_tokens", 512)
        if max_new_tokens <= 0 or max_tokens <= 0:
            raise ValueError("Token limits must be positive integers")
        
        # Validate paths exist if specified
        for path_key in ["hf_token_path", "api_key_path"]:
            if path_key in config and config[path_key]:
                path = config[path_key]
                if not os.path.exists(path):
                    print(f"Warning: {path_key} file not found: {path}")
        
        return config
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with override_config taking precedence.
        
        Args:
            base_config (Dict[str, Any]): Base configuration
            override_config (Dict[str, Any]): Override configuration
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    
    def get_config(self, 
                   config_source: Union[str, Dict[str, Any]], 
                   base_config: Optional[str] = None,
                   validate: bool = True) -> Dict[str, Any]:
        """
        Get a configuration from various sources.
        
        Args:
            config_source (Union[str, Dict[str, Any]]): Configuration source.
                Can be:
                - A predefined config name (e.g., "llama-3b")
                - A path to a JSON file
                - A configuration dictionary
            base_config (Optional[str]): Base configuration name to merge with
            validate (bool): Whether to validate the configuration
            
        Returns:
            Dict[str, Any]: Final configuration
        """
        # Get the configuration
        if isinstance(config_source, dict):
            # Direct configuration dictionary
            config = config_source.copy()
        elif isinstance(config_source, str):
            if config_source.endswith('.json') or '/' in config_source or '\\' in config_source:
                # File path
                config = self.load_config_from_file(config_source)
            else:
                # Predefined config name
                try:
                    config = ModelConfigs.get_config(config_source)
                except KeyError:
                    raise ValueError(f"Unknown configuration: {config_source}")
        else:
            raise ValueError("config_source must be a string or dictionary")
        
        # Merge with base config if specified
        if base_config:
            try:
                base = ModelConfigs.get_config(base_config)
                config = self.merge_configs(base, config)
            except KeyError:
                raise ValueError(f"Unknown base configuration: {base_config}")
        
        # Validate if requested
        if validate:
            config = self.validate_config(config)
        
        return config
    
    def create_config_template(self, model_family: str, output_path: str) -> None:
        """
        Create a configuration template file for a model family.
        
        Args:
            model_family (str): Model family ("llama", "gpt", etc.)
            output_path (str): Path where to save the template
        """
        # Get a base config for the family
        base_config = ModelConfigs.get_recommended_config_for_task("jailbreaking", model_family)
        
        # Add comments as a special key (will need manual editing)
        template = {
            "_comments": {
                "model_id": "Hugging Face model identifier or OpenAI model name",
                "quantization": "Memory optimization: none, auto, 4bit, 8bit",
                "temperature": "Sampling temperature (0.0-2.0, lower = more focused)",
                "max_new_tokens": "Maximum tokens to generate",
                "hf_token_path": "Path to Hugging Face token file",
                "api_key_path": "Path to OpenAI API key file"
            },
            **base_config
        }
        
        self.save_config_to_file(template, output_path)
        print(f"Configuration template saved to: {output_path}")
    
    def list_available_configs(self) -> None:
        """Print all available predefined configurations."""
        configs = ModelConfigs.list_configs()
        
        print("Available Model Configurations:")
        print("=" * 40)
        
        for category, config_names in configs.items():
            print(f"\n{category.upper()}:")
            for name in config_names:
                print(f"  - {name}")
    
    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a configuration.
        
        Args:
            config_name (str): Name of the configuration
            
        Returns:
            Dict[str, Any]: Configuration details
        """
        try:
            config = ModelConfigs.get_config(config_name)
            
            # Add metadata
            info = {
                "name": config_name,
                "config": config,
                "model_id": config.get("model_id", "Unknown"),
                "quantization": config.get("quantization", "auto"),
                "estimated_memory": self._estimate_memory_usage(config),
                "recommended_for": self._get_use_cases(config_name)
            }
            
            return info
        except KeyError:
            raise ValueError(f"Configuration not found: {config_name}")
    
    def _estimate_memory_usage(self, config: Dict[str, Any]) -> str:
        """Estimate memory usage based on configuration."""
        model_id = config.get("model_id", "").lower()
        quantization = config.get("quantization", "none")
        
        # Simple estimation based on model name
        if "70b" in model_id:
            base_memory = 140
        elif any(size in model_id for size in ["32b", "30b"]):
            base_memory = 64
        elif any(size in model_id for size in ["14b", "13b"]):
            base_memory = 28
        elif any(size in model_id for size in ["8b", "7b"]):
            base_memory = 16
        elif any(size in model_id for size in ["4b", "3b"]):
            base_memory = 8
        elif "1b" in model_id or "0.6b" in model_id:
            base_memory = 2
        else:
            base_memory = 16  # Default estimate
        
        # Apply quantization factor
        if quantization == "4bit":
            estimated = base_memory * 0.25
        elif quantization == "8bit":
            estimated = base_memory * 0.5
        else:
            estimated = base_memory
        
        return f"~{estimated:.1f} GB"
    
    def _get_use_cases(self, config_name: str) -> List[str]:
        """Get recommended use cases for a configuration."""
        name_lower = config_name.lower()
        
        use_cases = []
        
        if "jailbreaking" in name_lower:
            use_cases.append("Security testing")
        if "reasoning" in name_lower or "deepseek" in name_lower:
            use_cases.extend(["Mathematical reasoning", "Logic problems"])
        if "creative" in name_lower:
            use_cases.extend(["Creative writing", "Story generation"])
        if "gpt" in name_lower or "openai" in name_lower:
            use_cases.extend(["General chat", "Content generation"])
        if "aya" in name_lower:
            use_cases.extend(["Multilingual tasks", "Cross-cultural communication"])
        if any(size in name_lower for size in ["1b", "3b"]):
            use_cases.append("Edge deployment")
        if any(size in name_lower for size in ["70b", "32b"]):
            use_cases.extend(["Research", "High-quality generation"])
        
        return use_cases if use_cases else ["General purpose"]
