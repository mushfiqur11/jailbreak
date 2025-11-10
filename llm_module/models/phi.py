"""
Phi model implementations.

This module provides the Phi class for Microsoft Phi model variants,
including Phi-3-mini, Phi-4, and other Phi series models.
"""

from typing import Dict, Any, List, Optional
from ..base.hf_base import HuggingFaceBase


class Phi(HuggingFaceBase):
    """
    Phi language model implementation.
    
    Supports Microsoft Phi variants including:
    - microsoft/Phi-3-mini-4k-instruct
    - microsoft/phi-4
    - microsoft/Phi-4-reasoning-plus
    """
    
    # Supported model variants
    SUPPORTED_VARIANTS = [
        'microsoft/Phi-3-mini-4k-instruct',
        'microsoft/phi-4',
        'microsoft/Phi-4-reasoning-plus'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Phi model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        # Validate model ID
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in the list of known Phi variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")
        
        # Set default model if not specified
        if not model_id:
            config["model_id"] = "microsoft/Phi-3-mini-4k-instruct"
            print(f"No model_id specified, using default: {config['model_id']}")
        
        # Apply Phi-specific optimizations
        self._apply_phi_optimizations(config)
        
        # Initialize parent class
        super().__init__(config)
        
        print(f"Phi model initialized: {self.model_id}")
    
    def _apply_phi_optimizations(self, config: Dict[str, Any]) -> None:
        """Apply Phi-specific optimizations to the configuration."""
        
        # Phi-specific generation defaults
        phi_defaults = {
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": 512
        }
        
        # Apply defaults if not already specified
        for key, default_value in phi_defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Model-specific optimizations
        model_id = config.get("model_id", "").lower()
        
        if "phi-4" in model_id:
            # Phi-4 optimizations
            if "quantization" not in config:
                config["quantization"] = "auto"
            print("Applied Phi-4 model optimizations")
        else:
            # Phi-3 mini optimizations
            if "quantization" not in config:
                config["quantization"] = "none"  # Usually no quantization needed for mini
            print("Applied Phi-3-mini model optimizations")
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt with Phi-specific formatting.
        
        Args:
            prompt (str): The system prompt to use
        """
        # Phi models work well with concise, clear instructions
        formatted_prompt = f"You are a precise and helpful assistant. {prompt}"
        super().set_system_prompt(formatted_prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Phi-specific model information.
        
        Returns:
            Dict[str, Any]: Enhanced model information
        """
        info = super().get_model_info()
        info.update({
            "model_family": "Phi",
            "provider": "Microsoft",
            "supported_variants": self.SUPPORTED_VARIANTS,
            "estimated_parameters": self._get_parameter_count(),
            "context_length": self._get_context_length(),
            "special_features": self._get_special_features()
        })
        return info
    
    def _get_parameter_count(self) -> str:
        """Get estimated parameter count based on model name."""
        model_id_lower = self.model_id.lower()
        
        if "phi-4" in model_id_lower:
            return "~14B parameters"
        elif "phi-3-mini" in model_id_lower:
            return "~3.8B parameters"
        else:
            return "Unknown"
    
    def _get_context_length(self) -> int:
        """Get context length based on model variant."""
        model_id_lower = self.model_id.lower()
        
        if "4k" in model_id_lower:
            return 4096
        elif "phi-4" in model_id_lower:
            return 16384
        else:
            return 4096  # Default
    
    def _get_special_features(self) -> List[str]:
        """Get special features of the model."""
        features = []
        model_id_lower = self.model_id.lower()
        
        if "phi-4" in model_id_lower:
            features.extend([
                "Enhanced reasoning capabilities",
                "Improved mathematical performance",
                "Better code generation"
            ])
        
        if "phi-3-mini" in model_id_lower:
            features.extend([
                "Compact and efficient",
                "Good performance-to-size ratio",
                "Mobile/edge friendly"
            ])
        
        if "reasoning" in model_id_lower:
            features.append("Specialized for reasoning tasks")
        
        # All Phi models
        features.extend([
            "Microsoft optimized",
            "Efficient architecture",
            "Good instruction following"
        ])
        
        return features
    
    @classmethod
    def get_recommended_config(cls, model_variant: str = "phi-3-mini") -> Dict[str, Any]:
        """
        Get recommended configuration for different Phi variants.
        
        Args:
            model_variant (str): Model variant ("phi-3-mini", "phi-4", etc.)
            
        Returns:
            Dict[str, Any]: Recommended configuration
        """
        model_variant = model_variant.lower()
        
        base_config = {
            "hf_model_path": "./models",
            "hf_token_path": "./tokens/hf_token.txt",
            "temperature": 0.6,
            "max_new_tokens": 512
        }
        
        if "phi-4" in model_variant:
            base_config.update({
                "model_id": "microsoft/phi-4",
                "quantization": "auto",
                "max_new_tokens": 1024
            })
        else:
            # Default to Phi-3-mini
            base_config.update({
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "quantization": "none"
            })
        
        return base_config
