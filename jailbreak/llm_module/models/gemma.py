"""
Gemma model implementations.

This module provides the Gemma class for Google Gemma model variants.
"""

from typing import Dict, Any, List
from ..base.hf_base import HuggingFaceBase


class Gemma(HuggingFaceBase):
    """Gemma language model implementation (HuggingFace)."""

    SUPPORTED_VARIANTS = [
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-27b-it",
    ]

    def __init__(self, config: Dict[str, Any]):
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in known Gemma variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")

        if not model_id:
            config["model_id"] = "google/gemma-3-4b-it"
            print(f"No model_id specified, using default: {config['model_id']}")

        self._apply_gemma_optimizations(config)
        super().__init__(config)
        print(f"Gemma model initialized: {self.model_id}")

    def _apply_gemma_optimizations(self, config: Dict[str, Any]) -> None:
        defaults = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "max_new_tokens": 512,
        }
        for key, value in defaults.items():
            if key not in config:
                config[key] = value

        model_id = config.get("model_id", "").lower()
        if "27b" in model_id and "quantization" not in config:
            config["quantization"] = "4bit"
        elif any(s in model_id for s in ["4b", "1b"]) and "quantization" not in config:
            config["quantization"] = "none"

    def set_system_prompt(self, prompt: str) -> None:
        formatted_prompt = f"You are Gemma, a helpful and safe assistant. {prompt}"
        super().set_system_prompt(formatted_prompt)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_family": "Gemma",
                "provider": "Google",
                "supported_variants": self.SUPPORTED_VARIANTS,
            }
        )
        return info
