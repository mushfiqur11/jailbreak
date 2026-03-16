"""
GPT-OSS model implementations.

This module provides a HuggingFace-backed class for open-weight GPT-OSS models
such as openai/gpt-oss-20b.
"""

from typing import Dict, Any
from ..base.hf_base import HuggingFaceBase


class GPTOSS(HuggingFaceBase):
    """GPT-OSS language model implementation (HuggingFace)."""

    SUPPORTED_VARIANTS = [
        "openai/gpt-oss-20b",
    ]

    def __init__(self, config: Dict[str, Any]):
        model_id = config.get("model_id", "")
        if model_id and model_id not in self.SUPPORTED_VARIANTS:
            print(f"Warning: {model_id} is not in known GPT-OSS variants")
            print(f"Supported variants: {self.SUPPORTED_VARIANTS}")

        if not model_id:
            config["model_id"] = "openai/gpt-oss-20b"
            print(f"No model_id specified, using default: {config['model_id']}")

        self._apply_gptoss_optimizations(config)
        super().__init__(config)
        print(f"GPT-OSS model initialized: {self.model_id}")

    def _apply_gptoss_optimizations(self, config: Dict[str, Any]) -> None:
        defaults = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "max_new_tokens": 1024,
        }
        for key, value in defaults.items():
            if key not in config:
                config[key] = value

        if "quantization" not in config:
            config["quantization"] = "4bit"

    def set_system_prompt(self, prompt: str) -> None:
        formatted_prompt = f"You are GPT-OSS, a helpful assistant. {prompt}"
        super().set_system_prompt(formatted_prompt)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_family": "GPT-OSS",
                "provider": "OpenAI (open-weight via HF)",
                "supported_variants": self.SUPPORTED_VARIANTS,
            }
        )
        return info
