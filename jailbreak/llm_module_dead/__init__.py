"""LLM module entrypoint.

Keep this package import lightweight and robust so pipeline imports do not fail
due to optional utilities/fine-tuning helpers.
"""

from .model_factory import llm_model_factory
from .config.model_configs import ModelConfigs

__version__ = "1.0.1"
__author__ = "Jailbreak Framework Team"

__all__ = [
    "llm_model_factory",
    "ModelConfigs",
]
