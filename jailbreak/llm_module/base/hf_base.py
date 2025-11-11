"""
Abstract base class for HuggingFace model implementations.

This module provides the base class for all HuggingFace-based language models,
handling model loading, quantization, tokenization, and generation.
"""

import os
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from .base_llm import BaseLLM
from ..quantization.quantizers import QuantizationConfig, QuantizationMode
from ..quantization.memory_optimizer import MemoryOptimizer


class HuggingFaceBase(BaseLLM):
    """
    Abstract base class for HuggingFace language models.
    
    This class handles the common functionality for all HuggingFace models,
    including model loading, quantization, tokenization, and generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace base model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(config)
        
        # HuggingFace specific configuration
        self.hf_token = None
        self.model_path = None
        self.quantization_config = None
        self.is_quantized = False
        self.device_map = "auto"
        
        # Generation configuration
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "repetition_penalty": config.get("repetition_penalty", 1.1),
            "do_sample": config.get("do_sample", True),
            "max_new_tokens": self.max_tokens,
            "pad_token_id": None  # Will be set after tokenizer loading
        }
        
        # Setup quantization
        self._setup_quantization()
        
        # Load model automatically
        self._load_model()
    
    def _setup_quantization(self) -> None:
        """Setup quantization configuration based on config."""
        quantization_setting = self.config.get("quantization", "none")
        
        if quantization_setting == "auto":
            # Auto-select quantization based on available memory
            mode = QuantizationConfig.auto_select_quantization(self.model_id)
            print(f"Auto-selected quantization mode: {mode.value}")
        else:
            # Use specified quantization mode
            try:
                mode = QuantizationMode(quantization_setting)
            except ValueError:
                print(f"Unknown quantization mode: {quantization_setting}, using 'none'")
                mode = QuantizationMode.NONE
        
        # Get quantization config
        self.quantization_config = QuantizationConfig.get_quantization_config(mode)
        self.is_quantized = self.quantization_config is not None
        
        if self.is_quantized:
            print(f"Using quantization: {mode.value}")
            # Memory usage info
            memory_info = QuantizationConfig.get_memory_usage_info(self.model_id, mode)
            print(f"Estimated memory usage: {memory_info['quantized_size_gb']:.1f} GB "
                  f"(reduced from {memory_info['base_size_gb']:.1f} GB)")
    
    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        if self.is_loaded:
            return
        
        print(f"Loading model: {self.model_id}")
        
        # Check memory requirements
        quantization_mode = "none"
        if self.is_quantized:
            if hasattr(self.quantization_config, 'load_in_4bit') and self.quantization_config.load_in_4bit:
                quantization_mode = "4bit"
            elif hasattr(self.quantization_config, 'load_in_8bit') and self.quantization_config.load_in_8bit:
                quantization_mode = "8bit"
        
        memory_check = MemoryOptimizer.check_memory_requirements(
            self.model_id, quantization_mode
        )
        
        if not memory_check["is_sufficient"]:
            print("WARNING: Insufficient memory for model!")
            for suggestion in memory_check["suggestions"]:
                print(f"  - {suggestion}")
        
        # Setup model path
        self.model_path = self._setup_model_path()
        
        # Load with memory monitoring
        with MemoryOptimizer.memory_monitor(f"Loading {self.model_id}"):
            self._load_tokenizer()
            self._load_model_weights()
        
        # Optimize for inference
        MemoryOptimizer.optimize_for_inference()
        
        self.is_loaded = True
        print(f"Model loaded successfully: {self.model_id}")
    
    def _setup_model_path(self) -> str:
        """Setup the model path, downloading if necessary."""
        # Get model directory from config, with new default path
        if "hf_model_path" not in self.config:
            # Default to the jailbreak hf_models directory
            import pathlib
            current_file = pathlib.Path(__file__).resolve()
            # Navigate from jailbreak/jailbreak/llm_module/base/hf_base.py to jailbreak/hf_models/
            jailbreak_root = current_file.parents[3]  # Go up to the outer jailbreak directory
            default_hf_path = jailbreak_root / "hf_models"
            hf_model_path = str(default_hf_path)
        else:
            hf_model_path = self.config["hf_model_path"]
        
        # Ensure the directory exists
        os.makedirs(hf_model_path, exist_ok=True)
        
        model_path = os.path.join(hf_model_path, self.model_id)
        
        # Check if model exists locally
        if not os.path.exists(model_path) or self.config.get("redownload", False):
            print(f"Downloading model to: {model_path}")
            
            # Get HF token
            token_path = self.config.get("hf_token_path")
            if token_path and os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    self.hf_token = f.read().strip()
            
            # Download model
            snapshot_download(
                repo_id=self.model_id,
                local_dir=model_path,
                token=self.hf_token
            )
        
        return model_path
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update generation config with pad token
        self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
    
    def _load_model_weights(self) -> None:
        """Load the model weights."""
        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }
        
        # Add quantization config if enabled
        if self.quantization_config:
            load_kwargs["quantization_config"] = self.quantization_config
            load_kwargs["device_map"] = "auto"  # Required for quantization
        else:
            # Use optimal device mapping
            device_map = MemoryOptimizer.get_optimal_device_map(
                self.model_id, 
                quantization_enabled=False
            )
            load_kwargs["device_map"] = device_map
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs
        )
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for conversations.
        
        Args:
            prompt (str): The system prompt to use
        """
        self.system_prompt = prompt
        print(f"System prompt set: {prompt[:100]}...")
    
    def forward(self, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the language model.
        
        Args:
            messages (Optional[List[Dict[str, str]]]): Optional messages to use.
                If None, uses the internal conversation.
                
        Returns:
            str: Generated response text
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")
        
        # Prepare messages
        prepared_messages = self._prepare_messages(messages)
        
        if not prepared_messages:
            raise ValueError("No messages to process")
        
        # Apply chat template
        try:
            tokenized_input = self.tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Warning: Chat template failed ({e}), using fallback formatting")
            # Fallback: manual formatting
            text = self._format_messages_fallback(prepared_messages)
            tokenized_input = self.tokenizer.encode(text, return_tensors="pt")
        
        # Move to appropriate device
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        tokenized_input = tokenized_input.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_tokens = self.model.generate(
                tokenized_input,
                **self.generation_config
            )
        
        # Decode response
        input_length = tokenized_input.shape[1]
        generated_tokens = generated_tokens[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Fallback message formatting when chat template is not available.
        
        Args:
            messages (List[Dict[str, str]]): Messages to format
            
        Returns:
            str: Formatted text
        """
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        formatted_parts.append("Assistant:")  # Add generation prompt
        return "\n\n".join(formatted_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = super().get_model_info()
        info.update({
            "model_type": "HuggingFace",
            "model_path": self.model_path,
            "is_quantized": self.is_quantized,
            "device_map": self.device_map,
            "generation_config": self.generation_config,
            "hf_token_configured": self.hf_token is not None
        })
        
        if self.is_loaded and hasattr(self.model, 'device'):
            info["device"] = str(self.model.device)
        
        return info
    
    def prepare_for_lora(self, lora_config: Dict[str, Any]):
        """
        Prepare the model for LoRA fine-tuning (future implementation).
        
        Args:
            lora_config (Dict[str, Any]): LoRA configuration
            
        Returns:
            Any: PEFT model ready for fine-tuning
        """
        # This is a placeholder for future LoRA implementation
        print("LoRA fine-tuning preparation - feature coming soon!")
        print(f"LoRA config: {lora_config}")
        
        # Future implementation will use PEFT library:
        # from peft import LoraConfig, get_peft_model, TaskType
        # config = LoraConfig(...)
        # peft_model = get_peft_model(self.model, config)
        # return peft_model
        
        return None
