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
        # Check if this is a Vision-Language model
        if self._is_vision_language_model():
            raise ValueError(
                f"Vision-Language model detected: {self.model_id}\n"
                f"VL models (containing 'VL', 'Vision', or marked with supports_vision=True) "
                f"are not yet supported by the current HuggingFaceBase implementation.\n\n"
                f"Supported alternatives:\n"
                f"- For Qwen: Use text-only models like 'Qwen/Qwen2.5-7B-Instruct'\n"
                f"- Use ModelConfigs.get_config('qwen-7b') for pre-configured text-only models\n"
                f"- Or wait for Vision-Language model support in future updates"
            )
        
        # Log CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            print(f"🔧 CUDA available: {gpu_count} GPUs detected")
            print(f"🔧 Current GPU: {current_device} ({gpu_name}, {gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA not available - model will load on CPU")
        
        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }
        
        # Add quantization config if enabled
        if self.quantization_config:
            load_kwargs["quantization_config"] = self.quantization_config
            load_kwargs["device_map"] = "auto"  # Required for quantization
            print(f"🔧 Using device_map='auto' (required for quantization)")
        else:
            # Use optimal device mapping
            device_map = MemoryOptimizer.get_optimal_device_map(
                self.model_id, 
                quantization_enabled=False
            )
            load_kwargs["device_map"] = device_map
            print(f"🔧 Using device_map: {device_map}")
        
        # Log loading configuration
        print(f"🔧 Model loading config: {load_kwargs}")
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            # Log actual device placement after loading
            self._log_device_placement()
            
        except Exception as e:
            if "Unrecognized configuration class" in str(e) and "VL" in str(e):
                # This is likely a VL model that wasn't caught by our detection
                raise ValueError(
                    f"Vision-Language model loading failed: {self.model_id}\n"
                    f"Error: {str(e)}\n\n"
                    f"This appears to be a Vision-Language model that requires special handling.\n"
                    f"Please use a text-only model instead, such as:\n"
                    f"- Qwen/Qwen2.5-7B-Instruct (instead of Qwen3-VL models)\n"
                    f"- Or use ModelConfigs.get_config('qwen-7b') for a pre-configured model"
                ) from e
            else:
                # Re-raise other errors as-is
                raise
    
    def _log_device_placement(self) -> None:
        """Log detailed device placement information for the model."""
        if not hasattr(self, 'model') or self.model is None:
            print("❌ Model not loaded - cannot check device placement")
            return
        
        print("\n" + "="*50)
        print("🔍 DEVICE PLACEMENT REPORT")
        print("="*50)
        
        try:
            # Check if model has device attribute (for single device models)
            if hasattr(self.model, 'device'):
                model_device = self.model.device
                print(f"📍 Model device (direct): {model_device}")
            else:
                # For models with device_map, check parameter devices
                devices = set()
                for name, param in self.model.named_parameters():
                    devices.add(param.device)
                
                if len(devices) == 1:
                    model_device = list(devices)[0]
                    print(f"📍 Model device (parameters): {model_device}")
                else:
                    print(f"📍 Model spans multiple devices: {devices}")
                    # Log device mapping for multi-device models
                    device_map = {}
                    for name, param in self.model.named_parameters():
                        layer = name.split('.')[0] if '.' in name else name
                        if layer not in device_map:
                            device_map[layer] = param.device
                    
                    print("📍 Device mapping by layer:")
                    for layer, device in device_map.items():
                        print(f"   {layer}: {device}")
                    
                    model_device = list(devices)[0]  # Use first device as primary
            
            # Check tokenizer (if it has device-related attributes)
            if hasattr(self.tokenizer, 'device'):
                print(f"📍 Tokenizer device: {self.tokenizer.device}")
            else:
                print(f"📍 Tokenizer: CPU-based (no device attribute)")
            
            # Log GPU memory usage if on CUDA
            if model_device.type == 'cuda':
                gpu_id = model_device.index if model_device.index is not None else 0
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                print(f"🔋 GPU {gpu_id} memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
            # Store primary device for later use
            self.primary_device = model_device
            
        except Exception as e:
            print(f"❌ Error checking device placement: {e}")
        
        print("="*50 + "\n")
    
    def _is_vision_language_model(self) -> bool:
        """
        Check if the model is a Vision-Language model.
        
        Returns:
            bool: True if the model is a VL model, False otherwise
        """
        # Check configuration flag
        if self.config.get("supports_vision", False):
            return True
        
        # Check model ID for VL indicators
        model_id_lower = self.model_id.lower()
        vl_indicators = [
            "vl",           # Qwen-VL, LLaVA-VL, etc.
            "vision",       # Vision models
            "llava",        # LLaVA models
            "blip",         # BLIP models
            "flamingo",     # Flamingo models
            "kosmos",       # Kosmos models
            "git-",         # GIT models
            "pix2struct"    # Pix2Struct models
        ]
        
        return any(indicator in model_id_lower for indicator in vl_indicators)
    
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
        
        # Start timing for performance monitoring
        import time
        start_time = time.time()
        
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
        
        # Device verification and logging
        model_device = self._get_model_device()
        input_device = tokenized_input.device
        
        print(f"🔍 Forward pass device check:")
        print(f"   Model device: {model_device}")
        print(f"   Input device (before): {input_device}")
        
        # Check for device mismatch
        if input_device != model_device:
            print(f"⚡ Moving inputs from {input_device} to {model_device}")
            move_start = time.time()
            tokenized_input = tokenized_input.to(model_device)
            move_time = time.time() - move_start
            print(f"   Input transfer took: {move_time:.3f}s")
        else:
            print(f"✅ Inputs already on correct device")
        
        # Log input shape and sequence length
        input_shape = tokenized_input.shape
        print(f"🔢 Input tensor shape: {input_shape}, sequence length: {input_shape[1]}")
        
        # Check GPU memory before generation (if on CUDA)
        if model_device.type == 'cuda':
            gpu_id = model_device.index if model_device.index is not None else 0
            memory_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            print(f"🔋 GPU {gpu_id} memory before generation: {memory_before:.2f}GB")
        
        # Generate response
        generation_start = time.time()
        with torch.no_grad():
            generated_tokens = self.model.generate(
                tokenized_input,
                **self.generation_config
            )
        generation_time = time.time() - generation_start
        
        # Log generation performance
        output_length = generated_tokens.shape[1] - input_shape[1]
        tokens_per_second = output_length / generation_time if generation_time > 0 else 0
        print(f"⚡ Generation completed:")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Tokens generated: {output_length}")
        print(f"   Speed: {tokens_per_second:.1f} tokens/second")
        
        # Check GPU memory after generation (if on CUDA)
        if model_device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            memory_used = memory_after - memory_before
            print(f"🔋 GPU {gpu_id} memory after generation: {memory_after:.2f}GB (+{memory_used:.2f}GB)")
        
        # Decode response
        input_length = tokenized_input.shape[1]
        generated_tokens = generated_tokens[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Total time
        total_time = time.time() - start_time
        print(f"🏁 Total forward pass time: {total_time:.2f}s\n")
        
        return response.strip()
    
    def batch_forward(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generate responses for multiple message sequences efficiently using batching.
        
        Args:
            batch_messages (List[List[Dict[str, str]]]): List of message sequences to process
                
        Returns:
            List[str]: List of generated response texts
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")
        
        if not batch_messages:
            return []
        
        batch_size = len(batch_messages)
        print(f"🔄 Processing batch of {batch_size} message sequences")
        
        # Start timing for performance monitoring
        import time
        start_time = time.time()
        
        # Prepare all message sequences
        prepared_batch = []
        for messages in batch_messages:
            prepared_messages = self._prepare_messages(messages)
            if prepared_messages:
                prepared_batch.append(prepared_messages)
        
        if not prepared_batch:
            raise ValueError("No valid message sequences to process")
        
        # Tokenize all sequences
        tokenized_inputs = []
        for prepared_messages in prepared_batch:
            try:
                tokenized_input = self.tokenizer.apply_chat_template(
                    prepared_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            except Exception as e:
                print(f"Warning: Chat template failed ({e}), using fallback formatting")
                text = self._format_messages_fallback(prepared_messages)
                tokenized_input = self.tokenizer.encode(text, return_tensors="pt")
            
            tokenized_inputs.append(tokenized_input)
        
        # Device verification and batch preparation
        model_device = self._get_model_device()
        
        # Pad sequences to same length for batching
        max_length = max(inp.shape[1] for inp in tokenized_inputs)
        batch_input_ids = []
        attention_masks = []
        
        for tokenized_input in tokenized_inputs:
            # Pad sequence
            seq_len = tokenized_input.shape[1]
            if seq_len < max_length:
                pad_length = max_length - seq_len
                # Pad on the left (common for generation)
                padded_input = torch.cat([
                    torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=tokenized_input.dtype),
                    tokenized_input
                ], dim=1)
                attention_mask = torch.cat([
                    torch.zeros(1, pad_length, dtype=torch.long),
                    torch.ones(1, seq_len, dtype=torch.long)
                ], dim=1)
            else:
                padded_input = tokenized_input
                attention_mask = torch.ones_like(tokenized_input, dtype=torch.long)
            
            batch_input_ids.append(padded_input)
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)
        
        # Move to model device
        print(f"🔍 Batch device check:")
        print(f"   Model device: {model_device}")
        print(f"   Batch input device (before): {batch_input_ids.device}")
        
        if batch_input_ids.device != model_device:
            print(f"⚡ Moving batch inputs to {model_device}")
            move_start = time.time()
            batch_input_ids = batch_input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            move_time = time.time() - move_start
            print(f"   Batch transfer took: {move_time:.3f}s")
        else:
            print(f"✅ Batch inputs already on correct device")
        
        # Log batch details
        print(f"🔢 Batch tensor shape: {batch_input_ids.shape}")
        print(f"🔢 Max sequence length: {max_length}")
        
        # Check GPU memory before generation (if on CUDA)
        if model_device.type == 'cuda':
            gpu_id = model_device.index if model_device.index is not None else 0
            memory_before = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            print(f"🔋 GPU {gpu_id} memory before batch generation: {memory_before:.2f}GB")
        
        # Generate batch response
        generation_start = time.time()
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                **self.generation_config
            )
        generation_time = time.time() - generation_start
        
        # Log generation performance
        total_output_length = generated_tokens.shape[1] * batch_size - batch_input_ids.shape[1] * batch_size
        tokens_per_second = total_output_length / generation_time if generation_time > 0 else 0
        print(f"⚡ Batch generation completed:")
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Total tokens generated: {total_output_length}")
        print(f"   Batch throughput: {tokens_per_second:.1f} tokens/second")
        print(f"   Per-sequence throughput: {tokens_per_second/batch_size:.1f} tokens/second")
        
        # Check GPU memory after generation (if on CUDA)
        if model_device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            memory_used = memory_after - memory_before
            print(f"🔋 GPU {gpu_id} memory after generation: {memory_after:.2f}GB (+{memory_used:.2f}GB)")
        
        # Decode responses
        responses = []
        original_lengths = [inp.shape[1] for inp in tokenized_inputs]
        
        for i, original_length in enumerate(original_lengths):
            # Find actual start of original sequence (accounting for padding)
            actual_start = max_length - original_length
            generated_sequence = generated_tokens[i][actual_start + original_length:]
            response = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            responses.append(response.strip())
        
        # Total time
        total_time = time.time() - start_time
        speedup = (total_time / batch_size) if batch_size > 1 else 1
        print(f"🏁 Total batch processing time: {total_time:.2f}s")
        print(f"🚀 Estimated speedup vs individual calls: {1/speedup:.1f}x\n")
        
        return responses
    
    def _get_model_device(self) -> torch.device:
        """Get the primary device of the model."""
        if hasattr(self, 'primary_device'):
            return self.primary_device
        elif hasattr(self.model, 'device'):
            return self.model.device
        else:
            return next(self.model.parameters()).device
    
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
        Get comprehensive model information including detailed device placement.
        
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
        
        # Add detailed device information
        if self.is_loaded:
            device_info = self._get_device_info()
            info.update(device_info)
        
        return info
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information for the loaded model."""
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if not hasattr(self, 'model') or self.model is None:
            device_info["model_loaded"] = False
            return device_info
        
        device_info["model_loaded"] = True
        
        try:
            # Get model device(s)
            if hasattr(self.model, 'device'):
                primary_device = self.model.device
                device_info["primary_device"] = str(primary_device)
                device_info["single_device"] = True
            else:
                # Multi-device model
                devices = set()
                device_map = {}
                for name, param in self.model.named_parameters():
                    devices.add(param.device)
                    layer = name.split('.')[0] if '.' in name else name
                    if layer not in device_map:
                        device_map[layer] = str(param.device)
                
                device_info["devices"] = [str(d) for d in devices]
                device_info["single_device"] = len(devices) == 1
                device_info["primary_device"] = str(list(devices)[0]) if devices else "unknown"
                device_info["layer_device_map"] = device_map
            
            # GPU memory information if using CUDA
            primary_device = torch.device(device_info["primary_device"])
            if primary_device.type == 'cuda':
                gpu_id = primary_device.index if primary_device.index is not None else 0
                device_info["gpu_memory"] = {
                    "allocated_gb": torch.cuda.memory_allocated(gpu_id) / (1024**3),
                    "cached_gb": torch.cuda.memory_reserved(gpu_id) / (1024**3),
                    "total_gb": torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3),
                    "gpu_name": torch.cuda.get_device_name(gpu_id)
                }
            
        except Exception as e:
            device_info["device_error"] = str(e)
        
        return device_info
    
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
