"""
Quantization configurations and utilities for memory-efficient model loading.

This module provides different quantization modes and configurations to reduce
memory usage when loading large language models.
"""

import torch
import psutil
from enum import Enum
from typing import Optional, Dict, Any
from transformers import BitsAndBytesConfig


class QuantizationMode(Enum):
    """Supported quantization modes."""
    NONE = "none"
    AUTO = "auto"
    INT4_BNB = "4bit"
    INT8_BNB = "8bit"
    INT4_AWQ = "int4_awq"
    INT8_GPTQ = "int8_gptq"


class QuantizationConfig:
    """
    Comprehensive quantization configuration manager.
    
    This class provides various quantization configurations optimized for
    different hardware setups and memory constraints.
    """
    
    # Model size estimates in GB (approximate)
    MODEL_SIZE_ESTIMATES = {
        # Llama models
        "llama-1b": 2.0,
        "llama-3b": 6.0,
        "llama-7b": 14.0,
        "llama-8b": 16.0,
        "llama-13b": 26.0,
        "llama-70b": 140.0,
        
        # Phi models
        "phi-3-mini": 8.0,
        "phi-4": 14.0,
        
        # Qwen models
        "qwen-0.6b": 1.2,
        "qwen-1.5b": 3.0,
        "qwen-4b": 8.0,
        "qwen-7b": 14.0,
        "qwen-14b": 28.0,
        "qwen-32b": 64.0,
        
        # DeepSeek models
        "deepseek-1.5b": 3.0,
        "deepseek-14b": 28.0,
        "deepseek-32b": 64.0,
        "deepseek-70b": 140.0,
    }
    
    @staticmethod
    def get_bnb_4bit_config(
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
        use_double_quant: bool = True
    ) -> BitsAndBytesConfig:
        """
        Get BitsAndBytes 4-bit quantization configuration.
        
        Args:
            compute_dtype (torch.dtype): Compute data type for quantized model
            quant_type (str): Quantization type ("fp4" or "nf4")
            use_double_quant (bool): Whether to use double quantization
            
        Returns:
            BitsAndBytesConfig: 4-bit quantization configuration
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    
    @staticmethod
    def get_bnb_8bit_config(enable_fp32_cpu_offload: bool = True) -> BitsAndBytesConfig:
        """
        Get BitsAndBytes 8-bit quantization configuration.
        
        Args:
            enable_fp32_cpu_offload (bool): Enable CPU offloading for FP32 weights
            
        Returns:
            BitsAndBytesConfig: 8-bit quantization configuration
        """
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=enable_fp32_cpu_offload
        )
    
    @staticmethod
    def get_available_memory_gb() -> float:
        """
        Get available system memory in GB.
        
        Returns:
            float: Available memory in GB
        """
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)
    
    @staticmethod
    def get_gpu_memory_gb() -> float:
        """
        Get available GPU memory in GB.
        
        Returns:
            float: Available GPU memory in GB, 0 if no GPU available
        """
        if not torch.cuda.is_available():
            return 0.0
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        return gpu_memory / (1024**3)
    
    @staticmethod
    def estimate_model_size(model_id: str) -> float:
        """
        Estimate model size in GB based on model ID.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            float: Estimated model size in GB
        """
        model_id_lower = model_id.lower()
        
        # Extract size indicators from model name
        for size_key, size_gb in QuantizationConfig.MODEL_SIZE_ESTIMATES.items():
            if size_key in model_id_lower:
                return size_gb
        
        # Fallback: parse numbers from model name
        import re
        numbers = re.findall(r'(\d+(?:\.\d+)?)b', model_id_lower)
        if numbers:
            # Take the largest number as likely model size
            max_number = max(float(num) for num in numbers)
            return max_number * 2.0  # Rough estimate: 2GB per billion parameters
        
        # Default fallback
        return 14.0  # Assume ~7B model size
    
    @staticmethod
    def auto_select_quantization(
        model_id: str,
        available_memory_gb: Optional[float] = None,
        prefer_gpu: bool = True
    ) -> QuantizationMode:
        """
        Automatically select the best quantization mode based on available resources.
        
        Args:
            model_id (str): Model identifier
            available_memory_gb (Optional[float]): Available memory in GB. If None, auto-detect.
            prefer_gpu (bool): Whether to prefer GPU memory over system memory
            
        Returns:
            QuantizationMode: Recommended quantization mode
        """
        model_size_gb = QuantizationConfig.estimate_model_size(model_id)
        
        if available_memory_gb is None:
            if prefer_gpu and torch.cuda.is_available():
                available_memory_gb = QuantizationConfig.get_gpu_memory_gb()
            else:
                available_memory_gb = QuantizationConfig.get_available_memory_gb()
        
        # Calculate memory ratio (available / required)
        memory_ratio = available_memory_gb / model_size_gb
        
        if memory_ratio >= 1.5:
            # Plenty of memory - no quantization needed
            return QuantizationMode.NONE
        elif memory_ratio >= 0.8:
            # Moderate memory - 8-bit quantization
            return QuantizationMode.INT8_BNB
        elif memory_ratio >= 0.4:
            # Limited memory - 4-bit quantization
            return QuantizationMode.INT4_BNB
        else:
            # Very limited memory - aggressive 4-bit quantization
            return QuantizationMode.INT4_BNB
    
    @staticmethod
    def get_quantization_config(
        mode: QuantizationMode,
        extra_options: Optional[Dict[str, Any]] = None
    ) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration for the specified mode.
        
        Args:
            mode (QuantizationMode): Quantization mode
            extra_options (Optional[Dict[str, Any]]): Additional options
            
        Returns:
            Optional[BitsAndBytesConfig]: Quantization configuration or None
        """
        if mode == QuantizationMode.NONE:
            return None
        
        extra_options = extra_options or {}
        
        if mode == QuantizationMode.INT4_BNB:
            return QuantizationConfig.get_bnb_4bit_config(
                compute_dtype=extra_options.get("compute_dtype", torch.float16),
                quant_type=extra_options.get("quant_type", "nf4"),
                use_double_quant=extra_options.get("use_double_quant", True)
            )
        elif mode == QuantizationMode.INT8_BNB:
            return QuantizationConfig.get_bnb_8bit_config(
                enable_fp32_cpu_offload=extra_options.get("enable_fp32_cpu_offload", True)
            )
        else:
            raise NotImplementedError(f"Quantization mode {mode} not yet implemented")
    
    @staticmethod
    def get_memory_usage_info(model_id: str, quantization_mode: QuantizationMode) -> Dict[str, Any]:
        """
        Get estimated memory usage information for a model with quantization.
        
        Args:
            model_id (str): Model identifier
            quantization_mode (QuantizationMode): Quantization mode
            
        Returns:
            Dict[str, Any]: Memory usage information
        """
        base_size_gb = QuantizationConfig.estimate_model_size(model_id)
        
        # Quantization reduction factors
        reduction_factors = {
            QuantizationMode.NONE: 1.0,
            QuantizationMode.INT8_BNB: 0.5,
            QuantizationMode.INT4_BNB: 0.25,
            QuantizationMode.INT4_AWQ: 0.25,
            QuantizationMode.INT8_GPTQ: 0.5,
        }
        
        reduction = reduction_factors.get(quantization_mode, 1.0)
        quantized_size_gb = base_size_gb * reduction
        
        return {
            "base_size_gb": base_size_gb,
            "quantized_size_gb": quantized_size_gb,
            "reduction_factor": reduction,
            "memory_saved_gb": base_size_gb - quantized_size_gb,
            "memory_saved_percent": (1 - reduction) * 100,
            "quantization_mode": quantization_mode.value
        }
