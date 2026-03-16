"""
Memory optimization utilities for efficient model loading and inference.

This module provides utilities to optimize memory usage during model loading
and inference, including device mapping and memory monitoring.
"""

import torch
import gc
import psutil
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class MemoryOptimizer:
    """
    Utilities for memory optimization during model loading and inference.
    
    This class provides methods to monitor memory usage, clear caches,
    and optimize device mapping for efficient model loading.
    """
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """
        Get comprehensive memory information.
        
        Returns:
            Dict[str, Any]: Memory information including system and GPU memory
        """
        info = {
            "system_memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent_used": psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = gpu_props.total_memory / (1024**3)
                
                gpu_info[f"gpu_{i}"] = {
                    "name": gpu_props.name,
                    "total_gb": total,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "free_gb": total - allocated,
                    "percent_used": (allocated / total) * 100 if total > 0 else 0
                }
            info["gpu_memory"] = gpu_info
        else:
            info["gpu_memory"] = {}
        
        return info
    
    @staticmethod
    def clear_cache() -> None:
        """Clear Python and CUDA caches to free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_optimal_device_map(
        model_id: str,
        quantization_enabled: bool = False,
        max_memory_per_gpu: Optional[Dict[int, str]] = None
    ) -> str:
        """
        Get optimal device mapping for model loading.
        
        Args:
            model_id (str): Model identifier
            quantization_enabled (bool): Whether quantization is enabled
            max_memory_per_gpu (Optional[Dict[int, str]]): Maximum memory per GPU
            
        Returns:
            str: Device map ("auto", "cpu", or specific mapping)
        """
        if not torch.cuda.is_available():
            return "cpu"
        
        # For quantized models, use auto device mapping
        if quantization_enabled:
            return "auto"
        
        # For non-quantized models, check available memory
        memory_info = MemoryOptimizer.get_memory_info()
        gpu_memory = memory_info.get("gpu_memory", {})
        
        if not gpu_memory:
            return "cpu"
        
        # If we have multiple GPUs, use auto mapping
        if len(gpu_memory) > 1:
            return "auto"
        
        # Single GPU - check if model fits
        from ..quantization.quantizers import QuantizationConfig
        model_size = QuantizationConfig.estimate_model_size(model_id)
        
        gpu_0_info = list(gpu_memory.values())[0]
        available_memory = gpu_0_info["free_gb"]
        
        if available_memory > model_size * 1.2:  # 20% buffer
            return "cuda:0"
        else:
            return "auto"  # Let transformers handle the mapping
    
    @staticmethod
    @contextmanager
    def memory_monitor(operation_name: str = "Model Operation"):
        """
        Context manager to monitor memory usage during an operation.
        
        Args:
            operation_name (str): Name of the operation being monitored
        """
        print(f"Starting {operation_name}...")
        
        # Clear cache before starting
        MemoryOptimizer.clear_cache()
        
        # Get initial memory state
        initial_memory = MemoryOptimizer.get_memory_info()
        
        try:
            yield
        finally:
            # Get final memory state
            final_memory = MemoryOptimizer.get_memory_info()
            
            # Calculate memory usage
            system_used_diff = (
                final_memory["system_memory"]["used_gb"] - 
                initial_memory["system_memory"]["used_gb"]
            )
            
            print(f"{operation_name} completed.")
            print(f"System memory change: {system_used_diff:.2f} GB")
            
            if final_memory["gpu_memory"]:
                for gpu_id, gpu_info in final_memory["gpu_memory"].items():
                    initial_gpu = initial_memory["gpu_memory"].get(gpu_id, {})
                    gpu_used_diff = (
                        gpu_info.get("allocated_gb", 0) - 
                        initial_gpu.get("allocated_gb", 0)
                    )
                    print(f"{gpu_info['name']} ({gpu_id}) memory change: {gpu_used_diff:.2f} GB")
    
    @staticmethod
    def optimize_for_inference() -> None:
        """Optimize settings for inference."""
        if torch.cuda.is_available():
            # Disable gradient computation for inference
            torch.set_grad_enabled(False)
            
            # Set CUDA memory optimization flags
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def check_memory_requirements(
        model_id: str,
        quantization_mode: str = "none",
        warn_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        Check if system has enough memory for the model.
        
        Args:
            model_id (str): Model identifier
            quantization_mode (str): Quantization mode
            warn_threshold (float): Warning threshold (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Memory requirement check results
        """
        from ..quantization.quantizers import QuantizationConfig, QuantizationMode
        
        # Get model size estimate
        mode_enum = QuantizationMode(quantization_mode) if quantization_mode != "none" else QuantizationMode.NONE
        memory_info = QuantizationConfig.get_memory_usage_info(model_id, mode_enum)
        required_memory = memory_info["quantized_size_gb"]
        
        # Get available memory
        system_memory = MemoryOptimizer.get_memory_info()
        available_system = system_memory["system_memory"]["available_gb"]
        
        gpu_available = 0.0
        gpu_suitable = False
        if torch.cuda.is_available() and system_memory["gpu_memory"]:
            # Use the GPU with the most free memory
            gpu_available = max(
                gpu["free_gb"] for gpu in system_memory["gpu_memory"].values()
            )
            gpu_suitable = gpu_available >= required_memory
        
        system_suitable = available_system >= required_memory
        
        # Determine best option
        if gpu_suitable:
            recommended_device = "cuda"
            available_memory = gpu_available
        elif system_suitable:
            recommended_device = "cpu"
            available_memory = available_system
        else:
            recommended_device = "insufficient"
            available_memory = max(available_system, gpu_available)
        
        memory_ratio = available_memory / required_memory if required_memory > 0 else float('inf')
        
        return {
            "model_id": model_id,
            "quantization_mode": quantization_mode,
            "required_memory_gb": required_memory,
            "available_system_gb": available_system,
            "available_gpu_gb": gpu_available,
            "recommended_device": recommended_device,
            "memory_ratio": memory_ratio,
            "is_sufficient": memory_ratio >= 1.0,
            "needs_warning": memory_ratio > warn_threshold and memory_ratio < 1.0,
            "suggestions": MemoryOptimizer._get_memory_suggestions(memory_ratio, quantization_mode)
        }
    
    @staticmethod
    def _get_memory_suggestions(memory_ratio: float, current_quantization: str) -> List[str]:
        """Get suggestions for memory optimization."""
        suggestions = []
        
        if memory_ratio < 1.0:
            suggestions.append("Model requires more memory than available")
            
            if current_quantization == "none":
                suggestions.append("Try 8-bit quantization to reduce memory usage by ~50%")
                suggestions.append("Try 4-bit quantization to reduce memory usage by ~75%")
            elif current_quantization == "8bit":
                suggestions.append("Try 4-bit quantization to further reduce memory usage")
            else:
                suggestions.append("Consider using a smaller model variant")
                suggestions.append("Close other applications to free memory")
                
        elif memory_ratio < 1.2:
            suggestions.append("Memory usage will be tight - consider closing other applications")
            
        return suggestions
