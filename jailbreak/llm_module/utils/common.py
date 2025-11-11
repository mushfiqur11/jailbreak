"""
Common utility functions for the LLM module.

This module provides various utility functions used across different
components of the LLM module.
"""

import re
import os
import sys
import hashlib
import platform
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class ModelUtils:
    """
    Collection of utility functions for model operations.
    
    This class provides static methods for common operations like
    text processing, model identification, and system information.
    """
    
    @staticmethod
    def clean_text(text: str, remove_extra_whitespace: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Text to clean
            remove_extra_whitespace (bool): Whether to remove extra whitespace
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove or replace problematic characters
        text = text.replace('\r\n', '\n')  # Normalize line endings
        text = text.replace('\r', '\n')
        
        if remove_extra_whitespace:
            # Remove multiple consecutive spaces
            text = re.sub(r' +', ' ', text)
            # Remove multiple consecutive newlines
            text = re.sub(r'\n+', '\n', text)
            # Strip leading/trailing whitespace
            text = text.strip()
        
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to a maximum length.
        
        Args:
            text (str): Text to truncate
            max_length (int): Maximum length
            suffix (str): Suffix to add if truncated
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Account for suffix length
        effective_length = max_length - len(suffix)
        if effective_length <= 0:
            return suffix[:max_length]
        
        return text[:effective_length] + suffix
    
    @staticmethod
    def extract_model_family(model_id: str) -> str:
        """
        Extract model family from model ID.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            str: Model family name
        """
        model_id_lower = model_id.lower()
        
        # Common patterns for model families
        if "llama" in model_id_lower:
            return "llama"
        elif "gpt" in model_id_lower or "openai" in model_id_lower:
            return "gpt"
        elif "phi" in model_id_lower:
            return "phi"
        elif "qwen" in model_id_lower:
            return "qwen"
        elif "deepseek" in model_id_lower:
            return "deepseek"
        elif "aya" in model_id_lower:
            return "aya"
        elif "claude" in model_id_lower:
            return "claude"
        elif "gemini" in model_id_lower:
            return "gemini"
        else:
            return "unknown"
    
    @staticmethod
    def extract_model_size(model_id: str) -> Optional[str]:
        """
        Extract model size from model ID.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Optional[str]: Model size (e.g., "7b", "70b") or None
        """
        # Look for size patterns like "7b", "70b", "1.5b", etc.
        size_pattern = r'(\d+(?:\.\d+)?)[bB]'
        matches = re.findall(size_pattern, model_id)
        
        if matches:
            # Return the largest number found (usually the parameter count)
            sizes = [float(m) for m in matches]
            largest = max(sizes)
            return f"{largest:g}b".replace('.', '.')
        
        return None
    
    @staticmethod
    def is_openai_model(model_id: str) -> bool:
        """
        Check if a model ID corresponds to an OpenAI model.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            bool: True if it's an OpenAI model
        """
        openai_patterns = [
            "gpt-3.5", "gpt-4", "gpt-4o", "text-davinci", "text-curie",
            "text-babbage", "text-ada", "o3-mini", "o1-mini", "o1-preview",
            "chatgpt"
        ]
        
        model_id_lower = model_id.lower()
        return any(pattern in model_id_lower for pattern in openai_patterns)
    
    @staticmethod
    def is_huggingface_model(model_id: str) -> bool:
        """
        Check if a model ID corresponds to a HuggingFace model.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            bool: True if it's a HuggingFace model
        """
        # HuggingFace models typically have org/model format
        return "/" in model_id and not ModelUtils.is_openai_model(model_id)
    
    @staticmethod
    def validate_model_id(model_id: str) -> Dict[str, Any]:
        """
        Validate and analyze a model ID.
        
        Args:
            model_id (str): Model identifier to validate
            
        Returns:
            Dict[str, Any]: Validation results and metadata
        """
        if not model_id or not isinstance(model_id, str):
            return {
                "valid": False,
                "error": "Model ID must be a non-empty string"
            }
        
        info = {
            "valid": True,
            "model_id": model_id,
            "family": ModelUtils.extract_model_family(model_id),
            "size": ModelUtils.extract_model_size(model_id),
            "is_openai": ModelUtils.is_openai_model(model_id),
            "is_huggingface": ModelUtils.is_huggingface_model(model_id),
            "provider": "openai" if ModelUtils.is_openai_model(model_id) else "huggingface"
        }
        
        # Additional validation
        if not info["is_openai"] and not info["is_huggingface"]:
            info["warning"] = "Model ID format not recognized"
        
        return info
    
    @staticmethod
    def format_memory_size(bytes_size: Union[int, float]) -> str:
        """
        Format memory size in human-readable format.
        
        Args:
            bytes_size (Union[int, float]): Size in bytes
            
        Returns:
            str: Formatted size (e.g., "1.5 GB")
        """
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        size = float(bytes_size)
        
        for unit in units:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        
        return f"{size:.1f} {units[-1]}"
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information relevant for model deployment.
        
        Returns:
            Dict[str, Any]: System information
        """
        import psutil
        import torch
        
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            info["gpu_count"] = gpu_count
            info["gpus"] = []
            
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "id": i,
                    "name": gpu_props.name,
                    "memory_total_gb": gpu_props.total_memory / (1024**3),
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
                }
                info["gpus"].append(gpu_info)
        
        return info
    
    @staticmethod
    def create_model_hash(model_id: str, config: Dict[str, Any]) -> str:
        """
        Create a unique hash for a model configuration.
        
        Args:
            model_id (str): Model identifier
            config (Dict[str, Any]): Model configuration
            
        Returns:
            str: Unique hash string
        """
        # Create a deterministic string representation
        config_str = f"{model_id}_{sorted(config.items())}"
        
        # Generate hash
        hash_obj = hashlib.md5(config_str.encode('utf-8'))
        return hash_obj.hexdigest()[:12]  # First 12 characters
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
        """
        Find files matching a pattern in a directory.
        
        Args:
            directory (str): Directory to search
            pattern (str): Glob pattern to match
            recursive (bool): Whether to search recursively
            
        Returns:
            List[str]: List of matching file paths
        """
        path = Path(directory)
        
        if not path.exists():
            return []
        
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        return [str(f) for f in files if f.is_file()]
    
    @staticmethod
    def ensure_directory(directory: str) -> str:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory (str): Directory path
            
        Returns:
            str: Absolute path to the directory
        """
        path = Path(directory).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @staticmethod
    def load_text_file(file_path: str, encoding: str = 'utf-8') -> str:
        """
        Load text content from a file.
        
        Args:
            file_path (str): Path to the text file
            encoding (str): File encoding
            
        Returns:
            str: File content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If encoding is incorrect
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def save_text_file(content: str, file_path: str, encoding: str = 'utf-8') -> None:
        """
        Save text content to a file.
        
        Args:
            content (str): Content to save
            file_path (str): Path to save the file
            encoding (str): File encoding
        """
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ModelUtils.ensure_directory(directory)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def count_tokens_rough(text: str) -> int:
        """
        Get a rough token count estimate for text.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Rough token count estimate
        """
        # Rough approximation: ~4 characters per token for most languages
        # This is a very rough estimate and actual tokenization will vary
        return max(1, len(text) // 4)
    
    @staticmethod
    def split_text_by_tokens(text: str, max_tokens: int = 1000) -> List[str]:
        """
        Split text into chunks based on estimated token count.
        
        Args:
            text (str): Text to split
            max_tokens (int): Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
        
        # Rough character count per chunk (4 chars per token estimate)
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find a good breaking point (sentence or paragraph end)
            end_pos = min(current_pos + max_chars, len(text))
            
            if end_pos < len(text):
                # Look for sentence boundaries
                for boundary in ['. ', '.\n', '!\n', '?\n']:
                    last_boundary = text.rfind(boundary, current_pos, end_pos)
                    if last_boundary > current_pos:
                        end_pos = last_boundary + len(boundary)
                        break
            
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = end_pos
        
        return chunks
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any], 
                     deep_merge: bool = True) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base (Dict[str, Any]): Base configuration
            override (Dict[str, Any]): Override configuration
            deep_merge (bool): Whether to recursively merge nested dicts
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if (deep_merge and 
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = ModelUtils.merge_configs(result[key], value, deep_merge)
            else:
                result[key] = value
        
        return result
