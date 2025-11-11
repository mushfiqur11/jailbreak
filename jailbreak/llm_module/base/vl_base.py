"""
Vision-Language base class for HuggingFace VL model implementations.

This module provides the base class for Vision-Language models that support
both text and image inputs, handling model loading, tokenization, and generation.
"""

import os
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

from .hf_base import HuggingFaceBase


class VisionLanguageBase(HuggingFaceBase):
    """
    Base class for Vision-Language models.
    
    Extends HuggingFaceBase to support models that can process both text and images,
    such as Qwen-VL, LLaVA, and other multimodal models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Vision-Language base model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        # Mark as vision model to bypass the VL detection block in parent
        config["supports_vision"] = True
        self.processor = None
        
        # Don't call parent's __init__ directly as it will try to load as text-only
        # Instead, initialize components manually
        self._init_without_loading(config)
        
        # Load VL model using our custom method
        self._load_vl_model()
    
    def _init_without_loading(self, config: Dict[str, Any]):
        """Initialize base components without loading the model."""
        # Initialize BaseLLM components manually
        self.config = config
        self.model_id = config.get("model_id", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_new_tokens", config.get("max_tokens", 512))
        self.system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
        self.conversation = []
        self.is_loaded = False
        
        # HuggingFace specific configuration
        self.hf_token = None
        self.model_path = None
        self.quantization_config = None
        self.is_quantized = False
        self.device_map = "auto"
        
        # Generation configuration for VL models
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50),
            "repetition_penalty": config.get("repetition_penalty", 1.1),
            "do_sample": config.get("do_sample", True),
            "max_new_tokens": self.max_tokens,
            "pad_token_id": None  # Will be set after processor loading
        }
        
        # Setup quantization
        self._setup_quantization()
    
    def _load_vl_model(self) -> None:
        """Load the Vision-Language model and processor."""
        if self.is_loaded:
            return
        
        print(f"Loading Vision-Language model: {self.model_id}")
        
        # Setup model path
        self.model_path = self._setup_model_path()
        
        # Load processor and model
        self._load_processor()
        self._load_vl_model_weights()
        
        self.is_loaded = True
        print(f"Vision-Language model loaded successfully: {self.model_id}")
    
    def _load_processor(self) -> None:
        """Load the processor for Vision-Language models."""
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print("Processor loaded successfully")
            
            # Update generation config with pad token if available
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'pad_token_id'):
                self.generation_config["pad_token_id"] = self.processor.tokenizer.pad_token_id
                
        except Exception as e:
            print(f"Warning: Failed to load processor ({e}), falling back to tokenizer")
            # Fallback to regular tokenizer
            self._load_tokenizer()
    
    def _load_vl_model_weights(self) -> None:
        """Load the Vision-Language model weights."""
        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }
        
        # Add quantization config if enabled
        if self.quantization_config:
            load_kwargs["quantization_config"] = self.quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"
        
        try:
            # Try AutoModelForVision2Seq first (modern VL models)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            print("Loaded model using AutoModelForVision2Seq")
            
        except Exception as e1:
            print(f"AutoModelForVision2Seq failed ({e1}), trying AutoModel...")
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
                print("Loaded model using AutoModel")
                
            except Exception as e2:
                print(f"AutoModel failed ({e2}), trying model-specific loader...")
                
                # Try Qwen-VL specific loader
                if "qwen" in self.model_id.lower() and "vl" in self.model_id.lower():
                    try:
                        from transformers import Qwen2VLForConditionalGeneration
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_path,
                            **load_kwargs
                        )
                        print("Loaded model using Qwen2VLForConditionalGeneration")
                    except ImportError:
                        raise ValueError(
                            f"Failed to load Qwen-VL model. Please ensure you have the latest transformers version.\n"
                            f"Try: pip install transformers>=4.37.0"
                        )
                    except Exception as e3:
                        raise ValueError(
                            f"Failed to load Vision-Language model with all available loaders:\n"
                            f"AutoModelForVision2Seq: {e1}\n"
                            f"AutoModel: {e2}\n"
                            f"Qwen2VLForConditionalGeneration: {e3}"
                        )
                else:
                    raise ValueError(
                        f"Failed to load Vision-Language model:\n"
                        f"AutoModelForVision2Seq: {e1}\n"
                        f"AutoModel: {e2}\n"
                        f"Model {self.model_id} may not be supported yet."
                    )
    
    def forward(self, messages: Optional[List[Dict[str, str]]] = None, images: Optional[List[Union[str, Image.Image]]] = None) -> str:
        """
        Generate a response using the Vision-Language model.
        
        Args:
            messages (Optional[List[Dict[str, str]]]): Optional messages to use
            images (Optional[List[Union[str, Image.Image]]]): Optional images to process
                
        Returns:
            str: Generated response text
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call _load_vl_model() first.")
        
        # Prepare messages
        prepared_messages = self._prepare_messages(messages)
        
        if not prepared_messages:
            raise ValueError("No messages to process")
        
        # Process images if provided
        processed_images = self._process_images(images) if images else []
        
        # Create input text from messages
        text_input = self._format_vl_messages(prepared_messages)
        
        try:
            # Process inputs with the processor
            if self.processor:
                if processed_images:
                    inputs = self.processor(
                        text=text_input,
                        images=processed_images,
                        return_tensors="pt",
                        padding=True
                    )
                else:
                    inputs = self.processor(
                        text=text_input,
                        return_tensors="pt",
                        padding=True
                    )
            else:
                # Fallback: tokenize text only
                inputs = self.tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding=True
                )
            
            # Move inputs to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # Decode response
            if self.processor:
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response (remove input text if it's repeated)
            response = self._clean_vl_response(response, text_input)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error during VL generation: {e}")
            # Fallback to text-only generation
            return super().forward(messages)
    
    def _process_images(self, images: List[Union[str, Image.Image]]) -> List[Image.Image]:
        """Process images for the model."""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # Load image from path or URL
                try:
                    if img.startswith(('http://', 'https://')):
                        import requests
                        from io import BytesIO
                        
                        response = requests.get(img)
                        img = Image.open(BytesIO(response.content))
                    else:
                        img = Image.open(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img}: {e}")
                    continue
            
            if isinstance(img, Image.Image):
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                processed_images.append(img)
        
        return processed_images
    
    def _format_vl_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Vision-Language models."""
        # This is a basic implementation - can be overridden by specific models
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
        
        formatted_parts.append("Assistant:")
        return "\n\n".join(formatted_parts)
    
    def _clean_vl_response(self, response: str, input_text: str) -> str:
        """Clean the VL model response."""
        # Remove input text if it appears at the beginning of response
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Assistant:", "Response:", "Output:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get VL model information."""
        info = super().get_model_info()
        info.update({
            "model_type": "Vision-Language",
            "supports_images": True,
            "processor_available": self.processor is not None
        })
        return info
    
    def process_image_with_text(self, image: Union[str, Image.Image], text: str) -> str:
        """
        Convenience method for processing a single image with text.
        
        Args:
            image: Image file path, URL, or PIL Image
            text: Text prompt for the image
            
        Returns:
            str: Model response
        """
        messages = [{"role": "user", "content": text}]
        return self.forward(messages, images=[image])
