"""
Abstract base class for Vision-Language model implementations.

This module provides the base class for all Vision-Language models,
extending HuggingFaceBase with image processing capabilities.
"""

from typing import Dict, Any, Optional, List, Union
from PIL import Image

from .hf_base import HuggingFaceBase


class VisionLanguageBase(HuggingFaceBase):
    """
    Abstract base class for Vision-Language models.
    
    This class extends HuggingFaceBase to handle vision-language models
    that can process both text and images.
    
    Note: Full VL model support is a placeholder for future implementation.
    Currently, this class provides the interface that VL models like QwenVL
    can extend, but actual VL functionality (image processing) is not yet
    implemented and will raise NotImplementedError.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Vision-Language base model.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        # Ensure supports_vision is set
        config["supports_vision"] = True
        super().__init__(config)
        
        # VL-specific attributes
        self.processor = None  # Will hold the image processor
        self.image_size = config.get("image_size", 224)
        self.max_image_tiles = config.get("max_image_tiles", 4)
        
        print(f"VisionLanguageBase initialized for: {self.model_id}")
        print("⚠️  Note: Full VL support is not yet implemented. Use text-only models for production.")
    
    def _load_model(self) -> None:
        """
        Load the VL model and processor.
        
        Note: This currently raises an error as full VL support is not implemented.
        """
        raise NotImplementedError(
            f"Vision-Language model loading is not yet implemented for {self.model_id}.\n\n"
            f"Current workaround:\n"
            f"- Use text-only models instead (e.g., Qwen/Qwen2.5-7B-Instruct)\n"
            f"- VL support will be added in a future update\n\n"
            f"If you need VL capabilities now, consider using the qwen_vl_utils library directly."
        )
    
    def forward(
        self, 
        messages: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[Union[str, Image.Image]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the vision-language model.
        
        Args:
            messages: Optional messages to use
            images: Optional list of images (file paths, URLs, or PIL Images)
                
        Returns:
            Dict[str, Any]: Dictionary containing response and metrics
        """
        if images:
            raise NotImplementedError(
                "Image processing is not yet implemented for VL models. "
                "Use text-only forward() without images parameter."
            )
        
        # Fall back to text-only processing
        return super().forward(messages)
    
    def process_image(self, image: Union[str, Image.Image]) -> Any:
        """
        Process an image for input to the model.
        
        Args:
            image: Image file path, URL, or PIL Image
            
        Returns:
            Processed image tensor
        """
        raise NotImplementedError(
            "Image processing is not yet implemented. "
            "This feature will be available in a future update."
        )
    
    def process_image_with_text(
        self, 
        image: Union[str, Image.Image], 
        text: str
    ) -> str:
        """
        Process an image with accompanying text prompt.
        
        Args:
            image: Image file path, URL, or PIL Image
            text: Text prompt for the image
            
        Returns:
            str: Model response
        """
        raise NotImplementedError(
            "Image-text processing is not yet implemented. "
            "This feature will be available in a future update."
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict[str, Any]: Model information including VL-specific details
        """
        info = super().get_model_info()
        info.update({
            "model_type": "VisionLanguage",
            "supports_vision": True,
            "image_size": self.image_size,
            "max_image_tiles": self.max_image_tiles,
            "vl_support_status": "placeholder - not fully implemented"
        })
        return info
