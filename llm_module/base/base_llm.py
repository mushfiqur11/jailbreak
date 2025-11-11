"""
Abstract base class for all LLM implementations.

This module defines the core interface that all language model classes must implement,
providing a unified API for conversation management and text generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import copy


class BaseLLM(ABC):
    """
    Abstract base class for all language model implementations.
    
    This class defines the core interface that must be implemented by all
    language model classes, ensuring consistency across different model types
    (HuggingFace, OpenAI, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base LLM with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.system_prompt = None
        self.conversation = []
        self.is_loaded = False
        
        # Model information
        self.model_id = config.get("model_id", "")
        self.max_tokens = config.get("max_new_tokens", config.get("max_tokens", 512))
        self.temperature = config.get("temperature", 0.7)
        
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the language model and tokenizer.
        
        This method must be implemented by subclasses to handle the specific
        model loading process for their respective platforms (HuggingFace, OpenAI, etc.).
        """
        pass
    
    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for the model.
        
        Args:
            prompt (str): The system prompt to use for conversations
        """
        pass
    
    def get_system_prompt(self) -> Optional[str]:
        """
        Get the current system prompt.
        
        Returns:
            Optional[str]: The current system prompt, or None if not set
        """
        return self.system_prompt
    
    @abstractmethod
    def forward(self, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the language model.
        
        Args:
            messages (Optional[List[Dict[str, str]]]): Optional list of messages.
                If provided, uses these messages for generation without affecting
                the internal conversation state. If None, uses the internal conversation.
        
        Returns:
            str: Generated response text
        """
        pass
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role (str): Role of the message sender ("user", "assistant", "system")
            content (str): Content of the message
        """
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Role and content must be strings")
            
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be one of: user, assistant, system")
            
        self.conversation.append({"role": role, "content": content})
    
    def add_conversation(self, conversation: List[Dict[str, str]]) -> None:
        """
        Add multiple messages to the conversation history.
        
        Args:
            conversation (List[Dict[str, str]]): List of message dictionaries
        """
        if not isinstance(conversation, list):
            raise ValueError("Conversation must be a list")
            
        for message in conversation:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must have 'role' and 'content' keys")
                
            self.add_message(message["role"], message["content"])
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation = []
    
    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Get a copy of the current conversation.
        
        Returns:
            List[Dict[str, str]]: Copy of the conversation history
        """
        return copy.deepcopy(self.conversation)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        return {
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "conversation_length": len(self.conversation),
            "has_system_prompt": self.system_prompt is not None
        }
    
    def reset_conversation(self) -> None:
        """Reset conversation but keep system prompt."""
        self.conversation = []
    
    def _prepare_messages(self, messages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Prepare messages for generation, handling system prompt injection.
        
        Args:
            messages (Optional[List[Dict[str, str]]]): Optional external messages
            
        Returns:
            List[Dict[str, str]]: Prepared messages with system prompt if applicable
        """
        if messages is not None:
            # External messages - don't modify conversation state
            prepared_messages = copy.deepcopy(messages)
        else:
            # Use internal conversation
            prepared_messages = copy.deepcopy(self.conversation)
        
        # Add system prompt if it exists and not already present
        if self.system_prompt and (not prepared_messages or prepared_messages[0]["role"] != "system"):
            prepared_messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        return prepared_messages
    
    def __str__(self) -> str:
        """String representation of the LLM instance."""
        return f"{self.__class__.__name__}(model_id={self.model_id}, loaded={self.is_loaded})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the LLM instance."""
        return (f"{self.__class__.__name__}(model_id='{self.model_id}', "
                f"loaded={self.is_loaded}, conversations={len(self.conversation)})")
