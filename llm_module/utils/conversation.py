"""
Conversation management utilities.

This module provides tools for managing conversations, including history tracking,
conversation serialization, and conversation analytics.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class MessageRole(Enum):
    """Enumeration of message roles in conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role (str): The role of the message sender (system, user, assistant)
        content (str): The content of the message
        timestamp (datetime): When the message was created
        metadata (Dict[str, Any]): Additional metadata for the message
    """
    role: str
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary format."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ConversationManager:
    """
    Manager for handling conversations with language models.
    
    This class provides functionality for conversation tracking, history management,
    serialization, and analytics.
    """
    
    def __init__(self, conversation_id: Optional[str] = None, max_history: int = 1000):
        """
        Initialize the conversation manager.
        
        Args:
            conversation_id (Optional[str]): Unique identifier for the conversation
            max_history (int): Maximum number of messages to keep in history
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_history = max_history
        self.messages: List[Message] = []
        self.system_prompt: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: Union[str, MessageRole], content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role (Union[str, MessageRole]): Role of the message sender
            content (str): Content of the message
            metadata (Optional[Dict[str, Any]]): Additional metadata
        """
        if isinstance(role, MessageRole):
            role = role.value
        
        if role not in [r.value for r in MessageRole]:
            raise ValueError(f"Invalid role: {role}. Must be one of {[r.value for r in MessageRole]}")
        
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # Trim history if it exceeds max_history
        if len(self.messages) > self.max_history:
            # Keep system messages and trim oldest user/assistant messages
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM.value]
            other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM.value]
            
            # Keep the most recent messages
            messages_to_keep = self.max_history - len(system_messages)
            if messages_to_keep > 0:
                other_messages = other_messages[-messages_to_keep:]
            
            self.messages = system_messages + other_messages
    
    def add_conversation(self, conversation: List[Dict[str, str]]) -> None:
        """
        Add multiple messages to the conversation.
        
        Args:
            conversation (List[Dict[str, str]]): List of message dictionaries
        """
        for message in conversation:
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must have 'role' and 'content' keys")
            
            metadata = {k: v for k, v in message.items() if k not in ["role", "content"]}
            self.add_message(message["role"], message["content"], metadata or None)
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.
        
        Args:
            prompt (str): The system prompt
        """
        self.system_prompt = prompt
        
        # Remove existing system messages and add new one
        self.messages = [m for m in self.messages if m.role != MessageRole.SYSTEM.value]
        self.add_message(MessageRole.SYSTEM, prompt)
    
    def get_messages(self, include_system: bool = True, 
                    format_for_api: bool = True) -> List[Dict[str, Any]]:
        """
        Get messages in the conversation.
        
        Args:
            include_system (bool): Whether to include system messages
            format_for_api (bool): Whether to format for API calls (remove metadata/timestamp)
            
        Returns:
            List[Dict[str, Any]]: List of messages
        """
        messages = self.messages.copy()
        
        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM.value]
        
        if format_for_api:
            return [{"role": m.role, "content": m.content} for m in messages]
        else:
            return [m.to_dict() for m in messages]
    
    def get_last_messages(self, count: int, include_system: bool = False) -> List[Dict[str, Any]]:
        """
        Get the last N messages from the conversation.
        
        Args:
            count (int): Number of messages to retrieve
            include_system (bool): Whether to include system messages
            
        Returns:
            List[Dict[str, Any]]: Last N messages
        """
        messages = self.messages.copy()
        
        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM.value]
        
        last_messages = messages[-count:] if count > 0 else messages
        return [{"role": m.role, "content": m.content} for m in last_messages]
    
    def clear_conversation(self, keep_system_prompt: bool = True) -> None:
        """
        Clear the conversation history.
        
        Args:
            keep_system_prompt (bool): Whether to keep the system prompt
        """
        if keep_system_prompt and self.system_prompt:
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM.value]
            self.messages = system_messages
        else:
            self.messages = []
            self.system_prompt = None
        
        self.updated_at = datetime.now()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation.
        
        Returns:
            Dict[str, Any]: Conversation statistics
        """
        total_messages = len(self.messages)
        role_counts = {}
        total_tokens = 0
        
        for message in self.messages:
            role_counts[message.role] = role_counts.get(message.role, 0) + 1
            # Rough token estimation (4 characters per token)
            total_tokens += len(message.content) // 4
        
        duration = (self.updated_at - self.created_at).total_seconds()
        
        return {
            "conversation_id": self.conversation_id,
            "total_messages": total_messages,
            "role_distribution": role_counts,
            "estimated_tokens": total_tokens,
            "duration_seconds": duration,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "has_system_prompt": self.system_prompt is not None
        }
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save conversation to a JSON file.
        
        Args:
            file_path (str): Path to save the conversation
        """
        conversation_data = {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ConversationManager':
        """
        Load conversation from a JSON file.
        
        Args:
            file_path (str): Path to the conversation file
            
        Returns:
            ConversationManager: Loaded conversation manager
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversation = cls(
            conversation_id=data["conversation_id"],
            max_history=1000  # Default max_history
        )
        
        conversation.created_at = datetime.fromisoformat(data["created_at"])
        conversation.updated_at = datetime.fromisoformat(data["updated_at"])
        conversation.system_prompt = data.get("system_prompt")
        conversation.metadata = data.get("metadata", {})
        
        # Load messages
        for message_data in data["messages"]:
            message = Message.from_dict(message_data)
            conversation.messages.append(message)
        
        return conversation
    
    def export_for_training(self, format_type: str = "jsonl") -> List[Dict[str, Any]]:
        """
        Export conversation in a format suitable for training.
        
        Args:
            format_type (str): Export format ("jsonl", "alpaca", "sharegpt")
            
        Returns:
            List[Dict[str, Any]]: Formatted conversation data
        """
        if format_type == "jsonl":
            # Each message as a separate JSON line
            return [{"role": m.role, "content": m.content} for m in self.messages]
        
        elif format_type == "alpaca":
            # Alpaca format with instruction/output pairs
            result = []
            current_instruction = None
            system_context = self.system_prompt or ""
            
            for message in self.messages:
                if message.role == MessageRole.USER.value:
                    current_instruction = message.content
                elif message.role == MessageRole.ASSISTANT.value and current_instruction:
                    result.append({
                        "instruction": current_instruction,
                        "input": "",
                        "output": message.content,
                        "system": system_context
                    })
                    current_instruction = None
            
            return result
        
        elif format_type == "sharegpt":
            # ShareGPT format
            conversations = []
            current_conversation = []
            
            for message in self.messages:
                if message.role != MessageRole.SYSTEM.value:
                    current_conversation.append({
                        "from": "human" if message.role == MessageRole.USER.value else "gpt",
                        "value": message.content
                    })
            
            if current_conversation:
                conversations.append({"conversations": current_conversation})
            
            return conversations
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def get_conversation_turns(self) -> List[Dict[str, str]]:
        """
        Get conversation as user-assistant turns.
        
        Returns:
            List[Dict[str, str]]: List of conversation turns
        """
        turns = []
        current_turn = {}
        
        for message in self.messages:
            if message.role == MessageRole.USER.value:
                if current_turn:
                    turns.append(current_turn)
                current_turn = {"user": message.content}
            elif message.role == MessageRole.ASSISTANT.value and current_turn:
                current_turn["assistant"] = message.content
        
        if current_turn:
            turns.append(current_turn)
        
        return turns
    
    def analyze_conversation_flow(self) -> Dict[str, Any]:
        """
        Analyze the flow and patterns in the conversation.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        turns = self.get_conversation_turns()
        
        analysis = {
            "total_turns": len(turns),
            "avg_user_message_length": 0,
            "avg_assistant_message_length": 0,
            "user_message_lengths": [],
            "assistant_message_lengths": [],
            "conversation_topics": [],  # Could be enhanced with NLP
            "response_patterns": {}
        }
        
        user_lengths = []
        assistant_lengths = []
        
        for turn in turns:
            if "user" in turn:
                user_lengths.append(len(turn["user"]))
            if "assistant" in turn:
                assistant_lengths.append(len(turn["assistant"]))
        
        if user_lengths:
            analysis["avg_user_message_length"] = sum(user_lengths) / len(user_lengths)
            analysis["user_message_lengths"] = user_lengths
        
        if assistant_lengths:
            analysis["avg_assistant_message_length"] = sum(assistant_lengths) / len(assistant_lengths)
            analysis["assistant_message_lengths"] = assistant_lengths
        
        return analysis
    
    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)
    
    def __str__(self) -> str:
        """String representation of the conversation."""
        return f"Conversation {self.conversation_id} ({len(self.messages)} messages)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        stats = self.get_conversation_stats()
        return (f"ConversationManager(id='{self.conversation_id}', "
                f"messages={stats['total_messages']}, "
                f"tokens~{stats['estimated_tokens']})")
