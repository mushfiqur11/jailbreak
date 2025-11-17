"""
Conversation management for multi-agent jailbreaking pipeline.

This module provides conversation management with role switching support
for attacker-target interactions.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history with role switching between attacker and target perspectives.
    
    The conversation maintains turns with proper role assignment:
    - For attacker: attacker=assistant, target=user
    - For target: target=assistant, attacker=user  
    - For judge: entire conversation as single message
    """
    
    def __init__(self):
        """Initialize empty conversation manager."""
        self.turns = []
        logger.debug("ConversationManager initialized")
    
    def add_turn(self, attacker_prompt: str, target_response: str, 
                 turn_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn.
        
        Args:
            attacker_prompt (str): The attacker's prompt
            target_response (str): The target's response
            turn_metadata (Optional[Dict[str, Any]]): Additional metadata for the turn
        """
        turn_id = len(self.turns) + 1
        
        turn_data = {
            "turn_id": turn_id,
            "attacker_prompt": attacker_prompt,
            "target_response": target_response,
            "metadata": turn_metadata or {}
        }
        
        self.turns.append(turn_data)
        logger.debug(f"Added turn {turn_id} to conversation")
    
    def get_attacker_conversation(self) -> List[Dict[str, str]]:
        """
        Get conversation from attacker's perspective.
        
        In this format:
        - attacker prompts are 'assistant' messages  
        - target responses are 'user' messages
        
        Returns:
            List[Dict[str, str]]: Conversation in OpenAI format for attacker
        """
        conversation = []
        
        for turn in self.turns:
            # Attacker prompt as assistant message
            conversation.append({
                "role": "assistant",
                "content": turn["attacker_prompt"]
            })
            
            # Target response as user message  
            conversation.append({
                "role": "user", 
                "content": turn["target_response"]
            })
        
        logger.debug(f"Generated attacker conversation with {len(conversation)} messages")
        return conversation
    
    def get_target_conversation(self) -> List[Dict[str, str]]:
        """
        Get conversation from target's perspective.
        
        In this format:
        - attacker prompts are 'user' messages
        - target responses are 'assistant' messages
        
        Returns:
            List[Dict[str, str]]: Conversation in OpenAI format for target
        """
        conversation = []
        
        for turn in self.turns:
            # Attacker prompt as user message
            conversation.append({
                "role": "user",
                "content": turn["attacker_prompt"] 
            })
            
            # Target response as assistant message
            conversation.append({
                "role": "assistant",
                "content": turn["target_response"]
            })
        
        logger.debug(f"Generated target conversation with {len(conversation)} messages")
        return conversation
    
    def get_judge_conversation_string(self) -> str:
        """
        Get conversation as formatted string for judge evaluation.
        
        Returns:
            str: Formatted conversation string for judge
        """
        if not self.turns:
            return "No conversation yet."
        
        conversation_parts = []
        for turn in self.turns:
            conversation_parts.append(f"Attacker: {turn['attacker_prompt']}")
            conversation_parts.append(f"Target: {turn['target_response']}")
        
        conversation_str = "\n\n".join(conversation_parts)
        logger.debug(f"Generated judge conversation string with {len(self.turns)} turns")
        return conversation_str
    
    def get_last_turn(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent conversation turn.
        
        Returns:
            Optional[Dict[str, Any]]: Last turn data or None if no turns
        """
        return self.turns[-1] if self.turns else None
    
    def get_turn_count(self) -> int:
        """
        Get the number of conversation turns.
        
        Returns:
            int: Number of turns in conversation
        """
        return len(self.turns)
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()
        logger.debug("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the conversation.
        
        Returns:
            Dict[str, Any]: Conversation summary statistics
        """
        if not self.turns:
            return {
                "turn_count": 0,
                "total_attacker_chars": 0,
                "total_target_chars": 0,
                "avg_attacker_length": 0,
                "avg_target_length": 0
            }
        
        total_attacker_chars = sum(len(turn["attacker_prompt"]) for turn in self.turns)
        total_target_chars = sum(len(turn["target_response"]) for turn in self.turns)
        
        return {
            "turn_count": len(self.turns),
            "total_attacker_chars": total_attacker_chars,
            "total_target_chars": total_target_chars, 
            "avg_attacker_length": total_attacker_chars / len(self.turns),
            "avg_target_length": total_target_chars / len(self.turns)
        }
    
    def export_for_storage(self) -> List[Dict[str, Any]]:
        """
        Export conversation in format suitable for JSON storage.
        
        Returns:
            List[Dict[str, Any]]: Complete conversation data for storage
        """
        return [turn.copy() for turn in self.turns]
    
    def import_from_storage(self, stored_turns: List[Dict[str, Any]]) -> None:
        """
        Import conversation from stored format.
        
        Args:
            stored_turns (List[Dict[str, Any]]): Stored conversation data
        """
        self.turns = [turn.copy() for turn in stored_turns]
        logger.info(f"Imported conversation with {len(self.turns)} turns")
