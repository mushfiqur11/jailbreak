"""
Utility functions for the jailbreak agents module.

This module provides helper functions for loading and validating
tactics configuration and other agent utilities.
"""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_initial_tactics(file_path: str) -> Dict[str, Any]:
    """
    Load initial tactics from a JSON file.
    
    Args:
        file_path (str): Path to the tactics JSON file
        
    Returns:
        Dict[str, Any]: Dictionary containing tactics data
        
    Raises:
        FileNotFoundError: If the tactics file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tactics_data = json.load(f)
        
        logger.info(f"Successfully loaded tactics from {file_path}")
        return tactics_data
        
    except FileNotFoundError:
        logger.error(f"Tactics file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in tactics file {file_path}: {e}")
        raise


def validate_tactics_structure(tactics_data: Dict[str, Any]) -> bool:
    """
    Basic validation of tactics data structure.
    
    Args:
        tactics_data (Dict[str, Any]): Tactics data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Simple validation: check if it's a dict and not empty
    if not isinstance(tactics_data, dict):
        logger.warning("Tactics data is not a dictionary")
        return False
    
    if not tactics_data:
        logger.warning("Tactics data is empty")
        return False
    
    logger.debug("Tactics structure validation passed")
    return True
