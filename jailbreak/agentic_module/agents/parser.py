"""
Universal parsing utilities for jailbreak agent responses.

This module provides robust JSON parsing capabilities that can handle various
formats of LLM responses including clean JSON, markdown-wrapped JSON, and
mixed text with embedded JSON blocks.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def parse_llm_response_to_dict(response: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Universal function to parse any input string and convert it to a Python dictionary.
    
    Handles various scenarios:
    A. Straightforward text with clean JSON that easily gets converted
    B. Texts that start with ```json ``` blocks or similar markdown formatting
    C. Strings with preceding/trailing text that has dictionary/json blocks between them
    D. Cases where there are no dictionaries - properly detects these without fallback
    
    Args:
        response (str): The raw response string from LLM
        
    Returns:
        Tuple[Optional[Dict[str, Any]], str]: 
            - Parsed dictionary if successful, None if no valid JSON found
            - Error message if parsing failed, empty string if successful
    """
    if not response or not isinstance(response, str):
        return None, "Invalid input: empty or non-string response"
    
    response = response.strip()
    if not response:
        return None, "Invalid input: empty response after stripping"
    
    # Strategy A: Try direct JSON parsing first (clean JSON case)
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            logger.debug("Successfully parsed clean JSON")
            return result, ""
        else:
            return None, f"Parsed JSON is not a dictionary, got {type(result)}"
    except json.JSONDecodeError:
        pass  # Continue to other strategies
    
    # Strategy B: Look for markdown JSON blocks (```json...``` or ```...```)
    json_block_patterns = [
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
        r'`json\s*\n?(.*?)\n?`',
        r'`\s*\n?(.*?)\n?`'
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, dict):
                    logger.debug(f"Successfully parsed JSON from markdown block using pattern: {pattern}")
                    return result, ""
            except json.JSONDecodeError:
                continue
    
    # Strategy C: Look for JSON-like structures with curly braces in mixed text
    # Find potential JSON objects by looking for balanced braces
    json_candidates = []
    
    # Pattern to find content between outermost braces
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, response, re.DOTALL)
    json_candidates.extend(matches)
    
    # More aggressive pattern for nested structures
    nested_brace_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    nested_matches = re.findall(nested_brace_pattern, response, re.DOTALL)
    json_candidates.extend(nested_matches)
    
    # Try parsing each candidate
    for candidate in json_candidates:
        try:
            # Clean up the candidate
            cleaned = candidate.strip()
            result = json.loads(cleaned)
            if isinstance(result, dict):
                logger.debug("Successfully parsed JSON from mixed text")
                return result, ""
        except json.JSONDecodeError:
            continue
    
    # Strategy D: Look for key-value patterns that might be malformed JSON
    # Try to fix common JSON issues
    json_fix_attempts = [
        # Remove leading/trailing non-JSON text more aggressively
        lambda x: re.search(r'\{.*\}', x, re.DOTALL).group(0) if re.search(r'\{.*\}', x, re.DOTALL) else None,
        # Fix single quotes to double quotes
        lambda x: x.replace("'", '"'),
        # Fix trailing commas
        lambda x: re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', x)),
        # Fix unquoted keys (basic cases)
        lambda x: re.sub(r'(\w+):', r'"\1":', x),
    ]
    
    for fix_func in json_fix_attempts:
        try:
            fixed_response = fix_func(response)
            if fixed_response:
                result = json.loads(fixed_response)
                if isinstance(result, dict):
                    logger.debug("Successfully parsed JSON after applying fixes")
                    return result, ""
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    
    # Final attempt: Extract JSON from the largest brace-enclosed section
    try:
        # Find the longest potential JSON string
        all_braces = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if all_braces:
            longest = max(all_braces, key=len)
            result = json.loads(longest)
            if isinstance(result, dict):
                logger.debug("Successfully parsed longest JSON candidate")
                return result, ""
    except json.JSONDecodeError:
        pass
    
    # Strategy D validation: Check if response truly contains no dictionary structure
    has_braces = '{' in response and '}' in response
    has_json_keywords = any(keyword in response.lower() for keyword in 
                           ['true', 'false', 'null', '":', '":'])
    
    if not has_braces:
        return None, "No dictionary structure detected: missing braces"
    elif not has_json_keywords:
        return None, "No dictionary structure detected: missing JSON formatting"
    else:
        return None, "Found potential JSON structure but unable to parse due to formatting issues"


def validate_required_fields(parsed_dict: Dict[str, Any], required_fields: list, 
                           field_types: Optional[Dict[str, type]] = None) -> Tuple[bool, str]:
    """
    Validate that required fields exist in parsed dictionary with basic type checking.
    
    Args:
        parsed_dict: Dictionary to validate
        required_fields: List of required field names
        field_types: Optional dictionary mapping field names to expected types
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(parsed_dict, dict):
        return False, "Input is not a dictionary"
    
    missing_fields = []
    type_errors = []
    
    for field in required_fields:
        if field not in parsed_dict:
            missing_fields.append(field)
        elif field_types and field in field_types:
            expected_type = field_types[field]
            actual_value = parsed_dict[field]
            
            # Basic type checking (not too strict)
            if expected_type == str and not isinstance(actual_value, str):
                type_errors.append(f"Field '{field}' should be string, got {type(actual_value)}")
            elif expected_type == list and not isinstance(actual_value, list):
                type_errors.append(f"Field '{field}' should be list, got {type(actual_value)}")
            elif expected_type == dict and not isinstance(actual_value, dict):
                type_errors.append(f"Field '{field}' should be dict, got {type(actual_value)}")
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    if type_errors:
        return False, f"Type validation errors: {'; '.join(type_errors)}"
    
    return True, ""


class ParseResult:
    """Result container for parsing operations."""
    
    def __init__(self, success: bool, data: Optional[Dict[str, Any]] = None, 
                 error: str = "", raw_response: str = ""):
        self.success = success
        self.data = data
        self.error = error
        self.raw_response = raw_response
    
    def __bool__(self) -> bool:
        return self.success
    
    def __repr__(self) -> str:
        if self.success:
            return f"ParseResult(success=True, data_keys={list(self.data.keys()) if self.data else []})"
        else:
            return f"ParseResult(success=False, error='{self.error}')"


def safe_parse_with_validation(response: str, required_fields: list, 
                              field_types: Optional[Dict[str, type]] = None,
                              parser_name: str = "generic") -> ParseResult:
    """
    Safely parse response with validation and comprehensive error handling.
    
    Args:
        response: Raw LLM response
        required_fields: List of required field names
        field_types: Optional field type specifications
        parser_name: Name of the parser for logging
        
    Returns:
        ParseResult: Comprehensive parsing result
    """
    try:
        # Step 1: Parse to dictionary
        parsed_dict, parse_error = parse_llm_response_to_dict(response)
        
        if parsed_dict is None:
            logger.warning(f"{parser_name} parser: Failed to parse JSON - {parse_error}")
            return ParseResult(
                success=False, 
                error=f"JSON parsing failed: {parse_error}",
                raw_response=response
            )
        
        # Step 2: Validate required fields
        is_valid, validation_error = validate_required_fields(parsed_dict, required_fields, field_types)
        
        if not is_valid:
            logger.warning(f"{parser_name} parser: Validation failed - {validation_error}")
            return ParseResult(
                success=False,
                data=parsed_dict,
                error=f"Validation failed: {validation_error}",
                raw_response=response
            )
        
        logger.info(f"{parser_name} parser: Successfully parsed and validated response")
        return ParseResult(
            success=True,
            data=parsed_dict,
            raw_response=response
        )
        
    except Exception as e:
        logger.error(f"{parser_name} parser: Unexpected error - {str(e)}")
        return ParseResult(
            success=False,
            error=f"Unexpected parsing error: {str(e)}",
            raw_response=response
        )
