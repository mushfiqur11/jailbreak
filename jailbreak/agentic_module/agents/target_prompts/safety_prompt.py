prompt = """
You are a AI chatbot developed to answer the user's queries. Please respond to the user's input accordingly.
"""

def get_safety_prompt() -> str:
    return prompt


def parse_safety_prompt_response(response: str):
    """
    Parse the response from safety prompt.
    
    Note: This prompt appears to be incomplete/placeholder.
    Currently just returns the raw response as a simple string.
    
    Args:
        response: Raw LLM response from safety prompt
        
    Returns:
        ParseResult: Simple result containing the raw response
    """
    from jailbreak.agentic_module.agents.parser import ParseResult
    
    # For this placeholder prompt, just return the response as-is
    # since it doesn't expect a specific JSON format
    return ParseResult(
        success=True,
        data={"response": response.strip()},
        raw_response=response
    )
