prompt = """
Generalize the tactics
"""

def get_generalize_tactics_prompt() -> str:
    return prompt.format()


def parse_generalize_tactics_response(response: str):
    """
    Parse the response from generalize tactics prompt.
    
    Note: This prompt appears to be incomplete/placeholder.
    Currently just returns the raw response as a simple string.
    
    Args:
        response: Raw LLM response from generalize tactics prompt
        
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
