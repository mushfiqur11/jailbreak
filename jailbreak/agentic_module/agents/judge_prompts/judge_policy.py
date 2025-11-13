prompt = """
This is a judgement policy
"""

def get_judge_policy_prompt() -> str:
    return prompt


def parse_judge_policy_response(response: str):
    """
    Parse the response from judge policy prompt.
    
    Note: This prompt appears to be incomplete/placeholder.
    Currently just returns the raw response as a simple string.
    
    Args:
        response: Raw LLM response from judge policy prompt
        
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
