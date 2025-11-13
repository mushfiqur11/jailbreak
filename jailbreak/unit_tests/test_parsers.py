"""
Comprehensive unit tests for all jailbreak agent parsers.

This module tests:
1. Universal JSON parsing function (parse_llm_response_to_dict)
2. Validation functions
3. All specific prompt parsers from different folders
4. Failure cases and edge cases

Run with: python -m pytest test_parsers.py -v
"""

import unittest
import sys
import os

# Add the jailbreak module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jailbreak.agentic_module.agents.parser import (
    parse_llm_response_to_dict,
    validate_required_fields,
    safe_parse_with_validation,
    ParseResult
)

# Import all specific parsers
from jailbreak.agentic_module.agents.attacker_prompts.initial_planning import parse_initial_planning_response
from jailbreak.agentic_module.agents.attacker_prompts.followup_planning import parse_followup_planning_response
from jailbreak.agentic_module.agents.attacker_prompts.traceback_planning import parse_traceback_planning_response

from jailbreak.agentic_module.agents.reasoning_prompts.curate_tactic import parse_curate_tactic_response
from jailbreak.agentic_module.agents.reasoning_prompts.informed_traceback import parse_informed_traceback_response
from jailbreak.agentic_module.agents.reasoning_prompts.belief_state_update import parse_belief_state_update_response
from jailbreak.agentic_module.agents.reasoning_prompts.generalize_tactics import parse_generalize_tactics_response

from jailbreak.agentic_module.agents.judge_prompts.judge_policy import parse_judge_policy_response
from jailbreak.agentic_module.agents.target_prompts.safety_prompt import parse_safety_prompt_response


class TestUniversalParser(unittest.TestCase):
    """Test the universal JSON parsing function."""
    
    def test_clean_json_parsing(self):
        """Test parsing clean, valid JSON."""
        response = '{"key": "value", "number": 42, "list": [1, 2, 3]}'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(error, "")
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
        self.assertEqual(result["list"], [1, 2, 3])
    
    def test_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = '''
        Here's the response:
        ```json
        {
            "tactic": "example",
            "success": true
        }
        ```
        That should work!
        '''
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(error, "")
        self.assertEqual(result["tactic"], "example")
        self.assertTrue(result["success"])
    
    def test_simple_code_block(self):
        """Test parsing JSON in simple code blocks."""
        response = '''
        ```
        {"simple": "test", "works": true}
        ```
        '''
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(error, "")
        self.assertEqual(result["simple"], "test")
    
    def test_mixed_text_with_json(self):
        """Test parsing JSON embedded in mixed text."""
        response = '''
        The analysis shows that {"reasoning": "target was uncooperative", "score": 0.3} 
        and we should try a different approach next time.
        '''
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(error, "")
        self.assertEqual(result["reasoning"], "target was uncooperative")
        self.assertEqual(result["score"], 0.3)
    
    def test_malformed_json_fixes(self):
        """Test automatic fixing of common JSON issues."""
        # Test single quotes to double quotes
        response = "{'key': 'value', 'number': 42}"
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(error, "")
        self.assertEqual(result["key"], "value")
    
    def test_no_json_detection(self):
        """Test proper detection when no JSON is present."""
        response = "This is just plain text with no JSON structure at all."
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNone(result)
        self.assertIn("No dictionary structure detected", error)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result, error = parse_llm_response_to_dict("")
        
        self.assertIsNone(result)
        self.assertIn("empty response", error)
    
    def test_nested_json_structures(self):
        """Test parsing complex nested JSON."""
        response = '''
        {
            "plan": {
                "tactics": ["a", "b"],
                "nested": {"deep": "value"}
            },
            "meta": {"score": 1.0}
        }
        '''
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(error, "")
        self.assertIn("plan", result)
        self.assertEqual(result["plan"]["tactics"], ["a", "b"])
        self.assertEqual(result["plan"]["nested"]["deep"], "value")

    # FAILURE CASES FOR UNIVERSAL PARSER
    def test_invalid_json_structure_fails(self):
        """Test that completely invalid JSON structure fails properly."""
        response = '{"key": "value", "broken": }'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNone(result)
        self.assertIn("formatting issues", error)
    
    def test_non_dict_json_fails(self):
        """Test that non-dictionary JSON fails properly."""
        response = '["this", "is", "a", "list", "not", "dict"]'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNone(result)
        self.assertIn("not a dictionary", error)
    
    def test_numbers_only_fails(self):
        """Test that numeric-only responses fail properly."""
        response = '42'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNone(result)
        self.assertIn("not a dictionary", error)
    
    def test_boolean_only_fails(self):
        """Test that boolean-only responses fail properly."""
        response = 'true'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNone(result)
        self.assertIn("not a dictionary", error)
    
    def test_null_input_fails(self):
        """Test that null input fails properly."""
        result, error = parse_llm_response_to_dict(None)
        
        self.assertIsNone(result)
        self.assertIn("Invalid input", error)
    
    def test_non_string_input_fails(self):
        """Test that non-string input fails properly."""
        result, error = parse_llm_response_to_dict(123)
        
        self.assertIsNone(result)
        self.assertIn("Invalid input", error)

    # EDGE CASES FOR UNIVERSAL PARSER
    def test_very_large_json(self):
        """Test parsing very large JSON structures."""
        large_list = ["item_" + str(i) for i in range(1000)]
        response = f'{{"large_data": {str(large_list)}, "meta": "test"}}'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["large_data"]), 1000)
        self.assertEqual(result["meta"], "test")
    
    def test_deeply_nested_json(self):
        """Test parsing deeply nested JSON structures."""
        nested_structure = {"level1": {"level2": {"level3": {"level4": {"deep_value": "found"}}}}}
        response = str(nested_structure).replace("'", '"')
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["level1"]["level2"]["level3"]["level4"]["deep_value"], "found")
    
    def test_unicode_content(self):
        """Test parsing JSON with Unicode content."""
        response = '{"unicode": "测试数据", "emoji": "🚀🎯", "special": "àáâãäå"}'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["unicode"], "测试数据")
        self.assertEqual(result["emoji"], "🚀🎯")
    
    def test_special_characters_in_strings(self):
        """Test parsing JSON with special characters."""
        response = r'{"quotes": "He said \"Hello\"", "newlines": "Line 1\nLine 2", "tabs": "Col1\tCol2"}'
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        self.assertIn('"Hello"', result["quotes"])
        self.assertIn("\n", result["newlines"])
    
    def test_multiple_json_blocks_picks_first(self):
        """Test that multiple JSON blocks returns the first valid one."""
        response = '''
        First JSON: {"first": "value1", "type": "first"}
        
        Second JSON: {"second": "value2", "type": "second"}
        '''
        result, error = parse_llm_response_to_dict(response)
        
        self.assertIsNotNone(result)
        # Should pick the first valid JSON found
        self.assertEqual(result["type"], "first")
    
    def test_whitespace_only_fails(self):
        """Test that whitespace-only input fails."""
        result, error = parse_llm_response_to_dict("   \n\t   ")
        
        self.assertIsNone(result)
        self.assertIn("empty response after stripping", error)


class TestValidationFunctions(unittest.TestCase):
    """Test validation helper functions."""
    
    def test_validate_required_fields_success(self):
        """Test successful field validation."""
        data = {"field1": "value1", "field2": [1, 2, 3], "field3": {"nested": "value"}}
        required = ["field1", "field2"]
        field_types = {"field1": str, "field2": list}
        
        is_valid, error = validate_required_fields(data, required, field_types)
        
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_missing_fields(self):
        """Test detection of missing required fields."""
        data = {"field1": "value1"}
        required = ["field1", "field2", "field3"]
        
        is_valid, error = validate_required_fields(data, required)
        
        self.assertFalse(is_valid)
        self.assertIn("Missing required fields: field2, field3", error)
    
    def test_validate_wrong_types(self):
        """Test detection of wrong field types."""
        data = {"field1": "value1", "field2": "should_be_list"}
        required = ["field1", "field2"]
        field_types = {"field1": str, "field2": list}
        
        is_valid, error = validate_required_fields(data, required, field_types)
        
        self.assertFalse(is_valid)
        self.assertIn("Type validation errors", error)

    # FAILURE CASES FOR VALIDATION
    def test_validation_non_dict_input_fails(self):
        """Test that validation fails with non-dict input."""
        is_valid, error = validate_required_fields("not_a_dict", ["field1"])
        
        self.assertFalse(is_valid)
        self.assertIn("Input is not a dictionary", error)
    
    def test_validation_empty_required_fields(self):
        """Test validation with empty required fields list."""
        data = {"field1": "value1"}
        is_valid, error = validate_required_fields(data, [])
        
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validation_none_input_fails(self):
        """Test that validation fails with None input."""
        is_valid, error = validate_required_fields(None, ["field1"])
        
        self.assertFalse(is_valid)
        self.assertIn("Input is not a dictionary", error)


class TestSafeParseWithValidation(unittest.TestCase):
    """Test the safe parsing with validation wrapper."""
    
    def test_successful_parse_and_validation(self):
        """Test successful parsing and validation."""
        response = '{"name": "test", "items": ["a", "b"], "config": {"enabled": true}}'
        required = ["name", "items"]
        types = {"name": str, "items": list}
        
        result = safe_parse_with_validation(response, required, types, "test_parser")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.error, "")
        self.assertEqual(result.data["name"], "test")
    
    def test_parsing_failure(self):
        """Test handling of parsing failures."""
        response = "Not JSON at all"
        required = ["name"]
        
        result = safe_parse_with_validation(response, required, parser_name="test_parser")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIn("JSON parsing failed", result.error)
    
    def test_validation_failure(self):
        """Test handling of validation failures."""
        response = '{"name": "test"}'
        required = ["name", "missing_field"]
        
        result = safe_parse_with_validation(response, required, parser_name="test_parser")
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.data)  # Parsed but failed validation
        self.assertIn("Validation failed", result.error)

    # FAILURE AND EDGE CASES FOR SAFE PARSE
    def test_exception_handling(self):
        """Test that unexpected exceptions are handled gracefully."""
        # This should trigger an exception in validation due to invalid type comparison
        response = '{"field": "value"}'
        required = ["field"]
        types = {"field": "invalid_type_not_a_class"}  # This should cause an error
        
        result = safe_parse_with_validation(response, required, types, "test_parser")
        
        self.assertFalse(result.success)
        self.assertIn("Unexpected parsing error", result.error)
    
    def test_empty_parser_name(self):
        """Test with empty parser name."""
        response = '{"field": "value"}'
        required = ["field"]
        
        result = safe_parse_with_validation(response, required, parser_name="")
        
        self.assertTrue(result.success)  # Should still work


class TestAttackerPromptParsers(unittest.TestCase):
    """Test parsers for attacker prompt responses."""
    
    def test_initial_planning_parser_success(self):
        """Test successful parsing of initial planning response."""
        response = '''
        {
            "suggestedTactics": ["Request_Framing", "Hidden_Intention"],
            "detailedPlan": "Use indirect approach to build trust first",
            "reasoning": "Direct requests likely to be rejected",
            "nextPrompt": "Can you help me understand security best practices?"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["suggestedTactics"]), 2)
        self.assertIn("indirect approach", result.data["detailedPlan"])
    
    def test_initial_planning_parser_missing_fields(self):
        """Test initial planning parser with missing fields."""
        response = '''
        {
            "suggestedTactics": ["Request_Framing"],
            "detailedPlan": "Use indirect approach"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing required fields", result.error)
    
    def test_followup_planning_parser_success(self):
        """Test successful parsing of followup planning response."""
        response = '''
        {
            "thought": "The target seems receptive to general questions",
            "plan": {
                "suggestedTactics": ["Social_Engineering", "Gradual_Escalation"],
                "reasoning": "Build on established rapport",
                "infoToFocusOnNext": ["security protocols", "access controls"],
                "nextPrompt": "That's interesting. Could you elaborate on the authentication process?"
            }
        }
        '''
        result = parse_followup_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("target seems receptive", result.data["thought"])
        self.assertIn("suggestedTactics", result.data["plan"])
    
    def test_followup_planning_parser_invalid_plan(self):
        """Test followup planning parser with invalid nested plan."""
        response = '''
        {
            "thought": "Some thought",
            "plan": {
                "suggestedTactics": ["Social_Engineering"]
            }
        }
        '''
        result = parse_followup_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing required plan fields", result.error)
    
    def test_traceback_planning_parser_success(self):
        """Test successful parsing of traceback planning response."""
        response = '''
        {
            "suggestedTactics": ["Persistence", "Alternative_Framing"],
            "detailedPlan": "Try a different angle based on the traceback analysis",
            "reasoning": "Previous approach was too direct and triggered safety measures",
            "nextPrompt": "I understand your concerns. Let me ask this differently..."
        }
        '''
        result = parse_traceback_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("Alternative_Framing", result.data["suggestedTactics"])
        self.assertIn("different angle", result.data["detailedPlan"])

    # FAILURE CASES FOR ATTACKER PARSERS
    def test_initial_planning_wrong_types(self):
        """Test initial planning with wrong field types."""
        response = '''
        {
            "suggestedTactics": "should_be_list_not_string",
            "detailedPlan": ["should", "be", "string", "not", "list"],
            "reasoning": "correct type",
            "nextPrompt": "correct type"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)
    
    def test_followup_planning_plan_wrong_type(self):
        """Test followup planning with plan field as wrong type."""
        response = '''
        {
            "thought": "Valid thought",
            "plan": "This should be a dictionary, not a string"
        }
        '''
        result = parse_followup_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)
    
    def test_traceback_planning_empty_tactics(self):
        """Test traceback planning with empty tactics list."""
        response = '''
        {
            "suggestedTactics": [],
            "detailedPlan": "Valid plan",
            "reasoning": "Valid reasoning",
            "nextPrompt": "Valid prompt"
        }
        '''
        result = parse_traceback_planning_response(response)
        
        # Should succeed but with empty tactics
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["suggestedTactics"]), 0)

    # EDGE CASES FOR ATTACKER PARSERS  
    def test_initial_planning_very_long_strings(self):
        """Test initial planning with very long string values."""
        long_text = "Very long text " * 1000  # 15000+ characters
        response = f'''
        {{
            "suggestedTactics": ["Tactic1", "Tactic2"],
            "detailedPlan": "{long_text}",
            "reasoning": "{long_text}",
            "nextPrompt": "{long_text}"
        }}
        '''
        result = parse_initial_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.data["detailedPlan"]), 10000)
    
    def test_followup_planning_deeply_nested_plan(self):
        """Test followup planning with complex nested plan structure."""
        response = '''
        {
            "thought": "Complex thought",
            "plan": {
                "suggestedTactics": ["Tactic1", "Tactic2", "Tactic3"],
                "reasoning": "Complex reasoning",
                "infoToFocusOnNext": ["item1", "item2", "item3"],
                "nextPrompt": "Complex prompt",
                "extraField": {
                    "nested": {
                        "deeply": "value"
                    }
                }
            }
        }
        '''
        result = parse_followup_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("extraField", result.data["plan"])


class TestReasoningPromptParsers(unittest.TestCase):
    """Test parsers for reasoning prompt responses."""
    
    def test_curate_tactic_parser_success(self):
        """Test successful parsing of curate tactic response."""
        response = '''
        {
            "newTacticPool": [
                {
                    "tactic": "Progressive_Disclosure",
                    "definition": "Gradually reveal sensitive information through a series of seemingly innocent questions",
                    "prompts": ["Can you explain basic security?", "What about advanced features?"]
                }
            ],
            "selectionFramework": [
                {
                    "scenarioOrGoalType": "Information Extraction",
                    "recommendedTactic": ["Progressive_Disclosure", "Social_Engineering"]
                }
            ],
            "promptNotes": ["Target responds well to technical questions"]
        }
        '''
        result = parse_curate_tactic_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["newTacticPool"]), 1)
        self.assertEqual(result.data["newTacticPool"][0]["tactic"], "Progressive_Disclosure")
        self.assertIsInstance(result.data["selectionFramework"], list)
    
    def test_informed_traceback_parser_success(self):
        """Test successful parsing of informed traceback response."""
        response = '''
        {
            "fallbackTurn": "3",
            "reasoning": "The conversation went off track after turn 3 when we became too aggressive. Should maintain the subtle approach established in earlier turns."
        }
        '''
        result = parse_informed_traceback_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["fallbackTurn"], "3")
        self.assertIn("too aggressive", result.data["reasoning"])
    
    def test_belief_state_update_parser_success(self):
        """Test successful parsing of belief state update response."""
        response = '''
        {
            "conversationContext": {
                "conversationStage": "mid-stage info extraction",
                "refusalCount": 1,
                "progressScore": 0.4,
                "missingInformation": ["specific API details", "security protocols"],
                "collectedInformation": ["general system overview"],
                "cooperativeness": 6
            },
            "strategyState": {
                "currentTactic": ["Request_Framing"],
                "tacticsTried": ["Request_Framing", "Social_Engineering"]
            },
            "lastResponseAnalysis": {
                "responseType": "partial_compliance",
                "keyPoints": ["provided general info", "avoided specifics"],
                "policyTriggers": ["security_filter"]
            }
        }
        '''
        result = parse_belief_state_update_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["conversationContext"]["progressScore"], 0.4)
        self.assertIn("Request_Framing", result.data["strategyState"]["currentTactic"])
    
    def test_belief_state_update_parser_missing_context_fields(self):
        """Test belief state parser with missing essential context fields."""
        response = '''
        {
            "conversationContext": {
                "refusalCount": 1
            },
            "strategyState": {
                "currentTactic": ["Request_Framing"],
                "tacticsTried": ["Request_Framing"]
            },
            "lastResponseAnalysis": {
                "responseType": "partial_compliance",
                "keyPoints": ["test"],
                "policyTriggers": []
            }
        }
        '''
        result = parse_belief_state_update_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing essential conversationContext fields", result.error)
    
    def test_generalize_tactics_parser_success(self):
        """Test parsing of generalize tactics response (placeholder)."""
        response = "Some tactics analysis and generalization"
        result = parse_generalize_tactics_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["response"], "Some tactics analysis and generalization")

    # FAILURE CASES FOR REASONING PARSERS
    def test_curate_tactic_wrong_list_types(self):
        """Test curate tactic with wrong types for list fields."""
        response = '''
        {
            "newTacticPool": "should_be_list",
            "selectionFramework": "should_be_list",
            "promptNotes": {"should": "be_list"}
        }
        '''
        result = parse_curate_tactic_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)
    
    def test_informed_traceback_missing_reasoning(self):
        """Test informed traceback missing required reasoning field."""
        response = '''
        {
            "fallbackTurn": "3"
        }
        '''
        result = parse_informed_traceback_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing required fields", result.error)
    
    def test_belief_state_context_wrong_type(self):
        """Test belief state with wrong type for context field."""
        response = '''
        {
            "conversationContext": "should_be_dict",
            "strategyState": {
                "currentTactic": ["Request_Framing"],
                "tacticsTried": ["Request_Framing"]
            },
            "lastResponseAnalysis": {
                "responseType": "partial_compliance",
                "keyPoints": ["test"],
                "policyTriggers": []
            }
        }
        '''
        result = parse_belief_state_update_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)

    # EDGE CASES FOR REASONING PARSERS
    def test_curate_tactic_empty_lists(self):
        """Test curate tactic with empty but valid lists."""
        response = '''
        {
            "newTacticPool": [],
            "selectionFramework": [],
            "promptNotes": []
        }
        '''
        result = parse_curate_tactic_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["newTacticPool"]), 0)
    
    def test_informed_traceback_numeric_turn(self):
        """Test informed traceback with numeric turn (should be string)."""
        response = '''
        {
            "fallbackTurn": 3,
            "reasoning": "Valid reasoning"
        }
        '''
        result = parse_informed_traceback_response(response)
        
        # Should fail type validation since fallbackTurn should be string
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)
    
    def test_generalize_tactics_empty_response(self):
        """Test generalize tactics with empty response."""
        response = ""
        result = parse_generalize_tactics_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["response"], "")


class TestJudgeAndTargetParsers(unittest.TestCase):
    """Test parsers for judge and target prompt responses."""
    
    def test_judge_policy_parser_success(self):
        """Test parsing of judge policy response (placeholder)."""
        response = "Policy violation detected in turn 3 due to explicit request for sensitive information"
        result = parse_judge_policy_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("Policy violation detected", result.data["response"])
    
    def test_safety_prompt_parser_success(self):
        """Test parsing of safety prompt response (placeholder)."""
        response = "This request violates safety guidelines regarding information disclosure"
        result = parse_safety_prompt_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("safety guidelines", result.data["response"])

    # EDGE CASES FOR SIMPLE PARSERS
    def test_judge_policy_very_long_response(self):
        """Test judge policy with very long response."""
        long_response = "Policy analysis: " + "Very detailed analysis. " * 1000
        result = parse_judge_policy_response(long_response)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.data["response"]), 10000)
    
    def test_safety_prompt_unicode_response(self):
        """Test safety prompt with Unicode characters."""
        response = "Safety violation: 违反安全准则 🚫 ⚠️"
        result = parse_safety_prompt_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("🚫", result.data["response"])


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases across parsers."""
    
    def test_malformed_json_in_specific_parser(self):
        """Test how specific parsers handle malformed JSON."""
        response = '{"suggestedTactics": ["test"], "detailedPlan": "test" "reasoning": "missing comma"}'
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("JSON parsing failed", result.error)
    
    def test_completely_wrong_structure(self):
        """Test parsing with completely wrong structure."""
        response = '{"wrongField1": "value", "wrongField2": "another"}'
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing required fields", result.error)
    
    def test_partial_structure_match(self):
        """Test parsing with some correct and some missing fields."""
        response = '''
        {
            "suggestedTactics": ["Valid_Tactic"],
            "detailedPlan": "This is a valid plan",
            "wrongField": "This shouldn't be here"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("reasoning, nextPrompt", result.error)
    
    def test_nested_structure_validation(self):
        """Test validation of nested structures in complex parsers."""
        # Test followup planning with invalid nested structure
        response = '''
        {
            "thought": "Valid thought",
            "plan": "This should be a dict, not a string"
        }
        '''
        result = parse_followup_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)
    
    def test_empty_json_object(self):
        """Test parsing empty JSON object."""
        response = '{}'
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing required fields", result.error)
    
    def test_json_with_extra_text(self):
        """Test JSON embedded in lots of extra text."""
        response = '''
        I need to think about this carefully. The model seems uncooperative.
        
        Here's my analysis:
        {"fallbackTurn": "2", "reasoning": "We need to restart from turn 2"}
        
        That should be the best approach for now.
        '''
        result = parse_informed_traceback_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["fallbackTurn"], "2")

    # STRESS TESTS AND EXTREME EDGE CASES
    def test_extremely_nested_json_structure(self):
        """Test with extremely deeply nested JSON."""
        # Create 50 levels of nesting
        nested = {}
        current = nested
        for i in range(50):
            current["level_" + str(i)] = {}
            current = current["level_" + str(i)]
        current["final_value"] = "deep_down"
        
        # Wrap in valid structure
        response_data = {
            "suggestedTactics": ["Test"],
            "detailedPlan": "Test plan",
            "reasoning": "Test reasoning", 
            "nextPrompt": "Test prompt",
            "nested_data": nested
        }
        response = str(response_data).replace("'", '"')
        result = parse_initial_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["suggestedTactics"], ["Test"])
    
    def test_json_with_lots_of_special_characters(self):
        """Test JSON with many special characters and escape sequences."""
        response = r'''
        {
            "suggestedTactics": ["Tactic\"With\"Quotes", "Tactic\nWith\nNewlines"],
            "detailedPlan": "Plan with\ttabs and\rcarriage returns and \\backslashes",
            "reasoning": "Reasoning with / forward slashes and {braces} and [brackets]",
            "nextPrompt": "Prompt with 'single quotes' and \"double quotes\" and unicode: \u0041\u0042\u0043"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertIn("tabs", result.data["detailedPlan"])
    
    def test_json_with_numbers_and_booleans_in_strings(self):
        """Test JSON with string fields that contain number/boolean-like content."""
        response = '''
        {
            "fallbackTurn": "3.14159",
            "reasoning": "Turn number false, confidence true, score 0.95"
        }
        '''
        result = parse_informed_traceback_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["fallbackTurn"], "3.14159")
    
    def test_all_parsers_with_minimal_valid_input(self):
        """Test all parsers with minimal valid input to ensure they work."""
        test_cases = [
            (parse_initial_planning_response, {
                "suggestedTactics": ["T"], "detailedPlan": "P", "reasoning": "R", "nextPrompt": "N"
            }),
            (parse_followup_planning_response, {
                "thought": "T", "plan": {"suggestedTactics": ["T"], "reasoning": "R", "nextPrompt": "N"}
            }),
            (parse_traceback_planning_response, {
                "suggestedTactics": ["T"], "detailedPlan": "P", "reasoning": "R", "nextPrompt": "N"
            }),
            (parse_curate_tactic_response, {
                "newTacticPool": [], "selectionFramework": [], "promptNotes": []
            }),
            (parse_informed_traceback_response, {
                "fallbackTurn": "1", "reasoning": "R"
            }),
            (parse_belief_state_update_response, {
                "conversationContext": {"conversationStage": "S", "progressScore": 0.1},
                "strategyState": {"currentTactic": ["T"], "tacticsTried": ["T"]},
                "lastResponseAnalysis": {"responseType": "R", "keyPoints": ["K"], "policyTriggers": ["P"]}
            }),
        ]
        
        for parser_func, data in test_cases:
            response = str(data).replace("'", '"')
            result = parser_func(response)
            self.assertTrue(result.success, f"Parser {parser_func.__name__} failed with minimal input")
    
    def test_all_parsers_with_null_json(self):
        """Test all parsers with null JSON value."""
        response = 'null'
        
        parsers = [
            parse_initial_planning_response,
            parse_followup_planning_response, 
            parse_traceback_planning_response,
            parse_curate_tactic_response,
            parse_informed_traceback_response,
            parse_belief_state_update_response,
        ]
        
        for parser_func in parsers:
            result = parser_func(response)
            self.assertFalse(result.success, f"Parser {parser_func.__name__} should fail with null JSON")
    
    def test_mixed_case_field_names_fail(self):
        """Test that field names are case sensitive (should fail)."""
        response = '''
        {
            "SUGGESTEDTACTICS": ["Test"],
            "DETAILEDPLAN": "Test plan", 
            "REASONING": "Test reasoning",
            "NEXTPROMPT": "Test prompt"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Missing required fields", result.error)

    # ADDITIONAL STRESS TESTS
    def test_extremely_long_json_response(self):
        """Test with extremely long JSON response."""
        huge_list = ["tactic_" + str(i) for i in range(10000)]
        response = f'''
        {{
            "suggestedTactics": {str(huge_list)},
            "detailedPlan": "Plan",
            "reasoning": "Reasoning",
            "nextPrompt": "Prompt"
        }}
        '''
        result = parse_initial_planning_response(response)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["suggestedTactics"]), 10000)
    
    def test_json_with_null_values_in_required_fields_fails(self):
        """Test that null values in required fields cause failure."""
        response = '''
        {
            "suggestedTactics": null,
            "detailedPlan": "Valid plan",
            "reasoning": "Valid reasoning",
            "nextPrompt": "Valid prompt"
        }
        '''
        result = parse_initial_planning_response(response)
        
        self.assertFalse(result.success)
        self.assertIn("Type validation errors", result.error)
    
    def test_json_with_mixed_valid_invalid_structure(self):
        """Test JSON with mixed valid/invalid nested structures."""
        response = '''
        {
            "thought": "Valid thought",
            "plan": {
                "suggestedTactics": ["Valid"],
                "reasoning": "Valid",
                "nextPrompt": "Valid",
                "invalidNestedStructure": {
                    "deeply": {
                        "nested": {
                            "but": {
                                "valid": "structure"
                            }
                        }
                    }
                }
            }
        }
        '''
        result = parse_followup_planning_response(response)
        
        self.assertTrue(result.success)  # Should pass basic validation
        self.assertIn("invalidNestedStructure", result.data["plan"])


if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)
