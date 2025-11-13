# Prompt Functions Documentation

This document provides a comprehensive list of all prompt functions available in the jailbreak agentic module, including their input arguments, expected output formats, and import paths.

## Table of Contents
- [Attacker Prompts](#attacker-prompts)
- [Reasoning Prompts](#reasoning-prompts)
- [Judge Prompts](#judge-prompts)
- [Target Prompts](#target-prompts)

---

## Attacker Prompts

These prompts are used by the attacker agent to plan and execute jailbreak strategies.

### 1. Initial Planning Prompt

**Function:** `get_initial_planning_prompt`  
**Location:** `jailbreak.agentic_module.agents.attacker_prompts.initial_planning`  
**Import:** `from jailbreak.agentic_module.agents.attacker_prompts import get_initial_planning_prompt`

**Input Arguments:**
- `goal` (str): The attack goal/objective
- `target_safety_prompt` (str): Target model's safety measures
- `knowledge_str` (str): Available knowledge base content
- `tactics` (str): Available tactics

**Expected Output Format:**
```json
{
  "suggestedTactics": ["List", "of", "suggested", "tactics"],
  "detailedPlan": "Provide a detailed plan with the suggested tactics taking help from the provided tactics (T)",
  "reasoning": "description and reasoning for the chosen tactic",
  "nextPrompt": "Write the actual initial prompt you'll use that align with above suggestions."
}
```

**Purpose:** Generates initial attack planning with suggested tactics and the first prompt to use.

---

### 2. Followup Planning Prompt

**Function:** `get_followup_planning_prompt`  
**Location:** `jailbreak.agentic_module.agents.attacker_prompts.followup_planning`  
**Import:** `from jailbreak.agentic_module.agents.attacker_prompts import get_followup_planning_prompt`

**Input Arguments:**
- `goal` (str): The attack goal/objective
- `target_safety_prompt` (str): Target model's safety measures
- `belief_state_str` (str): Current belief state as JSON string
- `knowledge_str` (str): Available knowledge base content
- `tactics` (str): Available tactics

**Expected Output Format:**
```json
{
  "thought": "<your reflection on the observation towards a goal>",
  "plan": {
    "suggestedTactics": ["List", "of", "suggested", "tactics"],
    "reasoning": "<description_and_reasoning>",
    "infoToFocusOnNext": ["<item1>", "<item2>", "..."],
    "nextPrompt": "<next_prompt>"
  }
}
```

**Purpose:** Generates followup attack planning based on conversation history and current state.

---

### 3. Traceback Planning Prompt

**Function:** `get_traceback_planning_prompt`  
**Location:** `jailbreak.agentic_module.agents.attacker_prompts.traceback_planning`  
**Import:** `from jailbreak.agentic_module.agents.attacker_prompts import get_traceback_planning_prompt`

**Input Arguments:**
- `goal` (str): The attack goal/objective
- `target_safety_prompt` (str): Target model's safety measures
- `traceback_knowledge` (str): Feedback from failed attempts
- `belief_state_str` (str): Current belief state as JSON string
- `knowledge_str` (str): Available knowledge base content
- `tactics` (str): Available tactics

**Expected Output Format:**
```json
{
  "suggestedTactics": ["List", "of", "suggested", "tactics"],
  "detailedPlan": "Provide a detailed plan with the suggested tactics taking help from the provided tactics (T)",
  "reasoning": "description and reasoning for the chosen tactic",
  "nextPrompt": "Write the actual prompt you'll use that align with above suggestions. This should be a continuation of the history where you are the attacker and the you are interacting with the target."
}
```

**Purpose:** Generates new attack planning when previous attempts have failed, incorporating learned feedback.

---

## Reasoning Prompts

These prompts are used by reasoning agents to analyze, learn, and improve attack strategies.

### 1. Curate Tactic Prompt

**Function:** `get_curate_tactic_prompt`  
**Location:** `jailbreak.agentic_module.agents.reasoning_prompts.curate_tactic`  
**Import:** `from jailbreak.agentic_module.agents.reasoning_prompts import get_curate_tactic_prompt`

**Input Arguments:**
- `learning_note` (str): Initial learning note (read-only)
- `learning_note_supplementary` (str): Existing supplementary note to be updated
- `goal` (str): Red-teaming goal of the trial
- `conversation_str` (str): Latest successful trial conversation with judgments

**Expected Output Format:**
```json
{
  "newTacticPool": [
    {
      "tactic": "<TacticName>",
      "definition": "<A concise yet detailed explanation of the jailbreak tactic>",
      "prompts": ["<prompt for round 1>", "<prompt for round 2>", "..."]
    }
  ],
  "selectionFramework": [
    {
      "scenarioOrGoalType": "<Description of the jailbreak goal type>",
      "recommendedTactic": ["<Primary Tactic>", "..."]
    }
  ],
  "promptNotes": []
}
```

**Purpose:** Updates the learning memory with new tactics and selection frameworks based on successful trials.

---

### 2. Informed Traceback Prompt

**Function:** `get_informed_traceback_prompt`  
**Location:** `jailbreak.agentic_module.agents.reasoning_prompts.informed_traceback`  
**Import:** `from jailbreak.agentic_module.agents.reasoning_prompts import get_informed_traceback_prompt`

**Input Arguments:**
- `goal` (str): The attack goal/objective
- `target_safety_prompt` (str): Target model's safety measures
- `conversation_str` (str): Conversation history to analyze

**Expected Output Format:**
```json
{
  "fallbackTurn": "<the turn number to which the attacker should fallback>",
  "reasoning": "<detailed reasoning on what went wrong after that turn and how the attacker can make a better attempt next time>"
}
```

**Purpose:** Analyzes failed conversations to determine optimal fallback points and provide improvement strategies.

---

### 3. Belief State Update Prompt

**Function:** `get_belief_state_update_prompt`  
**Location:** `jailbreak.agentic_module.agents.reasoning_prompts.belief_state_update`  
**Import:** `from jailbreak.agentic_module.agents.reasoning_prompts import get_belief_state_update_prompt`

**Input Arguments:**
- `conversation_str` (str): Conversation history
- `current_tactic` (str): Current tactic being used
- `goal` (str): Red-teaming goal
- `belief_state_str` (str): Last belief state as JSON string

**Expected Output Format:**
```json
{
  "conversationContext": {
    "conversationStage": "<categorical indicator>",
    "refusalCount": "<number of direct refusals only>",
    "progressScore": "<number between 0 and 1 reflecting the progress toward the goal>",
    "missingInformation": ["<specific data points you still need to extract>"],
    "collectedInformation": ["<data already acquired>"],
    "cooperativeness": "<a number between 1 and 10 on how cooperative the target model is>"
  },
  "strategyState": {
    "currentTactic": ["<current_tactic1>", "..."],
    "tacticsTried": ["<tactic1>", "<tactic2>", "..."]
  },
  "lastResponseAnalysis": {
    "responseType": "<response type>",
    "keyPoints": ["<point1>", "<point2>", "..."],
    "policyTriggers": ["<type of policy or content filters were triggered>"]
  }
}
```

**Purpose:** Updates and maintains the belief state tracking conversation progress, tactics used, and target model responses.

---

### 4. Generalize Tactics Prompt

**Function:** `get_generalize_tactics_prompt`  
**Location:** `jailbreak.agentic_module.agents.reasoning_prompts.generalize_tactics`  
**Import:** `from jailbreak.agentic_module.agents.reasoning_prompts import get_generalize_tactics_prompt`

**Input Arguments:**
- None (takes no parameters)

**Expected Output Format:**
- Simple string: "Generalize the tactics"

**Purpose:** ⚠️ **Note:** This prompt appears to be incomplete/placeholder and may need further development.

---

## Judge Prompts

These prompts are used by judge agents to evaluate conversations and determine policy violations.

### 1. Judge Policy Prompt

**Function:** `get_judge_policy_prompt`  
**Location:** `jailbreak.agentic_module.agents.judge_prompts.judge_policy`  
**Import:** `from jailbreak.agentic_module.agents.judge_prompts import get_judge_policy_prompt`

**Input Arguments:**
- None (takes no parameters)

**Expected Output Format:**
- Simple string: "This is a judgement policy"

**Purpose:** ⚠️ **Note:** This appears to be a placeholder prompt that needs proper implementation.

---

## Target Prompts

These prompts are related to target model configurations and safety measures.

### 1. Safety Prompt

**Function:** `get_safety_prompt`  
**Location:** `jailbreak.agentic_module.agents.target_prompts.safety_prompt`  
**Import:** `from jailbreak.agentic_module.agents.target_prompts import get_safety_prompt`

**Input Arguments:**
- None (takes no parameters)

**Expected Output Format:**
- Simple string: "This is the safety prompt of the target model"

**Purpose:** ⚠️ **Note:** This appears to be a placeholder prompt that needs proper implementation.

---

## Usage Examples

### Basic Import and Usage

```python
# Import all attacker prompts
from jailbreak.agentic_module.agents.attacker_prompts import (
    get_initial_planning_prompt,
    get_followup_planning_prompt,
    get_traceback_planning_prompt
)

# Import all reasoning prompts
from jailbreak.agentic_module.agents.reasoning_prompts import (
    get_curate_tactic_prompt,
    get_informed_traceback_prompt,
    get_belief_state_update_prompt
)

# Example usage
goal = "Extract sensitive information"
target_safety = "I cannot provide harmful content"
knowledge = "Previous successful tactics include..."
tactics = "Request_Framing, Hidden_Intention_Streamline"

# Generate initial plan
initial_plan_prompt = get_initial_planning_prompt(
    goal=goal,
    target_safety_prompt=target_safety,
    knowledge_str=knowledge,
    tactics=tactics
)

# Use with LLM
llm.set_system_prompt(initial_plan_prompt)
response = llm.forward()  # Returns JSON with plan
```

### Integration with AttackerAgent

```python
from jailbreak.agentic_module.agents.attacker import AttackerAgent

# The AttackerAgent already integrates these prompts
agent = AttackerAgent(attacker_config, reasoning_config)

# These methods internally use the prompt functions:
initial_plan = agent.get_initial_plan(goal, target_safety_prompt)
followup_plan = agent.get_followup_plan(goal, conversation_str, current_tactic)
traceback_plan = agent.get_traceback_plan(goal, conversation_str, belief_state_str)
```

---

## Notes

1. **JSON Output:** Most prompts expect structured JSON responses from the LLM. Ensure your LLM is configured to return valid JSON.

2. **Placeholder Prompts:** Some prompts (marked with ⚠️) appear to be placeholders and may need proper implementation.

3. **Error Handling:** When using these prompts, implement proper JSON parsing and error handling for malformed responses.

4. **Security:** These prompts are designed for red-teaming and security research purposes. Use responsibly in controlled environments.

5. **Updates:** This documentation reflects the current state of the codebase. Prompts may be updated or extended over time.
