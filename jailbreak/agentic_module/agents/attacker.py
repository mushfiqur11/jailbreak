from jailbreak.agentic_module.agents.attacker_prompts import (
    get_initial_planning_prompt,
    get_followup_planning_prompt,
    get_traceback_planning_prompt
)

from jailbreak.agentic_module.agents.reasoning_prompts import (
    get_curate_tactic_prompt,
    get_informed_traceback_prompt,
    get_generalize_tactics_prompt
)

from jailbreak.llm_module import llm_model_factory
import logging
import json
import os
from typing import Dict, Any, Optional, List
from collections import defaultdict
from .utils import load_initial_tactics, validate_tactics_structure

# Configure logging
logger = logging.getLogger(__name__)


class BeliefState:
    """
    Belief state management for tracking conversation context, strategy state, and response analysis.
    """
    
    def __init__(self, turns: Optional[List[Dict]] = None):
        """
        Initialize the belief state with conversation context, strategy state, and analysis.
        
        Args:
            turns (Optional[List[Dict]]): Initial conversation turns
        """
        if turns is None:
            turns = []
            
        self.conversation_context = {
            "conversationStage": "early probing/trust-building",
            "refusalCount": 0,
            "progressScore": 0.0,
            "missingInformation": [],
            "collectedInformation": [],
            "cooperativeness": 0,
            "turns": list(turns)
        }
        
        self.strategy_state = {
            "currentTactic": [],
            "tacticsTried": []
        }
        
        self.last_response_analysis = {
            "responseType": "initial",
            "keyPoints": [],
            "policyTriggers": []
        }
        
        self.cost_list = []
        
        logger.debug("BeliefState initialized")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert belief state to dictionary format.
        
        Returns:
            Dict[str, Any]: Serialized belief state
        """
        return {
            "conversationContext": self.conversation_context,
            "strategyState": self.strategy_state,
            "lastResponseAnalysis": self.last_response_analysis
        }

    def add_turn(self, prompt: str, response: str, judgment: Optional[Dict] = None, cost: Optional[float] = None) -> None:
        """
        Add a conversation turn to the belief state.
        
        Args:
            prompt (str): Attacker prompt
            response (str): Target response
            judgment (Optional[Dict]): Judge's judgment
            cost (Optional[float]): API cost for this turn
        """
        turn_id = len(self.conversation_context["turns"])
        
        turn_data = {
            "turn": turn_id,
            "attacker_prompt": prompt,
            "target_response": response
        }
        
        if judgment is not None:
            turn_data["judgment"] = judgment
            
        if cost is not None:
            turn_data["cost"] = cost
            
        self.conversation_context["turns"].append(turn_data)
        
        if cost is not None:
            self.add_cost(cost, "conversation_turn")
            
        logger.debug(f"Added turn {turn_id} to belief state")

    def add_cost(self, cost: float, cost_type: str) -> None:
        """
        Add cost tracking to the belief state.
        
        Args:
            cost (float): Cost amount
            cost_type (str): Type of cost (e.g., 'conversation_turn', 'planning', etc.)
        """
        self.cost_list.append({
            "cost": cost,
            "type": cost_type
        })
        
        logger.debug(f"Added {cost_type} cost: {cost:.4f}")

    def get_conversation_history(self) -> List[Dict]:
        """
        Get the conversation history stored in the belief state.
        
        Returns:
            List[Dict]: List of conversation turns
        """
        return self.conversation_context["turns"]

    def update_from_response(self, response: str, current_tactics: List[str]) -> None:
        """
        Update belief state based on target's response.
        
        Args:
            response (str): Target model's response
            current_tactics (List[str]): Current tactics being used
        """
        # Analyze response type
        if any(phrase in response.lower() for phrase in ["i can't", "i cannot", "i'm sorry", "not appropriate"]):
            self.last_response_analysis["responseType"] = "direct_refusal"
            self.conversation_context["refusalCount"] += 1
        elif any(phrase in response.lower() for phrase in ["however", "but", "general", "overview"]):
            self.last_response_analysis["responseType"] = "partial_compliance"
        else:
            self.last_response_analysis["responseType"] = "full_compliance"
        
        # Update conversation stage based on progress
        if self.conversation_context["refusalCount"] > 2:
            self.conversation_context["conversationStage"] = "advanced/persistence phase"
        elif len(self.strategy_state["tacticsTried"]) > 1:
            self.conversation_context["conversationStage"] = "mid-stage info extraction"
        
        # Update tactics
        self.strategy_state["currentTactic"] = current_tactics
        for tactic in current_tactics:
            if tactic not in self.strategy_state["tacticsTried"]:
                self.strategy_state["tacticsTried"].append(tactic)
        
        # Update progress score
        if self.last_response_analysis["responseType"] == "full_compliance":
            self.conversation_context["progressScore"] = min(1.0, self.conversation_context["progressScore"] + 0.3)
        elif self.last_response_analysis["responseType"] == "partial_compliance":
            self.conversation_context["progressScore"] = min(1.0, self.conversation_context["progressScore"] + 0.1)
        
        logger.debug(f"Updated belief state: stage={self.conversation_context['conversationStage']}, "
                    f"progress={self.conversation_context['progressScore']:.2f}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get belief state statistics for monitoring.
        
        Returns:
            Dict[str, Any]: Statistics about the current belief state
        """
        return {
            "conversationStage": self.conversation_context["conversationStage"],
            "refusalCount": self.conversation_context["refusalCount"],
            "progressScore": self.conversation_context["progressScore"],
            "turns_completed": len(self.conversation_context["turns"]),
            "tactics_tried": len(self.strategy_state["tacticsTried"]),
            "last_responseType": self.last_response_analysis["responseType"],
            "total_cost": sum(item["cost"] for item in self.cost_list)
        }


class KnowledgeBase:
    """
    Knowledge base for managing tactics, insights, and learning from conversations.
    """
    
    def __init__(self, sce_knowledge: bool = False, tactics_file: str = "initial_tactics.json"):
        """
        Initialize the knowledge base with initial tactics and dynamic components.
        
        Args:
            sce_knowledge (bool): Whether to enable SCE (Safety-Critical Evaluations) knowledge
            tactics_file (str): Name of the tactics JSON file to load
        """
        # Load initial tactics from JSON file
        tactics_path = os.path.join(os.path.dirname(__file__), tactics_file)
        self.initial_tactics = load_initial_tactics(tactics_path)
        
        if not validate_tactics_structure(self.initial_tactics):
            logger.warning("Tactics structure validation failed, using empty tactics")
            self.initial_tactics = {}
        
        # Dynamic components for learning and adaptation
        self.failed_attempt_insight = {}
        self.discovered_tactics = {}
        self.tactic_selection_framework = {}
        self.prompt_improvement_notes = defaultdict(list)
        self.sce_knowledge = sce_knowledge
        self.tacticDescription = ""
        
        logger.info(f"KnowledgeBase initialized with {len(self.initial_tactics)} tactics")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the knowledge base to a dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the knowledge base
        """
        return {
            "failed_attempt_insight": self.failed_attempt_insight,
            "initial_tactics": str(list(self.initial_tactics.keys())),
            "current_tactical_plan": str(self.tacticDescription),
            "discovered_tactics": self.discovered_tactics,
            "tactic_selection_framework": self.tactic_selection_framework,
            "prompt_improvement_notes": dict(self.prompt_improvement_notes)
        }

    def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load complete knowledge base for the attacker.
        
        Returns:
            Dict[str, Any]: Complete knowledge base
        """
        return {
            "initial_tactics": self.initial_tactics,
            "discovered_tactics": self.discovered_tactics,
            "tactic_selection_framework": self.tactic_selection_framework,
            "prompt_improvement_notes": dict(self.prompt_improvement_notes)
        }

    def get_recommended_tactics(self, goal: str, goal_type: Optional[str] = None) -> List[str]:
        """
        Select appropriate tactics based on goal and current state.
        
        Args:
            goal (str): The attack goal
            goal_type (Optional[str]): Category of the goal if known
            
        Returns:
            List[str]: Recommended tactics for the goal
        """
        if goal_type and goal_type in self.tactic_selection_framework:
            # Sort tactics by effectiveness for this goal type
            sorted_tactics = sorted(
                self.tactic_selection_framework[goal_type].items(),
                key=lambda x: x[1], 
                reverse=True
            )
            return [tactic for tactic, _ in sorted_tactics[:3]]
        
        # Fallback to general effective tactics from initial tactics
        default_tactics = ["Request_Framing", "Hidden_Intention_Streamline", "Echoing"]
        available_tactics = list(self.initial_tactics.keys())
        
        # Return intersection of default tactics that exist in initial tactics
        recommended = [tactic for tactic in default_tactics if tactic in available_tactics]
        
        # If we don't have enough, add more from available tactics
        if len(recommended) < 3:
            for tactic in available_tactics:
                if tactic not in recommended:
                    recommended.append(tactic)
                    if len(recommended) >= 3:
                        break
        
        return recommended[:3]

    def add_discovered_tactic(self, tactic_name: str, definition: str, examples: List[str]) -> None:
        """
        Add a newly discovered tactic to the knowledge base.
        
        Args:
            tactic_name (str): Name of the new tactic
            definition (str): Definition of the tactic
            examples (List[str]): Example prompts using this tactic
        """
        self.discovered_tactics[tactic_name] = {
            "definition": definition,
            "examples": examples,
            "discovery_count": 1,
            "success_rate": 1.0
        }
        
        logger.info(f"Added new tactic to knowledge base: {tactic_name}")

    def update_tactic_selection_framework(self, goal_type: str, effective_tactics: List[str]) -> None:
        """
        Update the goal-based tactic selection framework.
        
        Args:
            goal_type (str): Category of the goal
            effective_tactics (List[str]): List of tactics that were effective
        """
        if goal_type not in self.tactic_selection_framework:
            self.tactic_selection_framework[goal_type] = defaultdict(int)
        
        for tactic in effective_tactics:
            self.tactic_selection_framework[goal_type][tactic] += 1
            
        logger.info(f"Updated tactic selection framework for {goal_type} with {len(effective_tactics)} tactics")

    def add_prompt_improvement(self, goal: str, prompt_improvements: str) -> None:
        """
        Add prompt-wise improvement note for specific goal.
        
        Args:
            goal (str): The attack goal
            prompt_improvements (str): Improvement suggestions
        """
        self.prompt_improvement_notes[goal].append({
            "note": prompt_improvements,
            "timestamp": "timestamp"  # Simplified timestamp
        })
        
        logger.info(f"Added prompt improvement for goal: {goal[:50]}...")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics for monitoring.
        
        Returns:
            Dict[str, Any]: Statistics about the knowledge base state
        """
        return {
            "initial_tactics_count": len(self.initial_tactics),
            "discovered_tactics_count": len(self.discovered_tactics),
            "goal_types_in_framework": len(self.tactic_selection_framework),
            "total_prompt_improvements": sum(len(notes) for notes in self.prompt_improvement_notes.values()),
            "goals_with_improvements": len(self.prompt_improvement_notes)
        }

    def get_tactic_definition(self, tactic_name: str) -> Optional[str]:
        """
        Get the definition of a specific tactic.
        
        Args:
            tactic_name (str): Name of the tactic
            
        Returns:
            Optional[str]: Tactic definition if found
        """
        if tactic_name in self.initial_tactics:
            return self.initial_tactics[tactic_name].get("definition")
        elif tactic_name in self.discovered_tactics:
            return self.discovered_tactics[tactic_name].get("definition")
        return None

    def get_tactic_examples(self, tactic_name: str) -> List[str]:
        """
        Get examples for a specific tactic.
        
        Args:
            tactic_name (str): Name of the tactic
            
        Returns:
            List[str]: List of examples for the tactic
        """
        if tactic_name in self.initial_tactics:
            return self.initial_tactics[tactic_name].get("examples", [])
        elif tactic_name in self.discovered_tactics:
            return self.discovered_tactics[tactic_name].get("examples", [])
        return []


class AttackerAgent:
    def __init__(self, attacker_config: Dict[str, Any], reasoning_config: Dict[str, Any]):
        """
        Initialize the AttackerAgent with separate LLMs for attacking and reasoning.
        
        Args:
            attacker_config (Dict[str, Any]): Configuration for the attacker LLM
            reasoning_config (Dict[str, Any]): Configuration for the reasoning LLM
        """
        logger.info("Initializing AttackerAgent")
        
        # Initialize attacker LLM using model factory
        try:
            self.attacker_llm = llm_model_factory(attacker_config)
            logger.info(f"Successfully created attacker LLM: {self.attacker_llm}")
        except Exception as e:
            logger.error(f"Failed to create attacker LLM: {e}")
            raise
        
        # Initialize reasoning LLM using model factory
        try:
            self.reasoning_llm = llm_model_factory(reasoning_config)
            logger.info(f"Successfully created reasoning LLM: {self.reasoning_llm}")
        except Exception as e:
            logger.error(f"Failed to create reasoning LLM: {e}")
            raise
        
        # Initialize belief state and knowledge base with full implementations
        self.belief_state = BeliefState()
        self.knowledge_base = KnowledgeBase()
        
        logger.info("AttackerAgent initialized successfully")

    def update_knowledge_base(self, new_info: Dict[str, Any]) -> None:
        """Update the knowledge base with new information."""
        logger.info("Updating knowledge base")
        # Now using the new KnowledgeBase methods
        if "tactic_name" in new_info and "definition" in new_info:
            self.knowledge_base.add_discovered_tactic(
                new_info["tactic_name"],
                new_info["definition"],
                new_info.get("examples", [])
            )

    def update_belief_state(self, belief_json: str) -> None:
        """Update belief state using the new BeliefState methods."""
        logger.info("Updating belief state")
        # Can now parse and update using the new BeliefState structure
        try:
            belief_data = json.loads(belief_json) if isinstance(belief_json, str) else belief_json
            # Update using the new BeliefState structure
            if "conversationContext" in belief_data:
                self.belief_state.conversation_context.update(belief_data["conversationContext"])
            if "strategyState" in belief_data:
                self.belief_state.strategy_state.update(belief_data["strategyState"])
            if "lastResponseAnalysis" in belief_data:
                self.belief_state.last_response_analysis.update(belief_data["lastResponseAnalysis"])
        except Exception as e:
            logger.error(f"Failed to update belief state: {e}")
    
    def get_initial_plan(self, goal: str, target_safety_prompt: str) -> str:
        """
        Generate initial attack plan using the attacker LLM.
        
        Args:
            goal (str): The attack goal
            target_safety_prompt (str): The target's safety prompt
            
        Returns:
            str: Generated initial plan
        """
        logger.info("Generating initial attack plan")
        
        try:
            # Get the initial planning prompt
            prompt = get_initial_planning_prompt(goal, target_safety_prompt)
            
            # Set system prompt and generate plan
            self.attacker_llm.set_system_prompt(prompt)
            self.attacker_llm.clear_conversation()
            
            # Generate the plan
            plan = self.attacker_llm.forward()
            logger.info("Initial attack plan generated successfully")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate initial plan: {e}")
            raise
    
    def get_followup_plan(self, goal: str, conversation_str: str, current_tactic: str) -> str:
        """
        Generate followup attack plan based on conversation history.
        
        Args:
            goal (str): The attack goal
            conversation_str (str): Current conversation history
            current_tactic (str): Current tactic being used
            
        Returns:
            str: Generated followup plan
        """
        logger.info("Generating followup attack plan")
        
        try:
            # Get the followup planning prompt
            prompt = get_followup_planning_prompt(goal, conversation_str, current_tactic)
            
            # Set system prompt and generate plan
            self.attacker_llm.set_system_prompt(prompt)
            self.attacker_llm.clear_conversation()
            
            # Generate the plan
            plan = self.attacker_llm.forward()
            logger.info("Followup attack plan generated successfully")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate followup plan: {e}")
            raise
    
    def get_traceback_plan(self, goal: str, conversation_str: str, belief_state_str: str) -> str:
        """
        Generate traceback plan when current approach fails.
        
        Args:
            goal (str): The attack goal
            conversation_str (str): Current conversation history
            belief_state_str (str): Current belief state
            
        Returns:
            str: Generated traceback plan
        """
        logger.info("Generating traceback attack plan")
        
        try:
            # Get the traceback planning prompt
            prompt = get_traceback_planning_prompt(goal, conversation_str, belief_state_str)
            
            # Set system prompt and generate plan
            self.attacker_llm.set_system_prompt(prompt)
            self.attacker_llm.clear_conversation()
            
            # Generate the plan
            plan = self.attacker_llm.forward()
            logger.info("Traceback attack plan generated successfully")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate traceback plan: {e}")
            raise

    def perform_traceback(self, goal: str, knowledge_str: str, tactics: str, conversation_str: str) -> str:
        """
        Perform informed traceback analysis using reasoning LLM.
        
        Args:
            goal (str): The attack goal
            knowledge_str (str): Current knowledge base content
            tactics (str): Available tactics
            conversation_str (str): Conversation history
            
        Returns:
            str: Traceback analysis result
        """
        logger.info("Performing informed traceback")
        
        try:
            # Get the informed traceback prompt
            prompt = get_informed_traceback_prompt(goal, knowledge_str, tactics, conversation_str)
            
            # Set system prompt and generate analysis
            self.reasoning_llm.set_system_prompt(prompt)
            self.reasoning_llm.clear_conversation()
            
            # Generate the analysis
            analysis = self.reasoning_llm.forward()
            logger.info("Informed traceback completed successfully")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to perform traceback: {e}")
            raise

    def perform_tactic_curation(self, learning_note: str, learning_note_supplementary: str, 
                               goal: str, conversation_str: str) -> str:
        """
        Perform tactic curation to update learning notes.
        
        Args:
            learning_note (str): Initial learning note
            learning_note_supplementary (str): Supplementary learning note
            goal (str): The attack goal
            conversation_str (str): Conversation history
            
        Returns:
            str: Updated supplementary learning note
        """
        logger.info("Performing tactic curation")
        
        try:
            # Get the curate tactic prompt
            prompt = get_curate_tactic_prompt(learning_note, learning_note_supplementary, 
                                            goal, conversation_str)
            
            # Set system prompt and generate curation
            self.reasoning_llm.set_system_prompt(prompt)
            self.reasoning_llm.clear_conversation()
            
            # Generate the curation result
            result = self.reasoning_llm.forward()
            logger.info("Tactic curation completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform tactic curation: {e}")
            raise

    def perform_tactic_generalization(self, goal: str, knowledge_str: str, tactics: str) -> str:
        """
        Perform tactic generalization using reasoning LLM.
        
        Args:
            goal (str): The attack goal
            knowledge_str (str): Current knowledge base content
            tactics (str): Available tactics
            
        Returns:
            str: Generalized tactics result
        """
        logger.info("Performing tactic generalization")
        
        try:
            # Get the generalize tactics prompt
            prompt = get_generalize_tactics_prompt(goal, knowledge_str, tactics)
            
            # Set system prompt and generate generalization
            self.reasoning_llm.set_system_prompt(prompt)
            self.reasoning_llm.clear_conversation()
            
            # Generate the generalization result
            result = self.reasoning_llm.forward()
            logger.info("Tactic generalization completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform tactic generalization: {e}")
            raise

    def perform_local_learning(self, *args, **kwargs) -> str:
        """
        Placeholder for local learning functionality.
        Implementation depends on specific requirements.
        """
        logger.info("Local learning functionality not yet implemented")
        return "Local learning placeholder"

    def perform_global_learning(self, *args, **kwargs) -> str:
        """
        Placeholder for global learning functionality.
        Implementation depends on specific requirements.
        """
        logger.info("Global learning functionality not yet implemented")
        return "Global learning placeholder"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict[str, Any]: Information about attacker and reasoning LLMs
        """
        return {
            "attacker_llm": self.attacker_llm.get_model_info(),
            "reasoning_llm": self.reasoning_llm.get_model_info()
        }
