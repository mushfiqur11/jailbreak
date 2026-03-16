"""
Tactic Generalization Module for multi-turn red-teaming pipeline.

Processes trial trajectories (successes and failures) in batches at the end of runs
to extract generalized multi-turn tactics with severity-aware context.
"""

import logging
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GeneralizedTactic:
    """A generalized tactic extracted from successful jailbreaks."""
    tactic: str
    definition: str
    multi_turn_structure: List[str]
    key_patterns: List[str]
    effective_for_goals: List[str]
    source_batch: int = 0


@dataclass
class GeneralizationResult:
    """Result of tactic generalization across all batches."""
    batches_processed: int
    total_records: int
    total_successes: int
    generalized_tactics: List[GeneralizedTactic]
    analysis_summaries: List[str] = field(default_factory=list)


class TacticGeneralizationModule:
    """
    Module for batch-wise tactic generalization from successful jailbreaks.
    
    Flow:
    1. Collect all trial histories (successes + failures)
    2. Process in batches of size N
    3. For each batch, extract generalized tactics via reasoning
    4. Combine all batch results
    """
    
    def __init__(self, reasoning_agent, batch_size: int = 20):
        """
        Initialize the TacticGeneralizationModule.
        
        Args:
            reasoning_agent: ReasoningAgent instance for generalization
            batch_size: Number of successes to process per batch
        """
        self.reasoning = reasoning_agent
        self.batch_size = batch_size
        
        logger.info(f"TacticGeneralizationModule initialized with batch_size={batch_size}")
    
    def generalize_all(
        self, 
        trial_records: List[Dict[str, Any]]
    ) -> GeneralizationResult:
        """
        Process all trial records in batches and combine results.
        
        Args:
            trial_records: List of trial records (successes and failures)
                Each record should have: goal, turns, tactics_used, success, violation_result
                
        Returns:
            GeneralizationResult with all generalized tactics
        """
        total_records = len(trial_records)
        logger.info(f"Generalizing tactics from {total_records} trial records")
        
        if total_records == 0:
            logger.warning("No trial records to generalize")
            return GeneralizationResult(
                batches_processed=0,
                total_records=0,
                total_successes=0,
                generalized_tactics=[],
                analysis_summaries=[]
            )

        total_successes = sum(1 for r in trial_records if r.get("success", False))
        
        all_tactics = []
        all_summaries = []
        batch_num = 0
        
        # Process in batches
        for i in range(0, total_records, self.batch_size):
            batch = trial_records[i:i + self.batch_size]
            batch_num += 1
            
            logger.info(f"Processing batch {batch_num}: {len(batch)} records")
            
            # Generalize tactics for this batch
            batch_result = self._process_batch(batch, batch_num)
            
            all_tactics.extend(batch_result["tactics"])
            all_summaries.append(batch_result["summary"])
        
        # Combine tactics (deduplication can be added)
        combined_tactics = all_tactics  # For now, keep all
        
        logger.info(f"Generalization complete: {len(combined_tactics)} tactics from {batch_num} batches")
        
        return GeneralizationResult(
            batches_processed=batch_num,
            total_records=total_records,
            total_successes=total_successes,
            generalized_tactics=combined_tactics,
            analysis_summaries=all_summaries
        )
    
    def _process_batch(
        self, 
        batch: List[Dict[str, Any]], 
        batch_num: int
    ) -> Dict[str, Any]:
        """
        Process a single batch of trial records.
        
        Args:
            batch: List of trial records
            batch_num: Batch number for tracking
            
        Returns:
            Dictionary with tactics and summary
        """
        # Format batch for reasoning
        batch_str = self._format_batch_for_generalization(batch)
        
        # Call reasoning to generalize
        result = self.reasoning.generalize_tactics(batch_str)
        
        # Parse result into GeneralizedTactic objects
        tactics = []
        for tactic_data in result.get("generalized_tactics", []):
            tactic = GeneralizedTactic(
                tactic=tactic_data.get("tactic", "Unknown"),
                definition=tactic_data.get("definition", ""),
                multi_turn_structure=tactic_data.get("multi_turn_structure", []),
                key_patterns=tactic_data.get("key_patterns", []),
                effective_for_goals=tactic_data.get("effective_for_goals", []),
                source_batch=batch_num
            )
            tactics.append(tactic)
        
        summary = result.get("analysis_summary", "")
        
        return {"tactics": tactics, "summary": summary}
    
    def _format_batch_for_generalization(self, batch: List[Dict[str, Any]]) -> str:
        """
        Format a batch of trial records for the generalization prompt.
        
        Args:
            batch: List of trial records
            
        Returns:
            Formatted string for the prompt
        """
        formatted_parts = []
        
        for idx, record in enumerate(batch, 1):
            goal = record.get("goal", "Unknown goal")
            turns = record.get("turns", [])
            tactics_used = record.get("tactics_used", [])
            success = bool(record.get("success", False))
            violation_result = record.get("violation_result", 0)
            max_pv = record.get("max_pv", violation_result)
            avg_pv = record.get("avg_pv", float(violation_result))
            
            part = f"=== Trial Record {idx} ===\n"
            part += f"Goal: {goal}\n"
            part += f"Outcome: {'SUCCESS' if success else 'FAILURE'}\n"
            part += f"Final PV: {violation_result}\n"
            part += f"Max PV observed: {max_pv}\n"
            part += f"Avg PV observed: {avg_pv:.3f}\n"
            part += f"Tactics Used: {', '.join(tactics_used) if tactics_used else 'Unknown'}\n"
            part += "Conversation:\n"
            
            for i, turn in enumerate(turns, 1):
                if "attacker_prompt" in turn:
                    part += f"  Turn {i} Attacker: {turn['attacker_prompt'][:500]}...\n"
                if "target_response" in turn:
                    part += f"  Turn {i} Target: {turn['target_response'][:500]}...\n"
                if "judge_evaluation" in turn:
                    pv = turn["judge_evaluation"].get("violation_intensity", 0)
                    part += f"  Turn {i} Judge PV: {pv}\n"
            
            formatted_parts.append(part)
        
        return "\n\n".join(formatted_parts)


def create_tactic_generalization_module(
    reasoning_agent, 
    config: Dict[str, Any]
) -> TacticGeneralizationModule:
    """
    Factory function to create a TacticGeneralizationModule from config.
    
    Args:
        reasoning_agent: ReasoningAgent instance
        config: Configuration dictionary
        
    Returns:
        Configured TacticGeneralizationModule
    """
    batch_size = config.get("features", {}).get("tactic_generalization_batch_size", 20)
    return TacticGeneralizationModule(reasoning_agent, batch_size=batch_size)


def save_generalization_result(result: GeneralizationResult, output_path: str) -> None:
    """
    Save generalization result to JSON file.
    
    Args:
        result: GeneralizationResult to save
        output_path: Path to output file
    """
    data = {
        "batches_processed": result.batches_processed,
        "total_records": result.total_records,
        # Backward-compatible field name retained for downstream consumers.
        "total_successes": result.total_successes,
        "generalized_tactics": [
            {
                "tactic": t.tactic,
                "definition": t.definition,
                "multi_turn_structure": t.multi_turn_structure,
                "key_patterns": t.key_patterns,
                "effective_for_goals": t.effective_for_goals,
                "source_batch": t.source_batch
            }
            for t in result.generalized_tactics
        ],
        "analysis_summaries": result.analysis_summaries
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Generalization result saved to {output_path}")
