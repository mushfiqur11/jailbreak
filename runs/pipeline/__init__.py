"""
Pipeline module for multi-turn red-teaming.

Contains:
- orchestrator: Main pipeline orchestrator
- orchestrator_utils: Utility functions and data classes
- traceback_module: Traceback mechanism for conversation rewind
- tactic_generalization: Batch tactic generalization
- basic: Basic pipeline (original)
- GALA: GALA-specific pipeline (original)
"""

from .orchestrator import PipelineOrchestrator, main
from .orchestrator_utils import TrialResult, GoalResult, ResultStorage
from .traceback_module import TracebackModule, TracebackResult, create_traceback_module
from .tactic_generalization import (
    TacticGeneralizationModule,
    GeneralizedTactic,
    GeneralizationResult,
    create_tactic_generalization_module,
    save_generalization_result
)

__all__ = [
    # Orchestrator
    'PipelineOrchestrator',
    'main',
    
    # Data classes
    'TrialResult',
    'GoalResult',
    'ResultStorage',
    
    # Traceback
    'TracebackModule',
    'TracebackResult',
    'create_traceback_module',
    
    # Tactic Generalization
    'TacticGeneralizationModule',
    'GeneralizedTactic',
    'GeneralizationResult',
    'create_tactic_generalization_module',
    'save_generalization_result'
]
