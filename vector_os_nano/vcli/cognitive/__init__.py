"""VGG cognitive layer — types, verifier, decomposer, strategy selector, executor, stats, sandbox, and experience.

Public surface::

    from vector_os_nano.vcli.cognitive import (
        SubGoal,
        GoalTree,
        StepRecord,
        ExecutionTrace,
        GoalVerifier,
        GoalDecomposer,
        GoalExecutor,
        StrategySelector,
        StrategyResult,
        StrategyStats,
        StrategyRecord,
        CodeExecutor,
        CodeResult,
        SubGoalTemplate,
        GoalTemplate,
        ExperienceCompiler,
        TemplateLibrary,
    )
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.code_executor import CodeExecutor, CodeResult
from vector_os_nano.vcli.cognitive.experience_compiler import (
    ExperienceCompiler,
    GoalTemplate,
    SubGoalTemplate,
)
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult, StrategySelector
from vector_os_nano.vcli.cognitive.strategy_stats import StrategyRecord, StrategyStats
from vector_os_nano.vcli.cognitive.template_library import TemplateLibrary
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

__all__ = [
    "CodeExecutor",
    "CodeResult",
    "ExperienceCompiler",
    "ExecutionTrace",
    "GoalDecomposer",
    "GoalExecutor",
    "GoalTemplate",
    "GoalTree",
    "GoalVerifier",
    "StepRecord",
    "StrategyRecord",
    "StrategyResult",
    "StrategySelector",
    "StrategyStats",
    "SubGoal",
    "SubGoalTemplate",
    "TemplateLibrary",
]
