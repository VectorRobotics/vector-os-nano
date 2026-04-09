"""VGG cognitive layer — types, verifier, and decomposer.

Public surface::

    from vector_os_nano.vcli.cognitive import (
        SubGoal,
        GoalTree,
        StepRecord,
        ExecutionTrace,
        GoalVerifier,
        GoalDecomposer,
    )
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

__all__ = [
    "ExecutionTrace",
    "GoalDecomposer",
    "GoalTree",
    "GoalVerifier",
    "StepRecord",
    "SubGoal",
]
