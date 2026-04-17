"""VGG cognitive layer — frozen dataclasses.

All types are immutable (frozen=True) to ensure safe sharing across
async executor threads without defensive copying.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubGoal:
    """A single verifiable step in a goal decomposition tree."""

    name: str
    description: str
    verify: str  # Python expression evaluated by GoalVerifier
    timeout_sec: float = 30.0
    depends_on: tuple[str, ...] = ()
    strategy: str = ""
    strategy_params: dict = field(default_factory=dict)
    fail_action: str = ""


@dataclass(frozen=True)
class GoalTree:
    """Full decomposition of a high-level task into ordered SubGoals."""

    goal: str
    sub_goals: tuple[SubGoal, ...]
    context_snapshot: str = ""


@dataclass(frozen=True)
class StepRecord:
    """Execution record for a single SubGoal attempt."""

    sub_goal_name: str
    strategy: str
    success: bool
    verify_result: bool
    duration_sec: float
    error: str = ""
    fallback_used: bool = False


@dataclass(frozen=True)
class ExecutionTrace:
    """Complete record of a goal execution run."""

    goal_tree: GoalTree
    steps: tuple[StepRecord, ...]
    success: bool
    total_duration_sec: float
