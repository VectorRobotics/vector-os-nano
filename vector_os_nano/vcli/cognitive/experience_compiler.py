"""ExperienceCompiler — compile successful ExecutionTraces into reusable GoalTemplates.

Compilation pipeline:
1. Filter: only success=True traces
2. Group: by structural signature (sub-goal name-prefix sequence)
3. Parameterize: find varying parts across group members → ${param} placeholders
4. Return: list of GoalTemplate
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, SubGoal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubGoalTemplate:
    """Parameterized template for a single sub-goal step."""

    name_pattern: str           # "reach_${room}"
    description_pattern: str    # "go to ${room}"
    verify_pattern: str         # "nearest_room() == '${room}'"
    strategy: str = ""
    timeout_sec: float = 30.0
    depends_on: tuple[str, ...] = ()
    fail_action: str = ""


@dataclass(frozen=True)
class GoalTemplate:
    """Reusable, parameterized template compiled from successful ExecutionTraces."""

    name: str                                       # "find_object_in_room"
    description: str                                # "Find an object in a specific room"
    parameters: tuple[str, ...]                     # ("object", "room")
    sub_goal_templates: tuple[SubGoalTemplate, ...]
    success_count: int = 0
    fail_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REACH_PREFIXES = {"reach", "navigate", "go", "move"}
_DETECT_PREFIXES = {"detect", "find", "locate", "search"}
_OBSERVE_PREFIXES = {"observe", "look", "scan", "watch"}


def _name_prefix(name: str) -> str:
    """Return the first '_'-separated token of a sub-goal name."""
    return name.split("_")[0]


def _structural_signature(tree: GoalTree) -> tuple[str, ...]:
    """Compute a structural signature from the prefix of each sub-goal name."""
    return tuple(_name_prefix(sg.name) for sg in tree.sub_goals)


def _suffix(name: str) -> str:
    """Return the portion of name after the first underscore (or empty string)."""
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _param_name_for_prefix(prefix: str, used_names: set[str]) -> str:
    """Heuristically assign a parameter name based on the sub-goal prefix."""
    if prefix in _REACH_PREFIXES or prefix in _OBSERVE_PREFIXES:
        candidate = "room"
    elif prefix in _DETECT_PREFIXES:
        candidate = "object"
    else:
        # Generic fallback — param1, param2, ...
        i = 1
        while f"param{i}" in used_names:
            i += 1
        return f"param{i}"

    if candidate not in used_names:
        return candidate
    # Already taken — use numbered variant
    i = 2
    while f"{candidate}{i}" in used_names:
        i += 1
    return f"{candidate}{i}"


def _replace_value(text: str, value: str, param: str) -> str:
    """Replace all occurrences of `value` in `text` with `${param}`."""
    if not value:
        return text
    return text.replace(value, f"${{{param}}}")


def _build_sub_goal_template(
    sub_goal: SubGoal,
    value_to_param: dict[str, str],
) -> SubGoalTemplate:
    """Build a SubGoalTemplate from a concrete SubGoal by substituting known values."""
    name_pat = sub_goal.name
    desc_pat = sub_goal.description
    verify_pat = sub_goal.verify

    for value, param in value_to_param.items():
        name_pat = _replace_value(name_pat, value, param)
        desc_pat = _replace_value(desc_pat, value, param)
        verify_pat = _replace_value(verify_pat, value, param)

    # Parameterize depends_on entries
    dep_pats: list[str] = []
    for dep in sub_goal.depends_on:
        dep_pat = dep
        for value, param in value_to_param.items():
            dep_pat = _replace_value(dep_pat, value, param)
        dep_pats.append(dep_pat)

    return SubGoalTemplate(
        name_pattern=name_pat,
        description_pattern=desc_pat,
        verify_pattern=verify_pat,
        strategy=sub_goal.strategy,
        timeout_sec=sub_goal.timeout_sec,
        depends_on=tuple(dep_pats),
        fail_action=sub_goal.fail_action,
    )


# ---------------------------------------------------------------------------
# ExperienceCompiler
# ---------------------------------------------------------------------------

class ExperienceCompiler:
    """Compile successful ExecutionTraces into reusable GoalTemplate objects."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, traces: list[ExecutionTrace]) -> list[GoalTemplate]:
        """Extract parameterized templates from successful traces.

        Steps:
        1. Filter: only success=True traces
        2. Group: by structural signature (sub-goal name-prefix sequence)
        3. Parameterize: identify varying parts as parameters
        4. Return: list of GoalTemplate
        """
        successful = [t for t in traces if t.success]
        if not successful:
            return []

        # Group by structural signature
        groups: dict[tuple[str, ...], list[GoalTree]] = {}
        for trace in successful:
            sig = _structural_signature(trace.goal_tree)
            groups.setdefault(sig, []).append(trace.goal_tree)

        templates: list[GoalTemplate] = []
        for sig, trees in groups.items():
            tmpl = self._build_template(sig, trees)
            if tmpl is not None:
                templates.append(tmpl)

        return templates

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _build_template(
        self,
        sig: tuple[str, ...],
        trees: list[GoalTree],
    ) -> GoalTemplate | None:
        """Build a GoalTemplate from a group of structurally similar GoalTrees."""
        if not trees:
            return None

        if len(trees) == 1:
            return self._build_concrete_template(sig, trees[0])

        return self._build_parameterized_template(sig, trees)

    def _build_concrete_template(
        self,
        sig: tuple[str, ...],
        tree: GoalTree,
    ) -> GoalTemplate:
        """Build a concrete (no-parameter) template from a single GoalTree."""
        name = "_".join(sig)
        sub_goal_templates = tuple(
            SubGoalTemplate(
                name_pattern=sg.name,
                description_pattern=sg.description,
                verify_pattern=sg.verify,
                strategy=sg.strategy,
                timeout_sec=sg.timeout_sec,
                depends_on=sg.depends_on,
                fail_action=sg.fail_action,
            )
            for sg in tree.sub_goals
        )
        return GoalTemplate(
            name=name,
            description=tree.goal,
            parameters=(),
            sub_goal_templates=sub_goal_templates,
        )

    def _build_parameterized_template(
        self,
        sig: tuple[str, ...],
        trees: list[GoalTree],
    ) -> GoalTemplate:
        """Build a parameterized template by finding varying values across trees."""
        # For each sub-goal position, collect suffixes from all trees
        n_sub_goals = len(sig)
        position_suffixes: list[list[str]] = [[] for _ in range(n_sub_goals)]

        for tree in trees:
            for i, sg in enumerate(tree.sub_goals):
                position_suffixes[i].append(_suffix(sg.name))

        # Find which positions have varying suffixes → candidates for parameters
        # Map value → param_name (greedy assignment by position)
        value_to_param: dict[str, str] = {}
        used_params: set[str] = set()

        for i, (prefix, suffixes) in enumerate(zip(sig, position_suffixes)):
            unique_suffixes = set(s for s in suffixes if s)
            if len(unique_suffixes) <= 1:
                continue  # this position is constant — skip

            # All different values at this position → a parameter
            param = _param_name_for_prefix(prefix, used_params)
            used_params.add(param)

            for val in unique_suffixes:
                if val and val not in value_to_param:
                    value_to_param[val] = param

        # Build sub-goal templates from the first tree as reference
        reference_tree = trees[0]
        sub_goal_templates = tuple(
            _build_sub_goal_template(sg, value_to_param)
            for sg in reference_tree.sub_goals
        )

        # Template name is derived from the signature
        name = "_".join(sig)
        description = f"Template for: {reference_tree.goal}"

        return GoalTemplate(
            name=name,
            description=description,
            parameters=tuple(sorted(used_params)),
            sub_goal_templates=sub_goal_templates,
        )
