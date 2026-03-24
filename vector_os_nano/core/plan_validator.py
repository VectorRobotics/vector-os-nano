"""Plan validation and auto-repair for LLM-generated task plans.

Pure symbolic validation -- no LLM calls. Runs between plan generation
and execution in the agent pipeline.

Validation checks (in order):
  1. Skill existence — skill_name must be in the registry
  2. Required parameters — warn if a param has no default and no required=False
  3. Enum conformance — enum params must match an allowed value
  4. Type conformance — typed params must match the declared type
  5. Dependency references — depends_on must point to step_ids in the plan
  6. Cycle detection — depends_on graph must be acyclic
  7. Precondition satisfiability — predicates are checked against world model

Auto-repair actions (validate_and_repair):
  1. Alias-based skill name fix — e.g. 'grab' -> 'pick'
  2. Fuzzy enum value fix — case/whitespace/underscore normalisation + substring
  3. Default fill — missing params with declared defaults are filled in
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vector_os_nano.core.types import TaskPlan, TaskStep


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """A single validation error or warning."""

    step_id: str
    field: str          # "skill_name", "parameters.location", etc.
    code: str           # "unknown_skill", "invalid_enum", "missing_required", etc.
    message: str
    suggestion: str


@dataclass(frozen=True)
class Repair:
    """Record of an auto-repair applied to a plan."""

    step_id: str
    field: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    """Result of plan validation."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PlanValidator
# ---------------------------------------------------------------------------


class PlanValidator:
    """Validate LLM-generated task plans before execution.

    Pure symbolic -- no LLM dependency.
    """

    def __init__(self, skill_registry: Any, world_model: Any) -> None:
        self._registry = skill_registry
        self._world_model = world_model
        self._skill_names: set[str] = set(skill_registry.list_skills())

        # Build alias -> canonical_name map (lowercase keys)
        self._alias_map: dict[str, str] = {}
        for name in self._skill_names:
            skill = skill_registry.get(name)
            if skill:
                for alias in getattr(skill, '__skill_aliases__', []):
                    self._alias_map[alias.lower()] = name
                self._alias_map[name.lower()] = name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, plan: TaskPlan) -> ValidationResult:
        """Validate the plan and return a ValidationResult.

        Never raises -- all issues are captured as errors or warnings.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        step_ids = {s.step_id for s in plan.steps}

        for step in plan.steps:
            # 1. Skill existence
            if step.skill_name not in self._skill_names:
                errors.append(ValidationError(
                    step_id=step.step_id,
                    field="skill_name",
                    code="unknown_skill",
                    message=f"Skill {step.skill_name!r} not found in registry",
                    suggestion=self._suggest_skill_name(step.skill_name),
                ))
            else:
                skill = self._registry.get(step.skill_name)
                if skill:
                    skill_params: dict = getattr(skill, 'parameters', {}) or {}

                    # 2. Required parameters
                    for pname, pdef in skill_params.items():
                        if isinstance(pdef, dict):
                            has_default = "default" in pdef
                            explicitly_optional = pdef.get("required") is False
                            if not has_default and not explicitly_optional:
                                if pname not in step.parameters:
                                    warnings.append(ValidationError(
                                        step_id=step.step_id,
                                        field=f"parameters.{pname}",
                                        code="missing_required",
                                        message=f"Required parameter {pname!r} is missing",
                                        suggestion=f"Add {pname!r} to parameters",
                                    ))

                    # 3. Enum conformance
                    for pname, pvalue in step.parameters.items():
                        pdef = skill_params.get(pname, {})
                        if isinstance(pdef, dict) and "enum" in pdef:
                            if pvalue not in pdef["enum"]:
                                errors.append(ValidationError(
                                    step_id=step.step_id,
                                    field=f"parameters.{pname}",
                                    code="invalid_enum",
                                    message=f"Value {pvalue!r} not in allowed enum {pdef['enum']}",
                                    suggestion=f"Use one of: {pdef['enum']}",
                                ))

                    # 4. Type conformance
                    for pname, pvalue in step.parameters.items():
                        pdef = skill_params.get(pname, {})
                        if isinstance(pdef, dict) and "type" in pdef:
                            if not self._check_type(pvalue, pdef["type"]):
                                errors.append(ValidationError(
                                    step_id=step.step_id,
                                    field=f"parameters.{pname}",
                                    code="type_mismatch",
                                    message=(
                                        f"Expected type {pdef['type']!r} for {pname!r}, "
                                        f"got {type(pvalue).__name__!r}"
                                    ),
                                    suggestion=f"Convert {pname!r} to {pdef['type']}",
                                ))

            # 5. Dependency references
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(ValidationError(
                        step_id=step.step_id,
                        field="depends_on",
                        code="missing_dependency",
                        message=f"Dependency {dep!r} not in plan step_ids",
                        suggestion="Remove or fix the dependency reference",
                    ))

        # 6. Cycle detection (only if no missing_dependency errors, to avoid confusion)
        if not any(e.code == "missing_dependency" for e in errors):
            if self._has_cycle(plan.steps):
                errors.append(ValidationError(
                    step_id="plan",
                    field="depends_on",
                    code="circular_dependency",
                    message="Circular dependency detected in plan steps",
                    suggestion="Remove the circular dependency to allow execution",
                ))

        # 7. Precondition satisfiability
        for step in plan.steps:
            for pred in step.preconditions:
                if not self._world_model.check_predicate(pred):
                    warnings.append(ValidationError(
                        step_id=step.step_id,
                        field="preconditions",
                        code="precondition_unsatisfiable",
                        message=f"Precondition {pred!r} is not currently satisfied",
                        suggestion="Ensure prior steps in the plan establish this condition",
                    ))

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_and_repair(self, plan: TaskPlan) -> tuple[TaskPlan, list[Repair]]:
        """Validate and auto-repair common LLM errors.

        Returns:
            (repaired_plan, repairs) -- repaired_plan is a NEW TaskPlan (original unchanged).
        """
        repairs: list[Repair] = []
        new_steps: list[TaskStep] = []

        for step in plan.steps:
            skill_name = step.skill_name
            parameters = dict(step.parameters)

            # Repair 1: Fix unknown skill names via alias map
            if skill_name not in self._skill_names:
                fixed = self._alias_map.get(skill_name.lower())
                if fixed:
                    repairs.append(Repair(
                        step_id=step.step_id,
                        field="skill_name",
                        old_value=skill_name,
                        new_value=fixed,
                        reason="alias match",
                    ))
                    skill_name = fixed

            # Repair 2: Fix enum values + Repair 3: Fill missing defaults
            skill = self._registry.get(skill_name)
            if skill:
                skill_params: dict = getattr(skill, 'parameters', {}) or {}

                # Repair 2: fuzzy enum fix
                for pname, pvalue in list(parameters.items()):
                    pdef = skill_params.get(pname, {})
                    if isinstance(pdef, dict) and "enum" in pdef:
                        if pvalue not in pdef["enum"]:
                            fixed_val = self._fuzzy_enum_match(pvalue, pdef["enum"])
                            if fixed_val is not None:
                                repairs.append(Repair(
                                    step_id=step.step_id,
                                    field=f"parameters.{pname}",
                                    old_value=pvalue,
                                    new_value=fixed_val,
                                    reason="fuzzy enum match",
                                ))
                                parameters[pname] = fixed_val

                # Repair 3: fill missing defaults
                for pname, pdef in skill_params.items():
                    if isinstance(pdef, dict) and pname not in parameters and "default" in pdef:
                        default_val = pdef["default"]
                        parameters[pname] = default_val
                        repairs.append(Repair(
                            step_id=step.step_id,
                            field=f"parameters.{pname}",
                            old_value=None,
                            new_value=default_val,
                            reason="default fill",
                        ))

            new_steps.append(TaskStep(
                step_id=step.step_id,
                skill_name=skill_name,
                parameters=parameters,
                depends_on=list(step.depends_on),
                preconditions=list(step.preconditions),
                postconditions=list(step.postconditions),
            ))

        repaired = TaskPlan(
            goal=plan.goal,
            steps=new_steps,
            message=plan.message,
            requires_clarification=plan.requires_clarification,
            clarification_question=plan.clarification_question,
        )
        return repaired, repairs

    # ------------------------------------------------------------------
    # Static helpers -- fuzzy matching
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_enum_match(value: str, enum_values: list) -> str | None:
        """Case-insensitive, whitespace-normalised match against enum values.

        Priority:
          1. Exact normalised match (lowercase, strip, underscore->space)
          2. Substring match (query in enum_value or vice versa)

        Returns the matched enum value, or None if no match found.
        """
        if not isinstance(value, str):
            return None
        norm = value.lower().strip().replace("_", " ")
        # Pass 1: exact normalised match
        for ev in enum_values:
            if isinstance(ev, str) and ev.lower().strip().replace("_", " ") == norm:
                return ev
        # Pass 2: substring match
        for ev in enum_values:
            if isinstance(ev, str):
                ev_norm = ev.lower().strip().replace("_", " ")
                if norm in ev_norm or ev_norm in norm:
                    return ev
        return None

    # ------------------------------------------------------------------
    # Static helpers -- type checking
    # ------------------------------------------------------------------

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        """Return True if value is compatible with the expected type string.

        Type strings follow JSON Schema conventions plus common aliases.
        Unknown type strings always return True (no false rejections).
        """
        type_map: dict[str, Any] = {
            "string": str,
            "str": str,
            "float": (int, float),
            "number": (int, float),
            "int": int,
            "integer": int,
            "bool": bool,
            "boolean": bool,
        }
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True  # Unknown type -- pass through
        return isinstance(value, expected_types)

    # ------------------------------------------------------------------
    # Static helpers -- cycle detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_cycle(steps: list[TaskStep]) -> bool:
        """Return True if the depends_on graph contains a cycle.

        Uses recursive DFS with an explicit recursion stack (visited + in_stack sets).
        Only considers nodes that are actually in the step list.
        """
        adj: dict[str, list[str]] = {s.step_id: list(s.depends_on) for s in steps}
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for dep in adj.get(node, []):
                if dfs(dep):
                    return True
            in_stack.remove(node)
            return False

        return any(dfs(s.step_id) for s in steps if s.step_id not in visited)

    # ------------------------------------------------------------------
    # Helper -- skill name suggestion
    # ------------------------------------------------------------------

    def _suggest_skill_name(self, name: str) -> str:
        """Return a human-readable suggestion for an unknown skill name."""
        name_lower = name.lower()
        # Exact alias match
        if name_lower in self._alias_map:
            return f"Did you mean {self._alias_map[name_lower]!r}?"
        # Substring match against known skill names
        for known in sorted(self._skill_names):
            if name_lower in known.lower() or known.lower() in name_lower:
                return f"Did you mean {known!r}?"
        return "No close match found"
