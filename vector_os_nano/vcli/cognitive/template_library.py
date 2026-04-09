"""TemplateLibrary — store, persist, and retrieve GoalTemplates.

Phase 2 matching strategy: simple keyword matching (no LLM required).
Phase 3 can extend with LLM-based semantic matching.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal
from vector_os_nano.vcli.cognitive.experience_compiler import GoalTemplate, SubGoalTemplate

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.expanduser("~/.vector_os_nano/goal_templates.json")

# ---------------------------------------------------------------------------
# Known keyword sets for simple parameter extraction
# ---------------------------------------------------------------------------

_KNOWN_ROOMS = {
    "kitchen", "bedroom", "hallway", "bathroom", "livingroom", "living_room",
    "garage", "office", "garden", "dining", "corridor", "lobby", "lab",
    "storage", "library", "entrance", "room",
}

_KNOWN_OBJECTS = {
    "cup", "book", "key", "phone", "laptop", "bag", "bottle", "pen", "pencil",
    "plate", "glass", "chair", "table", "box", "ball", "toy", "remote",
    "tablet", "charger", "wallet", "hat", "shoes", "jacket",
}

# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _sub_goal_template_to_dict(sgt: SubGoalTemplate) -> dict:
    return {
        "name_pattern": sgt.name_pattern,
        "description_pattern": sgt.description_pattern,
        "verify_pattern": sgt.verify_pattern,
        "strategy": sgt.strategy,
        "timeout_sec": sgt.timeout_sec,
        "depends_on": list(sgt.depends_on),
        "fail_action": sgt.fail_action,
    }


def _sub_goal_template_from_dict(d: dict) -> SubGoalTemplate:
    return SubGoalTemplate(
        name_pattern=d["name_pattern"],
        description_pattern=d["description_pattern"],
        verify_pattern=d["verify_pattern"],
        strategy=d.get("strategy", ""),
        timeout_sec=float(d.get("timeout_sec", 30.0)),
        depends_on=tuple(d.get("depends_on", [])),
        fail_action=d.get("fail_action", ""),
    )


def _goal_template_to_dict(t: GoalTemplate) -> dict:
    return {
        "name": t.name,
        "description": t.description,
        "parameters": list(t.parameters),
        "sub_goal_templates": [_sub_goal_template_to_dict(s) for s in t.sub_goal_templates],
        "success_count": t.success_count,
        "fail_count": t.fail_count,
    }


def _goal_template_from_dict(d: dict) -> GoalTemplate:
    return GoalTemplate(
        name=d["name"],
        description=d.get("description", ""),
        parameters=tuple(d.get("parameters", [])),
        sub_goal_templates=tuple(
            _sub_goal_template_from_dict(s) for s in d.get("sub_goal_templates", [])
        ),
        success_count=int(d.get("success_count", 0)),
        fail_count=int(d.get("fail_count", 0)),
    )


# ---------------------------------------------------------------------------
# Keyword extraction from task string
# ---------------------------------------------------------------------------

def _tokenize(task: str) -> set[str]:
    """Split task string into lowercase word tokens."""
    import re
    return set(re.findall(r"[a-z]+", task.lower()))


def _extract_param_values(
    task: str,
    parameters: tuple[str, ...],
) -> dict[str, str] | None:
    """Attempt to extract values for each parameter from the task string.

    Returns a mapping {param_name: value} if all parameters can be filled,
    or None if any parameter cannot be resolved.
    """
    tokens = _tokenize(task)
    result: dict[str, str] = {}

    for param in parameters:
        value: str | None = None

        if param == "room":
            for token in tokens:
                if token in _KNOWN_ROOMS:
                    value = token
                    break
        elif param == "object":
            for token in tokens:
                if token in _KNOWN_OBJECTS:
                    value = token
                    break
        else:
            # Generic param — try to find any content word not already used
            used = set(result.values())
            for token in tokens:
                if token not in _KNOWN_ROOMS and token not in _KNOWN_OBJECTS:
                    if token not in used:
                        value = token
                        break

        if value is None:
            return None  # Could not fill this parameter
        result[param] = value

    return result


# ---------------------------------------------------------------------------
# Substitution helpers
# ---------------------------------------------------------------------------

def _substitute(pattern: str, params: dict[str, str]) -> str:
    """Replace ${param} placeholders in pattern with actual values."""
    result = pattern
    for key, value in params.items():
        result = result.replace(f"${{{key}}}", value)
    return result


def _instantiate_sub_goal(sgt: SubGoalTemplate, params: dict[str, str]) -> SubGoal:
    """Create a concrete SubGoal from a SubGoalTemplate and parameter values."""
    depends_on = tuple(_substitute(dep, params) for dep in sgt.depends_on)
    return SubGoal(
        name=_substitute(sgt.name_pattern, params),
        description=_substitute(sgt.description_pattern, params),
        verify=_substitute(sgt.verify_pattern, params),
        strategy=sgt.strategy,
        timeout_sec=sgt.timeout_sec,
        depends_on=depends_on,
        fail_action=sgt.fail_action,
    )


# ---------------------------------------------------------------------------
# TemplateLibrary
# ---------------------------------------------------------------------------

class TemplateLibrary:
    """Store, retrieve, and persist GoalTemplates."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._path = persist_path or _DEFAULT_PATH
        self._templates: list[GoalTemplate] = []
        self.load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, template: GoalTemplate) -> None:
        """Add or replace a template (matched by name)."""
        self._templates = [t for t in self._templates if t.name != template.name]
        self._templates.append(template)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, task: str) -> tuple[GoalTemplate, dict[str, str]] | None:
        """Match task description to a stored template using keyword extraction.

        Returns (template, extracted_params) or None if no confident match.

        For concrete templates (0 parameters), any task containing a keyword
        related to the template name is considered a match.
        """
        best: tuple[GoalTemplate, dict[str, str]] | None = None

        for template in self._templates:
            if not template.parameters:
                # Concrete template — match based on template name keywords vs task tokens
                if self._matches_concrete(template, task):
                    best = (template, {})
                    break
            else:
                params = _extract_param_values(task, template.parameters)
                if params is not None:
                    # Confirm there's enough signal — at least one param matched something
                    if params:
                        best = (template, params)
                        break

        return best

    def _matches_concrete(self, template: GoalTemplate, task: str) -> bool:
        """Check if a task string is plausibly matched by a concrete template."""
        tokens = _tokenize(task)
        # Use sub-goal name patterns (concrete, no placeholders) as signal
        for sgt in template.sub_goal_templates:
            name_words = set(sgt.name_pattern.lower().replace("_", " ").split())
            if name_words & tokens:
                return True
        return False

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def instantiate(self, template: GoalTemplate, params: dict[str, str]) -> GoalTree:
        """Create a concrete GoalTree from template + parameter values."""
        sub_goals = tuple(
            _instantiate_sub_goal(sgt, params)
            for sgt in template.sub_goal_templates
        )
        # Build a descriptive goal string
        goal = _substitute(template.description, params)
        return GoalTree(goal=goal, sub_goals=sub_goals)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist templates to JSON file."""
        path = Path(self._path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [_goal_template_to_dict(t) for t in self._templates]
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            logger.debug("TemplateLibrary: saved %d templates to %s", len(self._templates), path)
        except OSError as exc:
            logger.warning("TemplateLibrary: save failed: %s", exc)

    def load(self) -> None:
        """Load templates from JSON file (no-op if file absent or invalid)."""
        path = Path(self._path)
        if not path.exists():
            return
        try:
            raw = path.read_text()
            data = json.loads(raw)
            self._templates = [_goal_template_from_dict(d) for d in data]
            logger.debug("TemplateLibrary: loaded %d templates from %s", len(self._templates), path)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("TemplateLibrary: load failed (%s) — starting empty", exc)
            self._templates = []
