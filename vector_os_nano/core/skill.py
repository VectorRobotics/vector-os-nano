"""Skill protocol, @skill decorator, registry, and execution context.

The @skill decorator replaces hard-coded routing. Each skill declares:
- aliases: words/phrases that trigger it (Chinese + English)
- direct: if True, execute immediately without LLM planning
- auto_steps: default skill chain for common patterns (e.g. scan→detect→pick)

The SkillRegistry matches user input against aliases and routes accordingly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# @skill decorator
# ---------------------------------------------------------------------------

def skill(
    cls=None,
    *,
    aliases: list[str] | None = None,
    direct: bool = False,
    auto_steps: list[str] | None = None,
):
    """Decorator that marks a class as a skill with routing metadata.

    Args:
        aliases: Words/phrases that trigger this skill (multi-language).
                 Matched against user input for automatic routing.
        direct: If True, execute immediately without LLM planning.
                For simple commands like "home", "open", "close".
        auto_steps: Default skill chain. E.g. ["scan", "detect", "pick"]
                    means this skill auto-expands to that sequence.

    Example::

        @skill(aliases=["grab", "抓", "拿"], auto_steps=["scan", "detect", "pick"])
        class PickSkill:
            name = "pick"
            description = "Pick up an object"
            ...

        @skill(aliases=["close", "grip", "夹紧"], direct=True)
        class GripperCloseSkill:
            name = "gripper_close"
            ...
    """
    def wrapper(cls):
        cls.__skill_aliases__ = aliases or []
        cls.__skill_direct__ = direct
        cls.__skill_auto_steps__ = auto_steps or []
        return cls

    if cls is not None:
        # Called without arguments: @skill
        cls.__skill_aliases__ = []
        cls.__skill_direct__ = False
        cls.__skill_auto_steps__ = []
        return cls

    return wrapper


# ---------------------------------------------------------------------------
# Skill Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Skill(Protocol):
    """Abstract skill interface."""

    name: str
    description: str
    parameters: dict
    preconditions: list[str]
    postconditions: list[str]
    effects: dict
    failure_modes: list[str]

    def execute(self, params: dict, context: "SkillContext") -> SkillResult: ...


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------

@dataclass
class SkillContext:
    """Everything a skill needs during execution."""

    arm: Any
    gripper: Any
    perception: Any
    world_model: Any
    calibration: Any
    config: dict = field(default_factory=dict)
    arms: dict | None = None
    base: Any | None = None


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class SkillMatch:
    """Result of matching user input against skill aliases."""
    skill_name: str
    direct: bool
    auto_steps: list[str]
    extracted_arg: str  # remaining text after alias match (e.g. "杯子" from "抓杯子")


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Manages skills with alias-based routing.

    Replaces all hard-coded command routing with declarative alias matching.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Any] = {}
        # alias → (skill_name, is_direct, auto_steps)
        self._alias_map: dict[str, tuple[str, bool, list[str]]] = {}

    def register(self, skill_instance: Any) -> None:
        """Register a skill instance. Reads @skill decorator metadata."""
        name = skill_instance.name
        self._skills[name] = skill_instance

        # Read decorator metadata
        aliases = getattr(skill_instance, '__skill_aliases__', [])
        direct = getattr(skill_instance, '__skill_direct__', False)
        auto_steps = getattr(skill_instance, '__skill_auto_steps__', [])

        # Also register the skill name itself as an alias
        self._alias_map[name.lower()] = (name, direct, auto_steps)

        for alias in aliases:
            self._alias_map[alias.lower()] = (name, direct, auto_steps)

        logger.debug("Registered skill %r with %d aliases, direct=%s",
                      name, len(aliases), direct)

    def match(self, user_input: str) -> SkillMatch | None:
        """Match user input against skill aliases.

        Tries exact match first, then prefix match (longest alias wins).
        Returns SkillMatch with extracted argument, or None if no match.

        Examples:
            "home"      → SkillMatch(home, direct=True, arg="")
            "抓杯子"    → SkillMatch(pick, direct=False, arg="杯子")
            "close grip" → SkillMatch(gripper_close, direct=True, arg="")
            "你好"       → None (no match, goes to LLM)
        """
        text = user_input.strip().lower()

        # 1. Exact match
        if text in self._alias_map:
            name, direct, auto = self._alias_map[text]
            return SkillMatch(name, direct, auto, "")

        # 2. Longest prefix match (e.g. "抓" matches in "抓杯子")
        best_match: tuple[str, bool, list[str]] | None = None
        best_len = 0
        best_arg = ""

        for alias, (name, direct, auto) in self._alias_map.items():
            if text.startswith(alias) and len(alias) > best_len:
                best_match = (name, direct, auto)
                best_len = len(alias)
                best_arg = text[len(alias):].strip()

        if best_match is not None and best_len > 0:
            name, direct, auto = best_match
            return SkillMatch(name, direct, auto, best_arg)

        return None

    def get(self, name: str) -> Any | None:
        """Retrieve a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return all registered skill names."""
        return list(self._skills.keys())

    def to_schemas(self) -> list[dict]:
        """Serialize all skill schemas for LLM planner context.

        Includes aliases, auto_steps, and failure_modes metadata for richer LLM context.
        """
        schemas: list[dict] = []
        for s in self._skills.values():
            aliases = getattr(s, '__skill_aliases__', [])
            auto = getattr(s, '__skill_auto_steps__', [])
            failure_modes = getattr(s, 'failure_modes', [])
            schema = {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "preconditions": list(s.preconditions),
                "postconditions": list(s.postconditions),
                "effects": dict(s.effects),
            }
            if aliases:
                schema["aliases"] = aliases
            if auto:
                schema["auto_steps"] = auto
            if failure_modes:
                schema["failure_modes"] = failure_modes
            schemas.append(schema)
        return schemas
