"""Skill protocol, @skill decorator, registry, and execution context.

The @skill decorator replaces hard-coded routing. Each skill declares:
- aliases: words/phrases that trigger it (Chinese + English)
- direct: if True, execute immediately without LLM planning
- auto_steps: default skill chain for common patterns (e.g. scan→detect→pick)

The SkillRegistry matches user input against aliases and routes accordingly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# Sentinel for detecting whether legacy kwargs were explicitly supplied.
_UNSET: object = object()


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

class SkillContext:
    """Everything a skill needs during execution.

    Hardware is accessed via typed registries (dicts). Skills query capabilities
    before using hardware. Backward-compatible flat-field kwargs allow existing
    skill code (context.arm, context.base, etc.) to continue working unchanged.

    New-style construction (dict registries):
        ctx = SkillContext(
            arms={"so101": arm},
            grippers={"so101": gripper},
            bases={"go2": base},
            perception_sources={"realsense": cam},
            world_model=wm,
        )

    Legacy construction (flat fields, backward-compatible):
        ctx = SkillContext(arm=arm, gripper=gripper, base=base,
                           perception=cam, world_model=wm, calibration=cal)
    """

    def __init__(
        self,
        *,
        # New-style dict registries
        arms: dict | None = None,
        grippers: dict | None = None,
        bases: dict | None = None,
        perception_sources: dict | None = None,
        services: dict | None = None,
        # Shared state
        world_model: Any = None,
        calibration: Any = None,
        config: dict | None = None,
        # Legacy flat-field kwargs (backward-compatible)
        arm: Any = _UNSET,
        gripper: Any = _UNSET,
        perception: Any = _UNSET,
        base: Any = _UNSET,
    ) -> None:
        # Dict registries (new-style API)
        self.arms: dict = arms if arms is not None else {}
        self.grippers: dict = grippers if grippers is not None else {}
        self.bases: dict = bases if bases is not None else {}
        self.perception_sources: dict = (
            perception_sources if perception_sources is not None else {}
        )
        self.services: dict = services if services is not None else {}

        # Shared state
        self.world_model: Any = world_model
        self.calibration: Any = calibration
        self.config: dict = config if config is not None else {}

        # Legacy flat fields — stored as-is so old code can still read them;
        # property accessors prefer the dict registries when both are present.
        self._legacy_arm: Any = arm if arm is not _UNSET else None
        self._legacy_gripper: Any = gripper if gripper is not _UNSET else None
        self._legacy_perception: Any = perception if perception is not _UNSET else None
        self._legacy_base: Any = base if base is not _UNSET else None

        # Track whether legacy kwargs were explicitly supplied (for .arms / .base
        # backward-compat attribute access in old tests that assert arms is None).
        self._has_legacy_arm: bool = arm is not _UNSET
        self._has_legacy_gripper: bool = gripper is not _UNSET
        self._has_legacy_perception: bool = perception is not _UNSET
        self._has_legacy_base: bool = base is not _UNSET

    # --- Backward-compatible property accessors ---

    @property
    def arm(self) -> Any:
        """Return first arm from registry, falling back to legacy flat field."""
        if self.arms:
            return next(iter(self.arms.values()))
        return self._legacy_arm

    @property
    def gripper(self) -> Any:
        """Return first gripper from registry, falling back to legacy flat field."""
        if self.grippers:
            return next(iter(self.grippers.values()))
        return self._legacy_gripper

    @property
    def base(self) -> Any:
        """Return first base from registry, falling back to legacy flat field."""
        if self.bases:
            return next(iter(self.bases.values()))
        return self._legacy_base

    @base.setter
    def base(self, value: Any) -> None:
        """Set base — clears the bases dict and updates the legacy field."""
        self.bases.clear()
        self._legacy_base = value
        self._has_legacy_base = value is not None

    @property
    def perception(self) -> Any:
        """Return first perception source from registry, falling back to legacy."""
        if self.perception_sources:
            return next(iter(self.perception_sources.values()))
        return self._legacy_perception

    # --- Capability queries ---

    def has_arm(self, name: str | None = None) -> bool:
        if name is not None:
            return name in self.arms
        return bool(self.arms) or self._legacy_arm is not None

    def has_gripper(self, name: str | None = None) -> bool:
        if name is not None:
            return name in self.grippers
        return bool(self.grippers) or self._legacy_gripper is not None

    def has_base(self, name: str | None = None) -> bool:
        if name is not None:
            return name in self.bases
        return bool(self.bases) or self._legacy_base is not None

    def has_perception(self, name: str | None = None) -> bool:
        if name is not None:
            return name in self.perception_sources
        return bool(self.perception_sources) or self._legacy_perception is not None

    def get_arm(self, name: str | None = None) -> Any:
        if name is not None:
            return self.arms.get(name)
        return self.arm

    def get_gripper(self, name: str | None = None) -> Any:
        if name is not None:
            return self.grippers.get(name)
        return self.gripper

    def get_base(self, name: str | None = None) -> Any:
        if name is not None:
            return self.bases.get(name)
        return self.base

    def capabilities(self) -> dict:
        return {
            "has_arm": self.has_arm(),
            "has_gripper": self.has_gripper(),
            "has_base": self.has_base(),
            "has_perception": self.has_perception(),
            "arm_names": list(self.arms.keys()),
            "gripper_names": list(self.grippers.keys()),
            "base_names": list(self.bases.keys()),
            "perception_names": list(self.perception_sources.keys()),
        }

    def __repr__(self) -> str:
        parts = []
        if self.arms:
            parts.append(f"arms={list(self.arms.keys())!r}")
        elif self._legacy_arm is not None:
            parts.append("arm=<legacy>")
        if self.bases:
            parts.append(f"bases={list(self.bases.keys())!r}")
        elif self._legacy_base is not None:
            parts.append("base=<legacy>")
        if self.world_model is not None:
            parts.append("world_model=set")
        return f"SkillContext({', '.join(parts)})"


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
