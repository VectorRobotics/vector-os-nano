# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Skill protocol, @skill decorator, registry, and execution context.

Each skill declares (the model routes by the skill's description; nothing matches
user text against these):
- aliases: words/phrases for it (Chinese + English); the stop skill's are the P0
  stop phrases
- actuates: whether it moves the robot

A skill that needs a prerequisite calls it itself; the model sequences the rest.
"""
from __future__ import annotations

import logging
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
    actuates: bool = True,
):
    """Decorator that marks a class as a skill with routing metadata.

    Args:
        aliases: Words/phrases for this skill (multi-language); the stop skill's
                 are the P0 stop phrases.
        actuates: Whether the skill moves the robot (base, arm, gripper). Fail
                  closed: an undeclared skill actuates. A read-only perception /
                  status skill declares ``actuates=False``. Actuating skills are
                  serialized, confirmed on real hardware, and report post-state.

    Example::

        @skill(aliases=["close", "grip", "夹紧"])
        class GripperCloseSkill:
            name = "gripper_close"
            ...
    """
    def wrapper(cls):
        cls.__skill_aliases__ = aliases or []
        cls.__skill_actuates__ = bool(actuates)
        return cls

    if cls is not None:
        # Called without arguments: @skill
        cls.__skill_aliases__ = []
        cls.__skill_actuates__ = True
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
# SkillRegistry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """The skills of one body, by name."""

    def __init__(self) -> None:
        self._skills: dict[str, Any] = {}

    def register(self, skill_instance: Any) -> None:
        """Register a skill instance under its name."""
        self._skills[skill_instance.name] = skill_instance
        logger.debug("Registered skill %r", skill_instance.name)

    def get(self, name: str) -> Any | None:
        """Retrieve a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return all registered skill names."""
        return list(self._skills.keys())

    def to_schemas(self) -> list[dict]:
        """Serialize all skill schemas for LLM planner context.

        Includes aliases and failure_modes metadata for richer LLM context.
        """
        schemas: list[dict] = []
        for s in self._skills.values():
            aliases = getattr(s, '__skill_aliases__', [])
            failure_modes = getattr(s, 'failure_modes', [])
            schema = {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "preconditions": list(s.preconditions),
                "postconditions": list(s.postconditions),
                "effects": dict(s.effects),
                # Per-skill success predicate the planner should prefer for this
                # step's verify expression (single-source; optional attribute).
                # Defaults to the always-safe "True" when a skill declares none.
                "verify_hint": getattr(s, "verify_hint", "") or "True",
            }
            if aliases:
                schema["aliases"] = aliases
            if failure_modes:
                schema["failure_modes"] = failure_modes
            schemas.append(schema)
        return schemas
