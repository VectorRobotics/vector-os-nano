"""Primitives package — module-level function wrappers for hardware and world state.

All primitives read from a module-global PrimitiveContext, matching the CaP-X
pattern where generated code calls module-level functions directly.

Usage::

    from vector_os_nano.vcli.primitives import init_primitives, PrimitiveContext
    ctx = PrimitiveContext(base=robot_base, scene_graph=sg, vlm=vlm)
    init_primitives(ctx)

    from vector_os_nano.vcli.primitives import locomotion
    pos = locomotion.get_position()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrimitiveContext:
    """Runtime context for primitive APIs.

    All fields are optional so primitives can be used in partial configurations
    (e.g. perception-only, or world-query-only without hardware).
    """

    base: Any | None = None              # BaseProtocol (Go2 hardware)
    scene_graph: Any | None = None       # SceneGraph instance
    vlm: Any | None = None               # Go2VLMPerception instance
    nav_client: Any | None = None        # NavStackClient instance
    skill_registry: Any | None = None    # SkillRegistry instance


_ctx: PrimitiveContext | None = None


def init_primitives(ctx: PrimitiveContext) -> None:
    """Initialize all primitive modules with the provided context.

    Must be called once before any primitive functions are used.

    Args:
        ctx: The runtime context containing hardware and world references.
    """
    global _ctx
    _ctx = ctx
    from vector_os_nano.vcli.primitives import locomotion, navigation, perception, world
    locomotion._ctx = ctx
    navigation._ctx = ctx
    perception._ctx = ctx
    world._ctx = ctx


def get_context() -> PrimitiveContext:
    """Return the current primitive context.

    Returns:
        The active PrimitiveContext.

    Raises:
        RuntimeError: If init_primitives() has not been called.
    """
    if _ctx is None:
        raise RuntimeError("Primitives not initialized. Call init_primitives() first.")
    return _ctx
