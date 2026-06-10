# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Protocol-based provider resolution — Phase E W3.3 (`_base` seam first).

The kernel asks for a PROTOCOL, not a named private attribute. Instead of
``getattr(agent, "_base")`` duck-typing scattered through the cognitive
layer, callers resolve a provider against a NARROW ``runtime_checkable``
Protocol describing exactly the surface they call, and failures are LOUD
and diagnostic (rule 8): "no base provider met spec X" with what IS present
and which methods are missing — never an AttributeError or a silent None.

These protocols are deliberately narrower than ``hardware/base.py``'s full
``BaseProtocol`` HAL: a provider only needs what the calling layer uses, so
a future embodiment (e.g. the habitat kinematic base) satisfies the spec by
construction without faking the full HAL.

``ProviderError`` subclasses ``RuntimeError`` so every existing
``except RuntimeError`` recovery path keeps working.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseStateProvider(Protocol):
    """The read-only base surface the kernel queries (context building)."""

    def get_position(self) -> Any: ...

    def get_heading(self) -> float: ...


@runtime_checkable
class BaseMotionProvider(Protocol):
    """The motion surface the locomotion primitives drive."""

    def get_position(self) -> Any: ...

    def get_heading(self) -> float: ...

    def walk(self, vx: float, vy: float, vyaw: float, duration: float) -> bool: ...

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None: ...

    def stop(self) -> None: ...


class ProviderError(RuntimeError):
    """A required provider is absent or does not satisfy its spec."""


def _missing_methods(obj: Any, protocol: type) -> list[str]:
    attrs = getattr(protocol, "__protocol_attrs__", None) or ()
    return sorted(a for a in attrs if not callable(getattr(obj, a, None)))


def _conforms(obj: Any, protocol: type) -> bool:
    """True when *obj* satisfies *protocol*.

    ``isinstance`` first (py3.12 checks via ``inspect.getattr_static``), then a
    DYNAMIC attribute-presence fallback: test doubles like ``MagicMock`` create
    attributes on access, which the static check cannot see — for them, every
    protocol method resolving to a callable via plain ``getattr`` counts as
    conforming (exactly the duck-typing contract the old code implied), while
    ``None`` and genuinely partial objects still fail with named gaps.
    """
    if isinstance(obj, protocol):
        return True
    attrs = getattr(protocol, "__protocol_attrs__", None) or ()
    if not attrs:
        return False
    return all(callable(getattr(obj, a, None)) for a in attrs)


def ensure_provider(obj: Any, protocol: type, *, what: str) -> Any:
    """Return *obj* when it satisfies *protocol*; raise a diagnostic error otherwise."""
    if obj is None:
        raise ProviderError(
            f"no {what} provider met spec {protocol.__name__}: nothing is "
            f"connected (arm-only / baseless world?)"
        )
    if not _conforms(obj, protocol):
        missing = ", ".join(_missing_methods(obj, protocol)) or "<unknown>"
        raise ProviderError(
            f"no {what} provider met spec {protocol.__name__}: candidate "
            f"{type(obj).__name__} is missing [{missing}]"
        )
    return obj


def resolve_provider(
    agent: Any,
    protocol: type,
    *,
    what: str,
    hint_attrs: tuple[str, ...] = ("_base", "base"),
    required: bool = True,
) -> Any:
    """Resolve a provider from *agent* by PROTOCOL, fail-loud by default.

    ``hint_attrs`` keeps discovery compatible with today's attribute layout
    (the migration is incremental); conformance is decided ONLY by
    ``isinstance(candidate, protocol)``. ``required=False`` returns ``None``
    when nothing conforms (for fail-safe surfaces like context building).
    """
    candidates: list[tuple[str, Any]] = []
    for attr in hint_attrs:
        obj = getattr(agent, attr, None)
        if obj is not None:
            candidates.append((attr, obj))
    matching = [obj for _, obj in candidates if _conforms(obj, protocol)]
    if matching:
        return matching[0]
    if not required:
        return None
    if candidates:
        attr, obj = candidates[0]
        missing = ", ".join(_missing_methods(obj, protocol)) or "<unknown>"
        raise ProviderError(
            f"no {what} provider met spec {protocol.__name__}: candidate "
            f"{attr}={type(obj).__name__} is missing [{missing}]"
        )
    raise ProviderError(
        f"no {what} provider met spec {protocol.__name__}: agent exposes "
        f"none of {list(hint_attrs)} (arm-only / baseless world?)"
    )
