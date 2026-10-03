# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""``@oracle`` — the tag that makes a verify-namespace callable count as evidence.

The verify namespace holds every callable a verify expression may call. Only a
callable tagged ``@oracle(scope=...)`` whose scope matches the active world —
``"robot"`` in a robot world, ``"dev"`` in the dev world — is an evidence oracle
(``evidence_names``). Everything else stays callable but can never ground a step:
untagged constant stubs, the actor's own memory readers (SceneGraph / ObjectMemory /
predict), and dev file predicates inside a robot world all fail closed (rule 5).

The registry is the tag itself — no second list to drift from the namespace. A world
that brings a new GT predicate tags it at its definition; nothing in the kernel
changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

_SCOPES = frozenset({"robot", "dev"})
_TRANSITIONS = frozenset({"rise", "fall", "level"})


@dataclass(frozen=True)
class OracleTag:
    """What a tagged oracle reads. ``scope`` decides evidence membership today;
    ``channel`` / ``transition`` describe which actor state change the predicate
    measures (rise = becomes true, fall = becomes false, level = state read) for the
    causation grader; ``max_tol`` is the cap the oracle clamps a caller tolerance to.
    """

    scope: str
    channel: str = "none"
    transition: str = "level"
    max_tol: float | None = None


def oracle(
    *,
    scope: str,
    channel: str = "none",
    transition: str = "level",
    max_tol: float | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a verify callable as a ground-truth oracle (sets ``fn.__oracle__``)."""
    if scope not in _SCOPES:
        raise ValueError(f"oracle scope {scope!r} not in {sorted(_SCOPES)}")
    if transition not in _TRANSITIONS:
        raise ValueError(f"oracle transition {transition!r} not in {sorted(_TRANSITIONS)}")
    tag = OracleTag(scope=scope, channel=channel, transition=transition, max_tol=max_tol)

    def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__oracle__ = tag  # type: ignore[attr-defined]
        return fn

    return mark


def evidence_names(ns: Mapping[str, Any], scope: str | None = None) -> frozenset[str]:
    """Names in *ns* whose callable carries an ``@oracle`` tag.

    *scope* restricts to that scope (``"robot"`` / ``"dev"``) — the done-gate passes
    the active world's scope so a dev predicate can never ground in a robot world.
    ``None`` (the reward gate, which has no world handle) takes the namespace's one
    scope; a namespace mixing scopes fails closed to the empty set, since the engine
    never merges dev predicates into a robot world.
    """
    tagged = {
        name: fn.__oracle__.scope
        for name, fn in ns.items()
        if isinstance(getattr(fn, "__oracle__", None), OracleTag)
    }
    if scope is None and len(set(tagged.values())) > 1:
        return frozenset()
    return frozenset(n for n, s in tagged.items() if scope is None or s == scope)
