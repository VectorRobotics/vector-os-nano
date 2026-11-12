# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""``@oracle`` — the tag that makes a verify-namespace callable count as evidence.

The verify namespace holds every callable a verify expression may call. Only a
callable tagged ``@oracle(scope=...)`` whose scope matches the active world —
``"robot"`` in a robot world, ``"dev"`` in the dev world — is an evidence oracle
(``evidence_names``). Everything else stays callable but can never ground a step:
untagged helpers, the actor's own memory readers (SceneGraph), and dev file
predicates inside a robot world all fail closed (rule 5).

The registry is the tag itself — no second list to drift from the namespace. A world
that brings a new GT predicate tags it at its definition; nothing in the kernel
changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

_SCOPES = frozenset({"robot", "dev"})
_TRANSITIONS = frozenset({"rise", "fall", "level"})
# Return domains: a bool, a non-negative count, or a closed numeric range (lo, hi).
_DOMAINS = frozenset({"bool", "count"})
# The window a transition is judged over: the step (default), or the whole turn for an
# oracle whose truth is itself turn-scoped (kept_holding: never dropped since the turn
# started), so the actor caused it only by a transition since the turn began.
_WINDOWS = frozenset({"step", "turn"})


@dataclass(frozen=True)
class OracleTag:
    """What a tagged oracle reads. ``scope`` decides evidence membership today;
    ``channel`` / ``transition`` describe which actor state change the predicate
    measures (rise = becomes true, fall = becomes false, level = state read) for the
    causation grader; ``max_tol`` is the cap the oracle clamps a caller tolerance to;
    ``domain`` is the return domain ("bool", "count", a numeric ``(lo, hi)`` range, or
    None = unbounded) the classifier uses to reject a bound vacuous over it; ``window``
    is what the transition is judged over ("step" or "turn").
    """

    scope: str
    channel: str = "none"
    transition: str = "level"
    max_tol: float | None = None
    domain: Any = None
    window: str = "step"


def oracle(
    *,
    scope: str,
    channel: str = "none",
    transition: str = "level",
    max_tol: float | None = None,
    domain: Any = None,
    window: str = "step",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a verify callable as a ground-truth oracle (sets ``fn.__oracle__``)."""
    if scope not in _SCOPES:
        raise ValueError(f"oracle scope {scope!r} not in {sorted(_SCOPES)}")
    if transition not in _TRANSITIONS:
        raise ValueError(f"oracle transition {transition!r} not in {sorted(_TRANSITIONS)}")
    if not (domain is None or domain in _DOMAINS
            or (isinstance(domain, tuple) and len(domain) == 2 and domain[0] < domain[1])):
        raise ValueError(f"oracle domain {domain!r}: one of {sorted(_DOMAINS)}, (lo, hi) or None")
    if window not in _WINDOWS:
        raise ValueError(f"oracle window {window!r} not in {sorted(_WINDOWS)}")
    tag = OracleTag(scope=scope, channel=channel, transition=transition, max_tol=max_tol,
                    domain=domain, window=window)

    def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__oracle__ = tag  # type: ignore[attr-defined]
        return fn

    return mark


def oracle_tags(ns: Mapping[str, Any], scope: str) -> dict[str, OracleTag]:
    """``{name: OracleTag}`` for the evidence oracles of *ns* (same rule as
    ``evidence_names``) — what the causation grader reads channel/transition from."""
    names = evidence_names(ns, scope)
    return {n: ns[n].__oracle__ for n in names}


def evidence_names(ns: Mapping[str, Any], scope: str) -> frozenset[str]:
    """Names in *ns* whose callable carries an ``@oracle`` tag of *scope*
    (``"robot"`` / ``"dev"``): the done-gate passes the active world's scope, so a
    dev predicate can never ground in a robot world."""
    return frozenset(
        name for name, fn in ns.items()
        if isinstance(tag := getattr(fn, "__oracle__", None), OracleTag) and tag.scope == scope
    )
