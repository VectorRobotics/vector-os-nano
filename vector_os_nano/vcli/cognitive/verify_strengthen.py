# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Named-target verify strengthening (STATUS backlog #2, part (b)).

A plan step whose params bind a NAMED target must verify "holding the
REQUESTED object" — ``holding_object('<name>')`` — never the bare
``holding_object()``, which FALSE-PASSES when the step grabbed the wrong
object. Generic grabs (no target bound) legitimately keep the bare form.

The rewrite is:
  - structural — keyed on param presence, zero language matching (rule 3);
  - stricter-only — it narrows a passing predicate, never widens (rule 5);
  - conservative — anything it cannot strengthen SAFELY passes through
    unchanged (Blackboard ``${...}`` refs are unresolvable inside a verify
    expression and would FALSE-FAIL; ``__`` would be rejected by the AST
    validator and kill the step).

Applied at both plan chokepoints: ``GoalDecomposer._validate_sub_goal``
(every LLM decompose and replan) and the engine's deterministic 1-step
fast path.
"""
from __future__ import annotations

from typing import Any

_BARE_HOLDING = "holding_object()"

# Param names a planner may bind a target to — mirrors the language-neutral
# fallback chain pick.py itself resolves (object_label first, id last).
_TARGET_PARAM_KEYS = ("object_label", "object", "query", "target", "object_id")


def strengthen_target_verify(verify: str, strategy_params: dict[str, Any] | None) -> str:
    """Bind a named target into a bare ``holding_object()`` verify.

    Returns *verify* unchanged unless it is exactly the bare form AND the
    params carry a usable named target.
    """
    if verify is None or verify.strip() != _BARE_HOLDING or not strategy_params:
        return verify
    for key in _TARGET_PARAM_KEYS:
        val = strategy_params.get(key)
        if not isinstance(val, str):
            continue
        name = val.strip()
        if not name:
            continue
        if "${" in name or "__" in name:
            # Unresolvable Blackboard ref (would FALSE-FAIL) or AST-validator
            # poison (would kill the step) — bare form is the lesser evil.
            return verify
        return f"holding_object({name!r})"
    return verify


def backfill_target_params(
    strategy: str, strategy_params: "dict[str, Any]", verify: str
) -> "dict[str, Any]":
    """Copy navigate-style target coordinates from *verify* into missing params.

    The navigate contract binds the SAME coordinates into params and verify
    (anti-false-pass). LLM plans occasionally carry them only in the verify
    (N4 live finding: 'navigate_to requires a label OR numeric x and y' while
    the verify held geodesic_dist(-4.5, -3.2) < 0.5). Deterministic repair of
    the planner's own stated intent — never overwrites existing params, only
    fires for navigate_to* strategies with a coordinate-style verify.
    """
    import re

    if not str(strategy).startswith("navigate_to"):
        return strategy_params
    params = strategy_params if isinstance(strategy_params, dict) else {}
    if params.get("label") or ("x" in params and "y" in params):
        return params
    m = re.search(
        r"(?:geodesic_dist|at_position)\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        verify or "",
    )
    if m:
        out = dict(params)
        out.setdefault("x", float(m.group(1)))
        out.setdefault("y", float(m.group(2)))
        return out
    # Label-style repair (R8, owner GUI-test regression): the plan put the
    # target ONLY in a visited('<label>') verify — same stated-intent family.
    lm = re.search(r"visited\(\s*['\"]([^'\"]+)['\"]\s*\)", verify or "")
    if lm:
        out = dict(params)
        out.setdefault("label", lm.group(1))
        return out
    return params
