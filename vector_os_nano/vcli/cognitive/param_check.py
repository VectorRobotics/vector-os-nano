# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deterministic param-completeness check against the skill registry schema.

Batch 2, campaign #4: the decomposer validated strategies and verify
expressions but never the PARAMS — a step missing a required param sailed
into execution, failed bad_params, and burned a whole replan cycle. This
module compares each step's ``strategy_params`` against the skill's OWN
declared ``parameters`` (rule 3 — the registry is the single source of the
action space, including its schemas): ``required: True`` keys must be bound
to a non-null value, and a value bound to an ``enum`` key must be in the
enum. Pure schema comparison — no LLM, no defaults injected, no repair here
(the decomposer does ONE bounded re-ask with these issues; what remains
broken stays broken and fails loud at the skill's bad_params gate).
"""
from __future__ import annotations

from typing import Any

# Strategies the kernel dispatches itself — no skill schema to check.
_KERNEL_STRATEGIES = frozenset({"", "answer"})


def _skill_for(strategy: str, registry: Any) -> Any:
    """Resolve a plan strategy to its registry skill (suffix-normalized)."""
    name = strategy[:-6] if strategy.endswith("_skill") else strategy
    try:
        return registry.get(name)
    except Exception:  # noqa: BLE001 — fail-soft: no schema, no check
        return None


def _bound(params: dict, key: str) -> bool:
    """A param is bound only by a real value — null/"" is MISSING (Case 11)."""
    v = params.get(key)
    return v is not None and v != ""


def collect_param_issues(tree: Any, registry: Any) -> "list[dict]":
    """Issues for every step whose params violate its skill's schema.

    Returns a list of ``{"step", "strategy", "missing": [name...],
    "illegal": [{"param", "value", "legal"}...], "schema": {param: info}}``
    — everything the bounded re-ask needs to TEACH the LLM the contract it
    broke. Unknown strategies and kernel strategies are skipped (the
    strategy validator already handles those loudly).
    """
    issues: list[dict] = []
    if registry is None:
        return issues
    for sg in getattr(tree, "sub_goals", ()) or ():
        strategy = getattr(sg, "strategy", "") or ""
        if strategy in _KERNEL_STRATEGIES:
            continue
        skill = _skill_for(strategy, registry)
        spec = getattr(skill, "parameters", None)
        if not isinstance(spec, dict) or not spec:
            continue
        params = getattr(sg, "strategy_params", {}) or {}
        missing = [
            k for k, info in spec.items()
            if isinstance(info, dict) and info.get("required") is True
            and not _bound(params, k)
        ]
        illegal = [
            {"param": k, "value": params[k], "legal": list(info["enum"])}
            for k, info in spec.items()
            if isinstance(info, dict) and isinstance(info.get("enum"), (list, tuple))
            and _bound(params, k) and params[k] not in info["enum"]
        ]
        if missing or illegal:
            issues.append({
                "step": sg.name,
                "strategy": strategy,
                "missing": missing,
                "illegal": illegal,
                "schema": {
                    k: {f: v for f, v in info.items() if f != "description"}
                    for k, info in spec.items() if isinstance(info, dict)
                },
            })
    return issues


def render_issue_block(issues: "list[dict]") -> str:
    """Human/LLM-readable correction request for the bounded re-ask."""
    lines = [
        "Your plan bound step parameters incompletely or illegally.",
        "Respond with ONLY a JSON object mapping each listed step name to its",
        "FULL corrected params dict. Do not add, remove, or rename steps.",
        "",
    ]
    for issue in issues:
        parts = []
        if issue["missing"]:
            parts.append(f"missing required: {issue['missing']}")
        for bad in issue["illegal"]:
            parts.append(
                f"illegal {bad['param']}={bad['value']!r} "
                f"(legal: {bad['legal']})"
            )
        lines.append(
            f"- step {issue['step']!r} (strategy {issue['strategy']}): "
            + "; ".join(parts)
        )
        lines.append(f"  schema: {issue['schema']}")
    return "\n".join(lines)


def merge_corrections(tree: Any, issues: "list[dict]", corrections: Any) -> Any:
    """Apply re-ask corrections to the NAMED issue steps only.

    Structure is immutable here: only ``strategy_params`` of steps that were
    listed in *issues* may change; null/"" values are stripped (the same
    parse-seam rule); anything else in the correction payload is ignored.
    Returns a new tree (or the original when nothing applied).
    """
    import dataclasses

    if not isinstance(corrections, dict):
        return tree
    issue_steps = {i["step"] for i in issues}
    changed = False
    merged = []
    for sg in tree.sub_goals:
        fix = corrections.get(sg.name)
        if sg.name in issue_steps and isinstance(fix, dict):
            cleaned = {k: v for k, v in fix.items()
                       if v is not None and v != ""}
            if cleaned != sg.strategy_params:
                sg = dataclasses.replace(sg, strategy_params=cleaned)
                changed = True
        merged.append(sg)
    if not changed:
        return tree
    return dataclasses.replace(tree, sub_goals=tuple(merged))
