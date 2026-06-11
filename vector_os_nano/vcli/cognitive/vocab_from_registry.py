# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Build a DecomposeVocab from the live skill registry.

Single-sources the GoalDecomposer's decompose vocabulary from one place: the
skill registry schemas plus the engine's verify namespace. The prompt the LLM
reads, the validator's strategy allowlist, and the params-help block are all
derived here, so they can never drift apart (the split-brain Stage 2 fixes:
the prompt teaching one set of strategies while the allowlist is built from
another).

This mechanism is world-agnostic. The arm is the touchstone — an arm with no
mobile base must NOT be taught the base primitives (walk_forward/turn/scan_360),
while a Go2 (which has a base) must. Callers pass ``has_base`` to gate that.

Nothing here is robot-specific: there is no hardcoded GO2 strategy list, no
"去厨房" example, no navigate/look/explore prompt text. Every strategy and
every example is generated from real registry skills and real verify
signatures.
"""

from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.worlds.base import DecomposeVocab

# Base locomotion primitives a mobile robot exposes in addition to its skills.
# Included in the vocabulary only when the connected agent has a mobile base
# (``has_base=True``); an arm-only agent never sees them.
_BASE_PRIMITIVE_DESCRIPTIONS: dict[str, str] = {
    "walk_forward": "Walk straight forward a set distance",
    "turn": "Rotate in place by given angle",
    "scan_360": "Rotate 360 degrees while recording observations",
}

# Generic, world-agnostic planner guidance appended to every intro. Phrased
# without any embodiment-specific noun (no arm/go2 vocabulary) so it applies to
# every world's skills uniformly: (i) bind a known target into the chosen
# strategy's object param, (ii) prefer each strategy's declared success
# predicate for the step's verify.
_TARGET_BINDING_GUIDANCE: str = (
    "When a step acts on a SPECIFIC object/target named in the task, copy that "
    "target into the chosen strategy's object/object_label/query/target "
    "parameter — never leave a known target blank. "
    "When the world context carries an 'Objects (live):' list, RESOLVE the "
    "user's reference (any language, attributes like color/size included) to "
    "the EXACT listed name whose name/attributes match, and bind THAT name — "
    "the scene knows objects by those names, not by the user's wording. If "
    "nothing listed matches, bind the user's wording as-is; the step will "
    "fail loudly and you can re-bind on replan from the fresh list. "
    "For an instruction that targets a NAMED object's location, bind the object's LABEL into params when the strategy supports it (the skill resolves live coordinates and fails loudly if the object is unknown) — NEVER invent x/y for an object. "
    "Use each strategy's 'suggested verify' predicate EXACTLY as written for that "
    "step's verify expression: put the target ONLY in strategy_params, never as an "
    "argument inside the verify expression. The verifier checks deterministic "
    "ground-truth state (or the step's own recorded output via step_output()), "
    "not your target string."
)

_DEFAULT_PLANNER_INTRO: str = (
    "You are a robot task planner. Decompose the user's task into verifiable "
    "sub-goals, each with a deterministic verify predicate. Choose a strategy "
    "for every step that must act; leave strategy empty for pure checks. "
    + _TARGET_BINDING_GUIDANCE
)

_SKILL_SUFFIX = "_skill"

# Param names that name an object/target the planner should bind a task entity
# to. Used to (a) surface the right param to fill and (b) show a BOUND example
# so the planner learns the shape of binding. PRIORITY-ORDERED: the example binds
# the FIRST present (highest-priority) one. World-agnostic — these are conventional
# natural-language target param NAMES, not domain nouns. ``object_id`` is
# intentionally EXCLUDED: it is a world-model HANDLE, not a natural-language target
# the planner should bind a task noun into.
_OBJECT_PARAM_NAMES: tuple[str, ...] = (
    "object_label", "object", "query", "target", "label", "item",
)
# Neutral placeholder when a param's description carries no usable sample noun.
_NEUTRAL_TARGET_VALUE: str = "target"


def _strategy_name(skill_name: str) -> str:
    """Return the decompose strategy name for a skill (``<name>_skill``)."""
    return f"{skill_name}{_SKILL_SUFFIX}"


def _verify_hint(schema: dict[str, Any]) -> str:
    """Return a skill's declared success predicate, or the safe ``True`` literal.

    Single-sourced from the skill's ``verify_hint`` (surfaced by
    ``Skill.to_schemas``); a skill that declares none gets the always-safe
    truthy literal so the planner always has a concrete suggestion.
    """
    hint = str(schema.get("verify_hint", "") or "").strip()
    return hint or "True"


def _format_params_block(schema: dict[str, Any]) -> str:
    """Render one skill's parameters as a readable strategy-params help line.

    Mirrors the spirit of dev.py's ``_DEV_STRATEGY_PARAMS_HELP``: skill name
    then each param's name/type and whether it is required. Skills with no
    parameters render an explicit ``{}`` so the LLM knows none are needed. A
    final ``suggested verify:`` line surfaces the skill's declared success
    predicate so the planner can prefer it for that step's verify expression.
    """
    name = str(schema.get("name", ""))
    strat = _strategy_name(name)
    hint_line = f"      suggested verify: {_verify_hint(schema)}"
    params = schema.get("parameters") or {}
    if not isinstance(params, dict) or not params:
        return f"  - {strat}: {{}}\n{hint_line}"

    lines = [f"  - {strat}:"]
    for pname, spec in params.items():
        if isinstance(spec, dict):
            ptype = str(spec.get("type", "any"))
            required = bool(spec.get("required", False))
        else:
            ptype = "any"
            required = False
        flag = "required" if required else "optional"
        lines.append(f'      "{pname}": <{ptype}, {flag}>')
    lines.append(hint_line)
    return "\n".join(lines)


def _build_examples(
    schemas: list[dict[str, Any]],
    verify_signatures: dict[str, str],
) -> str:
    """Build a generic few-shot example from REAL registry skills.

    Picks the first one or two skills and shows a minimal valid JSON GoalTree
    using their ``<name>_skill`` strategies. The verify expression uses an
    actual verify-function name from ``verify_signatures`` when one exists,
    otherwise the always-safe ``True`` literal. No GO2 hardcoding.
    """
    if not schemas:
        return ""

    fallback_verify_fn = _pick_verify_fn(verify_signatures)
    chosen = schemas[:2]
    sub_goals: list[str] = []
    prev_name: str | None = None
    for idx, schema in enumerate(chosen):
        skill_name = str(schema.get("name", f"skill_{idx}"))
        strat = _strategy_name(skill_name)
        step_name = f"step_{idx + 1}_{skill_name}"
        # Prefer the skill's declared success predicate so the example teaches the
        # single-sourced verify; fall back to a real verify-fn name only when the
        # skill declares no hint.
        verify_expr = _example_verify_expr(schema, fallback_verify_fn)
        depends = f'["{prev_name}"]' if prev_name else "[]"
        params = _example_params(schema)
        sub_goals.append(
            "    {\n"
            f'      "name": "{step_name}",\n'
            f'      "description": "{_first_sentence(schema.get("description", skill_name))}",\n'
            f'      "verify": "{verify_expr}",\n'
            f'      "strategy": "{strat}",\n'
            '      "timeout_sec": 30,\n'
            f'      "depends_on": {depends},\n'
            f'      "strategy_params": {params},\n'
            '      "fail_action": ""\n'
            "    }"
        )
        prev_name = step_name

    goal = "run " + " then ".join(str(s.get("name", "")) for s in chosen)
    body = ",\n".join(sub_goals)
    return (
        f'Task: "{goal}"\n'
        "Response:\n"
        "{\n"
        f'  "goal": "{goal}",\n'
        '  "sub_goals": [\n'
        f"{body}\n"
        "  ],\n"
        '  "context_snapshot": ""\n'
        "}"
    )


def _example_verify_expr(
    schema: dict[str, Any], fallback_verify_fn: str | None
) -> str:
    """Return the verify expression to show for *schema*'s example step.

    Prefers the skill's declared ``verify_hint`` (single-source) so the example
    teaches the same predicate the params-help suggests; otherwise falls back to
    a real verify-function call, then to the safe ``True`` literal.
    """
    hint = str(schema.get("verify_hint", "") or "").strip()
    if hint:
        return hint
    if fallback_verify_fn:
        return f"{fallback_verify_fn}()"
    return "True"


def _is_object_param(pname: str, spec: Any) -> bool:
    """True if *pname*/*spec* is a natural-language target param a task binds to.

    World-agnostic, matched by the explicit conventional NAME set AND a string
    type. The type guard keeps numeric/coordinate/ID params (e.g. a float ``x``
    whose description happens to say "Target X in metres") from being bound a
    string. Matching is by name only — a description mention ("target"/"object"/
    "label") was too loose: it wrongly matched float coordinates and free-form
    ``question`` params.
    """
    if pname not in _OBJECT_PARAM_NAMES:
        return False
    if isinstance(spec, dict):
        ptype = str(spec.get("type", "")).strip().lower()
        # Only a string-typed param holds a natural-language target ("" = type
        # unstated, allowed since the name is already a strong signal).
        return ptype in ("", "string", "str")
    return True


def _sample_target_value(spec: Any) -> str:
    """Derive a concrete sample value for an object-ish param from its schema.

    Parses an ``e.g. 'X'`` token out of the param description when present (so
    the sample is the skill's own example, in whatever language it states);
    otherwise uses a neutral noun. Never hardcodes a domain noun here.
    """
    if isinstance(spec, dict):
        desc = str(spec.get("description", ""))
        lower = desc.lower()
        marker = "e.g."
        idx = lower.find(marker)
        if idx != -1:
            after = desc[idx + len(marker):]
            # Take the first single-quoted token after the marker, e.g. 'banana'.
            start = after.find("'")
            if start != -1:
                end = after.find("'", start + 1)
                if end != -1:
                    token = after[start + 1: end].strip()
                    if token:
                        return token
    return _NEUTRAL_TARGET_VALUE


def _example_params(schema: dict[str, Any]) -> str:
    """Return a JSON object literal for a skill's example params.

    Shows required params plus — crucially — an object-ish param BOUND to a
    concrete sample value (even when optional), so the planner learns the SHAPE
    of binding a task target into the strategy. The sample value is derived from
    the schema only (the param description's ``e.g. 'X'`` token, else a neutral
    noun), keeping this world-agnostic.
    """
    params = schema.get("parameters") or {}
    if not isinstance(params, dict):
        return "{}"

    pairs: list[str] = []
    seen: set[str] = set()
    # First: bind the HIGHEST-PRIORITY natural-language target param (if present)
    # to a concrete sample so the example shows the binding SHAPE. Iterate the
    # priority order, NOT declaration order — so e.g. pick binds object_label (the
    # NL label the planner_intro tells the model to fill), never object_id.
    for cand in _OBJECT_PARAM_NAMES:
        spec = params.get(cand)
        if spec is not None and _is_object_param(cand, spec):
            pairs.append(f'"{cand}": "{_sample_target_value(spec)}"')
            seen.add(cand)
            break
    # Then: include any remaining required params (empty string placeholders).
    for pname, spec in params.items():
        if pname in seen:
            continue
        if isinstance(spec, dict) and spec.get("required"):
            pairs.append(f'"{pname}": ""')
            seen.add(pname)

    if not pairs:
        return "{}"
    return "{" + ", ".join(pairs) + "}"


def _pick_verify_fn(verify_signatures: dict[str, str]) -> str | None:
    """Pick one verify-function name to demonstrate in the example, or None."""
    if not verify_signatures:
        return None
    return sorted(verify_signatures.keys())[0]


def _first_sentence(text: Any) -> str:
    """Return a short, JSON-safe first clause of *text* for the example."""
    s = str(text).strip().replace('"', "'").replace("\n", " ")
    for sep in (". ", "; "):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    return s[:80]


def build_decompose_vocab(
    schemas: list[dict],
    verify_signatures: dict[str, str],
    has_base: bool,
    planner_intro: str | None = None,
) -> DecomposeVocab:
    """Build a DecomposeVocab from skill schemas and verify signatures.

    Args:
        schemas: ``skill_registry.to_schemas()`` output — a list of dicts each
            with at least ``name``, ``description`` and ``parameters``.
        verify_signatures: name -> human-readable signature string for the
            callables available in verify expressions (e.g.
            ``{"detect_objects": "detect_objects(query='')"}``). The keys become
            the validator's verify-function allowlist.
        has_base: True if the connected agent has a mobile base. When True, the
            base primitives (walk_forward/turn/scan_360) are added to the
            strategies and their descriptions; when False they are omitted
            entirely (the arm must never be taught base primitives).
        planner_intro: Optional planner-intro override; a neutral robot-task
            default is used when None.

    Returns:
        A DecomposeVocab whose strategies, descriptions, params-help, examples
        and verify allowlist are all derived from the single source above.
    """
    strategy_names = {_strategy_name(str(s.get("name", ""))) for s in schemas}
    strategy_descriptions = {
        _strategy_name(str(s.get("name", ""))): str(s.get("description", ""))
        for s in schemas
    }
    if has_base:
        strategy_names |= set(_BASE_PRIMITIVE_DESCRIPTIONS.keys())
        strategy_descriptions.update(_BASE_PRIMITIVE_DESCRIPTIONS)

    params_help_blocks = [_format_params_block(s) for s in schemas]
    strategy_params_help = "\n".join(params_help_blocks)

    return DecomposeVocab(
        planner_intro=planner_intro or _DEFAULT_PLANNER_INTRO,
        verify_functions=frozenset(verify_signatures.keys()),
        verify_fn_signatures=dict(verify_signatures),
        strategy_descriptions=strategy_descriptions,
        strategies=frozenset(strategy_names),
        strategy_params_help=strategy_params_help,
        examples=_build_examples(schemas, verify_signatures),
        fallback_verify="True",
    )
