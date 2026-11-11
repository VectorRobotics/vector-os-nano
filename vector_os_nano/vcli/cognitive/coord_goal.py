# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Goal-authenticity helpers (STEP 13) — does the verify CONSTANT match the user's
real task goal?

The honest moat (R1 structural + R2b actor-causation) proves a step consumed a real
oracle AND the actor caused the state change — but NOT that the verify constant is the
coordinate the USER asked for. A model could do a real actor-caused walk to the WRONG
place and verify ``at_position(<its own landing>)`` to self-certify success. These PURE
helpers let the grader (``classify_step_evidence``) REJECT such a goal-mismatch for
COORDINATE goals.

Properties (rule 5 / honest-by-construction):
- STRICTER-only: the grader uses these ONLY to downgrade GROUNDED -> RAN, never the
  reverse.
- FAIL-OPEN: a None on either side (no coordinate goal, or a non-``at_position`` /
  non-literal verify) means "no mismatch" — the existing classification stands. So
  dev / grasp / facing / coordinate-less turns are untouched.
- Honest: the target is parsed from ``goal_tree.goal`` (the user's NL command, passed
  verbatim by the producer), NEVER from actor-authored state. The model authors only
  the verify expr, never the goal — so it cannot make the parsed target equal its
  landing.
"""
from __future__ import annotations

import ast
import math
import re
import unicodedata

# First (x, y) numeric pair in parentheses — matches "走到坐标 (11.0,3.0)" and
# "go to (11,3)" (optional sign / decimals / whitespace).
_COORD_RE = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")


# A coordinate verify may pass an explicit arrival tolerance (``at_position(x, y, tol)``).
# An HONEST arrival check uses a small radius (the oracle default 0.5 m, or up to a metre
# or two for a generous "reached the vicinity"); a model trying to self-certify from
# anywhere inflates it (``tol=99`` = "within 99 m" = true across the whole map). This is
# the line between the two: an explicit tol ABOVE it makes the predicate vacuous and the
# verify is not an honest "reached (x, y)" assertion. Generous on purpose (accepts honest
# 0.5–2 m), so it only ever rejects an EGREGIOUS tolerance (stricter-only, rule 5).
_MAX_ARRIVAL_TOL_M: float = 2.0
# The same line for a heading tolerance (``facing(h, tol)``): above 45 deg "facing h"
# no longer names a direction.
_MAX_FACING_TOL_RAD: float = math.radians(45.0)
# The same line for the object-distance radii: "near" an object within 1.5 m, an
# object "at" a point within 0.5 m, "localized" within 0.2 m (the apartment's pick
# table holds bottles 0.22 m apart: a looser match would credit the neighbour).
_MAX_NEAR_M: float = 1.5
_MAX_OBJECT_AT_M: float = 0.5
_MAX_LOCALIZED_M: float = 0.2


def _at_position_tol() -> float:
    """The SAME tolerance the ``at_position`` oracle uses (single-sourced, fail-safe).

    Re-read, never re-authored, so the check can never disagree with the oracle.
    """
    try:
        from vector_os_nano.vcli.worlds.sim_oracles import _AT_POSITION_TOL_M

        return float(_AT_POSITION_TOL_M)
    except Exception:  # noqa: BLE001
        return 0.5


def parse_goal_coord(goal_text: str | None) -> tuple[float, float] | None:
    """The user's commanded (x, y) target parsed from the NL goal, or None.

    None when there is no coordinate, or MORE THAN ONE DISTINCT coordinate (ambiguous)
    — both fail OPEN. Derived ONLY from the user's words.
    """
    if not goal_text:
        return None
    # NFKC: an IME's full-width （11，3） is the same coordinate as (11,3).
    matches = _COORD_RE.findall(unicodedata.normalize("NFKC", goal_text))
    if not matches:
        return None
    coords = {(float(x), float(y)) for x, y in matches}
    if len(coords) != 1:
        return None  # ambiguous -> fail open
    return next(iter(coords))


def _const_number(node: ast.AST) -> float | None:
    """A numeric literal (incl. a unary +/- on one), else None (never a bool)."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_number(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _is_at_position_call(node: ast.AST) -> bool:
    """True iff *node* is an ``at_position(...)`` Call (by name)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "at_position"
    )


def _at_position_call_xy(node: ast.Call) -> tuple[float, float] | None:
    """The literal (x, y) of ONE ``at_position(...)`` Call node, or None.

    Single-sourced extraction (positional or ``x=``/``y=`` kwarg form) with the STEP-15
    tolerance-bound: a 3rd positional / ``tol=`` literal above ``_MAX_ARRIVAL_TOL_M`` (or
    a non-literal coord / tol) makes the predicate vacuous -> None (fail CLOSED).
    """
    x = y = None
    if len(node.args) >= 2:
        x = _const_number(node.args[0])
        y = _const_number(node.args[1])
    else:
        kw = {k.arg: k.value for k in node.keywords if k.arg in ("x", "y")}
        if "x" in kw and "y" in kw:
            x = _const_number(kw["x"])
            y = _const_number(kw["y"])
    if x is None or y is None:
        return None
    tol_node: ast.AST | None = None
    if len(node.args) >= 3:
        tol_node = node.args[2]
    else:
        for k in node.keywords:
            if k.arg == "tol":
                tol_node = k.value
                break
    if tol_node is not None:
        tol_val = _const_number(tol_node)
        if tol_val is None or tol_val > _MAX_ARRIVAL_TOL_M:
            return None
    return (x, y)


def _necessary_at_position_consts(node: ast.AST) -> list[tuple[float, float]]:
    """The (x, y) consts of every ``at_position`` call that is a NECESSARY CONJUNCT of
    *node* — reachable from the root through ``BoolOp(And)`` nodes ONLY.

    Such a call's falsity forces the WHOLE verify False (an ``and``-chain is True only if
    every conjunct is True), so it genuinely GATES the verdict on reaching the
    coordinate. An ``at_position`` buried under an ``or`` (a satisfiable sibling makes it
    non-necessary), a ``not`` (true when NOT there), an arithmetic ``BinOp`` /
    ``Compare`` threshold (``at_position(g)+at_position(w)>=1``), or an ``IfExp`` is NOT a
    necessary conjunct and is excluded. STEP-16: this is the line between a constant that
    is PRESENT and one that is NECESSARY — closing the boolean-necessity false-green
    family (4th moat review). Purely structural; STRICTER-only.
    """
    if _is_at_position_call(node):
        xy = _at_position_call_xy(node)
        return [xy] if xy is not None else []
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out: list[tuple[float, float]] = []
        for v in node.values:
            out.extend(_necessary_at_position_consts(v))
        return out
    return []


def _necessary_consts(expr: str | None) -> list[tuple[float, float]]:
    """Parse *expr* and return its necessary-conjunct at_position consts ([] on error)."""
    if not expr:
        return []
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return []
    return _necessary_at_position_consts(tree.body)


def _expr_has_at_position(expr: str | None) -> bool:
    """True iff *expr* names ``at_position(...)`` anywhere (literal or not)."""
    if not expr:
        return False
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return False
    return any(_is_at_position_call(n) for n in ast.walk(tree))


def at_position_is_necessary(
    expr: str | None, goal_xy: tuple[float, float], tol: float | None = None
) -> bool:
    """True iff *expr* has a NECESSARY ``at_position`` conjunct within *tol* of *goal_xy*.

    The matching ``at_position`` must GATE the verify (a top-level or nested ``and``
    conjunct), so the verify cannot evaluate True with the robot away from the commanded
    coordinate. Fail-CLOSED (False) on syntax error / no necessary match. STEP-16: this
    replaces a mere-PRESENCE check that an ``or``/``not``/arithmetic decoy could
    satisfy with a goal-matching-but-inert constant.
    """
    t = _at_position_tol() if tol is None else float(tol)
    return any(math.dist(goal_xy, xy) <= t for xy in _necessary_consts(expr))


def has_necessary_at_position(expr: str | None) -> bool:
    """True iff *expr* has ANY necessary (``and``-conjunct) literal ``at_position`` —
    regardless of which coordinate.

    Used by the turn gate's PARSE-EVASION branch (paren-less / multi-coordinate goals
    whose single target ``parse_goal_coord`` cannot pin): it requires an HONEST gating
    ``at_position`` rather than a mere presence, so the boolean-necessity decoys are
    rejected there too (the wrong-VALUE multi-coord case is the separate parse-asymmetry
    residual, tracked for a later round).
    """
    return bool(_necessary_consts(expr))


def at_position_const_matches(
    expr: str | None, goal_xy: tuple[float, float], tol: float | None = None
) -> bool:
    """True iff *expr* NECESSARILY asserts the commanded coordinate: a NECESSARY
    ``at_position`` conjunct within *tol* of *goal_xy* (default = the oracle tolerance).

    The POSITIVE dual of ``coord_goal_mismatch`` used by the turn-level coordinate gate
    (``evidence_passed``): it confirms a verify actually GATES on reaching the commanded
    coordinate. STEP-16: necessity (not mere presence) — a turn whose only at_position is
    a non-necessary decoy (``... or visited(elsewhere)``) no longer satisfies the gate.
    Fail-CLOSED for any non-``at_position`` / non-literal / tol-inflated / decoy verify.
    """
    return at_position_is_necessary(expr, goal_xy, tol)


# A looser coordinate-INTENT detector (paren OR paren-less ``x,y``): a numeric pair
# separated by a comma. Used ONLY to decide whether a coordinate requirement applies to
# a turn whose exact target ``parse_goal_coord`` could not pin (paren-less "导航到 11,3"
# or >=1 distinct coords). It NEVER extracts a target — ``parse_goal_coord`` stays the
# single authority for the actual (x, y).
_COORD_INTENT_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?"
)


def goal_has_coordinate_intent(goal_text: str | None) -> bool:
    """True iff the NL goal contains a numeric ``x,y`` pair (parenthesised or not).

    A turn-gate helper for the parse-evasion case: when ``parse_goal_coord`` fails open
    (paren-less or multi-coordinate phrasing) the gate still requires SOME honest
    at_position evidence rather than silently falling back to the weaker all-GROUNDED
    rule. A goal with no numeric pair (room name, "explore", "turn") returns False ->
    fails open (unchanged).
    """
    if not goal_text:
        return False
    return bool(_COORD_INTENT_RE.search(unicodedata.normalize("NFKC", goal_text)))


def coord_goal_mismatch(
    goal_text: str | None, expr: str | None, tol: float | None = None
) -> bool:
    """True iff *expr* contains an ``at_position`` for a coordinate goal but does NOT
    NECESSARILY assert the commanded coordinate within *tol* (default = the oracle tol).

    FAIL-OPEN (returns False) when there is no coordinate goal, or the verify carries no
    ``at_position`` at all (a non-``at_position`` verify is handled by the turn-level
    gate, not here). Otherwise it is a mismatch UNLESS the verify has a NECESSARY
    at_position conjunct within tol of the goal — so it rejects both the WRONG-CONSTANT
    case (``at_position(<own landing>)``, STEP-13) AND the STEP-16 boolean-necessity
    decoys (``at_position(goal) or <always-true>``, ``not at_position(goal)``,
    ``at_position(goal)+at_position(wrong)>=1``) whose matching constant is PRESENT but
    not NECESSARY. Stricter-only (rule 5).
    """
    goal_xy = parse_goal_coord(goal_text)
    if goal_xy is None:
        return False
    # Only police a verify that actually names at_position; a non-at_position verify
    # (facing / state compare) fails open here and is handled by the turn-level gate.
    if not _expr_has_at_position(expr):
        return False
    return not at_position_is_necessary(expr, goal_xy, tol)
