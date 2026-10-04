# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Goal-authenticity helpers for OBJECT (grasp/pick) goals (D17) — the object-goal
analogue of ``coord_goal`` (D15/D16).

The honest moat (R1 structural + R2b actor-causation) proves a step consumed a real
oracle AND, for a ROBOT predicate, that the actor caused the change. But a grasp goal
whose verify is a NON-robot, author-writable predicate (``file_exists('grabbed.txt')``,
``len(describe_scene()) > 0``, ``path_contains(...)``) slips through: ``file_exists`` is
a PREDICATE oracle so it classifies GROUNDED structurally, and being a non-robot
predicate it is never actor-causation-graded. That is exactly the fakeable-grasp
false-green this module closes — deepseek satisfied "抓前面的东西" by writing a marker
file and verifying the file, never the gripper (D17, see docs/tricky-bugs.md).

These PURE helpers let the turn-level gate (``trace_store.evidence_passed``) REJECT such
a grasp goal: a goal commanding a physical grasp/pick must be GROUNDED via a GT
MANIPULATION oracle (``holding_object`` / ``placed_count``) that the actor cannot author
— never a self-authored file/scene/marker predicate.

Properties (rule 5 / honest-by-construction — IDENTICAL discipline to coord_goal):
- STRICTER-only: the gate uses these ONLY to turn a FALSELY-GREEN grasp turn RED, never
  the reverse. It can only REJECT (the turn already classified all-GROUNDED) and can
  never fake-pass (the satisfying step must ITSELF be GROUNDED).
- FAIL-OPEN: a goal with no grasp/pick intent returns False -> the gate does not apply,
  the existing classification stands. So nav / dev / place-only / detect goals are
  untouched.
- Honest: intent is parsed from the user's NL command (``goal_tree.goal``, passed verbatim
  by the producer), NEVER from actor-authored state. The model authors only the verify
  expr, never the goal — so it cannot make the parsed intent disappear.
- SCOPED by the caller to a world whose live oracle set contains ``holding_object`` (an
  arm world), so go2-coordinate / dev / g1-detect worlds never see this gate.
"""
from __future__ import annotations

import ast
import re

# Grasp / pick INTENT: the user commanded a physical grasp or pick-up of an object.
# Mirrors ``coord_goal.goal_has_coordinate_intent`` — this detects INTENT only, it
# NEVER extracts the target object (the oracle owns the canonical scene name). Covers
# the Chinese grasp verbs (抓/夹/拿/握/捡) and the English ones (pick up, grasp, grab).
# Deliberately does NOT match pure PLACE verbs (放/place/put/drop) — a place-only goal
# carries no grasp intent and fails open, so a release-completion verify (e.g.
# ``not holding_object()``) is never policed here.
_OBJECT_INTENT_RE = re.compile(
    r"抓|夹住|夹起|夹取|拿起|拿住|握住|捡起|pick\s*up|\bpick\b|grasp|grab|gripp",
    re.IGNORECASE,
)

# GROUND-TRUTH POSSESSION oracles — the predicates whose truth PROVES the gripper holds
# the object and which the ACTOR CANNOT AUTHOR: holding_object (weld + finger contacts
# + lifted), lifted (held clear of its support), kept_holding (held, never dropped this
# turn). A placement (object_on / object_at / released) never proves a grasp: an object
# can rest on a table without ever being picked. Anything else (file_exists /
# path_contains / a scene reader) is the means' own author-writable output.
_MANIP_ORACLES: frozenset[str] = frozenset({"holding_object", "lifted", "kept_holding"})


def goal_has_object_intent(goal_text: str | None) -> bool:
    """True iff the NL goal commands a physical GRASP / PICK of an object.

    A turn-gate helper: when True (and the world exposes ``holding_object``), the gate
    requires the turn to actually prove possession via a GT manipulation oracle rather
    than a self-authored file/scene predicate. A goal with no grasp/pick verb (a nav,
    a place-only "放到盒子里", a "detect", a dev file task) returns False -> fails open
    (the existing classification stands; stricter-only).
    """
    if not goal_text:
        return False
    return bool(_OBJECT_INTENT_RE.search(goal_text))


def _is_manip_call(node: ast.AST) -> bool:
    """True iff *node* is a call to a GT manipulation oracle (by name)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _MANIP_ORACLES
    )


def _necessary_manip(node: ast.AST) -> bool:
    """True iff a GT manipulation-oracle call GATES *node*'s truth value.

    Mirrors ``coord_goal._necessary_at_position_consts`` — a manip call is "necessary"
    when its result reaches the verdict, so the verify cannot be True without the
    gripper/scene actually being in the proven state. Reachable through:
      - the manip call itself;
      - an ``and`` chain (every conjunct must hold, so a conjunct gates the whole);
      - ``not X`` (a place-release ``not holding_object()`` — the call still gates it);
      - a Compare (``placed_count() == 2`` — the call's value gates the comparison).
    An ``or`` (a satisfiable sibling makes the manip call non-necessary) is EXCLUDED —
    so ``holding_object('x') or True`` is NOT a necessary manip (it also classifies RAN
    structurally upstream, but the necessity check is independent). Purely structural;
    STRICTER-only.
    """
    if _is_manip_call(node):
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return any(_necessary_manip(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _necessary_manip(node.operand)
    if isinstance(node, ast.Compare):
        return _necessary_manip(node.left) or any(
            _necessary_manip(c) for c in node.comparators
        )
    return False


def has_necessary_manip_oracle(expr: str | None) -> bool:
    """True iff *expr* has a NECESSARY GT-manipulation-oracle call (gates the verify).

    The positive test the grasp turn-gate uses: a verify that NECESSARILY asserts
    ``holding_object`` / ``placed_count`` proves physical possession/placement. Fail-
    CLOSED (False) on syntax error, no manip oracle, or a non-necessary (``or``-decoy)
    manip call. So a fabricated grasp verified by ``file_exists`` / ``describe_scene`` /
    ``path_contains`` (none of which name a manip oracle) returns False -> the gate
    rejects the turn (it cannot prove a real grasp).
    """
    if not expr:
        return False
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return False
    return _necessary_manip(tree.body)
