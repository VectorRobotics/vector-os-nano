# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""evidence_classifier — separate GROUNDED verify predicates from RAN (no real,
non-tautological world evidence).

A verify string is evidence only when an ORACLE's own value decides it and the
author cannot make it true by choice of string. *oracle_names* are the evidence
oracles of the live namespace (``trace_store.verify_oracle_names``: ``@oracle``-
tagged callables of the active world's scope); any other name is not an oracle.

The grammar is an ALLOWLIST — anything not listed is RAN:

- a bare PREDICATE-oracle call with literal arguments (``at_position(2.0, 0.0)``,
  ``holding_object('banana')``, ``arm_at_home()``);
- ``<predicate>(...) == True``;
- ``<term> <op> <literal>`` (either order, one comparison) where *term* is a STATE
  oracle call, an index of one (``get_position()[0]``) or ``len()`` of one, and op is
  ``==`` or an ordering against a number — except the vacuous lower bounds
  ``>= c`` (c <= 0) / ``> c`` (c < 0) that every count satisfies, any compare a
  COUNT (``len()`` or a count oracle) satisfies at its fail-safe 0
  (``len(detect_objects()) < 1000``, ``placed_count() <= 1e9``), and any bound that
  holds over a ranged oracle's whole domain (``get_heading() < 100``);
- ``'<non-empty literal>' in <state oracle call>``;
- ``and`` / ``or`` of grounded parts — except a disjunction of ``facing`` arcs whose
  union is the whole circle (true at every heading).

Rejected by construction: value-absorbing wrappers (``|``, ``*0``, ``max``/``min``/
``abs``/``bool``…), ``is`` / ``is not``, ``!=``, ``not in``, ``not``, a bool predicate
ordered against a constant, ``'' in ...``, oracle-vs-oracle, chained compares,
oracle arguments that are not literals (``at_position(get_position()[0], …)``), an
empty container as a literal (``detect_objects() == []`` — true on the fail-safe
value), and a literal tolerance above its cap (``at_position(0, 0, 1e6)``, FG-3).

SCOPE (honest — do not overclaim): this is STRUCTURAL. It does not prove the literal
is the TASK's goal (coord_goal / object_goal turn gates do that for coordinate and
grasp goals), nor that the actor caused the state (actor_causation), nor catch a
shape-trivial compare (``len(get_position()) == 3``) — that needs oracle return-type
metadata.
"""
from __future__ import annotations

import ast
import math
from typing import Literal

from vector_os_nano.vcli.cognitive.coord_goal import _MAX_ARRIVAL_TOL_M, _MAX_FACING_TOL_RAD

Verdict = Literal["GROUNDED", "RAN"]

# Sentinels the decomposer / robot motor steps emit when there is no predicate.
_NO_EVIDENCE: frozenset[str] = frozenset({"", "True"})

# Oracle names returning a goal-conditioned bool — a BARE call is real evidence.
_PREDICATE_ORACLES: frozenset[str] = frozenset(
    {
        "at_position",
        "facing",
        "visited",
        "holding_object",
        "arm_at_home",
        "file_exists",
        # path_contains(path, substr) is a goal-conditioned bool (the author
        # supplies the substring goal; True only if the file actually contains it).
        "path_contains",
        "released",
    }
)
# Every bool oracle: in a compare only ``== True`` grounds (an ordering or ``== False``
# on a bool is a tautology or a negation), and it is never a state term. g1's
# ``detection_matches_gt`` grounds only as ``== True`` (R7); its bare call stays RAN.
# Every ``@oracle``-tagged ``-> bool`` callable MUST be here, else it gets the looser
# state rules; test_spine_moat_corpus scans the tags and fails on drift.
# ponytail: these lists copy the @oracle(domain=...) tags because the classifier gets
# only names, not the live OracleTags; test_spine_moat_corpus pins tag == list. Read
# the tags directly once classify_verify_expr takes the oracle dict.
_BOOL_ORACLES: frozenset[str] = _PREDICATE_ORACLES | {"detection_matches_gt"}
# Return domains of the non-bool oracles (``@oracle(domain=...)``; test_spine_moat_corpus
# pins tag == table). A count is >= 0 and 0 is its fail-safe value; a range is closed.
_COUNT_ORACLES: frozenset[str] = frozenset({"placed_count", "grep_count"})
_RANGE_ORACLES: dict[str, tuple[float, float]] = {"get_heading": (-math.pi, math.pi)}

# Tolerance-taking predicates: positional index of ``tol`` and its cap (the cap the
# oracle clamps to = its ``@oracle(max_tol=...)``). A literal tol above the cap, or a
# non-numeric one, is "within 1e6 m" — vacuous, never evidence (FG-3).
_TOL_ARG: dict[str, tuple[int, float]] = {
    "at_position": (2, _MAX_ARRIVAL_TOL_M),
    "facing": (1, _MAX_FACING_TOL_RAD),
}

_ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
# Mirror an ordering when the literal is on the LEFT (``0 < len(x)`` == ``len(x) > 0``).
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}


def _literal(node: ast.AST) -> bool:
    """A literal the author wrote: a non-None constant, a signed number, or a
    NON-EMPTY tuple/list of literals (``placed_count((0, 0, 1, 1))``). An empty
    container is no goal: ``detect_objects() == []`` holds on the fail-safe value."""
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _number(node) is not None
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(_literal(e) for e in node.elts)
    return False


def _number(node: ast.AST) -> float | None:
    """The value of a numeric literal (never a bool), else None."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _oracle_call(node: ast.AST, oracle_names: frozenset[str]) -> str | None:
    """The oracle's name if *node* is a DIRECT oracle call with literal args."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in oracle_names
    ):
        return None
    args = [*node.args, *(k.value for k in node.keywords)]
    if any(isinstance(a, ast.Starred) for a in node.args) or any(
        k.arg is None for k in node.keywords
    ):
        return None
    if not all(_literal(a) for a in args):
        return None
    spec = _TOL_ARG.get(node.func.id)
    if spec is not None:
        i, cap = spec
        tols = [k.value for k in node.keywords if k.arg == "tol"] + node.args[i : i + 1]
        if any((v := _number(t)) is None or v > cap for t in tols):
            return None
    return node.func.id


def _predicate_call(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    return _oracle_call(node, oracle_names) in _PREDICATE_ORACLES


def _state_call(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    name = _oracle_call(node, oracle_names)
    return name is not None and name not in _BOOL_ORACLES


def _state_term(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    """A value read straight off a state oracle: the call, a constant index of it,
    or ``len()`` of it — never an arithmetic / reduction that can absorb it."""
    if _state_call(node, oracle_names):
        return True
    if isinstance(node, ast.Subscript):
        return _state_call(node.value, oracle_names) and isinstance(node.slice, ast.Constant)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
        and _state_call(node.args[0], oracle_names)
    )


def _term_domain(node: ast.AST, oracle_names: frozenset[str]) -> object:
    """``"count"`` for ``len(<state>)`` or a count oracle, the ``(lo, hi)`` of a ranged
    oracle, else None (unbounded / unknown)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
        return "count"
    name = _oracle_call(node, oracle_names)
    if name in _COUNT_ORACLES:
        return "count"
    return _RANGE_ORACLES.get(name or "")


def _holds(op: ast.cmpop, value: float, bound: float) -> bool:
    return {ast.Eq: value == bound, ast.Lt: value < bound, ast.LtE: value <= bound,
            ast.Gt: value > bound, ast.GtE: value >= bound}[type(op)]


def _grounded_compare(node: ast.Compare, oracle_names: frozenset[str]) -> bool:
    if len(node.ops) != 1:
        return False  # chained compares
    op, left, right = node.ops[0], node.left, node.comparators[0]
    if isinstance(op, ast.In):
        lit = left
        return (
            _literal(lit)
            and not (isinstance(lit, ast.Constant) and lit.value == "")
            and _state_call(right, oracle_names)
        )
    if not isinstance(op, (ast.Eq, *_ORDERING)):
        return False  # is / is not / != / not in: satisfied by almost anything
    if _literal(left) and not _literal(right):
        left, right = right, left
        if isinstance(op, _ORDERING):
            op = _FLIP[type(op)]()
    if not _literal(right):
        return False  # oracle-vs-oracle, or an absorbing expression on the other side
    if _oracle_call(left, oracle_names) in _BOOL_ORACLES:
        # A bool oracle: only `== True` (the bare call); `== False` is a negation
        # and an ordering on a bool is a tautology for some constant.
        return isinstance(op, ast.Eq) and isinstance(right, ast.Constant) and right.value is True
    if not _state_term(left, oracle_names):
        return False
    bound = _number(right)
    domain = _term_domain(left, oracle_names)
    if domain == "count" and bound is not None and _holds(op, 0.0, bound):
        return False  # the fail-safe 0 satisfies it: an upper bound / `== 0` proves nothing
    if isinstance(domain, tuple) and bound is not None and not isinstance(op, ast.Eq) \
            and _holds(op, domain[0], bound) and _holds(op, domain[1], bound):
        return False  # true over the oracle's whole domain
    if isinstance(op, ast.Eq):
        return True
    if bound is None:
        return False  # an ordering against a non-number
    if isinstance(op, ast.GtE) and bound <= 0 or isinstance(op, ast.Gt) and bound < 0:
        return False  # every count satisfies it
    return True


def _disjuncts(node: ast.AST) -> list[ast.AST]:
    """The leaves of a (possibly parenthesised) ``or`` chain."""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return [d for v in node.values for d in _disjuncts(v)]
    return [node]


def _facing_covers_circle(values: list[ast.AST], oracle_names: frozenset[str]) -> bool:
    """True iff the ``facing(h, tol)`` disjuncts' arcs cover every heading (FG-3:
    ``facing(0) or facing(1.57) or ...``). A missing tol counts as the cap."""
    arcs = []
    for v in values:
        if _oracle_call(v, oracle_names) != "facing":
            continue
        kw = {k.arg: k.value for k in v.keywords}  # type: ignore[attr-defined]
        h_node = v.args[0] if v.args else kw.get("heading")  # type: ignore[attr-defined]
        t_node = v.args[1] if len(v.args) > 1 else kw.get("tol")  # type: ignore[attr-defined]
        h = _number(h_node) if h_node is not None else None
        tol = _number(t_node) if t_node is not None else _MAX_FACING_TOL_RAD
        if h is None or tol is None:
            continue
        lo = (h - tol) % (2 * math.pi)
        arcs += [(lo, lo + 2 * tol)] if lo + 2 * tol <= 2 * math.pi else \
            [(lo, 2 * math.pi), (0.0, lo + 2 * tol - 2 * math.pi)]
    reach = 0.0
    for a, b in sorted(arcs):
        if a > reach:
            return False
        reach = max(reach, b)
    return reach >= 2 * math.pi


def _is_grounded_node(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    """True iff *node*'s truth is decided by an oracle value against an author
    literal, with nothing that can absorb or invert it (see the module grammar)."""
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.Or) and _facing_covers_circle(_disjuncts(node), oracle_names):
            return False
        return bool(node.values) and all(_is_grounded_node(v, oracle_names) for v in node.values)
    if isinstance(node, ast.Compare):
        return _grounded_compare(node, oracle_names)
    return _predicate_call(node, oracle_names)


def classify_verify_expr(expr: str, oracle_names: frozenset[str]) -> Verdict:
    """Classify a verify expression as GROUNDED or RAN (see the module grammar).

    *oracle_names* MUST be the live evidence-oracle names from
    ``trace_store.verify_oracle_names`` — never a hand-authored second copy (rule 3).
    An empty set fails closed (everything RAN).
    """
    s = (expr or "").strip()
    if s in _NO_EVIDENCE:
        return "RAN"
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return "RAN"
    return "GROUNDED" if _is_grounded_node(tree.body, oracle_names) else "RAN"
