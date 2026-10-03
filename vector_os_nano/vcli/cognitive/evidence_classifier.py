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
  ``>= c`` (c <= 0) / ``> c`` (c < 0) that every count satisfies;
- ``'<non-empty literal>' in <state oracle call>``;
- ``and`` / ``or`` of grounded parts.

Rejected by construction: value-absorbing wrappers (``|``, ``*0``, ``max``/``min``/
``abs``/``bool``…), ``is`` / ``is not``, ``!=``, ``not in``, ``not``, a bool predicate
ordered against a constant, ``'' in ...``, oracle-vs-oracle, chained compares, and
oracle arguments that are not literals (``at_position(get_position()[0], …)``).

SCOPE (honest — do not overclaim): this is STRUCTURAL. It does not prove the literal
is the TASK's goal (coord_goal / object_goal turn gates do that for coordinate and
grasp goals), nor that the actor caused the state (actor_causation), nor catch a
shape-trivial compare (``len(get_position()) == 3``) — that needs oracle return-type
metadata.
"""
from __future__ import annotations

import ast
from typing import Literal

Verdict = Literal["GROUNDED", "RAN"]

# Sentinels the decomposer / robot motor steps emit when there is no predicate.
_NO_EVIDENCE: frozenset[str] = frozenset({"", "True"})

# Oracle names returning a goal-conditioned bool — a BARE call is real evidence, and
# only the bare call / ``== True`` grounds (an ordering or ``== False`` on a bool is a
# tautology or a negation). Every ``@oracle``-tagged ``-> bool`` callable MUST be here,
# else it falls under the looser state rules; test_spine_moat_corpus scans the tags and
# fails on drift. ponytail: a kernel list until the @oracle tag grows a return kind.
_PREDICATE_ORACLES: frozenset[str] = frozenset(
    {
        "at_position",
        "facing",
        "visited",
        "holding_object",
        "arm_at_home",
        "detection_matches_gt",
        "file_exists",
        # path_contains(path, substr) is a goal-conditioned bool (the author
        # supplies the substring goal; True only if the file actually contains it).
        "path_contains",
    }
)

_ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
# Mirror an ordering when the literal is on the LEFT (``0 < len(x)`` == ``len(x) > 0``).
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}


def _literal(node: ast.AST) -> bool:
    """A literal the author wrote: a non-None constant, a signed number, or a
    tuple/list of literals (``placed_count((0, 0, 1, 1))``)."""
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _number(node) is not None
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal(e) for e in node.elts)
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
    return node.func.id if all(_literal(a) for a in args) else None


def _predicate_call(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    return _oracle_call(node, oracle_names) in _PREDICATE_ORACLES


def _state_call(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    name = _oracle_call(node, oracle_names)
    return name is not None and name not in _PREDICATE_ORACLES


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
    if _predicate_call(left, oracle_names):
        # A bool predicate: only `== True` (the bare call); `== False` is a negation
        # and an ordering on a bool is a tautology for some constant.
        return isinstance(op, ast.Eq) and isinstance(right, ast.Constant) and right.value is True
    if not _state_term(left, oracle_names):
        return False
    if isinstance(op, ast.Eq):
        return True
    bound = _number(right)
    if bound is None:
        return False  # an ordering against a non-number
    if isinstance(op, ast.GtE) and bound <= 0 or isinstance(op, ast.Gt) and bound < 0:
        return False  # every count satisfies it
    return True


def _is_grounded_node(node: ast.AST, oracle_names: frozenset[str]) -> bool:
    """True iff *node*'s truth is decided by an oracle value against an author
    literal, with nothing that can absorb or invert it (see the module grammar)."""
    if isinstance(node, ast.BoolOp):
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
