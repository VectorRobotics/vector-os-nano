# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Goal authenticity for MANIPULATION goals (D17 / D69 / WP07), and the turn-level
task contract — the object-goal analogue of ``coord_goal``.

A manipulation goal (grasp / fetch / place) must be proven by a ground-truth
MANIPULATION oracle the actor cannot author (a possession oracle for a grasp / fetch;
a release or placement for a place, both for a pick-and-place), never by a self-authored file / scene /
marker predicate — the fakeable-grasp false-green (D17) — nor by a detect.

- ``goal_intents`` reads the intents from the user's NL goal through ONE vocabulary
  (the manipulation skills' aliases: 拿/拿过来/拿给我/取/递/送/搬/放/fetch/bring/
  carry/take/get/deliver/place/put ...); test_object_goal pins that every alias of a
  gripper skill is covered, so a new alias cannot fail the gate open.
- ``has_necessary_manip_oracle`` checks that a GT manipulation oracle GATES a verify.
- ``task_contract`` compiles the goal into the GT predicates that must be False at
  turn start and True at turn end (trace_store checks them).

Stricter-only (rule 5): the gates built on these only REJECT a turn that already
classified; intent comes from the user's words (and the skills the producer
dispatched), never from actor-authored state.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from vector_os_nano.vcli.cognitive.coord_goal import parse_goal_coord

# The manipulation vocabulary, by intent. A fetch also needs the grasp proof.
_INTENT_RES: dict[str, re.Pattern[str]] = {
    "grasp": re.compile(
        r"抓|夹住|夹起|夹取|拿(?!手)|握住|捡起|(?<![获读提选领])取(?!消|决|得)"
        r"|\bpick\b|grasp|\bgrab|gripp"
        r"|\btake\b(?!\s+(?:a\s+)?(?:look|picture|photo|turn|step|seat|break|rest|me|us|off)\b)"
        r"|\bget\b(?!\s+(?:to|there|back|up|going|out|in|into|off|closer|started|ready)\b)",
        re.IGNORECASE,
    ),
    "fetch": re.compile(
        r"拿过来|拿来|拿给|取来|递|(?<![发运传])送|搬|带过来|带来|带给"
        r"|\bbring\b|\bfetch|\bcarry|\bdeliver|\bhand\s+(?:it\s+|that\s+|this\s+)?(?:me|over)\b"
        r"|\bget\s+me\b|\bgive\s+me\b",
        re.IGNORECASE,
    ),
    "place": re.compile(
        r"(?<![播开释解])放(?![松弃心大映假学电过])"
        r"|(?<!the )(?<!a )(?<!this )(?<!that )\bplace\b|\bput\b|\bdrop\b"
        r"|\bset\s+(?:it\s+|that\s+|this\s+)?down\b",
        re.IGNORECASE,
    ),
}

# GT oracles that PROVE possession / placement. A grasp or fetch goal needs a
# POSSESSION oracle — holding_object (weld + finger contacts + lifted), lifted (held
# clear of its support), kept_holding (held, never dropped this turn); an object resting
# somewhere proves no grasp (D69). A pure place goal ends on a release or a placement.
_HOLD: frozenset[str] = frozenset({"holding_object", "lifted", "kept_holding"})
_PLACE: frozenset[str] = frozenset({"released", "object_on", "object_at"})


def goal_intents(goal_text: str | None) -> frozenset[str]:
    """The manipulation intents ("grasp" / "fetch" / "place") the NL goal names."""
    if not goal_text:
        return frozenset()
    return frozenset(i for i, rx in _INTENT_RES.items() if rx.search(goal_text))


def goal_has_object_intent(goal_text: str | None) -> bool:
    """True iff the NL goal names any manipulation (grasp / fetch / place)."""
    return bool(goal_intents(goal_text))


def proof_oracles(goal: frozenset[str] | set[str], dispatched: frozenset[str] | set[str] = frozenset()
                  ) -> tuple[frozenset[str], ...]:
    """The oracle sets a turn must EACH prove (empty: no gate), from the *goal*'s
    intents and the *dispatched* skills' ones: a hold for a grasp / fetch, a release or
    placement for a place. A dispatched intent adds a proof, never replaces the goal's
    own (I4-V1: a place that also grasped still needs its place proof); a release
    dispatched inside a grasp / fetch goal (a pre-open) adds none."""
    intents = set(goal) | set(dispatched)
    hold = bool(intents & {"grasp", "fetch"})
    place = "place" in goal or ("place" in intents and not set(goal) & {"grasp", "fetch"})
    return tuple(names for names, need in ((_HOLD, hold), (_PLACE, place)) if need)


def _necessary_manip(node: ast.AST, names: frozenset[str]) -> bool:
    """True iff a call to one of *names* GATES *node*'s truth: the call itself, an
    ``and`` conjunct, under ``not``, or inside a compare. An ``or`` sibling makes it
    non-necessary (``holding_object('x') or True``)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return any(_necessary_manip(v, names) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _necessary_manip(node.operand, names)
    if isinstance(node, ast.Compare):
        return any(_necessary_manip(c, names) for c in (node.left, *node.comparators))
    return False


def _parse(expr: str | None) -> ast.AST | None:
    if not expr:
        return None
    try:
        return ast.parse(expr.strip(), mode="eval").body
    except SyntaxError:
        return None


def has_necessary_manip_oracle(expr: str | None, names: frozenset[str] = _HOLD) -> bool:
    """True iff *expr* has a NECESSARY call to one of *names* (default: the GT
    possession oracles). Fail-closed on a syntax error or an ``or`` decoy."""
    node = _parse(expr)
    return node is not None and _necessary_manip(node, names)


def _asserted_calls(node: ast.AST) -> list[ast.Call]:
    """Calls whose truth the verify ASSERTS: the call, an ``and`` conjunct, or
    ``<call> == True`` — never under ``not`` / ``or`` / an ordering."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return [node]
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [c for v in node.values for c in _asserted_calls(v)]
    if (
        isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
        and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is True
    ):
        return _asserted_calls(node.left)
    return []


def _manifest(world_id: str) -> Any:
    """The world manifest ``config/worlds/<world_id>.yaml`` (pure data: what the world
    contains and what NL calls it, never where it is), or None."""
    if not world_id:
        return None
    from vector_os_nano.vcli.worlds.world_manifest import WorldManifestError, load_world

    try:
        return load_world(world_id)
    except WorldManifestError:
        return None


def backs_contract(item: str, verify: str | None, world_id: str = "") -> bool:
    """True iff *verify* asserts the contract *item*: a coordinate item needs the
    same necessary ``at_position`` (coord_goal); any other item a necessary call to
    the same oracle whose leading positional arguments name the same things as the
    item's (an object by any name the world manifest resolves to it); a trailing
    tolerance is the verify's own (the contract's end value is read at the default)."""
    from vector_os_nano.vcli.cognitive.coord_goal import _const_number, at_position_const_matches

    want, node = _parse(item), _parse(verify)
    if not isinstance(want, ast.Call) or not isinstance(want.func, ast.Name) or node is None:
        return False
    if want.func.id == "at_position":
        xy = [_const_number(a) for a in want.args[:2]]
        return len(xy) == 2 and None not in xy and at_position_const_matches(verify, (xy[0], xy[1]))
    world = _manifest(world_id)

    def canon(arg: ast.AST) -> Any:
        if not isinstance(arg, ast.Constant):
            return ast.dump(arg)
        v = arg.value
        return (world.resolve(v) or v) if world is not None and isinstance(v, str) else v

    return any(
        c.func.id == want.func.id and len(c.args) >= len(want.args)
        and all(canon(a) == canon(b) for a, b in zip(want.args, c.args))
        for c in _asserted_calls(node)
    )


def task_contract(goal_text: str | None, oracle_names: frozenset[str], world_id: str = "") -> tuple[str, ...]:
    """Compile the NL goal into its END-STATE ground-truth predicates (WP07, I4).

    The object and surface are bound through the world manifest *world_id* (the
    object the goal names, ``WorldManifest.resolve``; the surface,
    ``resolve_surface``), never from ground truth:

    - place naming a surface (放到餐桌上 / put it on the dining table) ->
      ``object_on(obj, surface)``; a place naming no surface compiles none (the
      manipulation gate still needs a release / placement step);
    - fetch (拿过来 / bring ...) -> ``kept_holding(obj)`` and ``near_object('user')``:
      delivered = held, never dropped this turn, at the user. kept_holding is caused
      only by a grasp in this turn (its ``window="turn"`` tag), and each item must be
      False at turn start: an object already held, or dropped during the carry, never
      grounds a fetch;
    - grasp (拿起 / pick up ...) -> ``holding_object(obj)`` (``holding_object()``, any
      object, only in a world without a manifest);
    - a pure coordinate goal -> ``at_position(x, y)``. With a manipulation intent the
      coordinate is a waypoint, not an end state; the coordinate gate checks it.

    An object the manifest cannot resolve ("bring it to me", two bottles, an apple the
    world lacks) binds
    ``None``: that item is no ground-truth evidence, so the turn fails closed. Scoped
    to the world's oracles (rule 3): a world without ``holding_object`` /
    ``at_position`` gets no manipulation / coordinate contract (dev). A manipulation
    goal never falls back to its coordinate: no contract, so every graded step must
    ground.
    """
    intents = goal_intents(goal_text)
    if intents:
        if "holding_object" not in oracle_names:
            return ()
        world = _manifest(world_id)
        obj = world.resolve(goal_text) if world is not None else None
        if "place" in intents:
            surface = world.resolve_surface(goal_text) if world is not None else None
            return (f"object_on({obj!r}, {surface!r})",) if surface else ()
        if "fetch" in intents:
            return (f"kept_holding({obj!r})", "near_object('user')")
        # a manifest binds the object or None (fails closed: "the apple", two bottles)
        return (f"holding_object({obj!r})" if world is not None else "holding_object()",)
    xy = parse_goal_coord(goal_text)
    if xy is not None and "at_position" in oracle_names:
        return (f"at_position({xy[0]}, {xy[1]})",)
    return ()
