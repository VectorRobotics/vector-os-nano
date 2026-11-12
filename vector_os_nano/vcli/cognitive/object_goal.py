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
- ``task_contract`` compiles a goal made only of closed sentence forms into the GT
  predicates that must be False at turn start and True at turn end (trace_store's
  goal check); any other goal compiles none and is reported unchecked (charter 3.5).

Stricter-only (rule 5): the gates built on these only REJECT a turn that already
classified; intent comes from the user's words (and the skills the producer
dispatched), never from actor-authored state.
"""
from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any

from vector_os_nano.vcli.cognitive.coord_goal import parse_goal_coord
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr

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
        r"(?<![播开释解])放(?![松弃心大映假学电过])|松开|松手|丢下|扔下"
        r"|(?<!the )(?<!a )(?<!this )(?<!that )\bplace\b|\bput\b|\bdrop\b|\brelease\b|\blet\s+go\b"
        r"|\bset\s+(?:[\w']+\s+){0,4}?down\b",
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


def verify_intents(expr: str | None) -> frozenset[str]:
    """The manipulation *expr* proves was performed: "grasp" when it asserts a
    possession oracle, "place" when it asserts a release or placement (never under
    ``not`` / ``or``: "not holding" proves no grasp)."""
    node = _parse(expr)
    called = {c.func.id for c in _asserted_calls(node)} if node is not None else set()
    return frozenset(i for i, names in (("grasp", _HOLD), ("place", _PLACE)) if called & names)


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


# The closed whole-sentence forms a task contract compiles from (charter 3.5), by the
# end state each names. ``{o}`` is an object slot (an object name or a pronoun), ``{s}``
# a destination slot (a surface name). A goal is lead grasp clauses, then one clause of
# any kind; a phrasing nobody listed compiles nothing (unchecked), never a guess.
_FORMS: dict[str, tuple[str, ...]] = {
    "grasp": (r"pick up {o}", r"pick {o} up", r"(?:grab|grasp|take|hold|take hold of|get hold of) {o}",
              r"(?:拿起|抓起|捡起|拿住|抓住|抓取|夹起|夹住|握住|拿|抓|捡)(?:一下)?{o}",
              r"把{o}(?:拿|抓|捡|夹)(?:起来|起|住)"),
    "release": (r"(?:put|set) {o} down", r"(?:put|set) down {o}", r"(?:drop|release|let go of) {o}", r"let go",
                r"把{o}(?:放下|放开|松开|丢下|扔下)来?", r"(?:放下|放开|松开|丢下|扔下){o}", r"放下|放开|松开|松手|放手"),
    "place": (r"(?:put|place|set) {o} (?:down )?(?:on top of|onto|on) {s}",
              r"(?:把{o})?(?:放到|放在|搁到|搁在|摆到|摆在){s}(?:上面|上)"),
    "fetch": (r"(?:bring|get|give|hand|fetch) me {o}", r"(?:give|hand) {o} to me",
              r"(?:bring|fetch) {o}(?: to me| to the user| over here| here| over)?", r"把{o}(?:拿|递|带|送)(?:过来给我|来给我|过来|来|给我)(?:一下|一点)?",
              r"拿{o}(?:过来|给我)(?:一下|一点)?",
              r"(?:拿|递|带)给我{o}", r"给我拿(?:来|过来)?{o}"),
}
# One goal per form above, in a world's own words ({o} an object's English name, {zo} a
# CJK one, {s} a surface): what the manipulation gate offers a goal outside every form.
# checkable_goals offers one only where this world can prove its contract (test_manipulation_contract).
_EXAMPLES = ("pick up the {o}", "put the {o} down", "put the {o} on the {s}", "bring me the {o}", "把{zo}拿过来")
_CLAUSES = [(kind, re.compile(f.format(o="(?P<o>.+?)", s="(?P<s>.+)"))) for kind, fs in _FORMS.items() for f in fs]
_THEN = re.compile(r"\s*(?:,?\s*\band then\b|,?\s*\band\b|,?\s*\bthen\b|[,，、;；]|然后|再)\s*")
# Politeness stripped at either end (a closed list, like the forms): only words that
# cannot change the end state after any form. 一下 / 一点 can (放下一点: lower it a bit;
# 松开一下: let go for a moment), so they are part of the fetch forms alone (拿过来一点).
_LEAD = re.compile(r"^(?:(?:please|kindly|just|can you|could you|would you|will you)\b|请|麻烦你|麻烦|帮我|你)\s*")
_TAIL = re.compile(r"[\s,，]*(?:\b(?:please|thanks|thank you|for me|now)|吧|谢谢|好吗)$|[\s.!?。！？~～]+$")
_PRONOUNS = frozenset({"it", "that", "this", "them", "that one", "this one", "它", "它们", "这个", "那个"})


def _names(world: Any, phrase: str, slot: str) -> bool:
    """*phrase* fills *slot* and nothing else: an object slot ``o`` a pronoun or an
    object reference in *world*'s words (``WorldManifest.referents``: bound or not), a
    destination slot ``s`` a surface name or a word for furniture (its scenery)."""
    if slot == "o":
        return phrase in _PRONOUNS or world.referents(phrase) is not None
    x = re.sub(r"^(?:(?:the|a|an|this|that|my)\s+|这个|那个|这|那)", "", phrase)
    names = {w.lower() for n, s in world.surfaces.items() for w in (n, n.replace("_", " "), *s.aliases)}
    return x in names or re.fullmatch(rf"(?:{_alt(world.scenery)})(?:子|s)?", x) is not None


def _alt(words: Any) -> str:
    return "|".join(sorted({re.escape(w.lower()) for w in words}, key=len, reverse=True)) or "(?!)"


def _sentence(world: Any, goal_text: str) -> tuple[str, str | None, str | None] | None:
    """The whole goal as closed forms (grasp clauses, then any one clause; a grasp may be
    glued to it: 拿起X放到Y上), or None: ``(the last clause's kind, the object, the
    destination)``, each bound through *world* (None: unknown, ambiguous, a pronoun)."""
    def one(t: str) -> tuple[str, dict[str, str]] | None:
        for kind, rx in _CLAUSES:
            m = rx.fullmatch(t.strip())
            slots = {k: v.strip() for k, v in (m.groupdict() if m else {}).items() if v}
            if m and all(_names(world, v, k) for k, v in slots.items()):
                return kind, slots
        return None

    text, prev = unicodedata.normalize("NFKC", goal_text).lower().strip(), None
    while text != prev:
        prev, text = text, _TAIL.sub("", _LEAD.sub("", text)).strip()
    clauses: list[tuple[str, dict[str, str]]] = []
    for c in filter(None, _THEN.split(text)):
        glued = next(([h, t] for i in range(1, len(c)) if (h := one(c[:i])) and h[0] == "grasp"
                      and (t := one(c[i:]))), None)
        clauses += [whole] if (whole := one(c)) else glued or [("", {})]
    if not clauses or any(kind != "grasp" for kind, _ in clauses[:-1]) or not clauses[-1][0]:
        return None
    named = {world.resolve(v) for _, slots in clauses if (v := slots.get("o")) and v not in _PRONOUNS}
    dest = clauses[-1][1].get("s")
    return clauses[-1][0], named.pop() if len(named) == 1 else None, dest and world.resolve_surface(dest)


def checkable_goals(world: Any, oracle_names: frozenset[str]) -> list[str]:
    """``_EXAMPLES`` filled with *world*'s first object and surface, each only when this
    world can prove its contract (REF1c-V-03): every item a ground-truth predicate here
    with every slot bound, and a fetch only where the world has a user to deliver to."""
    pk = next(iter(world.pickables.values()))
    zo = next((a for a in pk.aliases if not a.isascii()), pk.english)
    s = next(iter(world.surfaces), "").replace("_", " ")
    goals = [e.format(o=pk.english, zo=zo, s=s) for e in _EXAMPLES]
    return [g for g in goals if (c := task_contract(g, oracle_names, world.id)) and all(
        classify_verify_expr(i, oracle_names) == "GROUNDED" and not unbound_reason(i, g, world.id)
        and (world.user is not None or i != "near_object('user')") for i in c)]


def unbound_reason(item: str, goal_text: str | None, world_id: str = "") -> str:
    """Why contract *item* is no evidence ("" when every slot bound): which slot of the
    goal the world does not resolve (the object, or a place's destination), the goal's
    words, and what the world has to choose from."""
    call = _parse(item)
    if not isinstance(call, ast.Call):
        return ""
    world = _manifest(world_id)
    slots = ("object", "destination") if item.startswith("object_on(") else ("object",)
    for slot, arg in zip(slots, call.args):
        if not (isinstance(arg, ast.Constant) and arg.value is None):
            continue
        has = (world.vocabulary() if slot == "object" else sorted(world.surfaces)) if world is not None else []
        listed = f"; {'objects' if slot == 'object' else 'surfaces'}: {', '.join(has)}" if has else ""
        return f"the goal {goal_text!r} names no {slot} the world resolves{listed}"
    return ""


def task_contract(goal_text: str | None, oracle_names: frozenset[str], world_id: str = "") -> tuple[str, ...]:
    """The goal's END-STATE ground-truth predicates (charter 3.5), ONLY when the WHOLE
    goal is closed forms (``_FORMS``) whose slots bind through the world manifest
    *world_id*; anything else compiles () -- unchecked, never guessed. grasp ->
    ``holding_object(obj)``; release -> ``released(obj)`` (a pick-and-place starts released:
    trace_store exempts it only after a grounded grasp of obj); place on a surface ->
    ``object_on(obj, surface)``; fetch to the user -> ``kept_holding(obj)`` (held, never
    dropped this turn) and ``near_object('user')``; a pure coordinate goal ->
    ``at_position(x, y)``. A slot bound to None (unknown, ambiguous, a pronoun) fails
    the goal check (``unbound_reason``). Scoped to the world's oracles (rule 3): no
    manipulation contract without ``holding_object`` and a manifest, no coordinate one
    without ``at_position`` or beside a manipulation intent (a waypoint)."""
    world = _manifest(world_id) if "holding_object" in oracle_names else None
    form = _sentence(world, goal_text or "") if world is not None else None
    if form is not None:
        kind, obj, surface = form
        return {"grasp": (f"holding_object({obj!r})",), "release": (f"released({obj!r})",),
                "place": (f"object_on({obj!r}, {surface!r})",),
                "fetch": (f"kept_holding({obj!r})", "near_object('user')")}[kind]
    xy = parse_goal_coord(goal_text)
    if xy is not None and "at_position" in oracle_names and not goal_intents(goal_text):
        return (f"at_position({xy[0]}, {xy[1]})",)
    return ()
