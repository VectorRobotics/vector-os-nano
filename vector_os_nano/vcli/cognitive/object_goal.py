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
- ``task_contract`` compiles a goal made only of closed sentence forms (manipulation,
  a room, a coordinate) into the GT predicates that must be False at turn start and
  True at turn end (trace_store's goal check), and ``contract_leads`` its lead clauses'
  predicates, each needing its own grounded step; any other goal compiles none and is
  reported unchecked (charter 3.5).

Stricter-only (rule 5): the gates built on these only REJECT a turn that already
classified; intent comes from the user's words (and the skills the producer
dispatched), never from actor-authored state.
"""
from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any

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
    """Calls whose truth the verify ASSERTS (a contract proof): the call, an ``and``
    conjunct, or ``<call> == True`` — never ``True == <call>`` (REF1e-V-01), nor under
    ``not`` / ``or`` / an ordering."""
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


def named_intents(expr: str | None) -> frozenset[str]:
    """The intents whose oracle NAME appears anywhere in *expr* (REF1e): "grasp" for a
    possession oracle, "place" for a release or placement. What it asserts is never read,
    so no spelling the classifier grounds hides a manipulation from the gate."""
    node = _parse(expr)
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if node is not None else set()
    return frozenset(i for i, ns in (("grasp", _HOLD), ("place", _PLACE)) if names & ns)


def _manifest(world_id: str) -> Any:
    """The world manifest ``worlds/<world_id>/world.yaml`` (pure data: what the world
    contains and what NL calls it, never where it is), or None."""
    if not world_id:
        return None
    from vector_os_nano.schemas.world import WorldManifestError, load_world

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
        if world is not None and isinstance(v, str):
            resolve = world.resolve_articulation if want.func.id in {'articulation_open', 'articulation_closed'} else world.resolve
            return resolve(v) or v
        return v

    return any(
        c.func.id == want.func.id and len(c.args) >= len(want.args)
        and all(canon(a) == canon(b) for a, b in zip(want.args, c.args))
        for c in _asserted_calls(node)
    )


# The closed whole-sentence forms a task contract compiles from (charter 3.5), by the
# end state each names. Slots: ``{o}`` an object (a name, a pronoun, or a name with its
# CHECKED location: 取物台上的红罐子, "the red can on the pick table"), ``{s}`` a surface,
# ``{r}`` a room, ``{xy}`` a coordinate. A goal is lead clauses (``_LEADS``), then one clause
# of any kind; a phrasing nobody listed compiles nothing (unchecked), never a guess. Read in
# this order: a place before a release ("put down X on S": S is where X goes, not where it is).
_FORMS: dict[str, tuple[str, ...]] = {
    "drawer_open": (r"open {a}", r"打开{a}", r"把{a}打开"),
    "drawer_close": (r"close {a}", r"(?:关闭|关好){a}", r"把{a}(?:关闭|关好)"),
    "place": (r"(?:put|place|set|drop) {o} (?:down )?(?:on top of|onto|on) {s}",
              r"(?:put|set) down {o} (?:on top of|onto|on) {s}",
              r"(?:把?{o})?(?:放到|放在|搁到|搁在|摆到|摆在){s}(?:上面|上)"),
    "grasp": (r"pick up {o}", r"pick {o} up", r"lift (?:up )?{o}",
              r"(?:take hold of|get hold of|grab|grasp|take|hold) {o}",
              r"(?:拿起|抓起|捡起|提起|拿出|拿住|抓住|抓取|夹起|夹住|握住|拿|抓|捡)(?:一下)?{o}",
              r"把{o}(?:拿|抓|捡|夹|提)(?:起来|起|住|出来)"),
    "release": (r"(?:put|set) {o} down", r"(?:put|set) down {o}", r"(?:drop|release|let go of) {o}", r"let go",
                r"把{o}(?:放下|放开|松开|丢下|扔下)来?", r"(?:放下|放开|松开|丢下|扔下){o}", r"放下|放开|松开|松手|放手"),
    "fetch": (r"(?:bring|get|give|hand|fetch|grab) me {o}", r"(?:give|hand) {o} to me", r"get {o} for me",
              r"(?:bring|fetch) {o}(?: to me| to the user| over here| here| over)?",
              r"把{o}(?:(?:拿|递|带|送)(?:过来给我|来给我|过来|来|给我)|给我)(?:一下|一点)?",
              r"拿{o}(?:过来|给我)(?:一下|一点)?", r"(?:拿起|拿出)(?:一下)?{o}给我",
              r"(?:拿|递|带)给我{o}", r"给我拿(?:来|过来)?{o}"),
    "room": (r"(?:go|walk|head|move|navigate|run|come|return|get)(?: back| over)? (?:in)?to {r}", r"enter {r}",
             r"(?:回到|走到|走去|走进|跑到|前往|进入|导航到|移动到|去|到|回|进)到?{r}(?:去|里面|里|那里|那边|一趟)?"),
    "coord": (r"(?:go|walk|head|move|navigate|run|drive)(?: back| over)? to (?:the )?"
              r"(?:point|position|coordinates?|location)?{xy}",
              r"(?:回到|走到|走去|跑到|前往|导航到|移动到|去|到)到?(?:坐标|位置|点)?{xy}(?:去|那里|处)?"),
}
# The clause kinds a goal may lead with: each needs its own GROUNDED step in the turn,
# before the step that proves the last clause (trace_store._contract_reason).
_LEADS = frozenset({"grasp", "room", "coord"})
# One goal per manipulation form, in a world's own words ({o} an object's English name, {zo}
# a CJK one, {s} a surface): what the manipulation gate offers a goal outside every form.
# checkable_goals offers one only where this world can prove its contract (test_manipulation_contract).
_EXAMPLES = ("pick up the {o}", "put the {o} down", "put the {o} on the {s}", "bring me the {o}", "把{zo}拿过来")
# A grasp may say where the object ends up: in the hand (拿在手里, "in your hand").
_IN_HAND = r"(?:(?:拿|握|抓)在手(?:里|上|中)| in (?:your|the|its) (?:hand|gripper))?"
_SLOTS = {"o": r"(?P<o>.+?)", "s": r"(?P<s>.+)", "r": r"(?P<r>.+?)", "a": r"(?P<a>.+?)",
          "xy": r"\s*\(\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s*,\s*(?P<y>[-+]?\d+(?:\.\d+)?)\s*\)"}
_CLAUSES = [(kind, re.compile(f.format(**_SLOTS) + (_IN_HAND if kind == "grasp" else "") + "(?: for me)?"))
            for kind, fs in _FORMS.items() for f in fs]
# Clause separators; never a comma inside a coordinate's parentheses.
_THEN = re.compile(r"\s*(?:,?\s*\band then\b|,?\s*\band\b|,?\s*\bthen\b|[,，、;；]|然后|再)(?![^(]*\))\s*")
# Politeness stripped at either end of the goal and of each clause (a closed list, like the
# forms): only words that cannot change the end state after any form. 一下 / 一点 can
# (放下一点: lower it a bit; 松开一下: let go for a moment), so they are part of the fetch
# forms alone (拿过来一点); "for me" can ("get X for me" is a fetch), so the forms carry it.
_LEAD = re.compile(r"^(?:(?:please|kindly|just|can you|could you|would you|will you)\b|请|麻烦你|麻烦|帮我|你)\s*")
_TAIL = re.compile(r"[\s,，]*(?:\b(?:please|thanks|thank you|now)|吧|谢谢|好吗)$|[\s.!?。！？~～]+$")
_PRONOUNS = frozenset({"it", "that", "this", "them", "that one", "this one", "它", "它们", "这个", "那个"})
_IT = "<it>"  # an object slot that is a pronoun (or a release / place that names none)
_DET = re.compile(r"^(?:(?:the|a|an|this|that|my)\s+|这个|那个|这|那)")
# A location inside an object slot: the surface it rests on in the manifest, never a guess.
_ON = (re.compile(r"(?P<s>.+?)(?:上面|上)的(?P<o>.+)"), re.compile(r"(?P<o>.+?) (?:on|from) (?:top of )?(?P<s>.+)"))
# A generic word for a room the world may not have ("the game room", 房间, 会议室): it fills
# a room slot that then binds nothing (failed, with the rooms), like scenery for a surface.
_ROOM_HEAD = re.compile(r"(?:[a-z]+ ?)?room|[一-鿿]{0,2}(?:房间|屋|室|厅|房)")


def _object(world: Any, phrase: str) -> tuple[bool, str | None]:
    """Whether *phrase* fills an object slot of *world* and what it binds: ``_IT`` for a
    pronoun, a body when exactly one fits every word (``WorldManifest.referents``), else
    None. A possessive (我的, my) is a filler; a location (取物台上的X) binds only when it
    is the surface the manifest puts X on."""
    p = re.sub(r"^(?:我的|my\s+)", "", phrase)
    if p in _PRONOUNS:
        return True, _IT
    if world.referents(p) is not None:
        return True, world.resolve(p)
    for rx in _ON:
        m = rx.fullmatch(p)
        if m and world.referents(m["o"]) is not None and _surface(world, m["s"]):
            obj, on = world.resolve(m["o"]), world.resolve_surface(m["s"])
            return True, obj if obj is not None and on is not None and world.pickables[obj].surface == on else None
    return False, None


def _surface(world: Any, phrase: str) -> bool:
    """*phrase* fills a destination slot: a surface name or a word for furniture (scenery)."""
    x = _DET.sub("", phrase)
    names = {w.lower() for n, s in world.surfaces.items() for w in (n, n.replace("_", " "), *s.aliases)}
    return x in names or re.fullmatch(rf"(?:{_alt(world.scenery)})(?:子|s)?", x) is not None


def _room(world: Any, phrase: str) -> tuple[bool, str | None]:
    """Whether *phrase* fills a room slot and the room it binds (a word naming exactly one
    room of *world*), else None; a generic room word fills it and binds nothing."""
    x, words = _DET.sub("", phrase), world.room_words()
    if x in words:
        return True, world.resolve_room(x)
    return _ROOM_HEAD.fullmatch(x) is not None and not any(w in x for w in words if not w.isascii()), None


def _alt(words: Any) -> str:
    return "|".join(sorted({re.escape(w.lower()) for w in words}, key=len, reverse=True)) or "(?!)"


def _strip(text: str) -> str:
    prev = None
    while text != prev:
        prev, text = text, _TAIL.sub("", _LEAD.sub("", text)).strip()
    return text


def _clause(world: Any, text: str) -> tuple[str, dict[str, Any]] | None:
    """The first form *text* matches whole with every slot filled: ``(kind, {slot: bound})``."""
    for kind, rx in _CLAUSES:
        m = rx.fullmatch(text.strip())
        if m is None:
            continue
        slots = {k: v.strip() for k, v in m.groupdict().items() if v}
        bound: dict[str, Any] = {}
        if "x" in slots:
            bound["xy"] = (float(slots.pop("x")), float(slots.pop("y")))
        if slots and world is None:
            continue
        fill = {"o": lambda v: _object(world, v), "s": lambda v: (_surface(world, v), world.resolve_surface(v)),
                "r": lambda v: _room(world, v),
                "a": lambda v: (world.resolve_articulation(v) is not None, world.resolve_articulation(v))}
        got = {k: fill[k](v) for k, v in slots.items()}
        if all(ok for ok, _ in got.values()):
            return kind, {**bound, **{k: b for k, (_, b) in got.items()}}
    return None


def _clauses(world: Any, goal_text: str) -> list[tuple[str, dict[str, Any]]] | None:
    """The whole goal as closed forms (lead clauses, then any one clause; a lead may be
    glued to the next: 拿起X放到Y上, 去客厅把X拿起来), or None."""
    out: list[tuple[str, dict[str, Any]]] = []
    for c in filter(None, _THEN.split(_strip(unicodedata.normalize("NFKC", goal_text).lower().strip()))):
        c = _strip(c)
        whole = _clause(world, c)
        glued = None if whole else next(
            # Keep a complete object phrase before trying a shorter, ambiguous prefix.
            ([h, t] for i in range(len(c) - 1, 0, -1) if (h := _clause(world, c[:i])) and h[0] in _LEADS
             and (t := _clause(world, c[i:]))), None)
        out += [whole] if whole else glued or [("", {})]
    if not out or any(kind not in _LEADS for kind, _ in out[:-1]) or not out[-1][0]:
        return None
    return out


def _sentence(world: Any, goal_text: str, held: tuple[str, ...] = ()) -> tuple[str, str | None, Any, tuple] | None:
    """The whole goal as closed forms, or None: ``(the last clause's kind, the object, its
    destination, the lead clauses as (kind, destination))``. A destination is a surface,
    a room or a coordinate. Every object slot names the same object or it is None; one
    named only as 'it' (or a release / place naming none) is the ONE object *held* at turn
    start (ground truth), never in a grasp ("pick it up": nothing says which)."""
    clauses = _clauses(world, goal_text)
    if clauses is None:
        return None
    objs = [s.get("o", _IT) for k, s in clauses if "o" in s or k in ("release", "place")]
    named = {o for o in objs if o != _IT}
    if named:
        obj = named.pop() if len(named) == 1 else None
    else:
        grasp = any(k == "grasp" for k, _ in clauses)
        obj = held[0] if objs and not grasp and len(held) == 1 else None

    def dest(slots: dict[str, Any]) -> Any:
        return next((slots[k] for k in ("s", "r", "xy", "a") if k in slots), None)

    return clauses[-1][0], obj, dest(clauses[-1][1]), tuple((k, dest(s)) for k, s in clauses[:-1])


def checkable_goals(world: Any, oracle_names: frozenset[str]) -> list[str]:
    """``_EXAMPLES`` filled with *world*'s first object and first surface other than the one
    it starts on (a place there holds at turn start: REF1d-V-02), each only when this
    world can prove its contract (REF1c-V-03): every item a ground-truth predicate here
    with every slot bound, and a fetch only where the world has a user to deliver to."""
    pk = next(iter(world.pickables.values()))
    zo = next((a for a in pk.aliases if not a.isascii()), pk.english)
    s = next((n for n in world.surfaces if n != pk.surface), "").replace("_", " ")
    goals = [e.format(o=pk.english, zo=zo, s=s) for e in _EXAMPLES]
    return [g for g in goals if (c := task_contract(g, oracle_names, world.id)) and all(
        classify_verify_expr(i, oracle_names) == "GROUNDED" and not unbound_reason(i, g, world.id)
        and (world.user is not None or i != "near_object('user')") for i in c)]


def unbound_reason(item: str, goal_text: str | None, world_id: str = "", held: tuple[str, ...] = ()) -> str:
    """Why contract *item* is no evidence ("" when every slot bound): which slot of the
    goal the world does not resolve (the object, a place's destination, a room), the
    goal's words, and what the world has to choose from (for an object named only as
    'it': what the robot held at turn start, *held*)."""
    call = _parse(item)
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return ""
    world = _manifest(world_id)
    slots = {"object_on": ("object", "destination"), "in_room": ("room",)}.get(call.func.id, ("object",))
    for slot, arg in zip(slots, call.args):
        if not (isinstance(arg, ast.Constant) and arg.value is None):
            continue
        clauses = _clauses(world, goal_text or "") if world is not None else None
        if slot == "object" and clauses and all(s.get("o", _IT) == _IT for _, s in clauses):
            return (f"the goal {goal_text!r} names its object only as 'it' (or not at all): that binds the one "
                    f"object held at turn start, in a place, release or fetch; held: {', '.join(held) or 'nothing'}")
        has = {"object": world.vocabulary, "destination": lambda: sorted(world.surfaces),
               "room": lambda: sorted(world.rooms)}[slot]() if world is not None else []
        listed = f"; {'surfaces' if slot == 'destination' else slot + 's'}: {', '.join(has)}" if has else ""
        return f"the goal {goal_text!r} names no {slot} the world resolves{listed}"
    return ""


_NEEDS = {"room": "in_room", "coord": "at_position",
          "drawer_open": "articulation_open", "drawer_close": "articulation_closed"}


def _items(kind: str, obj: str | None, dest: Any) -> tuple[str, ...]:
    if kind == "coord":
        return (f"at_position({dest[0]}, {dest[1]})",)
    return {"grasp": (f"holding_object({obj!r})",), "release": (f"released({obj!r})",),
            "drawer_open": (f"articulation_open({dest!r})",), "drawer_close": (f"articulation_closed({dest!r})",),
            "place": (f"object_on({obj!r}, {dest!r})",), "room": (f"in_room({dest!r})",),
            "fetch": (f"kept_holding({obj!r})", "near_object('user')")}[kind]


def _compile(goal_text: str | None, oracle_names: frozenset[str], world_id: str,
             held: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    world = _manifest(world_id)
    form = _sentence(world, goal_text or "", held)
    if form is None:
        return (), ()
    kind, obj, dest, leads = form
    if any(_NEEDS.get(k, "holding_object") not in oracle_names for k in (kind, *(k for k, _ in leads))):
        return (), ()
    return _items(kind, obj, dest), tuple(_items(k, obj, d)[0] for k, d in leads)


def task_contract(goal_text: str | None, oracle_names: frozenset[str], world_id: str = "",
                  held: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The goal's END-STATE ground-truth predicates (charter 3.5): the last clause's, ONLY
    when the WHOLE goal is closed forms (``_FORMS``) whose slots bind through the world
    manifest *world_id*; anything else compiles () -- unchecked, never guessed. grasp ->
    ``holding_object(obj)``; release -> ``released(obj)`` (a pick-and-place starts released:
    trace_store exempts it only after a grounded grasp of obj); place on a surface ->
    ``object_on(obj, surface)``; fetch to the user -> ``kept_holding(obj)`` (held, never
    dropped this turn) and ``near_object('user')``; go to a room -> ``in_room(room)``; go to
    a coordinate -> ``at_position(x, y)``. An object named only as 'it' binds the one
    object in *held* (the grasp attachments at turn start). A slot bound to None (unknown,
    ambiguous, a pronoun with nothing or several held) fails the goal check
    (``unbound_reason``). Scoped to the world's oracles (rule 3): every clause's oracle
    (``holding_object`` for a manipulation, ``in_room``, ``at_position``) must be live."""
    return _compile(goal_text, oracle_names, world_id, held)[0]


def contract_leads(goal_text: str | None, oracle_names: frozenset[str], world_id: str = "",
                   held: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The predicates of the goal's lead clauses (``_LEADS``: a grasp -> ``holding_object(obj)``,
    a room, a coordinate): each needs its own GROUNDED step in the turn, before the step that
    proves the contract (trace_store); () when the goal compiles no contract."""
    return _compile(goal_text, oracle_names, world_id, held)[1]
