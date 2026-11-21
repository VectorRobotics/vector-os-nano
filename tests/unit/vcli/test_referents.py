# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""REF1 — ONE rule for "which object does this phrase name" (WorldManifest.referents /
resolve), the one answer the oracles, the contract binder, the manipulation
gate and the grasp skill all ask.

A text is a reference only when every character of it is a word of this world (a
body name, an alias, a colour with 色, a category with a plural or 子) or a filler of
ONE closed list; the object must agree with every word. Three verification rounds each
closed one phrasing of the old resolver (which ignored the words it did not know) and
found the next: 'yellow banana' resolved to the yellow cup in the apartment.
"""
from __future__ import annotations

import random

import pytest

from vector_os_nano.vcli.worlds.world_manifest import WORLDS_DIR, load_world

G, B, R = "pickable_bottle_green", "pickable_bottle_blue", "pickable_can_red"
Y, O, P = "pickable_cup_yellow", "pickable_box_orange", "pickable_box_purple"
WORLDS = sorted(p.stem for p in WORLDS_DIR.glob("*.yaml"))

_APARTMENT = [
    # colour with category (no alias)
    ("绿瓶", G), ("绿色瓶", G), ("蓝瓶", B), ("the blue bottle", B), ("红罐", R), ("红色的罐", R), ("黄杯", Y),
    ("yellow mug", Y), ("the yellow mug", Y), ("橙盒", O), ("橘色的盒子", O), ("violet box", P), ("紫盒", P),
    ("Purple Box", P), ("ＧＲＥＥＮ ＢＯＴＴＬＥ", None), ("red罐子", R), ("红色 can", R), ("green  bottle", G),
    # colour with a generic head or 的
    ("the red one", R), ("红色的", R), ("红的", R), ("红色那个", R), ("the yellow thing", Y), ("黄色的东西", Y),
    ("那个黄色的", Y), ("a purple object", P), ("紫色的物体", P), ("the orange item", O), ("this green one", G),
    ("蓝色的东西", B), ("红色的东西", R), ("my red one", R),
    # a unique category (REF1: the one place the rule binds more than before)
    ("the cup", Y), ("cup", Y), ("mug", Y), ("杯子", Y), ("那个杯子", Y), ("the can", R), ("罐子", R),
    ("这个罐子", R), ("cans", R), ("my cup", Y), ("a can", R), ("杯", Y),
    # an ambiguous category
    ("the bottle", None), ("瓶子", None), ("bottles", None), ("box", None), ("盒子", None), ("这个盒子", None),
    # a noun of another world
    ("banana", None), ("香蕉", None), ("the duck", None), ("lego", None), ("螺丝刀", None), ("screwdriver", None),
    ("yellow banana", None), ("黄色的香蕉", None), ("马克杯", None), ("red lego", None), ("unicorn", None),
    # contradictions: colour against alias or category, category against alias
    ("blue can", None), ("red bottle", None), ("green cup", None), ("黄色的罐子", None), ("orange can", None),
    ("紫色的瓶子", None), ("blue red can", None), ("yellow red can", None), ("绿色的红罐子", None),
    ("red mug", None), ("green can", None), ("red can bottle", None), ("yellow cup box", None),
    ("pickable_can_red cup", None),
    # two objects
    ("red can green bottle", None), ("red can and green bottle", None), ("the cup or the can", None),
    ("红罐子绿瓶子", None), ("cup can", None), ("grab the red can, not the green bottle", None),
    ("把红色罐子旁边的绿色瓶子拿来", None),
    # a negation
    ("not the red can", None), ("不是红色的罐子", None), ("no red can", None), ("the red can not", None),
    # spatial words (the skill strips them; the rule does not)
    ("the can in front", None), ("前面的罐子", None), ("the red can on the left", None), ("桌上的杯子", None),
    ("nearest cup", None),
    # numbers
    ("2 cans", None), ("two cans", None), ("两个罐子", None), ("一个罐子", None), ("the 1 red can", None),
    ("red can 2", None),
    # plurals
    ("red cans", R), ("yellow cups", Y), ("the mugs", Y), ("orange boxes", O), ("blue bottles", B),
    # garbage before and after; a sentence is no reference
    ("xred can", None), ("red canx", None), ("red can!", None), ("red can.", None), ("please red can", None),
    ("red can please", None), ("抓红色的罐子", None), ("红色的罐子吧", None), ("the red can, thanks", None),
    ("grab the red can", None), ("bring me the red can", None), ("把绿色瓶子拿过来", None),
    ("the green bottle please", None), ("can you see anything", None),
    # look-alikes
    ("infrared can", None), ("scan", None), ("candle", None), ("粉红色的罐子", None), ("reddish can", None),
    ("cupboard", None), ("boxer", None), ("bottleneck", None), ("redcan", None), ("canned", None),
    ("the orange", None), ("orange", None), ("red", None), ("红色", None), ("红", None), ("the green", None),
    # scenery and colours this world has not
    ("red stool", None), ("红色的凳子", None), ("a red stool", None), ("red table", None), ("green chair", None),
    ("white cup", None), ("black can", None), ("白色的杯子", None),
    # fillers alone name nothing
    ("the", None), ("the thing", None), ("这个", None), ("东西", None), ("the one", None), ("的", None),
    # body names
    ("pickable_can_red", R), ("PICKABLE_CAN_RED", R), ("pickable_can_red can", R), ("the pickable_cup_yellow", Y),
    # empty, None and a non-string
    ("", None), ("   ", None), (None, None), (3, None), (["red can"], None),
]

_TABLETOP = [
    ("banana", "banana"), ("香蕉", "banana"), ("the yellow banana", "banana"), ("黄色的香蕉", "banana"),
    ("red banana", None), ("mug", "mug"), ("red mug", "mug"), ("blue mug", None), ("杯子", "mug"),
    ("马克杯", "mug"), ("红色的马克杯", "mug"), ("the red one", None), ("红色的", None), ("the yellow one", "banana"),
    ("the blue thing", "bottle"), ("绿色的", "screwdriver"), ("orange one", "duck"), ("rubber duck", "duck"),
    ("the ducks", "duck"), ("lego brick", "lego"), ("红色的乐高", "lego"), ("积木", "lego"), ("blue lego", None),
    ("瓶子", "bottle"), ("the bottle", "bottle"), ("red can", None), ("罐子", None), ("green bottle", None),
    ("the cup", None), ("duck and mug", None), ("screwdriver", "screwdriver"), ("螺丝刀", "screwdriver"),
    ("红色的螺丝刀", None), ("苹果", None), ("pick up the duck", None), ("purple duck", None),
]

_ROWS = [("apartment", t, b) for t, b in _APARTMENT] + [("tabletop", t, b) for t, b in _TABLETOP]


def _generated_rows() -> list[tuple[str, str, str]]:
    """Every alias and body name of every world names its own object."""
    return [(w, name, n) for w in WORLDS for n, pk in load_world(w).pickables.items() for name in (n, *pk.aliases)]


def test_the_table_has_at_least_150_rows() -> None:
    assert len(_ROWS) + len(_generated_rows()) >= 150


@pytest.mark.parametrize("world, text, body", _ROWS + _generated_rows())
def test_resolve(world, text, body) -> None:
    assert load_world(world).resolve(text) == body


@pytest.mark.parametrize("world, text, fits", [
    ("apartment", "瓶子", {G, B}), ("apartment", "the bottle", {G, B}), ("apartment", "red bottle", set()),
    ("apartment", "yellow banana", None), ("apartment", "red", None), ("tabletop", "the red one", {"mug", "lego"}),
])
def test_referents_tells_an_unbound_reference_from_no_reference(world, text, fits) -> None:
    got = load_world(world).referents(text)
    assert (None if got is None else set(got)) == fits


# REF1c (charter 3.5, "Manipulation needs a contract"): a free sentence names no object.
# These rows pinned WorldManifest.mentions, the scan that read an object out of a free
# sentence (deleted: a landmark, a part, an excluded or a negated object all mention
# one). The gate's object is now the closed form's bound slot: OUT when the sentence is
# no closed form (the manipulation gate rejects the turn), None when the form's slot
# binds nothing, else the object. The comment after a row gives its old mentions value.
OUT = "outside the closed forms"


def _gate_object(world: str, sentence: str) -> str | None:
    from vector_os_nano.vcli.cognitive.object_goal import _sentence

    form = _sentence(load_world(world), sentence)
    return OUT if form is None else form[1]


@pytest.mark.parametrize("sentence, bound", [
    ("pick up the yellow banana", OUT), ("grab the red can", R), ("把绿色瓶子拿过来", G),
    ("bring me the red can", R),
    ("can you pick up the red one", R),                                      # was set(): a closed form after all
    ("bring me the red can and the green bottle", OUT),                      # was {R, G}
    ("拿起红色的罐子", R),
    ("pick up the green can", None),                                         # a phrase fitting nothing
    ("pick up the cup", Y),                                                  # was set(): a whole slot binds
    ("put the yellow cup on the coffee table", Y), ("the infrared scanner", OUT),
    ("拿起红色的", R),                                                       # was set(): a whole slot binds
    ("grab the red can, not the green bottle", OUT), ("把红色罐子旁边的绿色瓶子拿来", OUT),  # were {R, G}
    ("pick up the pickable_bottle_blue now", B), ("拿起黄色的香蕉", OUT),
    # REF1V-01: a CJK world word inside a longer word or under a modifier the world has not
    ("拿起粉红色的罐子", OUT), ("拿起粉红的罐子", OUT), ("拿起粉红色罐头", OUT), ("拿起红色罐头", OUT),
    ("拿起绿色的瓶盖", OUT), ("拿起橙色的盒饭", OUT), ("把桌子上的粉红色罐头拿给我", OUT),
    ("拿起深红色的罐子", OUT), ("拿起橘红色的罐子", None), ("拿起黄色的杯垫", OUT), ("拿起粉红罐子", OUT),
    ("拿起红色罐子的盖子", OUT), ("拿起绿色的红罐子", None), ("拿起大的红罐子", OUT),
    ("拿起粉红色 罐子", OUT), ("把粉红色的罐子放到绿色瓶子旁边", OUT), ("pick up the green red can", None),
    # were {R}/{G}/{Y}: a phrase beside other words of a sentence is no closed form any more
    ("帮我拿红罐子吧", R), ("我要绿色的瓶子", OUT), ("拿起红色的红罐子", R),
    ("用 RynnBrain 找一下绿色瓶子", OUT), ("去绿色的瓶子那里", OUT), ("帮我拿一下红色的罐子", R),
    # K1 (REF1c-V-04): closed forms now; a location binds only when it is the object's
    # surface in the manifest ("桌子" names none of the four)
    ("把桌子上的红色罐子拿给我", None), ("把取物台上的红色罐子拿给我", R), ("去客厅把黄色杯子拿起来", Y),
    ("拿起我的红罐子", R), ("红色的罐子放到餐桌上", R), ("拿出红罐子", R), ("把红罐子提起来", R),
])
def test_a_sentence_names_an_object_only_through_its_closed_form(sentence, bound) -> None:
    assert _gate_object("apartment", sentence) == bound


@pytest.mark.parametrize("sentence, bound", [
    ("粉红色 mugs", OUT), ("拿起粉红色的马克杯", OUT), ("拿起白色的马克杯", OUT), ("拿起香蕉皮", OUT),
    ("拿起蓝色的马克杯", None), ("拿起红色的马克杯", "mug"), ("拿起香蕉", "banana"),
    ("red mugs", OUT),                                                       # was {"mug"}: no verb, no form
    ("拿起黄色的杯子和香蕉", OUT), ("把蓝色的马克杯和香蕉拿过来", OUT),
])
def test_a_sentence_names_an_object_only_through_its_closed_form_in_the_tabletop(sentence, bound) -> None:
    assert _gate_object("tabletop", sentence) == bound



# ---------------------------------------------------------------------------
# Properties over every world manifest
# ---------------------------------------------------------------------------

def _own_phrases(world, pk) -> list[str]:
    """Phrases built only from *pk*'s own words and the fillers that fit it alone."""
    out = [a for a in pk.aliases] + [f"the {a}" if a.isascii() else f"这个{a}" for a in pk.aliases]
    pairs = [(n, q) for n, q in world.pickables.items() if (q.colour, q.category) == (pk.colour, pk.category)]
    if len(pairs) == 1:
        for c in (pk.colour, *world.colours.get(pk.colour, ())):
            for k in (pk.category, *world.categories.get(pk.category, ())):
                if c.isascii() and k.isascii():
                    out += [f"{c} {k}", f"the {c} {k}s"]
                elif c.isascii():
                    out += [f"{c}{k}", f"{c} {k}子", f"那个{c}{k}"]
                else:
                    out += [f"{c}{k}", f"{c}色的{k}", f"那个{c}色{k}", f"{c}的{k}" + ("" if k.isascii() else "子")]
    return out


def _cases():
    return [(w, n, t) for w in WORLDS for n, pk in load_world(w).pickables.items() for t in _own_phrases(load_world(w), pk)]


@pytest.mark.parametrize("world, body, text", _cases())
def test_every_phrase_of_an_objects_own_words_resolves_to_it(world, body, text) -> None:
    assert load_world(world).resolve(text) == body


_FOREIGN = ["unicorn", "stool", "apple", "2", "not", "pink", "infrared", "苹果", "凳", "粉", "两"]


@pytest.mark.parametrize("world, body, text", _cases())
def test_replacing_any_one_token_with_a_foreign_word_gives_none(world, body, text) -> None:
    w = load_world(world)
    words = {e[0] for e in w._lexicon}
    foreign = [f for f in _FOREIGN if f not in words]
    tokens = text.split() if text.isascii() else [text[i] for i in range(len(text))]
    glue = " " if text.isascii() else ""
    for i in range(len(tokens)):
        for f in foreign:
            mutant = glue.join([*tokens[:i], f, *tokens[i + 1:]])
            assert w.resolve(mutant) is None, (text, mutant)


@pytest.mark.parametrize("world", WORLDS)
def test_whatever_resolve_returns_agrees_with_every_word(world) -> None:
    """Random phrases of this world's own words (and the fillers): a referent returned
    has the colour of every colour word, the category of every category word, and is
    the object of every alias (read from the manifest data, not the parser's table)."""
    w, rng = load_world(world), random.Random(7)
    words = [(a, "name", n) for n, pk in w.pickables.items() for a in (n, *pk.aliases)]
    words += [(cw, "colour", c) for c, ws in w.colours.items() for cw in (c, *ws)]
    words += [(kw, "category", k) for k, ws in w.categories.items() for kw in (k, *ws)]
    words += [(f, "filler", None) for f in ("the", "a", "this", "one", "thing", "的", "东西", "那个", "个")]
    hits = 0
    for _ in range(3000):
        picked = rng.sample(words, rng.randint(1, 4))
        text = " ".join(t for t, _, _ in picked)
        body = w.resolve(text)
        if body is None:
            continue
        hits += 1
        pk = w.pickables[body]
        for _word, kind, value in picked:
            assert {"name": body, "colour": pk.colour, "category": pk.category}.get(kind, value) == value, (text, body)
    assert hits > 50


def test_no_legitimate_name_of_any_world_is_refused() -> None:
    """(R2F2V scratch, side B) every alias and colour-plus-category of every world names its object."""
    for world in WORLDS:
        w = load_world(world)
        for name, pk in w.pickables.items():
            for a in pk.aliases:
                assert w.resolve(a) == name, (world, a)
            if sum((q.colour, q.category) == (pk.colour, pk.category) for q in w.pickables.values()) == 1:
                assert w.resolve(f"{pk.colour} {pk.category}") == name


@pytest.mark.parametrize("query", ["yellow banana", "黄色的香蕉"])
def test_the_oracle_target_of_a_banana_is_not_the_yellow_cup(query) -> None:
    """(R2F2V scratch) sim_oracles.obj(target) = world.resolve(target)."""
    assert load_world("apartment").resolve(query) is None
