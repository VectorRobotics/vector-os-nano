# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""World manifest — what a sim world contains, as data (``config/worlds/<id>.yaml``).

Rooms (free seed + floor box), doors, support surfaces, pickables (MJCF body name
-> category / colour / zh+en aliases) and the user location. A body names its
world in ``robot.yaml`` (``world: apartment``). Pure data: no MuJoCo import, so
skills, the scene build and the oracles share one description of the world.

It is NOT ground truth: it says which objects exist and what they are called,
never where they are now (that lives in ``sim.ground_truth``).
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

WORLDS_DIR = Path(__file__).resolve().parents[3] / "config" / "worlds"


class WorldManifestError(ValueError):
    """A world manifest is missing or malformed (fail loud, Rule 8)."""


@dataclass(frozen=True)
class Pickable:
    name: str
    category: str
    colour: str
    aliases: tuple[str, ...]
    surface: str = ""  # the support it rests on at scene load

    @property
    def english(self) -> str:
        """Its canonical English name (its first ASCII alias), for detectors and the model."""
        return next((a for a in self.aliases if a.isascii()), f"{self.colour} {self.category}")


@dataclass(frozen=True)
class Surface:
    name: str
    body: str
    box: tuple[float, float, float, float]
    height: float
    aliases: tuple[str, ...] = ()  # what NL calls it ("餐桌", "dining table")


# The ONE closed list of words a reference may carry besides this world's own: articles
# and demonstratives, and the generic heads that (like 的) let a colour refer ("the red one").
_FILLERS = {
    "det": ("the", "a", "an", "this", "that", "my", "这个", "那个", "这", "那"),
    "head": ("one", "thing", "object", "item", "东西", "物体", "物品", "个", "的"),
}


def _norm(text: Any) -> str:
    return text.strip().lower() if isinstance(text, str) else ""


def _forms(word: str, zh_suffix: str) -> tuple[str, ...]:
    """*word* and its suffixed forms: an English plural ("cans", "boxes") or the CJK suffix (红色, 罐子)."""
    return (word, f"{word}s", f"{word}es") if word.isascii() else (word, word + zh_suffix)


def _match(text: str, i: int, word: str) -> int | None:
    """Where *word* ends when it starts at *text*[i], else None; an ASCII word only on
    word boundaries ("scan", "candle" and "infrared" hold no "can" or "red")."""
    j = i + len(word)
    if not text.startswith(word, i):
        return None
    if word.isascii() and any(0 <= k < len(text) and _ascii_word(text[k]) for k in (i - 1, j)):
        return None
    return j


def _ascii_word(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch == "_")


def _has(text: str, word: str) -> bool:
    """*word* occurs in *text*: on word boundaries for ASCII words, as a substring for CJK."""
    w = word.lower()
    if w.isascii():
        return re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text) is not None
    return w in text


@dataclass(frozen=True)
class WorldManifest:
    id: str
    rooms: dict[str, tuple[float, float]]
    boxes: dict[str, tuple[float, float, float, float]]
    doors: dict[str, tuple[float, float]]
    surfaces: dict[str, Surface]
    pickables: dict[str, Pickable]
    colours: dict[str, tuple[str, ...]]
    categories: dict[str, tuple[str, ...]]
    user: tuple[float, float] | None
    path: Path
    # Words for furniture: a place goal's destination slot ("on the table") that binds no surface.
    scenery: tuple[str, ...] = ()

    @functools.cached_property
    def _lexicon(self) -> tuple[tuple[str, str, frozenset[str]], ...]:
        """``(word, kind, the pickables it fits)`` for every word of a reference in this
        world: kind ``name`` (a body name or an alias), ``category`` (with a plural or 子),
        ``colour`` (with 色), ``det`` / ``head`` (the fillers); longest word first."""
        p = self.pickables
        out = {(w.lower(), "name", frozenset({n})) for n, pk in p.items() for w in (n, *pk.aliases)}
        for c in {pk.colour for pk in p.values()} | set(self.colours):
            fits = frozenset(n for n, pk in p.items() if pk.colour == c)
            out |= {(f, "colour", fits) for w in (c, *self.colours.get(c, ())) for f in _forms(w.lower(), "色")}
        for k in {pk.category for pk in p.values()} | set(self.categories):
            fits = frozenset(n for n, pk in p.items() if pk.category == k)
            out |= {(f, "category", fits) for w in (k, *self.categories.get(k, ())) for f in _forms(w.lower(), "子")}
        out |= {(w, kind, frozenset(p)) for kind, ws in _FILLERS.items() for w in ws}
        return tuple(sorted(out, key=lambda e: (-len(e[0]), e[0], e[1])))

    def referents(self, text: Any) -> frozenset[str] | None:
        """The pickables a reference fits, or None when *text* is no reference here.

        *text* is a reference only when every character of it is a word of this world
        (body name, alias, colour, category) or a filler (``_FILLERS``), and it has a
        head: a name or a category, or a colour marked by a generic head or 的 ("the red
        one", 红色的; a lone colour word is a noun: "the orange"). Anything left over (a
        noun of another world, a number, a negation, a spatial word, a second clause)
        makes it no reference. The pickables returned agree with EVERY word ("blue mug"
        fits nothing): one is the referent; none or several leave it unbound.
        """
        q = _norm(text)
        if not q:
            return None
        memo: dict[int, set[tuple[frozenset[str], bool, bool, bool]]] = {}

        def rest(i: int) -> set[tuple[frozenset[str], bool, bool, bool]]:
            # (fits, a head, a colour, a marker) of every reading of q[i:]
            while i < len(q) and q[i].isspace():
                i += 1
            if i == len(q):
                return {(frozenset(self.pickables), False, False, False)}
            if i not in memo:
                memo[i] = {(fits & f, h or kind in ("name", "category"), c or kind == "colour", m or kind == "head")
                           for w, kind, fits in self._lexicon if (j := _match(q, i, w)) is not None
                           for f, h, c, m in rest(j)}
            return memo[i]

        reads = [fits for fits, head, colour, marker in rest(0) if head or (colour and marker)]
        return frozenset().union(*reads) if reads else None

    def resolve(self, text: Any) -> str | None:
        """The pickable body the WHOLE *text* names (``referents``: exactly one), or None."""
        fits = self.referents(text)
        return next(iter(fits)) if fits and len(fits) == 1 else None

    def mentions(self, sentence: Any) -> frozenset[str]:
        """The pickables named inside a free *sentence* by complete noun phrases only: a
        body name or an alias, or a colour word attached to a category word ("red can",
        红色的罐子). A lone colour, a category alone and a generic head are references
        only as a whole slot (in a sentence "can" is a verb and 的 an attributive)."""
        q, spans = _norm(sentence), []
        for i in range(len(q)):
            for w, kind, fits in self._lexicon:
                if (j := _match(q, i, w)) is None or kind not in ("name", "colour"):
                    continue
                if kind == "name":
                    spans.append((i, j, fits))
                    continue
                k = j + q.startswith("的", j)
                k += q.startswith(" ", k)
                spans += [(i, e, fits & f) for w2, kind2, f in self._lexicon
                          if kind2 == "category" and (e := _match(q, k, w2)) is not None]
        inner = {s for s in spans for t in spans if t[0] <= s[0] and s[1] <= t[1] and t[:2] != s[:2]}
        return frozenset().union(*(s[2] for s in spans if s not in inner))

    def resolve_surface(self, text: Any) -> str | None:
        """The support surface *text* names (its name, or the longest alias inside the
        text), or None (none named, or two: "the table" in a house of tables)."""
        q = str(text or "").strip().lower()
        hits = [(w.lower(), n) for n, s in self.surfaces.items()
                for w in (n, n.replace("_", " "), *s.aliases) if q and _has(q, w)]
        if not hits:
            return None
        best_word, best = max(hits, key=lambda h: len(h[0]))
        return best if all(n == best or w in best_word for w, n in hits) else None

    def vocabulary(self) -> list[str]:
        """One canonical English name per pickable, for detectors and the model."""
        return [p.english for p in self.pickables.values()]


def _xy(v: Any, ctx: str) -> tuple[float, float]:
    try:
        x, y = (float(c) for c in v)
    except (TypeError, ValueError) as exc:
        raise WorldManifestError(f"{ctx}: expected [x, y], got {v!r}") from exc
    return x, y


def _box(v: Any, ctx: str) -> tuple[float, float, float, float]:
    try:
        b = tuple(float(c) for c in v)
    except (TypeError, ValueError) as exc:
        raise WorldManifestError(f"{ctx}: expected [x_min, y_min, x_max, y_max], got {v!r}") from exc
    if len(b) != 4 or b[0] > b[2] or b[1] > b[3]:
        raise WorldManifestError(f"{ctx}: expected [x_min, y_min, x_max, y_max], got {v!r}")
    return b  # type: ignore[return-value]


def parse_world(raw: dict[str, Any], world_id: str, path: Path) -> WorldManifest:
    ctx = f"config/worlds/{world_id}.yaml"
    surfaces: dict[str, Surface] = {}
    for name, s in (raw.get("surfaces") or {}).items():
        try:
            surfaces[str(name)] = Surface(
                str(name), str(s["body"]), _box(s["box"], f"{ctx}: surfaces.{name}.box"),
                float(s["height"]), tuple(str(a) for a in s.get("aliases") or ()),
            )
        except (KeyError, TypeError) as exc:
            raise WorldManifestError(f"{ctx}: surfaces.{name} needs body, box, height ({exc})") from exc
    pickables: dict[str, Pickable] = {}
    for name, p in (raw.get("pickables") or {}).items():
        try:
            aliases = tuple(str(a) for a in p["aliases"])
            pk = Pickable(str(name), str(p["category"]), str(p["colour"]), aliases, str(p.get("surface", "")))
        except (KeyError, TypeError) as exc:
            raise WorldManifestError(
                f"{ctx}: pickables.{name} needs category, colour, aliases ({exc})") from exc
        if pk.surface and pk.surface not in surfaces:
            raise WorldManifestError(
                f"{ctx}: pickables.{name}.surface={pk.surface!r}; surfaces: {sorted(surfaces)}")
        pickables[pk.name] = pk
    vocab = raw.get("vocab") or {}
    user = raw.get("user")
    return WorldManifest(
        id=world_id,
        rooms={str(k): _xy(v, f"{ctx}: rooms.{k}") for k, v in (raw.get("rooms") or {}).items()},
        boxes={str(k): _box(v, f"{ctx}: boxes.{k}") for k, v in (raw.get("boxes") or {}).items()},
        doors={str(k): _xy(v, f"{ctx}: doors.{k}") for k, v in (raw.get("doors") or {}).items()},
        surfaces=surfaces,
        pickables=pickables,
        colours={str(k): tuple(str(w) for w in v) for k, v in (vocab.get("colours") or {}).items()},
        categories={str(k): tuple(str(w) for w in v) for k, v in (vocab.get("categories") or {}).items()},
        user=_xy(user["xy"], f"{ctx}: user.xy") if user else None,
        path=path,
        scenery=tuple(str(w) for w in (vocab.get("scenery") or ())),
    )


def world_of(agent: Any) -> WorldManifest | None:
    """The manifest of the world *agent* stands in (its config names it), or None.

    The same lookup skills use (``SkillContext.config['world']``): what the map
    says, never ground truth.
    """
    cfg = getattr(agent, "_config", None)
    world_id = cfg.get("world") if isinstance(cfg, dict) else None
    return load_world(str(world_id)) if world_id else None


@functools.lru_cache(maxsize=8)
def load_world(world_id: str) -> WorldManifest:
    """Load ``config/worlds/<world_id>.yaml``; unknown id fails loud with the valid set."""
    path = WORLDS_DIR / f"{world_id}.yaml"
    if not path.is_file():
        valid = sorted(p.stem for p in WORLDS_DIR.glob("*.yaml"))
        raise WorldManifestError(f"unknown world {world_id!r}; valid: {valid}")
    return parse_world(yaml.safe_load(path.read_text()) or {}, world_id, path)
