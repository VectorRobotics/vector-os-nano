# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""World manifest — what a sim world contains, as data (``worlds/<id>/world.yaml``).

Rooms (free seed + floor box), doors, support surfaces, pickables (MJCF body name
-> category / colour / zh+en aliases) and the user location. A body names its
world in ``robot.yaml`` (``world: apartment``). Pure data: no MuJoCo import, so
skills, the scene build and the oracles share one description of the world.

It is NOT ground truth: it says which objects exist and what they are called,
never where they are now (that lives in ``sim.ground_truth``).
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vector_os_nano.schemas.strict import yaml_loader
from vector_os_nano.schemas.paths import manifest_folders

WORLDS_DIR = Path(__file__).resolve().parents[1] / "worlds"


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
    place_point: tuple[float, float] | None = None  # authored world XY, never inferred from missing data


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
    # The robot-free scene MJCF and where a body without a base is mounted
    # (x, y, z, heading): what sim/scene/compose.py builds the world from.
    scene: Path | None = None
    spawn: tuple[float, float, float, float] | None = None
    # What NL calls a room (``vocab.rooms``: room -> words), besides its own name.
    room_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

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

    def room_words(self) -> dict[str, frozenset[str]]:
        """``{word: the rooms it names}`` for every name ("living_room", "living room") and
        ``vocab.rooms`` alias of every room, lower-cased."""
        out: dict[str, set[str]] = {}
        for r in self.rooms:
            for w in (r, r.replace("_", " "), *self.room_aliases.get(r, ())):
                out.setdefault(w.lower(), set()).add(r)
        return {w: frozenset(rs) for w, rs in out.items()}

    def resolve_room(self, text: Any) -> str | None:
        """The room the WHOLE *text* names (one word of ``room_words`` naming exactly one
        room), or None."""
        rooms = self.room_words().get(_norm(text), frozenset())
        return next(iter(rooms)) if len(rooms) == 1 else None

    def vocabulary(self) -> list[str]:
        """One canonical English name per pickable, for detectors and the model."""
        return [p.english for p in self.pickables.values()]


def _mapping(value: Any, ctx: str, valid: tuple[str, ...] | None = None,
             required: tuple[str, ...] = ()) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
        raise WorldManifestError(f"{ctx}: expected a mapping with nonempty string keys")
    if valid is not None:
        extra = set(value) - set(valid)
        missing = set(required) - set(value)
        if extra or missing:
            raise WorldManifestError(f"{ctx}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}; "
                                     f"valid: {list(valid)}")
    return value


def _number(value: Any, ctx: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise WorldManifestError(f"{ctx}: expected a finite number")
    return float(value)


def _text(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldManifestError(f"{ctx}: expected a nonempty string")
    return value


def _words(value: Any, ctx: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise WorldManifestError(f"{ctx}: expected a list of strings")
    return tuple(_text(v, ctx) for v in value)


def _vector(value: Any, size: int, ctx: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise WorldManifestError(f"{ctx}: expected {size} finite coordinates")
    return tuple(_number(v, ctx) for v in value)


def _xy(v: Any, ctx: str) -> tuple[float, float]:
    x, y = _vector(v, 2, ctx)
    return x, y


def _box(v: Any, ctx: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _vector(v, 4, ctx)
    if x0 > x1 or y0 > y1:
        raise WorldManifestError(f"{ctx}: expected [x_min, y_min, x_max, y_max]")
    return x0, y0, x1, y1


def parse_world(raw: dict[str, Any], world_id: str, path: Path) -> WorldManifest:
    ctx = str(path)
    keys = ('schema', 'scene', 'spawn', 'rooms', 'boxes', 'doors', 'surfaces', 'pickables', 'user', 'vocab')
    raw = _mapping(raw, ctx, keys, ('schema', 'scene', 'spawn'))
    schema = raw['schema']
    if type(schema) is not int or schema != 1:
        advice = 'upgrade vector_os_nano' if type(schema) is int and schema > 1 else 'use schema: 1'
        raise WorldManifestError(f"{ctx}: schema {schema!r}; valid: [1]; {advice}")
    scene = Path(_text(raw['scene'], f'{ctx}: scene'))
    if scene.is_absolute() or '..' in scene.parts:
        raise WorldManifestError(f'{ctx}: scene must be a relative path inside the world folder')
    spawn = _mapping(raw['spawn'], f'{ctx}: spawn', ('xy', 'z', 'heading'), ('xy', 'z', 'heading'))
    spawn = (*_xy(spawn['xy'], f'{ctx}: spawn.xy'), _number(spawn['z'], f'{ctx}: spawn.z'),
             _number(spawn['heading'], f'{ctx}: spawn.heading'))
    surfaces: dict[str, Surface] = {}
    for name, value in _mapping(raw.get('surfaces', {}), f'{ctx}: surfaces').items():
        key = f'{ctx}: surfaces.{name}'
        row = _mapping(value, key, ('body', 'box', 'height', 'aliases', 'place_point'), ('body', 'box', 'height'))
        point = _xy(row['place_point'], f'{key}.place_point') if 'place_point' in row else None
        box = _box(row['box'], f'{key}.box')
        if point is not None and not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
            raise WorldManifestError(f'{key}.place_point: must lie within the declared surface box')
        surfaces[name] = Surface(name, _text(row['body'], f'{key}.body'), box,
                                 _number(row['height'], f'{key}.height'), _words(row.get('aliases', []), f'{key}.aliases'), point)
    pickables: dict[str, Pickable] = {}
    for name, value in _mapping(raw.get('pickables', {}), f'{ctx}: pickables').items():
        key = f'{ctx}: pickables.{name}'
        row = _mapping(value, key, ('category', 'colour', 'aliases', 'surface'), ('category', 'colour', 'aliases'))
        pk = Pickable(name, _text(row['category'], f'{key}.category'), _text(row['colour'], f'{key}.colour'),
                      _words(row['aliases'], f'{key}.aliases'), _text(row['surface'], f'{key}.surface') if 'surface' in row else '')
        if pk.surface and pk.surface not in surfaces:
            raise WorldManifestError(f"{key}.surface={pk.surface!r}; valid: {sorted(surfaces)}")
        pickables[name] = pk
    vocab = _mapping(raw.get('vocab', {}), f'{ctx}: vocab', ('rooms', 'colours', 'categories', 'scenery'))
    words = {kind: {name: _words(value, f'{ctx}: vocab.{kind}.{name}')
                    for name, value in _mapping(vocab.get(kind, {}), f'{ctx}: vocab.{kind}').items()}
             for kind in ('rooms', 'colours', 'categories')}
    rooms = {name: _xy(value, f'{ctx}: rooms.{name}') for name, value in _mapping(raw.get('rooms', {}), f'{ctx}: rooms').items()}
    if set(words['rooms']) - set(rooms):
        raise WorldManifestError(f"{ctx}: vocab.rooms names {sorted(set(words['rooms']) - set(rooms))}; valid: {sorted(rooms)}")
    user = raw.get('user')
    if user is not None:
        user = _xy(_mapping(user, f'{ctx}: user', ('xy',), ('xy',))['xy'], f'{ctx}: user.xy')
    return WorldManifest(
        id=world_id, rooms=rooms,
        boxes={name: _box(value, f'{ctx}: boxes.{name}') for name, value in _mapping(raw.get('boxes', {}), f'{ctx}: boxes').items()},
        doors={name: _xy(value, f'{ctx}: doors.{name}') for name, value in _mapping(raw.get('doors', {}), f'{ctx}: doors').items()},
        surfaces=surfaces, pickables=pickables, colours=words['colours'], categories=words['categories'],
        user=user, path=path, scenery=_words(vocab.get('scenery', []), f'{ctx}: vocab.scenery'),
        scene=path.parent / scene, spawn=spawn, room_aliases=words['rooms'],
    )


def world_of(agent: Any) -> WorldManifest | None:
    """The manifest of the world *agent* stands in (its config names it), or None.

    The same lookup skills use (``SkillContext.config['world']``): what the map
    says, never ground truth.
    """
    cfg = getattr(agent, "_config", None)
    world_id = cfg.get("world") if isinstance(cfg, dict) else None
    return load_world(str(world_id)) if world_id else None


def world_folders() -> dict[str, Path]:
    return manifest_folders(WORLDS_DIR, 'worlds', 'world.yaml', error=WorldManifestError)


def load_world(world_id: str) -> WorldManifest:
    """Load ``worlds/<world_id>/world.yaml``; unknown id fails loud with the valid set."""
    folders = world_folders()
    if world_id not in folders:
        raise WorldManifestError(f"unknown world {world_id!r}; valid: {sorted(folders)}")
    return load_world_file(folders[world_id] / 'world.yaml')


def load_world_file(path: str | Path) -> WorldManifest:
    """Read a world file through the same strict schema, including layout-seeding callers."""
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise WorldManifestError(f"{path}: unreadable ({exc})") from exc
    return _parse_world_text(path, text)


@functools.lru_cache(maxsize=16)
def _parse_world_text(path: Path, text: str) -> WorldManifest:
    try:
        raw = yaml.load(text, Loader=yaml_loader(path, WorldManifestError))
    except yaml.YAMLError as exc:
        raise WorldManifestError(f'{path}: unreadable ({exc})') from exc
    return parse_world(raw, path.parent.name, path)
