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


@dataclass(frozen=True)
class Surface:
    name: str
    body: str
    box: tuple[float, float, float, float]
    height: float
    aliases: tuple[str, ...] = ()  # what NL calls it ("餐桌", "dining table")


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
    # Words naming things that are NOT pickables (furniture): "red stool" is no red can.
    scenery: tuple[str, ...] = ()

    def resolve(self, text: Any) -> str | None:
        """The pickable body *text* names, or None (unknown or ambiguous).

        A body name, else the longest alias inside the text, else a colour (plus
        optional category) that fits exactly one pickable ("绿瓶" -> green bottle,
        "红色的" -> the only red object). A category alone never picks one ("罐子",
        the verb "can"), nor does a description that fits two ("瓶子") or one that
        names scenery ("红色的凳子" is not the red can).
        """
        q = str(text or "").strip().lower()
        if not q:
            return None
        for name in self.pickables:
            if q == name.lower():
                return name
        hits = [(a.lower(), n) for n, p in self.pickables.items() for a in p.aliases if _has(q, a)]
        if hits:
            best_alias, best = max(hits, key=lambda h: len(h[0]))
            # Two objects named ("the red can, not the green bottle"): ambiguous,
            # unless the other name is only part of the longer one ("cup" in "yellow cup").
            return best if all(n == best or a in best_alias for a, n in hits) else None
        colours = {c for c, ws in self.colours.items() if any(_has(q, w) for w in ws)}
        cats = {c for c, ws in self.categories.items() if any(_has(q, w) for w in ws)}
        if not colours or any(_has(q, w) for w in self.scenery):
            return None
        cands = [
            n for n, p in self.pickables.items()
            if (not colours or p.colour in colours) and (not cats or p.category in cats)
        ]
        return cands[0] if len(cands) == 1 else None

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
        """One canonical English name per pickable (its first ASCII alias), for detectors."""
        out = []
        for p in self.pickables.values():
            out.append(next((a for a in p.aliases if a.isascii()), f"{p.colour} {p.category}"))
        return out


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
