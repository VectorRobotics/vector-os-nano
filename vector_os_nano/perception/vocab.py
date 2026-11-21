# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""One zh/en object vocabulary (nouns + colours), built from the world manifests.

Every ``worlds/*/world.yaml`` contributes its ``vocab`` colour and category words,
plus each pickable alias that names its object without a category word ("香蕉" ->
banana). The grounding-dino prompt and colour parsing read this ONE table, so
"绿色的瓶子", "green bottle" and "绿瓶" mean the same thing everywhere. It decides no
target in a world with a manifest: there the manifest's own ``referents`` does.
English words match on word boundaries ("infrared" names no colour), CJK words as
substrings. Pure data: no MuJoCo, no torch.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from vector_os_nano.schemas.world import WORLDS_DIR, _has, load_world

# Words that never name the object: articles, pronouns and the verbs of a request.
_STOP = frozenset(
    "a an the this that these those it its please me my you your i to up down for of "
    "on in at with and or from pick grab take get bring fetch find locate detect look "
    "search show see go hold lift can't".split()
)
# Words that name "something" without saying what: the generic prompt, not a noun.
_GENERIC = frozenset({"object", "objects", "thing", "things", "something", "everything",
                      "all", "item", "items", "stuff"})
_GENERIC_ZH = ("东西", "物体", "物品")


@dataclass(frozen=True)
class Vocab:
    """``(word, canonical)`` pairs, longest word first, for colours and nouns."""

    colours: tuple[tuple[str, str], ...]
    nouns: tuple[tuple[str, str], ...]

    @staticmethod
    def _find(text: str, table: tuple[tuple[str, str], ...]) -> str | None:
        q = str(text or "").lower()
        return next((c for w, c in table if _has(q, w)), None)

    def colour(self, text: str | None) -> str | None:
        """The canonical (English) colour *text* names, or None."""
        return self._find(text or "", self.colours)

    def noun(self, text: str | None) -> str | None:
        """The canonical (English) object noun *text* names, or None."""
        return self._find(text or "", self.nouns)

    def without_colours(self, text: str | None) -> str:
        """*text* with its colour words removed ("red unicorn" -> " unicorn")."""
        q = str(text or "").lower()
        for w, _ in self.colours:
            q = re.sub(rf"(?<![a-z]){re.escape(w)}(?![a-z])", " ", q) if w.isascii() else q.replace(w, " ")
        return q

    def _words(self, q: str) -> list[str]:
        """The content words of an ASCII query (stop words dropped)."""
        return [w for w in re.findall(r"[a-z0-9]+", q) if w not in _STOP]

    def english(self, text: str | None) -> str | None:
        """An English noun phrase for *text* ("绿色的瓶子" -> "green bottle").

        ``""`` means "something" (empty, deictic or generic: the caller's generic
        prompt). None means *text* is CJK the vocabulary cannot translate ("拿那个苹果"):
        the caller must fail loud rather than hand a detector untranslated text.
        """
        q = str(text or "").strip().lower()
        if not q:
            return ""
        colour, noun = self.colour(q), self.noun(q)
        if noun:
            return f"{colour} {noun}" if colour else noun
        if q.isascii():
            colour_words = {w for w, _ in self.colours}
            words = self._words(q)
            if all(w in _GENERIC or w in colour_words for w in words):
                return f"{colour} object" if colour else ""
            return " ".join(words)
        if colour:
            return f"{colour} object"
        if any(g in q for g in _GENERIC_ZH):
            return ""
        return None


def _world_ids() -> list[str]:
    return sorted(p.parent.name for p in WORLDS_DIR.glob("*/world.yaml"))


@functools.lru_cache(maxsize=1)
def vocab() -> Vocab:
    """The vocabulary of every world manifest in ``worlds/*/world.yaml`` (their colour and
    category words and aliases)."""
    colours: dict[str, str] = {}
    nouns: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for wid in _world_ids():
        w = load_world(wid)
        for colour, ws in w.colours.items():
            for word in (colour, *ws):
                colours.setdefault(word.lower(), colour)
        for cat, ws in w.categories.items():
            for word in (cat, *ws):
                nouns.setdefault(word.lower(), cat)
        for p in w.pickables.values():
            for a in (p.category, *p.aliases):
                aliases.setdefault(a.lower(), p.category)
    cat_words = tuple(nouns)
    for a, cat in aliases.items():  # "香蕉" names a banana; "绿色的瓶子" is a bottle by 瓶
        if not any(_has(a, w) for w in cat_words):
            nouns.setdefault(a, cat)

    def longest_first(d: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(d.items(), key=lambda kv: -len(kv[0])))

    return Vocab(colours=longest_first(colours), nouns=longest_first(nouns))


def is_reference(text: str) -> bool:
    """*text* is an object reference in the words of some world manifest (every
    character accounted for: ``WorldManifest.referents``). For a robot with no manifest
    of its own, which must not hand a detector CJK it cannot translate whole."""
    return any(load_world(w).referents(text) is not None for w in _world_ids())
