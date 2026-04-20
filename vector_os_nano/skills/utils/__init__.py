"""Shared helpers for skills.

Exports:
    label_to_en_query(label) -> str | None
        Convert a possibly-Chinese/mixed object label into an English VLM
        query, using the color normaliser from pick_top_down and a small
        noun map. Returns None for empty/None input.
"""
from __future__ import annotations

from vector_os_nano.skills.pick_top_down import _normalise_color_keyword

# Small Chinese noun map — extend as needed.
# Keys sorted longest-first to avoid prefix collisions (e.g. "罐子" before "罐").
_CN_NOUN_MAP: dict[str, str] = {
    "瓶子": "bottle",
    "杯子": "cup",
    "碗": "bowl",
    "盘子": "plate",
    "罐子": "can",
    "盒子": "box",
    "球": "ball",
}


def label_to_en_query(label: str | None) -> str | None:
    """Convert a CN/mixed label to an English VLM query.

    Steps:
    1. If empty/None/whitespace → return None.
    2. Strip "的" (possessive) from anywhere in the string.
    3. Apply _normalise_color_keyword (CN colors → EN, in place).
    4. Replace known CN nouns via _CN_NOUN_MAP (longest key first).
    5. Collapse whitespace, lowercase English parts, return.

    Examples::

        label_to_en_query("蓝色瓶子")   # "blue bottle"
        label_to_en_query("红色的杯子") # "red cup"
        label_to_en_query("bottle")     # "bottle"
        label_to_en_query(None)         # None
        label_to_en_query("")           # None
        label_to_en_query("blue 瓶子")  # "blue bottle"
        label_to_en_query("奇怪的东西") # "奇怪东西"
        label_to_en_query("all objects")# "all objects"
    """
    if label is None:
        return None
    s = label.strip()
    if not s:
        return None

    # Strip the Chinese possessive "的"
    s = s.replace("的", "")

    # Apply color normaliser; returns modified string or None if no color keyword
    coloured = _normalise_color_keyword(s)
    s = coloured if coloured is not None else s

    # Apply noun map — longest key first to avoid prefix collisions
    for cn, en in sorted(_CN_NOUN_MAP.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(cn, " " + en)

    # Collapse whitespace and lowercase
    s = " ".join(s.split()).lower()
    return s or None
