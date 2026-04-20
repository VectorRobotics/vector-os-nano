"""Shared helpers for skills.

Exports:
    label_to_en_query(label) -> str | None
        Convert a possibly-Chinese/mixed object label into an English VLM
        query, using the color normaliser from pick_top_down and a small
        noun map. Returns None for empty/None input.

    run_autodetect_retry(params, context, log_tag="SKILL") -> int
        On a world_model resolve miss, invoke DetectSkill with a
        CN→EN translated query. Side-effect: DetectSkill writes
        discovered ObjectStates into context.world_model. Returns the
        number of objects DetectSkill reports; 0 if perception is
        unavailable, query is empty, or DetectSkill crashed.
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.skills.pick_top_down import _normalise_color_keyword

logger = logging.getLogger(__name__)

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


def run_autodetect_retry(
    params: dict,
    context: Any,
    log_tag: str = "SKILL",
) -> int:
    """Invoke DetectSkill with a translated query to populate world_model.

    Intended as a one-shot retry mechanism for pick/place skills whose
    initial ``_resolve_target`` missed. No-ops gracefully if perception
    or calibration is unavailable, or if the query cannot be derived.

    Args:
        params: Skill params dict; must contain ``object_label`` and/or
                ``object_id`` for query derivation.
        context: SkillContext (must expose ``perception``, ``calibration``,
                 and ``world_model``).
        log_tag: Short prefix for log lines (e.g. "MOBILE-PICK",
                 "PICK-TD") so callers can be traced.

    Returns:
        Number of objects DetectSkill reports (from its ``result_data.count``).
        Returns 0 whenever perception/calibration is missing, query is
        empty, DetectSkill is unsuccessful, or DetectSkill raises.
    """
    if context.perception is None or context.calibration is None:
        return 0

    query_raw = params.get("object_label") or params.get("object_id")
    en_query = label_to_en_query(query_raw)
    if not en_query:
        return 0

    logger.info("[%s] world_model miss; auto-detect query=%r", log_tag, en_query)
    try:
        # Lazy import avoids circular dependency at module load
        from vector_os_nano.skills.detect import DetectSkill
        det_result = DetectSkill().execute({"query": en_query}, context)
    except Exception as exc:
        logger.warning("[%s] auto-detect crashed: %s", log_tag, exc)
        return 0

    if det_result is None or not det_result.success:
        return 0
    count = int(det_result.result_data.get("count", 0))
    return count
