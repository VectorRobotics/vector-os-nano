"""VisualVerifier — VLM-based verification fallback for VGG steps.

Called when GoalVerifier.verify() returns False. Takes a photo via the robot's
camera, sends it to VLM, and checks if the step's goal was actually achieved.

Only triggers under specific conditions (not every step).
Gracefully degrades when VLM is unavailable.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_VLM_TIMEOUT_SEC: float = 10.0

# Keywords in strategy/description that indicate perception steps
_PERCEPTION_KEYWORDS: frozenset[str] = frozenset({
    "look", "observe", "detect", "find", "scan", "check", "see",
    "看", "观察", "检测", "找", "检查", "扫描",
})

# Keywords in verify expressions that suggest visual confirmation would help
_VISUAL_VERIFY_FUNCTIONS: frozenset[str] = frozenset({
    "describe_scene", "detect_objects", "identify_room",
})


@dataclass(frozen=True)
class VisualVerifyResult:
    """Result of visual verification attempt."""

    triggered: bool       # True if verification was attempted
    success: bool         # True if VLM confirmed the goal
    description: str      # VLM scene description (or "")
    duration_sec: float   # Time taken
    error: str            # Error message if failed


def should_verify(
    sub_goal_name: str,
    sub_goal_description: str,
    strategy: str,
    verify_expr: str,
    verify_result: bool,
) -> bool:
    """Decide whether to trigger visual verification.

    Returns True when:
    1. verify_result is False (primary verify failed — second opinion needed)
       AND at least one of:
       a. strategy or description contains perception keywords
       b. verify expression references visual functions (describe_scene, detect_objects)

    Returns False when:
    - verify_result is True (no need for second opinion)
    - No perception-related context (visual check wouldn't help)
    """
    if verify_result:
        return False

    # Check perception keywords in strategy or description
    combined_text = f"{strategy} {sub_goal_description}".lower()
    for keyword in _PERCEPTION_KEYWORDS:
        if keyword in combined_text:
            return True

    # Check for visual function references in verify expression
    for fn_name in _VISUAL_VERIFY_FUNCTIONS:
        if fn_name in verify_expr:
            return True

    return False


def verify_visual(
    agent: Any,
    sub_goal_description: str,
    verify_expr: str,
) -> VisualVerifyResult:
    """Attempt visual verification using robot camera + VLM.

    Steps:
    1. Check agent has camera (agent._base) and VLM (agent._vlm)
    2. Capture image: agent._base.get_camera_frame()
    3. If verify_expr mentions detect_objects → vlm.find_objects(frame, extract_query(verify_expr))
       If verify_expr mentions describe_scene → vlm.describe_scene(frame)
       Otherwise → vlm.describe_scene(frame) and check if description is relevant
    4. Parse VLM result to determine success
    5. Return VisualVerifyResult

    On any error or timeout → return VisualVerifyResult(triggered=True, success=False, error=...)
    If VLM unavailable → return VisualVerifyResult(triggered=False, success=False, error="VLM not available")

    Args:
        agent: Agent instance with _base (camera) and _vlm (VLM perception)
        sub_goal_description: Human-readable step description
        verify_expr: The verify expression that failed

    Returns:
        VisualVerifyResult
    """
    # Check VLM availability
    vlm = getattr(agent, "_vlm", None)
    if vlm is None:
        return VisualVerifyResult(
            triggered=False,
            success=False,
            description="",
            duration_sec=0.0,
            error="VLM not available",
        )

    # Check camera availability
    base = getattr(agent, "_base", None)
    if base is None:
        return VisualVerifyResult(
            triggered=False,
            success=False,
            description="",
            duration_sec=0.0,
            error="Camera base not available",
        )

    t_start = time.monotonic()

    try:
        frame = base.get_camera_frame()

        # Dispatch based on verify expression content
        if "detect_objects" in verify_expr:
            query = _extract_query_from_verify(verify_expr)
            objects = vlm.find_objects(frame, query)
            success = len(objects) > 0
            description = (
                ", ".join(getattr(o, "name", str(o)) for o in objects)
                if objects
                else ""
            )
        elif "describe_scene" in verify_expr:
            scene = vlm.describe_scene(frame)
            raw_description = _extract_scene_text(scene)
            success = _check_description_relevance(
                raw_description, sub_goal_description, verify_expr
            )
            description = raw_description
        else:
            # Fallback: describe scene and check relevance
            scene = vlm.describe_scene(frame)
            raw_description = _extract_scene_text(scene)
            success = _check_description_relevance(
                raw_description, sub_goal_description, verify_expr
            )
            description = raw_description

        duration = time.monotonic() - t_start
        return VisualVerifyResult(
            triggered=True,
            success=success,
            description=description,
            duration_sec=duration,
            error="",
        )

    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - t_start
        logger.warning("VisualVerifier: VLM call failed: %s", exc)
        return VisualVerifyResult(
            triggered=True,
            success=False,
            description="",
            duration_sec=duration,
            error=str(exc),
        )


def _extract_query_from_verify(verify_expr: str) -> str:
    """Extract object query from verify expression.

    Examples:
        "len(detect_objects('cup')) > 0" → "cup"
        "detect_objects('red ball')" → "red ball"
        "'table' in describe_scene()" → "table"
        "nearest_room() == 'kitchen'" → ""  (not a detection query)

    Uses regex to find string literals inside detect_objects() or describe_scene() calls.
    Returns "" if no query found.
    """
    # Match string inside detect_objects(...) — single or double quotes
    match = re.search(r"detect_objects\(['\"]([^'\"]+)['\"]\)", verify_expr)
    if match:
        return match.group(1)

    # Match quoted string before or after describe_scene — e.g. 'table' in describe_scene()
    match = re.search(r"['\"]([^'\"]+)['\"]\s+in\s+describe_scene", verify_expr)
    if match:
        return match.group(1)

    # Also match describe_scene() with a string argument
    match = re.search(r"describe_scene\(['\"]([^'\"]+)['\"]\)", verify_expr)
    if match:
        return match.group(1)

    return ""


def _check_description_relevance(
    description: str,
    sub_goal_description: str,
    verify_expr: str,
) -> bool:
    """Check if VLM scene description confirms the sub-goal.

    Simple keyword overlap check:
    1. Extract nouns/keywords from sub_goal_description and verify_expr
    2. Check if any appear in the VLM description
    3. Return True if overlap suggests goal achieved

    This is a heuristic — not perfect, but better than nothing.
    """
    if not description:
        return False

    description_lower = description.lower()

    # Combine goal description and verify expression as keyword sources
    combined_source = f"{sub_goal_description} {verify_expr}".lower()

    # Extract meaningful words (length > 2, skip common stopwords)
    _STOPWORDS = frozenset({
        "the", "and", "for", "in", "is", "of", "to", "a", "an", "on",
        "at", "be", "it", "or", "are", "was", "has", "len", "not",
        "this", "that", "with", "from", "by", "as", "if", "than",
        "nearest_room", "describe_scene", "detect_objects", "identify_room",
    })

    # Split on non-alphanumeric boundaries, filter stopwords and short tokens
    tokens = re.findall(r"[a-z\u4e00-\u9fff]+", combined_source)
    keywords = {
        t for t in tokens
        if len(t) > 2 and t not in _STOPWORDS
    }

    # Also extract keywords from verify_expr string literals
    for literal in re.findall(r"['\"]([^'\"]+)['\"]", verify_expr):
        literal_tokens = re.findall(r"[a-z\u4e00-\u9fff]+", literal.lower())
        keywords.update(t for t in literal_tokens if len(t) > 1)

    if not keywords:
        return False

    return any(kw in description_lower for kw in keywords)


def _extract_scene_text(scene: Any) -> str:
    """Pull text content from a scene description object.

    Tries .details, .summary, then str() as fallbacks.
    """
    details = getattr(scene, "details", None)
    if details and isinstance(details, str):
        return details

    summary = getattr(scene, "summary", None)
    if summary and isinstance(summary, str):
        return summary

    return str(scene) if scene is not None else ""
