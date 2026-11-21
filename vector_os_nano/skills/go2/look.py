# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Look skills for a camera-bearing base: what is around, where, in which room.

- LookSkill: the detector looks for the world's own objects (its manifest
  vocabulary) on ONE RGB-D snapshot and every hit that localizes goes on the map;
  a VLM, when there is one, adds a caption, a room hint and more names to look for.
- DescribeSceneSkill: a detailed VLM description (optionally of one query).

Both go through :func:`observe_scene`, which explore's discovery uses too. Only
LOCALIZED objects are ever stored (never a name at a made-up position); the room
is the floor box the robot stands in, the VLM's room label only a hint resolved
against the known rooms; looking never counts as visiting a room. A name the
detector or segmenter could not check is reported with its cause, never as
"nothing localized"; when none could be checked the look fails (perception_failed).
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.base import failure_cause
from vector_os_nano.perception.object_localizer import localize_objects_3d

logger = logging.getLogger(__name__)


class ObserveFailed(Exception):
    """observe_scene could not look (``code``: camera_failed / vlm_failed / perception_failed)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def observe_scene(
    base: Any,
    perception: Any,
    memory: Any,
    vlm: Any,
    names: list[str],
    *,
    query: str | None = None,
) -> dict[str, Any]:
    """Look once and record what localized; returns the observation as result_data.

    One snapshot (``perception.capture()``, else the base's RGB frame with nothing
    to localize against). The detector looks for *names* plus whatever the VLM
    names (``query`` asks the VLM for that only). Localized objects go into
    *memory* via ``observe_with_viewpoint``; un-localized names are reported under
    ``unlocalized`` and never stored; names that could not be checked under
    ``failed`` (``{name: cause}``). Raises :class:`ObserveFailed` when none could be.
    """
    try:
        frame = perception.capture() if callable(getattr(perception, "capture", None)) else None
        rgb = frame.rgb if frame is not None else base.get_camera_frame()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[LOOK] camera capture raised: %s: %s", type(exc).__name__, exc)
        raise ObserveFailed("camera_failed", f"Camera capture failed: {failure_cause(exc)}") from exc
    pos, heading = base.get_position(), float(base.get_heading())

    scene = room_id = None
    seen: dict[str, Any] = {}  # VLM name -> its DetectedObject
    if vlm is not None:
        try:
            if query:
                seen = {o.name: o for o in vlm.find_objects(rgb, query) if o.name}
            else:
                scene, room_id = vlm.describe_scene(rgb), vlm.identify_room(rgb)
                seen = {o.name: o for o in scene.objects if o.name}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LOOK] VLM raised: %s: %s", type(exc).__name__, exc)
            raise ObserveFailed("vlm_failed", f"VLM inference failed: {failure_cause(exc)}") from exc

    asked = list(dict.fromkeys([*names, *seen]))
    found, failed = localize_objects_3d(perception, asked, frame) if frame is not None and asked else ([], {})
    if asked and len(failed) == len(asked):
        raise ObserveFailed("perception_failed", f"Perception could not run for any of the {len(asked)} "
                            "names, so nothing was checked (this is not an empty room): "
                            f"{'; '.join(dict.fromkeys(failed.values()))}")
    x, y = float(pos[0]), float(pos[1])
    room = _room_of(memory, x, y, getattr(room_id, "room", None))
    summary = scene.summary if scene is not None else (
        f"Seen (detector{', no VLM' if vlm is None else ''}): "
        f"{', '.join(r.label for r in found) or 'nothing localized'}."
    )
    if failed:
        summary += f" {len(failed)} of {len(asked)} names could not be checked: {_causes(failed)}."
    if memory is not None and hasattr(memory, "observe_with_viewpoint"):
        try:
            memory.observe_with_viewpoint(
                room, x, y, heading, [],
                # only a scene caption describes the room (never a query, never a list)
                scene.summary if scene is not None and not query else "",
                detected_objects=[(r.label, r.x, r.y, r.z) for r in found],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LOOK] spatial memory update failed: %s", exc)
    localized = {r.label for r in found}
    return {
        "room": room,
        "summary": summary,
        "details": getattr(scene, "details", ""),
        "room_confidence": float(getattr(room_id, "confidence", 0.0) or 0.0),
        "room_reasoning": getattr(room_id, "reasoning", ""),
        "objects": [
            {
                "name": r.label,
                "description": getattr(seen.get(r.label), "description", ""),
                "confidence": round(r.score, 3),
                "world_x": r.x, "world_y": r.y, "world_z": r.z,
                "segmenter": r.segmenter,
            }
            for r in found
        ],
        "unlocalized": [n for n in seen if n not in localized and n not in failed],
        "failed": failed,
    }


def _causes(failed: dict[str, str]) -> str:
    """'green bottle, red can: the detector failed: <cause>; ...' (one clause per cause)."""
    by_cause: dict[str, list[str]] = {}
    for name, cause in failed.items():
        by_cause.setdefault(cause, []).append(name)
    return "; ".join(f"{', '.join(names)}: {cause}" for cause, names in by_cause.items())


def _room_of(memory: Any, x: float, y: float, hint: str | None) -> str:
    """The room (x, y) stands in; the VLM's *hint* only when it names a known room."""
    from vector_os_nano.skills.navigate import _detect_current_room, _resolve_room

    inside = memory.room_at(x, y) if callable(getattr(memory, "room_at", None)) else None
    if inside:
        return inside
    if hint and hint != "unknown" and memory is not None:
        resolved = _resolve_room(hint, memory)
        if resolved:
            return resolved
    return _detect_current_room(x, y, sg=memory)


def _world_vocabulary(context: SkillContext) -> list[str]:
    """The names of the objects in this world (its manifest), or [] outside a sim world."""
    world_id = (context.config or {}).get("world")
    if not world_id:
        return []
    from vector_os_nano.schemas.world import load_world

    return load_world(world_id).vocabulary()


@skill(requires=('base', 'camera'),
    aliases=["look", "看", "看看", "看一下", "看一看", "what do you see", "describe"],
    actuates=False,
)
class LookSkill:
    """Look around: localize the world's objects (and the VLM's) on one snapshot."""

    name: str = "look"
    description: str = (
        "Look at what the camera sees: the detector localizes the world's objects "
        "(and a VLM, if configured, captions the scene and suggests the room); every "
        "object that localizes goes on the map with its world position."
    )
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"scene_observed": True}
    failure_modes: list[str] = ["no_base", "no_vlm", "camera_failed", "vlm_failed", "perception_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(success=False, error_message="No base connected",
                               diagnosis_code="no_base")
        vlm = context.services.get("vlm")
        vocab = _world_vocabulary(context)
        if vlm is None and not (vocab and context.perception is not None):
            return SkillResult(
                success=False, diagnosis_code="no_vlm",
                error_message="Nothing to look with: no VLM, and no detector over a world vocabulary",
            )
        try:
            data = observe_scene(context.base, context.perception,
                                 context.services.get("spatial_memory"), vlm, vocab)
        except ObserveFailed as exc:
            logger.error("[LOOK] %s", exc)
            return SkillResult(success=False, error_message=str(exc), diagnosis_code=exc.code)
        logger.info("[LOOK] room=%s localized=%d %s", data["room"], len(data["objects"]),
                    [o["name"] for o in data["objects"]])
        return SkillResult(success=True, result_data=data)


@skill(requires=('base', 'camera'),
    aliases=["describe scene", "描述场景", "描述环境"],
    actuates=False,
)
class DescribeSceneSkill:
    """Get a detailed VLM description of the current scene."""

    name: str = "describe_scene"
    description: str = "Get a detailed VLM description of the current scene."
    parameters: dict = {
        "query": {
            "type": "string",
            "required": False,
            "description": "Optional: what to look for",
        }
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"scene_observed": True}
    failure_modes: list[str] = ["no_base", "no_vlm", "camera_failed", "vlm_failed", "perception_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """The VLM's description (or its answer to ``query``); what it names is
        localized on the same snapshot and only localized objects are stored."""
        if context.base is None:
            return SkillResult(success=False, error_message="No base connected",
                               diagnosis_code="no_base")
        vlm = context.services.get("vlm")
        if vlm is None:
            return SkillResult(success=False, error_message="VLM service not available",
                               diagnosis_code="no_vlm")
        query: str | None = params.get("query") or None
        try:
            data = observe_scene(context.base, context.perception,
                                 context.services.get("spatial_memory"), vlm, [], query=query)
        except ObserveFailed as exc:
            logger.error("[DESCRIBE_SCENE] %s", exc)
            return SkillResult(success=False, error_message=str(exc), diagnosis_code=exc.code)
        if query is not None:
            return SkillResult(success=True, result_data={
                "query": query, "objects": data["objects"], "unlocalized": data["unlocalized"],
                "failed": data["failed"], "count": len(data["objects"]) + len(data["unlocalized"]),
            })
        return SkillResult(success=True, result_data=data)
