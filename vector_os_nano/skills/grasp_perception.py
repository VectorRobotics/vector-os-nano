# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Exact target resolution and one RGB-D plausibility gate for grasp."""
from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.base import failure_cause
from vector_os_nano.perception.grasp_point import object_points, grasp_point_from_points
from vector_os_nano.perception.object_localizer import capture_rgbd, implausible

logger = logging.getLogger(__name__)
ALIASES = ["抓", "拿", "抓起", "抓住", "抓取", "拿起", "取", "grab", "grasp", "pick up", "pick"]


def _fail(code, message, **data):
    return SkillResult(False, {"diagnosis": code, **data}, message, diagnosis_code=code)


# A grasp query is ONE object reference, optionally led by one of this skill's own verbs
# ("grab the red can") and beside a spatial word ("the can in front"): nothing else.
_SPATIAL = re.compile(r"(?<![a-z0-9_])(?:in front(?: of (?:me|you))?|ahead|nearest|closest)(?![a-z0-9_])"
                      r"|前面|前方|面前|眼前|前边|最近")
# A query naming nothing but "what is in front" (once its spatial words are gone).
_DEICTIC = re.compile(r"(?:\s*(?:(?:the|that|this|a|one|thing|object|item|something)(?![a-z0-9_])"
                      r"|的|那个|这个|那|这|东西|物体|物品|个))*\s*")


class _Target(NamedTuple):
    """What the grasp goes for, decided before any motion."""

    front: bool          # the front-object resolver: by hue (colour), else by space
    colour: str | None   # the hue it is, for the resolver, the box choice and the box check
    prompt: str | None   # what the detector is asked for (the detector route; a stale box)
    label: str | None    # the verify label (None: the detector's own label)


def _reference(query: str) -> str:
    """*query* without a leading grasp verb of this skill and without its spatial words."""
    q = " ".join(query.lower().split())
    for verb in sorted(ALIASES, key=len, reverse=True):
        if q.startswith(verb) and not (verb.isascii() and q[len(verb):len(verb) + 1].isalnum()):
            q = q[len(verb):]
            break
    return " ".join(_SPATIAL.sub(" ", q).split())


def _target(query: str, world, have_front: bool) -> _Target | SkillResult:
    """The grasp's target, or the refusal (before any motion).

    With a world manifest the reference must name ONE object of this world
    (``WorldManifest.referents``): it is perceived by its own colour (the hue route, when
    the front resolver has a band for it) or else its English name (the detector), and
    labelled by its body name. A query naming only what is in front takes the front
    route. A reference fitting several objects is refused ``ambiguous_object``, anything
    else ``unknown_object`` with what the world has, so the model can retry by name.
    Without a manifest: only deictic -> the front route, only a colour -> the hue route,
    else the detector on the query's English phrase (CJK only when some world's words
    account for all of it). No route ever falls back to another.
    """
    from vector_os_nano.perception.front_object import parse_color
    from vector_os_nano.perception.vocab import is_reference, vocab

    ref = _reference(query)
    if _DEICTIC.fullmatch(ref):
        return _Target(True, None, None, query or "front object")
    if world is not None:
        fits = world.referents(ref)
        if fits and len(fits) > 1:
            names = [world.pickables[n].english for n in sorted(fits)]
            return _fail("ambiguous_object", f"{query!r} fits {len(names)} objects here: {names}; name one.",
                         query=query, candidates=names)
        if not fits:
            names = world.vocabulary()
            return _fail("unknown_object", f"{query!r} names no object of this world; name one of {names} "
                         "(or say 'the thing in front').", query=query, known_objects=names)
        pk = world.pickables[next(iter(fits))]
        hue = parse_color(pk.colour)
        unique_hue = sum(parse_color(o.colour) == hue for o in world.pickables.values()) == 1
        return _Target(have_front and hue is not None and unique_hue, hue, pk.english, pk.name)
    colour = parse_color(ref)
    if colour and _DEICTIC.fullmatch(vocab().without_colours(ref).replace("色", "")):
        return _Target(True, colour, None, query)
    phrase = vocab().english(ref)
    if not phrase or not (ref.isascii() or is_reference(ref)):
        return _fail("unknown_object", f"{query!r} has no English name to ask the detector for; "
                     "name the object in English.", query=query)
    return _Target(False, colour, phrase, None)


def _fresh(frame, rgb, pos, rot):
    """Only a box with valid camera provenance can be consumed without re-detection."""
    if not isinstance(frame, dict):
        return False
    try:
        a, b = np.asarray(frame["cam_mat"], float).reshape(3, 3), np.asarray(rot).reshape(3, 3)
        distance = float(np.linalg.norm(np.asarray(frame["cam_pos"], float) - pos))
        angle = float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))
        return list(frame["size"]) == [rgb.shape[1], rgb.shape[0]] and distance <= .02 and angle <= .035
    except (KeyError, TypeError, ValueError):
        return False


def _passed(params, target):
    dets = params.get("detections")
    if isinstance(dets, list):
        # A colour alone cannot distinguish two objects of the same colour.
        names = {n.lower() for n in (target.label, target.prompt) if n}
        fits = [d for d in dets if isinstance(d, dict) and str(d.get("label", "")).lower() in names]
        if fits:
            det = max(fits, key=lambda d: float(d.get("confidence", 0)))
            return det, det.get("frame") or params.get("frame")
    if "bbox" in params and target.front:
        return {"bbox": params["bbox"], "confidence": 1.0}, params.get("frame")
    return None, None


def perceive(perception, query, target, params=None, *, cancel=lambda: False, expected_bounds=None):
    """(point, metadata, failure), all coordinates from one sensor instant."""
    try:
        if cancel():
            return None, {}, _fail("cancelled", "Grasp cancelled before perception.")
        rgb, depth, K, pos, rot = capture_rgbd(perception)
    except Exception as exc:
        return None, {}, _fail("no_camera", f"Cannot read RGB-D: {failure_cause(exc)}; check the camera.")
    meta = {"perceived": False, "consumed_bbox": False, "reperceived": True,
            "detection_label": target.label or query}
    try:
        prior = None if expected_bounds is None else np.asarray(expected_bounds, dtype=float)
        if prior is not None and (prior.shape != (2, 3) or not np.isfinite(prior).all() or np.any(prior[0] >= prior[1])):
            raise ValueError('invalid prior observed bounds')
        passed, frame = _passed(params or {}, target)
        if passed is not None and _fresh(frame, rgb, pos, rot):
            detections = [SimpleNamespace(**{k: passed[k] for k in ("bbox", "confidence")})]
            meta.update(consumed_bbox=True, reperceived=False)
        elif target.front:
            mask = perception.front_object_mask(rgb, depth, color=target.colour)
            if mask is None or not np.any(mask):
                return None, meta, _fail("no_detections", f"Cannot see {query!r}; use look from another view, then grasp again.")
            yy, xx = np.nonzero(mask)
            detections = [SimpleNamespace(bbox=(xx.min(), yy.min(), xx.max()+1, yy.max()+1), confidence=1., mask=mask)]
        else:
            detections = perception.detect_in(rgb, target.prompt or query)
        if not detections:
            return None, meta, _fail("no_detections", f"Cannot see {query!r}; use look from another view, then grasp again.")
        rejected = []
        for det in sorted(detections, key=lambda d: float(d.confidence), reverse=True):
            if cancel():
                return None, meta, _fail("cancelled", "Grasp cancelled during perception.")
            mask = getattr(det, "mask", None)
            if mask is None:
                mask = perception.segment(rgb, det.bbox)
            if mask is None or not np.any(mask):
                rejected.append("empty segmentation")
                continue
            points = object_points(depth, rgb, mask, K, pos, rot)
            if points is None:
                rejected.append("no depth points")
                continue
            why = implausible(det, mask, *points, depth, rgb, colour=target.colour)
            if why:
                rejected.append(why)
                continue
            point = grasp_point_from_points(*points, pos, rot)
            if point is not None:
                bounds = np.percentile(points[1], [2, 98], axis=0)
                centre = bounds.mean(axis=0)
                if prior is not None and np.any((centre < prior[0]-.03) | (centre > prior[1]+.03)):
                    rejected.append('detection moved outside prior observed bounds; reacquire its identity')
                    continue
                if cancel():
                    return None, meta, _fail("cancelled", "Grasp cancelled during perception.")
                meta.update(perceived=True, segmenter=getattr(perception, "last_segmenter", None),
                            grasp_world=[point.x, point.y, point.z],
                            visible_bounds=bounds.tolist(), association_bounds=expected_bounds)
                return point, meta, None
        return None, meta, _fail("implausible_target", f"No plausible grasp point for {query!r}: {rejected}; look again.")
    except Exception as exc:
        logger.warning("Grasp perception failed: %s", exc)
        return None, meta, _fail("perception_failed", f"Cannot check {query!r}: {failure_cause(exc)}; check perception, then look again.")


