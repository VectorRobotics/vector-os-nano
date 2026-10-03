# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""g1 perception verify oracle — the HONEST GROUNDED for a camera-only embodiment.

R7. g1 (humanoid, head camera, NO arm) has no weld-causation path, so the only
honest GROUNDED for it is a GT-BACKED PERCEPTION MATCH: the learned detector
localizes an object on g1's camera (its box = the CLAIM), and the verify oracle
judges that claim against INDEPENDENT SIM ground truth the detector cannot author.

This is NOT the R4 self-read that minted a false green (D61). R4 bound
``detect_objects()`` over the detector's OWN stashed boxes and graded
``len(detect_objects()) > 0`` — the actor certifying itself (a tautology). Here:

  - the TRUTH is the MuJoCo SEGMENTATION render — the renderer's OWN per-pixel
    geom-id image of the scene geometry, taken from the SAME camera as the RGB. It
    is physics+render ground truth: it knows EXACTLY which pixels show the red
    object, with ZERO projection-convention guesswork (the engine does the
    projection, not us). The detector never sees it.
  - the CLAIM is the detector's box, obtained by running grounding-dino on the
    live RGB frame ONLY (the firewall: ``G1HeadPerception`` exposes only
    ``get_color_frame`` to the detector, never the segmentation/geom data).
  - the oracle returns True IFF a detected box CENTER lands within ``tol`` pixels of
    the SEGMENTATION CENTROID of the matching-colour geoms.

So the oracle returns "the detector's localization MATCHES where the SIM render says
the object actually is" — not "the detector found something". A WRONG detection (a
box on a wall, the wrong object) or the object OUT OF VIEW (no matching-colour geom
in the segmentation) → no match → False → the step grades RAN/FAILED, never
GROUNDED. GROUNDED must be EARNED by a spatially-correct localization (the refutation
proof).

WHY SEGMENTATION, NOT A HAND-ROLLED world→pixel PROJECTION (R7 honest pivot): the
freejoint pickables the GT-body accessor tracks turned out to be OCCLUDED in g1's
spawn view (segmentation: 0 px) — the red the detector sees is the static red bar
STOOL, which has no freejoint. And the head camera's pose-vs-frame relationship did
not reconcile under a standard pinhole convention (a 44° depression angle rendering
at mid-frame), so a hand-rolled ``world_to_pixel`` was UNRELIABLE — staking GROUNDED
on it would be a convincing-but-wrong result. The MuJoCo segmentation render sidesteps
both: it localizes whatever red geometry is ACTUALLY in view, in the renderer's own
exact pixels, no convention assumed. It is strictly MORE independent + MORE reliable
than the projection plan, and just as invisible to the detector.

Moat placement (kernel rule 2): this lives WORLD-side (``vcli.worlds``), bound into
the verify namespace by ``RobotWorld.build_verify_namespace``. The frozen
``vcli/cognitive/`` spine is BYTE-UNCHANGED — the oracle reaches GROUNDED purely by
being a STATE oracle the model compares against a constant
(``detection_matches_gt('red') == True``), which the existing
``evidence_classifier`` already grades GROUNDED.

Fail-safe contract (same as the arm/go2 oracles): a missing base / renderer / detector,
no matching-colour geom in view, or any read failure → the oracle returns ``False``
(never raises into the GoalVerifier sandbox, never a spurious True). Stricter-only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from vector_os_nano.vcli.cognitive.oracle_registry import oracle

logger = logging.getLogger(__name__)

# Pixel tolerance for the box-center ↔ segmentation-centroid match. The red stool in
# g1's spawn view spans ~900 seg px around (321, 260); the detector box center sits
# within ~30 px. 60 px absorbs box-center / centroid jitter yet is far tighter than a
# wrong-object miss (hundreds of px).
_DEFAULT_TOL_PX: float = 60.0

_RENDER_W: int = 640
_RENDER_H: int = 480

# Minimum matching-colour seg pixels for the GT to count as "in view" — guards against
# a stray handful of stale/edge pixels masquerading as the object.
_MIN_SEG_PX: int = 80

# Colour → an RGBA predicate over a geom's base colour (the dominant-channel test the
# room's coloured furniture/pickables satisfy). NL colour words (en + zh) map here.
_COLOUR_ZH: dict[str, str] = {
    "红": "red", "红色": "red",
    "绿": "green", "绿色": "green",
    "蓝": "blue", "蓝色": "blue",
}


def _colour_token(target: str) -> str | None:
    """Map an NL ``target`` to a base colour token ('red'/'green'/'blue'), or None."""
    q = str(target or "").strip().lower()
    if not q:
        return None
    for tok in ("red", "green", "blue"):
        if tok in q:
            return tok
    for zh, en in _COLOUR_ZH.items():
        if zh in str(target):
            return en
    return None


def _colour_matches(rgba: Any, token: str) -> bool:
    """True iff a geom RGBA is dominantly *token*-coloured (red/green/blue)."""
    try:
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    except (TypeError, ValueError, IndexError):
        return False
    if token == "red":
        return r > 0.55 and g < 0.5 and b < 0.5
    if token == "green":
        return g > 0.5 and r < 0.5 and b < 0.5
    if token == "blue":
        return b > 0.55 and r < 0.5 and g < 0.6
    return False


def _g1_base(agent: Any) -> Any | None:
    """Return the connected g1 sim base (exposes ``_model``/``_data`` + camera), else None.

    Reached only via the duck-typed ``agent._base`` accessor — no embodiment import.
    Requires the live MuJoCo model/data + the head-camera name so the oracle can run
    a segmentation render from the SAME camera as the RGB the detector sees.
    """
    if agent is None:
        return None
    base = getattr(agent, "_base", None)
    if base is None:
        return None
    if getattr(base, "_model", None) is None or getattr(base, "_data", None) is None:
        return None
    return base


def _red_geom_seg_centroid(base: Any, token: str) -> tuple[float, float, int] | None:
    """Segmentation centroid (u, v, px) of the *token*-coloured geoms in g1's view.

    Renders a MuJoCo SEGMENTATION image (per-pixel geom id) from the head camera —
    the renderer's OWN ground truth of what geometry is visible where — selects the
    pixels whose geom is *token*-coloured, and returns their centroid. None when the
    renderer is unavailable, no matching geom exists, or fewer than ``_MIN_SEG_PX``
    such pixels are visible (the object is not meaningfully in view → refutation).
    """
    try:
        import mujoco as mj

        model = base._model
        data = base._data
        cam_name = getattr(
            __import__(
                "vector_os_nano.hardware.sim.mujoco_g1", fromlist=["_SCENE_CAM_NAME"]
            ),
            "_SCENE_CAM_NAME",
        )
        try:
            cam_id = model.cam(cam_name).id
        except Exception:  # noqa: BLE001
            cam_id = 0
        red_geoms = [
            g for g in range(model.ngeom) if _colour_matches(model.geom_rgba[g], token)
        ]
        if not red_geoms:
            return None
        renderer = mj.Renderer(model, _RENDER_H, _RENDER_W)
        try:
            renderer.update_scene(data, camera=cam_id)
            renderer.enable_segmentation_rendering()
            seg = renderer.render()
        finally:
            try:
                renderer.disable_segmentation_rendering()
            except Exception:  # noqa: BLE001
                pass
            try:
                renderer.close()
            except Exception:  # noqa: BLE001
                pass
        objid = seg[:, :, 0]
        mask = np.isin(objid, red_geoms)
        ys, xs = np.where(mask)
        if len(xs) < _MIN_SEG_PX:
            return None
        return float(xs.mean()), float(ys.mean()), int(len(xs))
    except Exception as exc:  # noqa: BLE001
        logger.debug("detection_matches_gt: segmentation GT failed: %s", exc)
        return None


def _detector_boxes(agent: Any, base: Any, query: str) -> list[tuple[float, float, float, float]]:
    """Run the LEARNED detector on g1's live RGB ONLY → list of xyxy boxes (the CLAIM).

    The firewall: the frame source is the perception adapter's ``get_color_frame()``
    (the rendered RGB), and the detector is the SAME shared grounding-dino singleton
    the bare-cli detect tool uses — its whole input is pixels + the text query, NEVER
    the segmentation / geom data. Fail-safe to [] on any failure.
    """
    perception = getattr(agent, "_perception", None)
    rgb = None
    if perception is not None and callable(getattr(perception, "get_color_frame", None)):
        try:
            rgb = perception.get_color_frame()
        except Exception as exc:  # noqa: BLE001
            logger.debug("detection_matches_gt: perception.get_color_frame failed: %s", exc)
    if rgb is None and callable(getattr(base, "get_camera_frame", None)):
        try:
            rgb = base.get_camera_frame(_RENDER_W, _RENDER_H)
        except Exception as exc:  # noqa: BLE001
            logger.debug("detection_matches_gt: base.get_camera_frame failed: %s", exc)
    if rgb is None:
        return []
    try:
        from vector_os_nano.perception.grounding_dino import get_shared_detector

        detector = get_shared_detector()
        detections = detector.detect(rgb, query)
    except Exception as exc:  # noqa: BLE001
        logger.debug("detection_matches_gt: detector run failed: %s", exc)
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for d in detections:
        bbox = getattr(d, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        try:
            boxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        except (TypeError, ValueError):
            continue
    return boxes


def make_detection_matches_gt(agent: Any) -> Callable[..., bool]:
    """Build ``detection_matches_gt(target, tol=60.0)`` bound to *agent*.

    True IFF the LEARNED detector's box for *target* on g1's live camera lands within
    *tol* pixels of the SEGMENTATION CENTROID of the matching-colour geoms (the
    INDEPENDENT, renderer-native GT). Fails safe to False (no base / no renderer / no
    matching-colour geom in view / no box / detector error) — never raises, never a
    spurious True.
    """

    @oracle(scope="robot", channel="perception")
    def detection_matches_gt(target: Any = None, tol: float = _DEFAULT_TOL_PX) -> bool:
        base = _g1_base(agent)
        if base is None:
            return False
        token = _colour_token(str(target or ""))
        if token is None:
            logger.debug("detection_matches_gt: no colour token in target %r", target)
            return False

        # INDEPENDENT GT: where the renderer says the coloured object is.
        gt = _red_geom_seg_centroid(base, token)
        if gt is None:
            # Object not meaningfully in view → cannot match (the refutation case).
            logger.debug("detection_matches_gt: %s not in segmentation view", token)
            return False
        gu, gv, gpx = gt

        # CLAIM: the detector's boxes (run on RGB only).
        boxes = _detector_boxes(agent, base, str(target or ""))
        if not boxes:
            return False

        best = None
        for x1, y1, x2, y2 in boxes:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            dist = ((cx - gu) ** 2 + (cy - gv) ** 2) ** 0.5
            if best is None or dist < best:
                best = dist
        match = best is not None and best <= float(tol)
        logger.info(
            "[G1-PERCEPT-ORACLE] target=%r token=%s seg_gt=(%.1f,%.1f,%dpx) "
            "nearest_box_center_dist=%.1f tol=%.1f -> %s",
            target, token, gu, gv, gpx, best if best is not None else -1.0, tol, match,
        )
        return bool(match)

    return detection_matches_gt
