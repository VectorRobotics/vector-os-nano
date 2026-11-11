# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""DetectorCapability — learned detectors as ONE routable capability, two backends.

``backend`` picks the model: ``grounding-dino`` (default; local open-vocab
detector, ~0.1 s, scored boxes, English noun phrases) or ``rynnbrain`` (the
external RynnBrain embodied VLM at :8786, ~1 s, one unscored box, reads zh and
relations). Behind the Capability contract (``vcli/cognitive/capabilities/types.py``)
so a ``detect`` step is ROUTED to a specialized model alongside the chat LLM.
Read-only (``side_effecting=False``): ``invoke`` reports only that the detector
RAN; the verify predicate decides real success — it never self-certifies.

Every box carries its frame's provenance (``frame``: id, size, sim time, camera
pose) so a consumer can tell a box from an earlier frame (perception_grasp
re-detects a stale one).

Lives OUTSIDE ``vcli/cognitive/`` (the frozen verify+routing seam): it is domain
perception code that the robot world registers into the kernel's registry.
"""
from __future__ import annotations

import time
from typing import Any

BACKENDS = ("grounding-dino", "rynnbrain")

import numpy as np

from vector_os_nano.vcli.cognitive.capabilities.types import CapabilityResult


class DetectorCapability:
    """Open-vocabulary detector as a routable, read-only capability.

    Payload (the sub-goal's ``strategy_params``):
        - ``query``: str  (REQUIRED) — natural-language target ("the can").
        - ``rgb``: (H,W,3) uint8 array  (one of rgb / perception required)
        - ``perception``: an object exposing ``get_color_frame()`` (alternative
          to a raw ``rgb`` — the live RGB-D backend supplies the frame).
    Output: ``{"detections": [Detection.to_dict, ...], "boxes": [...],
    "labels": [...], "scores": [...]}``.
    """

    name = "detect"
    kind = "detector"
    side_effecting = False
    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "backend": {"type": "string", "enum": list(BACKENDS)},
            "rgb": {"type": "array"},
            "perception": {"type": "object"},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "detections": {"type": "array"},
            "boxes": {"type": "array"},
            "labels": {"type": "array"},
            "scores": {"type": "array"},
            "frame": {"type": "object"},
            "backend": {"type": "string"},
        },
    }

    def __init__(
        self, detector: Any = None, perception: Any = None, agent: Any = None,
        rynnbrain: Any = None,
    ) -> None:
        # Default to the shared lazy singleton so this capability and the grasp
        # route share ONE loaded model. Injectable for tests (no torch load).
        self._detector = detector
        # Optional bound RGB source (exposes ``get_color_frame()``), supplied at
        # registration by the world. The capability-dispatch path builds a
        # SkillContext (not a perception), so the kernel context cannot itself
        # yield a frame; binding the world's perception here lets the routed
        # ``detect`` sub-goal perceive WITHOUT any kernel/cognitive change. Stays
        # None for dev/CI; ``_resolve_rgb`` then falls back to the payload/context.
        self._perception = perception
        # COLD-TURN rebind (R37 Task B): register_capabilities runs at init_vgg,
        # BEFORE a mid-session NL `启动 go2 带机械臂` boots the arm — so the agent's
        # ``_perception`` is None at bind time and a snapshot would stay None forever.
        # Hold the AGENT instead and pull its LIVE ``_perception`` lazily at invoke
        # time, so a cold product turn (sim started this same turn) perceives without
        # any pre-boot. World-side wiring only; the cognitive seam is untouched.
        self._agent = agent
        self._rynnbrain = rynnbrain

    def _live_perception(self) -> Any:
        """The current frame source: a snapshot perception, else the agent's LIVE one.

        Prefers an explicitly-bound ``perception`` (tests / a world that already had
        one); otherwise reads ``agent._perception`` fresh on every call so a sim
        started AFTER registration is picked up (cold-turn rebind). Returns None when
        neither yields a source (dev/CI) — the payload/context fallback then applies.
        """
        if self._perception is not None:
            return self._perception
        agent = self._agent
        if agent is not None:
            return getattr(agent, "_perception", None)
        return None

    def _get_detector(self, backend: str = "grounding-dino") -> Any:
        if backend == "rynnbrain":
            if self._rynnbrain is None:
                from vector_os_nano.perception.rynnbrain import RynnBrainDetector
                self._rynnbrain = RynnBrainDetector()
            return self._rynnbrain
        if self._detector is None:
            from vector_os_nano.perception.grounding_dino import get_shared_detector
            self._detector = get_shared_detector()
        return self._detector

    def estimate(self, payload: dict[str, Any]) -> tuple[float, float]:
        # Cheap, no I/O: a free, ~0.3 s prior for routing/tiebreak.
        return (0.0, 0.3)

    def invoke(self, payload: dict[str, Any], context: Any) -> CapabilityResult:
        start = time.monotonic()
        if not isinstance(payload, dict):
            return CapabilityResult(success=False, error="payload must be a dict")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return CapabilityResult(
                success=False,
                error="missing required input: query (non-empty string)",
            )

        backend = str(payload.get("backend") or "grounding-dino").strip().lower()
        if backend not in BACKENDS:
            return CapabilityResult(
                success=False, error=f"unknown detector backend {backend!r}; valid: {list(BACKENDS)}",
            )

        frame, rgb = self._resolve_frame(payload, context, self._live_perception())
        if rgb is None:
            return CapabilityResult(
                success=False,
                error="no RGB frame: supply 'rgb' or a 'perception' with capture()/get_color_frame()",
                latency_sec=time.monotonic() - start,
            )

        try:
            detections = self._get_detector(backend).detect(rgb, query)
        except Exception as exc:  # noqa: BLE001
            return CapabilityResult(
                success=False,
                error=f"{backend} error: {exc}",
                output={"backend": backend},
                latency_sec=time.monotonic() - start,
            )

        prov = _provenance(frame, rgb)
        output = {
            "detections": [dict(d.to_dict(), frame=prov) for d in detections],
            "boxes": [list(d.bbox) for d in detections],
            "labels": [d.label for d in detections],
            "scores": [d.confidence for d in detections],
            "frame": prov,
            "backend": backend,
        }
        # success reports only that it RAN AND found something (per the protocol
        # docstring); the sub-goal's verify decides the real outcome.
        return CapabilityResult(
            success=bool(detections),
            output=output,
            latency_sec=time.monotonic() - start,
        )

    @staticmethod
    def _resolve_frame(
        payload: dict[str, Any], context: Any, bound_perception: Any = None
    ) -> tuple[Any, np.ndarray | None]:
        """``(frame, rgb)``: an explicit ``rgb`` (no frame), else the first source's
        snapshot (``capture()``: pixels + pose of one instant), else its RGB.

        Source priority: the payload's ``perception``, the capability's
        REGISTRATION-BOUND perception, then a ``context`` that yields a frame (itself
        or its ``.perception``) — the kernel context is a SkillContext, so the bound
        source is how a routed ``detect`` step reaches the live camera.
        """
        rgb = payload.get("rgb")
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            return None, rgb
        for src in (payload.get("perception"), bound_perception, context,
                    getattr(context, "perception", None)):
            try:
                if callable(getattr(src, "capture", None)):
                    f = src.capture()
                    if isinstance(f.rgb, np.ndarray) and f.rgb.ndim == 3:
                        return f, f.rgb
                elif callable(getattr(src, "get_color_frame", None)):
                    img = src.get_color_frame()
                    if isinstance(img, np.ndarray) and img.ndim == 3:
                        return None, img
            except Exception:  # noqa: BLE001 — best-effort frame source; try the next
                continue
        return None, None


def _provenance(frame: Any, rgb: np.ndarray) -> dict[str, Any]:
    """What a box belongs to: frame size, and (from a snapshot) its id, time and pose."""
    prov: dict[str, Any] = {"size": [int(rgb.shape[1]), int(rgb.shape[0])]}
    if frame is not None:
        prov.update(
            frame_id=f"{getattr(frame, 'camera', 'camera')}@{float(frame.sim_time):.4f}",
            sim_time=float(frame.sim_time),
            cam_pos=[float(v) for v in np.asarray(frame.cam_pos).ravel()],
            cam_mat=[float(v) for v in np.asarray(frame.cam_mat).ravel()],
        )
    return prov
