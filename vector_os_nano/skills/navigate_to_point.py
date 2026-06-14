# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""NavigateToPointSkill — navigation to world (x, y) — M3, nav-stack N4.

Base-generic: any connected base exposing ``navigate_to(x, y, tol)`` can run
it. N4 transport selection: when the REAL nav stack is alive (the base's
``_nav_feed`` reports an active pathFollower), navigation goes through it —
/way_point in, sensor-driven /navigation_cmd_vel out, euclidean progress
monitoring only. The sim-oracle ``base.navigate_to`` is the FALLBACK
(nav stack down / pure-sim installs), and the verify criterion
``geodesic_dist(x, y) < 0.8`` stays the deterministic judge either way —
the planner binds the SAME coordinates into params and verify, so a stuck
or unreachable navigation can never false-pass. ``result_data.transport``
records which path ran (honest provenance).
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


def _resolve_base_target(base: object, label: str
                         ) -> "tuple[float, float] | None":
    """Resolve a label against the base's GROUND-TRUTH targets, if it exposes
    ``list_targets() -> {name: (x, y)}`` (g1_room). Matches the exact name,
    ``target_<label>``, or a color/word suffix (so 'blue' finds 'target_blue').
    Returns (x, y) or None. This is the substrate-agnostic go-to-target path —
    NOT photoreal recognition (that stays gated by DQ-10)."""
    fn = getattr(base, "list_targets", None)
    if not callable(fn):
        return None
    try:
        targets = fn() or {}
    except Exception as exc:  # noqa: BLE001 — a flaky base never breaks navigation
        logger.warning("list_targets() failed on %s: %s",
                       type(base).__name__, exc)
        return None
    key = label.strip().lower()
    # zh/en color aliases so '去蓝色目标'/'blue'/'blue_target' all resolve. The
    # planner emits varied label shapes (a bare color, a zh phrase, or a
    # compound 'blue_target'); tokenise both sides and match on the DISTINCTIVE
    # token (the color), ignoring the generic 'target'/'目标' filler.
    _ALIASES = {"红": "red", "蓝": "blue", "绿": "green"}
    _FILLER = {"target", "目标", "the", "go", "to", "去", "color", ""}
    key_tokens: set = set()
    for zh, en in _ALIASES.items():       # zh color words appear as substrings
        if zh in key:
            key_tokens.add(en)
    for t in key.replace(" ", "_").split("_"):   # en/compound tokens
        if t and t not in _FILLER:
            key_tokens.add(t)
    for name, xy in targets.items():
        n = str(name).lower()
        name_tokens = {t for t in n.split("_") if t not in _FILLER}
        if key == n or (key_tokens and key_tokens & name_tokens):
            return (float(xy[0]), float(xy[1]))
    return None


@skill(
    aliases=["navigate to", "go to", "走到", "导航到", "去坐标"],
    direct=False,
)
class NavigateToPointSkill:
    """Navigate the mobile base to a world coordinate along the navmesh."""

    name: str = "navigate_to"
    description: str = (
        "Navigate the mobile base to world coordinates (x, y) following the "
        "shortest navigable path. Use for any 'go to position / 走到坐标' "
        "instruction with explicit coordinates."
    )
    # A real obstacle-routed walk takes wall time: the G1 biped gait at ~0.4
    # m/s with pivots covers a ~5 m room route in ~120 s (campaign #8 R3/R5
    # measured). This is the timeout FLOOR (max'd with the planner's estimate)
    # — it only PREVENTS a premature cut, never forces slowness, so the faster
    # go2/habitat navigations are unaffected. The old 20 s cut G1 mid-route.
    typical_duration_sec: float = 130.0
    # The deterministic VLN success criterion. Bind the SAME x, y here and in
    # strategy_params (the planner copies the literal coordinates).
    # STANDOFF (N4): an object-label goal verifies < 1.5 — a real robot stops
    # at the furniture clearance boundary, never ON the object's coordinates.
    # Calibrated to the REAL controller (N4, five live runs): the follower
    # parks within ~0.4 m of the planned path END and local plans end early
    # near clutter — coordinate arrival is honestly ~0.7 m, not the oracle's
    # teleport precision.
    verify_hint: str = (
        "geodesic_dist(x, y) < 0.8 for coordinate goals; for ROOM goals "
        "(labels marked type=room) use visited('<label>') — true only when "
        "the robot is INSIDE the room's rectangle; for object-label goals "
        "use at_position(x, y, 1.6) with the object's listed coordinates — "
        "EUCLIDEAN standoff (an object's centre sits OFF the navmesh, which "
        "distorts geodesic distance; the robot parks at furniture clearance, "
        "never on the object)"
    )
    parameters: dict = {
        "label": {
            "type": "string",
            "required": False,
            "description": (
                "Target OBJECT label (e.g. 'sofa'). PREFER this for any "
                "'go to the <object>' instruction: the skill resolves the "
                "object's live coordinates from the world model and FAILS "
                "LOUDLY if no such object is known — never invent x/y for "
                "an object. Object goals stop at furniture STANDOFF "
                "(~1.5m) — verify them with at_position(x, y, 1.6)."
            ),
        },
        "x": {
            "type": "number",
            "required": False,
            "description": "Target world x coordinate in metres (coordinate-style goals).",
        },
        "y": {
            "type": "number",
            "required": False,
            "description": "Target world y coordinate in metres (coordinate-style goals).",
        },
        "tol": {
            "type": "number",
            "required": False,
            "default": 0.2,
            "description": "Waypoint tolerance in metres.",
        },
    }
    preconditions: list = ["base connected and navigate_to-capable"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_navigate_support", "no_path", "nav_stuck"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(
                success=False,
                error_message="no mobile base connected",
                result_data={"diagnosis": "no_base"},
            )
        if not callable(getattr(base, "navigate_to", None)):
            return SkillResult(
                success=False,
                error_message=(
                    f"base {type(base).__name__} has no navigate_to capability"
                ),
                result_data={"diagnosis": "no_navigate_support"},
            )

        # Semantic goal: resolve the label against the LIVE world model — the
        # navigate analogue of named-grasp #2b. An unknown object fails LOUDLY
        # (never silently navigate to invented coordinates: that is a FALSE
        # SUCCESS — went somewhere, reported arrival at a phantom).
        label = str(params.get("label", "") or "").strip()
        if label:
            wm = getattr(context, "world_model", None)
            matches = wm.get_objects_by_label(label) if wm is not None else []
            if matches:
                best = max(matches, key=lambda o: o.confidence)
                x, y = float(best.x), float(best.y)
                params = dict(params)
                props = getattr(best, "properties", None) or {}
                rect = props.get("rect") if props.get("type") == "room" else None
                if rect is not None:
                    # ROOM goal (batch 2 #3): a room is a REGION — drive INTO
                    # the rect, never park at the 1.5 m object standoff (which
                    # can exceed the start distance and degrade to a no-op; the
                    # '走到厨房' seed). Tolerance from the rect's half-dims keeps
                    # the arrival point inside; visited('<label>') is honest.
                    try:
                        x0, y0, x1, y1 = (float(v) for v in rect)
                        half_min = min(abs(x1 - x0), abs(y1 - y0)) / 2.0
                        params["tol"] = max(0.3, min(half_min * 0.8, 1.5))
                    except (TypeError, ValueError):
                        params["tol"] = max(float(params.get("tol", 0.0) or 0.0), 1.5)
                    goal_kind = "room"
                else:
                    # OBJECT goal — semantic standoff: "go to the sofa" means
                    # NEAR it; the follower halts at the furniture clearance.
                    params["tol"] = max(float(params.get("tol", 0.0) or 0.0), 1.5)
                    goal_kind = "object"
            else:
                # No semantic match. Substrate-agnostic fallback (campaign #8
                # R5): a base may expose GROUND-TRUTH labeled targets
                # (g1_room.list_targets) — the 'go to the target object's point'
                # half of req #5 WITHOUT photoreal recognition (DQ-10-gated).
                gt_target = _resolve_base_target(base, label)
                if gt_target is not None:
                    x, y = gt_target
                    params = dict(params)
                    params["tol"] = max(float(params.get("tol", 0.0) or 0.0), 0.35)
                    goal_kind = "object"
                else:
                    known = sorted({o.label for o in wm.get_objects()}) if wm else []
                    # An EMPTY world model means semantic perception never ran —
                    # replanning cannot help; name the actual fix.
                    hint = ("" if known else
                            " — the world model is EMPTY: semantic perception "
                            "is not running. Start it first (say '启动sysnav' / "
                            "'start sysnav'), wait for detections, then retry.")
                    return SkillResult(
                        success=False,
                        error_message=(
                            f"no object matching {label!r} in the live world "
                            f"model; known objects: {known or '<none>'}{hint}"),
                        result_data={"diagnosis": "object_not_found",
                                     "label": label},
                    )
        else:
            try:
                x, y = float(params["x"]), float(params["y"])
            except (KeyError, TypeError, ValueError):
                return SkillResult(
                    success=False,
                    error_message="navigate_to requires a label OR numeric x and y",
                    result_data={"diagnosis": "bad_params"},
                )
            goal_kind = "coordinate"
        tol = float(params.get("tol", 0.2) or 0.2)

        feed = getattr(base, "_nav_feed", None)
        use_nav = feed is not None and feed.nav_stack_active()
        logger.info(
            "[NAVIGATE_TO] -> (%.2f, %.2f) tol=%.2f via %s",
            x, y, tol, "nav_stack" if use_nav else "sim_oracle",
        )
        if use_nav:
            out = feed.navigate_to(x, y, tol)
        else:
            out = base.navigate_to(x, y, tol)
        reached = bool(out.get("reached", False))
        remaining = float(out.get("remaining", float("inf")))
        result_data = {
            "target": [x, y],
            "goal_kind": goal_kind,
            "reached": reached,
            "remaining_m": remaining,  # euclidean (room geodesic via geodesic_distance)
            "position": out.get("pos"),
            "transport": out.get("transport", "sim_oracle"),
            # Motion evidence (three-value contract, R6): 'drove there' and
            # 'never moved' are different shapes for verify/replan/user.
            "already_there": bool(out.get("already_there", False)),
            "moved_m": float(out.get("moved_m", 0.0) or 0.0),
            "elapsed_s": float(out.get("elapsed_s", 0.0) or 0.0),
            "diagnosis": "ok" if reached else str(out.get("reason", "nav_stuck")),
        }
        return SkillResult(
            success=reached,
            error_message="" if reached else (
                f"navigation stopped {remaining:.2f}m (geodesic) short of "
                f"({x:.2f}, {y:.2f})"
            ),
            result_data=result_data,
        )
