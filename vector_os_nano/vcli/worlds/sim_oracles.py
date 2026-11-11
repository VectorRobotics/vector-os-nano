# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""sim_oracles — every ground-truth verify predicate of a sim world, in ONE module.

Each predicate reads the session's :class:`~vector_os_nano.sim.ground_truth.GroundTruth`
(MjData at a step boundary) plus the world / embodiment manifests — never a device
or agent attribute, so nothing the actor writes can make one true. The one CLAIM an
oracle judges is the actor's belief for ``localized`` (a SceneGraph instance), and
it is judged against GT.

Every predicate is tagged ``@oracle`` (scope robot, its actor channel and the
transition that proves it); untagged helpers never ground (oracle_registry). Object
arguments accept the scene body name or anything the world manifest resolves
("green bottle", "绿色的瓶子"). Every predicate fails safe to False (or None for a
state read) instead of raising into the verifier.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

from vector_os_nano.sim.nav.follow import arrival_radius
from vector_os_nano.vcli.cognitive.coord_goal import (
    _MAX_ARRIVAL_TOL_M,
    _MAX_FACING_TOL_RAD,
    _MAX_LOCALIZED_M,
    _MAX_NEAR_M,
    _MAX_OBJECT_AT_M,
)
from vector_os_nano.vcli.cognitive.oracle_registry import oracle

logger = logging.getLogger(__name__)

# Default tolerances. Caps clamp a caller-supplied tolerance INSIDE the oracle, so an
# inflated one (``near_object('x', 1e6)``) is vacuous for every goal (FG-3); the
# classifier also refuses an over-cap literal (evidence_classifier._TOL_ARG).
# at_position: the navigation arrival radius (config/nav.yaml), never above 0.5 m.
_AT_POSITION_TOL_M: float = min(0.5, arrival_radius())
_FACING_TOL_RAD: float = math.radians(20.0)
_NEAR_OBJECT_M: float = 0.8
_OBJECT_AT_M: float = 0.3
_LOCALIZED_M: float = 0.15
# A grasped object counts as lifted once its lowest point clears the support below
# it by this much (the object's rest height + 3 cm, wherever it rests).
LIFT_M: float = 0.03
# Resting = not welded, touched by no robot body, moving slower than this.
_REST_SPEED: float = 0.05
_HOME_TOL_RAD: float = 0.10
# Segmentation pixels (at the manifest camera size) for "in view".
_MIN_VIEW_PX: int = 20


def _safe(default: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Fail safe: a GT read error returns *default*, never raises into the verifier."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.debug("oracle %s%r failed: %s", fn.__name__, args, exc)
                return default

        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner

    return wrap


def _clamp(value: Any, cap: float) -> float:
    return min(float(value), cap)


def _angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _camera(cfg: Any) -> tuple[str, int, int] | None:
    for s in getattr(cfg, "sensors", ()):
        if s.role == "camera" and s.name and "width" in s.params and "height" in s.params:
            return s.name, int(s.params["width"]), int(s.params["height"])
    return None


def build_namespace(
    gt: Any,
    world: Any,
    cfg: Any,
    *,
    belief: Callable[[], Any] = lambda: None,
    turn_start: Callable[[], tuple[float, float]] = lambda: (-math.inf, -math.inf),
) -> dict[str, Callable[..., Any]]:
    """The verify namespace of one sim session.

    *gt* the session's GroundTruth, *world* its WorldManifest, *cfg* its
    EmbodimentConfig. *belief* returns the actor's SceneGraph (the claim
    ``localized`` judges). *turn_start* returns ``(sim_time, wall_time)`` of the
    current REPL turn's start (``kept_holding`` / ``localized`` freshness).
    """
    ns: dict[str, Callable[..., Any]] = {}

    def obj(target: Any) -> str:
        name = world.resolve(target) if world is not None else None
        if name is None and str(target) in gt.bid:
            name = str(target)
        if name is None or name not in gt.bid:
            raise KeyError(f"no scene object for {target!r}")
        return name

    # ---------------------------------------------------------------- base
    if gt.has_base:

        @oracle(scope="robot", channel="base", transition="rise", max_tol=_MAX_ARRIVAL_TOL_M, domain="bool")
        @_safe(False)
        def at_position(x: Any, y: Any, tol: Any = _AT_POSITION_TOL_M) -> bool:
            """Base within *tol* m (xy) of (x, y)."""
            bx, by, _z, _yaw = gt.base_pose()
            return math.dist((bx, by), (float(x), float(y))) <= _clamp(tol, _MAX_ARRIVAL_TOL_M)

        @oracle(scope="robot", channel="base", transition="rise", max_tol=_MAX_FACING_TOL_RAD, domain="bool")
        @_safe(False)
        def facing(heading: Any, tol: Any = _FACING_TOL_RAD) -> bool:
            """Base yaw within *tol* rad of *heading* (wraps across +/-pi)."""
            return _angle_delta(gt.base_pose()[3], float(heading)) <= _clamp(tol, _MAX_FACING_TOL_RAD)

        @oracle(scope="robot", channel="base", transition="rise", domain="bool")
        @_safe(False)
        def in_room(room: Any) -> bool:
            """Base inside the named room's floor box (world manifest)."""
            x0, y0, x1, y1 = world.boxes[str(room)]
            bx, by, _z, _yaw = gt.base_pose()
            return x0 <= bx <= x1 and y0 <= by <= y1

        @oracle(scope="robot", channel="base", transition="rise", max_tol=_MAX_NEAR_M, domain="bool")
        @_safe(False)
        def near_object(name: Any, r: Any = _NEAR_OBJECT_M) -> bool:
            """Base within *r* m (xy) of a scene object, or of the user ('user')."""
            if str(name).strip().lower() == "user":
                if world.user is None:
                    return False
                tx, ty = world.user
            else:
                tx, ty = gt.object_positions()[obj(name)][:2]
            bx, by, _z, _yaw = gt.base_pose()
            return math.dist((bx, by), (tx, ty)) <= _clamp(r, _MAX_NEAR_M)

        @oracle(scope="robot", channel="base")
        @_safe(None)
        def get_position() -> tuple[float, float, float]:
            """Base (x, y, z) — a state read."""
            return tuple(gt.base_pose()[:3])

        @oracle(scope="robot", channel="base", domain=(-math.pi, math.pi))
        @_safe(None)
        def get_heading() -> float:
            """Base yaw (rad) — a state read."""
            return gt.base_pose()[3]

        ns.update(at_position=at_position, facing=facing, in_room=in_room,
                  near_object=near_object, get_position=get_position, get_heading=get_heading)

    # ---------------------------------------------------------------- camera
    cam = _camera(cfg)
    if cam is not None:

        @oracle(scope="robot", channel="base", transition="rise", domain="bool")
        @_safe(False)
        def in_view(target: Any) -> bool:
            """Object visible in the robot's camera right now (segmentation render)."""
            return gt.visible_pixels(obj(target), *cam) >= _MIN_VIEW_PX

        ns["in_view"] = in_view

    @oracle(scope="robot", channel="perception", transition="rise", max_tol=_MAX_LOCALIZED_M, domain="bool")
    @_safe(False)
    def localized(target: Any, tol: Any = _LOCALIZED_M) -> bool:
        """The actor's SceneGraph has an instance of *target*, observed this turn,
        within *tol* m (xy) of where GT says the object is."""
        name = obj(target)
        sg = belief()
        if sg is None:
            return False
        tx, ty = gt.object_positions()[name][:2]
        _t_sim, t_wall = turn_start()
        for o in sg.find_objects_by_category(""):  # every instance
            seen = float(getattr(o, "last_seen", None) or getattr(o, "first_seen", 0.0))
            text = f"{(getattr(o, 'attributes', {}) or {}).get('color', '')} {o.category}"
            if seen < t_wall or world.resolve(text) != name:
                continue
            if o.x is None or o.y is None or (o.x, o.y) == (0.0, 0.0):  # never localized
                continue
            if math.dist((o.x, o.y), (tx, ty)) <= _clamp(tol, _MAX_LOCALIZED_M):
                return True
        return False

    ns["localized"] = localized

    # ---------------------------------------------------------------- arm
    caps = getattr(cfg, "capabilities", None)
    if caps is not None and caps.has_arm:
        arm = cfg.arm
        joints = tuple(arm.joints) if arm else tuple(cfg.stance)
        home = tuple(arm.home) if arm else tuple(cfg.stance.values())

        @oracle(scope="robot", channel="arm", transition="rise", domain="bool")
        @_safe(False)
        def arm_at_home() -> bool:
            """Every arm joint within 0.1 rad of the manifest home pose."""
            q = gt.joint_positions(joints)
            return len(q) == len(home) and all(abs(a - b) <= _HOME_TOL_RAD for a, b in zip(q, home))

        ns["arm_at_home"] = arm_at_home

    # ---------------------------------------------------------------- gripper
    gripper = getattr(getattr(cfg, "arm", None), "gripper", None)
    fingers = tuple(gt.body_id(b) for b in (gripper.finger_bodies if gripper else ()))
    if caps is not None and caps.has_gripper and gt.weld_active():

        def welded(name: str) -> bool:
            return bool(gt.weld_active().get(name, False))

        def gripped(name: str) -> bool:
            # A gripper with finger bodies in its manifest grasps by contact: every
            # finger must touch the object. A weld-only gripper (SO-101: visual jaws,
            # no finger bodies declared) is graded on the weld + the lift alone.
            return not fingers or set(fingers) <= gt.contacts(name)

        def held(name: str) -> bool:
            return welded(name) and gripped(name) and gt.clearance(name) >= LIFT_M

        def resting(name: str) -> bool:
            return (not welded(name) and not (gt.contacts(name) & gt.robot_bodies)
                    and gt.object_speed(name) < _REST_SPEED)

        @oracle(scope="robot", channel="gripper", transition="rise", domain="bool")
        @_safe(False)
        def holding_object(target: Any = None) -> bool:
            """Object welded to the gripper AND gripped by every finger AND lifted
            >= 3 cm above its support. ``target=None``: any object."""
            if target is not None:
                return held(obj(target))
            return any(held(n) for n, on in gt.weld_active().items() if on)

        @oracle(scope="robot", channel="gripper", transition="rise", domain="bool")
        @_safe(False)
        def lifted(target: Any, dz: Any = LIFT_M) -> bool:
            """Object held by the gripper with its lowest point >= *dz* (at least
            3 cm) above the support under it."""
            name = obj(target)
            return welded(name) and gt.clearance(name) >= max(LIFT_M, float(dz))

        @oracle(scope="robot", channel="gripper", transition="rise", domain="bool")
        @_safe(False)
        def kept_holding(target: Any) -> bool:
            """Held now AND its weld never dropped since the turn started."""
            name = obj(target)
            t0, _wall = turn_start()
            dropped = any(n == name and not on for _t, n, on in gt.weld_events(since=t0))
            return held(name) and not dropped

        @oracle(scope="robot", channel="gripper", transition="fall", domain="bool")
        @_safe(False)
        def released(target: Any = None) -> bool:
            """No longer held: the object's weld is off and no robot body touches it.
            ``target=None``: the gripper holds nothing."""
            if target is None:
                return not any(gt.weld_active().values())
            name = obj(target)
            return not welded(name) and not (gt.contacts(name) & gt.robot_bodies)

        @oracle(scope="robot", channel="gripper", transition="fall", max_tol=_MAX_OBJECT_AT_M, domain="bool")
        @_safe(False)
        def object_at(target: Any, x: Any, y: Any, r: Any = _OBJECT_AT_M) -> bool:
            """Object resting within *r* m (xy) of (x, y)."""
            name = obj(target)
            ox, oy = gt.object_positions()[name][:2]
            return resting(name) and math.dist((ox, oy), (float(x), float(y))) <= _clamp(r, _MAX_OBJECT_AT_M)

        @oracle(scope="robot", channel="gripper", transition="fall", domain="bool")
        @_safe(False)
        def object_on(target: Any, surface: Any) -> bool:
            """Object resting ON the named support surface: in contact with its body,
            above its top height, inside its footprint (world manifest)."""
            name, s = obj(target), world.surfaces[str(surface)]
            ox, oy, oz = gt.object_positions()[name]
            x0, y0, x1, y1 = s.box
            return (resting(name) and gt.body_id(s.body) in gt.contacts(name)
                    and oz > s.height and x0 <= ox <= x1 and y0 <= oy <= y1)

        ns.update(holding_object=holding_object, lifted=lifted, kept_holding=kept_holding,
                  released=released, object_at=object_at, object_on=object_on)

    return ns


def rig_namespace(runtime: Any, embodiment_id: str, **kwargs: Any) -> dict[str, Callable[..., Any]]:
    """:func:`build_namespace` for a runtime built without a SimSession (tests, probe tools)."""
    import mujoco as mj

    from vector_os_nano.embodiments import registry
    from vector_os_nano.sim.ground_truth import GroundTruth
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    cfg = registry.load(embodiment_id)
    root = mj.mj_name2id(runtime.model, mj.mjtObj.mjOBJ_BODY, cfg.model.root_body)
    gt = GroundTruth(runtime, robot_root_bid=int(root))
    return build_namespace(gt, load_world(cfg.world) if cfg.world else None, cfg, **kwargs)
