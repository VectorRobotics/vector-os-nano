# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Typed, frozen schema + loader for per-embodiment ``robot.yaml`` manifests.

A robot morphology is data, not code (project CLAUDE.md Rule 11). This module
parses ``vector_os_nano/embodiments/<id>/robot.yaml`` into immutable dataclasses
so the generic drivers READ every morphology-specific parameter — spawn pose,
nominal stance (by joint name), cameras and sensor mounts, root body, the arm and
its gripper, gait/policy reference, capability profile — instead of hardcoding it.

The loader is FAIL-LOUD: a missing file, a missing required key, or an unknown key
(top level, ``arm``, ``arm.gripper``, ``arm.ee``, a camera entry) raises a clear
error naming the file, the key and the valid set, never a silent default. No MuJoCo /
ROS2 imports: pure data, importable and testable offline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vector_os_nano.schemas.strict import yaml_loader

# Directory holding the per-embodiment subfolders (go2/, g1/, …).
_EMBODIMENTS_DIR: Path = Path(__file__).resolve().parents[1] / "embodiments"


class EmbodimentConfigError(ValueError):
    """Raised when a robot.yaml is missing or malformed (fail-loud, Rule 8)."""


# ---------------------------------------------------------------------------
# Schema — frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Where the body's MuJoCo model + meshes live, and its root body."""

    path: str
    root_body: str
    meshes_dir: str = ""


@dataclass(frozen=True)
class LidarSpec:
    """A lidar (``sensors: [{role: lidar, ...}]``) on ``body`` at ``pos`` (body frame; the
    mount follows the body's heading, not its roll and pitch). Its beams are pitched
    ``tilt_deg``, one ring per ``elevations_deg`` (0 is the 2-D scan) of ``n_azimuth``
    rays; hits farther than ``range`` are dropped from the point cloud; a scan is recast
    when the last one is ``period_s`` sim seconds old (0: on every read)."""

    body: str
    pos: tuple[float, float, float]
    tilt_deg: float
    elevations_deg: tuple[float, ...]
    n_azimuth: int
    range: float
    period_s: float
    role: str = "lidar"


@dataclass(frozen=True)
class CameraSpec:
    """A camera (``sensors: [{role: camera, ...}]``) and the size it renders at.

    ``depth``: the (near, far) range in metres it reports depth over (outside it 0,
    invalid); None for a colour-only camera. With ``pos`` the scene compiler adds the
    camera to ``body`` (the world when None) at ``pos`` / ``quat`` (w x y z) with
    ``fovy`` (degrees); without it the camera must already be in the robot's model.
    """

    name: str
    width: int
    height: int
    depth: tuple[float, float] | None = None
    body: str | None = None
    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fovy: float = 45.0
    role: str = "camera"


@dataclass(frozen=True)
class BasePolicySpec:
    """``base.policy``: the control policy (a row of ``sim.policies.POLICIES``) at
    ``tick_hz`` over physics steps of ``timestep``. ``params`` are the policy's own keys
    (it parses them strictly). ``by_backend[<backend>]`` replaces ``ref`` and ``params``
    whole on that backend."""

    ref: str
    timestep: float
    tick_hz: float
    params: dict[str, Any] = field(default_factory=dict)
    by_backend: dict[str, dict[str, Any]] = field(default_factory=dict)

    def on(self, backend: str) -> tuple[str, dict[str, Any]]:
        """``(ref, params)`` of the policy the body runs on *backend*."""
        entry = self.by_backend.get(backend)
        if entry is None:
            return self.ref, self.params
        return str(entry["ref"]), {k: v for k, v in entry.items() if k != "ref"}


@dataclass(frozen=True)
class NavSpec:
    """``base.nav``: what the in-process follower (sim/nav/follow.py) plans and drives
    with: the planning radius, forward speed and yaw rate, the goal tolerance (None: the
    follower's own) and the explorer the body asks for (``tare`` or the coverage tour)."""

    body_radius: float
    speed: float
    vyaw_max: float
    explorer: str = ""
    goal_tol: float | None = None


@dataclass(frozen=True)
class BaseSpec:
    """A legged base (``base:``): what the ONE generic ``LeggedBase`` drives.

    ``joints`` in the order the policy sees them; ``stance`` and every posture give an
    angle for each. ``height``: the root height the body stands at in its stance.
    ``limits``: the largest (vx, vy, vyaw) a command may ask. Fallen: the root below
    ``fall_z``, or its up axis tilted more than ``fall_tilt_deg`` from the world's.
    ``kp`` / ``kd``: the PD gains (one for every joint, or one per joint in ``joints``
    order) of postures and stance holds; their torques are clipped to ``torque_scale``
    x the model's motor range.
    """

    height: float
    holonomic: bool
    limits: tuple[float, float, float]
    fall_z: float
    fall_tilt_deg: float
    joints: tuple[str, ...]
    stance: dict[str, float]
    postures: dict[str, dict[str, float]]
    kp: float | tuple[float, ...]
    kd: float | tuple[float, ...]
    policy: BasePolicySpec
    nav: NavSpec
    torque_scale: float = 1.0


@dataclass(frozen=True)
class GripperSpec:
    """A position-controlled jaw, by its joint (the driver commands the joint's own
    position actuator).

    ``finger_bodies``: every one must touch an object before a grasp counts (the
    contact gate; each needs a colliding geom). ``weld_body``: the body a grasped
    object is attached to. ``min_opening``: the jaws must be held at least this far
    open (joint units past ``closed``) by the object. ``assist=False`` disables grasp
    constraints; ``speed`` limits set-point travel in joint units per second.
    """

    joint: str
    open: float
    closed: float
    finger_bodies: tuple[str, ...]
    weld_body: str
    min_opening: float
    assist: bool = True
    speed: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.assist, bool):
            raise EmbodimentConfigError("gripper.assist must be boolean")
        if self.speed is not None and (not math.isfinite(self.speed) or self.speed <= 0):
            raise EmbodimentConfigError("gripper.speed must be finite and positive")


@dataclass(frozen=True)
class ArmSpec:
    """An arm: its joints in order, the end-effector site, home, IK seeds, the move
    tolerance and its gripper. Every name is the compiled scene's. Optional velocity,
    acceleration and jerk bounds select a quintic command profile; actual dynamic
    tracking and hardware limits still need independent verification."""

    joints: tuple[str, ...]
    ee_site: str
    home: tuple[float, ...]
    ik_seeds: tuple[tuple[float, ...], ...] = ()
    # move_joints succeeds when every joint settles within this (rad) of its target.
    track_tol: float = 0.05
    gripper: GripperSpec | None = None
    postures: dict[str, tuple[float, ...]] = field(default_factory=dict)
    velocity: tuple[float, ...] = ()
    acceleration: tuple[float, ...] = ()
    jerk: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        limits = (self.velocity, self.acceleration, self.jerk)
        if any(limits) and any(len(v) != self.dof or any(not math.isfinite(x) or x <= 0 for x in v)
                               for v in limits):
            raise EmbodimentConfigError("arm velocity, acceleration and jerk must each give a finite positive limit per joint")

    @property
    def dof(self) -> int:
        return len(self.joints)


@dataclass(frozen=True)
class CapabilityProfile:
    """Uniform capability flags — what this body can do (Rule 11: no method drift)."""

    has_base: bool = True
    has_arm: bool = False
    has_gripper: bool = False
    camera: bool = False
    lidar: bool = False


@dataclass(frozen=True)
class AssetsSpec:
    """Fetch recipe for files that are not shipped (charter section 5).

    ``files``: asset-folder path -> path in the upstream repository (a file or a
    folder). ``sha256``: every file the robot loads, by asset-folder path; exactly
    these are copied. ``license``: ``{spdx, file}`` of the upstream repository.
    """

    name: str
    git: str
    rev: str
    files: dict[str, str]
    sha256: dict[str, str]
    license: dict[str, str]

    def upstream(self, rel: str) -> str:
        """The upstream path of asset file *rel* (through its longest ``files`` entry)."""
        key = max((k for k in self.files if _under(rel, k)), key=len)
        return self.files[key] + rel[len(key):]


def _under(rel: str, key: str) -> bool:
    return rel == key or rel.startswith(key + "/")


@dataclass(frozen=True)
class EmbodimentConfig:
    """Full, immutable robot morphology; its placement belongs to the world."""

    id: str
    display_name: str
    model: ModelSpec
    sensors: tuple[CameraSpec | LidarSpec, ...]
    capabilities: CapabilityProfile
    # Arm mounted on the body (e.g. the Piper on go2_piper), or None.
    arm: ArmSpec | None = None
    # The legged base the generic LeggedBase drives, or None.
    base: BaseSpec | None = None
    # The world it stands in: worlds/<world>/world.yaml (rooms, surfaces, pickables).
    world: str = ""
    # Files not shipped in the package; model.path, model.meshes_dir and the policy's
    # files are then relative to the asset folder (embodiments/assets.py resolves it).
    assets: AssetsSpec | None = None
    skills: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def camera(self) -> CameraSpec | None:
        """The body's first camera, or None."""
        return next((s for s in self.sensors if isinstance(s, CameraSpec)), None)

    @property
    def lidar(self) -> LidarSpec | None:
        """The body's lidar, or None."""
        return next((s for s in self.sensors if isinstance(s, LidarSpec)), None)


# ---------------------------------------------------------------------------
# Loader — fail-loud
# ---------------------------------------------------------------------------


def _require(raw: dict[str, Any], key: str, ctx: str) -> Any:
    """Return raw[key] or raise a clear, embodiment-scoped error."""
    if key not in raw or raw[key] is None:
        raise EmbodimentConfigError(
            f"{ctx}: missing required field '{key}'. Present keys: {sorted(raw)}"
        )
    return raw[key]


def _keys(raw: Any, ctx: str, valid: set[str]) -> dict[str, Any]:
    """*raw* when it is a mapping of keys in *valid*; else fail naming the key and the set."""
    if not isinstance(raw, dict):
        raise EmbodimentConfigError(f"{ctx} must be a mapping, got {type(raw).__name__}")
    for key in raw:
        if key not in valid:
            raise EmbodimentConfigError(f"{ctx}: unknown key {key!r}; valid: {sorted(valid)}")
    return raw


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise EmbodimentConfigError(f"expected a list, got {type(value).__name__}: {value!r}")


def _floats(value: Any, n: int, ctx: str) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in _as_tuple(value))
    except (TypeError, ValueError) as exc:
        raise EmbodimentConfigError(f"{ctx}: expected {n} finite numbers: {exc}") from None
    if len(out) != n:
        raise EmbodimentConfigError(f"{ctx} must have {n} values, got {len(out)}: {value!r}")
    if not all(math.isfinite(v) for v in out):
        raise EmbodimentConfigError(f"{ctx} must contain finite numbers")
    return out


def _optional_mapping(raw: dict[str, Any], key: str, ctx: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EmbodimentConfigError(f"{ctx}.{key} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _parse_model(raw: dict[str, Any], ctx: str) -> ModelSpec:
    mctx = f"{ctx}.model"
    m = _keys(_require(raw, "model", ctx), mctx, {"path", "root_body", "meshes_dir"})
    return ModelSpec(
        path=str(_require(m, "path", mctx)),
        root_body=str(_require(m, "root_body", mctx)),
        meshes_dir=str(m.get("meshes_dir", "")),
    )


_CAMERA_KEYS = {"role", "name", "width", "height", "depth", "body", "pos", "quat", "fovy"}


def _parse_camera(c: dict[str, Any], ctx: str) -> CameraSpec:
    _keys(c, ctx, _CAMERA_KEYS)
    placed = c.get("pos") is not None
    for key in ("body", "quat", "fovy"):
        if c.get(key) is not None and not placed:
            raise EmbodimentConfigError(f"{ctx}.{key} places a camera in the scene: it needs pos too")
    depth = c.get("depth")
    return CameraSpec(
        name=str(_require(c, "name", ctx)),
        width=int(_require(c, "width", ctx)),
        height=int(_require(c, "height", ctx)),
        depth=None if depth is None else _floats(depth, 2, f"{ctx}.depth"),  # type: ignore[arg-type]
        body=None if c.get("body") is None else str(c["body"]),
        pos=_floats(c["pos"], 3, f"{ctx}.pos") if placed else None,  # type: ignore[arg-type]
        quat=_floats(c.get("quat") or (1.0, 0.0, 0.0, 0.0), 4, f"{ctx}.quat"),  # type: ignore[arg-type]
        fovy=float(c.get("fovy") or 45.0),
    )


_LIDAR_KEYS = {"role", "body", "pos", "tilt_deg", "elevations_deg", "n_azimuth", "range", "period_s"}


def _parse_lidar(s: dict[str, Any], ctx: str) -> LidarSpec:
    _keys(s, ctx, _LIDAR_KEYS)
    rings = tuple(float(v) for v in _as_tuple(_require(s, "elevations_deg", ctx)))
    if 0.0 not in rings:
        raise EmbodimentConfigError(f"{ctx}.elevations_deg must include 0 (the 2-D scan ring), got {rings!r}")
    return LidarSpec(
        body=str(_require(s, "body", ctx)),
        pos=_floats(_require(s, "pos", ctx), 3, f"{ctx}.pos"),  # type: ignore[arg-type]
        tilt_deg=float(_require(s, "tilt_deg", ctx)),
        elevations_deg=rings,
        n_azimuth=int(_require(s, "n_azimuth", ctx)),
        range=float(_require(s, "range", ctx)),
        period_s=float(_require(s, "period_s", ctx)),
    )


def _parse_sensors(raw: dict[str, Any], ctx: str) -> tuple[CameraSpec | LidarSpec, ...]:
    parsers = {"camera": _parse_camera, "lidar": _parse_lidar}
    out: list[CameraSpec | LidarSpec] = []
    for i, sraw in enumerate(raw.get("sensors", []) or []):
        sctx = f"{ctx}.sensors[{i}]"
        role = _require(sraw, "role", sctx)
        if role not in parsers:
            raise EmbodimentConfigError(f"{sctx}.role: unknown role {role!r}; valid: {sorted(parsers)}")
        out.append(parsers[role](sraw, sctx))
    return tuple(out)


_BASE_KEYS = {"height", "holonomic", "limits", "fall_z", "fall_tilt_deg", "joints", "stance", "postures",
              "kp", "kd", "torque_scale", "policy", "nav"}
_POLICY_KEYS = {"ref", "timestep", "tick_hz", "by_backend"}  # the rest is the policy's own
_NAV_KEYS = {"body_radius", "speed", "vyaw_max", "explorer", "goal_tol"}


def _angles(raw: Any, joints: tuple[str, ...], ctx: str) -> dict[str, float]:
    """A ``{joint: rad}`` mapping that names exactly *joints*."""
    got = _keys(raw, ctx, set(joints))
    missing = [j for j in joints if j not in got]
    if missing:
        raise EmbodimentConfigError(f"{ctx}: no angle for {missing}; every base joint needs one")
    return {j: float(got[j]) for j in joints}


def _gain(raw: Any, joints: tuple[str, ...], ctx: str) -> float | tuple[float, ...]:
    """A PD gain: one number for every joint, or ``{joint: gain}`` naming exactly
    *joints* (a tuple in their order)."""
    if isinstance(raw, dict):
        got = _angles(raw, joints, ctx)
        return tuple(got[j] for j in joints)
    return float(raw)


def _parse_base_policy(p: dict[str, Any], ctx: str) -> BasePolicySpec:
    timestep, tick_hz = float(_require(p, "timestep", ctx)), float(_require(p, "tick_hz", ctx))
    import math

    if not all(math.isfinite(v) and v > 0 for v in (timestep, tick_hz)):
        raise EmbodimentConfigError(f"{ctx}: timestep and tick_hz must be finite and positive")
    substeps = 1.0 / (timestep * tick_hz)
    if not math.isfinite(substeps) or round(substeps) < 1 or abs(substeps - round(substeps)) > 1e-9:
        raise EmbodimentConfigError(
            f"{ctx}: tick_hz={tick_hz} must divide the physics rate 1/timestep={1 / timestep:g} (whole substeps)")
    by_backend = _optional_mapping(p, "by_backend", ctx) or {}
    for name, entry in by_backend.items():
        if not isinstance(entry, dict) or "ref" not in entry:
            raise EmbodimentConfigError(f"{ctx}.by_backend.{name} must be a mapping with a 'ref'")
    spec = BasePolicySpec(
        ref=str(_require(p, "ref", ctx)), timestep=timestep, tick_hz=tick_hz,
        params={k: v for k, v in p.items() if k not in _POLICY_KEYS},
        by_backend={str(k): dict(v) for k, v in by_backend.items()},
    )
    return spec



def _parse_base(raw: dict[str, Any], ctx: str) -> BaseSpec | None:
    b = _optional_mapping(raw, "base", ctx)
    if b is None:
        return None
    bctx = f"{ctx}.base"
    _keys(b, bctx, _BASE_KEYS)
    joints = tuple(str(v) for v in _as_tuple(_require(b, "joints", bctx)))
    if not joints or len(set(joints)) != len(joints):
        raise EmbodimentConfigError(f"{bctx}.joints must name each base joint once, got {joints!r}")
    postures = _optional_mapping(b, "postures", bctx) or {}
    if "stand" in postures:
        raise EmbodimentConfigError(f"{bctx}.postures: 'stand' is the stance; name it only there")
    limits = _keys(_require(b, "limits", bctx), f"{bctx}.limits", {"vx", "vy", "vyaw"})
    nav = _keys(_require(b, "nav", bctx), f"{bctx}.nav", _NAV_KEYS)
    nctx = f"{bctx}.nav"
    return BaseSpec(
        height=float(_require(b, "height", bctx)),
        holonomic=bool(_require(b, "holonomic", bctx)),
        limits=tuple(float(_require(limits, k, f"{bctx}.limits")) for k in ("vx", "vy", "vyaw")),  # type: ignore[arg-type]
        fall_z=float(_require(b, "fall_z", bctx)),
        fall_tilt_deg=float(_require(b, "fall_tilt_deg", bctx)),
        joints=joints,
        stance=_angles(_require(b, "stance", bctx), joints, f"{bctx}.stance"),
        postures={str(n): _angles(q, joints, f"{bctx}.postures.{n}") for n, q in postures.items()},
        kp=_gain(_require(b, "kp", bctx), joints, f"{bctx}.kp"),
        kd=_gain(_require(b, "kd", bctx), joints, f"{bctx}.kd"),
        torque_scale=float(b.get("torque_scale", 1.0)),
        policy=_parse_base_policy(_optional_mapping(b, "policy", bctx) or _require(b, "policy", bctx),
                                  f"{bctx}.policy"),
        nav=NavSpec(
            body_radius=float(_require(nav, "body_radius", nctx)),
            speed=float(_require(nav, "speed", nctx)),
            vyaw_max=float(_require(nav, "vyaw_max", nctx)),
            explorer=str(nav.get("explorer") or ""),
            goal_tol=None if nav.get("goal_tol") is None else float(nav["goal_tol"]),
        ),
    )


_ARM_KEYS = {"joints", "ee", "home", "ik_seeds", "track_tol", "gripper", "postures", "velocity", "acceleration", "jerk"}
_GRIPPER_KEYS = {"joint", "open", "closed", "finger_bodies", "weld_body", "min_opening", "assist", "speed"}


def _parse_gripper(g: dict[str, Any], ctx: str) -> GripperSpec:
    _keys(g, ctx, _GRIPPER_KEYS)
    fingers = tuple(str(v) for v in _as_tuple(_require(g, "finger_bodies", ctx)))
    if not fingers:
        raise EmbodimentConfigError(f"{ctx}.finger_bodies must name at least one body")
    return GripperSpec(
        joint=str(_require(g, "joint", ctx)),
        open=float(_require(g, "open", ctx)),
        closed=float(_require(g, "closed", ctx)),
        finger_bodies=fingers,
        weld_body=str(_require(g, "weld_body", ctx)),
        min_opening=float(_require(g, "min_opening", ctx)),
        assist=g.get("assist", True),
        speed=None if g.get("speed") is None else float(g["speed"]),
    )


def _parse_arm(raw: dict[str, Any], ctx: str) -> ArmSpec | None:
    a = _optional_mapping(raw, "arm", ctx)
    if a is None:
        return None
    actx = f"{ctx}.arm"
    _keys(a, actx, _ARM_KEYS)
    joints = tuple(str(v) for v in _as_tuple(_require(a, "joints", actx)))
    ee = _keys(_require(a, "ee", actx), f"{actx}.ee", {"site"})
    g = _optional_mapping(a, "gripper", actx)
    return ArmSpec(
        joints=joints,
        ee_site=str(_require(ee, "site", f"{actx}.ee")),
        home=_floats(_require(a, "home", actx), len(joints), f"{actx}.home"),
        ik_seeds=tuple(
            _floats(s, len(joints), f"{actx}.ik_seeds[{i}]") for i, s in enumerate(a.get("ik_seeds") or ())
        ),
        track_tol=float(a.get("track_tol", 0.05)),
        gripper=None if g is None else _parse_gripper(g, f"{actx}.gripper"),
        postures={str(n): _floats(q, len(joints), f"{actx}.postures.{n}")
                  for n, q in (_optional_mapping(a, "postures", actx) or {}).items()},
        **{k: _floats(a[k], len(joints), f"{actx}.{k}") if k in a else ()
           for k in ("velocity", "acceleration", "jerk")},
    )


_TOP_KEYS = {"schema", "id", "display_name", "world", "model", "sensors", "assets", "arm",
             "base", "skills"}

_GRASP_POSITIVE = {"pre_grasp_height", "lift_height", "reach_m", "view_duration", "hold_duration"}
_GRASP_NONNEGATIVE = {"grasp_z_above", "approach_back"}
_GRASP_BASE_COUNTS = {"approach_steps", "seat_steps", "scan_steps", "retries"}
_GRASP_BASE_POSITIVE = {"standoff", "view_standoff", "step_speed", "step_duration", "stall_distance", "seat_distance",
                        "lateral_gain", "lateral_speed", "lateral_deadband", "yaw_speed", "yaw_deadband",
                        "scan_step", "scan_range"}


def _positive(raw: Any, ctx: str, *, zero: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise EmbodimentConfigError(f"{ctx}: expected a finite number") from None
    if isinstance(raw, bool) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise EmbodimentConfigError(f"{ctx}: expected a finite {'nonnegative' if zero else 'positive'} number")
    return value


def _parse_skills(raw: dict[str, Any], ctx: str, arm: ArmSpec | None) -> dict[str, dict[str, Any]]:
    skills = _keys(_optional_mapping(raw, "skills", ctx) or {}, f"{ctx}.skills", {"grasp"})
    if "grasp" not in skills:
        return {}
    gctx = f"{ctx}.skills.grasp"
    if arm is None or arm.gripper is None:
        raise EmbodimentConfigError(f"{gctx}: requires an arm with a gripper")
    g = _keys(skills["grasp"], gctx, _GRASP_POSITIVE | _GRASP_NONNEGATIVE | {"durations", "base"})
    parsed = {k: _positive(_require(g, k, gctx), f"{gctx}.{k}", zero=k in _GRASP_NONNEGATIVE)
              for k in _GRASP_POSITIVE | _GRASP_NONNEGATIVE}
    durations = _floats(_require(g, "durations", gctx), 4, f"{gctx}.durations")
    if any(d <= 0 for d in durations):
        raise EmbodimentConfigError(f"{gctx}.durations: all four motion durations must be positive")
    parsed["durations"] = durations
    if "base" in g:
        bctx = f"{gctx}.base"
        if raw.get("base") is None:
            raise EmbodimentConfigError(f"{bctx}: requires a base")
        b = _keys(g["base"], bctx, _GRASP_BASE_COUNTS | _GRASP_BASE_POSITIVE)
        values = {k: _positive(_require(b, k, bctx), f"{bctx}.{k}", zero=k in _GRASP_BASE_COUNTS)
                  for k in _GRASP_BASE_COUNTS | _GRASP_BASE_POSITIVE}
        for k in _GRASP_BASE_COUNTS:
            if not values[k].is_integer():
                raise EmbodimentConfigError(f"{bctx}.{k}: expected a nonnegative integer")
            values[k] = int(values[k])
        parsed["base"] = values
    return {"grasp": parsed}


def _parse_assets(raw: dict[str, Any], ctx: str) -> AssetsSpec | None:
    a = _optional_mapping(raw, "assets", ctx)
    if a is None:
        return None
    actx = f"{ctx}.assets"
    blocks = {k: _optional_mapping(a, k, actx) for k in ("files", "sha256", "license")}
    for key, val in blocks.items():
        if not val:
            raise EmbodimentConfigError(f"{actx}.{key}: required, a non-empty mapping")
    lic = blocks["license"]
    spec = AssetsSpec(
        name=str(_require(a, "name", actx)),
        git=str(_require(a, "git", actx)),
        rev=str(_require(a, "rev", actx)),
        files={str(k): str(v) for k, v in blocks["files"].items()},
        sha256={str(k): str(v) for k, v in blocks["sha256"].items()},
        license={k: str(_require(lic, k, f"{actx}.license")) for k in ("spdx", "file")},
    )
    if len(spec.rev) != 40:
        raise EmbodimentConfigError(f"{actx}.rev must be a full 40-hex commit, got {spec.rev!r}")
    for rel in spec.sha256:
        if not any(_under(rel, k) for k in spec.files):
            raise EmbodimentConfigError(f"{actx}.sha256: {rel!r} lies under no files entry {sorted(spec.files)}")
    for key in spec.files:
        if not any(_under(rel, key) for rel in spec.sha256):
            raise EmbodimentConfigError(f"{actx}.files: {key!r} covers no file listed in sha256")
    return spec


def parse_embodiment_config(raw: dict[str, Any], ctx: str = "robot.yaml") -> EmbodimentConfig:
    """Parse an already-loaded dict into an EmbodimentConfig (fail-loud, unknown keys too)."""
    _keys(raw, ctx, _TOP_KEYS)
    version = _require(raw, "schema", ctx)
    if type(version) is not int or version != 1:
        advice = 'upgrade vector_os_nano' if type(version) is int and version > 1 else 'use schema: 1'
        raise EmbodimentConfigError(f"{ctx}.schema: {version!r}; valid: [1]; {advice}")
    arm = _parse_arm(raw, ctx)
    base = _parse_base(raw, ctx)
    sensors = _parse_sensors(raw, ctx)
    caps = CapabilityProfile(has_base=base is not None, has_arm=arm is not None,
                             has_gripper=arm is not None and arm.gripper is not None,
                             camera=any(isinstance(s, CameraSpec) for s in sensors),
                             lidar=any(isinstance(s, LidarSpec) for s in sensors))
    return EmbodimentConfig(
        id=str(_require(raw, "id", ctx)),
        display_name=str(_require(raw, "display_name", ctx)),
        model=_parse_model(raw, ctx),
        sensors=sensors,
        capabilities=caps,
        arm=arm,
        world=str(raw.get("world", "") or ""),
        assets=_parse_assets(raw, ctx),
        base=base,
        skills=_parse_skills(raw, ctx, arm),
    )


def load_embodiment_config(embodiment_id: str) -> EmbodimentConfig:
    """Load ``vector_os_nano/embodiments/<embodiment_id>/robot.yaml``.

    Raises:
        EmbodimentConfigError: if the file is missing, unreadable, not a mapping,
            missing a required field or carries an unknown key (fail-loud, Rule 8).
    """
    yaml_path = _EMBODIMENTS_DIR / embodiment_id / "robot.yaml"
    if not yaml_path.is_file():
        raise EmbodimentConfigError(
            f"embodiment '{embodiment_id}': robot.yaml not found at {yaml_path}"
        )
    try:
        raw = yaml.load(yaml_path.read_text(), Loader=yaml_loader(yaml_path, EmbodimentConfigError))
    except yaml.YAMLError as exc:  # pragma: no cover - malformed-yaml guard
        raise EmbodimentConfigError(
            f"embodiment '{embodiment_id}': failed to parse {yaml_path}: {exc}"
        ) from exc
    if raw is None:
        raise EmbodimentConfigError(
            f"embodiment '{embodiment_id}': {yaml_path} is empty"
        )
    return parse_embodiment_config(raw, ctx=f"{embodiment_id}/robot.yaml")
