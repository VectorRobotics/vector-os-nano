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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Directory holding the per-embodiment subfolders (go2/, g1/, …).
_EMBODIMENTS_DIR: Path = Path(__file__).resolve().parent


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
class SpawnSpec:
    """Where a body with a base is placed in the scene on connect()."""

    xy: tuple[float, float]
    base_height: float
    heading: float = 0.0


@dataclass(frozen=True)
class SensorSpec:
    """A non-camera sensor (lidar) mounted on the body: role, mount body, offset, params."""

    role: str
    mount_body: str
    name: str | None = None
    pos: tuple[float, ...] | None = None
    euler: tuple[float, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)


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
class PolicySpec:
    """How the body is driven (RL policy / MPC / scripted gait)."""

    ref: str
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperSpec:
    """A position-controlled jaw, by its joint (the driver commands the joint's own
    position actuator).

    ``finger_bodies``: every one must touch an object before a grasp counts (the
    contact gate; each needs a colliding geom). ``weld_body``: the body a grasped
    object is attached to. ``min_opening``: the jaws must be held at least this far
    open (joint units past ``closed``) by the object.
    """

    joint: str
    open: float
    closed: float
    finger_bodies: tuple[str, ...]
    weld_body: str
    min_opening: float


@dataclass(frozen=True)
class ArmSpec:
    """An arm: its joints in order, the end-effector site, home, IK seeds, the move
    tolerance and its gripper. Every name is the compiled scene's."""

    joints: tuple[str, ...]
    ee_site: str
    home: tuple[float, ...]
    ik_seeds: tuple[tuple[float, ...], ...] = ()
    # move_joints succeeds when every joint settles within this (rad) of its target.
    track_tol: float = 0.05
    gripper: GripperSpec | None = None

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
    """Full, immutable description of one robot morphology. ``spawn``, ``stance`` and
    ``policy`` belong to a body with a base; a fixed arm stands at its world's spawn."""

    id: str
    display_name: str
    model: ModelSpec
    sensors: tuple[SensorSpec | CameraSpec, ...]
    capabilities: CapabilityProfile
    spawn: SpawnSpec | None = None
    stance: dict[str, float] = field(default_factory=dict)
    policy: PolicySpec | None = None
    # Which in-process builder stands the body up (mujoco_go2 | mujoco_g1 | mujoco_arm |
    # generic). Transitional: one generic builder for every body (S4 / S5).
    driver: str = ""
    # Arm mounted on the body (e.g. the Piper on go2_piper), or None.
    arm: ArmSpec | None = None
    # Navigation capabilities the body asks for, e.g. {"explorer": "tare"}.
    nav: dict[str, Any] = field(default_factory=dict)
    # The world it stands in: config/worlds/<world>.yaml (rooms, surfaces, pickables).
    world: str = ""
    # Files not shipped in the package; model.path and policy.ref are then
    # relative to the asset folder (embodiments/assets.py resolves it).
    assets: AssetsSpec | None = None

    @property
    def camera(self) -> CameraSpec | None:
        """The body's first camera, or None."""
        return next((s for s in self.sensors if isinstance(s, CameraSpec)), None)


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
    out = tuple(float(v) for v in _as_tuple(value))
    if len(out) != n:
        raise EmbodimentConfigError(f"{ctx} must have {n} values, got {len(out)}: {value!r}")
    return out


def _optional_mapping(raw: dict[str, Any], key: str, ctx: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EmbodimentConfigError(f"{ctx}.{key} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _parse_model(raw: dict[str, Any], ctx: str) -> ModelSpec:
    m = _require(raw, "model", ctx)
    mctx = f"{ctx}.model"
    return ModelSpec(
        path=str(_require(m, "path", mctx)),
        root_body=str(_require(m, "root_body", mctx)),
        meshes_dir=str(m.get("meshes_dir", "")),
    )


def _parse_spawn(s: dict[str, Any], ctx: str) -> SpawnSpec:
    xy = _as_tuple(_require(s, "xy", ctx))
    if len(xy) != 2:
        raise EmbodimentConfigError(f"{ctx}.xy must have exactly 2 values, got {xy!r}")
    return SpawnSpec(
        xy=(float(xy[0]), float(xy[1])),
        base_height=float(_require(s, "base_height", ctx)),
        heading=float(s.get("heading", 0.0)),
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


def _parse_sensors(raw: dict[str, Any], ctx: str) -> tuple[SensorSpec | CameraSpec, ...]:
    out: list[SensorSpec | CameraSpec] = []
    for i, sraw in enumerate(raw.get("sensors", []) or []):
        sctx = f"{ctx}.sensors[{i}]"
        if _require(sraw, "role", sctx) == "camera":
            out.append(_parse_camera(sraw, sctx))
            continue
        pos = sraw.get("pos")
        euler = sraw.get("euler")
        out.append(SensorSpec(
            role=str(sraw["role"]),
            mount_body=str(_require(sraw, "mount_body", sctx)),
            name=(str(sraw["name"]) if sraw.get("name") is not None else None),
            pos=(tuple(float(v) for v in _as_tuple(pos)) if pos is not None else None),
            euler=(tuple(float(v) for v in _as_tuple(euler)) if euler is not None else None),
            params=dict(sraw.get("params", {}) or {}),
        ))
    return tuple(out)


def _parse_capabilities(raw: dict[str, Any], ctx: str) -> CapabilityProfile:
    c = _require(raw, "capabilities", ctx)
    return CapabilityProfile(
        has_base=bool(c.get("has_base", True)),
        has_arm=bool(c.get("has_arm", False)),
        has_gripper=bool(c.get("has_gripper", False)),
        camera=bool(c.get("camera", False)),
        lidar=bool(c.get("lidar", False)),
    )


_ARM_KEYS = {"joints", "ee", "home", "ik_seeds", "track_tol", "gripper"}
_GRIPPER_KEYS = {"joint", "open", "closed", "finger_bodies", "weld_body", "min_opening"}


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
    )


_TOP_KEYS = {"id", "display_name", "driver", "world", "model", "spawn", "stance", "sensors", "assets",
             "policy", "capabilities", "arm", "nav"}


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
    stance = raw.get("stance") or {}
    if not isinstance(stance, dict):
        raise EmbodimentConfigError(f"{ctx}.stance must be a mapping of joint_name -> angle")
    spawn = _optional_mapping(raw, "spawn", ctx)
    policy = _optional_mapping(raw, "policy", ctx)
    caps = _parse_capabilities(raw, ctx)
    if caps.has_base:  # a base stands at its own spawn, in its stance, driven by its policy
        for key in ("spawn", "stance", "policy"):
            _require(raw, key, ctx)
    return EmbodimentConfig(
        id=str(_require(raw, "id", ctx)),
        display_name=str(_require(raw, "display_name", ctx)),
        model=_parse_model(raw, ctx),
        sensors=_parse_sensors(raw, ctx),
        capabilities=caps,
        spawn=None if spawn is None else _parse_spawn(spawn, f"{ctx}.spawn"),
        stance={str(k): float(v) for k, v in stance.items()},
        policy=None if policy is None else PolicySpec(
            ref=str(_require(policy, "ref", f"{ctx}.policy")), spec=dict(policy.get("spec") or {})),
        driver=str(raw.get("driver", "") or ""),
        arm=_parse_arm(raw, ctx),
        nav=_optional_mapping(raw, "nav", ctx) or {},
        world=str(raw.get("world", "") or ""),
        assets=_parse_assets(raw, ctx),
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
        raw = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as exc:  # pragma: no cover - malformed-yaml guard
        raise EmbodimentConfigError(
            f"embodiment '{embodiment_id}': failed to parse {yaml_path}: {exc}"
        ) from exc
    if raw is None:
        raise EmbodimentConfigError(
            f"embodiment '{embodiment_id}': {yaml_path} is empty"
        )
    return parse_embodiment_config(raw, ctx=f"{embodiment_id}/robot.yaml")
