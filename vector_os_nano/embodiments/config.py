# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Typed, frozen schema + loader for per-embodiment ``robot.yaml`` manifests.

A robot morphology is data, not code (project CLAUDE.md Rule 11). This module
parses ``vector_os_nano/embodiments/<id>/robot.yaml`` into immutable dataclasses
so the generic embodiment driver (Stage 2) can READ every morphology-specific
parameter — spawn pose, nominal stance (by joint name), sensor mounts, root body,
gait/policy reference, capability profile — instead of hardcoding it.

Design notes:
  * All dataclasses are ``frozen=True`` (Rule 6 — additive change only: new field
    last, with a default).
  * The loader is FAIL-LOUD: a missing file or a missing required key raises a
    clear error naming the embodiment + the offending key, never a silent default.
  * No MuJoCo / ROS2 imports — pure data, importable + testable offline.
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
# Schema — frozen dataclasses (Rule 6: additive only — new field LAST + default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Where the body's MuJoCo model + meshes live, and its root (free) body."""

    path: str
    root_body: str
    meshes_dir: str = ""


@dataclass(frozen=True)
class SpawnSpec:
    """Where the body is placed in the scene on connect()."""

    xy: tuple[float, float]
    base_height: float
    heading: float = 0.0


@dataclass(frozen=True)
class SensorSpec:
    """One sensor mounted on the body (camera, depth, lidar, …).

    role: logical role ('camera', 'depth', 'lidar', …) — what the runtime routes to.
    mount_body: the body the sensor is rigidly attached to.
    name: the named MuJoCo element (camera name), if any.
    pos / euler: mount offset (metres / radians) from ``mount_body``, if any.
    params: sensor-specific tuning (tilt, fov, update interval, …).
    """

    role: str
    mount_body: str
    name: str | None = None
    pos: tuple[float, ...] | None = None
    euler: tuple[float, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySpec:
    """How the body is driven (RL policy / MPC / scripted gait)."""

    ref: str
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspSpec:
    """Manipulation surface (only for bodies with a gripper)."""

    gripper_body: str
    graspable: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityProfile:
    """Uniform capability flags — what this body can do (Rule 11: no method drift)."""

    has_base: bool = True
    has_arm: bool = False
    has_gripper: bool = False
    camera: bool = False
    lidar: bool = False


@dataclass(frozen=True)
class EmbodimentConfig:
    """Full, immutable description of one robot morphology.

    Faithfully captures what a per-robot driver currently hardcodes, so the
    generic driver (Stage 2) can stand the body up from data alone.
    """

    id: str
    display_name: str
    model: ModelSpec
    spawn: SpawnSpec
    stance: dict[str, float]
    sensors: tuple[SensorSpec, ...]
    policy: PolicySpec
    capabilities: CapabilityProfile
    grasp: GraspSpec | None = None
    # Which in-process driver stands the body up (mujoco_go2 | mujoco_g1 |
    # mujoco_arm). Transitional: removed once one generic driver reads the
    # manifest (WP17).
    driver: str = ""
    # Arm mounted on the body (e.g. the Piper on go2_piper), or None.
    arm: dict[str, Any] | None = None
    # Navigation capabilities the body asks for, e.g. {"explorer": "tare"}.
    nav: dict[str, Any] = field(default_factory=dict)


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


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise EmbodimentConfigError(f"expected a list, got {type(value).__name__}: {value!r}")


def _parse_model(raw: dict[str, Any], ctx: str) -> ModelSpec:
    m = _require(raw, "model", ctx)
    mctx = f"{ctx}.model"
    return ModelSpec(
        path=str(_require(m, "path", mctx)),
        root_body=str(_require(m, "root_body", mctx)),
        meshes_dir=str(m.get("meshes_dir", "")),
    )


def _parse_spawn(raw: dict[str, Any], ctx: str) -> SpawnSpec:
    s = _require(raw, "spawn", ctx)
    sctx = f"{ctx}.spawn"
    xy = _as_tuple(_require(s, "xy", sctx))
    if len(xy) != 2:
        raise EmbodimentConfigError(f"{sctx}.xy must have exactly 2 values, got {xy!r}")
    return SpawnSpec(
        xy=(float(xy[0]), float(xy[1])),
        base_height=float(_require(s, "base_height", sctx)),
        heading=float(s.get("heading", 0.0)),
    )


def _parse_sensors(raw: dict[str, Any], ctx: str) -> tuple[SensorSpec, ...]:
    out: list[SensorSpec] = []
    for i, sraw in enumerate(raw.get("sensors", []) or []):
        sctx = f"{ctx}.sensors[{i}]"
        pos = sraw.get("pos")
        euler = sraw.get("euler")
        out.append(
            SensorSpec(
                role=str(_require(sraw, "role", sctx)),
                mount_body=str(_require(sraw, "mount_body", sctx)),
                name=(str(sraw["name"]) if sraw.get("name") is not None else None),
                pos=(tuple(float(v) for v in _as_tuple(pos)) if pos is not None else None),
                euler=(tuple(float(v) for v in _as_tuple(euler)) if euler is not None else None),
                params=dict(sraw.get("params", {}) or {}),
            )
        )
    return tuple(out)


def _parse_policy(raw: dict[str, Any], ctx: str) -> PolicySpec:
    p = _require(raw, "policy", ctx)
    pctx = f"{ctx}.policy"
    return PolicySpec(
        ref=str(_require(p, "ref", pctx)),
        spec=dict(p.get("spec", {}) or {}),
    )


def _parse_capabilities(raw: dict[str, Any], ctx: str) -> CapabilityProfile:
    c = _require(raw, "capabilities", ctx)
    return CapabilityProfile(
        has_base=bool(c.get("has_base", True)),
        has_arm=bool(c.get("has_arm", False)),
        has_gripper=bool(c.get("has_gripper", False)),
        camera=bool(c.get("camera", False)),
        lidar=bool(c.get("lidar", False)),
    )


def _parse_grasp(raw: dict[str, Any], ctx: str) -> GraspSpec | None:
    g = raw.get("grasp")
    if g is None:
        return None
    gctx = f"{ctx}.grasp"
    return GraspSpec(
        gripper_body=str(_require(g, "gripper_body", gctx)),
        graspable=tuple(str(v) for v in _as_tuple(g.get("graspable"))),
    )


def _optional_mapping(raw: dict[str, Any], key: str, ctx: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EmbodimentConfigError(f"{ctx}.{key} must be a mapping, got {type(value).__name__}")
    return dict(value)


def parse_embodiment_config(raw: dict[str, Any], ctx: str = "robot.yaml") -> EmbodimentConfig:
    """Parse an already-loaded dict into an EmbodimentConfig (fail-loud)."""
    if not isinstance(raw, dict):
        raise EmbodimentConfigError(f"{ctx}: top-level YAML must be a mapping, got {type(raw).__name__}")
    stance_raw = _require(raw, "stance", ctx)
    if not isinstance(stance_raw, dict):
        raise EmbodimentConfigError(f"{ctx}.stance must be a mapping of joint_name -> angle")
    return EmbodimentConfig(
        id=str(_require(raw, "id", ctx)),
        display_name=str(_require(raw, "display_name", ctx)),
        model=_parse_model(raw, ctx),
        spawn=_parse_spawn(raw, ctx),
        stance={str(k): float(v) for k, v in stance_raw.items()},
        sensors=_parse_sensors(raw, ctx),
        policy=_parse_policy(raw, ctx),
        capabilities=_parse_capabilities(raw, ctx),
        grasp=_parse_grasp(raw, ctx),
        driver=str(raw.get("driver", "") or ""),
        arm=_optional_mapping(raw, "arm", ctx),
        nav=_optional_mapping(raw, "nav", ctx) or {},
    )


def load_embodiment_config(embodiment_id: str) -> EmbodimentConfig:
    """Load ``vector_os_nano/embodiments/<embodiment_id>/robot.yaml``.

    Raises:
        EmbodimentConfigError: if the file is missing, unreadable, not a mapping,
            or missing any required field (fail-loud, Rule 8 — clear error, no
            silent default).
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
