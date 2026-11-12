# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Generic DoF-layout introspection for a free-floating robot in a MuJoCo scene.

A robot morphology is data, not code (project CLAUDE.md Rule 11). The per-robot
``mujoco_<id>.py`` drivers used to hardcode the qpos/qvel slice indices of the
floating base + leg joints (go2: ``qpos[0:3]`` spawn, ``qpos[7:19]`` legs; g1: a
bespoke ``_G1Offsets`` class). Those are NOT morphology constants — they are
mechanically derivable from the compiled model by locating the root body's
freejoint. This module is the ONE shared, config-parameterized implementation
(``DofLayout``) that replaces both, so adding/changing a robot needs no driver
edit (Rule 11): the driver passes ``root_body`` + ``num_actuated`` from the
embodiment manifest and reads the slice addresses back.

``DofLayout`` generalizes ``_G1Offsets``:
  * For the go2 (root_body="base_link", first freejoint in the room scene) it
    yields ``root_qpos_adr=0`` / ``joint_qpos_start=7`` — exactly the literals
    the go2 driver hardcoded.
  * For the g1 (root_body="g1_pelvis", attached with a "g1_" prefix) it yields
    the same introspected addresses ``_G1Offsets`` computed.

It also builds the nominal-stance vector FROM the manifest's ``stance`` dict, in
the model's actual leg-qpos order (matched by joint-name suffix so the "g1_"
attach prefix is transparent), in float64: for g1 it equals ``_DEFAULT_ANGLES``.
The go2 runs on the generic LeggedBase (canonical addresses); only the G1 driver
reads this module, until it does too (S4b).

No torch import; ``mujoco`` is imported lazily so the module stays importable
offline (the pure-logic ``build_stance_vector`` needs only a model object, which
tests can supply via a tiny in-memory MJCF).
"""
from __future__ import annotations

from typing import Any

import numpy as np

_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


def build_robot_geom_set(model: Any, root_bid: int) -> set[int]:
    """Return geom ids belonging to the robot body subtree rooted at ``root_bid``.

    Walks each geom's body up the kinematic tree to ``root_bid`` (mj_ray's
    ``bodyexclude`` filters only ONE body, so the lidar self-filter needs the
    full set). This is the generic form of the go2 ``_Go2Model`` inline loop and
    the g1 ``_build_robot_geom_set`` name-prefix loop, unified — subtree
    membership is morphology-agnostic and needs no name convention.
    """
    robot_geom_ids: set[int] = set()
    for gid in range(model.ngeom):
        check_bid = int(model.geom_bodyid[gid])
        while check_bid > 0:
            if check_bid == root_bid:
                robot_geom_ids.add(gid)
                break
            check_bid = int(model.body_parentid[check_bid])
    return robot_geom_ids


class DofLayout:
    """Pre-computed qpos/qvel offsets for a free-floating robot's root + joints.

    Locates ``root_body``'s freejoint in the COMPILED model and exposes every
    slice address the drivers need. ``num_actuated`` is the count of actuated
    (leg) joints that follow the 7-float floating-base block — 12 for both go2
    and g1.

    Attributes (all ints unless noted):
      root_qpos_adr   jnt_qposadr of the root freejoint (go2=0, g1 introspected)
      root_dof_adr    jnt_dofadr  of the root freejoint
      quat_start      root_qpos_adr + 3   (4-float w,x,y,z orientation quaternion)
      joint_qpos_start root_qpos_adr + 7  (leg qpos slice start; go2=7)
      angvel_start    root_dof_adr + 3    (3-float base angular-velocity slice)
      joint_dof_start root_dof_adr + 6    (leg qvel slice start; go2=6)
      root_bid        body id of ``root_body``
      robot_geom_ids  set[int] of all geoms in the root body's subtree
    """

    __slots__ = (
        "num_actuated",
        "root_bid",
        "root_qpos_adr",
        "root_dof_adr",
        "quat_start",
        "joint_qpos_start",
        "angvel_start",
        "joint_dof_start",
        "robot_geom_ids",
    )

    def __init__(self, model: Any, root_body: str, num_actuated: int) -> None:
        self.num_actuated: int = int(num_actuated)
        # Root body + its freejoint (the floating base).
        self.root_bid: int = int(model.body(root_body).id)
        jnt_adr = int(model.body_jntadr[self.root_bid])
        if jnt_adr < 0 or int(model.jnt_type[jnt_adr]) != int(_get_mujoco().mjtJoint.mjJNT_FREE):
            raise ValueError(f"DofLayout: root body {root_body!r} has no free joint (a fixed or "
                             f"jointed root cannot be addressed as a floating base)")
        self.root_qpos_adr: int = int(model.jnt_qposadr[jnt_adr])
        self.root_dof_adr: int = int(model.jnt_dofadr[jnt_adr])
        # Orientation quaternion (w,x,y,z): 4 floats after the 3 position floats.
        self.quat_start: int = self.root_qpos_adr + 3
        # Actuated joints start right after the 7-float root (pos[3] + quat[4]).
        self.joint_qpos_start: int = self.root_qpos_adr + 7
        # Base angular velocity: 3 floats after the 3 linear-velocity floats.
        self.angvel_start: int = self.root_dof_adr + 3
        # Actuated joint velocities start after the 6-DoF free joint.
        self.joint_dof_start: int = self.root_dof_adr + 6
        # All geoms in the root body's subtree (lidar self-filter).
        self.robot_geom_ids: set[int] = build_robot_geom_set(model, self.root_bid)

    # ------------------------------------------------------------------
    # Stance — built from the manifest, in the model's actual leg-qpos order
    # ------------------------------------------------------------------

    def _leg_joint_names_in_qpos_order(self, model: Any) -> list[str]:
        """Names of the ``num_actuated`` joints occupying the leg-qpos slice.

        Returns them sorted by qpos address, so the order EXACTLY matches the
        ``qpos[joint_qpos_start : joint_qpos_start + num_actuated]`` slice the
        drivers write — independent of actuator declaration order (go2's
        actuators are FR,FL,RR,RL while its qpos legs are FL,FR,RL,RR; building
        in qpos order is what makes the result byte-identical to ``_STAND_JOINTS``).
        """
        mj = _get_mujoco()
        lo = self.joint_qpos_start
        hi = self.joint_qpos_start + self.num_actuated
        found: list[tuple[int, str]] = []
        for jid in range(int(model.njnt)):
            qadr = int(model.jnt_qposadr[jid])
            if lo <= qadr < hi:
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or ""
                found.append((qadr, name))
        found.sort(key=lambda t: t[0])
        if len(found) != self.num_actuated:
            raise ValueError(
                f"DofLayout: expected {self.num_actuated} leg joints in qpos "
                f"slice [{lo}, {hi}), found {len(found)}: "
                f"{[n for _, n in found]}"
            )
        return [name for _, name in found]

    def build_stance_vector(
        self, model: Any, stance: dict[str, float]
    ) -> np.ndarray:
        """Assemble the ``num_actuated``-vector of nominal joint angles.

        For each leg joint (in qpos-slice order) looks up its angle in the
        manifest ``stance`` dict, matching by joint-name SUFFIX so the model's
        attach prefix (e.g. "g1_") is transparent — the manifest authors names
        in the un-prefixed model convention (``left_hip_pitch_joint``,
        ``FL_hip``). Fails loud (Rule 8) if any leg joint has no stance entry or
        an ambiguous match.

        The result, written to ``qpos[joint_qpos_start : +num_actuated]``, is
        byte-identical to the old hardcoded arrays (go2: ``_STAND_JOINTS``,
        g1: ``_DEFAULT_ANGLES``).
        """
        names = self._leg_joint_names_in_qpos_order(model)
        out = np.empty(self.num_actuated, dtype=np.float64)
        for i, jname in enumerate(names):
            out[i] = float(_resolve_stance_angle(jname, stance))
        return out


def _normalize_joint_token(name: str) -> str:
    """Strip a leading ``<embodiment>_`` attach prefix and a trailing ``_joint``.

    Maps both naming conventions to a common canonical token so a model joint
    and its manifest stance key compare equal:
      * go2 model joint ``FL_hip_joint``  -> ``FL_hip``; manifest key ``FL_hip``.
      * g1 model joint ``g1_left_hip_pitch_joint`` -> ``left_hip_pitch_joint``;
        manifest key ``left_hip_pitch_joint`` (already canonical).
    Only ONE leading prefix segment is stripped (``g1_`` / ``go2_``), never an
    intrinsic ``left_``/``right_``/``FL_`` segment, because those are part of the
    canonical name on BOTH sides — so a manifest key keeps its leading segment.
    """
    token = name[:-6] if name.endswith("_joint") else name
    return token


def _canonical_variants(key: str) -> set[str]:
    """All model-joint spellings a manifest ``key`` may legitimately appear as."""
    base = _normalize_joint_token(key)
    return {base, key}


def _resolve_stance_angle(joint_name: str, stance: dict[str, float]) -> float:
    """Look up ``joint_name``'s nominal angle in ``stance`` (naming-tolerant).

    Compares the model joint name against each manifest key after normalizing
    both (drop a trailing ``_joint``, tolerate a leading ``<embodiment>_`` attach
    prefix on the model side). This bridges the two conventions in play:
      * go2: model ``FL_hip_joint`` <-> manifest ``FL_hip``.
      * g1:  model ``g1_left_hip_pitch_joint`` <-> manifest ``left_hip_pitch_joint``.
    Raises on no-match or ambiguous-match (Rule 8 — clear error with the offending
    name + the valid set), never a silent default.
    """
    if joint_name in stance:
        return stance[joint_name]
    jtok = _normalize_joint_token(joint_name)
    matches: list[str] = []
    for key in stance:
        variants = _canonical_variants(key)
        ktok = _normalize_joint_token(key)
        # Exact (post-normalize) OR the key is the model name minus an attach
        # prefix (model token ends with "_<key token>").
        if (
            joint_name in variants
            or jtok == ktok
            or jtok.endswith("_" + ktok)
        ):
            matches.append(key)
    if matches:
        # Prefer the most specific (longest) canonical key when several apply.
        matches.sort(key=lambda k: len(_normalize_joint_token(k)), reverse=True)
        m0 = _normalize_joint_token(matches[0])
        if len(matches) == 1 or len(m0) > len(_normalize_joint_token(matches[1])):
            return stance[matches[0]]
        raise ValueError(
            f"DofLayout: ambiguous stance match for joint '{joint_name}': "
            f"{matches}. Disambiguate the manifest stance keys."
        )
    raise ValueError(
        f"DofLayout: no stance angle for leg joint '{joint_name}'. "
        f"Manifest stance keys: {sorted(stance)}"
    )
