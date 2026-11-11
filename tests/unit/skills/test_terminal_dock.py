# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R40 (D54) — the CLOSED-LOOP terminal dock + POSE-VERIFICATION GATE (unit, no sim).

FAR un-parks the dog and brings it to within ~0.8 m of a goal at an ARBITRARY pose
(probe: heading ~169 deg off, or R39 t3: heading 86 deg + x back from the table → the
d435 frames the floor/wall → perception mislocalizes the can). The dock was NOT
repeatable (~1/3) because it ran a FIXED SEQUENCE and the quadruped gait DRIFTS.

R40 rebuilds the dock as a true CLOSED-LOOP controller (iterate: MEASURE the live pose
→ CORRECT toward the FIXED proven head-on pose) that returns a structured DockResult,
and adds a strict GATE (dock_converged): heading within ±12° of +X, |y - centerline| <
8 cm, x in the perceive band. A grasp proceeds ONLY when the gate passes; otherwise it
aborts cleanly "dock_not_converged" (an honest RAN, never a false grasp).

These tests pin the new contract WITHOUT a sim:
  1. from a RANGE of bad arrivals (incl. heading 86° / x back-from-table / off-center)
     the closed loop CONVERGES — final pose within the gate tolerances, dock_converged
     True. NEVER returns "ready" (converged) from a bad pose.
  2. a NON-convergeable arrival (pathological gait drift overwhelming each correction)
     → the gate FIRES: converged False after the iteration budget. The dock never
     claims convergence it did not achieve.
  3. the dock is a BENIGN no-op when already at the dock pose (scripted-from-spawn) —
     no walks, converged True — so D34-D51 is preserved.
  4. a base lacking the surface is skipped gracefully (ran False, no crash, no walks).
  5. the grasp wires the dock + GATE before the perceive: a non-converged dock aborts
     the grasp RAN (no perceive), a converged/benign dock proceeds.
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.skills.utils.terminal_dock import (
    _DOCK_LATERAL_DEADBAND_M,
    _DOCK_POS_DEADBAND_M,
    _GATE_HEADING_TOL_RAD,
    _GATE_LATERAL_TOL_M,
    _GATE_X_BAND_M,
    DockResult,
    dock_converged,
    terminal_dock,
)


# ---------------------------------------------------------------------------
# A kinematic base double: walk() integrates the commanded velocity over the
# duration, so a sequence of turn/walk commands moves a simulated (x, y, yaw).
# A per-step drift can be injected to prove the CLOSED loop still converges
# (it re-reads the live pose each step) — and, at large drift, that the GATE
# fires when convergence is impossible.
# ---------------------------------------------------------------------------

class _KinematicBase:
    def __init__(self, *, pose=(10.0, 3.0), heading=0.0, drift=0.0, drift_y=0.0):
        self._x, self._y = float(pose[0]), float(pose[1])
        self._yaw = float(heading)
        self._drift = float(drift)
        self._drift_y = float(drift_y)
        self.walks: list[tuple[float, float, float, float]] = []

    def get_position(self):
        return (self._x, self._y, 0.35)

    def get_heading(self):
        return self._yaw

    def walk(self, vx=0.0, vy=0.0, vyaw=0.0, duration=1.0):
        self.walks.append((float(vx), float(vy), float(vyaw), float(duration)))
        n = 8
        dt = duration / n
        for _ in range(n):
            self._yaw = math.atan2(
                math.sin(self._yaw + vyaw * dt), math.cos(self._yaw + vyaw * dt))
            self._x += vx * dt * math.cos(self._yaw) - vy * dt * math.sin(self._yaw)
            self._y += vx * dt * math.sin(self._yaw) + vy * dt * math.cos(self._yaw)
        self._x += self._drift
        self._y += self._drift_y
        return True


_DOCK_X, _DOCK_Y, _DOCK_HD = 10.0, 3.0, 0.0


# ---------------------------------------------------------------------------
# 1. closed-loop convergence from a RANGE of bad arrivals (the core fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pose,heading,desc",
    [
        ((9.74, 3.30), math.radians(86), "R39 t3: heading 86°, x back from table, off-Y"),
        ((10.75, 3.67), -2.95, "R39 t1: 169° off, overshot +y"),
        ((9.4, 2.7), 1.2, "far back + off-center"),
        ((10.9, 2.6), 2.0, "overshot +x, low y"),
        ((10.05, 3.18), 0.5, "close but +29° + off-Y (R39 t2 residual)"),
    ],
)
def test_dock_converges_from_bad_arrivals(pose, heading, desc):
    """From a range of bad FAR arrivals the CLOSED loop converges onto the FIXED proven
    head-on pose: final pose within ALL gate tolerances and dock_converged True."""
    base = _KinematicBase(pose=pose, heading=heading)
    result = terminal_dock(base, (_DOCK_X, _DOCK_Y), _DOCK_HD)

    assert result.ran is True, desc
    assert dock_converged(result) is True, (
        f"{desc}: dock did not converge; errs hd={math.degrees(result.heading_err):.0f}° "
        f"y={result.lateral_err:.3f}m x={result.x_err:.3f}m")
    # Belt-and-braces: the final pose really is head-on at the centerline.
    px, py, _ = base.get_position()
    assert result.heading_err <= _GATE_HEADING_TOL_RAD
    assert abs(py - _DOCK_Y) <= _GATE_LATERAL_TOL_M
    assert abs(px - _DOCK_X) <= _GATE_X_BAND_M


def test_dock_converges_with_drift_injected():
    """With per-step dead-reckoning drift injected, the CLOSED loop still converges —
    proving it re-measures + corrects rather than running open-loop (the ~1/3 failure)."""
    base = _KinematicBase(pose=(9.74, 3.30), heading=math.radians(86),
                          drift=-0.02, drift_y=0.015)
    result = terminal_dock(base, (_DOCK_X, _DOCK_Y), _DOCK_HD)
    assert dock_converged(result) is True


# ---------------------------------------------------------------------------
# 2. the GATE fires on a NON-convergeable arrival — never claims "ready" from a
#    bad pose (the honesty property: a t3-type bad dock → an honest RAN)
# ---------------------------------------------------------------------------

def test_gate_fires_when_dock_cannot_converge():
    """A pathological base whose per-step drift overwhelms every correction can never
    reach the pose; the dock must exhaust its budget and report converged=False so the
    gate aborts the grasp (NEVER 'ready' from a bad pose)."""
    base = _KinematicBase(pose=(9.74, 3.30), heading=math.radians(86), drift=-0.6)
    result = terminal_dock(base, (_DOCK_X, _DOCK_Y), _DOCK_HD)

    assert result.ran is True
    assert result.converged is False, "dock claimed convergence it did not achieve"
    assert dock_converged(result) is False
    assert result.iterations >= 1  # it actually tried


def test_gate_rejects_a_pose_outside_each_tolerance():
    """dock_converged is the AND of all three tolerances — failing ANY one rejects."""
    # within all → pass
    ok = DockResult(ran=True, converged=True, final_pose=(10.0, 3.0, 0.0),
                    heading_err=0.05, lateral_err=0.02, x_err=0.05)
    assert dock_converged(ok) is True
    # heading out of band
    assert dock_converged(DockResult(ran=True, converged=False, heading_err=0.5,
                                     lateral_err=0.02, x_err=0.05)) is False
    # lateral out of band
    assert dock_converged(DockResult(ran=True, converged=False, heading_err=0.05,
                                     lateral_err=0.20, x_err=0.05)) is False
    # x out of band (R39 t3: 0.26 m back from the table)
    assert dock_converged(DockResult(ran=True, converged=False, heading_err=0.05,
                                     lateral_err=0.02, x_err=0.40)) is False
    # did not run → not a converged dock
    assert dock_converged(DockResult(ran=False, converged=False)) is False


# ---------------------------------------------------------------------------
# 3. benign near no-op when already at the dock pose (scripted-from-spawn)
# ---------------------------------------------------------------------------

def test_dock_is_benign_when_already_at_pose():
    """When the dog is already at the dock pose (the scripted-from-spawn path), the dock
    issues NO walk at all and reports converged — so the proven grasp does not regress."""
    base = _KinematicBase(pose=(10.0, 3.0), heading=0.0)
    result = terminal_dock(base, (10.0, 3.0), 0.0)

    assert base.walks == [], f"dock issued motion from the already-docked pose: {base.walks}"
    assert result.ran is True
    assert result.converged is True
    assert dock_converged(result) is True
    px, py, _ = base.get_position()
    assert math.hypot(px - 10.0, py - 3.0) < 1e-9


def test_dock_benign_within_deadband():
    """A dog just inside both deadbands of the dock pose triggers no walk and converges."""
    base = _KinematicBase(pose=(10.05, 3.03), heading=0.05)
    result = terminal_dock(base, (10.0, 3.0), 0.0)
    assert base.walks == [], f"dock moved a within-deadband dog: {base.walks}"
    assert dock_converged(result) is True


# ---------------------------------------------------------------------------
# 4. graceful skip when the base lacks the surface
# ---------------------------------------------------------------------------

def test_dock_skips_base_without_heading():
    """A base lacking get_heading must not crash — it returns ran=False so the caller
    proceeds unchanged (scripted-from-spawn path, no +X assumption, no regression)."""
    class _NoHeading:
        def __init__(self):
            self.walks = []

        def get_position(self):
            return (10.0, 3.0, 0.35)

        def walk(self, vx=0.0, vy=0.0, vyaw=0.0, duration=1.0):
            self.walks.append((vx, vy, vyaw, duration))
            return True

    base = _NoHeading()
    result = terminal_dock(base, (10.0, 3.0), 0.0)
    assert result.ran is False
    assert base.walks == [], "dock issued walks on a base it should have skipped"


def test_dock_skips_none_base():
    """A None base is skipped gracefully (ran=False)."""
    result = terminal_dock(None, (10.0, 3.0), 0.0)
    assert result.ran is False


def test_dock_lateral_recenter_fires_when_off_centerline():
    """The closed-loop lateral re-center closes a y offset — final y within deadband."""
    base = _KinematicBase(pose=(9.88, 3.18), heading=0.02)
    terminal_dock(base, (_DOCK_X, _DOCK_Y), _DOCK_HD)
    _, py, _ = base.get_position()
    assert abs(py - _DOCK_Y) <= _DOCK_LATERAL_DEADBAND_M + 0.03, (
        f"lateral re-center did not close y offset: py={py:.3f}")


# ---------------------------------------------------------------------------
# 5. the grasp wires the dock + GATE before the perceive
# ---------------------------------------------------------------------------

def test_grasp_resolves_dock_pose_param():
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

    assert PerceptionGraspSkill._resolve_dock_pose(
        {"dock_pose": [10.0, 3.0, 0.0]}) == (10.0, 3.0, 0.0)
    assert PerceptionGraspSkill._resolve_dock_pose(
        {"dock_pose": [10.0, 3.0]}) == (10.0, 3.0, 0.0)
    assert PerceptionGraspSkill._resolve_dock_pose({}) is None
    assert PerceptionGraspSkill._resolve_dock_pose({"dock_pose": "bad"}) is None


def test_grasp_execute_docks_and_gates_before_perceiving():
    """The dock + GATE must run BEFORE the perceive: dock first, gate the result, abort
    if not converged, and only THEN _perceive_with_scan."""
    import inspect

    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

    src = inspect.getsource(PerceptionGraspSkill.execute)
    assert "terminal_dock(" in src, "the dock is not called from execute()"
    assert "dock_converged(" in src, "the pose-verification gate is not wired into execute()"
    assert '"dock_not_converged"' in src or "'dock_not_converged'" in src, (
        "the gate does not abort with a dock_not_converged diagnosis")
    dock_at = src.index("terminal_dock(")
    gate_at = src.index("dock_converged(")
    perceive_at = src.index("_perceive_with_scan")
    assert dock_at < gate_at < perceive_at, (
        "order must be: terminal_dock → dock_converged gate → _perceive_with_scan")


class _PerceiveSpy:
    """Records whether any frame-acquiring perceive method was called."""

    def __init__(self):
        self.perceived = False

    # The _REQUIRED_PERCEPTION surface (so the skill does not bail no_camera first).
    def detect(self, query):
        self.perceived = True
        return []

    def segment(self, rgb, box):
        self.perceived = True
        return None

    def capture(self):
        from types import SimpleNamespace
        self.perceived = True
        return SimpleNamespace(rgb=None, depth=None, K=None, cam_pos=None, cam_mat=None)

    def detect_in(self, rgb, query):
        return self.detect(query)

    def get_color_frame(self):
        self.perceived = True
        return None

    def get_depth_frame(self):
        self.perceived = True
        return None

    def get_intrinsics(self):
        return None

    def get_camera_pose(self):
        return (None, None)

    def front_object_mask(self, rgb=None, depth=None, *, color=None):
        self.perceived = True
        return None


class _FakeArm:
    name = "FakeArm"

    def ik_top_down(self, xyz):
        return None

    def move_joints(self, q, duration=1.0):
        return True

    def get_joint_positions(self):
        return [0.0] * 6


class _FakeGripper:
    def is_holding(self):
        return False


def test_grasp_aborts_RAN_when_dock_does_not_converge():
    """The pose-verification GATE: a dock that RAN but did NOT converge must abort the
    grasp with diagnosis 'dock_not_converged' and WITHOUT perceiving — converting a bad
    dock into an honest RAN, never a false/garbage grasp (the R39 t3 class)."""
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

    # A base that can NEVER converge (drift overwhelms each correction).
    base = _KinematicBase(pose=(9.74, 3.30), heading=math.radians(86), drift=-0.6)
    spy = _PerceiveSpy()
    ctx = SkillContext(arm=_FakeArm(), gripper=_FakeGripper(),
                       base=base, perception=spy)

    res = PerceptionGraspSkill().execute(
        {"query": "绿色的瓶子", "dock_pose": [_DOCK_X, _DOCK_Y, _DOCK_HD]}, ctx)

    assert res.success is False
    assert res.result_data.get("diagnosis") == "dock_not_converged", (
        f"expected dock_not_converged, got {res.result_data.get('diagnosis')}")
    assert spy.perceived is False, (
        "the grasp PERCEIVED from a non-converged dock pose — the gate did not abort "
        "before perceiving (this is the exact bad-pose-perceive the gate must prevent)")


def test_grasp_proceeds_to_perceive_when_dock_converges():
    """A converged dock must NOT abort — execution proceeds to the perceive (which then
    fails on the stub backend, but with a PERCEPTION diagnosis, not dock_not_converged)."""
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

    base = _KinematicBase(pose=(9.74, 3.30), heading=math.radians(86))  # converges
    spy = _PerceiveSpy()
    ctx = SkillContext(arm=_FakeArm(), gripper=_FakeGripper(),
                       base=base, perception=spy)

    res = PerceptionGraspSkill().execute(
        {"query": "绿色的瓶子", "dock_pose": [_DOCK_X, _DOCK_Y, _DOCK_HD]}, ctx)

    assert res.result_data.get("diagnosis") != "dock_not_converged", (
        "a converged dock must NOT abort dock_not_converged")
    assert spy.perceived is True, "a converged dock should proceed to perceive"
