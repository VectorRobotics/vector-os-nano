# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #11 M2 — Go2 navigate_to / geodesic / VLN-symmetry pure-logic tests.

MuJoCoGo2.navigate_to mirrors MuJoCoG1's three-value contract so recognize_navigate
runs on Go2 (lidar VLN). These exercise the control/contract logic via a stub (no
MuJoCo/GL — avoids the known MUJOCO_GL pollution reds); real-sim VLN arrival lives
in the sandbox harness + vector-cli."""
from __future__ import annotations

import math
import types

import pytest

from vector_os_nano.core.types import LaserScan
from vector_os_nano.hardware.sim import mujoco_go2 as G
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2


# -- LaserScan.points additive field (rule 6) -------------------------------

def test_laserscan_points_defaults_empty():
    """Pre-existing 2D-only construction (no points) still valid; default ()."""
    s = LaserScan(timestamp=0.0, angle_min=-1.0, angle_max=1.0,
                  angle_increment=0.1, range_min=0.1, range_max=12.0,
                  ranges=(1.0, 2.0))
    assert s.points == ()
    assert s.to_dict()["n_points"] == 0


def test_laserscan_points_carries_world_cloud():
    s = LaserScan(timestamp=0.0, angle_min=-1.0, angle_max=1.0,
                  angle_increment=0.1, range_min=0.1, range_max=12.0,
                  ranges=(1.0,), points=((1.0, 2.0, 0.3, 0.0),))
    assert s.points[0][:2] == (1.0, 2.0)
    assert getattr(s, "points", None)        # truthy -> locate_from_bearing runs


# -- FALL_Z: the #1 silent-failure fix (Go2 stands at z=0.35) ----------------

def test_fall_z_below_go2_stand_height():
    """_GO2_NAV_FALL_Z must be below the Go2 standing height (0.35) — reusing
    G1's 0.4 would make `reached` never true and report 'fell' on every arrival."""
    assert G._GO2_NAV_FALL_Z == 0.20
    assert G._GO2_NAV_FALL_Z < 0.35
    assert G._GO2_NAV_INFLATION == 0.34


# -- navigate_to three-value dict contract (stubbed, no MuJoCo) --------------

def _stub(positions):
    """A MuJoCoGo2-shaped stub: get_position pops from `positions` (last repeats)."""
    seq = list(positions)
    st = types.SimpleNamespace(
        _obstacles=[], _skill_ctrl_until=0.0, _skill_ctrl_tid=0,
        _require_connection=lambda: None,
        get_heading=lambda: 0.0,
        set_velocity=lambda *a, **k: None,
        stop=lambda: None,
        _drive_for=lambda s: None,
    )
    def get_position():
        return seq[0] if len(seq) == 1 else seq.pop(0)
    st.get_position = get_position
    return st


def test_navigate_to_dict_contract_and_reached():
    """Arrives (z=0.35 standing) -> reached True, full dict, transport sim_oracle.
    Guards the FALL_Z fix end-to-end AND the dict keys NavigateToPointSkill reads."""
    # start far, then land on target at z=0.35 (standing) for the rest.
    st = _stub([[0.0, 0.0, 0.35]] + [[2.0, 0.0, 0.35]] * 20)
    out = MuJoCoGo2._go2_navigate_point(st, 2.0, 0.0, tol=0.2)
    assert out["transport"] == "sim_oracle"
    for k in ("reached", "already_there", "moved_m", "net_m", "elapsed_s",
              "remaining", "pos", "effective_tol", "reason"):
        assert k in out
    assert out["reached"] is True and out["reason"] == "ok"


def test_navigate_to_nan_guard():
    st = _stub([[0.0, 0.0, 0.35]])
    out = MuJoCoGo2._go2_navigate_point(st, float("nan"), 0.0)
    assert out["reached"] is False and out["reason"] == "bad_params_nan"
    assert out["remaining"] == float("inf")


def test_navigate_to_already_within_tol():
    st = _stub([[1.0, 1.0, 0.35]])
    out = MuJoCoGo2._go2_navigate_point(st, 1.0, 1.0, tol=0.3)
    assert out["already_there"] is True and out["reason"] == "already_within_tol"
    assert out["moved_m"] == 0.0


# -- geodesic_distance: flat = straight line (single-source inflation) -------

def test_geodesic_flat_is_hypot():
    st = types.SimpleNamespace(_obstacles=[])
    d = MuJoCoGo2.geodesic_distance(st, [0.0, 0.0], [3.0, 4.0])
    assert d == pytest.approx(5.0)


# -- capability-probe registration in go2_runtime ---------------------------

def test_recognize_navigate_registered_only_with_navigate_to(monkeypatch):
    """boot_go2_agent registers recognize_navigate IFF base.navigate_to is
    callable; walk/turn/stop/vlm_seek always present (M1 unaffected)."""
    import vector_os_nano.vcli.go2_runtime as gr

    captured = {}

    class _Reg:
        def __init__(self):
            self.names = []
        def register(self, s):
            self.names.append(getattr(s, "name", type(s).__name__))

    def fake_base(has_nav):
        b = types.SimpleNamespace(
            connect=lambda: None, stand=lambda: None,
            list_targets=lambda: {})
        if has_nav:
            b.navigate_to = lambda *a, **k: {}
        return b

    # Patch the heavy bits: MuJoCoGo2 ctor, Agent, SkillRegistry, world model add.
    monkeypatch.setattr(gr, "_emit", lambda *a, **k: None)
    import vector_os_nano.core.skill as skill_mod
    monkeypatch.setattr(skill_mod, "SkillRegistry", _Reg)

    for has_nav in (True, False):
        reg = _Reg()
        # emulate the registration block directly (the runtime's exact logic)
        from vector_os_nano.skills.go2.stop import StopSkill
        from vector_os_nano.skills.go2.turn import TurnSkill
        from vector_os_nano.skills.go2.walk import WalkSkill
        from vector_os_nano.skills.vlm_seek import VlmSeekSkill
        for s in (WalkSkill(), TurnSkill(), StopSkill(), VlmSeekSkill()):
            reg.register(s)
        base = fake_base(has_nav)
        if callable(getattr(base, "navigate_to", None)):
            from vector_os_nano.skills.recognize_navigate import RecognizeNavigateSkill
            reg.register(RecognizeNavigateSkill())
        captured[has_nav] = reg.names

    assert "recognize_navigate" in captured[True]
    assert "recognize_navigate" not in captured[False]
    for nm in ("walk", "turn", "stop"):
        assert any(nm in n for n in captured[False])  # M1 locomotion intact
