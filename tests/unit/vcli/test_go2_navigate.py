# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #11 M2 — Go2 navigate_to / geodesic / VLN-symmetry pure-logic tests.

MuJoCoGo2.navigate_to mirrors MuJoCoG1's three-value contract so recognize_navigate
runs on Go2 (lidar VLN). Most tests exercise the control/contract logic via a stub
(no MuJoCo/GL). The capability-probe registration tests drive the REAL
go2_runtime.boot_go2_agent / _build_go2_skill_registry instead of a copy (R15): the
negative branch uses a navigate_to-less fake (no GL); the positive is a real headless
MuJoCoGo2 boot, marked @pytest.mark.integration (deselectable, disconnect in finally)."""
from __future__ import annotations

import math
import os
import types

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")   # headless render (sensors/conftest.py:17)

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
    import contextlib
    st = types.SimpleNamespace(
        _obstacles=[], _skill_ctrl_until=0.0, _skill_ctrl_tid=0,
        _require_connection=lambda: None,
        get_heading=lambda: 0.0,
        set_velocity=lambda *a, **k: None,
        stop=lambda: None,
        _drive_for=lambda s: None,
        _ctrl_token=lambda: contextlib.nullcontext(),
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


# -- capability-probe registration: drive the REAL boot_go2_agent (R15) ------
# (was a test that COPIED go2_runtime's registration block — a copy catches no
#  change to the real probe. Now both branches run the real code.)

def test_boot_go2_agent_omits_nav_skills_without_navigate_to(monkeypatch):
    """NEGATIVE branch of the REAL capability probe: a base WITHOUT navigate_to
    must NOT get the nav skills (only reachable via a navigate_to-less base, since
    a real MuJoCoGo2 always has navigate_to). Drives the actual boot_go2_agent —
    only the heavy ctor is swapped for a fake base."""
    import vector_os_nano.vcli.go2_runtime as gr

    fake_base = types.SimpleNamespace(
        connect=lambda: None, stand=lambda: None, list_targets=lambda: {})
    # boot_go2_agent's import is FUNCTION-LOCAL — patch the SOURCE module so the
    # in-function `from ...mujoco_go2 import MuJoCoGo2` picks up the fake.
    monkeypatch.setattr(
        "vector_os_nano.hardware.sim.mujoco_go2.MuJoCoGo2",
        lambda *a, **k: fake_base)
    agent = gr.boot_go2_agent(world=None, gui=False)
    names = set(agent.skills)
    assert "navigate_to" not in names and "recognize_navigate" not in names
    assert {"walk", "turn", "stop", "vlm_seek"}.issubset(names)  # M1 intact


def test_build_go2_skill_registry_probes_navigate_to():
    """The single-source helper itself: nav skills registered IFF navigate_to is
    callable. Pure (no boot) — pins the probe contract directly."""
    import vector_os_nano.vcli.go2_runtime as gr

    with_nav = gr._build_go2_skill_registry(
        types.SimpleNamespace(navigate_to=lambda *a, **k: {}))
    assert {"navigate_to", "recognize_navigate"}.issubset(set(with_nav.list_skills()))
    without = gr._build_go2_skill_registry(types.SimpleNamespace())
    got = set(without.list_skills())
    assert "navigate_to" not in got and "recognize_navigate" not in got
    assert {"walk", "turn", "stop", "vlm_seek"}.issubset(got)


@pytest.mark.integration
def test_boot_go2_agent_registers_navigate_skills_real_boot():
    """POSITIVE branch on a REAL headless MuJoCoGo2 (real-sim mandate): a booted
    Go2 has navigate_to, so the full skill set is registered. Disconnect in
    finally to release the physics daemon + GL (no leak)."""
    import threading

    import vector_os_nano.vcli.go2_runtime as gr
    agent = gr.boot_go2_agent(world=None, gui=False)
    try:
        names = set(agent.skills)
        assert {"walk", "turn", "stop", "vlm_seek",
                "navigate_to", "recognize_navigate"}.issubset(names)
    finally:
        agent._base.disconnect()
    assert not any(t.name == "mujoco_go2_physics" for t in threading.enumerate())


def test_navigate_skill_loud_fail_on_bool_contract():
    """M3 guard (rule 8): NavigateToPointSkill against a base whose navigate_to
    returns a bare bool (raw Go2ROS2Proxy / FAR track-B) fails LOUDLY with a
    clear diagnosis, never an AttributeError on out.get()."""
    from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill
    base = types.SimpleNamespace(
        navigate_to=lambda x, y, tol=0.2: True,    # bool, not the 3-value dict
        get_position=lambda: [0.0, 0.0, 0.3],
        list_targets=lambda: {})
    ctx = types.SimpleNamespace(base=base, world_model=None, world=None,
                                scene_graph=None, perception=None, config={})
    res = NavigateToPointSkill().execute({"x": 2.0, "y": 0.0}, ctx)
    assert res.success is False
    assert res.result_data.get("diagnosis") == "nav_contract_violation"
