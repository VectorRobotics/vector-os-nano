# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #11 R5 — shared _nav_controller pure-logic tests.

drive_to_point is the campaign-#6 G1 controller extracted verbatim + parameterized.
These pin the contract + the per-embodiment injection (fall_z, ctrl-token) so a
transcription slip fails here, not as a subtle real-sim drift."""
from __future__ import annotations

import contextlib
import types

import pytest

from vector_os_nano.hardware.sim import _nav_controller as NC
from vector_os_nano.hardware.sim._nav_controller import NavConsts, drive_to_point


def _consts(**over):
    base = dict(fall_z=0.4, tol_floor=0.3, speed=0.5, vyaw_max=0.6, k_yaw=2.0,
                face_tol=0.35, yaw_deadband=0.12, capture_r=0.5, tick_s=0.05,
                settle_s=0.0, timeout_s=5.0, stall_window_s=6.0, stall_min_m=0.1,
                inflation=0.4, waypoint_tol=0.45)
    base.update(over)
    return NavConsts(**base)


def _driver(positions, heading=0.0):
    seq = list(positions)
    calls = {"set_velocity": [], "stop": 0, "ticks": 0}
    def get_position():
        return seq[0] if len(seq) == 1 else seq.pop(0)
    def set_velocity(vx, vy, vyaw):
        calls["set_velocity"].append((vx, vy, vyaw))
    def stop():
        calls["stop"] += 1
    def tick(s):
        calls["ticks"] += 1
    return dict(get_position=get_position, get_heading=lambda: heading,
                set_velocity=set_velocity, stop=stop, tick_fn=tick), calls


def test_nan_guard_never_commands():
    kw, calls = _driver([[0, 0, 0.5]])
    out = drive_to_point(x=float("nan"), y=0.0, tol=0.2, speed=0.5, consts=_consts(), **kw)
    assert out["reached"] is False and out["reason"] == "bad_params_nan"
    assert out["remaining"] == float("inf")
    assert calls["set_velocity"] == [] and calls["ticks"] == 0


def test_already_within_tol_no_gait():
    kw, calls = _driver([[1.0, 1.0, 0.5]])
    out = drive_to_point(x=1.0, y=1.0, tol=0.3, speed=0.5, consts=_consts(), **kw)
    assert out["already_there"] is True and out["reason"] == "already_within_tol"
    assert calls["set_velocity"] == []         # no spurious step


def test_reached_walks_to_goal():
    # far, then lands on goal at z=0.5 (standing > fall_z 0.4)
    kw, calls = _driver([[0, 0, 0.5]] + [[2.0, 0, 0.5]] * 30)
    out = drive_to_point(x=2.0, y=0.0, tol=0.2, speed=0.5, consts=_consts(), **kw)
    assert out["reached"] is True and out["reason"] == "ok"
    assert out["transport"] == "sim_oracle"
    assert calls["stop"] >= 1


def test_fall_z_injection_is_honored():
    # z=0.30 sample. With fall_z=0.20 -> NOT fallen, reaches. With 0.40 -> 'fell'.
    kw_ok, _ = _driver([[0, 0, 0.30]] + [[2.0, 0, 0.30]] * 30)
    out_ok = drive_to_point(x=2.0, y=0.0, tol=0.2, speed=0.5,
                            consts=_consts(fall_z=0.20), **kw_ok)
    assert out_ok["reached"] is True and out_ok["reason"] == "ok"

    kw_fell, _ = _driver([[0, 0, 0.30]] * 5)
    out_fell = drive_to_point(x=2.0, y=0.0, tol=0.2, speed=0.5,
                              consts=_consts(fall_z=0.40), **kw_fell)
    assert out_fell["reached"] is False and out_fell["reason"] == "fell"


def test_stall_detection():
    # never makes progress while walking forward -> stalled_no_progress
    kw, _ = _driver([[0, 0, 0.5]] * 400)
    out = drive_to_point(x=5.0, y=0.0, tol=0.2, speed=0.5,
                         consts=_consts(stall_window_s=0.0), **kw)
    assert out["reason"] == "stalled_no_progress"


def test_ctrl_token_default_not_entered():
    kw, _ = _driver([[1.0, 1.0, 0.5]])   # already-there short-circuits before token
    # use a far target so the drive body (with ctrl_token) runs
    kw, calls = _driver([[0, 0, 0.5]] + [[2.0, 0, 0.5]] * 30)
    events = []
    @contextlib.contextmanager
    def spy():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")
    drive_to_point(x=2.0, y=0.0, tol=0.2, speed=0.5, consts=_consts(),
                   ctrl_token=spy, **kw)
    assert events == ["enter", "exit"]          # entered once, exited once


def test_ctrl_token_default_nullcontext():
    kw, _ = _driver([[0, 0, 0.5]] + [[2.0, 0, 0.5]] * 30)
    # default factory must be a no-op nullcontext (G1 path) — just runs clean
    out = drive_to_point(x=2.0, y=0.0, tol=0.2, speed=0.5, consts=_consts(), **kw)
    assert out["reached"] is True


def test_navconsts_pinned_values():
    """The two embodiments' constant packs — a wrong number fails HERE, not in sim."""
    from vector_os_nano.hardware.sim.mujoco_g1 import _G1_NAV
    from vector_os_nano.hardware.sim.mujoco_go2 import _GO2_NAV
    assert _G1_NAV.fall_z == 0.4 and _G1_NAV.tol_floor == 0.30 and _G1_NAV.inflation == 0.40
    assert _GO2_NAV.fall_z == 0.20 and _GO2_NAV.tol_floor == 0.20 and _GO2_NAV.inflation == 0.34
    assert _G1_NAV.speed == 0.5 and _GO2_NAV.speed == 0.5
    # R16 go2-trot arrival: Go2 breaks 5cm deeper (coast lands in-tol); G1 unchanged.
    assert _GO2_NAV.arrive_margin == 0.05 and _G1_NAV.arrive_margin == 0.0


# --------------------------------------------------------------------------
# arrive_margin — break DEEPER so the go2-trot coast settles in-tol (R16)
# --------------------------------------------------------------------------

def test_arrive_margin_drives_deeper_through_coast():
    """Models the Go2 trot: a gait dip triggers the 'arrived' break, but the
    SETTLED re-sample sits a few cm farther (the R6/R10/R13 straddle). With no
    margin the settled pose lands OUTSIDE eff_tol; arrive_margin forces a DEEPER
    break so the SAME coast lands INSIDE the unchanged eff_tol. tol_floor<eff_tol
    so the margin bites. The verify ring (reached=remaining<=eff_tol) is unchanged."""
    # Both runs approach through the SAME point (rem 0.39). eff_tol=0.4.
    # no margin (arrive_tol=0.4): STOPS at 0.39, coasts +0.05 to rem 0.44 -> False
    kw0, _ = _driver([[0, 0, 0.5], [1.61, 0, 0.5], [1.56, 0, 0.5]])
    out0 = drive_to_point(x=2.0, y=0.0, tol=0.4, speed=0.5,
                          consts=_consts(tol_floor=0.2, arrive_margin=0.0, settle_s=0.0), **kw0)
    assert out0["reason"] == "arrived" and out0["reached"] is False
    assert out0["remaining"] > 0.4              # the documented boundary straddle

    # margin 0.05 (arrive_tol=0.35): keeps going past 0.39 to rem 0.34, coasts
    # +0.05 back to 0.39 -> inside the UNCHANGED eff_tol 0.4 -> reached
    kwm, _ = _driver([[0, 0, 0.5], [1.61, 0, 0.5], [1.66, 0, 0.5], [1.61, 0, 0.5]])
    outm = drive_to_point(x=2.0, y=0.0, tol=0.4, speed=0.5,
                          consts=_consts(tol_floor=0.2, arrive_margin=0.05, settle_s=0.0), **kwm)
    assert outm["reached"] is True and outm["reason"] == "ok"
    assert outm["remaining"] <= 0.4             # eff_tol UNCHANGED — genuinely arrived


def test_arrive_margin_default_zero_no_op():
    """The additive field defaults to 0.0 (rule 6): _consts()/existing constructors
    omit it, so arrive_tol == eff_tol byte-identical (G1 path unchanged)."""
    assert _consts().arrive_margin == 0.0
    kw, _ = _driver([[0, 0, 0.5]] + [[2.0, 0, 0.5]] * 30)
    out = drive_to_point(x=2.0, y=0.0, tol=0.2, speed=0.5, consts=_consts(), **kw)
    assert out["reached"] is True and out["reason"] == "ok"


def test_arrive_margin_clamped_to_tol_floor():
    """A margin larger than eff_tol clamps to tol_floor — never <0, so a fat
    margin can never make arrival impossible or breach the gait COM floor."""
    # eff_tol = max(0.4, 0.3) = 0.4; arrive_tol = max(0.4-1.0, 0.3) = 0.3 (floor)
    kw, _ = _driver([[0, 0, 0.5], [1.65, 0, 0.5], [1.73, 0, 0.5]])
    out = drive_to_point(x=2.0, y=0.0, tol=0.4, speed=0.5,
                         consts=_consts(tol_floor=0.3, arrive_margin=1.0, settle_s=0.0), **kw)
    assert out["reached"] is True and out["reason"] == "ok"   # arrived at the floor


# --------------------------------------------------------------------------
# route_and_drive per-leg terminal-reason early-break (campaign #11 R10 debt)
# --------------------------------------------------------------------------

def _route_kw():
    """Fixed far standing pose; the drive_to_point itself is monkeypatched, so
    pose/heading/cmd callables are inert."""
    return dict(get_position=lambda: [0.0, 0.0, 0.5], get_heading=lambda: 0.0,
                set_velocity=lambda *a: None, stop=lambda: None,
                tick_fn=lambda s: None)


def _stub_two_legs(monkeypatch, first_reason):
    """Plan a 3-point path (= 2 legs) and record each leg drive_to_point drives.
    Leg 1 returns ``first_reason``; any later leg returns 'ok'."""
    from vector_os_nano.hardware.sim import g1_vgraph
    monkeypatch.setattr(g1_vgraph, "plan_path",
                        lambda a, b, obs, infl: ([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], 2.0))
    driven = []

    def fake_drive(*, x, y, **kw):
        driven.append((x, y))
        reason = first_reason if len(driven) == 1 else "ok"
        return {"moved_m": 0.1, "reason": reason, "reached": reason == "ok"}

    monkeypatch.setattr(NC, "drive_to_point", fake_drive)
    return driven


def test_route_and_drive_breaks_on_leg_timeout(monkeypatch):
    driven = _stub_two_legs(monkeypatch, "timeout")
    out = NC.route_and_drive(x=2.0, y=0.0, tol=0.3, speed=0.5, obstacles=[],
                             consts=_consts(), **_route_kw())
    assert len(driven) == 1                     # leg 2 NEVER driven after a timeout
    assert out["reason"] == "timeout"


def test_route_and_drive_continues_past_benign_leg_reason(monkeypatch):
    driven = _stub_two_legs(monkeypatch, "ok")
    NC.route_and_drive(x=2.0, y=0.0, tol=0.3, speed=0.5, obstacles=[],
                       consts=_consts(), **_route_kw())
    assert len(driven) == 2                     # a reached intermediate leg → continue


@pytest.mark.parametrize("terminal", ["fell", "stalled_no_progress"])
def test_route_and_drive_still_breaks_on_fell_and_stall(monkeypatch, terminal):
    driven = _stub_two_legs(monkeypatch, terminal)
    NC.route_and_drive(x=2.0, y=0.0, tol=0.3, speed=0.5, obstacles=[],
                       consts=_consts(), **_route_kw())
    assert len(driven) == 1                     # pre-existing terminals still break


def test_drive_to_point_default_reason_is_timeout():
    # far standing pose, never arrives/falls/stalls; timeout_s=0 → loop exits at once
    kw, _ = _driver([[0, 0, 0.5]])
    out = drive_to_point(x=5.0, y=0.0, tol=0.2, speed=0.5,
                         consts=_consts(timeout_s=0.0, settle_s=0.0), **kw)
    assert out["reason"] == "timeout" and out["reached"] is False
