# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R2 (track C) — Go2 in the furnished VLM room (double embodiment).

The SAME world-agnostic furnished room (g1_room.build_furnished_room_model) and
the SAME recognition→approach skill (VlmSeekSkill) the g1 furnished room uses,
now on the Go2 quadruped — proving the kernel is embodiment-agnostic (rule
#2/#7). The model-build test needs MuJoCo + the go2 assets (always vendored, so
not gated like g1). The seek test injects a fake VLM + base (no sim, no network).
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.hardware.sim import g1_room

_GO2_FLAT = (Path(__file__).resolve().parents[2] / "vector_os_nano" / "hardware"
             / "sim" / "mjcf" / "go2" / "scene_flat.xml")


class TestSharedFurnishedBuilder:
    """One builder, two embodiments — the world-agnostic invariant in code."""

    @pytest.fixture(scope="class")
    def go2_room(self):
        import mujoco
        m = g1_room.build_furnished_room_model(_GO2_FLAT)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d

    def test_furniture_targets_and_d435_present_no_head_cam(self, go2_room):
        import mujoco
        m, _ = go2_room
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(m.nbody)}
        assert {"target_chair", "target_sofa", "target_plant"} <= names
        assert {"wall_back", "obstacle_center"} <= names
        # Go2 ships its own first-person camera; no g1 pelvis head-cam is added.
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "d435_rgb") >= 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "g1_head_cam") < 0

    def test_furniture_centred_and_on_floor(self, go2_room):
        import mujoco
        m, d = go2_room
        for f in g1_room.FURNITURE:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f.name}_geom")
            mid = int(m.geom_dataid[gid])
            adr, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            R = np.array(d.geom_xmat[gid]).reshape(3, 3)
            v = np.array(m.mesh_vert[adr:adr + n]).reshape(-1, 3) @ R.T \
                + np.array(d.geom_xpos[gid])
            cx = (v[:, 0].min() + v[:, 0].max()) / 2.0
            cy = (v[:, 1].min() + v[:, 1].max()) / 2.0
            assert abs(cx - f.cx) < 0.05 and abs(cy - f.cy) < 0.05
            assert abs(float(v[:, 2].min())) < 0.02     # rests on the floor


class TestScenario:
    def test_go2_room_vlm_registered(self):
        from vector_os_nano.playground.catalog import get_scenario
        sc = get_scenario("go2_room_vlm")
        assert sc.embodiment == "go2"
        assert "target_chair" in sc.object_names


class _FakeDetector:
    def __init__(self, det):
        self._det = det
        self.queries = []

    def detect_targets(self, frame, query, min_area_frac=0.0025):  # noqa: ARG002
        self.queries.append(query)
        return list(self._det)


class _FakeGo2:
    """Blocking-walk go2 stand-in: advances on forward, blocked at x≈3.0."""

    seek_step_duration = 1.2

    def __init__(self):
        self._x = 0.0
        self.durations = []

    def get_camera_frame(self, width=640, height=480):
        return "frame"

    def walk(self, vx, vy, vyaw, duration=2.0):
        self.durations.append(duration)
        if vx > 0 and self._x < 3.0:
            self._x = min(3.0, self._x + 0.5)
        return True

    def stop(self):
        pass

    def get_position(self):
        return [self._x, 0.0, 0.35]


class _HeadingGo2:
    """Go2 stand-in WITH closed-loop heading control: simulates gait yaw-drift so
    the test exercises the P-controller. Target is dead-ahead at world heading 0;
    the camera bearing x_norm is derived from the (drifting) heading. Forward only
    advances x when roughly aligned; blocked at x≈3.0 → progress-stall arrival."""

    seek_heading_hold = True
    seek_step_duration = 1.2

    def __init__(self, start_heading=0.6, drift=0.10):
        self._h = float(start_heading)
        self._drift = float(drift)
        self._x = 0.0
        self.vyaws = []              # every commanded vyaw (turn + forward trim)
        self.heading_calls = 0

    def get_heading(self):
        self.heading_calls += 1
        return self._h

    def get_camera_frame(self, width=640, height=480):
        return "frame"

    def detect_targets_bearing(self):
        # x_norm = h / half_fov (face left of a dead-ahead target → it appears right)
        import math as _m
        return max(-1.0, min(1.0, self._h / 0.80))

    def walk(self, vx, vy, vyaw, duration=2.0):
        self.vyaws.append(float(vyaw))
        # apply commanded yaw + persistent gait drift
        self._h += vyaw * duration + self._drift
        # forward advances while roughly aligned (walk-while-correcting band)
        if vx > 0 and abs(self._h) < 0.4 and self._x < 3.0:
            self._x = min(3.0, self._x + vx * duration)
        return True

    def stop(self):
        pass

    def get_position(self):
        return [self._x, 0.0, 0.30]


class TestVlmSeekOnGo2:
    def test_closed_loop_heading_converges_under_drift(self):
        from vector_os_nano.skills.vlm_seek import VlmSeekSkill
        from types import SimpleNamespace as NS

        base = _HeadingGo2(start_heading=0.5, drift=0.15)

        class _Det:
            def detect_targets(self, frame, query, min_area_frac=0.0025):  # noqa: ARG002
                return [{"label": "chair", "x_norm": base.detect_targets_bearing(),
                         "area_frac": 0.03}]

        res = VlmSeekSkill(detector=_Det()).execute(
            {"label": "chair", "max_iters": 60}, NS(base=base, world_model=None, agent=None))
        assert res.success is True and res.result_data["reason"] == "arrived"
        # converged FORWARD (not wandered backward): ended near the blocked target
        assert base.get_position()[0] >= 2.9
        assert base.heading_calls > 0                 # closed loop really ran
        # commands are P-computed (within cap, and NOT the open-loop constant
        # 0.5) and converge toward zero as heading error shrinks
        assert base.vyaws and max(abs(v) for v in base.vyaws) <= 0.8 + 1e-9
        assert abs(base.vyaws[-1]) < abs(base.vyaws[0])   # shrank (P convergence)
        assert not all(abs(abs(v) - 0.5) < 1e-9 for v in base.vyaws)  # not open-loop

    def test_base_without_hint_never_calls_heading(self):
        # path-isolation guard: a base lacking seek_heading_hold uses the open-loop
        # shared path and must NEVER touch get_heading (Case 19 — g1 also has it).
        from vector_os_nano.skills.vlm_seek import VlmSeekSkill
        from types import SimpleNamespace as NS

        class _NoHintBase(_HeadingGo2):
            seek_heading_hold = False

        base = _NoHintBase()
        det = _FakeDetector([{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}])
        VlmSeekSkill(detector=det).execute(
            {"label": "chair", "max_iters": 15}, NS(base=base, world_model=None, agent=None))
        assert base.heading_calls == 0

    def test_same_skill_drives_go2_to_arrival(self):
        from vector_os_nano.skills.vlm_seek import VlmSeekSkill
        det = _FakeDetector([{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}])
        base = _FakeGo2()
        ctx = SimpleNamespace(base=base, world_model=None, agent=None)
        res = VlmSeekSkill(detector=det).execute(
            {"label": "chair", "max_iters": 30}, ctx)
        assert res.success is True
        assert res.result_data["reason"] == "arrived"
        assert res.result_data["transport"] == "vlm_vision"
        # the VLM was really queried, and the go2 step hint (1.2s) was used —
        # NOT the g1 3.0s default (proves per-embodiment motion profile).
        assert det.queries and det.queries[0] == "chair"
        assert base.durations and max(base.durations) <= 1.2
