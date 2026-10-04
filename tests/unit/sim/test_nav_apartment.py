# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The planner + follower + coverage tour on the REAL go2 apartment geometry.

The scene is compiled (no physics, no driver) and its obstacle set extracted
exactly as MuJoCoGo2.navigate_to does; a kinematic body replays the follower.

Pins: skills-nav:room-centers-in-furniture (every layout point is free floor in
its own room), the clearance of every planned leg (>= the inflation radius) and of
every executed leg (>= the go2 half-width; sim-locomotion:capture-radius-cuts-
corners), and an 8/8 coverage tour graded by room boxes.
"""
from __future__ import annotations

import json
import math

import pytest

from tests.harness.kin_base import KinBase
from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.sim.nav import vgraph as vg
from vector_os_nano.sim.nav.coverage import CoverageExplorer, room_at, room_layout
from vector_os_nano.sim.nav.follow import follow

_HALF_WIDTH = 0.19  # go2 body half-width (m)
_SPAWN = (10.0, 3.0)
BODY = load_embodiment_config("go2").nav
R = float(BODY["body_radius"])


class Never:
    @staticmethod
    def is_set():
        return False


@pytest.fixture(scope="module")
def obstacles():
    import mujoco

    from vector_os_nano.embodiments.dof_layout import DofLayout
    from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml

    model = mujoco.MjModel.from_xml_path(str(_build_room_scene_xml(with_arm=False)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    polys = vg.obstacles_from_model(model, data, DofLayout(model, "base_link", 12).robot_geom_ids)
    assert len(polys) > 40  # walls + furniture, robot and pickables excluded
    return polys


def _clearance(points, polys) -> float:
    return min(min(vg.distance_to_polygon(p, q) for q in polys) for p in points)


def _sample(path, step=0.05):
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.dist(a, b) / step))
        for k in range(n + 1):
            yield (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)


def test_every_layout_room_point_is_free_floor_in_its_own_room(obstacles):
    layout = room_layout()
    inflated = [vg.inflate_polygon(p, R) for p in obstacles]
    assert len(layout["rooms"]) == 8 and set(layout["rooms"]) == set(layout["boxes"])
    for name, pt in layout["rooms"].items():
        assert not vg.point_in_any(pt, inflated), f"{name} {pt} is inside (inflated) furniture"
        assert room_at(*pt, layout["boxes"]) == name


@pytest.mark.parametrize("room", sorted(room_layout()["rooms"]))
def test_planned_leg_keeps_the_inflation_radius(obstacles, room):
    path, length = vg.plan_path(_SPAWN, room_layout()["rooms"][room], obstacles, R, snap_goal=True)
    assert path is not None and math.isfinite(length)
    assert _clearance(_sample(path), obstacles) >= R - 1e-3


@pytest.mark.parametrize("room", sorted(room_layout()["rooms"]))
def test_executed_leg_keeps_the_body_half_width(obstacles, room):
    base = KinBase(*_SPAWN)
    res = follow(base, room_layout()["rooms"][room], obstacles, body=BODY, cancel=Never())
    assert res, res
    assert _clearance(base.trace[::5], obstacles) >= _HALF_WIDTH


def test_coverage_tour_enters_every_room_by_ground_truth(obstacles, tmp_path):
    class Body(KinBase):
        def navigate_to(self, x, y, cancel=None, **_):
            return follow(self, (x, y), obstacles, body=BODY, cancel=cancel)

    base = Body(*_SPAWN)
    events = []
    looks = []  # explore's discovery runs after every look turn (skills-nav:explore-auto-observe)
    log = tmp_path / "cov.jsonl"
    out = CoverageExplorer(base, cancel=Never(), on_event=lambda e, d: events.append(e),
                           log_path=str(log), observe=lambda: looks.append(base.get_heading())).run()
    assert sorted(out["visited"]) == sorted(room_layout()["rooms"]) and not out["failed"]
    assert len(looks) >= len(out["visited"]) and len(set(round(h, 1) for h in looks)) > 1
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    boxes = room_layout()["boxes"]
    assert all(room_at(*r["gt_pos"], boxes) == r["room"] and r["looked"] for r in rows)
    assert events[0] == "started" and events[-1] == "completed"
    assert _clearance(base.trace[::10], obstacles) >= _HALF_WIDTH


def test_an_extended_arm_or_held_object_grows_the_footprint():
    """(WP08 graft e) the navigation inflation grows by how far a mounted arm, or a
    body welded to it, reaches past the dog; the stowed Piper adds nothing."""
    import mujoco

    from vector_os_nano.embodiments.dof_layout import DofLayout
    from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml

    m = mujoco.MjModel.from_xml_path(str(_build_room_scene_xml(with_arm=True)))
    d = mujoco.MjData(m)
    lay = DofLayout(m, "base_link", 12)
    d.qpos[lay.root_qpos_adr:lay.root_qpos_adr + 3] = [10.0, 3.0, 0.35]
    d.qpos[lay.joint_qpos_start:lay.joint_qpos_start + 12] = lay.build_stance_vector(
        m, load_embodiment_config("go2").stance)
    extra = lay.joint_qpos_start + lay.num_actuated
    mujoco.mj_forward(m, d)
    assert vg.overhang(m, d, lay.root_bid, extra) == 0.0  # stowed
    d.qpos[extra + 1], d.qpos[extra + 2] = 1.57, -1.0     # reach forward and down
    mujoco.mj_forward(m, d)
    reach = vg.overhang(m, d, lay.root_bid, extra)
    assert reach > 0.02
    weld = next(i for i in range(m.neq) if m.eq(i).name == "grasp_pickable_can_red")
    d.eq_active[weld] = 1  # the can counts as part of the arm once welded to it
    assert vg.overhang(m, d, lay.root_bid, extra) > reach
