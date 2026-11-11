# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Navigation skills on fakes (no MuJoCo): navigate / explore / walk / turn / stop /
patrol / where_am_i / lie_down, each pinned against its WP08 ledger defect."""
from __future__ import annotations

import math
import threading
from unittest.mock import MagicMock, patch

import pytest

from tests.harness.kin_base import KinBase
from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult
from vector_os_nano.sim.nav.coverage import _LAYOUT_YAML, room_layout
from vector_os_nano.sim.nav.follow import NavResult, arrival_radius


@pytest.fixture(autouse=True)
def _own_run_dir(tmp_path, monkeypatch):
    """The coverage tour logs to the run dir: never the machine-global /tmp."""
    monkeypatch.setenv("VECTOR_RUN_DIR", str(tmp_path))


def _sg() -> SceneGraph:
    sg = SceneGraph()
    sg.load_layout(str(_LAYOUT_YAML))
    return sg


def _nav(reached: bool, reason: str, pos=(0.0, 0.0)) -> NavResult:
    return NavResult(reached=reached, reason=reason, target=list(pos), snapped=False,
                     goal=list(pos), pos=[*pos, 0.3])


class Teleport(KinBase):
    """A base whose planner lands exactly on every target (the coverage plumbing)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.goals = []

    def navigate_to(self, x, y, cancel=None, **_):
        self.goals.append((x, y))
        self.x, self.y = x, y
        return _nav(True, "arrived", (x, y))


# ---------------------------------------------------------------------------
# NavigateSkill
# ---------------------------------------------------------------------------

def test_failed_planner_navigation_never_moves_a_room_centre_and_reports_the_reason():
    # skills-nav:navigate-authors-own-oracle
    from vector_os_nano.skills.navigate import NavigateSkill

    sg = _sg()
    before = (sg.get_room("kitchen").center_x, sg.get_room("kitchen").center_y)
    base = MagicMock()
    base.get_position.return_value = [12.0, 3.0, 0.3]
    base.navigate_to.return_value = _nav(False, "stalled", (12.0, 3.0))
    res = NavigateSkill().execute({"room": "厨房"}, SkillContext(base=base, services={"spatial_memory": sg}))
    assert not res.success and res.result_data["reason"] == "stalled"
    assert "stalled" in res.error_message
    base.navigate_to.assert_called_once()  # one global planner: no door-chain retry
    assert (sg.get_room("kitchen").center_x, sg.get_room("kitchen").center_y) == before


def test_arrival_records_the_visit():
    from vector_os_nano.skills.navigate import NavigateSkill

    sg = _sg()
    kx, ky = room_layout()["rooms"]["kitchen"]
    n0 = sg.get_room("kitchen").visit_count
    base = MagicMock()
    base.get_position.return_value = [kx + 0.1, ky, 0.3]
    base.navigate_to.return_value = _nav(True, "arrived", (kx, ky))
    res = NavigateSkill().execute({"room": "kitchen"}, SkillContext(base=base, services={"spatial_memory": sg}))
    assert res.success and res.result_data["reason"] == "arrived"
    assert sg.get_room("kitchen").visit_count == n0 + 1


def test_coordinate_navigate_cancels_a_running_exploration_and_stops_via_stop():
    # skills-nav:missed2 + skills-nav:stop-identity-check
    from vector_os_nano.skills.navigate import NavigateSkill

    base = MagicMock()
    base.get_position.return_value = [1.0, 1.0, 0.3]
    base.navigate_to.return_value = _nav(True, "arrived", (1.0, 1.0))
    with patch("vector_os_nano.skills.go2.explore.is_exploring", return_value=True), \
            patch("vector_os_nano.skills.go2.explore.cancel_exploration") as cancel:
        res = NavigateSkill().execute({"x": 1.0, "y": 1.0}, SkillContext(base=base))
    assert res.success and res.result_data["reason"] == "arrived"
    cancel.assert_called_once()
    base.stop.assert_called_once_with()

    bare = MagicMock(spec=["get_position", "navigate_to", "set_velocity"])
    bare.get_position.return_value = [1.0, 1.0, 0.3]
    bare.navigate_to.return_value = _nav(True, "arrived", (1.0, 1.0))
    NavigateSkill().execute({"x": 1.0, "y": 1.0}, SkillContext(base=bare))
    bare.set_velocity.assert_called_once_with(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# ExploreSkill
# ---------------------------------------------------------------------------

def test_coverage_explore_blocks_and_reports_the_rooms_entered():
    from vector_os_nano.skills.go2 import explore

    base = Teleport(10.0, 3.0)
    sg = _sg()
    res = explore.ExploreSkill().execute({}, SkillContext(base=base, services={"spatial_memory": sg}))
    assert res.success and res.result_data["backend"] == "coverage"
    assert sorted(res.result_data["rooms_visited"]) == sorted(room_layout()["rooms"])
    assert res.result_data["coverage"] == "8/8 rooms"
    assert not explore.is_exploring() and explore.get_explored_rooms() == sorted(room_layout()["rooms"])


def test_coverage_explore_reached_no_room_is_a_failure():
    from vector_os_nano.skills.go2 import explore

    class Stuck(Teleport):
        def navigate_to(self, x, y, cancel=None, **_):
            return _nav(False, "unreachable", (self.x, self.y))

    res = explore.ExploreSkill().execute({}, SkillContext(base=Stuck(-5.0, -5.0)))
    assert not res.success and res.diagnosis_code == "exploration_failed"


def test_a_stop_cancels_the_coverage_tour():
    from vector_os_nano.skills.go2 import explore
    from vector_os_nano.vcli.cognitive import abort

    class StopAfterTwo(Teleport):
        def navigate_to(self, x, y, cancel=None, **_):
            if len(self.goals) == 2:
                abort.request_abort()
            return super().navigate_to(x, y, cancel=cancel)

    base = StopAfterTwo(10.0, 3.0)
    try:
        res = explore.ExploreSkill().execute({}, SkillContext(base=base))
    finally:
        abort.clear_abort()
    assert not res.success and res.diagnosis_code == "aborted"
    assert len(base.goals) == 3 and not explore.is_exploring()


def test_tare_explore_is_running_before_execute_returns():
    # legacy-cognitive:B14-explore-wait-race: the running flag was set inside the
    # thread, so a waiter polling is_exploring() right after execute saw False.
    from vector_os_nano.skills.go2 import explore

    release = threading.Event()
    with patch.object(explore, "_tare_running", return_value=True), \
            patch.object(explore, "_exploration_loop", side_effect=lambda base: release.wait(5)):
        res = explore.ExploreSkill().execute({}, SkillContext(base=Teleport()))
        try:
            assert res.success and res.result_data["backend"] == "tare"
            assert explore.is_exploring()
        finally:
            release.set()
            explore._explore_thread.join(5)
            explore._explore_running = False


# ---------------------------------------------------------------------------
# Walk / Turn / Stop / LieDown
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"direction": "sideways"}, {"distance": -2.0},
                                 {"distance": 0}, {"speed": -0.3}, {"distance": "far"}])
def test_walk_rejects_unknown_direction_and_non_positive_distance(bad):
    # skills-nav:turn-direction-silent + skills-nav:walk-runaway-on-exception
    from vector_os_nano.skills.go2.walk import WalkSkill

    base = MagicMock()
    res = WalkSkill().execute(bad, SkillContext(base=base))
    assert not res.success and res.diagnosis_code in ("bad_direction", "bad_distance")
    base.walk.assert_not_called()


@pytest.mark.parametrize("alias,sign", [("后退", -1.0), ("back", -1.0), ("向前", 1.0), ("Forward ", 1.0)])
def test_walk_normalises_direction_aliases(alias, sign):
    from vector_os_nano.skills.go2.walk import WalkSkill

    base = MagicMock()
    base.walk.return_value = True
    res = WalkSkill().execute({"direction": alias, "distance": 1.0}, SkillContext(base=base))
    assert res.success and math.copysign(1.0, base.walk.call_args[0][0]) == sign


@pytest.mark.parametrize("alias,want", [
    ("左转", "left"), ("逆时针", "left"), ("ccw", "left"), ("LEFT", "left"),
    ("右转", "right"), ("顺时针", "right"), ("cw", "right"),
])
def test_turn_normalises_aliases_and_closes_the_loop(alias, want):
    from vector_os_nano.skills.go2.turn import TurnSkill

    base = KinBase()
    res = TurnSkill().execute({"direction": alias, "angle": 90}, SkillContext(base=base))
    assert res.success and res.result_data["direction"] == want
    assert abs(base.yaw - (math.pi / 2 if want == "left" else -math.pi / 2)) <= math.radians(5)


def test_turn_rejects_an_unknown_direction():
    # skills-nav:turn-direction-silent: anything but 'left' used to turn RIGHT.
    from vector_os_nano.skills.go2.turn import TurnSkill

    base = KinBase()
    res = TurnSkill().execute({"direction": "forward", "angle": 90}, SkillContext(base=base))
    assert not res.success and res.diagnosis_code == "bad_direction" and base.yaw == 0.0


def test_turn_fails_when_the_heading_never_arrives():
    from vector_os_nano.skills.go2.turn import TurnSkill

    class Stuck(KinBase):
        def walk(self, vx, vy, vyaw, duration):
            return True  # the body never rotates

    res = TurnSkill().execute({"direction": "left", "angle": 90}, SkillContext(base=Stuck()))
    assert not res.success and res.diagnosis_code == "turn_failed"


def test_lie_down_on_a_base_without_it_fails_loud():
    # skills-nav:missed5: the proxy base has no lie_down -> AttributeError
    from vector_os_nano.skills.go2.stance import LieDownSkill

    res = LieDownSkill().execute({}, SkillContext(base=MagicMock(spec=["get_position"])))
    assert not res.success and res.diagnosis_code == "lie_down_failed"


@pytest.mark.parametrize("cls, posture", [("StandSkill", "stand"), ("SitSkill", "sit")])
def test_stand_and_sit_on_a_base_without_them_fail_loud(cls, posture):
    """skills-nav:stance-no-posture-check: stand/sit called the method bare (AttributeError)."""
    from vector_os_nano.skills.go2 import stance

    res = getattr(stance, cls)().execute({}, SkillContext(base=MagicMock(spec=["get_position"])))
    assert not res.success and res.diagnosis_code == f"{posture}_failed"


def test_a_posture_reports_what_it_measured_and_never_claims_the_posture():
    """skills-nav:stance-false-green: success was the base's bool, with no measurement."""
    from vector_os_nano.skills.go2.stance import SitSkill

    heights = iter([0.30, 0.18])
    base = MagicMock()
    base.get_position.side_effect = lambda: [0.0, 0.0, next(heights)]
    base.sit.return_value = True
    res = SitSkill().execute({}, SkillContext(base=base))
    assert res.success and res.result_data["posture_verified"] is False
    assert (res.result_data["height_before_m"], res.result_data["height_after_m"]) == (0.30, 0.18)


def test_a_blocked_walk_fails_with_the_distance_it_moved():
    """skills-nav:walk-success-unverified: a blocked robot reported success."""
    from vector_os_nano.skills.go2.walk import WalkSkill

    wall = KinBase(blocked=lambda x, y, t: x > 0.1)
    res = WalkSkill().execute({"direction": "forward", "distance": 1.0, "speed": 0.5}, SkillContext(base=wall))
    assert not res.success and res.diagnosis_code == "walk_blocked"
    assert res.result_data["moved_m"] == pytest.approx(0.1, abs=0.02)

    free = KinBase()
    res = WalkSkill().execute({"direction": "forward", "distance": 1.0, "speed": 0.5}, SkillContext(base=free))
    assert res.success and res.result_data["moved_m"] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Patrol / where_am_i / compass / at_position
# ---------------------------------------------------------------------------

def _patrol_ctx(nav_results, look_ok=False):
    sg = _sg()
    navigate = MagicMock()
    navigate.execute.side_effect = nav_results
    look = MagicMock()
    look.execute.return_value = (
        SkillResult(success=True, result_data={"summary": "a kitchen"}) if look_ok
        else SkillResult(success=False, error_message="no VLM"))
    registry = MagicMock()
    registry.get.side_effect = lambda name: {"navigate": navigate, "look": look}.get(name)
    base = MagicMock()
    base.get_position.return_value = [10.0, 3.0, 0.3]
    ctx = SkillContext(base=base, services={"spatial_memory": sg, "skill_registry": registry})
    return ctx, navigate, sg


def test_patrol_resolves_aliases_and_never_stores_a_failed_look():
    # skills-nav:patrol-defects
    from vector_os_nano.skills.go2.patrol import PatrolSkill

    ctx, navigate, sg = _patrol_ctx([SkillResult(success=True)])
    desc_before = sg.get_room("kitchen").representative_description
    res = PatrolSkill().execute({"rooms": ["厨房", "nowhere"]}, ctx)
    assert navigate.execute.call_args[0][0] == {"room": "kitchen"}
    assert res.success and res.result_data["rooms_visited"] == ["kitchen"]
    assert res.result_data["rooms_unknown"] == ["nowhere"]
    assert sg.get_room("kitchen").representative_description == desc_before


def test_patrol_stops_between_rooms():
    from vector_os_nano.skills.go2.patrol import PatrolSkill
    from vector_os_nano.vcli.cognitive import abort

    def nav_then_stop(params, context):
        abort.request_abort()
        return SkillResult(success=True)

    ctx, navigate, _sg_ = _patrol_ctx(None)
    navigate.execute.side_effect = nav_then_stop
    try:
        res = PatrolSkill().execute({"rooms": ["kitchen", "study", "bathroom"]}, ctx)
    finally:
        abort.clear_abort()
    assert navigate.execute.call_count == 1
    assert not res.success and res.diagnosis_code == "aborted"


@pytest.mark.parametrize("xy,room", [((17.0, 2.5), "kitchen"), ((3.0, 7.5), "dining_room"),
                                     ((10.0, 3.0), "hallway"), ((3.0, 2.5), "living_room")])
def test_where_am_i_assigns_the_room_by_geometry(xy, room):
    from vector_os_nano.skills.go2.where_am_i import WhereAmISkill

    base = MagicMock()
    base.get_position.return_value = [*xy, 0.3]
    base.get_heading.return_value = 0.0
    res = WhereAmISkill().execute({}, SkillContext(base=base))
    assert res.result_data["room"] == room
    assert res.result_data["heading"] == "east"


@pytest.mark.parametrize("yaw_deg,label", [(0, "east"), (90, "north"), (180, "west"), (-90, "south")])
def test_robot_context_compass_matches_where_am_i(yaw_deg, label):
    # skills-nav:compass-mismatch: robot_context called yaw 0 "N"
    from vector_os_nano.vcli.robot_context import _heading_to_compass

    assert _heading_to_compass(yaw_deg) == label


def test_at_position_default_tol_is_the_arrival_radius():
    # producers:arrival-radius-vs-tol-drift: navigate 'arrived' at 0.8 m, verify at 0.5 m
    from vector_os_nano.vcli.worlds.sim_oracles import _AT_POSITION_TOL_M

    assert _AT_POSITION_TOL_M == arrival_radius() <= 0.5


def test_nav_stack_detection_ignores_processes_that_are_not_ours(monkeypatch):
    # periphery:missed2: `pgrep -f localPlanner` took another project's nav node on the
    # shared stack for ours. Detection now walks only this process's descendants.
    import subprocess
    import sys

    from vector_os_nano.skills.go2 import explore

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "localPlanner"])
    try:
        assert explore.is_nav_stack_running()  # our child: ours
        monkeypatch.setattr(explore.os, "getpid", lambda: other.pid)  # seen from a process
        assert not explore.is_nav_stack_running()  # with no such child: not ours
        assert not explore._tare_running()
    finally:
        other.kill()
        other.wait()
