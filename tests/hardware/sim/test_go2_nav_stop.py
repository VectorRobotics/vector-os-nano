# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Real go2 sim (LeggedBase + convex_mpc): a navigate or walk that is interrupted always leaves the
dog stopped and the control token free.

Pins skills-nav:inprocess-nav-no-abort-stall (a stop from another thread ends the
navigate as 'aborted'), skills-nav:walk-runaway-on-exception (Ctrl-C mid-walk or
mid-navigate zeroes the command) and the StopSkill gate (its stop is honoured
while a navigate holds the skill token).
"""
from __future__ import annotations

import _thread
import math
import threading
import time

import pytest

mujoco = pytest.importorskip("mujoco")

from tests.hardware.sim.dog import stand_up  # noqa: E402
from vector_os_nano.core.skill import SkillContext  # noqa: E402
from vector_os_nano.skills.go2.stop import StopSkill  # noqa: E402
from vector_os_nano.vcli.cognitive import abort  # noqa: E402

_STUDY = (17.0, 7.5)  # a ~9 m leg from the spawn


pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def go2():
    dog = stand_up("go2")
    yield dog.base
    dog.close()


def _drift(dog, seconds: float) -> float:
    a = dog.get_position()
    dog.wait_sim(seconds)
    b = dog.get_position()
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _released(dog) -> bool:
    """The command is zero, no thread holds skill control and none holds the clock."""
    return dog._cmd == (0.0, 0.0, 0.0) and not dog._owners and dog._rt._hold is None


def test_ctrl_c_mid_navigate_stops_the_dog(go2) -> None:
    timer = threading.Timer(3.0, _thread.interrupt_main)
    start = go2.get_position()
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        go2.navigate_to(*_STUDY)
    assert _released(go2)
    moved = math.hypot(go2.get_position()[0] - start[0], go2.get_position()[1] - start[1])
    assert moved > 0.2  # it was walking when the Ctrl-C landed
    go2.wait_sim(1.0)  # the legged body settles
    assert _drift(go2, 1.0) < 0.05


def test_ctrl_c_mid_walk_stops_the_dog(go2) -> None:
    timer = threading.Timer(1.0, _thread.interrupt_main)
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        go2.walk(-0.3, 0.0, 0.0, 6.0)
    assert _released(go2)
    go2.wait_sim(1.0)
    assert _drift(go2, 1.0) < 0.05


def test_stop_skill_from_another_thread_aborts_the_navigate(go2) -> None:
    out: dict = {}

    def run() -> None:  # a navigate runs inside a run (the turn worker binds one), and
        abort.new_run()  # StopSkill spares only the stopper's own run (WP05)
        out["r"] = go2.navigate_to(*_STUDY)

    th = threading.Thread(target=run)
    th.start()
    time.sleep(3.0)
    try:
        assert StopSkill().execute({}, SkillContext(base=go2)).success
        t0 = time.monotonic()
        th.join(10.0)
        latency = time.monotonic() - t0
    finally:
        abort.clear_abort()
    assert not th.is_alive() and out["r"].reason == "aborted", out
    assert latency < 1.0
    assert _released(go2)
    go2.wait_sim(1.0)
    assert _drift(go2, 1.0) < 0.05
