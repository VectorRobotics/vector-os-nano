# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundTruth's own guards (S1), adversarial cases first: the attach log fails closed
when it no longer reaches back far enough, and a grasp counts only on the manifest's
weld body."""
from __future__ import annotations

import collections
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld  # noqa: E402
from vector_os_nano.sim.ground_truth import GroundTruth  # noqa: E402
from vector_os_nano.sim.runtime import SimRuntime  # noqa: E402
from vector_os_nano.sim.scene import compiled_scene  # noqa: E402

_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot" pos="0 0 0.5">
      <joint name="lift" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="hand" pos="0 0 -0.2"><geom type="box" size="0.02 0.02 0.02"/></body>
    </body>
    <body name="box" pos="0.5 0 0.05"><freejoint/><geom type="box" size="0.05 0.05 0.05"/></body>
  </worldbody>
  <equality><weld body1="hand" body2="box" active="false"/></equality>
</mujoco>
"""


@pytest.fixture
def rt(tmp_path):
    path = tmp_path / "gt.xml"
    path.write_text(_XML)
    r = SimRuntime(MuJoCoWorld(compiled_scene(path, robot_root="robot", grasp_pairs=[("hand", "box")])),
                   threaded=False, paced=False, name="gt_test")
    yield r
    r.close()


def test_the_robot_is_never_an_object_and_attachments_are_named(rt) -> None:
    gt = GroundTruth(rt)
    sc = rt.world.scene
    assert set(gt.bid) == {"box"} and gt.robot_bodies == {sc.body("robot"), sc.body("hand")}
    assert gt.has_base is False and gt.base_pose() is None  # a slide-jointed root is fixed
    rt.call(lambda w: w.attach(sc.body("hand"), sc.body("box")))
    assert gt.attachments() == {"box": "hand"} and gt.attached_to("box") == "hand"
    assert [(n, p) for _t, n, p in gt.attach_events()] == [("box", "hand")]


def test_attach_events_fail_closed_when_the_log_was_truncated(rt) -> None:
    gt = GroundTruth(rt)
    sc = rt.world.scene
    rt.attach_log = collections.deque(maxlen=2)  # a tiny log, so it overflows here
    hand, box = sc.body("hand"), sc.body("box")
    for _ in range(2):  # attach, detach, attach, detach: the first two drop out
        rt.call(lambda w: w.attach(hand, box))
        rt.step(1)
        rt.call(lambda w: w.detach(box))
        rt.step(1)
    first = rt.attach_log[0][0]
    with pytest.raises(RuntimeError, match="truncated"):
        gt.attach_events()  # since -inf: the dropped events are inside the window
    with pytest.raises(RuntimeError, match="truncated"):
        gt.attach_events(since=first - 1e-6)
    assert [p for _t, _n, p in gt.attach_events(since=first)] == [None]  # fully covered


def test_raw_weld_writes_in_one_drained_batch_each_log(tmp_path) -> None:
    """Adversarial: two jobs that flip the weld by a raw ``eq_active`` write (the SO-101
    gripper's way), drained by the stepper in ONE batch, still log both transitions."""
    path = tmp_path / "gt.xml"
    path.write_text(_XML)
    rt = SimRuntime(MuJoCoWorld(compiled_scene(path, robot_root="robot", grasp_pairs=[("hand", "box")])),
                    threaded=True, paced=False, name="gt_batch")
    sc = rt.world.scene
    try:
        rt.start()
        with rt.paused(), rt.lock:  # the parked stepper drains only after both are queued
            on = rt.submit(lambda w: w.data.eq_active.__setitem__(0, 1))
            off = rt.submit(lambda w: w.data.eq_active.__setitem__(0, 0))
        on.result(5.0), off.result(5.0)
        assert [p for _t, _c, p in rt.attach_log] == [sc.body("hand"), None]
    finally:
        rt.close()


def test_a_job_cannot_set_attributes_on_the_world_it_is_handed(rt) -> None:
    """Jobs see the world through the runtime's logging view: a stray attribute write
    fails loud instead of landing on the view and silently doing nothing."""
    with pytest.raises(AttributeError):
        rt.call(lambda w: setattr(w, "attach", None))


def test_a_job_cannot_reach_the_unlogged_world_through_its_view(rt) -> None:
    """S1F1V-1: no attribute of the view holds the runtime or the bare world, so a job
    cannot attach / detach around the call-time log; private names are refused."""
    sc = rt.world.scene
    for name in ("_rt", "_world", "_note_attachments", "_welds"):
        with pytest.raises(AttributeError):
            rt.call(lambda w, n=name: getattr(w, n))
    view = rt.call(lambda w: w)
    assert all(getattr(view, n, None) is not rt for n in dir(view))
    assert all(getattr(view, n, None) is not rt.world for n in dir(view))
    rt.call(lambda w: w.attach(sc.body("hand"), sc.body("box")))  # the public path still logs
    assert [p for _t, _c, p in rt.attach_log] == [sc.body("hand")]


def test_clearance_support_is_a_shape_top_at_most_1cm_above_the_bottom(tmp_path) -> None:
    """S1F1V-2: a support top 8 mm ABOVE the object's lowest point (the object resting
    sunk into it) is still its support, a shape top 28 mm above it never is: the 1 cm
    support epsilon is pinned to [8 mm, 28 mm) (the resting scenarios only sink ~0.1 mm)."""
    xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="table" type="box" pos="0 0 0.2" size="0.3 0.3 0.2"/>
    <geom name="pillar" type="box" pos="0.2 0 0.21" size="0.02 0.02 0.21"/>
    <body name="low" pos="-0.1 0 0.437"><freejoint/><geom type="box" size="0.05 0.05 0.045"/></body>
    <body name="sunk" pos="0.2 0 0.427"><freejoint/><geom type="box" size="0.01 0.01 0.035"/></body>
  </worldbody>
</mujoco>
"""
    path = tmp_path / "eps.xml"
    path.write_text(xml)
    rt = SimRuntime(MuJoCoWorld(compiled_scene(path)), threaded=False, paced=False, name="eps")
    try:
        gt = GroundTruth(rt)
        # bottoms at 0.392, the table top at 0.40: both rest 8 mm deep in the table
        assert gt.clearance("low") == pytest.approx(-0.008, abs=1e-9)
        # the pillar top (0.42) is 28 mm above the sunk box's bottom: not its support
        assert gt.clearance("sunk") == pytest.approx(-0.008, abs=1e-9)
    finally:
        rt.close()


def _stub_gt(parent: str) -> SimpleNamespace:
    """A GroundTruth that says the green bottle is gripped by both Piper fingers, 10 cm
    up, attached to *parent*."""
    held = {"pickable_bottle_green": parent}
    fingers = {9001, 9002}
    ids = {"piper_link7": 9001, "piper_link8": 9002}
    return SimpleNamespace(
        bid={"pickable_bottle_green": 1}, has_base=False, robot_bodies=frozenset(fingers),
        body_id=lambda n: ids.get(n, 77), attachments=lambda: dict(held),
        attached_to=lambda n: held.get(n), attach_events=lambda since=-math.inf: [],
        contacts=lambda n: frozenset(fingers), clearance=lambda n: 0.10, object_speed=lambda n: 0.0,
        object_positions=lambda: {"pickable_bottle_green": [10.9, 3.0, 0.42]},
    )


def test_a_grasp_counts_only_on_the_manifest_weld_body() -> None:
    """MC1: attached to any other body (the wrist link before the hand) is not held."""
    from vector_os_nano.embodiments import registry
    from vector_os_nano.vcli.worlds.sim_oracles import build_namespace
    from vector_os_nano.schemas.world import load_world

    cfg = registry.load("go2_piper")
    world = load_world(cfg.world)
    assert cfg.arm.gripper.weld_body == "piper_link6"
    wrong = build_namespace(_stub_gt("piper_link5"), world, cfg)
    assert not wrong["holding_object"]("green bottle") and not wrong["holding_object"]()
    assert not wrong["lifted"]("green bottle") and not wrong["kept_holding"]("green bottle")
    assert wrong["released"]()  # nothing is on the gripper
    right = build_namespace(_stub_gt("piper_link6"), world, cfg)
    assert right["holding_object"]("green bottle") and right["holding_object"]()
    assert right["kept_holding"]("green bottle") and not right["released"]()
