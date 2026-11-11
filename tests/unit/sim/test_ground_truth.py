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
    from vector_os_nano.vcli.worlds.world_manifest import load_world

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
