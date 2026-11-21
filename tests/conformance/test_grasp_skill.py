# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The same camera-driven grasp on both backends; truth only grades the final state."""
import pytest
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.embodiments import registry
from vector_os_nano.perception.camera_perception import CameraPerception
from vector_os_nano.sim.drivers.arm import Arm
from vector_os_nano.sim.drivers.camera import Camera
from vector_os_nano.sim.drivers.gripper import Gripper
from vector_os_nano.sim.ground_truth import GroundTruth
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.skills.grasp import GraspSkill
from vector_os_nano.schemas.world import load_world

pytestmark = pytest.mark.sim


def test_fixed_piper_grasp_from_rgbd_on_each_backend(backend, load):
    cfg, world = registry.load("piper"), load_world("workbench")
    scene = compose(world.scene, cfg, spawn=world.spawn, pickables=tuple(world.pickables))
    sim = load(backend, scene)
    rt = SimRuntime(sim, substeps=5, threaded=False, paced=False, name="grasp-conformance")
    try:
        rt.wait_sim(.5)
        arm, gripper = Arm(cfg.arm, rt), Gripper(cfg.arm.gripper, rt)
        ctx = SkillContext(arm=arm, gripper=gripper, perception=CameraPerception(Camera(cfg.camera, rt)),
                           config={"skills": cfg.skills, "arm": {"postures": cfg.arm.postures}},
                           services={"object_world": world})
        result = GraspSkill().execute({"query": "red box"}, ctx)
        assert result.success, result
        gt = GroundTruth(rt)
        assert gt.attached_to("red_box") == cfg.arm.gripper.weld_body
        assert {scene.body(b) for b in cfg.arm.gripper.finger_bodies} <= gt.contacts("red_box")
        assert gt.clearance("red_box") >= .03
        assert result.result_data["segmenter"] == "hsv"
    finally:
        rt.close()
