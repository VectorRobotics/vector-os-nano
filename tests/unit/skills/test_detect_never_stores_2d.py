# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""skills-manip:detect-zero-sentinel-pollution — DetectSkill stored 2D-only boxes at
(0, 0, 0), and pick then "found" them there."""
from __future__ import annotations

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.perception.calibration import Calibration
from vector_os_nano.skills.detect import DetectSkill


class _BoxesOnly:
    """A detector without 3D: tracking fails, as on a go2 camera without a tracker."""

    def detect(self, query):
        return [Detection(label="red can", bbox=(10, 10, 40, 60), confidence=0.8)]

    def track(self, detections):
        raise RuntimeError("no depth tracker")


def test_a_box_without_a_3d_pose_never_enters_the_world_model() -> None:
    wm = WorldModel()
    res = DetectSkill().execute({"query": "red can"},
                                SkillContext(perception=_BoxesOnly(), world_model=wm, calibration=Calibration()))
    assert res.success and res.result_data["count"] == 1
    assert res.result_data["objects"][0]["has_3d"] is False
    assert wm.get_objects() == []


def test_detect_without_a_calibration_refuses_and_stores_nothing() -> None:
    """perception:calibration-missing-silent-camera-frame: with no calibration, detect
    stored CAMERA-frame coordinates as arm-base ones (has_3d True)."""
    from vector_os_nano.core.types import Pose3D, TrackedObject

    class _Tracked(_BoxesOnly):
        def track(self, detections):
            return [TrackedObject(track_id=1, label="red can", bbox_2d=(10, 10, 40, 60),
                                  pose=Pose3D(x=0.1, y=0.0, z=0.4))]

    wm = WorldModel()
    res = DetectSkill().execute({"query": "red can"}, SkillContext(perception=_Tracked(), world_model=wm))
    assert not res.success and res.result_data["diagnosis"] == "calibration_error"
    assert wm.get_objects() == []


def test_a_mobile_base_gets_no_identity_calibration() -> None:
    from vector_os_nano.core.agent import Agent

    class _Base:
        name = "base"

    agent = Agent(base=_Base())
    assert agent._build_context().calibration is None


class _Snapshot:
    """Go2GraspPerception's surface after WP11: capture/detect_in/segment, no detect(query)."""

    def __init__(self) -> None:
        self.seen = []

    def capture(self):
        from types import SimpleNamespace

        return SimpleNamespace(rgb="frame-1")

    def detect_in(self, rgb, query):
        self.seen.append((rgb, query))
        return [Detection(label="red can", bbox=(10, 10, 40, 60), confidence=0.8)]


def test_detect_runs_on_a_snapshot_perception() -> None:
    """I23: DetectSkill called perception.detect(query), which the snapshot backend no
    longer has, so every detect on go2_piper failed with a perception error."""
    p, wm = _Snapshot(), WorldModel()
    res = DetectSkill().execute({"query": "red can"}, SkillContext(perception=p, world_model=wm, calibration=Calibration()))
    assert res.success and res.result_data["count"] == 1, res.error_message
    assert p.seen == [("frame-1", "red can")]  # one capture, detected in that frame
    assert res.result_data["objects"][0]["has_3d"] is False and wm.get_objects() == []


def test_pick_live_path_detects_on_a_snapshot_perception() -> None:
    """I23: PickSkill's live re-detect also called perception.detect(query)."""
    from types import SimpleNamespace

    from vector_os_nano.skills.pick import PickSkill

    p = _Snapshot()
    ctx = SimpleNamespace(perception=p, config={})
    assert PickSkill()._sample_from_perception({"object_label": "red can"}, ctx) is None  # no tracker: no 3D
    assert p.seen == [("frame-1", "red can")]


class _Arm:
    name = "arm"

    def get_joint_positions(self):
        return [0.0] * 5

    def move_joints(self, q, duration=1.0):
        return True


class _TrackedCan(_BoxesOnly):
    def track(self, detections):
        from vector_os_nano.core.types import Pose3D, TrackedObject

        return [TrackedObject(track_id=1, label="red can", bbox_2d=(10, 10, 40, 60),
                              pose=Pose3D(x=0.1, y=0.0, z=0.4))]


def _detect_on(agent):
    return DetectSkill().execute({"query": "red can"}, agent._build_context())


def test_the_sim_arm_perception_needs_no_calibration_file() -> None:
    """R2V-03: the sim arm (MuJoCoPerception reports arm-base points) built with the
    default config (whose calibration.file does not ship) refused every detect."""
    import numpy as np

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception

    p = np.array([0.1, -0.2, 0.3])
    assert np.allclose(MuJoCoPerception(None).calibration.camera_to_base(p), p)  # identity
    perception = _TrackedCan()
    perception.calibration = Calibration()  # what a base-frame perception declares
    res = _detect_on(Agent(arm=_Arm(), perception=perception))  # config=None: default.yaml
    assert res.success and res.result_data["objects"][0]["has_3d"] is True, res.error_message


def test_a_config_naming_no_calibration_file_refuses_and_says_so() -> None:
    """R2V-03 (adversarial): a dict config without calibration.file got an identity
    calibration silently, so a real arm would drive to camera-frame coordinates."""
    from vector_os_nano.core.agent import Agent

    res = _detect_on(Agent(arm=_Arm(), perception=_TrackedCan(), config={}))
    assert not res.success and res.result_data["diagnosis"] == "calibration_error"
    assert "names no calibration.file" in res.error_message


def test_an_unreadable_calibration_file_refuses_and_names_it(tmp_path) -> None:
    from vector_os_nano.core.agent import Agent

    missing = str(tmp_path / "no_such_calibration.yaml")
    agent = Agent(arm=_Arm(), perception=_TrackedCan(), config={"calibration": {"file": missing}})
    res = _detect_on(agent)
    assert not res.success and res.result_data["diagnosis"] == "calibration_error"
    assert missing in res.error_message
