# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""perception:look-reports-crashed-detector-as-nothing — on every perception path a
camera, detector or segmenter that RAISES is a failure with its cause, never an
empty result: look and describe_scene told the model 'nothing localized' when the
detector was out of GPU memory or had no weights (gates e10 / e11, 2026-11-12)."""
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest import mock

import pytest

from tests.harness.kin_base import KinBase
from tests.unit.perception.test_object_localizer import Raising, no_weights, oom
from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.perception.calibration import Calibration
from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.skills.go2.look import DescribeSceneSkill, LookSkill, observe_scene
from vector_os_nano.vcli.worlds.world_manifest import load_world

_APT = load_world("apartment")
_OOM_CAUSE = "out of GPU memory on cuda:0 (another process may hold it)"
_PREFETCH = "pre-fetch with: python scripts/fetch_assets.py --weights"


def _sg() -> SceneGraph:
    sg = SceneGraph(resolve=_APT.resolve)
    sg.load_layout(str(_APT.path))
    return sg


class _Base:
    def get_position(self):
        return (10.0, 3.0, 0.3)

    def get_heading(self):
        return 0.0


def _ctx(perception, vlm=None) -> SkillContext:
    return SkillContext(base=_Base(), perception=perception, config={"world": "apartment"},
                        services={"spatial_memory": _sg(), **({"vlm": vlm} if vlm else {})})


@pytest.mark.parametrize("where, exc, cause", [
    ("detect", oom, f"the detector failed: {_OOM_CAUSE}"),
    ("detect", lambda: RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has"),
     f"the detector failed: {_OOM_CAUSE}"),
    ("detect", no_weights, "the detector failed: IDEA-Research/grounding-dino-tiny@a2bb814"),
    ("segment", no_weights, "the segmenter failed: IDEA-Research/grounding-dino-tiny@a2bb814"),
])
def test_look_with_a_model_that_cannot_run_fails_with_its_cause(where, exc, cause) -> None:
    res = LookSkill().execute({}, _ctx(Raising(where, exc())))
    assert not res.success and res.diagnosis_code == "perception_failed"
    assert cause in res.error_message and "nothing localized" not in res.error_message
    assert "this is not an empty room" in res.error_message
    if exc is no_weights:
        assert _PREFETCH in res.error_message


def test_look_whose_capture_raises_is_a_camera_failure_with_its_cause() -> None:
    res = LookSkill().execute({}, _ctx(Raising("capture", OSError("EGL context lost"))))
    assert not res.success and res.diagnosis_code == "camera_failed"
    assert "EGL context lost" in res.error_message


def _oom_on_gpu1_with_a_path() -> BaseException:
    torch = pytest.importorskip("torch")
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB. GPU 1 has a "
                                       "total capacity of 23.6 GiB; /home/alice/x/y.py")


def test_a_capture_out_of_gpu_memory_is_named_with_its_device_and_no_path() -> None:
    res = LookSkill().execute({}, _ctx(Raising("capture", _oom_on_gpu1_with_a_path())))
    assert not res.success and res.diagnosis_code == "camera_failed"
    assert "Camera capture failed: out of GPU memory on cuda:1" in res.error_message, res.error_message
    assert "/home" not in res.error_message


def test_a_vlm_that_raises_is_a_vlm_failure_with_no_machine_path() -> None:
    vlm = mock.MagicMock()
    vlm.describe_scene.side_effect = RuntimeError("cannot read /data/models/m.bin or /tmp/zz/q")
    res = DescribeSceneSkill().execute({}, _ctx(Raising("none", None), vlm))
    assert not res.success and res.diagnosis_code == "vlm_failed"
    assert res.error_message == "VLM inference failed: RuntimeError: cannot read <path> or <path>", \
        res.error_message


def test_one_name_of_three_failing_is_named_and_the_rest_still_report() -> None:
    names = ["green bottle", "blue bottle", "red can"]
    perc = Raising("detect", RuntimeError("boom"), only=("blue bottle",))
    data = observe_scene(_Base(), perc, _sg(), None, names)
    assert data["failed"] == {"blue bottle": "the detector failed: RuntimeError: boom"}
    assert "1 of 3 names could not be checked: blue bottle: the detector failed: RuntimeError: boom" \
        in data["summary"], data["summary"]


class _FindsNothing(Raising):
    """The detector runs and finds nothing, except where it raises."""

    def detect_in(self, rgb, query):
        self._maybe("detect", query)
        return []


def test_nothing_localized_is_said_only_of_names_that_ran() -> None:
    names = ["green bottle", "red can"]
    perc = _FindsNothing("detect", oom(), only=("red can",))
    data = observe_scene(_Base(), perc, _sg(), None, names)
    assert data["summary"] == (f"Seen (detector, no VLM): nothing localized. 1 of 2 names could not be "
                               f"checked: red can: the detector failed: {_OOM_CAUSE}.")
    assert data["objects"] == [] and data["unlocalized"] == []


def test_describe_scene_whose_segmenter_cannot_load_fails_with_the_prefetch_command() -> None:
    vlm = mock.MagicMock()
    vlm.find_objects.return_value = [SimpleNamespace(name="red can", description="")]
    res = DescribeSceneSkill().execute({"query": "red things"}, _ctx(Raising("segment", no_weights()), vlm))
    assert not res.success and res.diagnosis_code == "perception_failed"
    assert _PREFETCH in res.error_message and "/home/" not in res.error_message


def test_a_vlm_named_object_that_could_not_be_checked_is_failed_not_unlocalized() -> None:
    vlm = mock.MagicMock()
    vlm.find_objects.return_value = [SimpleNamespace(name=n, description="") for n in ("red can", "sofa")]
    perc = _FindsNothing("detect", oom(), only=("red can",))
    res = DescribeSceneSkill().execute({"query": "what is here"}, _ctx(perc, vlm))
    assert res.success and res.result_data["unlocalized"] == ["sofa"]
    assert res.result_data["failed"] == {"red can": f"the detector failed: {_OOM_CAUSE}"}


class _Tour(KinBase):
    """A base whose planner lands exactly on every target (the coverage plumbing)."""

    def navigate_to(self, x, y, cancel=None, **_):
        from vector_os_nano.sim.nav.follow import NavResult

        self.x, self.y = x, y
        return NavResult(reached=True, reason="arrived", target=[x, y], snapped=False,
                         goal=[x, y], pos=[x, y, 0.3])


def test_explore_reports_the_looks_on_the_way_that_could_not_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_RUN_DIR", str(tmp_path))
    from vector_os_nano.skills.go2 import explore

    ctx = SkillContext(base=_Tour(10.0, 3.0), perception=Raising("detect", oom()),
                       config={"world": "apartment"}, services={"spatial_memory": _sg()})
    res = explore.ExploreSkill().execute({}, ctx)
    assert res.success, res.error_message  # the rooms were explored: that is explore's job
    note = res.result_data["perception_failed"]
    assert re.match(r"[1-9]\d* look\(s\) on the way could not check every name: Perception could not run "
                    r"for any of the \d+ names, so nothing was checked \(this is not an empty room\): "
                    "the detector failed: out of GPU memory on cuda:0", note), note
    assert "nothing localized" not in note
    assert explore.get_explore_status()["perception_failed"] == note


def test_explore_note_for_a_look_whose_capture_ran_out_of_gpu_memory_has_no_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_RUN_DIR", str(tmp_path))
    from vector_os_nano.skills.go2 import explore

    ctx = SkillContext(base=_Tour(10.0, 3.0), perception=Raising("capture", _oom_on_gpu1_with_a_path()),
                       config={"world": "apartment"}, services={"spatial_memory": _sg()})
    note = explore.ExploreSkill().execute({}, ctx).result_data["perception_failed"]
    assert "Camera capture failed: out of GPU memory on cuda:1" in note and "/home" not in note, note


def test_detect_skill_whose_detector_cannot_run_names_the_cause() -> None:
    class _Perc:
        def detect(self, query):
            raise no_weights()

    res = DetectSkill().execute(
        {"query": "red can"}, SkillContext(perception=_Perc(), world_model=WorldModel(), calibration=Calibration()))
    assert not res.success and res.result_data["diagnosis"] == "perception_failed"
    assert _PREFETCH in res.error_message and "nothing was checked" in res.error_message


def test_the_producers_detect_tool_carries_the_cause_of_a_capture_or_a_model() -> None:
    from vector_os_nano.perception.detector_capability import DetectorCapability
    from vector_os_nano.vcli.native_tools import _NativeDetectTool

    class _Model:
        def detect(self, rgb, query):
            raise oom()

    out = _NativeDetectTool(DetectorCapability(detector=_Model(), perception=Raising("none", oom()))).execute(
        {"query": "red can"}, None)
    assert out.is_error and _OOM_CAUSE in out.content and "RAN but localized nothing" not in out.content
    out = _NativeDetectTool(DetectorCapability(detector=_Model(), perception=Raising(
        "capture", OSError("Can't open /dev/dri/renderD128 for EGL")))).execute({"query": "red can"}, None)
    assert out.is_error
    assert "the camera capture failed: OSError: Can't open <path> for EGL" in out.content, out.content
