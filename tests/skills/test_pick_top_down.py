"""Unit tests for PickTopDownSkill using mocked hardware.

The skill's wiring (hardware preconditions, world-model lookup, motion
sequencing, return-data structure) is verified here with a mock arm and
gripper. The actual physics / MuJoCo behaviour is exercised by the E2E
stability runs in scripts/verify_pick_top_down.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.pick_top_down import PickTopDownSkill


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockArm:
    """ArmProtocol-compatible mock. Records calls; ik_top_down returns a
    pre-canned solution unless force_ik_fail is set for a specific phase.
    """

    name = "mock_arm"
    dof = 6
    joint_names = [f"piper_joint{i}" for i in range(1, 7)]

    def __init__(self, ik_fail_phase: str | None = None):
        self.calls: list[tuple[str, object]] = []
        self._joints = [0.0] * 6
        self._ik_call_count = 0
        self._ik_fail_phase = ik_fail_phase  # "pre" or "grasp" or None

    def get_joint_positions(self):
        return list(self._joints)

    def move_joints(self, positions, duration=3.0):
        self.calls.append(("move_joints", (list(positions), duration)))
        self._joints = list(positions)
        return True

    def ik_top_down(self, target, current_joints=None):
        self.calls.append(("ik_top_down", (tuple(target), current_joints)))
        self._ik_call_count += 1
        if self._ik_fail_phase == "pre" and self._ik_call_count == 1:
            return None
        if self._ik_fail_phase == "grasp" and self._ik_call_count == 2:
            return None
        return [0.1] * 6

    def stop(self):
        self.calls.append(("stop", None))

    def fk(self, joint_positions):
        return [0.0, 0.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class _MockGripper:
    def __init__(self, holding_after_close: bool = True):
        self.calls: list[str] = []
        self._holding_after_close = holding_after_close
        self._closed = False

    def open(self):
        self.calls.append("open")
        self._closed = False
        return True

    def close(self):
        self.calls.append("close")
        self._closed = True
        return True

    def is_holding(self):
        return self._closed and self._holding_after_close

    def get_position(self):
        return 0.0 if self._closed else 1.0

    def get_force(self):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mkctx(arm=None, gripper=None, world_model=None, config=None):
    return SkillContext(
        arm=arm, gripper=gripper, base=SimpleNamespace(),
        world_model=world_model, config=config or {},
    )


def _mkworld(*objs: ObjectState) -> WorldModel:
    wm = WorldModel()
    for o in objs:
        wm.add_object(o)
    return wm


# ---------------------------------------------------------------------------
# Preconditions / failure modes
# ---------------------------------------------------------------------------


def test_no_arm_returns_no_arm_diag():
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=None, gripper=_MockGripper(), world_model=_mkworld())
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.success is False
    assert r.result_data["diagnosis"] == "no_arm"


def test_no_gripper_returns_no_gripper_diag():
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=_MockArm(), gripper=None, world_model=_mkworld())
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.result_data["diagnosis"] == "no_gripper"


def test_no_world_model_returns_no_world_model_diag():
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=_MockArm(), gripper=_MockGripper(), world_model=None)
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.result_data["diagnosis"] == "no_world_model"


def test_arm_without_ik_top_down_rejected():
    """An arm missing ``ik_top_down`` must fail with arm_unsupported."""
    class _BareArm:
        name = "bare"
        dof = 5
    arm = _BareArm()
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=arm, gripper=_MockGripper(), world_model=_mkworld())
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.result_data["diagnosis"] == "arm_unsupported"


def test_object_not_found_lists_known():
    skill = PickTopDownSkill()
    wm = _mkworld(ObjectState(object_id="pickable_a", label="a cube", x=0.5, y=0, z=0.1))
    ctx = _mkctx(arm=_MockArm(), gripper=_MockGripper(), world_model=wm)
    r = skill.execute({"object_id": "pickable_nonexistent"}, ctx)
    assert r.result_data["diagnosis"] == "object_not_found"
    assert "a cube" in r.result_data["known_objects"]


def test_ik_unreachable_pregrasp():
    skill = PickTopDownSkill()
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=1.0, y=0, z=0.2))
    ctx = _mkctx(arm=_MockArm(ik_fail_phase="pre"), gripper=_MockGripper(), world_model=wm)
    r = skill.execute({"object_id": "o1"}, ctx)
    assert r.success is False
    assert r.result_data["diagnosis"] == "ik_unreachable"
    assert r.result_data["phase"] == "pre_grasp"


def test_ik_unreachable_grasp():
    skill = PickTopDownSkill()
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=1.0, y=0, z=0.2))
    ctx = _mkctx(arm=_MockArm(ik_fail_phase="grasp"), gripper=_MockGripper(), world_model=wm)
    r = skill.execute({"object_id": "o1"}, ctx)
    assert r.result_data["diagnosis"] == "ik_unreachable"
    assert r.result_data["phase"] == "grasp"


# ---------------------------------------------------------------------------
# Happy path — motion sequencing
# ---------------------------------------------------------------------------


def test_pick_sequence_opens_then_moves_then_closes():
    """Verify the call order: gripper.open → move(pre) → move(grasp) →
    gripper.close → move(pre lifted). No final "home" move."""
    arm = _MockArm()
    gripper = _MockGripper(holding_after_close=True)
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=0.5, y=0, z=0.2))
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"object_id": "o1"}, ctx)

    assert r.success is True
    assert r.result_data["grasped_heuristic"] is True
    assert r.result_data["object_id"] == "o1"

    # Gripper: open before motion, close after descent — exactly once each
    assert gripper.calls == ["open", "close"]

    # Arm: two IK solves (pre + grasp), three moves (pre, grasp, lift)
    ik_calls = [c for c in arm.calls if c[0] == "ik_top_down"]
    move_calls = [c for c in arm.calls if c[0] == "move_joints"]
    assert len(ik_calls) == 2
    assert len(move_calls) == 3, f"expected 3 moves (pre/grasp/lift), got {len(move_calls)}"


def test_pick_uses_object_label_when_id_missing():
    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld(ObjectState(object_id="pickable_bottle_blue",
                              label="blue bottle", x=0.5, y=0, z=0.2))
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"object_label": "blue bottle"}, ctx)
    assert r.success is True
    assert r.result_data["object_id"] == "pickable_bottle_blue"


def test_pick_target_xyz_override():
    """Explicit target_xyz bypasses world_model lookup."""
    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()  # empty
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"target_xyz": [0.5, 0.0, 0.2]}, ctx)
    assert r.success is True


def test_grasp_heuristic_false_reports_possibly_missed():
    arm = _MockArm()
    gripper = _MockGripper(holding_after_close=False)
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=0.5, y=0, z=0.2))
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"object_id": "o1"}, ctx)
    assert r.success is True  # motion sequence completed
    assert r.result_data["diagnosis"] == "possibly_missed"
    assert r.result_data["grasped_heuristic"] is False


# ---------------------------------------------------------------------------
# Metadata / descriptor
# ---------------------------------------------------------------------------


def test_skill_has_canonical_fields():
    skill = PickTopDownSkill()
    assert skill.name == "pick_top_down"
    assert "object_id" in skill.parameters
    assert "object_label" in skill.parameters
    assert "ik_unreachable" in skill.failure_modes
    assert "arm_unsupported" in skill.failure_modes


def test_skill_aliases_registered_for_chinese_and_english():
    aliases = PickTopDownSkill.__skill_aliases__
    assert "抓" in aliases
    assert "grab" in aliases
    assert "pick" in aliases


# ---------------------------------------------------------------------------
# _normalise_color_keyword — Chinese color token normaliser (T3)
# ---------------------------------------------------------------------------

from vector_os_nano.skills.pick_top_down import _normalise_color_keyword  # noqa: E402


def test_normalise_color_keyword_chinese_suffix():
    result = _normalise_color_keyword("绿色")
    assert result is not None
    assert "green" in result
    assert "绿" not in result
    assert "色" not in result


def test_normalise_color_keyword_single_char():
    result = _normalise_color_keyword("红")
    assert result is not None
    assert "red" in result
    assert "红" not in result


def test_normalise_color_keyword_returns_none_if_no_match():
    assert _normalise_color_keyword("bottle") is None
    assert _normalise_color_keyword("紫色") is None


@pytest.mark.parametrize("cn,en", [
    ("红色", "red"),
    ("绿色", "green"),
    ("蓝色", "blue"),
    ("黄色", "yellow"),
    ("白色", "white"),
    ("黑色", "black"),
])
def test_normalise_color_keyword_all_six_colors(cn: str, en: str):
    result = _normalise_color_keyword(cn)
    assert result is not None
    assert en in result


def test_normalise_color_keyword_mixed_input_preserves_other_chars():
    result = _normalise_color_keyword("抓前面绿色瓶子")
    assert result is not None
    assert "green" in result
    assert "抓前面" in result
    assert "瓶子" in result
    assert "绿" not in result


# ---------------------------------------------------------------------------
# _resolve_target — new fallback passes (T7)
# ---------------------------------------------------------------------------

import logging  # noqa: E402


def test_resolve_target_matches_chinese_color_after_normalise():
    """Chinese color label normalises to English and matches a stored object."""
    wm = _mkworld(
        ObjectState(object_id="pickable_bottle_green", label="green bottle",
                    x=11.0, y=3.0, z=0.2),
        ObjectState(object_id="pickable_bottle_blue", label="blue bottle",
                    x=11.0, y=2.85, z=0.2),
    )
    skill = PickTopDownSkill()
    result = skill._resolve_target({"object_label": "抓前面绿色"}, wm)
    assert result is not None
    obj_id, _xyz = result
    assert obj_id == "pickable_bottle_green"


def test_resolve_target_returns_none_when_label_unmatched():
    """Unmatched label returns None — no silent substitution (perception must
    populate the world model via detect_*, not be bypassed)."""
    wm = _mkworld(
        ObjectState(object_id="pickable_can_red", label="red can",
                    x=11, y=3.15, z=0.2),
    )
    skill = PickTopDownSkill()
    result = skill._resolve_target({"object_label": "紫色"}, wm)
    assert result is None


def test_resolve_target_returns_none_when_multiple_pickables_and_unmatched():
    """Multiple pickables + no exact/normalised match → None."""
    wm = _mkworld(
        ObjectState(object_id="pickable_a", label="red can", x=11, y=3, z=0.2),
        ObjectState(object_id="pickable_b", label="blue bottle", x=11, y=2.85, z=0.2),
        ObjectState(object_id="pickable_c", label="green cup", x=11, y=3.15, z=0.2),
    )
    skill = PickTopDownSkill()
    result = skill._resolve_target({"object_label": "紫色"}, wm)
    assert result is None


def test_resolve_target_prefers_explicit_label_match_over_color_normalise(caplog):
    """Direct English label match (step 3) must resolve before normaliser (step 4).
    Verified via caplog — no colour-normalisation log line should appear."""
    wm = _mkworld(
        ObjectState(object_id="pickable_bottle_green", label="green bottle",
                    x=5.0, y=1.0, z=0.3),
    )
    skill = PickTopDownSkill()
    with caplog.at_level(logging.INFO, logger="vector_os_nano.skills.pick_top_down"):
        result = skill._resolve_target({"object_label": "green bottle"}, wm)
    assert result is not None
    assert result[0] == "pickable_bottle_green"
    assert "colour-normalisation" not in caplog.text


# ---------------------------------------------------------------------------
# Error-message enrichment (helps VGG re-plan retry with a valid label)
# ---------------------------------------------------------------------------


def test_execute_object_not_found_error_message_lists_known_objects():
    """error_message must include the known-object labels so the VGG re-plan
    LLM can retry with a valid label rather than injecting detect_*."""
    arm = _MockArm()
    gripper = _MockGripper()
    # Use specific color label that doesn't match — triggers object_not_found
    wm = _mkworld(
        ObjectState(object_id="pickable_bottle_blue", label="blue bottle",
                    x=11, y=2.85, z=0.25),
        ObjectState(object_id="pickable_can_red", label="red can",
                    x=11, y=3.15, z=0.25),
    )
    ctx = _mkctx(arm, gripper, wm)
    skill = PickTopDownSkill()
    result = skill.execute({"object_label": "橙色"}, ctx)
    assert result.success is False
    assert result.result_data["diagnosis"] == "object_not_found"
    # error_message should contain the known labels inline
    assert "blue bottle" in result.error_message
    assert "red can" in result.error_message
    # result_data keeps the structured copy too
    assert "blue bottle" in result.result_data["known_objects"]


# ---------------------------------------------------------------------------
# v2.3 hot-fix: PickTopDown auto-detect retry (symmetric with MobilePick).
# Gaps LLM routing where "抓 X" is sent to pick_top_down instead of mobile_pick.
# ---------------------------------------------------------------------------


def test_pick_top_down_auto_detect_on_miss_then_hit():
    """world_model empty + perception+calibration wired → DetectSkill is invoked,
    world_model gets populated, retry resolves, and the skill proceeds."""
    from unittest.mock import MagicMock, patch

    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()  # empty
    # Extend SkillContext kwargs to include perception+calibration
    ctx = SkillContext(
        arm=arm, gripper=gripper, base=SimpleNamespace(),
        world_model=wm, config={},
        perception=MagicMock(),
        calibration=MagicMock(),
    )

    def fake_detect_execute(params, c):
        c.world_model.add_object(ObjectState(
            object_id="pickable_bottle_blue", label="blue bottle",
            x=11.0, y=3.0, z=0.25, confidence=0.9,
        ))
        from vector_os_nano.core.types import SkillResult as _SR
        return _SR(success=True, result_data={"count": 1, "diagnosis": "ok"})

    skill = PickTopDownSkill()
    with patch(
        "vector_os_nano.skills.detect.DetectSkill",
        return_value=MagicMock(execute=fake_detect_execute),
    ):
        result = skill.execute({"object_label": "blue bottle"}, ctx)

    # skill should proceed past resolve (i.e. not object_not_found).
    assert result.result_data.get("diagnosis") != "object_not_found"
    # one object persisted to world model
    assert len(wm.get_objects()) == 1


def test_pick_top_down_no_retry_when_perception_none():
    """Backward-compat: if perception is None, no auto-detect; keep existing UX."""
    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()
    ctx = _mkctx(arm, gripper, wm)  # perception not set → None
    skill = PickTopDownSkill()
    result = skill.execute({"object_label": "blue bottle"}, ctx)
    assert result.success is False
    assert result.result_data["diagnosis"] == "object_not_found"


def test_pick_top_down_no_retry_when_calibration_none():
    """perception wired but calibration None → guard prevents retry."""
    from unittest.mock import MagicMock

    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()
    ctx = SkillContext(
        arm=arm, gripper=gripper, base=SimpleNamespace(),
        world_model=wm, config={},
        perception=MagicMock(),
        calibration=None,
    )
    skill = PickTopDownSkill()
    result = skill.execute({"object_label": "blue bottle"}, ctx)
    assert result.success is False
    assert result.result_data["diagnosis"] == "object_not_found"


def test_pick_top_down_vlm_returns_empty_then_object_not_found():
    """DetectSkill returns count=0 → _resolve_target still None → not_found."""
    from unittest.mock import MagicMock, patch
    from vector_os_nano.core.types import SkillResult as _SR

    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()
    ctx = SkillContext(
        arm=arm, gripper=gripper, base=SimpleNamespace(),
        world_model=wm, config={},
        perception=MagicMock(),
        calibration=MagicMock(),
    )

    empty_detect = MagicMock(execute=MagicMock(
        return_value=_SR(success=True, result_data={"count": 0, "diagnosis": "no_detections"}),
    ))
    skill = PickTopDownSkill()
    with patch(
        "vector_os_nano.skills.detect.DetectSkill",
        return_value=empty_detect,
    ):
        result = skill.execute({"object_label": "blue bottle"}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "object_not_found"


def test_pick_top_down_detect_crash_does_not_crash_skill():
    """DetectSkill.execute raises → helper catches → skill reports object_not_found."""
    from unittest.mock import MagicMock, patch

    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()
    ctx = SkillContext(
        arm=arm, gripper=gripper, base=SimpleNamespace(),
        world_model=wm, config={},
        perception=MagicMock(),
        calibration=MagicMock(),
    )

    def boom(params, c):
        raise RuntimeError("VLM network exploded")

    skill = PickTopDownSkill()
    with patch(
        "vector_os_nano.skills.detect.DetectSkill",
        return_value=MagicMock(execute=boom),
    ):
        result = skill.execute({"object_label": "blue bottle"}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "object_not_found"
