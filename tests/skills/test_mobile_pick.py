"""Unit tests for MobilePickSkill — TDD Wave 3 T8.

8 primary tests covering guard clauses, navigation decisions, ordering, and
composition with PickTopDownSkill.  Additional helper-unit tests cover
_wait_stable and _ang_diff to reach >=80% coverage.
All hardware is mocked; time.sleep is patched to a no-op so _wait_stable
does not block in the primary tests.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch


from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.mobile_pick import (
    MobilePickSkill,
    _ang_diff,
    _dist_xy,
    _wait_stable,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_APPROACH_POSE = (1.0, 2.0, 0.5)   # (ax, ay, ayaw) returned by compute_approach_pose
_OBJ_XYZ = (1.55, 2.0, 0.25)       # world position of the target object
_OBJ_ID = "pickable_bottle_blue"


def _make_base(pos=(0.0, 0.0, 0.28), heading=0.0):
    """Build a mock base that returns configurable position/heading."""
    base = MagicMock()
    base.get_position.return_value = pos
    base.get_heading.return_value = heading
    base.navigate_to.return_value = True
    return base


def _make_world_model(obj_id=_OBJ_ID, obj_xyz=_OBJ_XYZ):
    """Build a mock world_model with one pickable object."""
    from vector_os_nano.core.world_model import ObjectState, WorldModel

    wm = WorldModel()
    obj = ObjectState(
        object_id=obj_id,
        label="blue bottle",
        x=obj_xyz[0],
        y=obj_xyz[1],
        z=obj_xyz[2],
        confidence=1.0,
    )
    wm.add_object(obj)
    return wm


def _make_context(base=None, arm=None, gripper=None, world_model=None, config=None):
    """Build a SkillContext with the given (possibly None) components."""
    return SkillContext(
        base=base,
        arm=arm,
        gripper=gripper,
        world_model=world_model,
        config=config or {},
    )


def _make_skill_with_mock_pick(
    pick_resolve_return=(_OBJ_ID, _OBJ_XYZ),
    pick_execute_return=None,
):
    """Instantiate MobilePickSkill and replace its _pick internals with mocks.

    Returns (skill, mock_pick) where mock_pick has:
      - _resolve_target.return_value
      - execute.return_value
    """
    if pick_execute_return is None:
        pick_execute_return = SkillResult(
            success=True,
            result_data={"diagnosis": "ok", "object_id": _OBJ_ID, "grasp_world": list(_OBJ_XYZ)},
        )

    skill = MobilePickSkill()

    mock_pick = MagicMock()
    mock_pick._resolve_target.return_value = pick_resolve_return
    mock_pick.execute.return_value = pick_execute_return
    skill._pick = mock_pick

    return skill, mock_pick


# Monotonic counter for _wait_stable fast termination
class _MonotonicCounter:
    """Advances by `step` each call; call count drives the stable loop."""

    def __init__(self, step: float = 0.3):
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


# ---------------------------------------------------------------------------
# Test 1 — already reachable: navigate_to NOT called
# ---------------------------------------------------------------------------


def test_mobile_pick_already_reachable_skips_navigate():
    """Dog is AT the approach pose — navigate_to must not be called."""
    ax, ay, ayaw = _APPROACH_POSE

    # Place dog exactly at approach pose
    base = _make_base(pos=(ax, ay, 0.28), heading=ayaw)
    wm = _make_world_model()
    skill, mock_pick = _make_skill_with_mock_pick()

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", return_value=True),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=wm
        )
        result = skill.execute({}, ctx)

    assert result.success
    base.navigate_to.assert_not_called()
    mock_pick.execute.assert_called_once()
    assert result.result_data["mobile_pick"]["skipped_navigate"] is True


# ---------------------------------------------------------------------------
# Test 2 — navigate → wait_stable → pick in order
# ---------------------------------------------------------------------------


def test_mobile_pick_calls_navigate_then_wait_then_pick_in_order():
    """Dog is FAR from approach pose → navigate → wait_stable → pick, in order."""
    # Dog starts far from approach
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=3.14)
    wm = _make_world_model()

    call_order: list[str] = []

    def nav_side_effect(*args, **kwargs):
        call_order.append("navigate_to")
        return True

    base.navigate_to.side_effect = nav_side_effect

    skill, mock_pick = _make_skill_with_mock_pick()

    def pick_side_effect(params, ctx):
        call_order.append("pick.execute")
        return SkillResult(success=True, result_data={"diagnosis": "ok", "object_id": _OBJ_ID, "grasp_world": list(_OBJ_XYZ)})

    mock_pick.execute.side_effect = pick_side_effect

    def wait_side_effect(base, **kwargs):
        call_order.append("wait_stable")
        return True

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", side_effect=wait_side_effect),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=wm
        )
        result = skill.execute({}, ctx)

    assert result.success
    assert call_order == ["navigate_to", "wait_stable", "pick.execute"]


# ---------------------------------------------------------------------------
# Test 3 — navigation failure → diagnosis=nav_failed, pick not called
# ---------------------------------------------------------------------------


def test_mobile_pick_nav_failed_returns_nav_failed():
    """navigate_to returns False → skill returns nav_failed, pick not called."""
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    base.navigate_to.return_value = False
    wm = _make_world_model()

    skill, mock_pick = _make_skill_with_mock_pick()

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", return_value=True),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=wm
        )
        result = skill.execute({}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "nav_failed"
    mock_pick.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4 — wait_stable timeout → diagnosis=wait_stable_timeout, pick not called
# ---------------------------------------------------------------------------


def test_mobile_pick_wait_stable_timeout_returns_wait_stable_timeout():
    """_wait_stable returns False → diagnosis=wait_stable_timeout, pick not called."""
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    wm = _make_world_model()

    skill, mock_pick = _make_skill_with_mock_pick()

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", return_value=False),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=wm
        )
        result = skill.execute({}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "wait_stable_timeout"
    mock_pick.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5 — ik_unreachable propagated from pick
# ---------------------------------------------------------------------------


def test_mobile_pick_propagates_pick_ik_unreachable_failure():
    """Pick skill returning ik_unreachable is propagated unchanged by mobile_pick."""
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    wm = _make_world_model()

    pick_result = SkillResult(
        success=False,
        error_message="IK unreachable for pre-grasp",
        result_data={"diagnosis": "ik_unreachable", "phase": "pre_grasp"},
    )
    skill, mock_pick = _make_skill_with_mock_pick(pick_execute_return=pick_result)

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", return_value=True),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=wm
        )
        result = skill.execute({}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "ik_unreachable"
    assert result.error_message == "IK unreachable for pre-grasp"


# ---------------------------------------------------------------------------
# Test 6 — object_not_found when world model is empty
# ---------------------------------------------------------------------------


def test_mobile_pick_object_not_found_returns_object_not_found():
    """_resolve_target returns None (empty world) → object_not_found, nav not called."""
    from vector_os_nano.core.world_model import WorldModel

    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    empty_wm = WorldModel()

    skill, mock_pick = _make_skill_with_mock_pick()
    # Override _resolve_target to return None (simulates empty WM lookup)
    mock_pick._resolve_target.return_value = None

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=empty_wm
        )
        result = skill.execute({"object_label": "nonexistent"}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "object_not_found"
    base.navigate_to.assert_not_called()
    mock_pick.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7 — no base guard
# ---------------------------------------------------------------------------


def test_mobile_pick_no_base_returns_no_base():
    """context.base is None → immediate failure with diagnosis=no_base."""
    skill, mock_pick = _make_skill_with_mock_pick()

    ctx = _make_context(base=None, arm=MagicMock(), gripper=MagicMock(), world_model=MagicMock())
    result = skill.execute({}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "no_base"
    mock_pick.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8 — skip_navigate param honoured
# ---------------------------------------------------------------------------


def test_mobile_pick_skip_navigate_param_honoured():
    """skip_navigate=True skips navigate_to even if dog is far from approach."""
    # Dog is FAR away, but skip_navigate=True
    base = _make_base(pos=(99.0, 99.0, 0.28), heading=0.0)
    wm = _make_world_model()

    skill, mock_pick = _make_skill_with_mock_pick()

    with (
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", return_value=True),
        patch("time.sleep"),
    ):
        ctx = _make_context(
            base=base, arm=MagicMock(), gripper=MagicMock(), world_model=wm
        )
        result = skill.execute({"skip_navigate": True}, ctx)

    assert result.success
    base.navigate_to.assert_not_called()
    mock_pick.execute.assert_called_once()
    assert result.result_data["mobile_pick"]["skipped_navigate"] is True


# ---------------------------------------------------------------------------
# Extra guard tests — no_arm, no_gripper, no_world_model
# ---------------------------------------------------------------------------


def test_mobile_pick_no_arm_returns_no_arm():
    """context.arm is None -> immediate failure with diagnosis=no_arm."""
    skill, mock_pick = _make_skill_with_mock_pick()
    ctx = _make_context(base=MagicMock(), arm=None, gripper=MagicMock(), world_model=MagicMock())
    result = skill.execute({}, ctx)
    assert not result.success
    assert result.result_data["diagnosis"] == "no_arm"
    mock_pick.execute.assert_not_called()


def test_mobile_pick_no_gripper_returns_no_gripper():
    """context.gripper is None -> immediate failure with diagnosis=no_gripper."""
    skill, mock_pick = _make_skill_with_mock_pick()
    ctx = _make_context(base=MagicMock(), arm=MagicMock(), gripper=None, world_model=MagicMock())
    result = skill.execute({}, ctx)
    assert not result.success
    assert result.result_data["diagnosis"] == "no_gripper"
    mock_pick.execute.assert_not_called()


def test_mobile_pick_no_world_model_returns_no_world_model():
    """context.world_model is None -> immediate failure with diagnosis=no_world_model."""
    skill, mock_pick = _make_skill_with_mock_pick()
    ctx = _make_context(base=MagicMock(), arm=MagicMock(), gripper=MagicMock(), world_model=None)
    result = skill.execute({}, ctx)
    assert not result.success
    assert result.result_data["diagnosis"] == "no_world_model"
    mock_pick.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Helper unit tests — _ang_diff, _dist_xy, _wait_stable
# ---------------------------------------------------------------------------


def test_ang_diff_normal():
    assert abs(_ang_diff(0.1, 0.0) - 0.1) < 1e-9


def test_ang_diff_wrap_positive():
    """Difference exceeds +pi -> should wrap to small positive range."""
    # a - b = 0.01 - 2*pi  => after wrapping: 0.01
    d = _ang_diff(0.1, 0.1 + 2 * math.pi - 0.01)
    assert abs(d - 0.01) < 1e-6


def test_ang_diff_wrap_negative():
    """Difference below -pi -> should wrap to small negative range."""
    # a - b = -2*pi - 0.01 => after wrapping: -0.01
    d = _ang_diff(0.1, 0.1 + 2 * math.pi + 0.01)
    assert abs(d - (-0.01)) < 1e-6


def test_dist_xy_basic():
    assert abs(_dist_xy(0.0, 0.0, 3.0, 4.0) - 5.0) < 1e-9


def test_wait_stable_returns_true_when_dog_still():
    """_wait_stable returns True when position doesn't change (speed=0)."""
    base = MagicMock()
    base.get_position.return_value = (1.0, 1.0, 0.0)

    _t = [0.0]

    def fake_monotonic():
        _t[0] += 0.21  # slightly more than poll_dt=0.2
        return _t[0]

    with (
        patch("vector_os_nano.skills.mobile_pick.time.monotonic", side_effect=fake_monotonic),
        patch("vector_os_nano.skills.mobile_pick.time.sleep"),
    ):
        result = _wait_stable(base, max_speed=0.05, settle_duration=0.5, timeout=10.0)

    assert result is True


def test_wait_stable_returns_false_on_timeout():
    """_wait_stable returns False when dog keeps moving past timeout."""
    base = MagicMock()
    _calls = [0]

    def moving_pos():
        _calls[0] += 1
        return (float(_calls[0]) * 0.5, 0.0, 0.0)

    base.get_position.side_effect = moving_pos

    _t = [0.0]

    def fast_clock():
        _t[0] += 1.5  # advances quickly so deadline is hit fast
        return _t[0]

    with (
        patch("vector_os_nano.skills.mobile_pick.time.monotonic", side_effect=fast_clock),
        patch("vector_os_nano.skills.mobile_pick.time.sleep"),
    ):
        result = _wait_stable(base, max_speed=0.05, settle_duration=2.0, timeout=3.0)

    assert result is False


# ---------------------------------------------------------------------------
# Auto-detect retry tests — T5 Deliverable B
# ---------------------------------------------------------------------------


def _make_context_with_perception(
    base=None, arm=None, gripper=None, world_model=None, config=None,
    perception=None, calibration=None,
):
    """Build a SkillContext with perception + calibration support."""
    return SkillContext(
        base=base,
        arm=arm,
        gripper=gripper,
        world_model=world_model,
        config=config or {},
        perception=perception,
        calibration=calibration,
    )


def test_mobile_pick_auto_detect_on_miss_then_hit():
    """world_model miss → auto-detect fires → object added → skill proceeds past resolve.

    DetectSkill.execute is patched to return count=1 and add an object to world_model.
    The second call to _resolve_target returns the target (side_effect: None then hit).
    """
    from vector_os_nano.core.world_model import ObjectState, WorldModel

    empty_wm = WorldModel()
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    perception = MagicMock()
    calibration = MagicMock()

    skill_inst = MobilePickSkill()
    mock_pick = MagicMock()

    # First _resolve_target call → None; second → valid target (after DetectSkill adds object)
    target_hit = (_OBJ_ID, _OBJ_XYZ)
    mock_pick._resolve_target.side_effect = [None, target_hit]
    mock_pick.execute.return_value = SkillResult(
        success=True,
        result_data={"diagnosis": "ok", "object_id": _OBJ_ID, "grasp_world": list(_OBJ_XYZ)},
    )
    skill_inst._pick = mock_pick

    # DetectSkill.execute: add object to world_model and return count=1
    def fake_detect_execute(params, ctx):
        obj = ObjectState(
            object_id=_OBJ_ID,
            label="blue bottle",
            x=_OBJ_XYZ[0],
            y=_OBJ_XYZ[1],
            z=_OBJ_XYZ[2],
            confidence=0.91,
        )
        ctx.world_model.add_object(obj)
        return SkillResult(
            success=True,
            result_data={"count": 1, "objects": [{"object_id": _OBJ_ID}], "diagnosis": "ok"},
        )

    ctx = _make_context_with_perception(
        base=base,
        arm=MagicMock(),
        gripper=MagicMock(),
        world_model=empty_wm,
        perception=perception,
        calibration=calibration,
    )

    with (
        patch(
            "vector_os_nano.skills.detect.DetectSkill",
            return_value=MagicMock(execute=fake_detect_execute),
        ),
        patch("vector_os_nano.skills.mobile_pick.compute_approach_pose", return_value=_APPROACH_POSE),
        patch("vector_os_nano.skills.mobile_pick._wait_stable", return_value=True),
        patch("time.sleep"),
    ):
        result = skill_inst.execute({"object_label": "blue bottle"}, ctx)

    # Skill should proceed past resolve (not object_not_found)
    assert result.success
    assert mock_pick._resolve_target.call_count == 2
    mock_pick.execute.assert_called_once()
    # Object was added to world_model
    assert len(empty_wm.get_objects()) == 1


def test_mobile_pick_no_retry_when_perception_none():
    """context.perception is None → no auto-detect retry → object_not_found."""
    from vector_os_nano.core.world_model import WorldModel

    empty_wm = WorldModel()
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)

    skill_inst = MobilePickSkill()
    mock_pick = MagicMock()
    mock_pick._resolve_target.return_value = None
    skill_inst._pick = mock_pick

    ctx = _make_context_with_perception(
        base=base,
        arm=MagicMock(),
        gripper=MagicMock(),
        world_model=empty_wm,
        perception=None,
        calibration=MagicMock(),
    )

    with patch("time.sleep"):
        result = skill_inst.execute({"object_label": "blue bottle"}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "object_not_found"
    # _resolve_target called exactly once — no retry
    assert mock_pick._resolve_target.call_count == 1
    mock_pick.execute.assert_not_called()


def test_mobile_pick_no_retry_when_calibration_none():
    """context.calibration is None → no auto-detect retry → object_not_found."""
    from vector_os_nano.core.world_model import WorldModel

    empty_wm = WorldModel()
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)

    skill_inst = MobilePickSkill()
    mock_pick = MagicMock()
    mock_pick._resolve_target.return_value = None
    skill_inst._pick = mock_pick

    ctx = _make_context_with_perception(
        base=base,
        arm=MagicMock(),
        gripper=MagicMock(),
        world_model=empty_wm,
        perception=MagicMock(),
        calibration=None,
    )

    with patch("time.sleep"):
        result = skill_inst.execute({"object_label": "blue bottle"}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "object_not_found"
    assert mock_pick._resolve_target.call_count == 1
    mock_pick.execute.assert_not_called()


def test_mobile_pick_vlm_returns_empty_then_object_not_found():
    """DetectSkill returns count=0 → _resolve_target still None → object_not_found."""
    from vector_os_nano.core.world_model import WorldModel

    empty_wm = WorldModel()
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    perception = MagicMock()
    calibration = MagicMock()

    skill_inst = MobilePickSkill()
    mock_pick = MagicMock()
    # Both calls to _resolve_target return None (no object added)
    mock_pick._resolve_target.return_value = None
    skill_inst._pick = mock_pick

    fake_det_empty = MagicMock()
    fake_det_empty.execute.return_value = SkillResult(
        success=True,
        result_data={"count": 0, "objects": [], "diagnosis": "no_detections"},
    )

    ctx = _make_context_with_perception(
        base=base,
        arm=MagicMock(),
        gripper=MagicMock(),
        world_model=empty_wm,
        perception=perception,
        calibration=calibration,
    )

    with (
        patch("vector_os_nano.skills.detect.DetectSkill", return_value=fake_det_empty),
        patch("time.sleep"),
    ):
        result = skill_inst.execute({"object_label": "blue bottle"}, ctx)

    assert not result.success
    assert result.result_data["diagnosis"] == "object_not_found"
    fake_det_empty.execute.assert_called_once()
    mock_pick.execute.assert_not_called()


def test_mobile_pick_detect_crash_does_not_crash_skill():
    """DetectSkill.execute raises RuntimeError → caught → object_not_found (graceful)."""
    from vector_os_nano.core.world_model import WorldModel

    empty_wm = WorldModel()
    base = _make_base(pos=(5.0, 5.0, 0.28), heading=0.0)
    perception = MagicMock()
    calibration = MagicMock()

    skill_inst = MobilePickSkill()
    mock_pick = MagicMock()
    mock_pick._resolve_target.return_value = None
    skill_inst._pick = mock_pick

    crashing_det = MagicMock()
    crashing_det.execute.side_effect = RuntimeError("VLM service unavailable")

    ctx = _make_context_with_perception(
        base=base,
        arm=MagicMock(),
        gripper=MagicMock(),
        world_model=empty_wm,
        perception=perception,
        calibration=calibration,
    )

    with (
        patch("vector_os_nano.skills.detect.DetectSkill", return_value=crashing_det),
        patch("time.sleep"),
    ):
        result = skill_inst.execute({"object_label": "blue bottle"}, ctx)

    # Must not raise — skill returns graceful failure
    assert not result.success
    assert result.result_data["diagnosis"] == "object_not_found"
    mock_pick.execute.assert_not_called()
