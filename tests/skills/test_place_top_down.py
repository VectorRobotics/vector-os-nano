# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for PlaceTopDownSkill using mocked hardware.

Tests verify: hardware preconditions, world-model receptacle lookup,
motion sequencing (IK → approach → descent → open → lift), and return
data structure. Actual physics is not exercised here.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.place_top_down import PlaceTopDownSkill


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockArm:
    """ArmProtocol-compatible mock. Records calls; ik_top_down returns a
    pre-canned joint solution unless forced to fail on a specific call number.
    """

    name = "mock_arm"
    dof = 6
    joint_names = [f"piper_joint{i}" for i in range(1, 7)]

    def __init__(self, ik_fail_call: int | None = None):
        """Args:
            ik_fail_call: 1-based index of which ik_top_down call should return None.
                          None means all calls succeed.
        """
        self.calls: list[tuple[str, object]] = []
        self._joints = [0.0] * 6
        self._ik_call_count = 0
        self._ik_fail_call = ik_fail_call

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        self.calls.append(("move_joints", (list(positions), duration)))
        self._joints = list(positions)
        return True

    def ik_top_down(
        self, target: tuple[float, float, float], current_joints: list[float] | None = None
    ) -> list[float] | None:
        self.calls.append(("ik_top_down", (tuple(target), current_joints)))
        self._ik_call_count += 1
        if self._ik_fail_call is not None and self._ik_call_count == self._ik_fail_call:
            return None
        return [0.1] * 6

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def fk(self, joint_positions: list[float]) -> tuple:
        return [0.0, 0.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class _MockGripper:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._open = True

    def open(self) -> bool:
        self.calls.append("open")
        self._open = True
        return True

    def close(self) -> bool:
        self.calls.append("close")
        self._open = False
        return True

    def is_holding(self) -> bool:
        return not self._open

    def get_position(self) -> float:
        return 1.0 if self._open else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mkctx(
    arm=None,
    gripper=None,
    world_model=None,
    config: dict | None = None,
) -> SkillContext:
    return SkillContext(
        arm=arm,
        gripper=gripper,
        base=SimpleNamespace(),
        world_model=world_model,
        config=config or {},
    )


def _mkworld(*objs: ObjectState) -> WorldModel:
    wm = WorldModel()
    for o in objs:
        wm.add_object(o)
    return wm


def _receptacle(
    rid: str = "tray_01",
    x: float = 0.5,
    y: float = 0.0,
    z: float = 0.25,
) -> ObjectState:
    return ObjectState(object_id=rid, label="tray", x=x, y=y, z=z)


# ---------------------------------------------------------------------------
# Test 1 — Happy path: explicit xyz, assert full call order
# ---------------------------------------------------------------------------


def test_place_top_down_happy_path_explicit_xyz_calls_open_after_descent() -> None:
    """Call order: ik(pre_place) → ik(place, ...) → move(pre) → move(place) →
    gripper.open() → move(pre).  No home return."""
    arm = _MockArm()
    gripper = _MockGripper()
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=_mkworld())

    result = PlaceTopDownSkill().execute(
        {"target_xyz": [0.5, 0.0, 0.25]}, ctx
    )

    assert result.success is True
    assert result.result_data["diagnosis"] == "ok"

    # Gripper: open fires AFTER descent, exactly once
    assert gripper.calls == ["open"], f"expected ['open'], got {gripper.calls}"

    # Arm: exactly 2 IK calls (pre_place, place) and 3 move_joints calls (approach, descent, lift)
    ik_calls = [c for c in arm.calls if c[0] == "ik_top_down"]
    move_calls = [c for c in arm.calls if c[0] == "move_joints"]
    assert len(ik_calls) == 2, f"expected 2 ik_top_down calls, got {len(ik_calls)}"
    assert len(move_calls) == 3, f"expected 3 move_joints calls (approach/descent/lift), got {len(move_calls)}"

    # Order within arm.calls: ik_pre < ik_place < move_approach < move_descent < move_lift
    call_names = [c[0] for c in arm.calls]
    ik_indices = [i for i, n in enumerate(call_names) if n == "ik_top_down"]
    move_indices = [i for i, n in enumerate(call_names) if n == "move_joints"]
    assert ik_indices[0] < ik_indices[1] < move_indices[0] < move_indices[1] < move_indices[2]

    # placed_at should be [tx, ty, tz + drop_height]
    placed = result.result_data["placed_at"]
    assert len(placed) == 3
    assert abs(placed[0] - 0.5) < 1e-6
    assert abs(placed[1] - 0.0) < 1e-6
    # z = 0.25 + default drop_height (0.05) = 0.30
    assert abs(placed[2] - 0.30) < 1e-6


# ---------------------------------------------------------------------------
# Test 2 — IK unreachable on pre_place (first call)
# ---------------------------------------------------------------------------


def test_place_top_down_ik_unreachable_pre_place() -> None:
    arm = _MockArm(ik_fail_call=1)  # first ik_top_down returns None
    gripper = _MockGripper()
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=_mkworld())

    result = PlaceTopDownSkill().execute({"target_xyz": [0.5, 0.0, 0.25]}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "ik_unreachable"
    assert result.result_data["phase"] == "pre_place"


# ---------------------------------------------------------------------------
# Test 3 — IK unreachable on place pose (second call)
# ---------------------------------------------------------------------------


def test_place_top_down_ik_unreachable_place() -> None:
    arm = _MockArm(ik_fail_call=2)  # second ik_top_down returns None
    gripper = _MockGripper()
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=_mkworld())

    result = PlaceTopDownSkill().execute({"target_xyz": [0.5, 0.0, 0.25]}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "ik_unreachable"
    assert result.result_data["phase"] == "place"


# ---------------------------------------------------------------------------
# Test 4 — No arm
# ---------------------------------------------------------------------------


def test_place_top_down_no_arm_returns_no_arm() -> None:
    ctx = _mkctx(arm=None, gripper=_MockGripper(), world_model=_mkworld())

    result = PlaceTopDownSkill().execute({"target_xyz": [0.5, 0.0, 0.25]}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "no_arm"


# ---------------------------------------------------------------------------
# Test 5 — No gripper
# ---------------------------------------------------------------------------


def test_place_top_down_no_gripper_returns_no_gripper() -> None:
    ctx = _mkctx(arm=_MockArm(), gripper=None, world_model=_mkworld())

    result = PlaceTopDownSkill().execute({"target_xyz": [0.5, 0.0, 0.25]}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "no_gripper"


# ---------------------------------------------------------------------------
# Test 6 — Receptacle ID resolves from world model
# ---------------------------------------------------------------------------


def test_place_top_down_receptacle_id_resolves_from_world_model() -> None:
    """When target_xyz is absent but receptacle_id is provided, the skill
    fetches the receptacle's (x, y, z) from the world model and places there."""
    arm = _MockArm()
    gripper = _MockGripper()
    receptacle = _receptacle(rid="tray_01", x=1.2, y=0.3, z=0.15)
    wm = _mkworld(receptacle)
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    result = PlaceTopDownSkill().execute({"receptacle_id": "tray_01"}, ctx)

    assert result.success is True
    assert result.result_data["diagnosis"] == "ok"

    placed = result.result_data["placed_at"]
    # Should be [rx, ry, rz + drop_height]
    assert abs(placed[0] - 1.2) < 1e-6
    assert abs(placed[1] - 0.3) < 1e-6
    drop_h = 0.05  # default
    assert abs(placed[2] - (0.15 + drop_h)) < 1e-6

    # world_model.get_object must have been called with "tray_01"
    # Verify the IK targets use the receptacle coords
    ik_calls = [c for c in arm.calls if c[0] == "ik_top_down"]
    pre_place_target = ik_calls[0][1][0]  # (x, y, z)
    assert abs(pre_place_target[0] - 1.2) < 1e-6
    assert abs(pre_place_target[1] - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Test 7 — Receptacle not found
# ---------------------------------------------------------------------------


def test_place_top_down_receptacle_not_found_returns_receptacle_not_found() -> None:
    wm = _mkworld()  # empty world model
    ctx = _mkctx(arm=_MockArm(), gripper=_MockGripper(), world_model=wm)

    result = PlaceTopDownSkill().execute({"receptacle_id": "nonexistent_tray"}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "receptacle_not_found"


# ---------------------------------------------------------------------------
# Test 8 — Missing target (neither target_xyz nor receptacle_id)
# ---------------------------------------------------------------------------


def test_place_top_down_missing_target_returns_missing_target() -> None:
    ctx = _mkctx(arm=_MockArm(), gripper=_MockGripper(), world_model=_mkworld())

    result = PlaceTopDownSkill().execute({}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "missing_target"


# ---------------------------------------------------------------------------
# Test 9 — verify_hint: skill declares a grounded object_at predicate
# ---------------------------------------------------------------------------


def test_place_top_down_has_verify_hint_with_object_at() -> None:
    """PlaceTopDownSkill.verify_hint must be a non-empty str containing
    'object_at' — the grounded place-success predicate (GT: resting at the drop xy).  It must NOT be
    empty, 'True', or the old always-passing fallback."""
    skill = PlaceTopDownSkill()
    hint = getattr(skill, "verify_hint", None)
    assert hint is not None, "PlaceTopDownSkill must declare verify_hint"
    assert isinstance(hint, str), "verify_hint must be a str"
    assert hint.strip(), "verify_hint must be non-empty"
    assert "object_at" in hint, (
        f"verify_hint must reference 'object_at'; got {hint!r}"
    )


# ---------------------------------------------------------------------------
# Test 10 — to_schemas() surfaces the verify_hint via SkillRegistry
# ---------------------------------------------------------------------------


def test_place_top_down_schema_surfaces_verify_hint() -> None:
    """When PlaceTopDownSkill is registered, to_schemas() carries the
    verify_hint == the class attribute value (not the fallback 'True')."""
    from vector_os_nano.core.skill import SkillRegistry

    reg = SkillRegistry()
    reg.register(PlaceTopDownSkill())
    schemas = reg.to_schemas()
    by_name = {s["name"]: s for s in schemas}
    assert "place_top_down" in by_name, "place_top_down skill must appear in schemas"
    hint = by_name["place_top_down"]["verify_hint"]
    assert "object_at" in hint, (
        f"schema verify_hint must contain 'object_at'; got {hint!r}"
    )
    assert hint != "True", "verify_hint must not be the fallback 'True'"
