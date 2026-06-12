# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 3 — cross-layer contract single-sourcing (design review #7/#12/#13/#14).

Same value, same meaning, at every layer: degrees are CONVERTED at the
selector seam (never renamed rad without converting); unknown primitive
params fail loud instead of silently dropping; enum/default reach the LLM's
schema; motor-ness is an explicit Skill declaration (E-stop never gated
behind a confirmation prompt); lateral motion on a non-holonomic base fails
fast with the capability named.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import SubGoal


def _sg(name, strategy="", params=None, description=""):
    return SubGoal(name=name, description=description or name,
                   strategy=strategy, strategy_params=params or {},
                   verify="True")


class TestDegreesConvertedAtSeam:
    def test_explicit_turn_angle_degrees_to_radians(self):
        sel = StrategySelector(primitives_executable=lambda: True)
        r = sel.select(_sg("turn_q", strategy="turn", params={"angle": -90}))
        assert r.executor_type == "primitive"
        assert math.isclose(r.params["angle_rad"], math.radians(-90))

    def test_keyword_ladder_turn_converts(self):
        sel = StrategySelector(primitives_executable=lambda: True)
        r = sel.select(_sg("turn_left", description="turn left",
                           params={"angle": 45}))
        assert r.executor_type == "primitive"
        assert math.isclose(r.params["angle_rad"], math.radians(45))

    def test_already_rad_param_passes_through(self):
        sel = StrategySelector(primitives_executable=lambda: True)
        r = sel.select(_sg("turn_q", strategy="turn",
                           params={"angle_rad": 1.57}))
        assert math.isclose(r.params["angle_rad"], 1.57)


class TestPrimitiveUnknownParamsFailLoud:
    def test_unknown_param_is_error_with_accepted_set(self):
        from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor

        ex = GoalExecutor(strategy_selector=None, verifier=None,
                          primitives={"noop": lambda known=1: True})
        ok, err, _ = ex._execute_primitive("noop", {"unknwon": 2})
        assert ok is False
        assert "known" in err  # the accepted params, fed to the replan


class TestWrapperSchemaEnumDefault:
    def test_enum_and_default_passed_through(self):
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        schema = SkillWrapperTool._build_schema({
            "direction": {"type": "string", "enum": ["forward", "backward"],
                          "default": "forward"},
        })
        prop = schema["properties"]["direction"]
        assert prop["enum"] == ["forward", "backward"]
        assert prop["default"] == "forward"


class TestExplicitIsMotor:
    def test_explicit_declaration_wins_over_sniffing(self):
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        class _Quiet:  # description full of motor words, declared NOT motor
            name = "observe"
            description = "move the arm gripper base motor joint"
            parameters: dict = {}
            is_motor = False

            def execute(self, p, c):
                return None

        assert SkillWrapperTool._detect_motor(_Quiet()) is False

    def test_go2_motion_skills_declare_motor(self):
        from vector_os_nano.skills.go2.stop import StopSkill
        from vector_os_nano.skills.go2.walk import WalkSkill

        assert WalkSkill.is_motor is True
        assert StopSkill.is_motor is True

    def test_stop_is_confirm_exempt(self):
        # E-stop must never wait behind a confirmation prompt (safety rule).
        from vector_os_nano.skills.go2.stop import StopSkill

        assert StopSkill.confirm_exempt is True


class TestLateralCapabilityCheck:
    def test_lateral_on_non_holonomic_fails_fast(self):
        from vector_os_nano.skills.go2.walk import WalkSkill

        class _Base:
            supports_holonomic = False

            def walk(self, vx, vy, vyaw, duration):  # pragma: no cover
                raise AssertionError("must fail BEFORE commanding")

        res = WalkSkill().execute(
            {"direction": "left", "distance": 1.0},
            SimpleNamespace(base=_Base(), world_model=None, agent=None))
        assert res.success is False
        assert res.result_data.get("diagnosis") == "lateral_unsupported"

    def test_habitat_base_declares_non_holonomic(self):
        from vector_os_nano.playground.habitat.base import HabitatBase

        base = HabitatBase.__new__(HabitatBase)  # property needs no __init__
        assert base.supports_holonomic is False


class TestConfirmExemptGate:
    def test_stop_never_asks_even_on_real_hardware(self):
        from vector_os_nano.skills.go2.stop import StopSkill
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        w = SkillWrapperTool(StopSkill(), agent=None)
        w._robot_is_simulated = lambda: False  # pretend REAL hardware
        res = w.check_permissions({}, None)
        assert res.behavior == "allow"


class TestDirectionEnumValidation:
    """#13 second half: skills validate enum values — an unknown direction
    must fail loud with the legal set, never silently walk forward (the GUI
    test caught '左' coercing to forward and burning 3.3s on a wrong walk)."""

    def _ctx(self):
        class _Base:
            supports_holonomic = True

            def walk(self, vx, vy, vyaw, duration):
                return True

            def get_position(self):
                return [0.0, 0.0, 0.0]

        return SimpleNamespace(base=_Base(), world_model=None, agent=None)

    def test_unknown_direction_is_bad_params(self):
        from vector_os_nano.skills.go2.walk import WalkSkill

        res = WalkSkill().execute({"direction": "sideways", "distance": 1.0},
                                  self._ctx())
        assert res.success is False
        assert "forward" in (res.error_message or "")  # legal set named

    def test_chinese_aliases_resolve(self):
        from vector_os_nano.skills.go2.walk import _DIRECTION_MAP

        for zh in ("前", "后", "左", "右"):
            assert zh in _DIRECTION_MAP

    def test_turn_unknown_direction_bad_params(self):
        from vector_os_nano.skills.go2.turn import TurnSkill

        res = TurnSkill().execute({"direction": "around", "angle": 90},
                                  self._ctx())
        assert res.success is False
