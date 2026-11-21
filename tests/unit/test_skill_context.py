# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for redesigned SkillContext with dict registries."""
import pytest
from unittest.mock import MagicMock
from vector_os_nano.core.skill import SkillContext



class TestSkillContextRegistries:
    def test_empty_context(self):
        ctx = SkillContext()
        assert ctx.arm is None
        assert ctx.gripper is None
        assert ctx.base is None
        assert ctx.perception is None

    def test_single_arm(self):
        arm = MagicMock()
        ctx = SkillContext(arms={"so101": arm})
        assert ctx.arm is arm
        assert ctx.has_arm()
        assert ctx.has_arm("so101")
        assert not ctx.has_arm("other")

    def test_single_base(self):
        base = MagicMock()
        ctx = SkillContext(bases={"go2": base})
        assert ctx.base is base
        assert ctx.has_base()
        assert ctx.has_base("go2")

    def test_multiple_arms(self):
        arm1 = MagicMock()
        arm2 = MagicMock()
        ctx = SkillContext(arms={"so101": arm1, "ur5": arm2})
        assert ctx.arm is arm1  # first one
        assert ctx.has_arm("ur5")
        assert ctx.get_arm("ur5") is arm2

    def test_no_base(self):
        ctx = SkillContext()
        assert not ctx.has_base()
        assert ctx.base is None
        assert ctx.get_base("go2") is None

    def test_capabilities(self):
        arm = MagicMock()
        base = MagicMock()
        ctx = SkillContext(
            arms={"so101": arm},
            bases={"go2": base},
        )
        caps = ctx.capabilities()
        assert caps["has_arm"] is True
        assert caps["has_base"] is True
        assert caps["has_gripper"] is False
        assert "so101" in caps["arm_names"]
        assert "go2" in caps["base_names"]

    def test_services_registry(self):
        nav = MagicMock()
        ctx = SkillContext(services={"nav": nav})
        assert ctx.services["nav"] is nav

    def test_perception_sources(self):
        cam = MagicMock()
        ctx = SkillContext(perception_sources={"realsense": cam})
        assert ctx.perception is cam
        assert ctx.has_perception()

    def test_get_arm_default(self):
        arm = MagicMock()
        ctx = SkillContext(arms={"a": arm})
        assert ctx.get_arm() is arm
        assert ctx.get_arm("a") is arm
        assert ctx.get_arm("nonexistent") is None

    def test_get_base_default(self):
        base = MagicMock()
        ctx = SkillContext(bases={"b": base})
        assert ctx.get_base() is base

    def test_backward_compat_config(self):
        ctx = SkillContext(config={"key": "val"})
        assert ctx.config["key"] == "val"

    def test_calibration_field(self):
        cal = MagicMock()
        ctx = SkillContext(calibration=cal)
        assert ctx.calibration is cal


class TestSkillContextBackwardCompat:
    """Ensure existing skill code that uses context.arm / context.base still works."""

    def test_walk_skill_pattern(self):
        """WalkSkill does: context.base.walk(vx, vy, vyaw, dur)"""
        base = MagicMock()
        base.walk.return_value = True
        ctx = SkillContext(bases={"go2": base})
        # Simulates what WalkSkill does
        assert ctx.base is not None
        result = ctx.base.walk(0.3, 0, 0, 2.0)
        assert result is True

    def test_pick_skill_pattern(self):
        """PickSkill does: context.arm, context.gripper, context.perception, context.calibration"""
        arm = MagicMock()
        gripper = MagicMock()
        perception = MagicMock()
        cal = MagicMock()
        ctx = SkillContext(
            arms={"so101": arm},
            grippers={"so101": gripper},
            perception_sources={"realsense": perception},
            calibration=cal,
        )
        assert ctx.arm is arm
        assert ctx.gripper is gripper
        assert ctx.perception is perception
        assert ctx.calibration is cal

    def test_legacy_flat_kwargs_arm(self):
        """Old-style arm= kwarg populates arm property via legacy fallback."""
        arm = MagicMock()
        ctx = SkillContext(arm=arm)
        assert ctx.arm is arm

    def test_legacy_flat_kwargs_base(self):
        """Old-style base= kwarg populates base property via legacy fallback."""
        base = MagicMock()
        ctx = SkillContext(base=base)
        assert ctx.base is base

    def test_legacy_flat_kwargs_gripper(self):
        """Old-style gripper= kwarg populates gripper property via legacy fallback."""
        gripper = MagicMock()
        ctx = SkillContext(gripper=gripper)
        assert ctx.gripper is gripper

    def test_legacy_flat_kwargs_perception(self):
        """Old-style perception= kwarg populates perception property via legacy fallback."""
        perception = MagicMock()
        ctx = SkillContext(perception=perception)
        assert ctx.perception is perception

    def test_dict_registry_takes_priority_over_legacy(self):
        """Dict registry arm takes priority over legacy arm= kwarg."""
        legacy_arm = MagicMock()
        dict_arm = MagicMock()
        ctx = SkillContext(
            arm=legacy_arm,
            arms={"so101": dict_arm},
        )
        assert ctx.arm is dict_arm

    def test_legacy_none_perception(self):
        """Legacy perception=None yields perception property None."""
        ctx = SkillContext(
            arm=MagicMock(),
            gripper=MagicMock(),
            perception=None,
            calibration=None,
        )
        assert ctx.perception is None

    def test_has_arm_with_legacy(self):
        """has_arm() returns True when arm is provided via legacy kwarg."""
        arm = MagicMock()
        ctx = SkillContext(arm=arm)
        assert ctx.has_arm()

    def test_has_base_with_legacy(self):
        """has_base() returns True when base is provided via legacy kwarg."""
        base = MagicMock()
        ctx = SkillContext(base=base)
        assert ctx.has_base()

    def test_has_gripper_with_legacy(self):
        """has_gripper() returns True when gripper is provided via legacy kwarg."""
        gripper = MagicMock()
        ctx = SkillContext(gripper=gripper)
        assert ctx.has_gripper()
