# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Home never synthesizes another arm's joint vector."""
import pytest
from tests.unit.skills.test_grasp import context
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.vcli.cognitive import abort


@pytest.mark.parametrize('body', ['so101', 'piper', 'go2_piper'])
def test_home_uses_the_selected_arm_controller(body):
    abort.clear_abort()
    ctx = context(body)
    assert HomeSkill().execute({}, ctx).success
    ctx.arm.home.assert_called_once_with()
    ctx.arm.move_joints.assert_not_called()
