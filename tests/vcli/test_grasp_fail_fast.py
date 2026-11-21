# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""An unresolved grasp is refused before moving or looking in the real session."""
import pytest
from unittest.mock import patch
from tests.harness.sim_session import started

pytestmark = pytest.mark.sim

@pytest.mark.parametrize("body", ["piper", "so101", "go2_piper"])
def test_absent_target_is_refused_before_any_motion(body):
    with started(body) as s:
        agent = s.agent
        with patch.object(agent._arm, "move_joints") as move, \
             patch.object(agent._gripper, "open") as opened, \
             patch.object(agent._perception, "capture") as capture:
            result = agent.execute_skill("grasp", {"query": "absent purple kettle"})
        assert not result.success
        assert result.diagnosis_code == "unknown_object"
        move.assert_not_called()
        opened.assert_not_called()
        capture.assert_not_called()
