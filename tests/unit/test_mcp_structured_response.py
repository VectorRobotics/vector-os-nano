"""T8: MCP structured JSON response tests.

Verifies that _format_execution_result() returns structured JSON
for ExecutionResult inputs, with per-step diagnostics, world state,
and correct backward-compat string passthrough.
"""
from __future__ import annotations

import json

from vector_os_nano.core.types import ExecutionResult, StepTrace


def test_format_execution_result_returns_json():
    """ExecutionResult input produces valid JSON string."""
    result = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=1,
        steps_total=1,
        trace=[
            StepTrace(
                step_id="s1",
                skill_name="home",
                status="success",
                duration_sec=3.0,
                result_data={"diagnosis": "ok"},
            )
        ],
    )
    from vector_os_nano.mcp.tools import _format_execution_result

    output = _format_execution_result("home", result)
    data = json.loads(output)  # must not raise
    assert data["success"] is True


def test_format_result_json_has_steps_with_result_data():
    """Parsed JSON has steps list with result_data."""
    result = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=1,
        steps_total=1,
        trace=[
            StepTrace(
                step_id="s1",
                skill_name="scan",
                status="success",
                duration_sec=2.0,
                result_data={"diagnosis": "ok", "joint_values": [0.1]},
            )
        ],
    )
    from vector_os_nano.mcp.tools import _format_execution_result

    data = json.loads(_format_execution_result("scan", result))
    assert len(data["steps"]) == 1
    assert data["steps"][0]["result_data"]["diagnosis"] == "ok"


def test_format_result_json_has_world_state():
    """When world_state is provided, JSON includes it."""
    result = ExecutionResult(success=True, status="completed")
    from vector_os_nano.mcp.tools import _format_execution_result

    ws = {"objects": [{"label": "mug"}], "robot": {"gripper_state": "open"}}
    data = json.loads(_format_execution_result("home", result, world_state=ws))
    assert "world_state" in data
    assert data["world_state"]["objects"][0]["label"] == "mug"


def test_format_result_string_passthrough():
    """String input returns unchanged (backward compat)."""
    from vector_os_nano.mcp.tools import _format_execution_result

    assert _format_execution_result("test", "hello world") == "hello world"


def test_format_result_json_has_failure_reason():
    """Failed execution JSON has failure_reason."""
    result = ExecutionResult(
        success=False,
        status="failed",
        failure_reason="IK failed",
        trace=[
            StepTrace(
                step_id="s1",
                skill_name="pick",
                status="execution_failed",
                duration_sec=1.0,
                error="IK failed",
                result_data={"diagnosis": "ik_unreachable"},
            )
        ],
    )
    from vector_os_nano.mcp.tools import _format_execution_result

    data = json.loads(_format_execution_result("pick banana", result))
    assert data["success"] is False
    assert data["failure_reason"] == "IK failed"
    assert data["steps"][0]["result_data"]["diagnosis"] == "ik_unreachable"


def test_format_result_json_success_case():
    """Successful multi-step execution has complete trace."""
    result = ExecutionResult(
        success=True,
        status="completed",
        steps_completed=3,
        steps_total=3,
        trace=[
            StepTrace(
                step_id="s1",
                skill_name="scan",
                status="success",
                duration_sec=3.0,
                result_data={"diagnosis": "ok"},
            ),
            StepTrace(
                step_id="s2",
                skill_name="detect",
                status="success",
                duration_sec=1.0,
                result_data={"diagnosis": "ok", "count": 2},
            ),
            StepTrace(
                step_id="s3",
                skill_name="pick",
                status="success",
                duration_sec=5.0,
                result_data={"diagnosis": "ok", "position_cm": [22, 5]},
            ),
        ],
    )
    from vector_os_nano.mcp.tools import _format_execution_result

    data = json.loads(_format_execution_result("pick banana", result))
    assert data["steps_completed"] == 3
    assert len(data["steps"]) == 3
    assert data["total_duration_sec"] == 9.0
