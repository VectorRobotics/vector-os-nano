"""Unit tests for Gazebo backend registration in SimStartTool.

TDD — RED phase: these tests define the expected behaviour before
implementation.  All six tests should FAIL until sim_tool.py is updated.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.vcli.tools.sim_tool import SimStartTool


# ---------------------------------------------------------------------------
# 1. input_schema enum includes "gazebo"
# ---------------------------------------------------------------------------

def test_sim_tool_backend_enum_has_gazebo() -> None:
    """backend enum must list 'gazebo' as a valid value."""
    backend_prop = SimStartTool.input_schema["properties"]["backend"]
    assert "gazebo" in backend_prop["enum"], (
        f"'gazebo' not in backend enum: {backend_prop['enum']}"
    )


# ---------------------------------------------------------------------------
# 2. _start_gazebo_go2 method exists on the class
# ---------------------------------------------------------------------------

def test_start_gazebo_go2_method_exists() -> None:
    """SimStartTool must expose a _start_gazebo_go2 method."""
    assert hasattr(SimStartTool, "_start_gazebo_go2"), (
        "SimStartTool has no attribute '_start_gazebo_go2'"
    )


# ---------------------------------------------------------------------------
# 3. _start_gazebo_go2 must be a static method
# ---------------------------------------------------------------------------

def test_start_gazebo_go2_is_static_method() -> None:
    """_start_gazebo_go2 must be declared as a staticmethod."""
    raw = inspect.getattr_static(SimStartTool, "_start_gazebo_go2")
    assert isinstance(raw, staticmethod), (
        f"_start_gazebo_go2 is {type(raw).__name__}, expected staticmethod"
    )


# ---------------------------------------------------------------------------
# 4. execute() routes backend="gazebo" + sim_type="go2" → _start_gazebo_go2
# ---------------------------------------------------------------------------

def test_execute_routes_to_gazebo() -> None:
    """execute() must call _start_gazebo_go2 for backend='gazebo', sim_type='go2'."""
    tool = SimStartTool()
    mock_agent = MagicMock()
    mock_agent._skill_registry.list_skills.return_value = []
    # stats() must return a dict so the rooms > 0 comparison works
    mock_agent._spatial_memory.stats.return_value = {"rooms": 0, "objects": 0}
    context = MagicMock()
    context.app_state = {"agent": None, "registry": None, "engine": None}

    with patch.object(SimStartTool, "_start_gazebo_go2", return_value=mock_agent) as mock_method:
        result = tool.execute({"sim_type": "go2", "backend": "gazebo"}, context)

    mock_method.assert_called_once()
    assert not result.is_error, f"Expected success, got error: {result.content}"


# ---------------------------------------------------------------------------
# 5. Gazebo backend rejects non-go2 sim types
# ---------------------------------------------------------------------------

def test_gazebo_only_supports_go2() -> None:
    """Requesting backend='gazebo' with sim_type='arm' must return an error."""
    tool = SimStartTool()
    context = MagicMock()
    context.app_state = {"agent": None, "registry": None, "engine": None}

    result = tool.execute({"sim_type": "arm", "backend": "gazebo"}, context)

    assert result.is_error, "Expected is_error=True for gazebo + arm"
    assert "go2" in result.content.lower(), (
        f"Error message should mention 'go2', got: {result.content!r}"
    )


# ---------------------------------------------------------------------------
# 6. Default backend is now "gazebo"
# ---------------------------------------------------------------------------

def test_mujoco_is_default_backend() -> None:
    """Default backend must be 'mujoco' (MuJoCo is primary sim)."""
    backend_prop = SimStartTool.input_schema["properties"]["backend"]
    assert backend_prop.get("default") == "mujoco", (
        f"Default backend should be 'mujoco', got {backend_prop.get('default')!r}"
    )
