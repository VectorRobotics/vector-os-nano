"""Tests for simulation backend switching in SimStartTool.

Level: Isaac-L5
Verifies that SimStartTool correctly routes to Isaac Sim or MuJoCo backends
depending on the 'backend' parameter.

All tests mock hardware instantiation — no Docker, MuJoCo, or ROS2 needed.
"""
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _import_sim_start_tool() -> type:
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool
    return SimStartTool


def _make_tool() -> Any:
    cls = _import_sim_start_tool()
    return cls()


def _make_app_state() -> dict:
    registry = MagicMock()
    registry.register = MagicMock()
    engine = MagicMock()
    engine._system_prompt = ""
    engine.init_vgg = MagicMock()
    return {"agent": None, "registry": registry, "engine": engine,
            "scene_graph": None, "skill_registry": None}


def _make_context(app_state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.app_state = app_state if app_state is not None else _make_app_state()
    ctx.cwd = "/tmp"
    return ctx


def _make_mock_agent(has_arm: bool = False) -> MagicMock:
    agent = MagicMock()
    agent._arm = MagicMock() if has_arm else None
    agent._base = None if has_arm else MagicMock()
    agent._spatial_memory = None
    agent._skill_registry = MagicMock()
    agent._skill_registry.list_skills.return_value = []
    return agent


def _execute(tool: Any, params: dict, ctx: MagicMock) -> Any:
    """Execute SimStartTool with local skill/prompt imports mocked."""
    with patch("vector_os_nano.vcli.tools.skill_wrapper.wrap_skills", return_value=[]), \
         patch("vector_os_nano.vcli.prompt.build_system_prompt", return_value=""):
        return tool.execute(params, ctx)


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

class TestSimStartToolSchema:
    def test_schema_has_properties(self) -> None:
        assert "properties" in _make_tool().input_schema

    def test_schema_has_sim_type_property(self) -> None:
        assert "sim_type" in _make_tool().input_schema["properties"]

    def test_schema_has_backend_property(self) -> None:
        assert "backend" in _make_tool().input_schema["properties"]

    def test_backend_enum_contains_mujoco(self) -> None:
        prop = _make_tool().input_schema["properties"]["backend"]
        assert "mujoco" in prop.get("enum", [])

    def test_backend_enum_contains_isaac(self) -> None:
        prop = _make_tool().input_schema["properties"]["backend"]
        assert "isaac" in prop.get("enum", [])

    def test_backend_default_is_isaac(self) -> None:
        prop = _make_tool().input_schema["properties"]["backend"]
        assert prop.get("default") == "isaac"

    def test_backend_enum_has_exactly_two_values(self) -> None:
        prop = _make_tool().input_schema["properties"]["backend"]
        assert len(prop.get("enum", [])) == 2

    def test_backend_property_has_description(self) -> None:
        prop = _make_tool().input_schema["properties"]["backend"]
        assert "description" in prop

    def test_backend_description_is_non_empty_string(self) -> None:
        desc = _make_tool().input_schema["properties"]["backend"].get("description", "")
        assert isinstance(desc, str) and len(desc) > 0

    def test_backend_description_mentions_backends(self) -> None:
        desc = _make_tool().input_schema["properties"]["backend"].get("description", "").lower()
        assert "isaac" in desc or "mujoco" in desc

    def test_sim_type_required(self) -> None:
        assert "sim_type" in _make_tool().input_schema.get("required", [])

    def test_backend_is_not_required(self) -> None:
        assert "backend" not in _make_tool().input_schema.get("required", [])

    def test_gui_parameter_still_present(self) -> None:
        assert "gui" in _make_tool().input_schema["properties"]

    def test_backend_type_is_string(self) -> None:
        prop = _make_tool().input_schema["properties"]["backend"]
        assert prop.get("type") == "string"


# ---------------------------------------------------------------------------
# 2. Routing — isaac backend
# ---------------------------------------------------------------------------

class TestIsaacBackendRouting:
    def test_isaac_go2_calls_start_isaac_go2(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_go2",
                          return_value=_make_mock_agent()) as mock_start:
            _execute(tool, {"sim_type": "go2", "backend": "isaac"}, ctx)
        mock_start.assert_called_once()

    def test_isaac_arm_calls_start_isaac_arm(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_arm",
                          return_value=_make_mock_agent(has_arm=True)) as mock_start:
            _execute(tool, {"sim_type": "arm", "backend": "isaac"}, ctx)
        mock_start.assert_called_once()

    def test_mujoco_not_called_for_isaac(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_go2", return_value=_make_mock_agent()), \
             patch.object(tool.__class__, "_start_go2", return_value=_make_mock_agent()) as m_mujoco:
            _execute(tool, {"sim_type": "go2", "backend": "isaac"}, ctx)
        m_mujoco.assert_not_called()

    def test_isaac_not_running_returns_error(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_go2",
                          side_effect=ConnectionError("container not running")):
            result = tool.execute({"sim_type": "go2", "backend": "isaac"}, ctx)
        assert result.is_error is True

    def test_isaac_error_message_is_descriptive(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_go2",
                          side_effect=ConnectionError("vector-isaac-sim not running. Start: launch_isaac.sh")):
            result = tool.execute({"sim_type": "go2", "backend": "isaac"}, ctx)
        assert result.is_error is True
        assert len(result.content) > 15

    def test_isaac_error_contains_sim_type_context(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_go2",
                          side_effect=RuntimeError("timeout")):
            result = tool.execute({"sim_type": "go2", "backend": "isaac"}, ctx)
        assert result.is_error is True
        assert "go2" in result.content.lower() or "sim" in result.content.lower()

    def test_no_backend_param_uses_isaac_default(self) -> None:
        """Omitting backend must use the schema default ('isaac')."""
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_isaac_go2",
                          return_value=_make_mock_agent()) as m_isaac, \
             patch.object(tool.__class__, "_start_go2",
                          return_value=_make_mock_agent()) as m_mujoco:
            _execute(tool, {"sim_type": "go2"}, ctx)  # no 'backend' key
        # Default is 'isaac', so isaac fires and mujoco does not
        m_isaac.assert_called_once()
        m_mujoco.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Routing — mujoco backend
# ---------------------------------------------------------------------------

class TestMuJoCoBackendRouting:
    def test_mujoco_go2_calls_start_go2(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_go2",
                          return_value=_make_mock_agent()) as mock_start:
            _execute(tool, {"sim_type": "go2", "backend": "mujoco"}, ctx)
        mock_start.assert_called_once()

    def test_mujoco_arm_calls_start_arm(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_arm",
                          return_value=_make_mock_agent(has_arm=True)) as mock_start:
            _execute(tool, {"sim_type": "arm", "backend": "mujoco"}, ctx)
        mock_start.assert_called_once()

    def test_isaac_not_called_for_mujoco(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_go2", return_value=_make_mock_agent()), \
             patch.object(tool.__class__, "_start_isaac_go2",
                          return_value=_make_mock_agent()) as m_isaac:
            _execute(tool, {"sim_type": "go2", "backend": "mujoco"}, ctx)
        m_isaac.assert_not_called()

    def test_mujoco_result_is_not_error(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_go2", return_value=_make_mock_agent()):
            result = _execute(tool, {"sim_type": "go2", "backend": "mujoco"}, ctx)
        assert not result.is_error


# ---------------------------------------------------------------------------
# 4. Unknown sim_type
# ---------------------------------------------------------------------------

class TestUnknownSimType:
    def test_unknown_sim_type_mujoco_returns_error(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        result = tool.execute({"sim_type": "drone", "backend": "mujoco"}, ctx)
        assert result.is_error is True

    def test_unknown_sim_type_isaac_returns_error(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        result = tool.execute({"sim_type": "drone", "backend": "isaac"}, ctx)
        assert result.is_error is True

    def test_error_message_mentions_sim_type(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        result = tool.execute({"sim_type": "hovercraft", "backend": "mujoco"}, ctx)
        assert ("hovercraft" in result.content.lower() or
                "unknown" in result.content.lower())


# ---------------------------------------------------------------------------
# 5. _start_isaac_go2 source-level checks
# ---------------------------------------------------------------------------

class TestStartIsaacGo2:
    def test_start_isaac_go2_method_exists(self) -> None:
        assert hasattr(_import_sim_start_tool(), "_start_isaac_go2")

    def test_start_isaac_go2_is_callable(self) -> None:
        assert callable(_import_sim_start_tool()._start_isaac_go2)

    def test_start_isaac_go2_references_isaac_sim_proxy(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_go2)
        assert "IsaacSimProxy" in source

    def test_start_isaac_go2_calls_connect(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_go2)
        assert "connect" in source

    def test_start_isaac_go2_creates_agent(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_go2)
        assert "Agent" in source

    def test_start_isaac_go2_sets_scene_graph(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_go2)
        assert "scene_graph" in source.lower() or "SceneGraph" in source

    def test_start_isaac_go2_raises_on_connection_failure(self) -> None:
        cls = _import_sim_start_tool()
        with patch("vector_os_nano.hardware.sim.isaac_sim_proxy.IsaacSimProxy.connect",
                   side_effect=ConnectionError("not running")):
            with pytest.raises((ConnectionError, Exception)):
                cls._start_isaac_go2()


# ---------------------------------------------------------------------------
# 6. _start_isaac_arm source-level checks
# ---------------------------------------------------------------------------

class TestStartIsaacArm:
    def test_start_isaac_arm_method_exists(self) -> None:
        assert hasattr(_import_sim_start_tool(), "_start_isaac_arm")

    def test_start_isaac_arm_is_callable(self) -> None:
        assert callable(_import_sim_start_tool()._start_isaac_arm)

    def test_start_isaac_arm_references_arm_proxy(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_arm)
        assert "IsaacSim" in source or "Proxy" in source

    def test_start_isaac_arm_creates_agent(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_arm)
        assert "Agent" in source

    def test_start_isaac_arm_calls_connect(self) -> None:
        source = inspect.getsource(_import_sim_start_tool()._start_isaac_arm)
        assert "connect" in source


# ---------------------------------------------------------------------------
# 7. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_app_state_returns_error(self) -> None:
        tool = _make_tool()
        ctx = _make_context(app_state=None)
        result = tool.execute({"sim_type": "go2"}, ctx)
        assert result.is_error is True

    def test_gui_parameter_forwarded_to_mujoco(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        captured: list[dict] = []

        def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return _make_mock_agent()

        with patch.object(tool.__class__, "_start_go2", side_effect=capture):
            _execute(tool, {"sim_type": "go2", "backend": "mujoco", "gui": False}, ctx)

        if captured:
            assert captured[0].get("gui") is False

    def test_result_has_is_error_attribute(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_go2", return_value=_make_mock_agent()):
            result = _execute(tool, {"sim_type": "go2", "backend": "mujoco"}, ctx)
        assert hasattr(result, "is_error")
        assert hasattr(result, "content")

    def test_already_running_go2_returns_message(self) -> None:
        tool = _make_tool()
        existing_agent = _make_mock_agent()
        ctx = _make_context(app_state={
            "agent": existing_agent,
            "registry": MagicMock(),
            "engine": MagicMock(),
        })
        result = tool.execute({"sim_type": "go2", "backend": "mujoco"}, ctx)
        assert result is not None
        assert isinstance(result.content, str)

    def test_mujoco_result_content_not_error(self) -> None:
        tool = _make_tool()
        ctx = _make_context()
        with patch.object(tool.__class__, "_start_go2", return_value=_make_mock_agent()):
            result = _execute(tool, {"sim_type": "go2", "backend": "mujoco"}, ctx)
        assert not result.is_error
