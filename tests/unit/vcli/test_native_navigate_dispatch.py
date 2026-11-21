"""Coordinate navigation uses the same registered skill as go_to_room/MCP."""
from types import SimpleNamespace
import threading
from unittest.mock import Mock

from vector_os_nano.core.types import SkillResult
from vector_os_nano.vcli.native_tools import _NativeBaseNavigateTool


def test_coordinate_navigation_preserves_registered_skill_failure_and_cancel():
    skill = SimpleNamespace(name='navigate', parameters={}, execute=Mock(return_value=
        SkillResult(False, {'reason': 'contact_interlock'}, 'Lift clear before moving', 'navigation_failed')))
    skill_context = SimpleNamespace(services={})
    base = Mock()
    agent = SimpleNamespace(_skill_registry={'navigate': skill}, _base=base,
                            _build_context=lambda: skill_context)
    token = threading.Event()
    context = SimpleNamespace(agent=None, app_state={'agent': agent}, abort=token)
    result = _NativeBaseNavigateTool().execute({'x': 1., 'y': 2.}, context)
    assert result.is_error and result.metadata['reason'] == 'contact_interlock'
    skill.execute.assert_called_once_with({'x': 1., 'y': 2.}, skill_context)
    assert skill_context.services['cancel'] is token
    base.navigate_to.assert_not_called()
