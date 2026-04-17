"""CodeExecutor TDD tests — Phase 2 T1.

Tests:
  AC-1   print(get_position()) → success, stdout contains "(10.0, 5.0, 0.3)"
  AC-2   import os → success=False, blocked by AST
  AC-3   set_velocity(5.0, 0, 0) → clamped to (1.0, 0, 0)
  AC-4   while True: pass → timeout, success=False
  AC-5   x = nearest_room(); print(x) → stdout contains "kitchen"
  AC-6   __builtins__ access → blocked (dunder attr)
  AC-7   import math; print(math.pi) → success, stdout contains "3.14"
  AC-8   multi-line loop → success, stdout contains each room
  AC-9   from math import pi; print(pi) → allowed
  AC-10  from os import path → blocked
  AC-11  x.__class__ → blocked (dunder attr access)
  AC-12  empty code string → success=True, empty stdout
  AC-13  syntax error → success=False
  AC-14  exception in primitive → success=False, error captured
  AC-15  return value: last expression get_position() → return_value = (10.0, 5.0, 0.3)
  AC-16  duration tracked (> 0)
  AC-17  set_velocity with negative clamping (-5.0, -5.0, -5.0) → clamped to (-1.0, -1.0, -2.0)
  AC-18  multi-arg call detect_objects(query="cup") → success
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from vector_os_nano.vcli.cognitive.code_executor import CodeExecutor, CodeResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_set_velocity() -> MagicMock:
    return MagicMock(return_value=None)


@pytest.fixture()
def mock_stop() -> MagicMock:
    return MagicMock(return_value=None)


@pytest.fixture()
def mock_primitives(mock_set_velocity, mock_stop) -> dict:
    return {
        "get_position": lambda: (10.0, 5.0, 0.3),
        "get_heading": lambda: 1.57,
        "nearest_room": lambda: "kitchen",
        "set_velocity": mock_set_velocity,
        "stop": mock_stop,
        "describe_scene": lambda: "A kitchen with a table",
        "detect_objects": lambda query="": [{"name": "cup", "confidence": 0.9}],
        "get_visited_rooms": lambda: ["kitchen", "hallway"],
        "walk_forward": lambda distance_m=1.0, speed=0.4: True,
        "turn": lambda angle_rad=1.57: True,
    }


@pytest.fixture()
def executor(mock_primitives) -> CodeExecutor:
    return CodeExecutor(mock_primitives, timeout_sec=5.0)


# ---------------------------------------------------------------------------
# AC-1: print(get_position()) → success, stdout contains "(10.0, 5.0, 0.3)"
# ---------------------------------------------------------------------------


def test_ac1_print_get_position(executor: CodeExecutor) -> None:
    result = executor.execute("print(get_position())")
    assert result.success is True
    assert "(10.0, 5.0, 0.3)" in result.stdout


# ---------------------------------------------------------------------------
# AC-2: import os → blocked
# ---------------------------------------------------------------------------


def test_ac2_import_os_blocked(executor: CodeExecutor) -> None:
    result = executor.execute("import os")
    assert result.success is False
    assert result.error != ""


# ---------------------------------------------------------------------------
# AC-3: set_velocity clamped to MAX (1.0, 1.0, 2.0)
# ---------------------------------------------------------------------------


def test_ac3_set_velocity_clamped_positive(
    executor: CodeExecutor, mock_set_velocity: MagicMock
) -> None:
    result = executor.execute("set_velocity(5.0, 0, 0)")
    assert result.success is True
    # Underlying mock should have been called with clamped value
    mock_set_velocity.assert_called_once_with(1.0, 0, 0)


# ---------------------------------------------------------------------------
# AC-4: infinite loop → timeout
# ---------------------------------------------------------------------------


def test_ac4_infinite_loop_timeout() -> None:
    # Use very short timeout to keep test fast
    from vector_os_nano.vcli.cognitive.code_executor import CodeExecutor

    primitives: dict = {}
    short_executor = CodeExecutor(primitives, timeout_sec=0.5)
    result = short_executor.execute("while True: pass")
    assert result.success is False
    assert "timeout" in result.error.lower() or result.error != ""


# ---------------------------------------------------------------------------
# AC-5: assignment + print → stdout contains "kitchen"
# ---------------------------------------------------------------------------


def test_ac5_assignment_and_print(executor: CodeExecutor) -> None:
    code = "x = nearest_room()\nprint(x)"
    result = executor.execute(code)
    assert result.success is True
    assert "kitchen" in result.stdout


# ---------------------------------------------------------------------------
# AC-6: __builtins__ access → blocked
# ---------------------------------------------------------------------------


def test_ac6_dunder_builtins_blocked(executor: CodeExecutor) -> None:
    result = executor.execute("x = __builtins__")
    assert result.success is False


# ---------------------------------------------------------------------------
# AC-7: import math allowed; math.pi printed
# ---------------------------------------------------------------------------


def test_ac7_import_math_allowed(executor: CodeExecutor) -> None:
    result = executor.execute("import math\nprint(math.pi)")
    assert result.success is True
    assert "3.14" in result.stdout


# ---------------------------------------------------------------------------
# AC-8: multi-line loop printing each room
# ---------------------------------------------------------------------------


def test_ac8_multiline_loop(executor: CodeExecutor) -> None:
    code = (
        "rooms = get_visited_rooms()\n"
        "for r in rooms:\n"
        "    print(f'Room: {r}')"
    )
    result = executor.execute(code)
    assert result.success is True
    assert "Room: kitchen" in result.stdout
    assert "Room: hallway" in result.stdout


# ---------------------------------------------------------------------------
# AC-9: from math import pi → allowed
# ---------------------------------------------------------------------------


def test_ac9_from_math_import_allowed(executor: CodeExecutor) -> None:
    result = executor.execute("from math import pi\nprint(pi)")
    assert result.success is True
    assert "3.14" in result.stdout


# ---------------------------------------------------------------------------
# AC-10: from os import path → blocked
# ---------------------------------------------------------------------------


def test_ac10_from_os_import_blocked(executor: CodeExecutor) -> None:
    result = executor.execute("from os import path")
    assert result.success is False
    assert result.error != ""


# ---------------------------------------------------------------------------
# AC-11: x.__class__ → blocked (dunder attribute access)
# ---------------------------------------------------------------------------


def test_ac11_dunder_attr_access_blocked(executor: CodeExecutor) -> None:
    result = executor.execute("x = 1\ny = x.__class__")
    assert result.success is False


# ---------------------------------------------------------------------------
# AC-12: empty code string → success, empty stdout
# ---------------------------------------------------------------------------


def test_ac12_empty_code(executor: CodeExecutor) -> None:
    result = executor.execute("")
    assert result.success is True
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# AC-13: syntax error → success=False
# ---------------------------------------------------------------------------


def test_ac13_syntax_error(executor: CodeExecutor) -> None:
    result = executor.execute("def (")
    assert result.success is False
    assert result.error != ""


# ---------------------------------------------------------------------------
# AC-14: exception in primitive → success=False, error captured
# ---------------------------------------------------------------------------


def test_ac14_primitive_exception(mock_primitives: dict) -> None:
    def boom():
        raise RuntimeError("sensor offline")

    mock_primitives["get_position"] = boom
    ex = CodeExecutor(mock_primitives, timeout_sec=5.0)
    result = ex.execute("get_position()")
    assert result.success is False
    assert "sensor offline" in result.error


# ---------------------------------------------------------------------------
# AC-15: return value — last expression is captured
# ---------------------------------------------------------------------------


def test_ac15_return_value_last_expression(executor: CodeExecutor) -> None:
    result = executor.execute("get_position()")
    assert result.success is True
    assert result.return_value == (10.0, 5.0, 0.3)


# ---------------------------------------------------------------------------
# AC-16: duration is tracked (> 0)
# ---------------------------------------------------------------------------


def test_ac16_duration_tracked(executor: CodeExecutor) -> None:
    result = executor.execute("x = 1 + 1")
    assert result.duration_sec > 0.0


# ---------------------------------------------------------------------------
# AC-17: set_velocity negative clamping
# ---------------------------------------------------------------------------


def test_ac17_set_velocity_clamped_negative(
    executor: CodeExecutor, mock_set_velocity: MagicMock
) -> None:
    result = executor.execute("set_velocity(-5.0, -5.0, -5.0)")
    assert result.success is True
    mock_set_velocity.assert_called_once_with(-1.0, -1.0, -2.0)


# ---------------------------------------------------------------------------
# AC-18: detect_objects with keyword arg
# ---------------------------------------------------------------------------


def test_ac18_detect_objects_kwarg(executor: CodeExecutor) -> None:
    code = 'objs = detect_objects(query="cup")\nprint(len(objs))'
    result = executor.execute(code)
    assert result.success is True
    assert "1" in result.stdout


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_return_value_is_none_when_last_is_assignment(executor: CodeExecutor) -> None:
    """Assignment as last statement → return_value is None."""
    result = executor.execute("x = get_position()")
    assert result.success is True
    assert result.return_value is None


def test_math_module_available_without_import(executor: CodeExecutor) -> None:
    """math is pre-injected into globals — usable without explicit import."""
    result = executor.execute("print(math.sqrt(4))")
    assert result.success is True
    assert "2.0" in result.stdout


def test_safe_builtins_len_available(executor: CodeExecutor) -> None:
    result = executor.execute("print(len([1,2,3]))")
    assert result.success is True
    assert "3" in result.stdout


def test_safe_builtins_range_available(executor: CodeExecutor) -> None:
    result = executor.execute("print(list(range(3)))")
    assert result.success is True
    assert "[0, 1, 2]" in result.stdout


def test_import_sys_blocked(executor: CodeExecutor) -> None:
    result = executor.execute("import sys")
    assert result.success is False


def test_code_result_is_frozen_dataclass() -> None:
    """CodeResult must be immutable (frozen=True)."""
    cr = CodeResult(
        success=True,
        stdout="hi",
        return_value=None,
        error="",
        duration_sec=0.1,
    )
    with pytest.raises((AttributeError, TypeError)):
        cr.success = False  # type: ignore[misc]
