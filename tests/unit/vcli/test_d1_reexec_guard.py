# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for Phase D.1 — NL-first visible sim re-exec guard.

Verifies:
- _maybe_reexec_under_mjpython never calls os.execve during pytest
- _wants_window logic (--sim / --sim-go2 / --headless combinations)
- --headless flag is present in parse_args; --gui is removed
- sim_tool schema: gui default is True
- sim_tool._viewer_available() returns True on non-macOS
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch


# ---------------------------------------------------------------------------
# parse_args / flag tests
# ---------------------------------------------------------------------------


def test_headless_flag_exists():
    from vector_os_nano.vcli.cli import parse_args
    args = parse_args(["--headless"])
    assert args.headless is True


def test_gui_flag_removed():
    from vector_os_nano.vcli.cli import parse_args
    import pytest
    with pytest.raises(SystemExit):
        parse_args(["--gui"])


def test_sim_default_no_headless():
    from vector_os_nano.vcli.cli import parse_args
    args = parse_args(["--sim"])
    assert args.sim is True
    assert args.headless is False


def test_sim_headless():
    from vector_os_nano.vcli.cli import parse_args
    args = parse_args(["--sim", "--headless"])
    assert args.headless is True


# ---------------------------------------------------------------------------
# _wants_window tests
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"sim": False, "sim_go2": False, "headless": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_wants_window_sim_no_headless():
    from vector_os_nano.vcli.cli import _wants_window
    assert _wants_window(_make_args(sim=True)) is True


def test_wants_window_sim_go2_no_headless():
    from vector_os_nano.vcli.cli import _wants_window
    assert _wants_window(_make_args(sim_go2=True)) is True


def test_wants_window_headless_suppresses():
    from vector_os_nano.vcli.cli import _wants_window
    assert _wants_window(_make_args(sim=True, headless=True)) is False


def test_wants_window_no_sim():
    from vector_os_nano.vcli.cli import _wants_window
    assert _wants_window(_make_args()) is False


# ---------------------------------------------------------------------------
# _maybe_reexec_under_mjpython must never fire during pytest
# ---------------------------------------------------------------------------


def test_reexec_guard_never_fires_in_pytest():
    """os.execve must NOT be called during test runs."""
    from vector_os_nano.vcli.cli import _maybe_reexec_under_mjpython

    execve_called = False

    def _fake_execve(*args, **kwargs):
        nonlocal execve_called
        execve_called = True

    with patch("os.execve", _fake_execve):
        _maybe_reexec_under_mjpython(_make_args(sim=True, headless=False))

    assert not execve_called, "os.execve must not be called during pytest"


def test_reexec_guard_skips_when_headless():
    from vector_os_nano.vcli.cli import _maybe_reexec_under_mjpython

    execve_called = False

    def _fake_execve(*args, **kwargs):
        nonlocal execve_called
        execve_called = True

    with patch("os.execve", _fake_execve):
        _maybe_reexec_under_mjpython(_make_args(sim=True, headless=True))

    assert not execve_called


def test_reexec_guard_skips_when_no_sim():
    from vector_os_nano.vcli.cli import _maybe_reexec_under_mjpython

    execve_called = False

    def _fake_execve(*args, **kwargs):
        nonlocal execve_called
        execve_called = True

    with patch("os.execve", _fake_execve):
        _maybe_reexec_under_mjpython(_make_args())

    assert not execve_called


def test_reexec_guard_skips_when_already_reexecd(monkeypatch):
    from vector_os_nano.vcli.cli import _maybe_reexec_under_mjpython

    monkeypatch.setenv("VECTOR_REEXEC", "1")
    execve_called = False

    def _fake_execve(*args, **kwargs):
        nonlocal execve_called
        execve_called = True

    with patch("os.execve", _fake_execve):
        _maybe_reexec_under_mjpython(_make_args(sim=True))

    assert not execve_called


# ---------------------------------------------------------------------------
# sim_tool — schema default and _viewer_available
# ---------------------------------------------------------------------------


def test_sim_tool_gui_default_true():
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool
    tool = SimStartTool()
    gui_prop = tool.input_schema["properties"]["gui"]
    assert gui_prop["default"] is True


def test_viewer_available_non_macos():
    """On non-macOS (or when mocked), _viewer_available returns True."""
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool
    with patch("sys.platform", "linux"):
        assert SimStartTool._viewer_available() is True


def test_viewer_available_macos_no_mjpython():
    """On macOS without mjpython, _viewer_available returns False."""
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    class _FakeMjViewer:
        _MJPYTHON = None

    with patch("sys.platform", "darwin"), \
         patch.dict("sys.modules", {"mujoco.viewer": _FakeMjViewer()}):
        assert SimStartTool._viewer_available() is False


def test_viewer_available_macos_with_mjpython():
    """On macOS under mjpython, _viewer_available returns True.

    Patches the _MJPYTHON attribute on the already-imported mujoco.viewer
    module object (works even when the real module is cached in sys.modules).
    """
    import mujoco.viewer as _real_mjv
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    original = getattr(_real_mjv, "_MJPYTHON", None)
    try:
        _real_mjv._MJPYTHON = True  # type: ignore[attr-defined]
        with patch("sys.platform", "darwin"):
            assert SimStartTool._viewer_available() is True
    finally:
        _real_mjv._MJPYTHON = original  # type: ignore[attr-defined]
