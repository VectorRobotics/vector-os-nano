# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for the mjpython probe (hardware/sim/viewer_mode): pure logic, no GL."""
from vector_os_nano.hardware.sim import viewer_mode as vm


class TestRunningUnderMjpython:
    def test_false_under_plain_python(self):
        # The test runner is plain python, never mjpython.
        assert vm.running_under_mjpython() is False

    def test_true_when_mjpython_marker_present(self, monkeypatch):
        import sys
        import types
        fake_viewer = types.SimpleNamespace(_MJPYTHON=object())
        fake_pkg = types.ModuleType("mujoco")
        fake_pkg.viewer = fake_viewer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mujoco", fake_pkg)
        monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer)
        assert vm.running_under_mjpython() is True
