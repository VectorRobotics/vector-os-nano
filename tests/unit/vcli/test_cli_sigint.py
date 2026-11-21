# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2-4: Ctrl-C handling under mjpython.

mjpython can leave SIGINT bound to its own Cocoa handler, swallowing a single
Ctrl-C so it never raises KeyboardInterrupt in the REPL. _ensure_sigint_under_
mjpython restores Python's default int handler under mjpython only.
"""
import signal

import pytest


@pytest.fixture
def restore_sigint():
    orig = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, orig)


def test_noop_when_not_under_mjpython(monkeypatch, restore_sigint):
    from vector_os_nano.sim.backends.mujoco import viewer_mode
    from vector_os_nano.vcli import cli

    monkeypatch.setattr(viewer_mode, "running_under_mjpython", lambda: False)

    def _sentinel(*_a):
        return None

    signal.signal(signal.SIGINT, _sentinel)
    cli._ensure_sigint_under_mjpython()
    # Off mjpython the handler is left untouched.
    assert signal.getsignal(signal.SIGINT) is _sentinel


def test_restores_default_handler_under_mjpython(monkeypatch, restore_sigint):
    from vector_os_nano.sim.backends.mujoco import viewer_mode
    from vector_os_nano.vcli import cli

    monkeypatch.setattr(viewer_mode, "running_under_mjpython", lambda: True)

    # Simulate mjpython having swallowed SIGINT (bound to ignore).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cli._ensure_sigint_under_mjpython()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
