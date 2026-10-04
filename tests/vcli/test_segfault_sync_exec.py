# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P0 segfault fix — synchronous skill execution when a viewer is live on mjpython.

MuJoCo mjData and GLFW are not thread-safe, and GLFW is main-thread-ONLY under
mjpython/macOS, where the passive viewer is pumped by the caller/main thread. The
REPL runs each turn on a worker thread (``cli._run_turn_busy``) so it can read a stop
mid-turn; under mjpython with a live viewer that worker would race the main-thread
render and segfault, so the turn runs INLINE on the caller's (viewer-owning) thread.
Everywhere else — headless/dev, and Linux/Windows even WITH a live viewer (their
viewer+physics run on dedicated daemons) — the worker is kept so the REPL stays
responsive (Test A2).

Pure kernel logic — no real MuJoCo, no real model.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli import cli
from vector_os_nano.vcli.engine import VectorEngine


def _make_engine() -> VectorEngine:
    return VectorEngine(backend=MagicMock(), intent_router=MagicMock())


def _turn_thread(monkeypatch, *, mjpython: bool, viewer: object) -> threading.Thread:
    """The thread one REPL turn ran on."""
    import vector_os_nano.hardware.sim.viewer_mode as vm

    monkeypatch.setattr(vm, "running_under_mjpython", lambda: mjpython)
    monkeypatch.setattr(cli, "_read_line", lambda timeout: None)  # nobody types
    eng = _make_engine()
    eng._vgg_agent = SimpleNamespace(_arm=SimpleNamespace(_viewer=viewer))
    ran: list[threading.Thread] = []
    cli._run_turn_busy(eng, lambda: ran.append(threading.current_thread()), {})
    assert len(ran) == 1
    return ran[0]


# ---------------------------------------------------------------------------
# Test A — viewer live UNDER mjpython => the turn runs on the caller's thread
# (the macOS GLFW-main-thread constraint; only there must skills run inline)
# ---------------------------------------------------------------------------


def test_viewer_live_under_mjpython_runs_synchronously(monkeypatch) -> None:
    assert _turn_thread(monkeypatch, mjpython=True, viewer=object()) is threading.current_thread()


# ---------------------------------------------------------------------------
# Test A2 — viewer live but NOT mjpython (Linux/Windows) => worker thread.
# On those platforms the viewer + physics run on their own daemons, so the
# skill thread never touches GLFW/mjData; forcing sync-on-main would only
# freeze the REPL (laggy Ctrl-C / "stop") with no safety benefit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("viewer", [object(), None], ids=["viewer", "no_viewer"])
def test_off_mjpython_or_without_a_viewer_the_turn_runs_on_a_worker(monkeypatch, viewer) -> None:
    # Test B folded in: no viewer => the worker too (the CLI stays responsive).
    assert _turn_thread(monkeypatch, mjpython=False, viewer=viewer) is not threading.current_thread()
    assert _turn_thread(monkeypatch, mjpython=True, viewer=None) is not threading.current_thread()


# ---------------------------------------------------------------------------
# Test C — _has_live_viewer duck-typing across arm/base, _viewer/viewer attrs
# ---------------------------------------------------------------------------


def test_has_live_viewer_detection() -> None:
    eng = _make_engine()

    # Live arm viewer (_viewer attr).
    eng._vgg_agent = SimpleNamespace(_arm=SimpleNamespace(_viewer=object()))
    assert eng._has_live_viewer() is True

    # Live base viewer (public `viewer` attr).
    eng._vgg_agent = SimpleNamespace(_base=SimpleNamespace(viewer=object()))
    assert eng._has_live_viewer() is True

    # No agent connected.
    eng._vgg_agent = None
    assert eng._has_live_viewer() is False

    # Arm + base present but both viewers None.
    eng._vgg_agent = SimpleNamespace(
        _arm=SimpleNamespace(_viewer=None),
        _base=SimpleNamespace(_viewer=None, viewer=None),
    )
    assert eng._has_live_viewer() is False
