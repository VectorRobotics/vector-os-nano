# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""LeggedBase skill control without a simulator: who may command, and when.

ledger sim-locomotion:skill-control-single-slot: one token slot let a finishing holder
(a foreground walk) clear another's (a background navigate), re-opening the bridge's
cmd_vel path in the middle of that navigate.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

from vector_os_nano.sim.drivers.base import LeggedBase


def _bare() -> LeggedBase:
    base = LeggedBase.__new__(LeggedBase)
    base._spec = SimpleNamespace(limits=(0.8, 0.4, 4.0))
    base._lock, base._owners, base._cmd, base._hold, base._cmd_motion = threading.Lock(), [], (0.0,) * 3, None, 0.0
    base._rt = SimpleNamespace(wait_sim=lambda *a, **k: True, release=lambda: None)
    return base


def _in_thread(fn) -> None:
    t = threading.Thread(target=fn)
    t.start()
    t.join()


def test_a_finishing_holder_leaves_the_other_holders_control():
    base = _bare()
    navigate_holds, walk_done, seen = threading.Event(), threading.Event(), []

    def navigate() -> None:  # a background navigate holds control throughout
        with base._control():
            navigate_holds.set()
            walk_done.wait(5)
            base.set_velocity(0.3, 0.0, 0.0)  # still its command after the walk ended
            seen.append(base._cmd)

    t = threading.Thread(target=navigate)
    t.start()
    navigate_holds.wait(5)
    with base._control():  # a foreground walk takes over, then ends
        base.set_velocity(0.1, 0.0, 0.0)
        _in_thread(lambda: base.set_velocity(0.5, 0.0, 0.0))  # a bridge write: ignored
        assert base._cmd == (0.1, 0.0, 0.0)
    walk_done.set()
    t.join(5)
    assert seen == [(0.3, 0.0, 0.0)]
    assert base._cmd == (0.0, 0.0, 0.0) and base._owners == []  # zeroed by the navigate's exit
    _in_thread(lambda: base.set_velocity(0.2, 0.0, 0.0))  # nobody holds control now
    assert base._cmd == (0.2, 0.0, 0.0)


def test_the_bridge_is_gated_while_the_navigate_still_holds():
    """Adversarial: after the foreground walk ends, a bridge write lands only if the
    walk's exit dropped the navigate's hold too (the single-slot bug)."""
    base = _bare()
    navigate_holds, walk_done, checked = threading.Event(), threading.Event(), threading.Event()

    def navigate() -> None:
        with base._control():
            navigate_holds.set()
            checked.wait(5)

    t = threading.Thread(target=navigate)
    t.start()
    navigate_holds.wait(5)
    with base._control():
        pass  # the walk ends
    walk_done.set()
    _in_thread(lambda: base.set_velocity(0.5, 0.0, 0.0))  # the bridge, between the two
    assert base._cmd == (0.0, 0.0, 0.0), "the walk's exit re-opened the bridge path"
    checked.set()
    t.join(5)


def test_a_stop_is_never_gated():
    base = _bare()
    with base._control():
        base.set_velocity(0.4, 0.0, 0.0)
        _in_thread(base.stop)
        assert base._cmd == (0.0, 0.0, 0.0)
