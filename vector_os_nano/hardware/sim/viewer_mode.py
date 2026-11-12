# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Is this process running under mjpython (the macOS launcher of a MuJoCo window)?

The REPL and the sim tool ask, because Apple GLFW is main-thread only: under mjpython
a window's physics is pumped by the main thread. The world itself decides whether a
stepper thread may drive it (``PhysicsWorld.caller_thread_only``).
"""
from __future__ import annotations


def running_under_mjpython() -> bool:
    """Return True when executing under mjpython (macOS Cocoa main-loop launcher).

    mjpython sets ``mujoco.viewer._MJPYTHON``; under plain python it is absent.
    This is the precise condition under which a passive viewer requires
    main-thread GL access -- more precise than a bare ``sys.platform`` check
    (a macOS process NOT under mjpython has no usable window anyway).
    """
    try:
        import mujoco.viewer as _mjv  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- mujoco may be absent (headless CI)
        return False
    return getattr(_mjv, "_MJPYTHON", None) is not None
