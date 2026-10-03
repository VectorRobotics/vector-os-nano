# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""fake_clock: skip waits without turning wall-clock loops into busy loops.

A bare ``patch("time.sleep")`` leaves ``time.time``/``time.monotonic`` real, so
every ``while time.monotonic() < deadline: ... sleep(dt)`` under test spins until
the REAL deadline, recording a MagicMock call per spin (gigabytes; this box OOM'd
twice). Here ``sleep`` advances a fake clock instead, so the deadline arrives at
once. The patch is process-wide (all threads), like the one it replaces.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch


@contextmanager
def fake_clock() -> Iterator[list[float]]:
    """Patch time.sleep/time/monotonic; sleep(s) advances both clocks by s (>= 1 ms)."""
    elapsed = [0.0]
    t0, m0 = time.time(), time.monotonic()

    def _sleep(seconds: float = 0.0) -> None:
        elapsed[0] += max(float(seconds), 1e-3)

    with patch("time.sleep", _sleep), patch("time.time", lambda: t0 + elapsed[0]), patch(
        "time.monotonic", lambda: m0 + elapsed[0]
    ):
        yield elapsed
