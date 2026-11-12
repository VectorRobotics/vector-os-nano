# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""A body started the product's way (``SimSession.start``), headless and caller-stepped.

Caller-stepped (``VECTOR_SIM_LOCKSTEP=1``): physics advances only inside the drivers'
timed waits, so a pick runs as fast as the machine allows and every test is
deterministic. The session owns the agent, its GroundTruth (``s.gt``) and its oracles
(``s.oracles()``).
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

import pytest


@contextlib.contextmanager
def started(embodiment: str) -> Iterator[Any]:
    from vector_os_nano.sim.runtime import LOCKSTEP_ENV
    from vector_os_nano.sim.session import SimSession, SimSpec

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(LOCKSTEP_ENV, "1")
        s = SimSession.start(SimSpec(embodiment, gui=False))
    try:
        yield s
    finally:
        s.stop()
