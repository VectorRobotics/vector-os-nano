"""Level 2 — Locomotion verification.

Tests verify that walk() produces meaningful displacement in the expected
direction and that the robot does not fall during motion.

All tests start from go2_standing (already connected + standing).
Timeout guards prevent indefinite hangs in CI.
"""
from __future__ import annotations

import math
import time

import pytest

_TIMEOUT_S = 60.0    # per-test hard timeout
_MIN_Z = 0.15        # metres — robot must stay above this to be considered upright


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xy_distance(p1: list[float], p2: list[float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def _heading_delta(h1: float, h2: float) -> float:
    """Shortest angular distance between two headings (rad), range [0, pi]."""
    delta = h2 - h1
    # Normalise to [-pi, pi]
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return abs(delta)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.level2
def test_walk_forward_displacement(go2_standing):
    """walk(vx=0.3, duration=5.0) moves > 0.3 m in XY plane."""
    start = go2_standing.get_position()
    deadline = time.monotonic() + _TIMEOUT_S

    result = go2_standing.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=5.0)

    assert time.monotonic() < deadline, "test_walk_forward_displacement exceeded timeout"

    end = go2_standing.get_position()
    dist = _xy_distance(start, end)

    assert dist > 0.3, (
        f"Forward walk displacement {dist:.3f} m is below 0.3 m minimum. "
        f"start={start[:2]}, end={end[:2]}"
    )


@pytest.mark.level2
def test_walk_stays_upright(go2_standing):
    """walk(vx=0.3, duration=5.0) — robot z stays above 0.15 m (no fall)."""
    deadline = time.monotonic() + _TIMEOUT_S

    go2_standing.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=5.0)

    assert time.monotonic() < deadline, "test_walk_stays_upright exceeded timeout"

    pos = go2_standing.get_position()
    assert pos[2] > _MIN_Z, (
        f"Robot fell during forward walk: z={pos[2]:.3f} m (minimum {_MIN_Z} m)"
    )


@pytest.mark.level2
def test_walk_backward(go2_standing):
    """walk(vx=-0.3, duration=3.0) moves robot in the negative-x direction."""
    start = go2_standing.get_position()
    heading = go2_standing.get_heading()
    deadline = time.monotonic() + _TIMEOUT_S

    go2_standing.walk(vx=-0.3, vy=0.0, vyaw=0.0, duration=3.0)

    assert time.monotonic() < deadline, "test_walk_backward exceeded timeout"

    end = go2_standing.get_position()

    # Project displacement onto the robot's forward axis at start
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    # Forward direction from heading
    fwd_x = math.cos(heading)
    fwd_y = math.sin(heading)
    # Dot product: negative means moved backward
    forward_displacement = dx * fwd_x + dy * fwd_y

    assert forward_displacement < -0.05, (
        f"Robot did not move backward: forward_displacement={forward_displacement:.3f} m "
        f"(expected < -0.05). start={start[:2]}, end={end[:2]}, heading={heading:.2f} rad"
    )

    # Still upright
    assert end[2] > _MIN_Z, f"Robot fell during backward walk: z={end[2]:.3f} m"


@pytest.mark.level2
def test_turn_in_place(go2_standing):
    """walk(vyaw=1.0, duration=4.0) changes heading by > 0.3 rad."""
    start_heading = go2_standing.get_heading()
    deadline = time.monotonic() + _TIMEOUT_S

    go2_standing.walk(vx=0.0, vy=0.0, vyaw=1.0, duration=4.0)

    assert time.monotonic() < deadline, "test_turn_in_place exceeded timeout"

    end_heading = go2_standing.get_heading()
    delta = _heading_delta(start_heading, end_heading)

    assert delta > 0.3, (
        f"Heading changed only {delta:.3f} rad (expected > 0.3 rad). "
        f"start={start_heading:.3f}, end={end_heading:.3f}"
    )

    # Still upright after turning
    pos = go2_standing.get_position()
    assert pos[2] > _MIN_Z, f"Robot fell during turn: z={pos[2]:.3f} m"


@pytest.mark.level2
def test_walk_returns_true_when_upright(go2_standing):
    """walk() returns True when robot stays upright."""
    result = go2_standing.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=2.0)
    assert result, (
        f"walk() returned {result!r} — expected truthy (robot should remain upright)"
    )
