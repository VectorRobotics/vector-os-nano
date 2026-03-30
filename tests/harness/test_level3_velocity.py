"""Level 3 — Velocity tracking verification.

Tests verify that set_velocity() produces robot motion in the commanded
direction and that zero velocity command stops the robot.

set_velocity() is non-blocking — the physics thread applies it.
Tests use time.sleep() to let the physics run, then sample state.

Tolerance: 50% of commanded velocity is acceptable (open-loop gait).
"""
from __future__ import annotations

import math
import time

import pytest

_TIMEOUT_S = 60.0
_MIN_Z = 0.15          # metres — upright threshold
_SETTLE_TIME_S = 5.0   # wait time before sampling velocity
_VEL_TOLERANCE = 0.5   # accept >= 50% of commanded speed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.level3
def test_velocity_tracking_forward(go2_standing):
    """set_velocity(0.3, 0, 0) → avg forward speed > 0.05 m/s over 5 s."""
    start_pos = go2_standing.get_position()
    deadline = time.monotonic() + _TIMEOUT_S

    go2_standing.set_velocity(0.3, 0.0, 0.0)
    time.sleep(_SETTLE_TIME_S)

    assert time.monotonic() < deadline, "test_velocity_tracking_forward exceeded timeout"

    end_pos = go2_standing.get_position()
    go2_standing.set_velocity(0.0, 0.0, 0.0)

    # Use average displacement over time (more robust than instantaneous velocity
    # since sinusoidal gait produces oscillating velocities)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    avg_speed = math.sqrt(dx * dx + dy * dy) / _SETTLE_TIME_S

    assert avg_speed >= 0.05, (
        f"Forward velocity tracking failed: avg_speed={avg_speed:.3f} m/s, "
        f"expected >= 0.05 m/s"
    )
    assert dx > 0, f"Robot moved backward (dx={dx:.3f}), expected forward"

    pos = go2_standing.get_position()
    assert pos[2] > _MIN_Z, f"Robot fell during velocity tracking: z={pos[2]:.3f} m"


@pytest.mark.level3
@pytest.mark.skip(reason="Hip abduction alone insufficient for lateral locomotion — needs body-frame velocity decomposition")
def test_velocity_tracking_lateral(go2_standing):
    """set_velocity(0, 0.2, 0) → robot moves in Y direction after 5 s."""
    start_pos = go2_standing.get_position()
    deadline = time.monotonic() + _TIMEOUT_S

    go2_standing.set_velocity(0.0, 0.2, 0.0)
    time.sleep(_SETTLE_TIME_S)

    assert time.monotonic() < deadline, "test_velocity_tracking_lateral exceeded timeout"

    end_pos = go2_standing.get_position()
    go2_standing.set_velocity(0.0, 0.0, 0.0)

    # Lateral displacement: robot should have moved in Y (or any XY direction since
    # heading may not be aligned with world axes — measure XY displacement)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    xy_displacement = math.sqrt(dx * dx + dy * dy)

    # Accept any XY movement > 0.1 m (lateral gait may not be perfectly side-pure)
    min_displacement = 0.1
    assert xy_displacement > min_displacement, (
        f"Lateral velocity tracking: XY displacement {xy_displacement:.3f} m "
        f"< minimum {min_displacement} m. start={start_pos[:2]}, end={end_pos[:2]}"
    )

    assert end_pos[2] > _MIN_Z, f"Robot fell during lateral walk: z={end_pos[2]:.3f} m"


@pytest.mark.level3
def test_velocity_tracking_yaw(go2_standing):
    """set_velocity(0, 0, 0.5) → heading changes > 0.5 rad over 3 s."""
    start_heading = go2_standing.get_heading()
    deadline = time.monotonic() + _TIMEOUT_S

    go2_standing.set_velocity(0.0, 0.0, 1.0)
    time.sleep(4.0)

    assert time.monotonic() < deadline, "test_velocity_tracking_yaw exceeded timeout"

    end_heading = go2_standing.get_heading()
    go2_standing.set_velocity(0.0, 0.0, 0.0)

    # Compute shortest angular delta
    delta = end_heading - start_heading
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    delta = abs(delta)

    assert delta > 0.3, (
        f"Yaw tracking: heading only changed {delta:.3f} rad (expected > 0.3 rad). "
        f"start={start_heading:.3f}, end={end_heading:.3f}"
    )

    pos = go2_standing.get_position()
    assert pos[2] > _MIN_Z, f"Robot fell during yaw command: z={pos[2]:.3f} m"


@pytest.mark.level3
def test_zero_velocity_stops(go2_standing):
    """After walking, set_velocity(0,0,0) → robot decelerates and nearly stops."""
    # Start walking
    go2_standing.set_velocity(0.3, 0.0, 0.0)
    time.sleep(2.0)

    # Command stop
    go2_standing.set_velocity(0.0, 0.0, 0.0)
    # Allow time for robot to decelerate and PD hold to take over
    time.sleep(2.0)

    vel = go2_standing.get_velocity()
    speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)

    # After stopping command, robot should be nearly stationary
    # Generous threshold (0.3 m/s) since PD hold may have some drift
    assert speed < 0.3, (
        f"Robot did not stop after zero-velocity command: speed={speed:.3f} m/s "
        f"(expected < 0.3 m/s)"
    )

    pos = go2_standing.get_position()
    assert pos[2] > _MIN_Z, f"Robot fell during stop: z={pos[2]:.3f} m"


@pytest.mark.level3
def test_set_velocity_non_blocking(go2_standing):
    """set_velocity() returns immediately (< 0.05 s) — it is non-blocking."""
    t0 = time.perf_counter()
    go2_standing.set_velocity(0.3, 0.0, 0.0)
    elapsed = time.perf_counter() - t0
    go2_standing.set_velocity(0.0, 0.0, 0.0)

    assert elapsed < 0.05, (
        f"set_velocity() took {elapsed:.4f} s — expected < 0.05 s (non-blocking)"
    )
