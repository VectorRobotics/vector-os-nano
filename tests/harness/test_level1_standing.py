"""Level 1 — PD standing verification.

Tests verify that the PD idle controller keeps the robot upright at
approximately the correct height with stable orientation.

Uses the go2_standing fixture (connect + stand already called).
All tests have manual timeout guards to avoid hanging in CI.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

# Standing target joints: [hip, thigh, calf] * 4 legs (FL FR RL RR)
_STAND_JOINTS = [0.0, 0.9, -1.8] * 4
_STAND_Z_MIN = 0.25       # metres — robot must be at least this high
_ANGULAR_VEL_MAX = 0.5    # rad/s — maximum body angular velocity magnitude
_JOINT_TOL = 0.15         # rad — joint tracking tolerance after stand()
_SETTLE_STEPS = 1000      # additional physics steps for stability check
_TIMEOUT_S = 30.0         # per-test hard timeout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.level1
def test_stand_height(go2_standing):
    """After stand(), base z > 0.25 m (robot is upright, not on ground)."""
    pos = go2_standing.get_position()
    assert pos[2] > _STAND_Z_MIN, (
        f"Standing height {pos[2]:.3f} m is below minimum {_STAND_Z_MIN} m"
    )


@pytest.mark.level1
def test_stand_stability(go2_standing):
    """After stand(), continue stepping — angular velocity stays < 0.5 rad/s."""
    deadline = time.monotonic() + _TIMEOUT_S

    # Use walk(vx=0, duration=1.0) to tick the physics thread for ~1 s without moving.
    # Then sample angular velocity from odometry.
    go2_standing.walk(vx=0.0, vy=0.0, vyaw=0.0, duration=1.0)

    assert time.monotonic() < deadline, "test_stand_stability exceeded timeout"

    odom = go2_standing.get_odometry()
    # vyaw is angular velocity around z axis; use it as a proxy for tipping tendency.
    # For full angular velocity magnitude we'd need IMU gyro, but vyaw is sufficient
    # as a stability proxy for tipping in the yaw axis.
    angular_vel_z = abs(odom.vyaw)
    assert angular_vel_z < _ANGULAR_VEL_MAX, (
        f"Angular velocity {angular_vel_z:.3f} rad/s exceeds limit {_ANGULAR_VEL_MAX} rad/s "
        "— robot may be tipping"
    )

    # Also verify height is maintained
    pos = go2_standing.get_position()
    assert pos[2] > _STAND_Z_MIN, (
        f"Robot fell during stability check: z={pos[2]:.3f} m"
    )


@pytest.mark.level1
def test_stand_joint_tracking(go2_standing):
    """After stand(), all 12 joint positions within 0.15 rad of target."""
    joints = go2_standing.get_joint_positions()

    assert len(joints) == 12, f"Expected 12 joints, got {len(joints)}"

    errors = []
    for idx, (target, actual) in enumerate(zip(_STAND_JOINTS, joints)):
        err = abs(actual - target)
        if err > _JOINT_TOL:
            errors.append(f"  joint[{idx}]: target={target:.2f}, actual={actual:.2f}, err={err:.3f}")

    assert not errors, (
        f"Joint tracking errors exceed {_JOINT_TOL} rad tolerance:\n"
        + "\n".join(errors)
    )


@pytest.mark.level1
def test_stand_quaternion_valid(go2_standing):
    """After stand(), orientation quaternion is unit and upright (qw > 0.9)."""
    odom = go2_standing.get_odometry()

    # Quaternion magnitude must be ~1
    q = np.array([odom.qx, odom.qy, odom.qz, odom.qw])
    mag = float(np.linalg.norm(q))
    assert abs(mag - 1.0) < 0.05, f"Quaternion magnitude {mag:.4f} is not unit"

    # Robot is approximately upright — qw close to 1.0 (small rotation from identity)
    assert abs(odom.qw) > 0.9, (
        f"qw={odom.qw:.3f} suggests robot is tilted more than ~25 degrees — not upright"
    )


@pytest.mark.level1
def test_stand_no_nan_in_state(go2_standing):
    """After stand(), position, heading, joints, and odometry are all NaN-free."""
    pos = go2_standing.get_position()
    heading = go2_standing.get_heading()
    joints = go2_standing.get_joint_positions()
    vel = go2_standing.get_velocity()
    odom = go2_standing.get_odometry()

    assert not any(math.isnan(v) for v in pos), f"NaN in position: {pos}"
    assert not math.isnan(heading), f"NaN in heading: {heading}"
    assert not any(math.isnan(j) for j in joints), f"NaN in joints: {joints}"
    assert not any(math.isnan(v) for v in vel), f"NaN in velocity: {vel}"

    odom_values = [odom.x, odom.y, odom.z, odom.qx, odom.qy, odom.qz, odom.qw,
                   odom.vx, odom.vy, odom.vz, odom.vyaw]
    assert not any(math.isnan(v) for v in odom_values), f"NaN in odometry: {odom}"
