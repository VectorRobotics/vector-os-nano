"""Level 4 — Navigation to waypoint (placeholder).

Implements a simple proportional controller to drive the robot from
(0, 0) to a goal point. This does NOT use Nav2 — Nav2 integration
is deferred until T10/T11 (ROS2 bridge + Nav2 launch file).

Proportional controller logic:
    heading_error = angle_to_goal - current_heading
    vyaw = K_yaw * heading_error  (if |heading_error| > 0.2 rad)
    vx = K_vx * dist_to_goal      (if roughly pointed at goal)

The test verifies the robot arrives within arrival_tolerance_m of the goal.
"""
from __future__ import annotations

import math
import time
from typing import Tuple

import pytest

# Navigation parameters (conservative for open-loop gait)
_GOAL_X = 2.0               # target x coordinate (metres from start)
_GOAL_Y = 0.0               # target y coordinate
_ARRIVAL_TOL_M = 0.5        # success if within this distance of goal
_NAV_TIMEOUT_S = 40.0       # abort if not arrived within this time
_MIN_Z = 0.15               # upright threshold

# Proportional controller gains
_K_YAW = 2.0                # rad/s per rad heading error
_K_VX = 0.5                 # m/s per metre distance to goal
_MAX_VX = 0.3               # m/s — velocity cap
_MAX_VYAW = 1.0             # rad/s — yaw rate cap
_ALIGN_THRESHOLD = 0.3      # rad — start moving forward when within this angle of goal
_STEP_DURATION_S = 0.2      # seconds between controller updates


# ---------------------------------------------------------------------------
# P-controller helpers
# ---------------------------------------------------------------------------

def _xy_dist(pos: list[float], goal: Tuple[float, float]) -> float:
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    return math.sqrt(dx * dx + dy * dy)


def _angle_to_goal(pos: list[float], heading: float, goal: Tuple[float, float]) -> float:
    """Signed angle error from current heading to direction toward goal (rad)."""
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    desired_heading = math.atan2(dy, dx)
    error = desired_heading - heading
    # Normalise to [-pi, pi]
    while error > math.pi:
        error -= 2 * math.pi
    while error < -math.pi:
        error += 2 * math.pi
    return error


def _p_navigate(robot: object, goal: Tuple[float, float], timeout_s: float) -> bool:
    """Drive robot to goal using a proportional controller.

    Args:
        robot: MuJoCoGo2 instance (must already be standing).
        goal: (x, y) target in world frame.
        timeout_s: Maximum time allowed.

    Returns:
        True if arrived within _ARRIVAL_TOL_M; False if timed out or fell.
    """
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        pos = robot.get_position()  # type: ignore[attr-defined]
        heading = robot.get_heading()  # type: ignore[attr-defined]

        # Upright check
        if pos[2] < _MIN_Z:
            return False

        dist = _xy_dist(pos, goal)
        if dist <= _ARRIVAL_TOL_M:
            robot.set_velocity(0.0, 0.0, 0.0)  # type: ignore[attr-defined]
            return True

        heading_err = _angle_to_goal(pos, heading, goal)

        # Compute proportional commands
        vyaw = float(_K_YAW * heading_err)
        vyaw = max(-_MAX_VYAW, min(_MAX_VYAW, vyaw))

        # Only drive forward when roughly aligned with goal
        if abs(heading_err) < _ALIGN_THRESHOLD:
            vx = float(_K_VX * dist)
            vx = min(_MAX_VX, vx)
        else:
            vx = 0.0

        robot.set_velocity(vx, 0.0, vyaw)  # type: ignore[attr-defined]
        time.sleep(_STEP_DURATION_S)

    # Timed out — stop robot
    robot.set_velocity(0.0, 0.0, 0.0)  # type: ignore[attr-defined]
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.level4
def test_navigate_to_point(go2_standing):
    """P-controller drives robot from (0,0) to (2,0) within 0.5 m tolerance.

    This is a placeholder for Nav2 integration. The proportional controller
    uses set_velocity() to steer toward the goal using heading feedback.
    """
    # Record start position as the reference origin
    start_pos = go2_standing.get_position()
    goal_world = (start_pos[0] + _GOAL_X, start_pos[1] + _GOAL_Y)

    arrived = _p_navigate(go2_standing, goal_world, timeout_s=_NAV_TIMEOUT_S)

    final_pos = go2_standing.get_position()
    dist_to_goal = _xy_dist(final_pos, goal_world)

    assert arrived, (
        f"Navigation timed out or robot fell before reaching goal. "
        f"goal={goal_world}, final_pos={final_pos[:2]}, "
        f"dist_to_goal={dist_to_goal:.3f} m"
    )
    assert dist_to_goal <= _ARRIVAL_TOL_M, (
        f"Navigation arrived flag set but distance {dist_to_goal:.3f} m "
        f"exceeds tolerance {_ARRIVAL_TOL_M} m"
    )
    assert final_pos[2] > _MIN_Z, (
        f"Robot fell during navigation: z={final_pos[2]:.3f} m"
    )


@pytest.mark.level4
def test_navigate_heading_convergence(go2_standing):
    """P-controller aligns heading toward goal within 3 s before driving forward.

    Verifies the yaw component of the controller works — robot should point
    roughly toward (2, 0) within the alignment threshold.
    """
    start_pos = go2_standing.get_position()
    goal_world = (start_pos[0] + _GOAL_X, start_pos[1] + _GOAL_Y)

    # Run only the alignment phase (yaw-only for 3 s)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        pos = go2_standing.get_position()
        heading = go2_standing.get_heading()

        heading_err = _angle_to_goal(pos, heading, goal_world)
        vyaw = float(_K_YAW * heading_err)
        vyaw = max(-_MAX_VYAW, min(_MAX_VYAW, vyaw))
        go2_standing.set_velocity(0.0, 0.0, vyaw)
        time.sleep(_STEP_DURATION_S)

    go2_standing.set_velocity(0.0, 0.0, 0.0)

    final_pos = go2_standing.get_position()
    final_heading = go2_standing.get_heading()
    final_err = abs(_angle_to_goal(final_pos, final_heading, goal_world))

    # After 3 s of yaw control, heading error should be small
    assert final_err < 0.5, (
        f"Heading did not converge: final_err={final_err:.3f} rad (expected < 0.5 rad). "
        f"pos={final_pos[:2]}, heading={final_heading:.3f}"
    )
    assert final_pos[2] > _MIN_Z, f"Robot fell during alignment: z={final_pos[2]:.3f} m"
