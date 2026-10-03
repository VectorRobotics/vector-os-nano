# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""v2.4 T6 integration — virtual sensors against a realistic MJCF.

These tests exercise :class:`MuJoCoLivox360`, :class:`MuJoCoPano360`,
and :class:`GroundTruthOdomPublisher` against a medium-sized inline
MJCF that has multiple geometries arranged at known positions. The
goal is to catch geometry / coordinate-frame regressions that the
unit-test fixtures (single trunk + one wall) cannot.

Inline MJCF deliberately avoids importing the Go2 model so the test
file stays light (per ``feedback_no_parallel_agents.md``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest


_ROOM_MJCF = """
<mujoco>
  <option timestep="0.002"/>
  <visual><global fovy="90"/></visual>
  <worldbody>
    <body name="trunk" pos="0 0 0.5">
      <freejoint name="trunk_root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
    <!-- Four walls forming a 6m × 6m room centred at origin -->
    <geom name="wall_pos_x" type="box" pos="3 0 1"  size="0.1 3 1" rgba="1 0 0 1"/>
    <geom name="wall_neg_x" type="box" pos="-3 0 1" size="0.1 3 1" rgba="0 1 0 1"/>
    <geom name="wall_pos_y" type="box" pos="0 3 1"  size="3 0.1 1" rgba="0 0 1 1"/>
    <geom name="wall_neg_y" type="box" pos="0 -3 1" size="3 0.1 1" rgba="1 1 0 1"/>
    <!-- Floor + ceiling -->
    <geom name="floor"   type="plane" pos="0 0 0"   size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="ceiling" type="box"   pos="0 0 2.1" size="3 3 0.05" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def room_model_data():
    """Compile a realistic 6×6 m room exactly once per module."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(_ROOM_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ---------------------------------------------------------------------------
# Lidar against a 4-wall room
# ---------------------------------------------------------------------------


def test_lidar_publishes_at_least_one_thousand_points_against_room(
    room_model_data,
) -> None:
    from vector_os_nano.hardware.sim.sensors import MuJoCoLivox360
    lidar = MuJoCoLivox360(*room_model_data, h_resolution=180, v_layers=8)
    sample = lidar.step(now=0.0)
    assert sample.points.shape[0] >= 1000


def test_lidar_hits_all_four_walls(room_model_data) -> None:
    """Trunk at origin → rays in ±X and ±Y directions hit the four walls
    at the expected near-face distances (≈ 2.9 m given the wall thickness)."""
    from vector_os_nano.hardware.sim.sensors import MuJoCoLivox360
    lidar = MuJoCoLivox360(
        *room_model_data,
        h_resolution=4, v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
        offset=(0.0, 0.0, 0.0),
        max_range=10.0,
    )
    sample = lidar.step(now=0.0)
    assert sample.points.shape[0] == 4
    distances = np.linalg.norm(sample.points[:, :3], axis=1)
    # Walls are 6 m apart centre-to-centre, half-thickness 0.1; so the
    # near face is at radius 2.9 m from origin.
    assert np.allclose(distances, 2.9, atol=0.1)


def test_lidar_clamps_at_max_range_inside_room(room_model_data) -> None:
    from vector_os_nano.hardware.sim.sensors import MuJoCoLivox360
    lidar = MuJoCoLivox360(
        *room_model_data,
        h_resolution=8, v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
        offset=(0.0, 0.0, 0.0),
        max_range=2.0,        # walls are 2.9 m → no hits expected
    )
    sample = lidar.step(now=0.0)
    assert sample.points.shape[0] == 0


# ---------------------------------------------------------------------------
# Pano camera against the same room
# ---------------------------------------------------------------------------


def test_pano_image_shape_after_render(room_model_data) -> None:
    from vector_os_nano.hardware.sim.sensors import MuJoCoPano360
    pano = MuJoCoPano360(
        *room_model_data,
        out_w=256, out_h=128, face_size=64, vfov_deg=120.0,
    )
    sample = pano.step(now=0.0)
    assert sample.image.shape == (128, 256, 3)
    assert sample.image.dtype == np.uint8
    pano.close()


def test_pano_centre_pixel_sees_red_wall_in_front(room_model_data) -> None:
    """Forward face renders the +X wall (red, rgba='1 0 0 1') so the
    output centre column samples should be predominantly red."""
    from vector_os_nano.hardware.sim.sensors import MuJoCoPano360
    pano = MuJoCoPano360(
        *room_model_data,
        out_w=256, out_h=128, face_size=64, vfov_deg=120.0,
        offset=(0.0, 0.0, 0.0),
    )
    sample = pano.step(now=0.0)
    centre = sample.image[64, 128]    # row mid, col mid
    assert centre[0] > centre[1]      # R > G
    assert centre[0] > centre[2]      # R > B
    pano.close()


def test_pano_rate_limit_caches(room_model_data) -> None:
    from vector_os_nano.hardware.sim.sensors import MuJoCoPano360
    pano = MuJoCoPano360(
        *room_model_data,
        out_w=128, out_h=64, face_size=32, rate_hz=2.0,
    )
    first = pano.step(now=0.0)
    second = pano.step(now=0.1)       # 100 ms < 500 ms
    assert second is first
    pano.close()


# ---------------------------------------------------------------------------
# Ground-truth odometry against a translated body
# ---------------------------------------------------------------------------


def test_gt_odom_after_simulated_translation(room_model_data) -> None:
    """Translate the trunk 0.5 m in 1 s → linear vx ≈ 0.5 m/s."""
    import mujoco
    from vector_os_nano.hardware.sim.sensors import GroundTruthOdomPublisher

    model, data = room_model_data
    # Reset trunk to known initial pose
    data.qpos[0:3] = [0.0, 0.0, 0.5]
    mujoco.mj_forward(model, data)

    pub = GroundTruthOdomPublisher(model, data, rate_hz=1e9)
    pub.step(now=0.0)

    # Translate
    data.qpos[0:3] = [0.5, 0.0, 0.5]
    mujoco.mj_forward(model, data)
    sample = pub.step(now=1.0)

    assert sample.position == (0.5, 0.0, 0.5)
    vx, vy, vz = sample.linear_twist
    assert vx == pytest.approx(0.5, rel=0.05)
    assert vy == pytest.approx(0.0, abs=1e-6)
    assert vz == pytest.approx(0.0, abs=1e-6)


def test_gt_odom_orientation_round_trip(room_model_data) -> None:
    """Set body quaternion to a known yaw rotation → ROS xyzw matches."""
    import mujoco
    from vector_os_nano.hardware.sim.sensors import GroundTruthOdomPublisher

    model, data = room_model_data
    # 90° yaw: wxyz = (cos(45°), 0, 0, sin(45°))
    half = math.cos(math.pi / 4)
    data.qpos[3:7] = [half, 0.0, 0.0, half]      # (w, x, y, z)
    mujoco.mj_forward(model, data)

    pub = GroundTruthOdomPublisher(model, data)
    sample = pub.step(now=0.0)
    qx, qy, qz, qw = sample.orientation

    assert qx == pytest.approx(0.0, abs=1e-6)
    assert qy == pytest.approx(0.0, abs=1e-6)
    assert qz == pytest.approx(half, abs=1e-6)
    assert qw == pytest.approx(half, abs=1e-6)


# ---------------------------------------------------------------------------
# All three sensors share one model — coexistence smoke
# ---------------------------------------------------------------------------


def test_all_three_sensors_can_step_in_same_model(room_model_data) -> None:
    """Construct lidar + pano + odom for the same model and step them
    once each; nothing crashes, nothing else interferes with the other."""
    import mujoco

    from vector_os_nano.hardware.sim.sensors import (
        GroundTruthOdomPublisher,
        MuJoCoLivox360,
        MuJoCoPano360,
    )

    model, data = room_model_data
    # Reset trunk to canonical pose (other module-scoped tests may have
    # left it translated/rotated)
    data.qpos[0:3] = [0.0, 0.0, 0.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    lidar = MuJoCoLivox360(
        model, data, h_resolution=72, v_layers=4, max_range=10.0,
    )
    pano = MuJoCoPano360(model, data, out_w=128, out_h=64, face_size=32)
    odom = GroundTruthOdomPublisher(model, data)

    lidar_sample = lidar.step(now=0.0)
    pano_sample = pano.step(now=0.0)
    odom_sample = odom.step(now=0.0)

    assert lidar_sample.points.shape[0] > 0
    assert pano_sample.image.shape == (64, 128, 3)
    assert odom_sample.position == (0.0, 0.0, 0.5)

    pano.close()
