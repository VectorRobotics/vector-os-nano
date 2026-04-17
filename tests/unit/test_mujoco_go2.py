"""Tests for MuJoCoGo2 — Go2 quadruped in MuJoCo simulation."""
import pytest

# Skip entire module if mujoco is not installed.
# convex_mpc tests are isolated in TestMuJoCoGo2ConvexMPC below.
pytest.importorskip("mujoco", reason="mujoco not installed")


def _has_convex_mpc() -> bool:
    try:
        import convex_mpc  # noqa: F401
        return True
    except ImportError:
        return False


class TestMuJoCoGo2Lifecycle:
    def test_connect_disconnect(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        assert go2._connected
        pos = go2.get_position()
        assert len(pos) == 3
        assert pos[2] > 0.1  # not on ground yet
        go2.disconnect()
        assert not go2._connected

    def test_get_heading(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        heading = go2.get_heading()
        assert isinstance(heading, float)
        go2.disconnect()

    def test_get_joint_positions(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        joints = go2.get_joint_positions()
        assert len(joints) == 12
        go2.disconnect()


class TestMuJoCoGo2ConvexMPC:
    @pytest.mark.skipif(not _has_convex_mpc(), reason="convex_mpc not installed")
    def test_imports(self):
        from convex_mpc.go2_robot_data import PinGo2Model
        from convex_mpc.mujoco_model import MuJoCo_GO2_Model
        go2 = PinGo2Model()
        assert go2.model.nq == 19
        mj_go2 = MuJoCo_GO2_Model()
        assert mj_go2.model.nu == 12


class TestMuJoCoGo2Posture:
    def test_stand(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        pos = go2.get_position()
        assert 0.2 < pos[2] < 0.4  # standing height ~0.27m
        joints = go2.get_joint_positions()
        assert len(joints) == 12
        go2.disconnect()

    def test_sit(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        stand_z = go2.get_position()[2]
        go2.sit()
        sit_z = go2.get_position()[2]
        assert sit_z < stand_z
        go2.disconnect()

    def test_pd_controller(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        import numpy as np
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        target = [0.0, 0.9, -1.8] * 4  # standing pose
        go2._pd_interpolate(np.array(target), duration=2.0)
        actual = go2.get_joint_positions()
        for t, a in zip(target, actual):
            assert abs(t - a) < 0.15, f"Joint error too large: target={t:.2f}, actual={a:.2f}"
        go2.disconnect()


class TestMuJoCoGo2Walk:
    def test_walk_forward(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        start_pos = go2.get_position()
        go2.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=2.0)
        end_pos = go2.get_position()
        displacement = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
        assert displacement > 0.1, f"Only moved {displacement:.3f}m in 2s"
        assert end_pos[2] > 0.15, f"Robot fell: z={end_pos[2]:.3f}"
        go2.disconnect()

    def test_walk_turn(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        start_heading = go2.get_heading()
        go2.walk(vx=0.0, vy=0.0, vyaw=0.5, duration=2.0)
        end_heading = go2.get_heading()
        delta = abs(end_heading - start_heading)
        assert delta > 0.3, f"Only turned {delta:.3f} rad"
        go2.disconnect()

    def test_walk_stability(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        go2.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=5.0)
        pos = go2.get_position()
        assert pos[2] > 0.15, f"Robot fell during 5s walk: z={pos[2]:.3f}"
        go2.disconnect()


class TestMuJoCoGo2HAL:
    """Tests for the new HAL interface (set_velocity, odometry, lidar)."""

    def test_name_property(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        assert go2.name == "mujoco_go2"
        go2.disconnect()

    def test_supports_holonomic(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        assert go2.supports_holonomic is True
        go2.disconnect()

    def test_supports_lidar(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        assert go2.supports_lidar is True
        go2.disconnect()

    def test_set_velocity_changes_position(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        import time
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        start = go2.get_position()
        go2.set_velocity(0.3, 0, 0)
        time.sleep(2.0)
        go2.set_velocity(0, 0, 0)
        time.sleep(0.2)  # let physics settle
        end = go2.get_position()
        dist = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
        assert dist > 0.1, f"Robot didn't move: dist={dist}"
        assert end[2] > 0.15, "Robot fell"
        go2.disconnect()

    def test_get_odometry(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        from vector_os_nano.core.types import Odometry
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        odom = go2.get_odometry()
        assert isinstance(odom, Odometry)
        assert odom.timestamp > 0
        # Standing robot should have qw close to 1
        assert abs(odom.qw) > 0.5
        go2.disconnect()

    def test_get_lidar_scan(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        from vector_os_nano.core.types import LaserScan
        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()
        scan = go2.get_lidar_scan()
        assert isinstance(scan, LaserScan)
        assert len(scan.ranges) == 360
        assert scan.range_max == 12.0
        # In the room scene, some rays should hit walls (finite range)
        finite_ranges = [r for r in scan.ranges if r < scan.range_max]
        assert len(finite_ranges) > 0, "No walls detected by lidar"
        go2.disconnect()

    def test_satisfies_base_protocol(self):
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        from vector_os_nano.hardware.base import BaseProtocol
        go2 = MuJoCoGo2(gui=False)
        # Check protocol satisfaction (structural typing)
        assert isinstance(go2, BaseProtocol)


# ---------------------------------------------------------------------------
# New test classes — sinusoidal backend (always available when mujoco present)
# ---------------------------------------------------------------------------

import time
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2


class TestMuJoCoGo2SpeedCommand:
    """Core velocity command tests using the sinusoidal backend."""

    def test_set_velocity_forward(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            start = go2.get_position()
            go2.set_velocity(0.3, 0, 0)
            time.sleep(2.0)
            go2.set_velocity(0, 0, 0)
            end = go2.get_position()
            displacement = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            assert displacement > 0.1, f"Forward displacement too small: {displacement:.3f}m"
        finally:
            go2.disconnect()

    def test_set_velocity_lateral(self):
        """Lateral vy command — sinusoidal gait hip abduction is small,
        so we only verify the robot stays upright (doesn't crash)."""
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            go2.set_velocity(0, 0.2, 0)
            time.sleep(2.0)
            go2.set_velocity(0, 0, 0)
            pos = go2.get_position()
            assert pos[2] > 0.15, f"Robot fell during lateral walk: z={pos[2]:.3f}"
        finally:
            go2.disconnect()

    def test_set_velocity_turn(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            start_heading = go2.get_heading()
            go2.set_velocity(0, 0, 0.5)
            time.sleep(2.0)
            go2.set_velocity(0, 0, 0)
            end_heading = go2.get_heading()
            # Heading change can wrap; use absolute difference or angle distance
            delta = abs(end_heading - start_heading)
            # Wrap into [0, pi] range
            if delta > 3.14159:
                delta = abs(delta - 2 * 3.14159)
            assert delta > 0.3, f"Heading change too small: {delta:.3f} rad"
        finally:
            go2.disconnect()

    def test_set_velocity_stop(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            go2.set_velocity(0.3, 0, 0)
            time.sleep(1.0)
            go2.set_velocity(0, 0, 0)
            time.sleep(0.5)
            vel = go2.get_velocity()
            speed = (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5
            assert speed < 0.1, f"Robot still moving after stop: speed={speed:.3f} m/s"
        finally:
            go2.disconnect()


class TestMuJoCoGo2StateQueries:
    """Sensor and state interface tests."""

    def test_get_position_3d(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            pos = go2.get_position()
            assert len(pos) == 3
            assert all(isinstance(v, float) for v in pos)
        finally:
            go2.disconnect()

    def test_get_heading_float(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            heading = go2.get_heading()
            assert isinstance(heading, float)
        finally:
            go2.disconnect()

    def test_get_velocity_3d(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            vel = go2.get_velocity()
            assert len(vel) == 3
            assert all(isinstance(v, float) for v in vel)
        finally:
            go2.disconnect()

    def test_get_joint_positions_12(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            joints = go2.get_joint_positions()
            assert len(joints) == 12
            assert all(isinstance(v, float) for v in joints)
        finally:
            go2.disconnect()

    def test_odometry_timestamp_advances(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            odom1 = go2.get_odometry()
            time.sleep(0.1)
            odom2 = go2.get_odometry()
            assert odom2.timestamp > odom1.timestamp, (
                f"Timestamp did not advance: {odom1.timestamp} -> {odom2.timestamp}"
            )
        finally:
            go2.disconnect()

    def test_lidar_scan_structure(self):
        go2 = MuJoCoGo2(gui=False, room=True)  # needs walls for ray hits
        go2.connect()
        try:
            go2.stand()
            scan = go2.get_lidar_scan()
            assert len(scan.ranges) == 360, f"Expected 360 ranges, got {len(scan.ranges)}"
        finally:
            go2.disconnect()


class TestMuJoCoGo2LifecycleEdgeCases:
    """Edge case lifecycle tests."""

    def test_double_disconnect(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        go2.disconnect()
        # Second disconnect must not raise
        go2.disconnect()

    def test_require_connection_raises(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        # Not connected — get_position() must raise RuntimeError
        with pytest.raises(RuntimeError):
            go2.get_position()

    def test_velocity_clipping(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.set_velocity(100.0, 100.0, 100.0)
            with go2._cmd_lock:
                vx, vy, vyaw = go2._cmd_vel
            assert abs(vx) <= 0.8, f"vx not clipped: {vx}"
            assert abs(vy) <= 0.4, f"vy not clipped: {vy}"
            assert abs(vyaw) <= 4.0, f"vyaw not clipped: {vyaw}"
        finally:
            go2.disconnect()


class TestMuJoCoGo2ResetPose:
    """reset_pose() behaviour tests."""

    def test_reset_preserves_xy(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            go2.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=1.5)
            before = go2.get_position()
            go2.reset_pose()
            after = go2.get_position()
            assert abs(after[0] - before[0]) < 0.05, (
                f"x changed too much after reset: {before[0]:.3f} -> {after[0]:.3f}"
            )
            assert abs(after[1] - before[1]) < 0.05, (
                f"y changed too much after reset: {before[1]:.3f} -> {after[1]:.3f}"
            )
        finally:
            go2.disconnect()

    def test_reset_restores_height(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            go2.reset_pose()
            z = go2.get_position()[2]
            assert abs(z - 0.35) < 0.05, f"Height after reset unexpected: z={z:.3f}"
        finally:
            go2.disconnect()

    def test_reset_zeros_velocity(self):
        go2 = MuJoCoGo2(gui=False, room=False)
        go2.connect()
        try:
            go2.stand()
            go2.set_velocity(0.3, 0, 0)
            time.sleep(0.5)
            go2.reset_pose()
            vel = go2.get_velocity()
            speed = (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5
            assert speed < 0.1, f"Velocity not zeroed after reset: speed={speed:.3f} m/s"
        finally:
            go2.disconnect()
