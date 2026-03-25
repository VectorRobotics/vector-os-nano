"""Tests for MuJoCoGo2 — Go2 quadruped in MuJoCo simulation."""
import pytest

# Skip if convex_mpc not installed
pytest.importorskip("convex_mpc")


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
