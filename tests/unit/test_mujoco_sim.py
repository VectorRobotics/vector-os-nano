"""Tests for MuJoCo simulation backend.

Tests MuJoCoArm and MuJoCoGripper against ArmProtocol / GripperProtocol
contracts. Requires mujoco to be installed.
"""
from __future__ import annotations

import math
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arm():
    """Connected MuJoCoArm instance (headless)."""
    a = MuJoCoArm(gui=False)
    a.connect()
    yield a
    a.disconnect()


@pytest.fixture
def gripper(arm):
    """MuJoCoGripper attached to the arm fixture."""
    return MuJoCoGripper(arm)


# ---------------------------------------------------------------------------
# MuJoCoArm — ArmProtocol tests
# ---------------------------------------------------------------------------

class TestMuJoCoArmProperties:
    def test_name(self, arm: MuJoCoArm):
        assert arm.name == "mujoco_so101"

    def test_joint_names(self, arm: MuJoCoArm):
        assert arm.joint_names == [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll",
        ]

    def test_dof(self, arm: MuJoCoArm):
        assert arm.dof == 5


class TestMuJoCoArmConnection:
    def test_connect_disconnect(self):
        a = MuJoCoArm(gui=False)
        a.connect()
        assert a._connected
        a.disconnect()
        assert not a._connected

    def test_double_disconnect(self):
        a = MuJoCoArm(gui=False)
        a.connect()
        a.disconnect()
        a.disconnect()  # should not raise

    def test_require_connection(self):
        a = MuJoCoArm(gui=False)
        with pytest.raises(RuntimeError, match="not connected"):
            a.get_joint_positions()


class TestMuJoCoArmJoints:
    def test_get_joint_positions_length(self, arm: MuJoCoArm):
        joints = arm.get_joint_positions()
        assert len(joints) == 5

    def test_get_joint_positions_zero(self, arm: MuJoCoArm):
        joints = arm.get_joint_positions()
        for j in joints:
            assert abs(j) < 0.1  # near zero at init

    def test_move_joints_wrong_length(self, arm: MuJoCoArm):
        with pytest.raises(ValueError, match="expected 5"):
            arm.move_joints([0.0, 0.0])


class TestMuJoCoArmMotion:
    def test_move_joints_tracks(self, arm: MuJoCoArm):
        target = [0.3, -0.3, 0.2, 0.1, -0.1]
        arm.move_joints(target, duration=4.0)
        actual = arm.get_joint_positions()
        for a, t in zip(actual, target):
            assert abs(a - t) < 0.05  # within ~3 degrees

    def test_move_cartesian_ik(self, arm: MuJoCoArm):
        result = arm.move_cartesian((0.25, 0.0, 0.12), duration=2.0)
        assert result is True

    def test_move_cartesian_unreachable(self, arm: MuJoCoArm):
        result = arm.move_cartesian((10.0, 10.0, 10.0), duration=1.0)
        assert result is False

    def test_stop(self, arm: MuJoCoArm):
        arm.move_joints([0.3, 0.0, 0.0, 0.0, 0.0], duration=0.5)
        arm.stop()
        j1 = arm.get_joint_positions()
        arm.step(100)
        j2 = arm.get_joint_positions()
        # After stop, joints shouldn't drift much
        for a, b in zip(j1, j2):
            assert abs(a - b) < 0.05


class TestMuJoCoArmFK:
    def test_fk_at_zero(self, arm: MuJoCoArm):
        pos, rot = arm.fk([0.0] * 5)
        assert len(pos) == 3
        assert len(rot) == 3
        assert len(rot[0]) == 3
        # EE should be roughly (0.39, 0, 0.25) at zero config
        assert 0.3 < pos[0] < 0.5
        assert abs(pos[1]) < 0.01
        assert 0.15 < pos[2] < 0.35

    def test_fk_does_not_change_state(self, arm: MuJoCoArm):
        original = arm.get_joint_positions()
        arm.fk([0.5, 0.5, 0.5, 0.5, 0.5])
        after = arm.get_joint_positions()
        for a, b in zip(original, after):
            assert abs(a - b) < 1e-6


class TestMuJoCoArmIK:
    def test_ik_reachable(self, arm: MuJoCoArm):
        target = (0.25, 0.0, 0.10)
        solution = arm.ik(target)
        assert solution is not None
        assert len(solution) == 5

    def test_ik_accuracy(self, arm: MuJoCoArm):
        target = (0.22, 0.05, 0.08)
        solution = arm.ik(target)
        assert solution is not None
        pos, _ = arm.fk(solution)
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, target)))
        assert err < 0.003  # within 3mm

    def test_ik_unreachable(self, arm: MuJoCoArm):
        solution = arm.ik((5.0, 5.0, 5.0))
        assert solution is None


class TestMuJoCoArmScene:
    def test_get_object_positions(self, arm: MuJoCoArm):
        objs = arm.get_object_positions()
        assert len(objs) >= 6
        assert "mug" in objs
        assert "banana" in objs
        for name, pos in objs.items():
            assert len(pos) == 3

    def test_render(self, arm: MuJoCoArm):
        img = arm.render()
        assert img is not None
        assert img.shape == (480, 640, 3)

    def test_render_different_camera(self, arm: MuJoCoArm):
        img = arm.render(camera_name="front", width=320, height=240)
        assert img is not None
        assert img.shape == (240, 320, 3)


# ---------------------------------------------------------------------------
# MuJoCoGripper — GripperProtocol tests
# ---------------------------------------------------------------------------

class TestMuJoCoGripper:
    def test_open(self, gripper: MuJoCoGripper):
        result = gripper.open()
        assert result is True
        assert gripper.get_position() > 0.5

    def test_close(self, gripper: MuJoCoGripper):
        result = gripper.close()
        assert result is True
        assert gripper.get_position() < 0.5

    def test_is_holding_when_open(self, gripper: MuJoCoGripper):
        gripper.open()
        assert gripper.is_holding() is False

    def test_get_force(self, gripper: MuJoCoGripper):
        force = gripper.get_force()
        assert force is not None
        assert isinstance(force, float)

    def test_open_close_cycle(self, gripper: MuJoCoGripper):
        gripper.open()
        pos_open = gripper.get_position()
        gripper.close()
        pos_closed = gripper.get_position()
        assert pos_open > pos_closed


class TestMuJoCoPickAndPlace:
    """Integration test: full pick-and-place in MuJoCo simulation."""

    @pytest.mark.skip(reason="Timing-sensitive; pick verified manually — see test_mujoco_sim_manual.py")
    def test_pick_cube(self, arm: MuJoCoArm, gripper: MuJoCoGripper):
        """Pick up the red cube and verify it lifts off the table."""
        # Let objects settle on the table first
        arm.step(500)
        objs = arm.get_object_positions()
        red = objs["red_cube"]
        original_z = red[2]  # ~0.035 (on table)

        # Open gripper, move above, descend — smooth incremental motion
        gripper.open()

        def _smooth_move(target, steps=8, dur=0.5):
            cur, _ = arm.fk(arm.get_joint_positions())
            for i in range(1, steps + 1):
                t = i / steps
                pt = tuple(c + t * (g - c) for c, g in zip(cur, target))
                ik = arm.ik(pt)
                if ik:
                    arm.move_joints(ik, duration=dur)
                    arm.step(50)

        _smooth_move((red[0], red[1], red[2] + 0.08))
        _smooth_move((red[0], red[1], red[2] - 0.02))

        # Close and lift incrementally
        gripper.close()
        arm.step(200)

        ee_pos, _ = arm.fk(arm.get_joint_positions())
        for step in range(12):
            target_z = ee_pos[2] + (step + 1) * 0.01
            ik = arm.ik((ee_pos[0], ee_pos[1], target_z))
            if ik:
                arm.move_joints(ik, duration=0.5)
                arm.step(80)

        new_objs = arm.get_object_positions()
        new_z = new_objs["red_cube"][2]
        assert new_z > original_z + 0.03, (
            f"Cube should have lifted >3cm, got delta={new_z - original_z:.3f}m"
        )


# ---------------------------------------------------------------------------
# TestMuJoCoArmPhysics — physical realism checks
# ---------------------------------------------------------------------------

class TestMuJoCoArmPhysics:
    def test_gravity_settles_objects(self, arm: MuJoCoArm):
        """After 500 steps all objects should have landed on the table (not through floor, not floating)."""
        arm.step(500)
        objs = arm.get_object_positions()
        assert len(objs) >= 6, "Expected at least 6 scene objects"
        for name, pos in objs.items():
            z = pos[2]
            assert z > 0, f"Object '{name}' fell through floor: z={z:.4f}"
            assert z < 0.2, f"Object '{name}' still floating: z={z:.4f}"

    def test_joint_limits_respected(self, arm: MuJoCoArm):
        """Commanding extreme targets should clamp to joint limits, not exceed pi."""
        arm.move_joints([3.0] * 5, duration=2.0)
        actual = arm.get_joint_positions()
        for i, q in enumerate(actual):
            assert abs(q) < math.pi, (
                f"Joint {i} exceeded pi: {q:.4f} rad"
            )

    def test_objects_dont_fall_through(self, arm: MuJoCoArm):
        """Every free-body object must stay above the floor (z > 0) after settling."""
        arm.step(500)
        objs = arm.get_object_positions()
        for name, pos in objs.items():
            assert pos[2] > 0, f"Object '{name}' has z={pos[2]:.4f} (below floor)"


# ---------------------------------------------------------------------------
# TestMuJoCoGripperWeld — grasp verification via weld constraints
# ---------------------------------------------------------------------------

class TestMuJoCoGripperWeld:
    def test_grasp_near_object(self, arm: MuJoCoArm, gripper: MuJoCoGripper):
        """Moving EE within grasp radius of banana then closing should register a hold."""
        # Let objects settle first
        arm.step(200)
        objs = arm.get_object_positions()
        banana_pos = objs["banana"]

        # Use IK to get EE close to the banana
        target = (banana_pos[0], banana_pos[1], banana_pos[2] + 0.02)
        solution = arm.ik(target)
        if solution is not None:
            arm.move_joints(solution, duration=2.0)
            arm.step(100)

        gripper.close()
        assert gripper.is_holding() is True

    def test_grasp_far_from_object(self, arm: MuJoCoArm, gripper: MuJoCoGripper):
        """Closing from the home position (far from all objects) should not register a hold."""
        # Ensure we are at home (joints near zero — default)
        arm.step(50)
        gripper.close()
        assert gripper.is_holding() is False

    def test_release_after_grasp(self, arm: MuJoCoArm, gripper: MuJoCoGripper):
        """Opening after a successful grasp should release the object."""
        # Move near banana and grasp
        arm.step(200)
        objs = arm.get_object_positions()
        banana_pos = objs["banana"]
        target = (banana_pos[0], banana_pos[1], banana_pos[2] + 0.02)
        solution = arm.ik(target)
        if solution is not None:
            arm.move_joints(solution, duration=2.0)
            arm.step(100)
        gripper.close()

        if not gripper.is_holding():
            pytest.skip("Grasp did not succeed — skipping release check")

        gripper.open()
        assert gripper.is_holding() is False

    @pytest.mark.skip(reason="Timing-sensitive — object follow requires many physics steps")
    def test_held_object_follows_arm(self, arm: MuJoCoArm, gripper: MuJoCoGripper):
        """After grasping, lifting the arm should increase the held object's z."""
        arm.step(200)
        objs = arm.get_object_positions()
        banana_pos = objs["banana"]

        # Move to banana and grasp
        target = (banana_pos[0], banana_pos[1], banana_pos[2] + 0.02)
        solution = arm.ik(target)
        if solution is not None:
            arm.move_joints(solution, duration=2.0)
            arm.step(100)
        gripper.close()
        arm.step(200)

        # Record z before lift
        pre_lift_z = arm.get_object_positions()["banana"][2]

        # Lift EE by 0.08 m
        cur_pos, _ = arm.fk(arm.get_joint_positions())
        lift_target = (cur_pos[0], cur_pos[1], cur_pos[2] + 0.08)
        lift_solution = arm.ik(lift_target)
        if lift_solution is not None:
            arm.move_joints(lift_solution, duration=2.0)
            arm.step(200)

        post_lift_z = arm.get_object_positions()["banana"][2]
        assert post_lift_z > pre_lift_z + 0.03, (
            f"Object z did not increase after lift: pre={pre_lift_z:.4f}, post={post_lift_z:.4f}"
        )
