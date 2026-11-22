"""Experimental hardware contracts exercised on a deterministic serial bus only."""
from unittest.mock import MagicMock
import pytest
from vector_os_nano.hardware.so101.arm import SO101Arm
from vector_os_nano.hardware.so101.joint_config import JOINT_CONFIG
from vector_os_nano.hardware.so101.gripper import SO101Gripper


def arm_with_bus():
    arm = SO101Arm()
    bus = MagicMock()
    bus.connect.return_value = bus.write_position.return_value = bus.set_torque.return_value = True
    bus.read_position.return_value = 2000
    arm._bus = bus
    return arm, bus


@pytest.mark.parametrize('failure', ['read', 'write', 'torque'])
def test_startup_failure_rolls_back_without_enabling_an_unseated_motor(failure):
    arm, bus = arm_with_bus()
    if failure == 'read':
        bus.read_position.return_value = -1
    elif failure == 'write':
        bus.write_position.return_value = False
    else:
        bus.set_torque.side_effect = lambda motor, enable: not enable
    with pytest.raises(RuntimeError):
        arm.connect()
    assert not arm._connected
    bus.disconnect.assert_called_once()
    if failure != 'torque':
        assert not any(call.kwargs.get('enable') for call in bus.set_torque.call_args_list)


def test_repeated_connect_does_not_reenable_motors():
    arm, bus = arm_with_bus()
    arm.connect()
    arm.connect()
    assert bus.connect.call_count == 1


@pytest.mark.parametrize('positions,duration', [([float('nan')]*5, 1.), ([float('inf')]*5, 1.),
    ([100.]*5, 1.), ([0.]*5, -1.), ([0.]*5, float('nan'))])
def test_invalid_trajectory_is_rejected_before_bus_io(positions, duration):
    arm, bus = arm_with_bus(); arm._connected = True
    with pytest.raises(ValueError):
        arm.move_joints(positions, duration)
    bus.read_position.assert_not_called()
    bus.write_position.assert_not_called()


def test_stop_preempts_remaining_waypoints(monkeypatch):
    arm, bus = arm_with_bus(); arm._connected = True
    stopped = []
    def interrupt(_):
        if not stopped:
            arm.stop()
            stopped.append(bus.write_position.call_count)
    monkeypatch.setattr('vector_os_nano.hardware.so101.arm.time.sleep', interrupt)
    assert arm.move_joints([0.]*5, .1) is False
    assert bus.write_position.call_count == stopped[0]


def test_missing_fk_is_not_an_invented_origin():
    arm, _ = arm_with_bus()
    with pytest.raises(RuntimeError, match='IK solver'):
        arm.fk([0.]*5)


def test_encoder_failure_is_not_a_closed_gripper():
    _, bus = arm_with_bus(); bus.read_position.return_value = -1
    with pytest.raises(RuntimeError, match='encoder'):
        SO101Gripper(bus).get_position()


def test_encoder_cannot_identify_a_named_object():
    _, bus = arm_with_bus()
    grip = SO101Gripper(bus)
    assert grip.is_holding() is True
    assert grip.is_holding('banana') is False


def test_failed_stop_read_still_attempts_all_other_motors():
    arm, bus = arm_with_bus(); arm._connected = True
    bus.read_position.side_effect = [-1, 2000, 2000, 2000, 2000]
    with pytest.raises(RuntimeError, match='shoulder_pan'):
        arm.stop()
    assert bus.read_position.call_count == 5
    assert bus.write_position.call_count == 4


def test_concurrent_motion_is_refused_without_queueing_after_stop():
    arm, bus = arm_with_bus(); arm._connected = True
    with arm._motion_lock:
        assert arm.move_joints([0.]*5) is False
    bus.write_position.assert_not_called()
