"""Only declared proprioceptive fingertip forces cross the actor boundary."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from vector_os_nano.sim.drivers.gripper import Gripper


def test_finger_load_excludes_other_contacts_and_cancels_internal_pairs():
    contacts = [NS(body_a=2, body_b=7, force_on_b=(2., 0., .9)),
                NS(body_a=7, body_b=3, force_on_b=(2., 0., -.9)),
                NS(body_a=2, body_b=3, force_on_b=(10., 20., 30.)),
                NS(body_a=7, body_b=8, force_on_b=(100., 100., 100.))]
    world = NS(time=lambda: 3., contact_loads=lambda: tuple(contacts))
    grip = object.__new__(Gripper)
    grip._spec = NS(force_feedback=True)
    grip._grasp = NS(finger_bodies=(2, 3))
    grip._rt = NS(call=lambda fn: fn(world))
    reading = grip.get_finger_load()
    assert reading.sim_time == 3.
    assert reading.force == pytest.approx((0., 0., -1.8))
    assert set(vars(reading)) == {'sim_time', 'force'}


def test_undeclared_force_sensor_never_reads_contacts():
    grip = object.__new__(Gripper)
    grip._spec = NS(force_feedback=False)
    grip._rt = Mock()
    assert grip.get_finger_load() is None
    grip._rt.call.assert_not_called()
