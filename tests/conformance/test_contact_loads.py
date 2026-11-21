"""Force direction and scale are checked from independent momentum/gravity, on both engines."""
import numpy as np
import pytest
from types import SimpleNamespace

from vector_os_nano.sim.scene import compiled_scene

pytestmark = pytest.mark.sim


def _force(world, body):
    force = np.zeros(3)
    for contact in world.contact_loads():
        if contact.body_b == body:
            force += contact.force_on_b
        elif contact.body_a == body:
            force -= contact.force_on_b
    return force


def test_solver_load_support_and_friction_direction(backend, load, tmp_path):
    path = tmp_path/'load.xml'
    path.write_text('''<mujoco><option timestep=".001" integrator="Euler"/>
      <worldbody><geom type="plane" size="3 3 .1" friction="1 .005 .0001"/>
      <body name="box" pos="0 0 .05"><freejoint/>
      <geom type="box" size=".05 .05 .05" mass="2" friction="1 .005 .0001"/>
      </body></worldbody></mujoco>''')
    scene = compiled_scene(path)
    world = load(backend, scene)
    body = scene.body('box')
    assert world.contact_loads() == ()
    world.step(1000)
    np.testing.assert_allclose(_force(world, body), [0, 0, 2*9.81], atol=.01)
    from vector_os_nano.sim.drivers.gripper import Gripper
    sensor = object.__new__(Gripper)
    sensor._spec = SimpleNamespace(force_feedback=True)
    sensor._grasp = SimpleNamespace(finger_bodies=(body,))
    sensor._rt = SimpleNamespace(call=lambda fn: fn(world))
    reading = sensor.get_finger_load()
    np.testing.assert_allclose(reading.force, [0, 0, 2*9.81], atol=.01)
    assert reading.sim_time == world.time()
    before = world.state()
    assert all({c.body_a, c.body_b} == {0, body} for c in world.contact_loads())
    for c in world.contact_loads():
        np.testing.assert_allclose(c.normal_on_b, [0, 0, 1 if c.body_b == body else -1], atol=1e-5)
    np.testing.assert_array_equal(world.state()[0], before[0])
    q, velocity = before
    velocity[0] = .5
    world.reset(q, velocity)
    assert world.contact_loads() == ()
    world.step()
    acceleration = (world.state()[1][:3]-velocity[:3])/scene.timestep
    np.testing.assert_allclose(_force(world, body), 2*(acceleration+[0, 0, 9.81]), atol=.03)
    assert _force(world, body)[0] < 0
    q[2] = .0505  # Inside the proximity threshold but not carrying load.
    world.reset(q)
    assert 0 in world.contacts(body) and not world.contact_loads()
    world.step()
    np.testing.assert_array_equal(_force(world, body), np.zeros(3))
