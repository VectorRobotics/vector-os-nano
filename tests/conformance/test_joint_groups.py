"""Shared scalar servo control and cancellation under both physics engines."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import pytest

from vector_os_nano.schemas.robot import JointGroupSpec
from vector_os_nano.sim.devices import joint_groups
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.scene import compiled_scene

pytestmark = pytest.mark.sim


def test_scalar_postures_and_cancel_hold_measured_state(backend, load, tmp_path):
    path = tmp_path/'groups.xml'
    path.write_text('''<mujoco><compiler angle="radian"/><option timestep=".001" gravity="0 0 0"/>
    <worldbody><body name="root"><geom type="box" size=".1 .1 .1"/>
      <body name="slider"><joint name="lift" type="slide" axis="1 0 0" range="-.3 .3"/>
        <geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
        <body name="head"><joint name="pan" axis="0 0 1" range="-1 1"/>
          <geom type="box" size=".02 .01 .01" mass="1" contype="0" conaffinity="0"/>
        </body></body></body></worldbody><actuator>
      <position joint="lift" kp="1000" kv="60" ctrlrange="-.3 .3" forcerange="-50 50"/>
      <position joint="pan" kp="1" kv=".01" ctrlrange="-1 1" forcerange="-1 1"/>
    </actuator></mujoco>''')
    scene = compiled_scene(path, robot_root='root')
    rt = SimRuntime(load(backend, scene), substeps=1, threaded=False, paced=False)
    token = threading.Event()
    spec = JointGroupSpec(('lift', 'pan'), {'work': (.2, .4)}, (.1, .2), (.2, .4), (.5, 1.), .003)
    cfg = SimpleNamespace(joint_groups={'assembly': spec}, path='fixture')
    try:
        group = joint_groups(cfg, rt, abort=token.is_set)['assembly']
        assert group.move_joints([.2, .4], duration=.1)
        assert group.get_joint_positions() == pytest.approx([.2, .4], abs=.003)
        wait = rt.wait_sim
        def cancel_after_step(*a, **kw):
            result = wait(*a, **kw)
            token.set()
            return result
        rt.wait_sim = cancel_after_step
        assert not group.move_joints([0., 0.])
        held = group.get_joint_positions()
        assert rt.call(lambda w: w.targets(group._joints)) == pytest.approx(held)
        rt.wait_sim = wait
        wait(.5)
        assert group.get_joint_positions() == pytest.approx(held, abs=.003)
        cfg.joint_groups['assembly'] = replace(spec, postures={'invalid': (.4, .4)})
        with pytest.raises(ValueError, match='outside model limits'):
            joint_groups(cfg, rt, abort=token.is_set)
    finally:
        rt.close()
