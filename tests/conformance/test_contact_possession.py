"""Instrument validation: a force-limited fixture holds then releases a real free body."""
from types import SimpleNamespace

import pytest

from vector_os_nano.interfaces.physics import GraspSpec
from vector_os_nano.schemas.robot import GripperSpec
from vector_os_nano.sim.ground_truth import GroundTruth
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.vcli.worlds.sim_oracles import build_namespace

pytestmark = pytest.mark.sim


def test_force_only_possession_and_release_on_both_engines(backend, load, tmp_path):
    path = tmp_path/'grip.xml'
    path.write_text('''<mujoco><option timestep=".001" integrator="Euler"/>
    <default><geom friction="1 .005 .0001" solref=".004 1"/></default>
    <worldbody><geom type="plane" size="3 3 .1"/>
      <body name="hand" pos="0 0 .4">
        <body name="left" pos="-.03 0 0"><joint name="left" type="slide" axis="-1 0 0" range="-.01 .03"/>
          <geom type="box" size=".005 .04 .06" mass=".1"/></body>
        <body name="right" pos=".03 0 0"><joint name="right" type="slide" axis="1 0 0" range="-.01 .03"/>
          <geom type="box" size=".005 .04 .06" mass=".1"/></body>
      </body>
      <body name="box" pos="0 0 .4"><freejoint/><geom type="box" size=".025 .025 .025" mass=".05"/></body>
    </worldbody><actuator>
      <position joint="left" kp="1000" kv="2" forcerange="-5 5"/>
      <position joint="right" kp="1000" kv="2" forcerange="-5 5"/>
    </actuator></mujoco>''')
    scene = compiled_scene(path, robot_root='hand')
    w = load(backend, scene)
    rt = SimRuntime(w, substeps=5, threaded=False, paced=False)
    try:
        spec = GraspSpec(scene.body('hand'), (scene.body('left'), scene.body('right')),
                         scene.joint('left'), -.004, .001, assist=False)
        rt.watch_contact_grip(spec)
        gt = GroundTruth(rt)
        gripper = GripperSpec('left', .03, -.004, ('left', 'right'), 'hand', .001, assist=False)
        cfg = SimpleNamespace(arm=SimpleNamespace(joints=('left',), home=(0,), gripper=gripper), camera=None)
        ns = build_namespace(gt, None, cfg, turn_start=lambda: (0, 0))
        joints = [scene.joint('left'), scene.joint('right')]
        rt.call(lambda world: world.set_targets(joints, [-.004, -.004]))
        assert not ns['holding_object']('box')
        rt.wait_sim(.5)
        assert not w.attached()
        assert ns['holding_object']('box') and ns['holding_object']()
        assert ns['lifted']('box') and ns['kept_holding']('box')
        assert not ns['released']()
        rt.call(lambda world: world.set_targets(joints, [.03, .03]))
        rt.wait_sim(.4)
        assert not ns['holding_object']('box') and not ns['kept_holding']('box')
        assert ns['released']('box') and ns['released']()
        assert gt.object_positions()['box'][2] < .1
    finally:
        rt.close()
