"""Containment follows the moving drawer and checks corners, not just a centre."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from vector_os_nano.schemas.world import Surface, SurfaceAnchor
from vector_os_nano.sim.scene.compiled import compiled_scene
from vector_os_nano.sim.ground_truth import GroundTruth
from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld
from vector_os_nano.sim.backends.pybullet.world import PyBulletWorld
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.scene.canonical import relative_bounds

pytestmark=pytest.mark.sim


def test_rotated_drawer_motion_and_protruding_package(tmp_path):
    xml=tmp_path/'bins.xml'
    xml.write_text('''<mujoco><compiler angle="radian"/><worldbody>
      <body name="cart" pos="1 2 .5" euler="0 0 1.5707963267948966"><freejoint/><geom size=".1"/>
        <body name="drawer" pos=".1 0 0"><joint name="slide" type="slide" axis="1 0 0" range="0 .3"/>
          <geom type="box" size=".1 .1 .01" pos="0 0 -.01"/>
        </body>
      </body>
      <body name="package" pos="1 2.15 .55"><freejoint/><geom type="box" size=".04 .03 .02"/></body>
    </worldbody></mujoco>''')
    scene=compiled_scene(xml);q=scene.model.qpos0.copy()
    def bounds():return relative_bounds(scene,q,scene.body('package'),scene.body('drawer'))
    np.testing.assert_allclose(bounds(),[[.02,-.04,.03],[.08,.04,.07]],atol=1e-12)
    obj=scene.free_q(scene.body('package'));slide=scene.joint_q(scene.joint('slide'))
    q[obj.start+1]+=.04  # centre is still inside; the box's far edge protrudes
    assert bounds()[1][0]>.1
    q[obj.start+1]-=.04;q[slide]=.15
    assert bounds()[0][0]<-.1
    q[obj.start+1]+=.15
    np.testing.assert_allclose(bounds(),[[.02,-.04,.03],[.08,.04,.07]],atol=1e-12)
    q[obj.start+2]+=.04  # top protrudes even though the centre remains below the ceiling
    assert bounds()[1][2]>.1
    q[obj.start]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):bounds()


@pytest.mark.parametrize('backend',[MuJoCoWorld,PyBulletWorld])
def test_contained_box_needs_actual_floor_support(tmp_path,backend):
    xml=tmp_path/'support.xml'
    xml.write_text('''<mujoco><worldbody>
      <body name="tray" pos="1 2 .3"><geom type="box" size=".15 .15 .01" pos="0 0 -.01"/></body>
      <body name="package" pos="1 2 .38"><freejoint/><geom type="box" size=".03 .03 .02" mass=".1"/></body>
    </worldbody></mujoco>''')
    sc=compiled_scene(xml)
    rt=SimRuntime(backend(sc),threaded=False,paced=False,name='container_support')
    original_call=rt.call
    try:
        gt=GroundTruth(rt)
        surface=Surface('bin','tray',(-.15,-.15,.15,.15),0.,place_point=(0,0),
                        local_anchor=SurfaceAnchor('knob',(0,0,0),0,.06),ceiling=.15)
        assert not gt.supported_inside_surface('package',surface)
        rt.step(400)
        assert gt.supported_inside_surface('package',surface)
        # Reuse the real settled state, but make its independently supplied load lateral.
        from dataclasses import replace
        loads=rt.world.contact_loads();state=rt.world.state()
        rt.call=lambda f:f(NS(state=lambda:state,contact_loads=lambda:[
            replace(c,normal_on_b=(1.,0.,0.),force_on_b=(1.,0.,0.)) for c in loads]))
        assert not gt.supported_inside_surface('package',surface)
        rt.call=lambda f:f(NS(state=lambda:state,contact_loads=lambda:[
            replace(c,normal_on_b=(0.,0.,float('nan'))) for c in loads]))
        assert not gt.supported_inside_surface('package',surface)
    finally:
        rt.call=original_call
        rt.close()
