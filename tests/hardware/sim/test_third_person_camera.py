"""The chase camera stays oblique and orbits out of a rear wall."""
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('wall', [None, -.5, -1.])
def test_chase_view_preserves_robot_and_clears_wall(tmp_path, wall):
    path=tmp_path/'camera.xml'
    obstacle=f'<geom type="box" pos="{wall} 0 1.4" size=".08 3 1.4"/>' if wall else ''
    path.write_text('<mujoco><worldbody>'+obstacle+'<body name="robot"><freejoint/>'
                    '<geom type="box" pos="0 0 .4" size=".3 .3 .4"/>'
                    '</body></worldbody></mujoco>')
    world=MuJoCoWorld(compiled_scene(path,robot_root='robot'))
    cam=mj.MjvCamera();cam.elevation=-25.
    world._viewer=SimpleNamespace(cam=cam,close=lambda:None)
    try:
        world._follow_camera()
        assert cam.distance>=2. and cam.elevation==-25.
        a,e=np.radians([cam.azimuth,cam.elevation])
        outward=np.array([-np.cos(a)*np.cos(e),-np.sin(a)*np.cos(e),-np.sin(e)])
        position=cam.lookat+cam.distance*outward
        assert .8<position[2]<2.5
        if wall:
            assert position[0]>wall+.08 and abs(cam.azimuth)>25.
        else:
            assert cam.azimuth==25.
        world.data.qpos[1]=1.;mj.mj_forward(world.model,world.data)
        world._follow_camera()
        assert cam.lookat[1]==1.
    finally:
        world.close()
