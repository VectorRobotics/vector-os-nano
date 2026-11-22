"""A reachable half-turn must not be a zero-error stationary point of IK."""
import numpy as np
import pytest

from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.sim.scene.kinematics import ArmKinematics

pytestmark=pytest.mark.sim


@pytest.mark.parametrize('axis', [(0, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize('limit', [.5, 3.2])
def test_half_turn_reaches_only_when_joint_limits_allow_it(tmp_path, axis, limit):
    path=tmp_path/'hinge.xml'
    path.write_text(f'''<mujoco><compiler angle="radian"/><worldbody><body name="base">
      <body name="arm"><joint name="hinge" axis="{' '.join(map(str,axis))}" range="{-limit} {limit}"/>
      <geom type="sphere" size=".02" mass=".1"/><site name="tcp"/></body>
    </body></worldbody></mujoco>''')
    scene=compiled_scene(path,robot_root='base')
    kin=ArmKinematics(scene,['hinge'],'tcp');q=scene.model.qpos0.copy()
    n=np.asarray(axis)/np.linalg.norm(axis);target=2*np.outer(n,n)-np.eye(3)
    solved=kin.ik(q,[0,0,0],[0],target_rotation=target)
    if limit<np.pi:
        assert solved is None
    else:
        assert solved is not None
        pos,rotation=kin.fk(q,solved)
        np.testing.assert_allclose(pos,[0,0,0],atol=1e-8)
        angle=np.arccos(np.clip((np.trace(target@np.asarray(rotation).T)-1)/2,-1,1))
        assert angle<.02
        assert abs(solved[0])<=limit
