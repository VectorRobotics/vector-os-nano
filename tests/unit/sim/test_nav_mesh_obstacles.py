"""The shared planner must see mesh furniture while leaving overhead clearance."""
import pytest
from vector_os_nano.sim.nav import vgraph as vg
from vector_os_nano.sim.scene.compiled import compiled_scene

pytestmark = pytest.mark.sim


def test_route_avoids_mesh_furniture_and_passes_below_overhead_mesh(tmp_path):
    xml = tmp_path/'mesh_world.xml'
    xml.write_text('''<mujoco><asset><mesh name="block" vertex="
      -.3 -.3 -.3  -.3 -.3 .3  -.3 .3 -.3  -.3 .3 .3
       .3 -.3 -.3   .3 -.3 .3   .3 .3 -.3   .3 .3 .3"/></asset>
      <worldbody>
        <body name="robot" pos="0 0 .75"><geom type="box" size=".2 .2 .75"/></body>
        <geom name="furniture" type="mesh" mesh="block" pos="2 0 .3" euler="0 0 31"/>
        <geom name="overhead" type="mesh" mesh="block" pos="4 0 3"/>
        <body name="cart" pos="6 0 .3"><freejoint/><geom type="box" size=".1 .1 .1"/>
          <body name="cart_part"><geom type="mesh" mesh="block"/></body></body>
      </worldbody></mujoco>''')
    scene = compiled_scene(xml, robot_root='robot')
    obstacles = vg.scene_obstacles(scene, scene.model.qpos0)
    assert vg.point_in_any((2,0), obstacles)
    assert not vg.point_in_any((4,0), obstacles)
    assert not vg.point_in_any((6,0), obstacles), 'A movable cart is not part of a fixed map'
    route, distance = vg.plan_path((0,0), (4,0), obstacles, .25)
    assert route is not None and distance > 4.05
