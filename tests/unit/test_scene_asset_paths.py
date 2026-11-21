"""Textured worlds must survive serialization to a different cache directory."""
from pathlib import Path

import mujoco as mj
import pytest
from PIL import Image

from vector_os_nano.sim.scene.compose import robot_spec


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("cube", [False, True])
def test_texture_survives_cached_scene(tmp_path: Path, absolute: bool, cube: bool):
    source = tmp_path / "source"
    textures = source / "textures"
    textures.mkdir(parents=True)
    image = textures / "floor.png"
    Image.new("RGB", (4, 4), (80, 120, 160)).save(image)
    model = source / "world.xml"
    file = image if absolute else 'floor.png'
    attributes = ('type="cube" ' + ' '.join(f'file{face}="{file}"' for face in
                  ('right', 'left', 'up', 'down', 'front', 'back')) if cube else f'type="2d" file="{file}"')
    model.write_text(f'''<mujoco>
      <compiler texturedir="textures"/>
      <asset><texture name="albedo" {attributes}/>
      <material name="floor" texture="albedo"/></asset>
      <worldbody><geom type="box" size="1 1 .1" material="floor"/></worldbody>
    </mujoco>''')
    spec = robot_spec(model)
    spec.compile()
    cache = tmp_path / "cache"
    cache.mkdir()
    scene = cache / "world.xml"
    scene.write_text(spec.to_xml())
    restored = mj.MjModel.from_xml_path(str(scene))
    assert restored.ntex == 1
    assert tuple(restored.tex_data[:3]) == (80, 120, 160)
