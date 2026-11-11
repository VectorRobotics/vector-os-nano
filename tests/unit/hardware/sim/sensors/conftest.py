# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tiny inline MJCF fixtures for sensor unit tests.

These deliberately do NOT import the Go2 model. Per
``feedback_no_parallel_agents.md`` the full go2 / pipeline import chain
is OOM-prone under parallel pytest; sensor unit tests must stay light.
"""
from __future__ import annotations

import pytest


_TINY_MJCF = """
<mujoco>
  <worldbody>
    <body name="trunk" pos="0 0 0.5">
      <freejoint name="trunk_root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
    <geom name="wall_front" type="box" pos="3 0 0.5" size="0.1 5 5" rgba="1 0 0 1"/>
    <geom name="wall_back"  type="box" pos="-3 0 0.5" size="0.1 5 5" rgba="0 1 0 1"/>
    <geom name="floor"      type="plane" pos="0 0 0" size="20 20 0.1" rgba="0.5 0.5 0.5 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def tiny_world(tmp_path):
    """The single-body MJCF as a MuJoCoWorld (rays go through ``world.raycast``)."""
    from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld
    from vector_os_nano.sim.scene import compiled_scene

    path = tmp_path / "tiny.xml"
    path.write_text(_TINY_MJCF)
    world = MuJoCoWorld(compiled_scene(path))
    yield world
    world.close()
