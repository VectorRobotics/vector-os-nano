# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tiny inline MJCF fixtures for sensor unit tests.

These deliberately do NOT import the Go2 model. Per
``feedback_no_parallel_agents.md`` the full go2 / pipeline import chain
is OOM-prone under parallel pytest; sensor unit tests must stay light.
"""
from __future__ import annotations

import os

# EGL is the only headless render backend that ships in the .venv-nano
# image. Honoured by both lidar (mj_ray needs no GL) and pano (Renderer
# does need GL).
os.environ.setdefault("MUJOCO_GL", "egl")

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
def tiny_model_data():
    """Single-body MJCF + paired data — shared across sensor tests."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(_TINY_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def trunk_translated_to(tiny_model_data):
    """Helper that lets a test reposition the trunk and re-runs FK.

    Returns a callable ``move(x, y, z)`` that updates qpos and forwards
    kinematics; subsequent reads of ``data.xpos[trunk_id]`` reflect the
    new position.
    """
    import mujoco
    model, data = tiny_model_data
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    def _move(x: float, y: float, z: float) -> None:
        # qpos layout for a freejoint: [x, y, z, qw, qx, qy, qz]
        data.qpos[0:3] = [x, y, z]
        mujoco.mj_forward(model, data)

    return body_id, _move
