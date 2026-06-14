# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RED-first: the PhotorealRenderer adapter + scene builder + look-at helper.

The adapter is the drop-in for ``get_camera_observation()``: it reads the LIVE
MuJoCo camera pose and renders the photoreal frame from exactly that pose, so the
frame the VLM sees matches the robot's physics camera (no drift end-to-end). The
scene builder maps furniture placements -> a Blender scene spec, substituting
photoreal PBR assets for the toy meshes (R2/R3: asset fidelity is the lever).
Tested with a fake bridge + a hand-built tiny MJCF — no Blender, no network.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.playground.photoreal.camera import (
    look_at_camera_matrix,
    mujoco_camera_to_blender,
)
from vector_os_nano.playground.photoreal.renderer import PhotorealRenderer
from vector_os_nano.playground.photoreal.scene import build_room_scene_spec


# -- scene builder (pure) --------------------------------------------------

def test_scene_builder_places_mapped_assets_on_floor():
    placements = {"chair": (3.6, 0.0), "sofa": (3.6, 2.0), "plant": (3.6, -2.0)}
    asset_map = {
        "chair": {"path": "/a/chair.gltf", "scale": 1.0},
        "sofa": {"path": "/a/sofa.gltf", "scale": 1.2},
        # 'plant' deliberately absent — partial library must skip, not crash
    }
    spec = build_room_scene_spec(placements, asset_map,
                                 floor={"size": 12}, walls=[{"pos": [4, 0, 0.6]}])
    assets = {a["path"]: a for a in spec["assets"]}
    assert set(assets) == {"/a/chair.gltf", "/a/sofa.gltf"}     # plant skipped
    assert assets["/a/chair.gltf"]["pos"] == [3.6, 0.0, 0.0]
    assert assets["/a/chair.gltf"]["scale"] == 1.0
    assert assets["/a/sofa.gltf"]["pos"] == [3.6, 2.0, 0.0]
    assert assets["/a/sofa.gltf"]["scale"] == 1.2
    assert all(a["ground"] for a in spec["assets"])              # rest on floor
    assert spec["floor"]["size"] == 12
    assert spec["walls"] == [{"pos": [4, 0, 0.6]}]
    assert "sun" in spec and "sky" in spec                       # default lighting


def test_scene_builder_empty_asset_map_yields_no_assets():
    spec = build_room_scene_spec({"chair": (1.0, 2.0)}, {})
    assert spec["assets"] == []


# -- look-at helper --------------------------------------------------------

def test_look_at_matrix_points_forward_at_target():
    pos, mat = look_at_camera_matrix((0.0, 0.0, 1.0), (0.0, 5.0, 1.0))
    R = np.asarray(mat, float).reshape(3, 3)
    forward = -R[:, 2]                       # MuJoCo cam looks down -local-Z
    np.testing.assert_allclose(forward, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(np.asarray(pos), [0.0, 0.0, 1.0])
    # orthonormal proper rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert R[2, 1] > 0                       # up has a +world-Z component


# -- renderer adapter (fake bridge + tiny MJCF) ----------------------------

_TINY_MJCF = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="rig" pos="1.5 -0.5 0.85">
      <camera name="g1_head_cam" pos="0 0 0" xyaxes="1 0 0 0 0 1" fovy="45"/>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeBridge:
    """Duck-typed BlenderBridge: records the pose/scene, returns a known frame."""

    def __init__(self, h=48, w=64):
        self.calls = []
        self._frame = np.full((h, w, 3), 7, dtype=np.uint8)

    def render(self, pose, scene, *, width=640, height=480, samples=32):
        self.calls.append({"pose": pose, "scene": scene,
                           "width": width, "height": height, "samples": samples})
        return self._frame


@pytest.fixture
def mj_room():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_TINY_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_observe_returns_camera_observation_shape(mj_room):
    _, model, data = mj_room
    bridge = _FakeBridge()
    r = PhotorealRenderer(bridge, {"floor": {}}, cam_name="g1_head_cam",
                          width=64, height=48, samples=8)
    obs = r.observe(model, data)
    assert set(obs) >= {"rgb", "cam_pos", "cam_mat", "fovy"}
    assert obs["rgb"].shape == (48, 64, 3)
    assert obs["cam_pos"].shape == (3,)
    assert obs["cam_mat"].shape in {(9,), (3, 3)}
    assert np.isclose(obs["fovy"], 45.0)


def test_observe_renders_from_exact_mujoco_pose_no_drift(mj_room):
    """The matrix_world handed to the bridge == the transform of MuJoCo's own
    live camera pose. This is the whole co-sim invariant."""
    mujoco, model, data = mj_room
    bridge = _FakeBridge()
    r = PhotorealRenderer(bridge, {"floor": {}}, cam_name="g1_head_cam")
    r.observe(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "g1_head_cam")
    expected = mujoco_camera_to_blender(
        data.cam_xpos[cam_id], data.cam_xmat[cam_id], model.cam_fovy[cam_id])
    sent = bridge.calls[0]["pose"]
    np.testing.assert_allclose(sent.flat16(), expected.flat16(), atol=1e-12)
    # camera position really is the rig position
    np.testing.assert_allclose(bridge.calls[0]["pose"].matrix_world[:3, 3],
                               [1.5, -0.5, 0.85], atol=1e-9)


def test_observe_forwards_scene_and_render_params(mj_room):
    _, model, data = mj_room
    bridge = _FakeBridge()
    scene = {"floor": {"size": 9}, "assets": []}
    r = PhotorealRenderer(bridge, scene, width=320, height=240, samples=16)
    r.observe(model, data)
    call = bridge.calls[0]
    assert call["scene"] is scene
    assert call["width"] == 320 and call["height"] == 240 and call["samples"] == 16


def test_render_from_pose_explicit_no_mujoco():
    """The hybrid integration renders from an EXPLICIT MuJoCo cam pose (the same
    pose the depth was rendered at) — no model/data needed, callable off-thread."""
    bridge = _FakeBridge(h=24, w=32)
    r = PhotorealRenderer(bridge, {"floor": {}}, width=32, height=24, samples=8)
    cam_mat = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1.0])
    rgb = r.render_from_pose([1.5, -0.5, 0.85], cam_mat, 45.0)
    assert rgb.shape == (24, 32, 3)
    sent = bridge.calls[0]["pose"]
    expected = mujoco_camera_to_blender([1.5, -0.5, 0.85], cam_mat, 45.0)
    np.testing.assert_allclose(sent.flat16(), expected.flat16(), atol=1e-12)
