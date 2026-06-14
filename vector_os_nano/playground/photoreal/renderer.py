# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PhotorealRenderer — the co-sim world adapter.

A drop-in for a MuJoCo base's ``get_camera_observation()``: it reads the LIVE
MuJoCo camera pose (``cam_xpos`` / ``cam_xmat`` / ``cam_fovy``) and renders the
photoreal frame from EXACTLY that pose via the Blender bridge, so the image the
VLM grounds matches the robot's physics camera (no drift — the whole co-sim
invariant). Returns the same ``{rgb, cam_pos, cam_mat, fovy}`` dict shape the
recognise->navigate skill already consumes.

Furniture placement is static room config (the scene spec, built once via
``scene.build_room_scene_spec``); only the camera moves per frame. Depth is NOT
produced here — in the hybrid co-sim integration the MuJoCo base supplies depth
at the same pose (geometrically consistent because the pose is identical).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vector_os_nano.playground.photoreal.camera import mujoco_camera_to_blender


class PhotorealRenderer:
    """Render a MuJoCo camera's view as a photoreal frame through a BlenderBridge."""

    def __init__(
        self,
        bridge: Any,
        scene_spec: dict,
        *,
        cam_name: str = "g1_head_cam",
        width: int = 640,
        height: int = 480,
        samples: int = 32,
    ) -> None:
        self._bridge = bridge
        self._scene = scene_spec
        self._cam_name = cam_name
        self._width = width
        self._height = height
        self._samples = samples

    def render_from_pose(self, cam_pos, cam_mat, fovy_deg: float) -> np.ndarray:
        """Photoreal RGB from an EXPLICIT MuJoCo camera pose.

        The hybrid co-sim integration calls this with the pose MuJoCo already
        rendered the depth at, so the photoreal RGB and the MuJoCo depth share
        one camera frame. No ``bpy``/GL in our process → safe off the control
        thread (unlike MuJoCo's Renderer).
        """
        pose = mujoco_camera_to_blender(cam_pos, cam_mat, fovy_deg)
        return self._bridge.render(
            pose, self._scene,
            width=self._width, height=self._height, samples=self._samples)

    def observe(self, model: Any, data: Any) -> dict:
        """``{rgb, cam_pos, cam_mat, fovy}`` from the live MuJoCo camera pose."""
        import mujoco  # noqa: PLC0415 — adapter only; pure tests don't import it

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam_name)
        if cam_id < 0:
            raise ValueError(f"camera not found in model: {self._cam_name!r}")
        cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
        cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).copy()
        fovy = float(model.cam_fovy[cam_id])
        rgb = self.render_from_pose(cam_pos, cam_mat, fovy)
        return {"rgb": rgb, "cam_pos": cam_pos, "cam_mat": cam_mat, "fovy": fovy}
