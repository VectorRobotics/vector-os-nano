# SPDX-License-Identifier: Apache-2.0
"""Render a real arm grasp (before/after) to PNGs for visual verification (R10)."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco as mj  # noqa: E402

from vector_os_nano.core.agent import Agent  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception  # noqa: E402
from vector_os_nano.skills import get_default_skills  # noqa: E402
from vector_os_nano.skills.pick import SIM_PICK_CONFIG  # noqa: E402
from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace  # noqa: E402

arm = MuJoCoArm(gui=False)
arm.connect()
gripper = MuJoCoGripper(arm)
perception = MuJoCoPerception(arm)
agent = Agent(arm=arm, gripper=gripper, perception=perception,
              config={"skills": {"pick": dict(SIM_PICK_CONFIG)}})
for s in get_default_skills():
    agent._skill_registry.register(s)

import numpy as np  # noqa: E402

renderer = mj.Renderer(arm._model, height=480, width=640)
_CAMS = ("overhead", "front", "side")


# A free 3/4 camera that shows the LIFT (z) clearly — overhead hides it.
_free = mj.MjvCamera()
_free.type = mj.mjtCamera.mjCAMERA_FREE
_free.lookat[:] = [0.18, 0.06, 0.12]
_free.azimuth, _free.elevation, _free.distance = 130.0, -20.0, 0.85


def shot(tag: str) -> None:
    """Render the bright overhead cam + a 3/4 free cam (shows the lift); report brightness."""
    mj.mj_forward(arm._model, arm._data)
    for cam, name in (("overhead", "overhead"), (_free, "angle")):
        renderer.update_scene(arm._data, camera=cam)
        rgb = renderer.render()
        path = f"/tmp/grasp_{tag}_{name}.png"
        cv2.imwrite(path, rgb[:, :, ::-1])  # RGB -> BGR
        print(f"  {tag}/{name}: mean={float(np.mean(rgb)):.1f} -> {path}")


z0 = arm.get_object_positions().get("banana")
print("banana z BEFORE:", z0, "holding:", rig_namespace(arm.runtime, "so101")["holding_object"]())
shot("before")

res = agent.execute_skill("pick", {"object_label": "banana", "mode": "hold"})
print("pick success   :", res.success)

z1 = arm.get_object_positions().get("banana")
print("banana z AFTER :", z1)
print("lift (cm)      :", round((z1[2] - z0[2]) * 100, 1) if (z0 and z1) else "n/a")
print("holding AFTER  :", rig_namespace(arm.runtime, "so101")["holding_object"]())
shot("after")

try:
    arm.disconnect()
except Exception:  # noqa: BLE001
    pass
