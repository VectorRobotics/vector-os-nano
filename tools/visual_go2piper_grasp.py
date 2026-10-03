# SPDX-License-Identifier: Apache-2.0
"""Render the go2+Piper perception grasp (before/after) to PNGs for visual verification (R10)."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from vector_os_nano.core.agent import Agent  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper  # noqa: E402
from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception  # noqa: E402
from vector_os_nano.skills.go2 import get_go2_skills  # noqa: E402
from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill  # noqa: E402
from vector_os_nano.skills.pick_top_down import PickTopDownSkill  # noqa: E402
from vector_os_nano.vcli.worlds.arm_sim_oracle import make_holding_object  # noqa: E402

go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
go2.connect()
piper = MuJoCoPiper(go2)
piper.connect()
gripper = MuJoCoPiperGripper(go2)
gripper.connect()
perception = Go2GraspPerception(go2, width=320, height=240)
agent = Agent(base=go2, arm=piper, gripper=gripper, perception=perception, config={})
for s in get_go2_skills():
    agent._skill_registry.register(s)
agent._skill_registry.register(PickTopDownSkill())
agent._skill_registry.register(PerceptionGraspSkill())

model, data = go2._mj.model, go2._mj.data
renderer = mj.Renderer(model, height=480, width=640)
_free = mj.MjvCamera()
_free.type = mj.mjtCamera.mjCAMERA_FREE
_free.lookat[:] = [10.55, 3.0, 0.30]
_free.azimuth, _free.elevation, _free.distance = 150.0, -18.0, 2.4


def shot(tag: str) -> None:
    mj.mj_forward(model, data)
    renderer.update_scene(data, camera=_free)
    rgb = renderer.render()
    path = f"/tmp/go2grasp_{tag}.png"
    cv2.imwrite(path, rgb[:, :, ::-1])
    print(f"  {tag}: mean={float(np.mean(rgb)):.1f} -> {path}")


def bottle_z() -> float:
    try:
        return float(piper.get_object_positions().get("pickable_bottle_green", [0, 0, 0])[2])
    except Exception:  # noqa: BLE001
        return -1.0


print("green-bottle z BEFORE:", round(bottle_z(), 3), "holding:", make_holding_object(agent)())
shot("before")

res = agent.execute_skill("perception_grasp", {"query": "绿色的瓶子"})
print("perception_grasp success:", res.success)
print("green-bottle z AFTER :", round(bottle_z(), 3), "holding:", make_holding_object(agent)())
shot("after")

for dev in (gripper, piper, go2):
    try:
        dev.disconnect()
    except Exception:  # noqa: BLE001
        pass
