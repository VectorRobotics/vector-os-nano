# SPDX-License-Identifier: Apache-2.0
"""Measure the IK alignment tolerance (heading x lateral) at the grasp standoff (R12)."""
from __future__ import annotations

import math
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj  # noqa: E402

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402

go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True); go2.connect()
piper = MuJoCoPiper(go2); piper.connect()
go2._running = False; time.sleep(0.3)
m, d = go2._mj.model, go2._mj.data
rq = go2._mj.layout.root_qpos_adr

bp = piper.get_object_positions()["pickable_bottle_green"]
bx, by, bz = float(bp[0]), float(bp[1]), float(bp[2])
STAND = 0.42  # achievable jam standoff


def set_pose(dog_x, dog_y, yaw):
    d.qpos[rq] = dog_x
    d.qpos[rq + 1] = dog_y
    # base quat for yaw (w,x,y,z)
    d.qpos[rq + 3] = math.cos(yaw / 2)
    d.qpos[rq + 4] = 0.0
    d.qpos[rq + 5] = 0.0
    d.qpos[rq + 6] = math.sin(yaw / 2)
    mj.mj_forward(m, d)


print(f"green @ ({bx:.2f},{by:.2f},{bz:.2f}); standoff={STAND} (dog_x={bx-STAND:.2f})")
print("HEADING sweep (lateral=0):  yaw_deg -> ik")
for deg in (-8, -6, -4, -2, 0, 2, 4, 6, 8):
    set_pose(bx - STAND, by, math.radians(deg))
    q = piper.ik_top_down((bx, by, bz))
    print(f"  yaw={deg:+3d}deg : {'OK' if q is not None else 'FAIL'}")
print("LATERAL sweep (yaw=0):  dy(m) -> ik")
for dy in (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06):
    set_pose(bx - STAND, by + dy, 0.0)
    q = piper.ik_top_down((bx, by, bz))
    print(f"  dy={dy:+.2f}m : {'OK' if q is not None else 'FAIL'}")

go2.disconnect()
