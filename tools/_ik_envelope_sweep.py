# SPDX-License-Identifier: Apache-2.0
"""Map the Piper top-down IK reach envelope vs dog standoff (R12 — IK-margin design).

Teleports the dog to a sweep of standoff distances from the green bottle and checks
ik_top_down at the (pre-)grasp point — NO walking, NO grasp — so we SEE exactly which
standoff band IK can reach and where the MARGIN is. Confirms "理论上抓得到" + tells us
the target standoff the scalable approach should drive to (0-nudge reachable).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj  # noqa: E402

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402

import time as _time  # noqa: E402

go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
go2.connect()
piper = MuJoCoPiper(go2)
piper.connect()

# PAUSE the 1 kHz gait thread before teleporting qpos (writing live qpos while the
# physics daemon steps segfaults — concurrent MjData access).
go2._running = False
_time.sleep(0.3)

model, data = go2._mj.model, go2._mj.data
root = go2._mj.layout.root_qpos_adr

bottle = piper.get_object_positions().get("pickable_bottle_green")
bx, by, bz = float(bottle[0]), float(bottle[1]), float(bottle[2])
pre_z = bz + 0.02  # _GO2_PRE_GRASP_H
print(f"green bottle @ ({bx:.3f}, {by:.3f}, {bz:.3f}); pre-grasp z={pre_z:.3f}")
print("dog_x  standoff(m)  ik_top_down(grasp)  ik_top_down(pre-grasp)")

y0 = float(data.qpos[root + 1])
z0 = float(data.qpos[root + 2])
for i in range(18):
    dog_x = 10.30 + i * 0.02   # 10.30 .. 10.64
    data.qpos[root + 0] = dog_x
    data.qpos[root + 1] = by    # align laterally with the bottle
    data.qpos[root + 2] = z0
    mj.mj_forward(model, data)
    standoff = bx - dog_x
    q_grasp = piper.ik_top_down((bx, by, bz))
    q_pre = piper.ik_top_down((bx, by, pre_z))
    print(f"{dog_x:5.2f}   {standoff:6.3f}      {'OK ' if q_grasp is not None else 'FAIL':4s}            {'OK' if q_pre is not None else 'FAIL'}")

for dev in (piper, go2):
    try:
        dev.disconnect()
    except Exception:  # noqa: BLE001
        pass
