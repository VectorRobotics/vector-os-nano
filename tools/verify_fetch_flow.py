# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Step-#3 real-sim verification: the full find->navigate->arrival-perceive->grasp flow.

ONE headless go2+arm sim, composing the building blocks the way the model would:
  1. look  (stub VLM naming, real depth localization) -> scene graph has 'green bottle'
  2. drive the dog AWAY from the table
  3. navigate_to_object('green bottle')  -> dog at the ~0.7m standoff (object in view)
  4. perception_grasp('green bottle')    -> arrival-fresh perceive + R12 approach + grasp

Acceptance (MuJoCo GT, not a flag): the GREEN bottle is lifted off the table AND the
gripper reports holding (honest oracle make_holding_object). This proves the arrival
depth re-perception is satisfied BY COMPOSITION — perception_grasp's first perceive
happens at the navigate_to_object arrival standoff, the well-framed distance R12 wants.

Prints `RESULT {json}`.  Run:
  MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python tools/verify_fetch_flow.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

import mujoco as mj  # noqa: E402

from vector_os_nano.core.agent import Agent  # noqa: E402
from vector_os_nano.core.scene_graph import SceneGraph  # noqa: E402
from vector_os_nano.core.skill import SkillContext  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper_gripper import (  # noqa: E402
    MuJoCoPiperGripper,
)
from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception  # noqa: E402
from vector_os_nano.perception.object_localizer import localize_objects_3d  # noqa: E402
from vector_os_nano.perception.vlm_go2 import (  # noqa: E402
    DetectedObject, RoomIdentification, SceneDescription,
)
from vector_os_nano.skills.go2.look import LookSkill  # noqa: E402
from vector_os_nano.skills.navigate_to_object import NavigateToObjectSkill  # noqa: E402
from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill  # noqa: E402
from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace  # noqa: E402

_TARGET = "green bottle"
_GT_BODY = "pickable_bottle_green"
_FAR_POINT = (8.0, 3.0)


class _StubVLM:
    def describe_scene(self, frame):
        objs = [
            DetectedObject(name="green bottle", description="", confidence=0.9),
            DetectedObject(name="blue bottle", description="", confidence=0.9),
            DetectedObject(name="red can", description="", confidence=0.9),
        ]
        return SceneDescription(summary="containers on a table", objects=objs,
                                room_type="kitchen", details="")

    def identify_room(self, frame):
        return RoomIdentification(room="kitchen", confidence=0.9, reasoning="stub")


def _gt(go2, name):
    model, data = go2._mj.model, go2._mj.data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    return [float(v) for v in data.xpos[bid]]


def _ee_pos(go2):
    """World position of the Piper EE weld site (the point the weld measures from)."""
    model, data = go2._mj.model, go2._mj.data
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "piper_ee_site")
    if sid < 0:
        return None
    return [float(v) for v in data.site_xpos[sid]]


def main() -> int:
    out: dict = {"steps": {}}
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    gripper = MuJoCoPiperGripper(go2)
    gripper.connect()
    perception = Go2GraspPerception(go2, width=320, height=240)
    agent = Agent(base=go2, arm=piper, gripper=gripper, perception=perception, config={})
    holding_oracle = rig_namespace(go2.runtime, "go2_piper")["holding_object"]

    z0 = _gt(go2, _GT_BODY)[2]
    out["green_z_before"] = round(z0, 3)
    print(f"green bottle z BEFORE = {z0:.3f}  holding={holding_oracle()}")

    # context for direct skill calls (bypasses the planner/executor home 5-vs-6 bug)
    sg = SceneGraph()
    ctx = SkillContext(
        arms={"default": piper}, grippers={"default": gripper},
        bases={"default": go2}, perception_sources={"default": perception},
        services={"vlm": _StubVLM(), "spatial_memory": sg},
    )

    # settle + warm up detector before the first look
    time.sleep(2.5)
    _ = localize_objects_3d(perception, [_TARGET])

    # 1. look -> seed scene graph
    lr = LookSkill().execute({}, ctx)
    seeded = [(o.category, round(o.x, 3), round(o.y, 3)) for o in sg.find_objects_by_category(_TARGET)]
    print(f"1) look success={lr.success}; graph {_TARGET!r}={seeded}")
    out["steps"]["look"] = {"success": lr.success, "objects": seeded}

    # 2. drive away
    go2.navigate_to(_FAR_POINT[0], _FAR_POINT[1], timeout=60)
    p = go2.get_position()
    g = _gt(go2, _GT_BODY)
    away = math.hypot(p[0] - g[0], p[1] - g[1])
    print(f"2) moved away -> ({p[0]:.2f},{p[1]:.2f}) dist_to_green={away:.2f}m")
    out["steps"]["move_away"] = {"pos": [round(p[0], 2), round(p[1], 2)], "dist": round(away, 2)}

    # 3. navigate_to_object
    nr = NavigateToObjectSkill().execute({"object": _TARGET}, ctx)
    p = go2.get_position()
    near = math.hypot(p[0] - g[0], p[1] - g[1])
    print(f"3) navigate_to_object success={nr.success} -> ({p[0]:.2f},{p[1]:.2f}) dist_to_green={near:.2f}m")
    out["steps"]["navigate"] = {"success": nr.success, "pos": [round(p[0], 2), round(p[1], 2)],
                                "dist": round(near, 2)}

    # 4. perception_grasp (arrival-fresh perceive + R12 approach + grasp)
    # Snapshot the GT object centre BEFORE the grasp — the object is static until
    # welded, so this is the honest reference for the perceive error (comparing the
    # perceived point to the POST-lift GT would fold the +0.23 m lift into the z error).
    g_pre = _gt(go2, _GT_BODY)
    gr = PerceptionGraspSkill().execute({"query": _TARGET}, ctx)
    z1 = _gt(go2, _GT_BODY)[2]
    held = bool(gripper.is_holding())
    oracle = bool(holding_oracle())
    lifted = z1 - z0

    # --- error decomposition (OBSERVE): perceive error vs GT object centre.
    # gw = perceived grasp point fed to IK; g_now = live GT object centre. We capture
    # the planar (xy) and vertical (z) components so a hypothesis loop can tell a
    # perception miss (gw far from GT) from an execution/pose miss (gw≈GT but the EE
    # still lands off — read from the gripper's weld-distance log).
    g_now = _gt(go2, _GT_BODY)
    gw = (gr.result_data or {}).get("grasp_world")
    perceive = None
    if gw and len(gw) == 3:
        pxy = math.hypot(gw[0] - g_pre[0], gw[1] - g_pre[1])
        pz = gw[2] - g_pre[2]
        perceive = {"xy": round(pxy, 4), "z": round(pz, 4),
                    "d3": round(math.dist(gw, g_pre), 4)}
    out["steps"]["perceive_err"] = perceive
    print(f"   perceive_err vs GT centre (pre-grasp): {perceive}  (gw={gw} gt={[round(v,3) for v in g_pre]})")
    print(f"4) perception_grasp success={gr.success} diag={gr.diagnosis_code} err={gr.error_message!r}")
    print(f"   grasp result_data={json.dumps(gr.result_data or {})}")
    print(f"   green z AFTER={z1:.3f} (lifted {lifted:+.3f}m)  gripper.is_holding={held}  oracle={oracle}")
    out["steps"]["grasp"] = {"success": gr.success, "diag": gr.diagnosis_code,
                             "green_z_after": round(z1, 3), "lifted_m": round(lifted, 3),
                             "is_holding": held, "oracle_holding": oracle,
                             "grasp_world": (gr.result_data or {}).get("grasp_world")}

    # verdict: honest grasp = bottle lifted clear AND gripper holding (oracle)
    grasped = bool((held or oracle) and lifted > 0.05)
    out["overall_pass"] = grasped

    for dev in (gripper, piper, go2):
        try:
            dev.disconnect()
        except Exception:  # noqa: BLE001
            pass

    print("RESULT " + json.dumps(out))
    return 0 if grasped else 1


if __name__ == "__main__":
    raise SystemExit(main())
