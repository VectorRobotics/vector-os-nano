# SPDX-License-Identifier: Apache-2.0
"""Probe ONE perception-grasp's IK reach margin (R11 — Yusen's IK-distance hypothesis).

Calls perception_grasp DIRECTLY (no model) for GRASP_QUERY, capturing the [PGRASP] IK /
approach logs + the ground-truth outcome (holding + lift) + the dog's final distance to
the target. Tells us whether the grasp succeeds comfortably or marginally, and whether a
FARTHER / laterally-offset object actually fails IK. One object per subprocess.
"""
from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("MUJOCO_GL", "egl")

_QUERY = os.environ.get("GRASP_QUERY", "绿色的瓶子")
_TARGET = os.environ.get("GRASP_TARGET", "pickable_bottle_green")

# Capture the perception_grasp INFO logs ([PGRASP] reach/approach/grasp_world).
_pg_lines: list[str] = []


class _Cap(logging.Handler):
    def emit(self, rec):  # noqa: ANN001
        m = self.format(rec)
        if "PGRASP" in m or "ik" in m.lower() or "reach" in m.lower():
            _pg_lines.append(m)


logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(_Cap())


def main() -> int:
    import math

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
    from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.vcli.worlds.arm_sim_oracle import make_holding_object

    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True); go2.connect()
    piper = MuJoCoPiper(go2); piper.connect()
    gripper = MuJoCoPiperGripper(go2); gripper.connect()
    perception = Go2GraspPerception(go2, width=320, height=240)
    agent = Agent(base=go2, arm=piper, gripper=gripper, perception=perception, config={})
    agent._skill_registry.register(PickTopDownSkill())
    agent._skill_registry.register(PerceptionGraspSkill())

    objs0 = piper.get_object_positions()
    tgt0 = objs0.get(_TARGET)
    dog0 = go2.get_position()
    dist0 = math.hypot(tgt0[0] - dog0[0], tgt0[1] - dog0[1]) if tgt0 else -1

    res = agent.execute_skill("perception_grasp", {"query": _QUERY})

    objs1 = piper.get_object_positions()
    tgt1 = objs1.get(_TARGET)
    dog1 = go2.get_position()
    dist1 = math.hypot(tgt1[0] - dog1[0], tgt1[1] - dog1[1]) if tgt1 else -1
    holding = bool(make_holding_object(agent)())
    lift = (tgt1[2] - tgt0[2]) if (tgt0 and tgt1) else None
    rd = dict(getattr(res, "result_data", {}) or {})

    out = {
        "query": _QUERY, "target": _TARGET,
        "GROUNDED_proxy_holding": holding,
        "lift_m": round(lift, 3) if lift is not None else None,
        "dog_to_target_before_m": round(dist0, 3),
        "dog_to_target_after_m": round(dist1, 3),
        "grasp_world": rd.get("grasp_world"),
        "diagnosis": rd.get("diagnosis"),
        "skill_success_selfreport": bool(res.success),
        "ik_reach_log": [ln for ln in _pg_lines if any(k in ln for k in ("reach", "approach", "grasp_world", "unreach", "out of reach"))][-8:],
    }
    print("IKPROBE " + json.dumps(out, ensure_ascii=False))
    for dev in (gripper, piper, go2):
        try:
            dev.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
