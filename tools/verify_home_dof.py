# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Backlog-#2 real-sim verification: `home` works on the real 6-DoF Piper arm.

The home skill hard-defaulted the 5-DoF SO-101 home pose; on the go2+Piper (6-DoF)
arm move_joints raised 'expected 6 positions, got 5'. Because Agent.execute_skill()
APPENDS a trailing `home` step to most manipulation plans (pick/scan/...), this bug
crashed the WHOLE planner/executor path — blocking the bare-vector-cli + NL fetch.

This drives `home` through the real Agent/GoalExecutor against a REAL go2+Piper sim
(not a mock): it proves the previously-crashing executor->HomeSkill->real-arm path
now completes and the real 6-DoF arm actually reaches the neutral pose.

Prints `RESULT {json}`.  Run:
  MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python tools/verify_home_dof.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from vector_os_nano.core.agent import Agent  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_piper_gripper import (  # noqa: E402
    MuJoCoPiperGripper,
)


def main() -> int:
    out: dict = {}
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    gripper = MuJoCoPiperGripper(go2)
    gripper.connect()
    agent = Agent(base=go2, arm=piper, gripper=gripper, config={})

    out["arm_dof"] = int(piper.dof)
    time.sleep(1.5)

    # Drive the arm AWAY from neutral first, so reaching home is a real, observable move.
    piper.move_joints([0.3, -0.8, 0.4, 0.5, -0.3, 0.2], duration=2.0)
    before = [round(v, 3) for v in piper.get_joint_positions()]
    out["joints_before_home"] = before
    print(f"arm DoF={piper.dof}; joints before home={before}")

    # The previously-crashing path: Agent.execute_skill('home') -> GoalExecutor ->
    # HomeSkill -> real 6-DoF Piper.move_joints. Pre-fix this raised ValueError
    # 'expected 6 positions, got 5'.
    crashed = None
    try:
        res = agent.execute_skill("home")
        ok = bool(res.success)
        out["execute_skill_success"] = ok
        out["execute_skill_status"] = getattr(res, "status", None)
        print(f"execute_skill('home') success={ok} status={getattr(res,'status',None)}")
    except Exception as exc:  # noqa: BLE001  (the regression we are guarding against)
        crashed = f"{type(exc).__name__}: {exc}"
        out["execute_skill_success"] = False
        out["crash"] = crashed
        print(f"execute_skill('home') CRASHED: {crashed}")

    after = [round(v, 3) for v in piper.get_joint_positions()]
    out["joints_after_home"] = after
    print(f"joints after home={after}")

    # Acceptance: no crash AND the real arm actually moved toward the neutral pose
    # ([0]*dof). Position PD has steady-state error near limits, so use a tolerant
    # band: every joint must be closer to 0 than its driven-away start.
    moved_home = crashed is None and all(
        abs(a) <= abs(b) + 1e-3 for a, b in zip(after, before)
    )
    # And it must be genuinely near neutral (not just "didn't move further out").
    near_neutral = crashed is None and max(abs(a) for a in after) < 0.25
    out["moved_toward_home"] = moved_home
    out["near_neutral"] = near_neutral

    passed = bool(out.get("execute_skill_success") and moved_home and near_neutral)
    out["overall_pass"] = passed

    for dev in (gripper, piper, go2):
        try:
            dev.disconnect()
        except Exception:  # noqa: BLE001
            pass

    print("RESULT " + json.dumps(out))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
