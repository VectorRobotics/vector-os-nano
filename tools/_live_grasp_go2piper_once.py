# SPDX-License-Identifier: Apache-2.0
"""ONE live-model go2+Piper PERCEPTION-grasp attempt; prints `RESULT {json}` (R10/D81).

Builds the go2+Piper attach world IN-PROCESS exactly as the deterministic R1 test does,
but drives `run_turn_native` with the REAL DeepSeek backend (no scripted turns) so the
MODEL alone must route a perception grasp -> verify(holding_object(...)) -> GROUNDED on
the honest spine. Invoked once per subprocess by the orchestrator (MuJoCo can't realloc
worlds in one process). All sim/engine noise -> stderr; only `RESULT {json}` -> stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("VECTOR_FAKE_LLM_TOOLS", None)    # ensure the REAL backend (no scripted turns)
os.environ.pop("VECTOR_FAKE_LLM", None)

_GOAL = os.environ.get("GRASP_GOAL", "把前面的绿色瓶子抓起来")


def main() -> int:
    res: dict = {"goal": _GOAL}
    go2 = piper = gripper = None
    try:
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
        from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper
        from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception
        from vector_os_nano.skills.go2 import get_go2_skills
        from vector_os_nano.skills.mobile_pick import MobilePickSkill
        from vector_os_nano.skills.mobile_place import MobilePlaceSkill
        from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill
        from vector_os_nano.skills.pick_top_down import PickTopDownSkill
        from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
        from vector_os_nano.vcli.cli import create_backend_with_fake_seam
        from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
        from vector_os_nano.vcli.config import resolve_credentials
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.permissions import PermissionContext
        from vector_os_nano.vcli.session import Session
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
        from vector_os_nano.vcli.verdict import VerdictReport
        from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace
        from vector_os_nano.vcli.worlds.robot import RobotWorld

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
        for sk in (PickTopDownSkill(), PlaceTopDownSkill(), MobilePickSkill(), MobilePlaceSkill()):
            agent._skill_registry.register(sk)
        agent._skill_registry.register(PerceptionGraspSkill())

        assert rig_namespace(go2.runtime, "go2_piper")["holding_object"]() is False, "must boot not holding"

        api_key, provider, model, base_url = resolve_credentials()
        backend = create_backend_with_fake_seam(
            provider=provider, api_key=api_key, model=model, base_url=base_url
        )
        eng = VectorEngine(
            backend=backend, registry=CategorizedToolRegistry(), permissions=PermissionContext(),
        )
        eng._world = RobotWorld()
        eng.init_vgg(agent=agent, skill_registry=agent._skill_registry, world=RobotWorld())
        eng._vgg_agent = agent
        eng._backend = backend

        session = Session(
            session_id="live-go2piper-grasp", created_at="t", updated_at="t",
            path=Path("/tmp/live_go2piper_grasp.jsonl"),
        )
        trace = eng.run_turn_native(_GOAL, session=session)

        names = verify_oracle_names(agent, eng)
        report = VerdictReport.from_trace(trace, names)
        steps = list(trace.steps)
        res["evidence"] = report.evidence
        res["verified"] = bool(report.verified)
        res["n_steps"] = len(steps)
        res["strategies"] = [s.strategy for s in steps]
        res["holding_final"] = bool(rig_namespace(go2.runtime, "go2_piper")["holding_object"]())
    except Exception as exc:  # noqa: BLE001
        res["evidence"] = "ERROR"
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        for dev in (gripper, piper, go2):
            try:
                if dev is not None:
                    dev.disconnect()
            except Exception:  # noqa: BLE001
                pass

    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
