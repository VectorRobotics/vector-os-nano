#!/usr/bin/env python3
"""End-to-end verification for the Piper top-down grasp pipeline.

Runs a headless MuJoCo Go2+Piper sim, picks each of the three pickable_*
objects in the scene (one fresh subprocess per object — MuJoCo does not
tolerate multiple sim instances per process), and reports:

    - IK convergence
    - gripper close / is_holding state
    - vertical lift of the object (mm)
    - overall pass / fail per object

Usage:
    .venv-nano/bin/python scripts/verify_pick_top_down.py
    .venv-nano/bin/python scripts/verify_pick_top_down.py --repeat 5
    .venv-nano/bin/python scripts/verify_pick_top_down.py --object pickable_can_red

Exit code 0 when every run reports lift >= MIN_LIFT_CM and held=True.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Object ids to sweep. Initial z is measured at runtime (after settling)
# so scene edits don't silently break the baseline.
_TARGETS: list[str] = [
    "pickable_bottle_blue",
    "pickable_bottle_green",
    "pickable_can_red",
]

# Minimum vertical lift (in cm) to count as a successful grasp.
_MIN_LIFT_CM: float = 1.0

# Child script that actually runs the pick. Kept as inline template so the
# verification works whether the repo is cloned or symlinked anywhere.
_CHILD_TEMPLATE = textwrap.dedent("""
    import os, sys, time, logging, mujoco
    os.environ["VECTOR_SIM_WITH_ARM"] = "1"
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
    from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper
    from vector_os_nano.core.world_model import WorldModel, ObjectState
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    OBJ = "{obj_id}"

    # Direct hardware path — no ROS2 bridge, pure in-process MuJoCo. This
    # bypasses the production proxy flow but exercises the same IK / grasp
    # logic the proxy uses.
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc")
    go2.connect()
    piper = MuJoCoPiper(go2); piper.connect()
    gripper = MuJoCoPiperGripper(go2); gripper.connect()
    time.sleep(0.5)  # let dog stand + physics settle

    # Teleport the dog next to the pick_table (at x=11.0) so objects are
    # in arm reach. The verify script isn't meant to test walking — that's
    # exercised by Walk/Navigate skill tests elsewhere. We just want a
    # deterministic starting pose for the grasp pipeline.
    import mujoco as _mj
    go2._pause_physics()
    try:
        d = go2._mj.data
        d.qpos[0] = 10.65  # 35 cm behind table — all 3 objects reachable
        d.qpos[1] = 2.95   # 5 cm off the row centreline so the dog's
                           # trunk teleport doesn't induce a contact
                           # impulse on the bottle at y=3.00

        d.qpos[2] = 0.28
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quat (face +X)
        d.qvel[:] = 0.0
        _mj.mj_forward(go2._mj.model, d)
    finally:
        go2._resume_physics()
    time.sleep(0.8)  # let everything settle after teleport

    try:
        # Snapshot the object's settled z BEFORE the pick — used as the
        # baseline for lift measurement so MJCF edits don't silently break
        # the test.
        m = go2._mj.model
        d = go2._mj.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
        init_z_measured = float(d.body(bid).xpos[2])

        wm = WorldModel()
        SimStartTool._populate_pickables_from_mjcf(wm, go2._scene_xml_path)

        ctx = SkillContext(
            arm=piper, gripper=gripper, base=go2,
            world_model=wm, config={{}},
        )
        result = PickTopDownSkill().execute({{"object_id": OBJ}}, ctx)

        final_z = float(d.body(bid).xpos[2])
        lift_cm = (final_z - init_z_measured) * 100
        held = bool(result.result_data.get("grasped_heuristic"))

        print(f"RESULT obj={{OBJ}} success={{result.success}} "
              f"init_z={{init_z_measured:.3f}} final_z={{final_z:.3f}} "
              f"lift={{lift_cm:+.2f}}cm held={{held}} "
              f"diag={{result.result_data.get('diagnosis')}}")
        sys.exit(0 if result.success else 2)
    finally:
        try: gripper.disconnect()
        except Exception: pass
        try: piper.disconnect()
        except Exception: pass
        try: go2.disconnect()
        except Exception: pass
""").strip()


def _run_one(obj_id: str, repo: Path, verbose: bool) -> tuple[bool, str]:
    """Run a single pick in a fresh subprocess. Returns (ok, summary)."""
    child_script = _CHILD_TEMPLATE.format(obj_id=obj_id)
    venv_py = repo / ".venv-nano" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable

    proc = subprocess.run(
        [py, "-u", "-c", child_script],
        capture_output=True, text=True, cwd=str(repo), timeout=120,
    )
    if verbose:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            # Parse key=value pairs
            parts = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            lift = float(parts.get("lift", "0").rstrip("cm"))
            held = parts.get("held", "False") == "True"
            ok = proc.returncode == 0 and lift >= _MIN_LIFT_CM and held
            return ok, line
    # No RESULT line — crashed before skill finished
    code_txt = {139: "SEGFAULT", 124: "TIMEOUT"}.get(proc.returncode, str(proc.returncode))
    return False, f"RESULT obj={obj_id} CRASHED exit={code_txt}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1,
                        help="how many times to run each object (default 1)")
    parser.add_argument("--object",
                        help="only run this object id (default: all three)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream child process output")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    targets = _TARGETS
    if args.object:
        targets = [o for o in _TARGETS if o == args.object]
        if not targets:
            print(f"unknown object id: {args.object!r}", file=sys.stderr)
            return 2

    print(f"=== verify_pick_top_down: {len(targets)} objects × {args.repeat} repeats ===")
    results: list[tuple[str, bool, str]] = []
    for obj_id in targets:
        for i in range(args.repeat):
            ok, summary = _run_one(obj_id, repo, args.verbose)
            print(f"[{i+1}/{args.repeat}] {summary}  {'PASS' if ok else 'FAIL'}")
            results.append((obj_id, ok, summary))

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"=== pass={passed}/{total} ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
