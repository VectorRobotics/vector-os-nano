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

# Each tuple = (object_id, init_z_on_table)
_TARGETS: list[tuple[str, float]] = [
    ("pickable_bottle_blue",  0.279),
    ("pickable_bottle_green", 0.249),
    ("pickable_can_red",      0.244),
]

# Minimum vertical lift (in cm) to count as a successful grasp.
_MIN_LIFT_CM: float = 1.0

# Child script that actually runs the pick. Kept as inline template so the
# verification works whether the repo is cloned or symlinked anywhere.
_CHILD_TEMPLATE = textwrap.dedent("""
    import os, sys, logging, mujoco
    os.environ["VECTOR_SIM_WITH_ARM"] = "1"
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.core.skill import SkillContext

    OBJ = "{obj_id}"
    INIT_Z = {init_z}

    agent = SimStartTool._start_go2_local(gui=False)
    try:
        skill = PickTopDownSkill()
        ctx = SkillContext(
            arm=agent._arm, gripper=agent._gripper, base=agent._base,
            world_model=agent._world_model, config=agent._config,
        )
        result = skill.execute({{"object_id": OBJ}}, ctx)

        m, d = agent._base._mj.model, agent._base._mj.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ)
        lift_cm = (float(d.body(bid).xpos[2]) - INIT_Z) * 100
        held = bool(result.result_data.get("grasped_heuristic"))

        print(f"RESULT obj={{OBJ}} success={{result.success}} "
              f"lift={{lift_cm:+.2f}}cm held={{held}} "
              f"diag={{result.result_data.get('diagnosis')}}")
        sys.exit(0 if result.success else 2)
    finally:
        try: agent._base.disconnect()
        except Exception: pass
        try: agent._arm.disconnect()
        except Exception: pass
        try: agent._gripper.disconnect()
        except Exception: pass
""").strip()


def _run_one(obj_id: str, init_z: float, repo: Path, verbose: bool) -> tuple[bool, str]:
    """Run a single pick in a fresh subprocess. Returns (ok, summary)."""
    child_script = _CHILD_TEMPLATE.format(obj_id=obj_id, init_z=init_z)
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
        targets = [(o, z) for o, z in _TARGETS if o == args.object]
        if not targets:
            print(f"unknown object id: {args.object!r}", file=sys.stderr)
            return 2

    print(f"=== verify_pick_top_down: {len(targets)} objects × {args.repeat} repeats ===")
    results: list[tuple[str, bool, str]] = []
    for obj_id, init_z in targets:
        for i in range(args.repeat):
            ok, summary = _run_one(obj_id, init_z, repo, args.verbose)
            print(f"[{i+1}/{args.repeat}] {summary}  {'PASS' if ok else 'FAIL'}")
            results.append((obj_id, ok, summary))

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"=== pass={passed}/{total} ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
