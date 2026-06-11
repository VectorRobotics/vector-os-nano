#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M4 acceptance — the FULL SysNav graph on the photoreal habitat world.

Orchestrates four processes and probes the result:

  habitat feed (this repo)      → /camera/image /registered_scan /state_estimation
  detection_node (SysNav)       → DetectionResult        [YOLOE TensorRT]
  semantic_mapping_node (SysNav)→ /object_nodes_list     [SAM2]
  probe (this script)           → LiveSysnavBridge → WorldModel

PASS = WorldModel receives non-empty, labelled object nodes (the campaign
M4 criterion). Run from the repo root with ROS2 Jazzy + the SysNav install
sourced:

    source /opt/ros/jazzy/setup.bash && source ~/Desktop/SysNav/install/setup.bash
    .venv/bin/python scripts/verify_habitat_sysnav.py [--timeout 300]

The SysNav workspace stays a SIBLING (PolyForm-NC): we launch its installed
nodes, never copy its code.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSNAV = Path(os.environ.get("VECTOR_SYSNAV_WS", str(Path.home() / "Desktop" / "SysNav")))
VENV_PY = str(REPO / ".venv" / "bin" / "python")
SCENE = os.environ.get("VECTOR_HABITAT_SCENE_REF", "habitat-test-scenes/apartment_1.glb")


def _spawn(name: str, cmd: str, cwd: Path, env: dict, logdir: Path) -> subprocess.Popen:
    log = open(logdir / f"{name}.log", "w")
    proc = subprocess.Popen(
        ["bash", "-c", cmd], cwd=str(cwd), env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"  [{name}] pid={proc.pid} log={log.name}")
    return proc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds to wait for non-empty object nodes")
    args = ap.parse_args()

    run_id = f"m4-{uuid.uuid4().hex[:8]}"
    logdir = Path("/tmp") / f"vector_m4_{run_id}"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"M4 acceptance run {run_id} — logs in {logdir}")

    env = dict(os.environ)
    env["VECTOR_RUN_ID"] = run_id
    sam2_path = str(SYSNAV / "src/semantic_mapping/semantic_mapping/external/sam2")
    env_sysnav = dict(env)
    env_sysnav["PYTHONPATH"] = sam2_path + ":" + env.get("PYTHONPATH", "")

    procs: list[subprocess.Popen] = []
    try:
        procs.append(_spawn(
            "habitat_feed",
            f"{VENV_PY} -m vector_os_nano.playground.habitat.sysnav_bridge "
            f"--scene {SCENE} --hz 2 --wander",
            REPO, env, logdir,
        ))
        params = str(SYSNAV / "install/semantic_mapping/share/semantic_mapping/mapping_mecanum_sim.yaml")
        procs.append(_spawn(
            "detection_node",
            f"{VENV_PY} -c 'from semantic_mapping.detection_node import main; main()' "
            f"--ros-args --params-file {params}",
            SYSNAV, env_sysnav, logdir,
        ))
        procs.append(_spawn(
            "semantic_mapping_node",
            f"{VENV_PY} -c 'from semantic_mapping.semantic_mapping_node import main; main()' "
            f"--ros-args --params-file {params}",
            SYSNAV, env_sysnav, logdir,
        ))

        # ---- probe: the v2.4 consumption path, now fed by the REAL graph ----
        from vector_os_nano.core.world_model import WorldModel
        from vector_os_nano.integrations.sysnav_bridge.live_bridge import LiveSysnavBridge

        wm = WorldModel()
        bridge = LiveSysnavBridge(world_model=wm)
        if not bridge.start():
            print("FAIL: LiveSysnavBridge could not start (source the SysNav install?)")
            return 2

        deadline = time.time() + args.timeout
        last_report = 0.0
        while time.time() < deadline:
            objs = wm.get_objects()
            if objs and any(o.label for o in objs):
                print("\nM4 ACCEPTANCE PASS — SysNav scene graph produced object nodes:")
                for o in objs:
                    print(f"  {o.object_id:14s} {o.label:14s} "
                          f"({o.x:+.2f}, {o.y:+.2f}, {o.z:+.2f}) conf={o.confidence:.2f}")
                return 0
            if time.time() - last_report > 15:
                alive = all(p.poll() is None for p in procs)
                print(f"  ... waiting ({int(deadline - time.time())}s left, "
                      f"procs alive={alive}, objects={len(objs)})")
                if not alive:
                    for p, name in zip(procs, ("habitat_feed", "detection", "mapping")):
                        if p.poll() is not None:
                            print(f"FAIL: {name} exited rc={p.returncode} — see {logdir}")
                    return 3
                last_report = time.time()
            time.sleep(1.0)
        print(f"FAIL: no object nodes within {args.timeout}s — see {logdir}")
        return 1
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2.0)
        try:
            from vector_os_nano.vcli.watchdog import sweep_orphans
            swept = sweep_orphans(run_id)
            if swept:
                print(f"  watchdog swept {len(swept)} stray pid(s)")
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
