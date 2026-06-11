#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M5 — the flagship third-world demo: a VISIBLE, self-correcting long chain.

One process owns the photoreal agent; SysNav perceives; the VGG closed loop
drives everything from natural language:

  habitat (conda subprocess) ←─ navigate/walk ──┐
      │ pano/cloud/odom (in-process bridge)     │ VGG: decompose → execute →
      ▼                                         │ deterministic verify → replan
  SysNav detection+mapping (subprocesses)       │
      ▼ /object_nodes_list                      │
  agent.world_model ──→ world_context "Objects (live)" ──→ LLM binding

Artifacts land in ~/sandbox/m5_demo/<run>/: per-instruction verified traces
(the observation surface), egocentric frames at key moments, the live object
list, and a SUMMARY.md. Honest-failure exhibit included: an instruction for
an object the scene does not contain must FAIL LOUDLY, not false-pass.

Run (repo root, ROS2 Jazzy + SysNav install sourced, VECTOR_LIVE_LLM creds):
    .venv/bin/python scripts/demo_third_world.py [--skip-sysnav]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSNAV = Path(os.environ.get("VECTOR_SYSNAV_WS", str(Path.home() / "Desktop" / "SysNav")))
SYSNAV_PY = str(SYSNAV / ".venv-sysnav" / "bin" / "python")
SCENE_REF = os.environ.get("VECTOR_HABITAT_SCENE_REF", "habitat-test-scenes/apartment_1.glb")


def _spawn(name: str, cmd: str, cwd: Path, env: dict, logdir: Path) -> subprocess.Popen:
    log = open(logdir / f"{name}.log", "w")
    proc = subprocess.Popen(
        ["bash", "-c", cmd], cwd=str(cwd), env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"  [{name}] pid={proc.pid}")
    return proc


def _spawn_sysnav_nodes(env: dict, logdir: Path) -> list[subprocess.Popen]:
    pkg = str(SYSNAV / "src/semantic_mapping")
    sam2 = str(SYSNAV / "src/semantic_mapping/semantic_mapping/external/sam2")
    e = dict(env)
    e["PYTHONPATH"] = pkg + ":" + sam2 + ":" + env.get("PYTHONPATH", "")
    params = str(SYSNAV / "install/semantic_mapping/share/semantic_mapping/mapping_mecanum_sim.yaml")
    return [
        _spawn("detection_node",
               f"{SYSNAV_PY} -c 'from semantic_mapping.detection_node import main; main()' "
               f"--ros-args --params-file {params}", SYSNAV, e, logdir),
        _spawn("semantic_mapping_node",
               f"{SYSNAV_PY} -c 'from semantic_mapping.semantic_mapping_node import main; main()' "
               f"--ros-args --params-file {params}", SYSNAV, e, logdir),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sysnav", action="store_true",
                    help="run without the SysNav graph (no semantic objects)")
    ap.add_argument("--warmup", type=float, default=90.0,
                    help="seconds of wandering for the mapper to populate")
    args = ap.parse_args()

    run_id = f"m5-{uuid.uuid4().hex[:8]}"
    outdir = Path.home() / "sandbox" / "m5_demo" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VECTOR_RUN_ID", run_id)
    print(f"M5 demo {run_id} — artifacts in {outdir}")

    import rclpy

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.integrations.sysnav_bridge.live_bridge import LiveSysnavBridge
    from vector_os_nano.playground.catalog import get_scenario
    from vector_os_nano.playground.habitat import HabitatBase
    from vector_os_nano.playground.habitat.scenes import resolve_scene_ref
    from vector_os_nano.playground.habitat.sysnav_bridge import HabitatSysnavBridge
    from vector_os_nano.playground.world import PlaygroundWorld
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.skills.go2.turn import TurnSkill
    from vector_os_nano.skills.go2.walk import WalkSkill
    from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill
    from vector_os_nano.vcli.backends import create_backend
    from vector_os_nano.vcli.config import resolve_credentials
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.intent_router import IntentRouter

    procs: list[subprocess.Popen] = []
    feed_executor = None
    try:
        # --- the agent in the photoreal world --------------------------------
        base = HabitatBase(scene=resolve_scene_ref(SCENE_REF))
        base.connect()
        agent = Agent(base=base)
        registry = SkillRegistry()
        for s in (WalkSkill(), TurnSkill(), StopSkill(), NavigateToPointSkill()):
            registry.register(s)
        agent._skill_registry = registry

        # --- perception: same base feeds SysNav; nodes fill the world model --
        if not args.skip_sysnav:
            feed = HabitatSysnavBridge(base, hz=2.0)
            from rclpy.executors import MultiThreadedExecutor

            feed_executor = MultiThreadedExecutor()
            feed_executor.add_node(feed.node)
            threading.Thread(target=feed_executor.spin, daemon=True,
                             name="sysnav-feed-spin").start()
            procs += _spawn_sysnav_nodes(dict(os.environ), outdir)
            consumer = LiveSysnavBridge(world_model=agent._world_model)
            assert consumer.start(), "LiveSysnavBridge failed to start"

            print(f"  warming up the semantic mapper ({args.warmup:.0f}s of wandering)...")
            import random

            t_end = time.time() + args.warmup
            while time.time() < t_end:
                here = base.get_position()
                tgt = base.snap_point([
                    here[0] + random.uniform(-4, 4),
                    here[1] + random.uniform(-4, 4), here[2],
                ])
                base.navigate_to(tgt[0], tgt[1])
                time.sleep(1.0)
                objs = agent._world_model.get_objects()
                if objs:
                    print(f"    mapper sees {len(objs)} object(s): "
                          f"{sorted({o.label for o in objs})}")
            objs = agent._world_model.get_objects()
            (outdir / "objects.json").write_text(json.dumps(
                [dataclasses.asdict(o) for o in objs], indent=1, default=str))
            if not objs:
                print("WARN: no semantic objects after warmup — continuing anyway")

        # --- the closed-loop engine ------------------------------------------
        api_key, provider, model, base_url = resolve_credentials()
        if not api_key:
            print("FAIL: no LLM credentials"); return 2
        engine = VectorEngine(
            create_backend(provider, api_key=api_key, model=model, base_url=base_url),
            intent_router=IntentRouter(),
        )
        engine.init_vgg(agent=agent, skill_registry=registry,
                        world=PlaygroundWorld(get_scenario("apartment")))
        assert engine._vgg_enabled

        from vector_os_nano.vcli.cognitive.observation import run_snapshot
        from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

        labels = sorted({o.label for o in agent._world_model.get_objects()
                         if o.label and o.label != "unknown"})
        target_a = labels[0] if labels else None
        target_b = labels[1] if len(labels) > 1 else target_a

        instructions: list[tuple[str, str, bool]] = []
        if target_a:
            instructions.append(("semantic-goto", f"走到{target_a}那里", True))
        if target_b and target_b != target_a:
            instructions.append(
                ("long-chain", f"先去{target_b}那边看看，然后回到{target_a}附近", True))
        instructions += [
            ("relative-motion", "向前走半米然后左转90度", True),
            # honest-failure exhibit: the scene has no such object — the loop
            # must fail LOUDLY (or replan to a real failure), never false-pass.
            ("honest-failure", "走到游泳池旁边", False),
        ]

        summary_lines = [f"# M5 demo {run_id}", "",
                         f"scene: {SCENE_REF}", f"objects: {labels}", ""]
        ok_count = 0
        for tag, instr, expect_ok in instructions:
            print(f"\n=== [{tag}] {instr!r}")
            tree = engine.vgg_decompose(instr)
            if tree is None:
                outcome = "no-plan"
                trace = None
            else:
                trace = engine.vgg_execute(tree)
                outcome = "verified-done" if trace.success else "failed"
            ev = bool(trace and evidence_passed(trace))
            frame = outdir / f"{tag}.png"
            try:
                frame.write_bytes(base.render_rgb_png())
            except Exception:  # noqa: BLE001
                pass
            if trace is not None:
                (outdir / f"{tag}.trace.json").write_text(
                    json.dumps(run_snapshot(trace), indent=1, default=str))
            matched = (outcome == "verified-done") == expect_ok
            ok_count += int(matched)
            line = (f"[{tag}] {instr!r} → {outcome} (evidence={ev}) "
                    f"{'AS EXPECTED' if matched else 'UNEXPECTED'}")
            print("  " + line)
            summary_lines.append("- " + line)

        verdict = "PASS" if ok_count == len(instructions) else "PARTIAL"
        summary_lines += ["", f"verdict: {verdict} ({ok_count}/{len(instructions)})"]
        (outdir / "SUMMARY.md").write_text("\n".join(summary_lines))
        print(f"\nM5 {verdict} — artifacts: {outdir}")
        return 0 if verdict == "PASS" else 1
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        try:
            if feed_executor is not None:
                feed_executor.shutdown(timeout_sec=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            base.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            from vector_os_nano.vcli.watchdog import sweep_orphans
            sweep_orphans(run_id)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
