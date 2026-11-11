# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WP04 dead-track sweep: the retired tracks stay retired (git is the archive).

Each row names the ledger bug that lived in the deleted code; that code is the fix.
Reviving a track means a reviewed change that deletes its row here — never a stray
import or a file copied back. ``git log --all -- <path>`` recovers any of it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

# path -> the ledger key(s) closed by deleting it
_DEAD: dict[str, str] = {
    "vector_os_nano/hardware/sim/go2_ros2_proxy.py":
        "sim-arm:missed2 (stop() not an e-stop while the nav flag is armed), "
        "perception:missed2 (frozen frames, no freshness gate), the _publish_markers rclpy abort",
    "vector_os_nano/hardware/sim/piper_ros2_proxy.py": "the sim ROS2 producer (WP03 flipped NL launch in-process)",
    "vector_os_nano/hardware/sim/isaac_sim_proxy.py": "periphery:isaac-healthcheck-humble",
    "vector_os_nano/hardware/sim/isaac_sim_arm_proxy.py": "periphery:isaac-healthcheck-humble (ik() always None)",
    "vector_os_nano/hardware/sim/gazebo_go2_proxy.py": "dead gazebo env (no gz/go2_description)",
    "vector_os_nano/hardware/sim/pybullet_arm.py": "sim-arm:pybullet-ik-never-fails",
    "vector_os_nano/hardware/sim/pybullet_gripper.py": "sim-arm:pybullet-ik-never-fails (floor contacts = holding)",
    "vector_os_nano/hardware/sim/sensors/lidar360.py": "sim-locomotion:dead-sysnav-sensor-defects (no self-filter)",
    "vector_os_nano/hardware/sim/sensors/pano360.py": "sim-locomotion:dead-sysnav-sensor-defects (global fovy)",
    "vector_os_nano/hardware/sim/sensors/gt_odom.py": "sim-locomotion:dead-sysnav-sensor-defects (wall clock)",
    "vector_os_nano/integrations/sysnav_bridge": "periphery:sysnav-viewpoint-and-disconnect",
    "vector_os_nano/vcli/tools/sysnav_sim_tool.py": "periphery:sysnav-viewpoint-and-disconnect (unregistered)",
    "vector_os_nano/ros2": "periphery:ros2-agentnode-skillserver-dead-api, periphery:perception-node-topic-as-query, "
                           "periphery:hardware-bridge-trajectory-timing",
    "vector_os_nano/core/nav_client.py": "never constructed (services['nav'] is never populated)",
    "docker/isaac-sim": "periphery:isaac-healthcheck-humble",
    "gazebo": "dead gazebo env",
    "examples": "periphery:examples-all-broken",
    "scripts/verify_loco_pick_place.py": "periphery:verify-loco-self-report",
    "scripts/verify_pick_top_down.py": "calls the deleted SimStartTool._populate_pickables_from_mjcf",
    "scripts/launch_explore.sh": "periphery B-1/B-2 (hardcoded /home/yusen, cp into the shared nav stack)",
    "vector_os_nano/hardware/sim/mjcf/g1/g1.xml": "35 absolute /home/yusen mesh paths, unreferenced",
    "vector_os_nano/hardware/sim/mjcf/piper/scene.xml": "unused menagerie demo",
    # R1: the native tool-use loop is the only producer; the legacy VGG track is gone.
    "vector_os_nano/vcli/cognitive/goal_executor.py":
        "legacy-cognitive:B6-posthoc-timeout, B10-foreach-verify-splice, B15-fail-action-semantics, "
        "B16-non-dict-result-data, core:two-skillcontext-builders",
    "vector_os_nano/vcli/cognitive/goal_decomposer.py":
        "legacy-cognitive:B9-decompose-raises, B12-template-hijack, B22-class-default-drift, small-dead-code",
    "vector_os_nano/vcli/cognitive/vgg_harness.py":
        "legacy-cognitive:harness-runs-steps-after-dep-failure, harness-retry-keyword-reroute, "
        "B3-retry-keyword-reroute, pipeline-retry-replays-succeeded-steps, legacy-retry-ignores-stop, "
        "B25-harness-wrong-trace-stale-bb, harness-dead-config-and-hooks",
    "vector_os_nano/vcli/cognitive/strategy_selector.py": "legacy-cognitive:locomotion-primitives-uninitialized",
    "vector_os_nano/vcli/cognitive/strategy_stats.py": "legacy-cognitive:B17-stats-starvation",
    "vector_os_nano/vcli/cognitive/code_executor.py":
        "legacy-cognitive:CE-sandbox-unsafe, code-executor-inprocess-exec, tests:runaway-sandbox-thread",
    "vector_os_nano/vcli/cognitive/tool_dispatcher.py": "legacy-cognitive:B21-tool-output-dropped",
    "vector_os_nano/vcli/cognitive/template_library.py":
        "legacy-cognitive:B20-loose-template-match, producers:answer-traces-compiled-templates",
    "vector_os_nano/vcli/cognitive/experience_compiler.py": "producers:answer-traces-compiled-templates",
    "vector_os_nano/vcli/cognitive/vocab_from_registry.py": "legacy-cognitive:B18-fallback-verify-typeerror",
    "vector_os_nano/vcli/cognitive/object_memory.py": "legacy-cognitive:object-memory-stale-snapshot",
    "vector_os_nano/vcli/cognitive/visual_verifier.py": "spine:visual-keyword-overlap",
    "vector_os_nano/vcli/cognitive/predict.py": "untagged verify reader only the legacy vocab taught",
    "vector_os_nano/vcli/cognitive/blackboard.py": "the legacy executor's foreach binding",
    "vector_os_nano/vcli/cognitive/observation.py": "the legacy plan renderer",
    "vector_os_nano/vcli/cognitive/capabilities/chat.py": "the chat capability only the legacy router used",
    "vector_os_nano/vcli/intent_router.py": "legacy-cognitive:intent-router-substring-keywords",
    "vector_os_nano/vcli/primitives": "tools-worlds:primitives-track-dead, tools-worlds:primitives-dead-broken",
    "vector_os_nano/playground":
        "tools-worlds:playground-duplicate-world, sim-runtime:playground-dangling-scene-path",
    "vector_os_nano/vcli/eval_runner.py": "cli-entry:eval-runner-legacy-undocumented",
    "vector_os_nano/vcli/hooks.py": "cli-entry:hooks-without-registrants, producers:hooks-never-registered",
    "vector_os_nano/vcli/turn_status.py": "the ReAct turn's live region (run_turn is gone)",
    "vector_os_nano/vcli/dynamic_prompt.py": "the ReAct turn's system prompt (run_turn is gone)",
}

# dotted modules nothing may import (or import_module) any more
_DEAD_MODULES = {
    "vector_os_nano.hardware.sim." + m
    for m in ("go2_ros2_proxy", "piper_ros2_proxy", "isaac_sim_proxy", "isaac_sim_arm_proxy",
              "gazebo_go2_proxy", "pybullet_arm", "pybullet_gripper",
              "sensors.lidar360", "sensors.pano360", "sensors.gt_odom")
} | {
    "vector_os_nano.integrations", "vector_os_nano.ros2", "vector_os_nano.core.nav_client",
    "vector_os_nano.vcli.tools.sysnav_sim_tool",
} | {
    "vector_os_nano.vcli.cognitive." + m
    for m in ("goal_executor", "goal_decomposer", "vgg_harness", "strategy_selector",
              "strategy_stats", "code_executor", "tool_dispatcher", "template_library",
              "experience_compiler", "vocab_from_registry", "object_memory", "visual_verifier",
              "predict", "blackboard", "observation", "capabilities.chat")
} | {
    "vector_os_nano.vcli." + m
    for m in ("intent_router", "primitives", "eval_runner", "hooks", "turn_status", "dynamic_prompt")
} | {"vector_os_nano.playground"}

# R1: names of the deleted producer that nothing may define again (one producer).
_DEAD_ENGINE_API = (
    "run_turn", "run_turn_unified", "vgg_decompose", "vgg_execute", "try_vgg",
    "classify_intent", "init_vgg",
)
# R1: the switches and flags of the deleted second producer.
_DEAD_SWITCHES = (
    "VECTOR_NATIVE_LOOP", "VECTOR_NATIVE_FIRST", "VECTOR_REPL_NATIVE", "VECTOR_PRINT_NATIVE",
    "VECTOR_LEGACY_TURN", "--native-loop", "--native-first", "--scenario", "vector-eval",
)


def test_one_producer() -> None:
    from vector_os_nano.vcli.engine import VectorEngine

    back = [n for n in _DEAD_ENGINE_API if hasattr(VectorEngine, n)]
    assert not back, f"VectorEngine has a second producer again: {back}"


def test_dead_switches_are_gone() -> None:
    hits = []
    for p in [*sorted((_REPO / "vector_os_nano").rglob("*.py")), _REPO / "pyproject.toml"]:
        text = p.read_text(encoding="utf-8")
        hits += [f"{p.relative_to(_REPO)}: {s}" for s in _DEAD_SWITCHES if s in text]
    assert not hits, hits


@pytest.mark.parametrize("path", sorted(_DEAD))
def test_dead_track_is_deleted(path: str) -> None:
    assert not (_REPO / path).exists(), f"{path} is back; it held: {_DEAD[path]}"


def _dead(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in _DEAD_MODULES)


def test_nothing_imports_a_dead_module() -> None:
    hits = []
    for top in ("vector_os_nano", "scripts", "tools", "tests"):
        for p in sorted((_REPO / top).rglob("*.py")):
            if p.resolve() == Path(__file__).resolve():
                continue
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names = [node.value]  # importlib.import_module("...") / sys.modules keys
                else:
                    continue
                hits += [f"{p.relative_to(_REPO)}:{node.lineno} {n}" for n in names if _dead(n)]
    assert not hits, hits


def test_pyproject_drops_the_dead_backend_deps() -> None:
    import tomllib

    extras = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    deps = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for v in extras.values() for d in v}
    assert not deps & {"open3d"}  # pybullet is back as sim/backends/pybullet (owner decision, release 1.0)
