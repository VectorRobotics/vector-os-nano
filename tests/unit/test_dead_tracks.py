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
    # R8a periphery sweep (scripts/gt_grade.py is the one survivor of tools/)
    "tools": "periphery:tools-kill-by-name-and-rosm (pkill -9 -f mujoco, rosm nuke), "
             "periphery:tools-stale-ik-probes, periphery:tools-bypass-product-fixed-tmp",
    "sim.sh": "cli-entry:alias-scripts-redundant",
    "scripts/vector-sim": "cli-entry:alias-scripts-redundant (.venv-nano fallback)",
    "scripts/harness_run.py": "periphery:scripts-dead-dev-tools (a second gate)",
    "scripts/add_spdx_headers.py": "periphery:scripts-dead-dev-tools",
    ".mcp.json": "docs-packaging:dotmcp-hardcoded-paths (/home/<user> paths, a personal server)",
    "vector_os_nano/hardware/ros2/runtime.py": "periphery:hw-ros2-runtime-dead",
    "config/far_go2_indoor.yaml": "periphery:config-dead-planner-yamls (no launcher reads it)",
    "config/local_planner_go2.yaml": "periphery:config-dead-planner-yamls (no launcher reads it)",
    "tests/e2e": "tests-gate-e2e:e2e-dir-stale-direct-api (a hand-built rig, not the bare REPL)",
    "vector_os_nano/vcli/tools/ros2_tools.py": "tools-worlds:ros2-tools-broken-and-dead-logs, "
        "tools-worlds:ros2-tools-jazzy-broken (hz never returned, echo used a flag Jazzy lacks, "
        "the log tool read files nobody writes)",
    "tests/harness/test_path_follower.py": "tests-inventory:path-follower-tautology (tested a copy of the logic)",
    "vector_os_nano/hardware/sim/objects": "unreferenced byte-identical copies of hardware/urdf/meshes/*.stl",
    "vector_os_nano/hardware/urdf/so101_sim.urdf": "the PyBullet arm's URDF, unreferenced since WP04 deleted that arm",
    "foxglove/launch_foxglove.sh": "periphery:foxglove-check-before-source (open_foxglove launches the bridge itself)",
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
    # R2: the hidden mini-planner (a skill runs alone; the model sequences the rest).
    "vector_os_nano/core/executor.py":
        "core:auto-home-eats-tool-payload, core:diagnosis-code-lost-in-executor, core:executor-no-cancel, "
        "core:executor-uncaught-errors, skills-manip:home-after-detect-drops-held-object",
    # R2: dead skills and the dead code only they used.
    "vector_os_nano/skills/mobile_pick.py": "skills-manip:mobile-pick-place-nonfunctional",
    "vector_os_nano/skills/mobile_place.py": "skills-manip:mobile-pick-place-nonfunctional",
    "vector_os_nano/skills/utils/place_region.py": "unreachable (no caller)",
    "vector_os_nano/skills/utils/terminal_dock.py":
        "skills-nav:terminal-dock-dead-track, skills-manip:dock-pose-param-exposed",
    "vector_os_nano/skills/calibration.py":
        "a second camera->base loader beside perception.calibration (silent identity: "
        "perception:calibration-missing-silent-camera-frame)",
    "vector_os_nano/core/spatial_memory.py": "core:dead-scene-graph-and-worldmodel-api",
    "vector_os_nano/vcli/backends/text_llm_adapter.py":
        "core:dead-scene-graph-and-worldmodel-api (rank_rooms_for_goal's adapter, never constructed)",
    "tests/harness/test_level52_doorchain_navstack.py": "skills-nav:dead-doorchain-proxy-branch",
    "tests/harness/test_level4_patrol.py": "always skipped: it needed the deleted ROS2 proxy",
    "tests/unit/test_calibration_transform.py": "always skipped: no calibration file ships",
}

# dotted modules nothing may import (or import_module) any more
_DEAD_MODULES = {
    "vector_os_nano.hardware.sim." + m
    for m in ("go2_ros2_proxy", "piper_ros2_proxy", "isaac_sim_proxy", "isaac_sim_arm_proxy",
              "gazebo_go2_proxy", "pybullet_arm", "pybullet_gripper",
              "sensors.lidar360", "sensors.pano360", "sensors.gt_odom")
} | {
    "vector_os_nano.integrations", "vector_os_nano.ros2", "vector_os_nano.core.nav_client",
    "vector_os_nano.vcli.tools.sysnav_sim_tool", "vector_os_nano.hardware.ros2.runtime",
    "vector_os_nano.vcli.tools.ros2_tools",
} | {
    "vector_os_nano.vcli.cognitive." + m
    for m in ("goal_executor", "goal_decomposer", "vgg_harness", "strategy_selector",
              "strategy_stats", "code_executor", "tool_dispatcher", "template_library",
              "experience_compiler", "vocab_from_registry", "object_memory", "visual_verifier",
              "predict", "blackboard", "observation", "capabilities.chat")
} | {
    "vector_os_nano.vcli." + m
    for m in ("intent_router", "primitives", "eval_runner", "hooks", "turn_status", "dynamic_prompt")
} | {"vector_os_nano.playground", "vector_os_nano.core.executor", "vector_os_nano.core.spatial_memory",
     "vector_os_nano.vcli.backends.text_llm_adapter", "vector_os_nano.skills.calibration"} | {
    "vector_os_nano.skills." + m for m in ("mobile_pick", "mobile_place", "utils.place_region",
                                           "utils.terminal_dock")
}

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
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.vcli.engine import VectorEngine

    back = [n for n in _DEAD_ENGINE_API if hasattr(VectorEngine, n)]
    assert not back, f"VectorEngine has a second producer again: {back}"
    # core:alias-prefix-misroute: no keyword matcher routes user text to a skill.
    assert not hasattr(SkillRegistry, "match")


def test_the_dead_core_api_stays_deleted() -> None:
    """R2, core:dead-scene-graph-and-worldmodel-api and the mini-planner's leftovers."""
    import inspect

    from vector_os_nano.core import agent, config, scene_graph, types
    from vector_os_nano.skills import navigate, grasp

    dead = {
        scene_graph.SceneGraph: ("get_location", "get_all_locations", "remember_location",
                                 "get_unvisited_rooms", "rank_rooms_for_goal"),
        agent.Agent: ("stop", "home", "register_skill"),
        navigate.NavigateSkill: ("_dead_reckoning", "_navigate_with_proxy"),
        grasp.GraspSkill: ("_resolve_dock_pose",),
    }
    back = [f"{c.__name__}.{n}" for c, names in dead.items() for n in names if hasattr(c, n)]
    back += [f"{m.__name__}.{n}" for m, names in (
        (config, ("validate_config", "get_section")),
        (types, ("ActionRecord", "GoalResult", "TaskPlan", "TaskStep", "StepTrace", "ExecutionResult")),
        (scene_graph, ("TextLLM",)),
    ) for n in names if hasattr(m, n)]
    params = inspect.signature(agent.Agent).parameters
    back += [f"Agent({k}=)" for k in ("llm", "llm_api_key") if k in params]
    back += ["perception_grasp dock_pose"] * ("dock_pose" in grasp.GraspSkill.parameters)
    # the deleted producer's skill metadata (no reader since R1)
    from vector_os_nano.skills import get_piper_skills, get_so101_skills
    from vector_os_nano.skills.go2 import get_go2_skills

    back += [f"{type(s).__name__}.{a}" for s in (*get_so101_skills(), *get_go2_skills(), *get_piper_skills())
             for a in ("typical_duration_sec", "__skill_direct__", "__skill_auto_steps__") if hasattr(s, a)]
    assert not back, back


def test_dead_switches_are_gone() -> None:
    hits = []
    for p in [*sorted((_REPO / "vector_os_nano").rglob("*.py")), _REPO / "pyproject.toml"]:
        text = p.read_text(encoding="utf-8")
        hits += [f"{p.relative_to(_REPO)}: {s}" for s in _DEAD_SWITCHES if s in text]
    assert not hits, hits


def _is_back(p: Path) -> bool:
    """A deleted path is back as a file, or as a directory holding anything but stale
    bytecode (git leaves the ignored __pycache__ of a deleted package in a checkout)."""
    if p.is_dir():
        return any(f.is_file() and "__pycache__" not in f.parts for f in p.rglob("*"))
    return p.exists()


@pytest.mark.parametrize("path", sorted(_DEAD))
def test_dead_track_is_deleted(path: str) -> None:
    assert not _is_back(_REPO / path), f"{path} is back; it held: {_DEAD[path]}"


def test_stale_bytecode_is_not_a_dead_track_come_back(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__" / "old.cpython-312.pyc").write_bytes(b"")
    assert not _is_back(tmp_path / "pkg")
    (tmp_path / "pkg" / "new.py").write_text("")
    assert _is_back(tmp_path / "pkg")
    assert _is_back(tmp_path / "pkg" / "new.py") and not _is_back(tmp_path / "gone")


def _dead(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in _DEAD_MODULES)


def test_nothing_imports_a_dead_module() -> None:
    hits = []
    for top in ("vector_os_nano", "scripts", "tests"):
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
