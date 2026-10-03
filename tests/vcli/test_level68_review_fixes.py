# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 68 — hardening fixes from the Phase B adversarial review.

Each test pins a confirmed finding's fix so it cannot regress:
- bash deny-list survives --no-permission (intrinsic deny is an unconditional stop)
- file_write/file_edit hard-deny protected paths (write-path parity with read)
- ToolDispatcher ask-resolver fails closed (only y/a allow)
- visual-override steps are NOT deterministic evidence
- concrete tool_call templates require a full name match (no single-token hijack)
- TemplateLibrary() is in-memory (no home-dir writes)
- comment-only code is not a successful execution
- template value substitution is word-boundary aware
- the code sandbox carries only read-only dev predicates (tests_pass is removed)
- vector-eval distinguishes an empty suite from a failed case

Pure kernel logic — no robot, no network, no mujoco fixtures.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from vector_os_nano.vcli.cognitive.code_executor import CodeExecutor
from vector_os_nano.vcli.cognitive.experience_compiler import ExperienceCompiler
from vector_os_nano.vcli.cognitive.template_library import TemplateLibrary
from vector_os_nano.vcli.cognitive.tool_dispatcher import ToolDispatcher
from vector_os_nano.vcli.cognitive.trace_store import evidence_passed, load_trace, save_trace

# Live verify-namespace callable names for the R1 evidence gate (replaces is_robot).
ORACLES = frozenset({
    "at_position", "facing", "visited", "holding_object", "arm_at_home",
    "file_exists", "path_contains", "get_position", "get_heading",
    "describe_scene", "detect_objects", "placed_count", "nearest_room",
    "objects_in_room", "find_object", "room_coverage",
})
from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools.base import CategorizedToolRegistry, ToolContext
from vector_os_nano.vcli.tools.bash_tool import BashTool
from vector_os_nano.vcli.tools.file_tools import FileEditTool, FileWriteTool


def _ctx(tmp: Path) -> ToolContext:
    return ToolContext(agent=None, cwd=tmp, session=None,
                       permissions=PermissionContext(), abort=threading.Event())


# --- Fix A: intrinsic deny survives --no-permission -------------------------


def test_bash_deny_list_survives_no_permission() -> None:
    ctx = PermissionContext(no_permission=True)
    assert ctx.check(BashTool(), {"command": "rm -rf /"}).behavior == "deny"
    # a safe command under no_permission is still allowed
    assert ctx.check(BashTool(), {"command": "ls"}).behavior == "allow"


def test_user_deny_tools_still_overridable_by_no_permission() -> None:
    # The narrower change: a USER deny preference may still be overridden by
    # --no-permission (only intrinsic tool denies are unconditional).
    ctx = PermissionContext(deny_tools={"bash"}, no_permission=True)
    assert ctx.check(BashTool(), {"command": "ls"}).behavior == "allow"


# --- Fix B: dangerous-path guard on write/edit ------------------------------


def test_file_write_denies_protected_path_via_permissions() -> None:
    res = FileWriteTool().check_permissions({"file_path": "~/.ssh/authorized_keys", "content": "x"}, None)
    assert res.behavior == "deny"
    # survives --no-permission too (intrinsic deny)
    ctx = PermissionContext(no_permission=True)
    assert ctx.check(FileWriteTool(), {"file_path": "/etc/shadow", "content": "x"}).behavior == "deny"


def test_file_edit_denies_protected_path() -> None:
    res = FileEditTool().check_permissions(
        {"file_path": "~/.claude/settings.json", "old_string": "a", "new_string": "b"}, None)
    assert res.behavior == "deny"


def test_file_write_execute_blocks_protected_path(tmp_path: Path) -> None:
    r = FileWriteTool().execute({"file_path": "~/.ssh/authorized_keys", "content": "x"}, _ctx(tmp_path))
    assert r.is_error
    assert "protected" in r.content.lower()


# --- Fix D: ask-resolver fails closed ---------------------------------------


def _dispatcher(resolver: Any, tmp: Path) -> ToolDispatcher:
    reg = CategorizedToolRegistry()
    reg.register(FileWriteTool(), category="code")
    return ToolDispatcher(reg, PermissionContext(), allowlist=frozenset({"file_write"}),
                          ask_permission=resolver, cwd=tmp)


def test_ask_resolver_none_return_denies(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    ok, err = _dispatcher(lambda _n, _p: None, tmp_path).dispatch(
        "file_write", {"file_path": str(target), "content": "x"})
    assert ok is False and not target.exists()


def test_ask_resolver_unknown_string_denies(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    ok, _ = _dispatcher(lambda _n, _p: "maybe", tmp_path).dispatch(
        "file_write", {"file_path": str(target), "content": "x"})
    assert ok is False and not target.exists()


def test_ask_resolver_y_allows(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    ok, err = _dispatcher(lambda _n, _p: "y", tmp_path).dispatch(
        "file_write", {"file_path": str(target), "content": "ok\n"})
    assert ok is True and target.read_text() == "ok\n"


# --- Fix E: visual-override is not deterministic evidence -------------------


def _trace_with_override(visual_override: bool) -> ExecutionTrace:
    sg = SubGoal(name="see_cup", description="look for cup", verify="len(detect_objects('cup')) > 0")
    step = StepRecord("see_cup", "look_skill", success=True, verify_result=True,
                      duration_sec=0.1, visual_override=visual_override)
    return ExecutionTrace(goal_tree=GoalTree("find cup", (sg,)), steps=(step,),
                          success=True, total_duration_sec=0.1)


def test_visual_override_is_not_evidence() -> None:
    assert evidence_passed(_trace_with_override(visual_override=True), ORACLES) is False
    # a real deterministic pass on the same shape IS evidence
    assert evidence_passed(_trace_with_override(visual_override=False), ORACLES) is True


def test_visual_override_round_trips(tmp_path: Path) -> None:
    p = save_trace(_trace_with_override(True), tmp_path / "t.json")
    assert load_trace(p).steps[0].visual_override is True


# --- Fix F: concrete tool_call template requires full name match ------------


def _tool_template(name: str):
    tr = ExecutionTrace(
        goal_tree=GoalTree("create config file", (
            SubGoal(name=name, description="d", verify="file_exists('config.txt')",
                    strategy="tool_call",
                    strategy_params={"tool": "file_write", "args": {"file_path": "config.txt", "content": "x"}}),
        )),
        steps=(StepRecord(name, "tool_call", True, True, 0.1),),
        success=True, total_duration_sec=0.1)
    return ExperienceCompiler().compile([tr])[0]


def test_tool_call_template_requires_full_match(tmp_path: Path) -> None:
    lib = TemplateLibrary(persist_path=str(tmp_path / "t.json"))
    lib.add(_tool_template("create_config"))
    # Full subset present -> match.
    assert lib.match("create config now") is not None
    # Only one shared token ("create") -> NO hijack.
    assert lib.match("create a summary report") is None


# --- Fix M: TemplateLibrary() is in-memory ----------------------------------


def test_template_library_in_memory_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    lib = TemplateLibrary()  # no path -> in-memory
    lib.add(_tool_template("create_config"))
    lib.save()  # no-op
    # nothing written under the tmp cwd
    assert list(tmp_path.iterdir()) == []


# --- Fix H: comment-only code is not success --------------------------------


def test_comment_only_code_fails() -> None:
    res = CodeExecutor({}).execute("# step complete\n# all done")
    assert res.success is False
    assert "no executable" in res.error
    # genuinely empty code keeps no-op success semantics
    assert CodeExecutor({}).execute("").success is True
    # real code still succeeds
    assert CodeExecutor({}).execute("x = 1 + 1\nx").success is True


# --- Fix I: word-boundary substitution --------------------------------------


def test_parameterization_is_word_boundary_aware() -> None:
    # Two traces share signature "reach_*" with suffixes "room"/"hall". The value
    # "room" must replace the standalone 'room' in the verify string but must NOT
    # be carved out of the function name nearest_room() (a blind str.replace would
    # corrupt it to nearest_${room}()).
    def _t(place: str) -> ExecutionTrace:
        sg = SubGoal(name=f"reach_{place}", description=f"reach the {place}",
                     verify=f"nearest_room() == '{place}'", strategy="navigate_skill",
                     strategy_params={"target": place})
        return ExecutionTrace(goal_tree=GoalTree(f"reach {place}", (sg,)),
                              steps=(StepRecord(f"reach_{place}", "navigate_skill", True, True, 0.1),),
                              success=True, total_duration_sec=0.1)

    tmpl = ExperienceCompiler().compile([_t("room"), _t("hall")])[0]
    sgt = tmpl.sub_goal_templates[0]
    # The standalone value was parameterized, the function name was preserved.
    assert sgt.verify_pattern == "nearest_room() == '${room}'"
    assert "nearest_${" not in sgt.verify_pattern  # NOT corrupted mid-word


# --- Fix C: code sandbox excludes tests_pass --------------------------------


def test_code_sandbox_excludes_tests_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_DEV_ALLOW_TESTS", "1")
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.worlds import DevWorld

    eng = VectorEngine(backend=MagicMock(), intent_router=MagicMock())
    eng.init_vgg(agent=None, skill_registry=None, world=DevWorld(), persist_dir=tmp_path)
    ns = eng._goal_executor._code_executor._namespace
    assert "tests_pass" not in ns  # WP02 removed tests_pass entirely (an action, not a verify)
    assert "file_exists" in ns      # read-only predicates retained


# --- Fix K: vector-eval empty suite ----------------------------------------


def test_eval_main_empty_suite_distinct_code(tmp_path: Path) -> None:
    from vector_os_nano.vcli.eval_runner import main as eval_main

    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    assert eval_main([str(empty)]) == 2  # not 1 (a failed case) and not 0
