# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 STEP 6 (C) — native producer in the DEV world: the action leak is now CLOSED.

Steps 1-2 + 3A-B proved the native producer covers the go2 ACTION surface (walk,
turn -> at_position, facing). The legacy planner's other job is being WORLD-AGNOSTIC:
the same loop must ACT in the dev world too (write a file, run a tool) and prove it
with a dev verify predicate. Step 3 documented a REAL FINDING — through the REAL
``cli.main`` PTY the native loop offered NO dev action tools, so a dev
``file_write`` was an "Unknown tool" -> RAN / exit 2. STEP 6 CLOSES that leak; this
module now PINS the fix.

THE LEAK (was) AND THE FIX (now):

    ``native_loop._build_motor_tools(agent)`` USED to build the loop's action surface
    from ``wrap_skills(agent)`` ONLY. In the dev world ``runtime.build_runtime`` starts
    no body (no ``--sim``/``--sim-go2``), so there was no agent and no skill registry:
    the loop offered ONLY ``verify`` + ``finish`` and ZERO action tools. STEP 6 extends
    ``_build_motor_tools(agent, engine)`` to ALSO surface the engine registry's
    ``code``-category tools (file_read/file_write/file_edit/bash/glob/grep — the SAME
    set the legacy dev path dispatches via ``DEV_TOOL_ALLOWLIST`` + ``ToolDispatcher``).
    These are real ``Tool`` objects whose ``.execute(params, ctx)`` is the interface
    ``dispatch_skill`` already calls. World-agnostic BY CONSTRUCTION: native asks the
    ENGINE'S registry for its registered action tools — no "if dev" branch in the spine.
    So a dev write-then-verify script through ``cli.main`` now dispatches
    ``file_write`` for real, writes the file, and ``path_contains`` reads True ->
    GROUNDED / verified True / exit 0.

This was never a spine/grading defect (the verify spine, the dev predicates, and the
GROUNDED classification all worked — ``path_contains`` is a PREDICATE oracle and grades
GROUNDED on a True read). It was purely the missing dev ACTION surface; the fix wires
the kernel "code" toolset native already had access to via the engine.

THE PRODUCER IS WORLD-AGNOSTIC (the second test, unchanged, still proves it): when a
dev action is wrapped as a skill onto an agent, ``run_turn_native`` dispatches it, the
file is written, and ``verify(path_contains(...))`` grades GROUNDED / verified True /
exit 0 through the UNTOUCHED spine.

This file needs NO live key and NO sim (it is a dev PTY + an in-process run). Reproduce::

    PATH=/usr/bin:$PATH .venv/bin/python -m pytest \\
      tests/vcli/test_native_loop_devworld_pty.py -v -s
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tests.harness.pty_cli import run_cli_turn


# ---------------------------------------------------------------------------
# (C.1) THE FIX — through the REAL cli.main PTY, the dev world ACTS
# ---------------------------------------------------------------------------


def test_native_devworld_via_cli_main_covers_dev_action_via_native(tmp_path) -> None:
    """A dev write-then-verify script through cli.main -> GROUNDED / exit 0.

    STEP 6 closed the leak: ``_build_motor_tools`` now ALSO surfaces the engine
    registry's ``code``-category tools, so the native loop in the dev world (no agent)
    offers ``file_write`` for real. The script writes ``out.txt`` then verifies with
    ``path_contains`` — the file IS written, the predicate reads True, the spine grades
    GROUNDED, the verdict is verified True / exit 0. The producing strategy is now
    ``file_write`` (the action chain was recorded), proving the dev ACTION surface is
    wired through the REAL product, not just the in-process producer.
    """
    # Run in a fresh tmp cwd so out.txt is a brand-new file (FileWriteTool refuses to
    # overwrite an unread existing file) and the assertion-on-disk is unambiguous.
    out_path = tmp_path / "out.txt"
    r = run_cli_turn(
        "create out.txt containing the word ready",
        tool_script={
            "turns": [
                {"tool_calls": [
                    {"name": "file_write",
                     "input": {"file_path": "out.txt", "content": "ready\n"}}
                ]},
                {"tool_calls": [
                    {"name": "verify",
                     "input": {"expr": "path_contains('out.txt', 'ready')"}}
                ]},
                {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
            ]
        },
        timeout_sec=90.0,
        cwd=tmp_path,
    )
    print(f"\n[devworld coverage] cli.main -> {r.verdict}")
    # The dev action ran through the native loop: verified, GROUNDED, exit 0.
    assert r.verified is True, f"dev write must verify (code tool now wired); {r.verdict}"
    assert r.evidence == "GROUNDED", f"got evidence={r.evidence}; {r.verdict}"
    assert r.exit_code == 0, f"a grounded dev step must exit 0; got {r.exit_code}"
    per_step = r.verdict.get("per_step") or []
    assert per_step, f"expected one write->verify step; got {per_step}"
    step = per_step[0]
    # The TELL of the fix: the producing strategy is now ``file_write`` (the action
    # chain WAS recorded) and the verify passed (file written).
    assert step["strategy"] == "file_write", (
        f"native dev loop now surfaces the code tools -> strategy 'file_write'; got "
        f"{step['strategy']!r}. If this is empty, the leak has re-opened."
    )
    assert step["verify"].startswith("path_contains"), f"got verify={step['verify']!r}"
    assert step["verify_result"] is True, (
        "file_write is now a native tool, so the file was written and "
        "path_contains must be True"
    )
    assert step["evidence"] == "GROUNDED"
    # The file was ACTUALLY written to disk (the action had a real side effect).
    assert out_path.read_text(encoding="utf-8") == "ready\n", (
        f"out.txt must exist with the written content; cwd={tmp_path}"
    )


# ---------------------------------------------------------------------------
# (C.2) THE PRODUCER IS WORLD-AGNOSTIC — native ACTS in dev when a skill is wrapped
# ---------------------------------------------------------------------------


def test_native_devworld_grounds_when_dev_action_skill_is_wrapped() -> None:
    """run_turn_native covers a dev write -> verify(path_contains) when the action is a
    wrapped skill -> GROUNDED / verified True / exit 0.

    Same registration pattern as the trichotomy teleport case (a test-only skill on a
    real ``Agent``), but a DEV (non-robot) action: write a file. This isolates the
    leak to the cli.main dev-world wiring (no agent -> no action tools) and proves the
    native PRODUCER + the honest spine already work outside go2. ``path_contains`` is
    a PREDICATE oracle, so a True read grades GROUNDED; the step is NOT_GRADED for
    actor-causation (a dev predicate, not a robot predicate) -> no downgrade.
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.types import SkillResult
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import (
        classify_step_evidence,
        verify_oracle_names,
    )
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds.dev import DevWorld

    from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn

    # A test-only DEV action skill: writes a file (relative to cwd, the same tree the
    # dev ``path_contains`` predicate reads). No robot, no motor — a pure dev action.
    class _WriteFileSkill:
        name = "write_file"
        description = "TEST-ONLY: write text content to a file (a dev action skill)."
        parameters = {
            "file_path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
        }
        preconditions: list[str] = []
        effects: dict = {}

        def execute(self, params, context):
            Path(params["file_path"]).write_text(str(params.get("content", "")), encoding="utf-8")
            return SkillResult(success=True, result_data={"file_path": params.get("file_path")})

    prev_cwd = os.getcwd()
    work_dir = tempfile.mkdtemp(prefix="native_devworld_")
    os.chdir(work_dir)
    try:
        # A dev-style agent: NO hardware (arm/base/gripper all None), one dev action
        # skill registered — exactly the surface wrap_skills would expose if the dev
        # world handed the loop an agent.
        agent = Agent(config={})
        agent._skill_registry.register(_WriteFileSkill())

        backend = FakeToolScriptBackend.from_tool_script([
            tool_turn(("write_file", {"file_path": "out.txt", "content": "ready\n"})),
            tool_turn(("verify", {"expr": "path_contains('out.txt', 'ready')"})),
            tool_turn(end=True),
        ])
        eng = VectorEngine(
            backend=backend, registry=CategorizedToolRegistry(),
            permissions=PermissionContext(no_permission=True),
        )
        eng.bind(agent, DevWorld())

        session = Session(
            session_id="native-devworld", created_at="t", updated_at="t",
            path=Path(work_dir) / "native_devworld.jsonl",
        )
        trace = eng.run_turn_native("create out.txt containing ready", session=session)

        assert len(trace.steps) == 1, f"expected one write->verify step; got {len(trace.steps)}"
        names = verify_oracle_names(agent, eng)
        sub, step = trace.goal_tree.sub_goals[0], trace.steps[0]

        # The native loop dispatched the DEV action skill (routing held outside go2).
        assert step.strategy == "write_file", f"strategy={step.strategy!r}"
        assert sub.verify.startswith("path_contains"), f"verify={sub.verify!r}"
        # The action actually ran — the file exists with the content.
        assert Path("out.txt").read_text(encoding="utf-8") == "ready\n", "skill must write the file"
        assert step.verify_result is True, "path_contains must read True after the write"
        # A dev predicate is NOT a robot predicate -> NOT_GRADED -> no causation downgrade.
        assert step.actor_caused is ActorCaused.NOT_GRADED, (
            f"a dev predicate must be NOT_GRADED (not a robot predicate); got "
            f"{step.actor_caused.value}"
        )
        assert classify_step_evidence(step, sub, names) == "GROUNDED"

        # SAME verdict gate -> the dev path verifies cleanly through the native producer.
        report = VerdictReport.from_trace(trace, names)
        assert report.verified is True, f"a grounded dev step must verify; {report.evidence}"
        assert report.evidence == "GROUNDED"
        assert report.exit_code() == 0
    finally:
        os.chdir(prev_cwd)
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# (C.3) GOAL-AUTHENTICITY — a dev VERIFY-ONLY no-op (no action) must be RAN
# ---------------------------------------------------------------------------


def test_native_devworld_verify_only_noop_is_ran(tmp_path) -> None:
    """A dev VERIFY-ONLY no-op (no file_write) against a PRE-EXISTING file -> RAN / exit 2.

    STEP 9 goal-authenticity (closes a 2026-06-19 review gap): the native loop offers
    the code tools, but a turn that ONLY verifies pre-existing / ambient state — zero
    action skills dispatched — must NOT earn GROUNDED. The actor caused nothing, so
    ``NativeStepRunner._grade`` returns UNCAUSED for the non-robot predicate (empty
    action chain) and the spine R2b downgrade flips the otherwise-GROUNDED path to RAN.
    The file PRE-EXISTS, so ``path_contains`` reads True — the moat must still report
    RAN / exit 2 because the actor did not produce the state.
    """
    pre = tmp_path / "pre.txt"
    pre.write_text("seed value\n", encoding="utf-8")
    r = run_cli_turn(
        "confirm pre.txt contains seed",
        tool_script={
            "turns": [
                {"tool_calls": [
                    {"name": "verify", "input": {"expr": "path_contains('pre.txt', 'seed')"}}
                ]},
                {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
            ]
        },
        timeout_sec=90.0,
        cwd=tmp_path,
    )
    print(f"\n[devworld no-op] cli.main -> {r.verdict}")
    # No action ran -> not verified, RAN, exit 2 — even though the predicate reads True.
    assert r.verified is False, f"a verify-only dev no-op must NOT verify (no action); {r.verdict}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}; {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    per_step = r.verdict.get("per_step") or []
    assert per_step, f"expected one verify-only step; got {per_step}"
    step = per_step[0]
    # The TELLs: the predicate READS True (file pre-exists), the action chain is EMPTY,
    # and the step grades RAN (the goal-authenticity causation tie fired).
    assert step["verify_result"] is True, "path_contains should read True (file pre-exists)"
    assert step["strategy"] == "", f"verify-only step has no action chain; got {step['strategy']!r}"
    assert step["evidence"] == "RAN", f"a no-action dev verify must be RAN; got {step['evidence']}"
