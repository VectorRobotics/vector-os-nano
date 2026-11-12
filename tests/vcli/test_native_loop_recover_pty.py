# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""STEP 11 acceptance — the RECOVER pillar (north-star #4) through run_turn_native.

The native producer does within-turn agentic recover-by-retry (the model re-walks
after a FAIL verify; native_loop.py system prompt 626-631), driven ONLY by the model
reading the ``verify(...) -> FAIL`` tool_result — the loop itself has NO replan and NO
``finish``-while-FAIL guard (native_loop.py:389, 440-454). This file proves, on the
REAL go2 MuJoCo sim through the actual ``cli.main`` (-p/--json PTY harness, only the
network LLM faked via VECTOR_FAKE_LLM_TOOLS), that the honest moat is FAIL-CLOSED on
recovery — it can never be tricked into a false "recovered/done".

HONEST COVERAGE STATEMENT (made explicit, per the STEP-11 adversarial review — do NOT
let the test set over-claim the recover pillar):

  Deterministic acceptance here covers THREE honest outcomes:
    (b) FINISH-ON-FAIL      finish() while the latest at_position verify FAILs ->
                            RAN / verified False / exit 2. The moat catches the lie.
    (c) NEVER-SUCCEEDS      a recovery that never lands exhausts the script -> the
                            backend returns a terminal end_turn -> the loop breaks
                            (no infinite loop; max_turns=24 never reached) -> all
                            steps RAN / verified False / exit 2.
    (d) GENUINE IN-TURN     walk-short -> verify FAIL -> walk-again -> verify PASS ->
        RECOVERY            finish. WP07 attempt semantics: the re-verify of the SAME
        (VERIFIED)          predicate is the latest attempt of that sub-goal, graded
                            against the baseline before the FIRST walk; it is
                            actor-CAUSED and GROUNDED, so the turn verifies. The FAIL
                            row stays in the verdict as a superseded attempt (audit).
                            A PASS of a different, easier predicate never supersedes,
                            and an UNCAUSED PASS never verifies (test_spine_s2).

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after each
case via the ``sim_cleanup`` fixture (the generated scene xml is untracked). Do NOT parallelize with
any other sim or pytest-heavy job (64 GB host).
"""
from __future__ import annotations


import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

_SIM_TIMEOUT_SEC = 360.0

# go2 starts at (10, 3); target (9, 3) is a 1.0 m gap BEHIND it (the pick table
# blocks the way forward at x~10.47), at_position tol = 0.5 m. Walks count SIM time
# and the MPC gait tracks backward walks to a few cm. So:
#   - a 0.3 m "short" walk lands ~x=9.7 -> 0.7 m from target -> FAIL;
#   - a 0.7 m "recovery" walk closes the remaining gap -> PASS.

# (b) FINISH-ON-FAIL: one short walk that lands outside tol, then a lying finish().
_FINISH_AFTER_FAIL_SCRIPT = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 0.3, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}

# (c) NEVER-SUCCEEDS: two walk->verify pairs toward an UNREACHABLE target (50, 3),
# NO finish -> backend exhausts -> terminal end_turn -> the loop breaks (no hang).
_NEVER_SUCCEEDS_SCRIPT = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(50.0, 3.0)"}}]},
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(50.0, 3.0)"}}]},
        # (no finish — exhaustion terminates the loop)
    ]
}

# (d) GENUINE IN-TURN RECOVERY: short walk FAILs, recovery walk REACHES -> verified.
_GENUINE_RECOVERY_SCRIPT = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 0.3, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},   # FAIL (short)
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 0.7, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},   # PASS (reached)
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


# ---------------------------------------------------------------------------
# (b) FINISH-ON-FAIL — a fake "done" must NOT verify (the moat catches the lie)
# ---------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_finish_after_fail_is_not_verified(sim_cleanup) -> None:
    """The model finishes while the latest at_position verify still FAILs. The loop
    honors finish (no guard), but the recorded last step is RAN (verify_result False),
    so the spine grades verified False / exit 2 — recover cannot be claimed by lying."""
    r = run_cli_turn(
        "走到坐标 (9.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_FINISH_AFTER_FAIL_SCRIPT,
    )
    assert r.verified is False, f"finish-on-FAIL must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    assert r.verified == (r.exit_code == 0)
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    assert step["verify_result"] is False, "short walk must land outside tol (a real FAIL)"
    assert step["success"] is True and step["strategy"] == "walk"


# ---------------------------------------------------------------------------
# (c) NEVER-SUCCEEDS — recovery that never lands TERMINATES, all RAN (no fake-pass)
# ---------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_recovery_never_succeeds_terminates_ran_not_verified(sim_cleanup) -> None:
    """Two walk->verify pairs toward an unreachable target, NO finish. The fake
    backend exhausts -> terminal end_turn -> the loop's ``if not tool_calls: break``
    fires (max_turns=24 never reached) — proving recover-by-retry TERMINATES rather
    than spinning. Every step is a real walk that FAILed -> all RAN / verified False."""
    r = run_cli_turn(
        "走到坐标 (50.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_NEVER_SUCCEEDS_SCRIPT,
    )
    assert r.verified is False, f"never-succeeds must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f">=1 recorded-but-unreached step must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    per_step = r.verdict["per_step"]
    assert len(per_step) >= 2, f"expected >=2 walk->verify steps; got {len(per_step)}"
    assert all(s["evidence"] == "RAN" and s["verify_result"] is False for s in per_step)
    # Real walks ran (not a verify-only no-op) — distinguishes 'walked but missed' from 'did nothing'.
    assert all(s["strategy"] == "walk" for s in per_step)


# ---------------------------------------------------------------------------
# (d) GENUINE IN-TURN RECOVERY — FAIL -> retry -> PASS on the same predicate verifies
# ---------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_inturn_recovery_is_verified(sim_cleanup) -> None:
    """The recover pillar (north-star #4), WP07: walk-short -> verify FAIL ->
    walk-again -> verify PASS -> finish verifies. The PASS is the latest attempt of the
    same sub-goal and is actor-CAUSED against the baseline before the first walk (real
    commanded motion + displacement read off unauthored qpos); the FAIL row stays in
    the verdict, marked superseded (tests:recover-pinned-unverifiable)."""
    r = run_cli_turn(
        "走到坐标 (9.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_GENUINE_RECOVERY_SCRIPT,
    )
    assert r.verified is True, f"a genuine recovery must verify; got {r.verdict}"
    assert r.exit_code == 0 and r.evidence == "GROUNDED", r.verdict
    per_step = r.verdict["per_step"]
    assert len(per_step) == 2, f"expected exactly [short-FAIL, recovery-PASS]; got {len(per_step)}"
    # step 0 — the short walk that landed short: a real FAIL, kept as a superseded attempt.
    assert per_step[0]["verify_result"] is False and per_step[0]["superseded"] is True
    assert per_step[0]["strategy"] == "walk" and per_step[0]["reject_reason"] == "verify FAIL"
    # step 1 — the recovery walk that REACHED: GROUNDED, actor-CAUSED (a genuine flip).
    assert per_step[1]["evidence"] == "GROUNDED" and per_step[1]["verify_result"] is True
    assert per_step[1]["actor_caused"] == "CAUSED" and per_step[1]["superseded"] is False
    assert per_step[1]["strategy"] == "walk"
    assert r.verdict["n_steps"] == 1 and r.verdict["contract"] == ["at_position(9.0, 3.0)"]
