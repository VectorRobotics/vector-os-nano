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
        RECOVERY            finish records [RAN, GROUNDED]. The all-GROUNDED gate
        (FAIL-CLOSED)       (trace_store.evidence_passed 355-362) FAILS CLOSED on the
                            recorded FAIL row -> verified False / exit 2, EVEN THOUGH
                            the robot genuinely recovered (step-1 GROUNDED => the 2nd
                            walk was actor-CAUSED and reached the target read off
                            unauthored qpos).

  WHAT IS DELIBERATELY NOT CLAIMED: a deterministic recover-to-verified-True. By the
  moat architecture (native_loop records one StepRecord per verify with NO dedup,
  271-293; evidence_passed requires EVERY checked step GROUNDED), an honestly-recorded
  in-turn FAIL->PASS recovery grades RAN. Making it verified=True would require a
  per-sub-goal latest-verify gate that lets a later PASS overwrite a recorded FAIL —
  that is LOOSER, and rule 5 (verify only ever gets stricter) forbids it. The spine
  stays BYTE-UNCHANGED. Recover-to-verified-True is therefore demonstrated ONLY where
  the LIVE model verifies ONCE after recovering: ``test_native_loop_multistep_live_pty``.
  Goal-authenticity of the verify constant (does (9,3) match the real task goal)
  remains the documented deferred residual — STEP 11 does not close it.

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

# (d) GENUINE IN-TURN RECOVERY: short walk FAILs, recovery walk REACHES -> [RAN, GROUNDED].
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
# (d) GENUINE IN-TURN RECOVERY — records [RAN, GROUNDED], FAILS CLOSED to verified False
# ---------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_inturn_recovery_is_fail_closed_not_verified(sim_cleanup) -> None:
    """THE headline honest invariant. A genuine in-turn recovery (walk-short ->
    verify FAIL -> walk-again -> verify PASS -> finish) records exactly [RAN, GROUNDED].
    The all-GROUNDED moat gate FAILS CLOSED on the recorded FAIL row -> verified False
    / exit 2, EVEN THOUGH the robot really recovered. step-1 GROUNDED proves the
    recovery was genuine: the spine grants GROUNDED only when the verify PASSed AND the
    step was actor-CAUSED (real commanded motion + real displacement reaching the target
    read off unauthored qpos) — so this is a real flip, fail-closed, never a false green.

    This DOCUMENTS the moat's fail-closed-on-recovery behavior as an asserted invariant
    rather than an unstated gap. Recover-to-verified-True lives in the LIVE test
    (test_native_loop_multistep_live_pty), where the model verifies once after recovering.
    """
    r = run_cli_turn(
        "走到坐标 (9.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_GENUINE_RECOVERY_SCRIPT,
    )
    assert r.verified is False, f"a recorded-FAIL recovery must fail closed; got {r.verdict}"
    assert r.exit_code == 2, f"fail-closed recovery must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    per_step = r.verdict["per_step"]
    assert len(per_step) == 2, f"expected exactly [short-FAIL, recovery-PASS]; got {len(per_step)}"
    # step 0 — the short walk that landed short: a real recorded FAIL (the row the gate keys on).
    assert per_step[0]["evidence"] == "RAN" and per_step[0]["verify_result"] is False
    assert per_step[0]["strategy"] == "walk"
    # step 1 — the recovery walk that REACHED: GROUNDED => actor-CAUSED (genuine flip, not a no-op).
    assert per_step[1]["evidence"] == "GROUNDED" and per_step[1]["verify_result"] is True
    assert per_step[1]["strategy"] == "walk"
