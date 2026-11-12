# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""STEP 12 regression — two adversarial-review moat false-greens flip to RAN through
the REAL ``cli.main`` on real go2 MuJoCo (the 2nd moat review, loop-until-dry).

Both holes graded a FALSE turn ``verified=True`` BEFORE the fix; this pins that the
real product now grades them RAN / exit 2 (the moat may only get STRICTER, rule 5):

(A) COMPARE-MEMBERSHIP SHORT-CIRCUIT (evidence_classifier): a real walk (CAUSED) +
    ``verify('True in (at_position(99,99), True)')`` — the constant member satisfies
    the membership while the oracle is dead weight, the ``... or True`` short-circuit
    hidden in an ``in`` node. The classifier now rejects a constant-literal container
    -> RAN even though the predicate EVALUATES True and the walk was CAUSED.

(B) FACING CROSS-CHANNEL CAUSATION (actor_causation): a forward walk (planar move,
    heading preserved) + ``verify('facing(0.0, 0.78)')`` (true at spawn, tol under the 45-deg cap). The
    heading no-op used to grade CAUSED via the old ``max(planar, yaw)`` metric; the
    per-channel fix grades the FORWARD walk UNCAUSED for a facing predicate -> RAN.
    This ALSO empirically checks the yaw threshold against REAL forward-walk gait
    drift: if the straight walk drifted heading past ~10deg this would not flip.

SIM DISCIPLINE: serialized, headless, MuJoCo closed + the child process group reaped
after each case via the fixture.
"""
from __future__ import annotations


import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

_SIM_TIMEOUT_SEC = 360.0

# (A) real CAUSED walk + a membership short-circuit verify (predicate is dead weight).
# Backward: forward from the spawn the dog moves 0.48 m of a 2.5 m walk (measured),
# which walk fails as walk_blocked, so the step graded FAILED before the classifier ran.
_MEMBERSHIP_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "True in (at_position(99.0, 99.0), True)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}

# (B) a straight walk (no turn) + a facing predicate already true at spawn. The
# heading no-op must NOT grade CAUSED off the walk's planar displacement.
_FACING_STRAIGHT_WALK_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "facing(0.0, 0.78)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_membership_shortcircuit_false_green_flips_to_ran(sim_cleanup) -> None:
    """(A) `True in (at_position(99,99), True)` after a real walk -> RAN / verified
    False / exit 2. The membership predicate EVALUATES True and the walk is CAUSED,
    so ONLY the structural classifier can stop it — and it must."""
    r = run_cli_turn(
        "走到坐标 (99.0,99.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_MEMBERSHIP_FALSE_GREEN,
    )
    assert r.verified is False, f"membership short-circuit must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The predicate genuinely evaluates True (the dead-weight constant) and the walk was
    # actor-CAUSED — the classifier rejected the STRUCTURE, nothing else.
    assert step["verify_result"] is True
    assert step["actor_caused"] == "CAUSED"
    assert step["reject_reason"] == "the verify is not ground-truth evidence"


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_facing_straight_walk_false_green_flips_to_ran(sim_cleanup) -> None:
    """(B) a straight walk + `facing(0.0, 0.78)` (true at the spawn heading) -> RAN
    / verified False / exit 2. The walk moved planar, not heading, so a facing
    predicate must grade UNCAUSED — proving the per-channel causation fix holds on the
    real gait (incl. any straight-walk yaw drift below the ~10deg threshold)."""
    r = run_cli_turn(
        "保持当前朝向",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_FACING_STRAIGHT_WALK_FALSE_GREEN,
    )
    assert r.verified is False, f"straight-walk facing no-op must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The facing predicate is true (heading kept) — only actor-causation stops it.
    assert step["verify_result"] is True
    assert step["actor_caused"] == "UNCAUSED"
    assert step["reject_reason"].startswith("UNCAUSED (base:facing:UNCAUSED")


# (C) STEP-13 goal-authenticity: a real CAUSED walk to the WRONG place that verifies its
# OWN landing must NOT verify — only the verify-constant-vs-parsed-goal gate stops it.
# The landing (9,3) is 1 m from the (10,3) spawn, so the verify is False at the
# baseline and the F->T causation rule passes it: the goal (8,3) is what differs.
_WRONG_PLACE_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_wrong_place_self_verified_landing_flips_to_ran(sim_cleanup) -> None:
    """(C, STEP-13 goal-authenticity) a REAL actor-caused walk that lands near (9,3) —
    NOT the commanded goal (8,3) — and verifies at_position(9,3) (its OWN landing) ->
    RAN / verified False / exit 2. The walk is CAUSED and the predicate is TRUE at the
    landing, so R1+R2b both pass; ONLY the goal-authenticity gate (verify constant !=
    the user's parsed coordinate goal) can reject it."""
    r = run_cli_turn(
        "走到坐标 (8.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_WRONG_PLACE_FALSE_GREEN,
    )
    assert r.verified is False, f"verifying your own wrong landing must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The predicate is TRUE at the landing and the walk is a real CAUSED walk — only the
    # goal-authenticity gate (constant (9,3) != goal (8,3)) downgrades it.
    assert step["verify_result"] is True
    assert step["strategy"] == "walk"
    assert step["actor_caused"] == "CAUSED"
    assert step["reject_reason"] == "the verified coordinate is not the commanded one"


# (D) STEP-14: the STEP-12 membership fix is bypassed by a BUILTIN container constructor
# (tuple()/list()), an ast.Call rather than a literal collection — the constant member
# satisfies the `in` while the oracle is dead weight. Must flip to RAN through cli.main.
_CALLABLE_CONTAINER_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "forward", "distance": 0.4, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "True in tuple((at_position(2.0, 0.0), True))"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_callable_container_shortcircuit_flips_to_ran(sim_cleanup) -> None:
    """(D, STEP-14) the STEP-12 membership fix is bypassed by a builtin container
    constructor: `True in tuple((at_position(2,0), True))` — the constant True member
    satisfies the `in` while the oracle is dead weight (the `... or True` short-circuit
    wrapped in a tuple() call). A real CAUSED walk + this verify (goal '走到坐标 (2,0)',
    so goal-authenticity fails open as const==goal) must grade RAN / verified False /
    exit 2 — ONLY the structural classifier can stop it."""
    r = run_cli_turn(
        "走到坐标 (2.0,0.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_CALLABLE_CONTAINER_FALSE_GREEN,
    )
    assert r.verified is False, f"callable-container short-circuit must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The predicate EVALUATES True (the dead-weight constant) — the classifier rejected
    # the STRUCTURE, not the truth value.
    assert step["verify_result"] is True
    assert step["actor_caused"] == "CAUSED"
    assert step["reject_reason"] == "the verify is not ground-truth evidence"


# (E) STEP-15 WRONG-PREDICATE-TYPE: a COORDINATE goal verified ENTIRELY with a non-
# at_position predicate. A real 60-deg left TURN (yaw command -> facing graded CAUSED ->
# the step is honestly GROUNDED per-step) + a facing verify within the oracle's 45-deg
# tol cap: facing(1.05, 0.78) holds for any heading in 15..105 deg — False at the 0-deg
# spawn (F->T) and True after the closed-loop 60-deg turn. The coord '走到坐标 (11,3)' is NEVER asserted by
# an at_position step, so the TURN-LEVEL coordinate gate must reject the turn (verified
# False) EVEN THOUGH the step itself is GROUNDED — the distinction from R2b (which
# downgrades the step) and from STEP-13 (which needs a literal at_position to mismatch).
_WRONG_PREDICATE_TYPE_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "turn", "input": {"direction": "left", "angle": 60}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "facing(1.05, 0.78)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_wrong_predicate_type_coord_goal_flips_to_ran(sim_cleanup) -> None:
    """(E, STEP-15) a coordinate goal '走到坐标 (11,3)' verified ONLY with a facing
    predicate after a real CAUSED turn -> turn verified False / RAN / exit 2, while the
    facing STEP stays GROUNDED. The turn-level coordinate gate rejects a coord turn that
    never asserts the commanded coordinate with an at_position; per-step honesty is
    untouched (the distinction this round adds)."""
    r = run_cli_turn(
        "走到坐标 (11.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_WRONG_PREDICATE_TYPE_FALSE_GREEN,
    )
    assert r.verified is False, f"coord goal verified with only facing must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    # KEY DISTINCTION: the facing step is honestly GROUNDED (real CAUSED turn), yet the
    # TURN is rejected because the coordinate was never asserted by an at_position.
    assert step["strategy"] == "turn", f"got strategy={step['strategy']}"
    assert step["evidence"] == "GROUNDED", (
        f"the facing step itself is honest/GROUNDED; only the turn-level coord gate "
        f"rejects the turn. got per-step evidence={step['evidence']}"
    )
    assert step["verify_result"] is True
    assert step["actor_caused"] == "CAUSED"
    assert "at_position(11.0, 3.0)" in r.verdict["reject_reason"]


# (F) STEP-16 BOOLEAN-NECESSITY (4th moat review): a real CAUSED walk to the WRONG place +
# an OR-decoy verify whose goal-matching at_position(11,3) is NOT necessary — a second
# at_position disjunct around the spawn (10,3) with a 2 m tol (at the oracle's tol cap)
# carries the truth after the short walk, while the (11,3) constant is an inert decoy. The decoy uses only
# at_position (not facing), so the BASE channel grades CAUSED off the real forward walk;
# ONLY the necessity gate (matching at_position must be an AND-conjunct) can stop it.
_OR_DECOY_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(8.0, 3.0) or at_position(9.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_or_decoy_non_necessary_at_position_flips_to_ran(sim_cleanup) -> None:
    """(F, STEP-16) a coordinate goal '走到坐标 (8,3)' verified with
    `at_position(8,3) or at_position(9,3)` after a real CAUSED 1 m backward walk from the
    (10,3) spawn -> RAN / verified False / exit 2. The OR EVALUATES True (the decoy
    disjunct holds at the landing, and was False at the baseline) and the walk is CAUSED
    on the planar channel, so R1+R2b both pass and the goal-matching constant (8,3) is
    PRESENT — only the necessity gate (the matching at_position must be an AND-conjunct
    that gates the verdict) rejects it."""
    r = run_cli_turn(
        "走到坐标 (8.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_OR_DECOY_FALSE_GREEN,
    )
    assert r.verified is False, f"OR-decoy (non-necessary at_position) must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The OR genuinely evaluates True (the decoy disjunct) and the walk is a real
    # CAUSED walk. Since I4 (G3) a disjunction is never evidence, so the classifier
    # rejects it first; the necessity gate (coord_goal_mismatch) still stands behind it.
    assert step["verify_result"] is True
    assert step["strategy"] == "walk"
    assert step["actor_caused"] == "CAUSED"
    assert step["reject_reason"] == "the verify is not ground-truth evidence"


# (G) WP07 rework (verifier P8): a housekeeping stop re-verifying the SAME text supersedes
# a FAILED turn row. The stop row is exempt only when its whole attempt group is
# housekeeping, so it must GROUND on its own — a vacuous ``True`` never does.
_STOP_LAUNDERS_FAILED_TURN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},
        {"tool_calls": [{"name": "turn", "input": {"direction": "sideways", "angle": 90}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "True"}}]},
        {"tool_calls": [{"name": "stop", "input": {}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "True"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_a_stop_cannot_launder_a_failed_turn(sim_cleanup) -> None:
    """(G) '走到坐标 (9,3) 然后左转90度': the walk GROUNDs the contract, the turn skill
    errs ('sideways'), then stop + the same vacuous re-verify. Charter 3.5 (SP1): the turn
    erred before commanding motion, so it was REFUSED — not an action: the verify right
    after it is a probe (refused unevaluated), and the stop's row carries the refusal;
    nothing is laundered. A turn that fails AFTER motion is its own FAILED row and still
    rejects (tests/unit/vcli/test_native_failed_attempts.py)."""
    r = run_cli_turn(
        "走到坐标 (9.0,3.0) 然后左转90度",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_STOP_LAUNDERS_FAILED_TURN,
    )
    assert r.verified is True and r.exit_code == 0, f"got {r.exit_code}: {r.verdict}"
    assert r.verdict["contract"] == ["at_position(9.0, 3.0)"]
    rows = [(s["strategy"], s["evidence"], s["superseded"]) for s in r.verdict["per_step"]]
    assert rows == [("walk", "GROUNDED", False), ("stop", "RAN", False)]
    assert [x.split(":")[0] for x in r.verdict["per_step"][1]["refused"]] == ["turn"], r.verdict
