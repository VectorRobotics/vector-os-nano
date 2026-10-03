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
    heading preserved) + ``verify('facing(0.0, 3.2)')`` (tol clamped to 45 deg; true at spawn). The
    heading no-op used to grade CAUSED via the old ``max(planar, yaw)`` metric; the
    per-channel fix grades the FORWARD walk UNCAUSED for a facing predicate -> RAN.
    This ALSO empirically checks the yaw threshold against REAL forward-walk gait
    drift: if the straight walk drifted heading past ~10deg this would not flip.

SIM DISCIPLINE: serialized, headless, MuJoCo closed + rosm nuke + scene xml restored
after each case via the fixture.
"""
from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

_SIM_TIMEOUT_SEC = 360.0

# (A) real CAUSED walk + a membership short-circuit verify (predicate is dead weight).
_MEMBERSHIP_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "forward", "distance": 2.5, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "True in (at_position(99.0, 99.0), True)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}

# (B) forward walk (no turn) + a facing predicate already true at spawn. The
# heading no-op must NOT grade CAUSED off the walk's planar displacement.
_FACING_FORWARD_WALK_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "forward", "distance": 2.5, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "facing(0.0, 3.2)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


def _nuke() -> None:
    try:
        subprocess.run(["rosm", "nuke", "--yes"], timeout=30, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture()
def sim_cleanup():
    yield
    _nuke()
    try:
        subprocess.run(
            ["git", "checkout",
             "vector_os_nano/hardware/sim/mjcf/go2/scene_room_piper.xml"],
            timeout=20, capture_output=True,
        )
    except Exception:  # noqa: BLE001
        pass


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
        extra_args=["--headless", "--native-loop"],
        tool_script=_MEMBERSHIP_FALSE_GREEN,
    )
    assert r.verified is False, f"membership short-circuit must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The predicate genuinely evaluates True (the dead-weight constant) — the classifier
    # rejected the STRUCTURE, not the truth value.
    assert step["verify_result"] is True


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_facing_forward_walk_false_green_flips_to_ran(sim_cleanup) -> None:
    """(B) a forward walk + `facing(0.0, 3.2)` (true at the spawn heading) -> RAN
    / verified False / exit 2. The walk moved planar, not heading, so a facing
    predicate must grade UNCAUSED — proving the per-channel causation fix holds on the
    real gait (incl. any forward-walk yaw drift below the ~10deg threshold)."""
    r = run_cli_turn(
        "保持当前朝向",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless", "--native-loop"],
        tool_script=_FACING_FORWARD_WALK_FALSE_GREEN,
    )
    assert r.verified is False, f"forward-walk facing no-op must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The facing predicate is true (heading kept) — only actor-causation stops it.
    assert step["verify_result"] is True


# (C) STEP-13 goal-authenticity: a real CAUSED walk to the WRONG place that verifies its
# OWN landing must NOT verify — only the verify-constant-vs-parsed-goal gate stops it.
_WRONG_PLACE_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "forward", "distance": 0.4, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(10.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_wrong_place_self_verified_landing_flips_to_ran(sim_cleanup) -> None:
    """(C, STEP-13 goal-authenticity) a REAL actor-caused walk that lands near (10,3) —
    NOT the commanded goal (11,3) — and verifies at_position(10,3) (its OWN landing) ->
    RAN / verified False / exit 2. The walk is CAUSED and the predicate is TRUE at the
    landing, so R1+R2b both pass; ONLY the goal-authenticity gate (verify constant !=
    the user's parsed coordinate goal) can reject it."""
    r = run_cli_turn(
        "走到坐标 (11.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless", "--native-loop"],
        tool_script=_WRONG_PLACE_FALSE_GREEN,
    )
    assert r.verified is False, f"verifying your own wrong landing must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The predicate is TRUE at the landing and the walk is a real CAUSED walk — only the
    # goal-authenticity gate (constant (10,3) != goal (11,3)) downgrades it.
    assert step["verify_result"] is True
    assert step["strategy"] == "walk"


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
        extra_args=["--headless", "--native-loop"],
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


# (E) STEP-15 WRONG-PREDICATE-TYPE: a COORDINATE goal verified ENTIRELY with a non-
# at_position predicate. A real 90-deg left TURN (yaw command -> facing graded CAUSED ->
# the step is honestly GROUNDED per-step) + an honest facing verify of the post-turn
# heading (+y, within 0.78 rad: under the oracle's 45-deg tol cap, so true for a landing
# anywhere in 45..135 deg). The coord '走到坐标 (11,3)' is NEVER asserted by an
# at_position step, so the TURN-LEVEL coordinate gate must reject the turn (verified
# False) EVEN THOUGH the step itself is GROUNDED — the distinction from R2b (which
# downgrades the step) and from STEP-13 (which needs a literal at_position to mismatch).
_WRONG_PREDICATE_TYPE_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "turn", "input": {"direction": "left", "angle": 90}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "facing(1.5708, 0.78)"}}]},
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
        extra_args=["--headless", "--native-loop"],
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


# (F) STEP-16 BOOLEAN-NECESSITY (4th moat review): a real CAUSED walk to the WRONG place +
# an OR-decoy verify whose goal-matching at_position(11,3) is NOT necessary — a second
# at_position disjunct around the spawn (10,3) with a 2 m tol (at the oracle's tol cap)
# carries the truth after the short walk, while the (11,3) constant is an inert decoy. The decoy uses only
# at_position (not facing), so the BASE channel grades CAUSED off the real forward walk;
# ONLY the necessity gate (matching at_position must be an AND-conjunct) can stop it.
_OR_DECOY_FALSE_GREEN = {
    "turns": [
        {"tool_calls": [{"name": "walk", "input": {"direction": "forward", "distance": 0.4, "speed": 0.3}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(11.0, 3.0) or at_position(10.0, 3.0, 2.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_or_decoy_non_necessary_at_position_flips_to_ran(sim_cleanup) -> None:
    """(F, STEP-16) a coordinate goal '走到坐标 (11,3)' verified with
    `at_position(11,3) or at_position(10,3,2.0)` after a real CAUSED 0.4 m forward walk
    from the (10,3) spawn -> RAN / verified False / exit 2. The OR EVALUATES True (the
    decoy disjunct holds at the landing) and the walk is CAUSED on the planar channel, so R1+R2b both pass and
    the goal-matching constant (11,3) is PRESENT — only the necessity gate (the matching
    at_position must be an AND-conjunct that gates the verdict) rejects it."""
    r = run_cli_turn(
        "走到坐标 (11.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless", "--native-loop"],
        tool_script=_OR_DECOY_FALSE_GREEN,
    )
    assert r.verified is False, f"OR-decoy (non-necessary at_position) must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "RAN"
    # The OR genuinely evaluates True (the decoy disjunct) and the walk is a real
    # CAUSED walk — only the necessity gate (matching at_position is not a gating conjunct)
    # downgrades it, at the per-step level (coord_goal_mismatch) as well as the turn gate.
    assert step["verify_result"] is True
    assert step["strategy"] == "walk"
