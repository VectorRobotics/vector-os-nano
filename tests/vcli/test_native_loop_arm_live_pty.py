# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 STEP 4 (B) — the LIVE-LLM real-sim acceptance of the ARM/GRIPPER capability.

Step 4 (A) (``test_native_loop_arm_pty.py``) proved the PLUMBING with a SCRIPTED
backend (``FakeToolScriptBackend`` replays pick->verify->finish). It did NOT prove
the THESIS: that the REAL frontier model, handed the SO-101 arm tool-set + NL,
drives the manipulation loop ITSELF. This module is that proof — the arm analogue
of the live walk acceptance (``test_native_loop_live_llm_pty.py``).

It drives the REAL ``cli.main -p "<NL>" --json --sim`` PTY with NO
tool_script — the REAL provider backend resolved from the repo-root ``.env`` (the
product default) on the REAL SO-101 MuJoCo arm sim. The MODEL must AUTONOMOUSLY
decompose / route / verify the grasp:

  - some step's ``strategy == "pick"`` — a manipulation skill (the model chose to
    grasp, not to home/scan-only);
  - that step's ``verify`` starts with ``holding_object`` — the registry-derived
    GRIPPER oracle that MEASURES the goal (not a base position oracle, not a
    scene/describe oracle);
  - when the grasp really toggles the object's weld 0->1 -> that step is
    ``GROUNDED`` (the honest spine's gripper-channel causation grade is CAUSED).

The runner owns NONE of this: it does not inject the plan, does not replan, does
not sentinel-verify. The model is the only thing choosing ``pick`` and writing the
``holding_object`` predicate.

HONEST grading (reuses the walk test's philosophy, never loosens the moat): the
honest ``evidence_passed`` gate requires EVERY step GROUNDED for the whole turn to
verify. A model may emit extra read-only/exploratory steps (scan/detect/describe)
whose verify carries no causation and grades RAN — a RAN+GROUNDED mix honestly
reports the whole turn RAN/exit2 (the gate working, not a failure). We therefore pin
full-turn ``verified True`` ONLY when EVERY step grounded; we NEVER force a partial
run green. The always-true acceptance is: the model autonomously routed ``pick`` +
``holding_object`` and earned >=1 GROUNDED grasp step. ``verified == (exit==0)`` is
asserted inside ``run_cli_turn`` regardless.

GATE (LIVE, non-deterministic, billed). SAME double gate as the walk live test:
SKIPPED unless BOTH a real provider key is present in the repo-root ``.env`` AND
``VECTOR_LIVE_LLM=1`` is set. Reproduce::

    VECTOR_LIVE_LLM=1 MUJOCO_GL=egl PATH=/usr/bin:$PATH \\
      .venv/bin/python -m pytest tests/vcli/test_native_loop_arm_live_pty.py -v -s -m sim

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after the
case via the ``sim_cleanup`` fixture. The arm sim does NOT write its scene xml.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

# A live grasp: cold imports + sim boot + several LLM turns + the arm executing a
# multi-phase pick (scan/detect/pregrasp/descend/close/lift/home). Generous headroom.
_LIVE_TIMEOUT_SEC = 300.0

# The repo-root .env the child's resolve_credentials reads (single-sourced from the
# harness root, exactly as the walk live test does).
_REPO_ROOT = run_cli_turn.__globals__["_REPO_ROOT"]
_ENV_PATH = _REPO_ROOT / ".env"


def _env_has_provider_key() -> bool:
    """True iff the repo-root .env carries a real provider key (OpenRouter/DeepSeek).

    Reads the file directly (cheap, no dotenv import) so the gate matches what the
    child's resolve_credentials sees — fail-closed to False on any read error.
    """
    try:
        text = _ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY") and value.strip():
            return True
    return False


def _live_opted_in() -> bool:
    return os.environ.get("VECTOR_LIVE_LLM", "").strip() in ("1", "true", "True")


# The DOUBLE gate (same as the walk live test): a real key AND the explicit opt-in.
# Module scope so the routine suite skips it and collection stays cheap.
_SKIP_REASON = (
    "live-LLM acceptance: needs a real provider key in repo-root .env AND "
    "VECTOR_LIVE_LLM=1 (deliberate, billed opt-in)"
)
pytestmark = pytest.mark.skipif(
    not (_env_has_provider_key() and _live_opted_in()),
    reason=_SKIP_REASON,
)


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
@pytest.mark.live_llm
def test_live_arm_pick_routes_and_grounds_holding(sim_cleanup) -> None:
    """REAL model, CHINESE NL "把香蕉抓起来拿在手里" -> autonomous pick->verify(holding_object('banana')) -> GROUNDED.

    The SO-101 arm boots with a banana on the table and NOT holding anything; the
    model alone must decide to ``pick`` it and prove it with a ``holding_object``
    predicate. The honest spine grades the assembled trace via the gripper-weld
    causation channel — a real grasp (weld 0->1) is the only way to earn GROUNDED.

    STEP 7 — the cross-language gap is now CLOSED (not deferred). ``holding_object(target)``
    matches the scene's CANONICAL English name ("banana") with a STRICT case-insensitive
    EXACT match — a wrong name (e.g. ``holding_object('apple')``) still returns False even
    on a real grasp, and the oracle was NOT loosened. What closes the gap is the VOCAB the
    native loop now hands the model: ``_native_system_prompt`` lists the world's graspable
    object names (single-sourced from the arm's ``get_object_positions()`` keys, the SAME
    ground truth the oracle matches), so a model commanded in CHINESE translates 香蕉->banana
    and emits the CANONICAL ``holding_object('banana')`` itself. We do NOT weaken the prompt
    to a bare no-arg ``holding_object()`` (that would make any-object satisfy a specific-object
    goal — looser, banned by rule 5); the model verifies the STRICT object-specific predicate
    from a Chinese command. (If a real model still emitted the Chinese name despite the vocab,
    the documented fallback is a strict, single-sourced oracle-side alias map — see STATUS.)
    """
    prompt = "把香蕉抓起来拿在手里"
    r = run_cli_turn(
        prompt,
        sim=True,
        live=True,
        timeout_sec=_LIVE_TIMEOUT_SEC,
        extra_args=["--headless"],
    )
    # Echo the live verdict (visible under `pytest -s`) so a real run's autonomous
    # routing + grounding is inspectable in the log, not just pass/fail.
    print(f"\n[live verdict] {prompt!r} -> {r.verdict}")
    per_step = r.verdict.get("per_step") or []
    assert per_step, (
        f"live turn produced no per_step trace for {prompt!r}; "
        f"verdict={r.verdict} raw=\n{r.raw_output[-2000:]}"
    )

    # (1) ROUTING — the model autonomously routed a manipulation ``pick`` step and
    # verified it with the registry-derived GRIPPER oracle ``holding_object(...)``.
    pick_steps = [
        s for s in per_step
        if s["strategy"] == "pick" and str(s["verify"]).startswith("holding_object")
    ]
    assert pick_steps, (
        f"the model must route a pick step verified by holding_object(...); got "
        f"per_step={[(s['strategy'], s['verify']) for s in per_step]} for {prompt!r}. "
        f"verdict={r.verdict}"
    )

    # (1b) STEP 7 CROSS-LANGUAGE — from a CHINESE command the model must emit the
    # CANONICAL scene name (banana), not the Chinese word (香蕉). This is the whole
    # point of the step-7 vocab expose: the strict oracle matches only "banana", so
    # a Chinese command verifies the object-specific predicate ONLY because the model
    # translated 香蕉->banana using the object vocab the native loop handed it.
    canonical = [s for s in pick_steps if "banana" in str(s["verify"]).lower()]
    assert canonical, (
        f"from a CHINESE command the model must verify with the CANONICAL scene name "
        f"holding_object('banana'); got verify exprs="
        f"{[s['verify'] for s in pick_steps]} for {prompt!r}. verdict={r.verdict}"
    )

    # (2) HONEST GROUNDING — a real grasp (weld 0->1) earned a GROUNDED pick step.
    grounded = [
        s for s in pick_steps
        if s["evidence"] == "GROUNDED" and s["verify_result"] is True
    ]
    assert grounded, (
        f"a real grasp must earn >=1 GROUNDED holding_object pick step; got "
        f"per_step={per_step} for {prompt!r}. verdict={r.verdict}"
    )

    # FULL-TURN verified ONLY when EVERY step grounded (never forced green on a mix:
    # exploratory scan/detect/home steps the model may add grade RAN and honestly
    # leave the whole turn RAN/exit2). ``verified == (exit==0)`` is asserted inside
    # run_cli_turn regardless.
    if all(s["evidence"] == "GROUNDED" for s in per_step):
        assert r.verified is True, (
            f"every step grounded -> the whole turn must verify True; verdict={r.verdict}"
        )
        assert r.exit_code == 0
        assert r.evidence == "GROUNDED"
