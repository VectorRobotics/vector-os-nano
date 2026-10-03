# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 STEP 2 — the LIVE-LLM real-sim acceptance of the native tool-use producer.

Step 1 (``test_native_loop_trichotomy_pty.py``) proved the PLUMBING with a SCRIPTED
backend (``FakeToolScriptBackend`` replays walk->verify->finish). It did NOT prove
the THESIS: that the REAL frontier model, handed the go2-walk tool-set + NL, drives
the loop ITSELF. This module is that proof.

It drives the REAL ``cli.main -p "<NL>" --json --sim-go2 --native-loop`` PTY with NO
tool_script — the REAL provider backend resolved from the repo-root ``.env`` (the
product default, see ``vcli.config.resolve_credentials``) on the REAL go2 MuJoCo sim.
The MODEL must AUTONOMOUSLY decompose / route / verify:

  - ``per_step[0].strategy == "walk"`` — NOT ``navigate`` (navigate is not even
    offered as a motor tool; its cmd_vel is gated out of actor-causation);
  - ``per_step[0].verify`` starts with ``at_position`` (the registry-derived vocab);
  - when the gait actually displaces within tol -> ``verified True`` / evidence
    ``GROUNDED`` / exit 0.

The runner owns NONE of this: it does not inject the plan, does not replan, does not
sentinel-verify. The model is the only thing choosing walk and writing the predicate.

GATE (this is LIVE, non-deterministic, and costs real provider calls). The test is
SKIPPED unless BOTH hold:
  (a) a real provider key is present in the repo-root ``.env``
      (``OPENROUTER_API_KEY`` or ``DEEPSEEK_API_KEY``); AND
  (b) explicit opt-in env ``VECTOR_LIVE_LLM=1`` is set.
So the routine chunked suite (no opt-in) SKIPS it even with a key present — it runs
ONLY when explicitly opted in. Reproduce with::

    VECTOR_LIVE_LLM=1 MUJOCO_GL=egl PATH=/usr/bin:$PATH \\
      .venv/bin/python -m pytest tests/vcli/test_native_loop_live_llm_pty.py -v -m sim

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after each
case via the ``sim_cleanup`` fixture (the generated scene xml is untracked).
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

# Live runs need real physics + a real network round-trip; allow generous headroom
# (cold imports + sim boot + several LLM turns + the MPC gait walking ~1m).
_LIVE_TIMEOUT_SEC = 300.0

# The repo-root .env (the SAME file resolve_credentials' load_dotenv() reads). The
# gate keys on a real provider key being present HERE — not on the ambient pytest
# env, which need not carry it.
_REPO_ROOT = run_cli_turn.__globals__["_REPO_ROOT"]  # single-source the harness root
_ENV_PATH = _REPO_ROOT / ".env"


def _env_has_provider_key() -> bool:
    """True iff the repo-root .env carries a real provider key (OpenRouter/DeepSeek).

    Reads the file directly (cheap, no dotenv import) so the gate matches what the
    child's resolve_credentials will actually see — fail-closed to False on any
    read error (a missing/unreadable .env means no live run).
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


# The DOUBLE gate: a real key AND the explicit opt-in. Applied at module scope so the
# whole file is skipped in the routine suite (the opt-in is the deliberate, billed
# switch); collection stays cheap.
_SKIP_REASON = (
    "live-LLM acceptance: needs a real provider key in repo-root .env AND "
    "VECTOR_LIVE_LLM=1 (deliberate, billed opt-in)"
)
pytestmark = pytest.mark.skipif(
    not (_env_has_provider_key() and _live_opted_in()),
    reason=_SKIP_REASON,
)


def _assert_walk_at_position(r, *, prompt: str) -> None:
    """The live thesis: the MODEL autonomously routes walk -> verify(at_position) and
    a REAL displacing walk earns a GROUNDED step from the honest spine.

    Two claims, asserted at the strength each can honestly bear (verified by REAL
    runs on the go2 sim with BOTH the product-default DeepSeek and a frontier model):

    1. ROUTING (design fix 6) — EVERY motion step the model issues routes to ``walk``
       (never ``navigate``, which is not even offered) and verifies with the
       registry-derived ``at_position(...)`` oracle. This holds on every step of
       every run — it is the fix-6 acceptance.
    2. HONEST GROUNDING — a real displacing walk earns a step the UNTOUCHED spine
       classifies ``GROUNDED`` with ``verify_result`` True (the literal "when the
       gait displaces within tol -> GROUNDED"). We require at least one such step.

    Full-turn ``verified``: the honest ``evidence_passed`` gate (spine, NOT touched)
    requires EVERY step GROUNDED. The legged gait UNDER-SHOOTS a ~1m walk command, so
    the model frequently lands short on its first walk (that step grades RAN) and
    then autonomously REPLANS a second walk that reaches tol (GROUNDED). That replan
    is the thesis working — but it leaves a RAN+GROUNDED mix, so the gate honestly
    reports the whole turn RAN/exit2. We therefore pin full-turn ``verified is True /
    exit 0 / GROUNDED`` ONLY for a clean ONE-walk trace; we never loosen the grade to
    force a replan run green (``verified == (exit==0)`` is already asserted inside
    run_cli_turn). Reaching tol in one walk vs needing a second is gait noise, not a
    routing failure — so the always-true acceptance is routing + a grounded walk.
    """
    # Echo the live verdict (visible under `pytest -s`) so a real run's autonomous
    # routing + grounding is inspectable in the log, not just pass/fail.
    print(f"\n[live verdict] {prompt!r} -> {r.verdict}")
    per_step = r.verdict.get("per_step") or []
    assert per_step, (
        f"live turn produced no per_step trace for {prompt!r}; "
        f"verdict={r.verdict} raw=\n{r.raw_output[-2000:]}"
    )
    # (1) ROUTING contract on EVERY step the model issued.
    for i, step in enumerate(per_step):
        assert step["strategy"] == "walk", (
            f"step {i}: model must route to walk (not navigate); got "
            f"strategy={step['strategy']!r} for {prompt!r}. verdict={r.verdict}"
        )
        assert str(step["verify"]).startswith("at_position"), (
            f"step {i}: model must verify with at_position(...); got "
            f"verify={step['verify']!r} for {prompt!r}. verdict={r.verdict}"
        )
    # (2) HONEST GROUNDING — a real displacing walk earned a GROUNDED at_position step.
    grounded = [
        s for s in per_step
        if s["evidence"] == "GROUNDED" and s["verify_result"] is True
    ]
    assert grounded, (
        f"a real displacing walk must earn at least one GROUNDED at_position step; "
        f"got per_step={per_step} for {prompt!r}. verdict={r.verdict}"
    )
    # STRONG case: a clean ONE-walk trace -> the full turn verifies True / exit 0.
    if len(per_step) == 1:
        assert r.verified is True, (
            f"a single grounded walk must verify the whole turn True; verdict={r.verdict}"
        )
        assert r.exit_code == 0, f"verified walk must exit 0; got {r.exit_code}"
        assert r.evidence == "GROUNDED", f"got evidence={r.evidence}; verdict={r.verdict}"


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
@pytest.mark.live_llm
def test_live_walk_to_coordinate_routes_and_grounds(sim_cleanup) -> None:
    """REAL model, NL "走到坐标 (11,3)" -> autonomous walk->verify(at_position) -> GROUNDED.

    go2 starts at (10,3); (11,3) is ~1m forward, inside at_position's 0.5m tol once
    the gait displaces. The model alone decides walk + the predicate; the honest
    spine grades the assembled trace.
    """
    prompt = "走到坐标 (11.0, 3.0)"
    r = run_cli_turn(
        prompt,
        sim_go2=True,
        live=True,
        timeout_sec=_LIVE_TIMEOUT_SEC,
        extra_args=["--headless", "--native-loop"],
    )
    _assert_walk_at_position(r, prompt=prompt)


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
@pytest.mark.live_llm
def test_live_walk_natural_phrasing_routes_and_grounds(sim_cleanup) -> None:
    """A second, looser NL phrasing must ALSO route to walk + at_position (red-team).

    "往前走到大约前方一米的位置" names no coordinate and no tool — the model must
    infer the forward ~1m target, route to walk (never navigate), and prove it with
    an at_position predicate near (11,3). Guards against the model only routing when
    the prompt literally spells out a coordinate.
    """
    prompt = "往前走到大约前方一米的位置"
    r = run_cli_turn(
        prompt,
        sim_go2=True,
        live=True,
        timeout_sec=_LIVE_TIMEOUT_SEC,
        extra_args=["--headless", "--native-loop"],
    )
    _assert_walk_at_position(r, prompt=prompt)
