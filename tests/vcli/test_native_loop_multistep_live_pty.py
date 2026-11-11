# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 STEP 3 (B) — LIVE multi-STEP: the model autonomously decomposes a 2-leg journey.

Step 2 (``test_native_loop_live_llm_pty.py``) proved the REAL model drives ONE
autonomous walk->verify leg. The legacy planner's defining job is multi-STEP
DECOMPOSITION: given one instruction naming several waypoints, break it into ordered
verified legs. This module is that proof for the native producer — a SINGLE NL turn
naming TWO targets, decomposed by the MODEL (not the runner) into two walk->verify
legs on the REAL go2 sim, graded by the UNTOUCHED honest spine.

The prompt names BOTH coordinates and NO tool::

    先走到坐标 (11.0,3.0)，到了之后再走到 (12.0,3.0)

The runner owns NONE of the decomposition — it does not inject legs, does not
replan, does not sentinel-verify. The model alone must:
  - issue >= 2 walk->verify legs (the multi-step decomposition);
  - route EVERY leg to ``walk`` (never ``navigate``, not offered) verifying with the
    registry-derived ``at_position(...)`` oracle (the fix-6 routing contract);
  - target BOTH coordinates — at least one leg verifying near (11,3) AND at least
    one near (12,3) (proves it decomposed the journey, not just walked once);
  - earn >= 1 GROUNDED step from a real displacing walk.

HONEST grading (reuses step 2's philosophy, never loosens the moat): the honest
``evidence_passed`` gate requires EVERY step GROUNDED for the whole turn to verify.
The legged gait UNDER-SHOOTS each ~1m walk command, so a leg frequently lands short
(RAN) before the model autonomously walks again (GROUNDED). A RAN+GROUNDED mix
honestly reports the whole turn RAN/exit2 — that is the gate working, not a failure.
We therefore pin full-turn ``verified True`` ONLY when EVERY step is grounded; we
NEVER force a partial run green. ``verified == (exit==0)`` is asserted inside
run_cli_turn regardless.

GATE (LIVE, non-deterministic, billed). SAME double gate as step 2: SKIPPED unless
BOTH a real provider key is present in the repo-root ``.env`` AND ``VECTOR_LIVE_LLM=1``
is set. Reproduce::

    VECTOR_LIVE_LLM=1 MUJOCO_GL=egl PATH=/usr/bin:$PATH \\
      .venv/bin/python -m pytest tests/vcli/test_native_loop_multistep_live_pty.py -v -s -m sim

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after the
case via the ``sim_cleanup`` fixture (the generated scene xml is untracked).
"""
from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

# A live multi-step journey: cold imports + sim boot + several LLM turns + the MPC
# gait walking two ~1m legs. Generous headroom.
_LIVE_TIMEOUT_SEC = 300.0

# The repo-root .env the child's resolve_credentials reads (single-sourced from the
# harness root, exactly as the step-2 live test does).
_REPO_ROOT = run_cli_turn.__globals__["_REPO_ROOT"]
_ENV_PATH = _REPO_ROOT / ".env"

# at_position tol (single-sourced from the oracle) — a verify expr is "near" a target
# when its coordinate args are within this tol of the target. A safe literal fallback
# keeps the gate parse-only (it never imports the sim) if the const ever moves.
try:
    from vector_os_nano.vcli.worlds.sim_oracles import _AT_POSITION_TOL_M as _TOL_M
except Exception:  # noqa: BLE001
    _TOL_M = 0.5
_NEAR_TOL: float = float(_TOL_M)


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


# The DOUBLE gate (same as step 2): a real key AND the explicit opt-in. Module scope
# so the routine suite skips it and collection stays cheap.
_SKIP_REASON = (
    "live-LLM acceptance: needs a real provider key in repo-root .env AND "
    "VECTOR_LIVE_LLM=1 (deliberate, billed opt-in)"
)
pytestmark = pytest.mark.skipif(
    not (_env_has_provider_key() and _live_opted_in()),
    reason=_SKIP_REASON,
)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _verify_targets_near(verify: str, x: float, y: float) -> bool:
    """True iff *verify* is an ``at_position(...)`` whose first two numeric args are
    within ``_NEAR_TOL`` of ``(x, y)``.

    Parse-only (the first two numbers in the expr are the target coords) — never
    eval. A non-at_position expr, or one without two coords, is not "near".
    """
    s = str(verify or "")
    if not s.startswith("at_position"):
        return False
    nums = _NUM_RE.findall(s)
    if len(nums) < 2:
        return False
    try:
        tx, ty = float(nums[0]), float(nums[1])
    except ValueError:
        return False
    return abs(tx - x) <= _NEAR_TOL and abs(ty - y) <= _NEAR_TOL


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
@pytest.mark.live_llm
def test_live_multistep_two_leg_journey_decomposes_and_routes(sim_cleanup) -> None:
    """REAL model decomposes "走到 (11,3) 再走到 (12,3)" into >=2 walk->at_position legs.

    go2 starts at (10,3). The journey is two ~1m forward legs to (11,3) then (12,3).
    The model alone decides the legs + each predicate; the honest spine grades the
    assembled trace. The acceptance is the decomposition + routing + at least one
    grounded leg; full-turn verified is pinned only when every leg grounds (gait
    noise may leave a short leg RAN — tolerated, never forced green).
    """
    prompt = "先走到坐标 (11.0,3.0)，到了之后再走到 (12.0,3.0)"
    r = run_cli_turn(
        prompt,
        sim_go2=True,
        live=True,
        timeout_sec=_LIVE_TIMEOUT_SEC,
        extra_args=["--headless"],
    )
    # Echo the live verdict (visible under `pytest -s`) so a real run's autonomous
    # multi-step decomposition is inspectable in the log, not just pass/fail.
    print(f"\n[live verdict] {prompt!r} -> {r.verdict}")
    per_step = r.verdict.get("per_step") or []

    # (0) MULTI-STEP — the model decomposed the journey into >= 2 verified legs.
    assert len(per_step) >= 2, (
        f"a 2-leg journey must decompose into >=2 native steps; got "
        f"{len(per_step)}: {per_step}. verdict={r.verdict} raw=\n{r.raw_output[-2000:]}"
    )

    # (1) ROUTING contract on EVERY leg (fix 6): walk + at_position, never navigate.
    for i, step in enumerate(per_step):
        assert step["strategy"] == "walk", (
            f"step {i}: every leg must route to walk (not navigate); got "
            f"strategy={step['strategy']!r}. verdict={r.verdict}"
        )
        assert str(step["verify"]).startswith("at_position"), (
            f"step {i}: every leg must verify with at_position(...); got "
            f"verify={step['verify']!r}. verdict={r.verdict}"
        )

    # (2) BOTH TARGETS — at least one leg aims near (11,3) AND one near (12,3). This
    # is the multi-step proof: the model targeted the two distinct waypoints, not
    # just walked once or walked twice to the same place.
    near_first = [s for s in per_step if _verify_targets_near(s["verify"], 11.0, 3.0)]
    near_second = [s for s in per_step if _verify_targets_near(s["verify"], 12.0, 3.0)]
    assert near_first, (
        f"the model must verify a leg near (11,3); per_step verifies="
        f"{[s['verify'] for s in per_step]}. verdict={r.verdict}"
    )
    assert near_second, (
        f"the model must verify a leg near (12,3); per_step verifies="
        f"{[s['verify'] for s in per_step]}. verdict={r.verdict}"
    )

    # (3) HONEST GROUNDING — at least one real displacing walk earned a GROUNDED step.
    grounded = [
        s for s in per_step
        if s["evidence"] == "GROUNDED" and s["verify_result"] is True
    ]
    assert grounded, (
        f"a real multi-leg journey must earn >=1 GROUNDED at_position step; got "
        f"per_step={per_step}. verdict={r.verdict}"
    )

    # FULL-TURN verified ONLY when EVERY leg grounded (never forced green on a mix).
    if all(s["evidence"] == "GROUNDED" for s in per_step):
        assert r.verified is True, (
            f"every leg grounded -> the whole turn must verify True; verdict={r.verdict}"
        )
        assert r.exit_code == 0
        assert r.evidence == "GROUNDED"
