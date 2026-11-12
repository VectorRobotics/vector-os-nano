# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E3a as a paired-seed campaign: the grasp is physics, so its rate over seeds is the measure.

    VECTOR_GRASP_CAMPAIGN=8 GATE_PATHS=tests/e2e_repl/test_e3a_grasp_campaign.py \\
        with-sim-slot scripts/gate.sh e2e

Each seed is one ``vector-cli --embodiment go2_piper --headless -p 拿起绿色的瓶子 --json``
turn: the scripted model nudges the dog (walk back, sidestep, turn; the seed draws
how far), then grasps and verifies. A seed counts when its ``VECTOR_VERDICT`` says
so: the grasp's step is GROUNDED and actor-CAUSED, and the goal check is ``proven``.
A seed that does not fails with its diagnosis (what the grasp told the model, and
why the verdict did not ground it). The campaign ends with one line,
``campaign: k of N grounded``, naming each failed seed and its diagnosis; the same
seeds before and after a change are a paired comparison. Skipped unless
VECTOR_GRASP_CAMPAIGN is set (VECTOR_GRASP_CAMPAIGN_SEED0 shifts the seeds).
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import pytest

from tests.harness.pty_cli import run_cli_turn

_N = int(os.environ.get("VECTOR_GRASP_CAMPAIGN", "0") or 0)
_SEED0 = int(os.environ.get("VECTOR_GRASP_CAMPAIGN_SEED0", "0") or 0)
_TARGET = "pickable_bottle_green"
_GRASP = "拿起绿色的瓶子"
_RESULTS: dict[int, str] = {}  # seed -> "" (grounded) or its diagnosis


def _script(seed: int) -> dict:
    r = random.Random(seed)
    back, side_dir, side = r.uniform(0.0, 0.25), r.choice(["left", "right"]), r.uniform(0.0, 0.10)
    turn_dir, angle = r.choice(["left", "right"]), r.uniform(0.0, 12.0)
    return {"steps": [
        {"when": _GRASP, "tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": round(back, 3)}}]},
        {"when": "", "tool_calls": [{"name": "walk", "input": {"direction": side_dir, "distance": round(side, 3)}}]},
        {"when": "", "tool_calls": [{"name": "turn", "input": {"direction": turn_dir, "angle": round(angle, 1)}}]},
        {"when": "", "tool_calls": [{"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
        {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": f"holding_object('{_TARGET}')"}}]},
        {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}


def _diagnosis(verdict: dict, observations: list[str]) -> str:
    """'' when the grasp grounded, else why not: the verdict's own words and the grasp's."""
    steps = [s for s in verdict.get("per_step", []) if s["strategy"] == "perception_grasp" and not s["superseded"]]
    if steps and steps[-1]["evidence"] == "GROUNDED" and steps[-1]["actor_caused"] == "CAUSED" \
            and verdict.get("goal_check") == "proven":
        return ""
    step = steps[-1] if steps else {}
    # observations[4] is what the model saw after the grasp (the steps fire in order)
    told = observations[4].splitlines()[0] if len(observations) > 4 else "(the grasp never ran)"
    why = step.get("reject_reason") or verdict.get("reject_reason") or ""
    return (f"goal_check={verdict.get('goal_check')} step={step.get('evidence', 'none')}/"
            f"{step.get('actor_caused', '-')}{f' ({why})' if why else ''} | grasp said: {told}")


def test_a_seed_counts_only_when_its_verdict_grounds_the_grasp() -> None:
    """R2V-doc: the campaign asserted a rendered line that no longer appears, so every
    seed read failed while each was GROUNDED. It reads the verdict data now."""
    ok = {"strategy": "perception_grasp", "superseded": False, "evidence": "GROUNDED",
          "actor_caused": "CAUSED", "reject_reason": ""}
    said = ["", "", "", "", "Skill 'perception_grasp' succeeded."]
    assert _diagnosis({"per_step": [ok], "goal_check": "proven"}, said) == ""
    for verdict in ({"per_step": [ok], "goal_check": "failed"},
                    {"per_step": [{**ok, "actor_caused": "UNCAUSED", "reject_reason": "not caused"}],
                     "goal_check": "proven"},
                    {"per_step": [{**ok, "evidence": "FAILED"}], "goal_check": "proven"},
                    {"per_step": [{**ok, "strategy": "walk"}], "goal_check": "proven"}):
        assert _diagnosis(verdict, said).endswith("| grasp said: Skill 'perception_grasp' succeeded."), verdict
    failed = said[:4] + ["The detector failed for '绿色的瓶子': ... (perception_failed)\nData: {}"]
    assert _diagnosis({"per_step": [], "goal_check": "failed"}, failed) == (
        "goal_check=failed step=none/- | grasp said: The detector failed for '绿色的瓶子': ... (perception_failed)")


@pytest.fixture(scope="module")
def campaign(request):
    yield _RESULTS
    if not _RESULTS:
        return
    failed = {s: d for s, d in sorted(_RESULTS.items()) if d}
    line = f"campaign: {len(_RESULTS) - len(failed)} of {len(_RESULTS)} grounded"
    if failed:
        line += "; failed: " + "; ".join(f"seed {s}: {d}" for s, d in failed.items())
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    (reporter.write_line if reporter else print)(line)


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.skipif(not _N, reason="set VECTOR_GRASP_CAMPAIGN=N to run the N-seed grasp campaign")
@pytest.mark.parametrize("seed", range(_SEED0, _SEED0 + max(_N, 1)))
def test_grasp_campaign_seed(seed: int, campaign: dict[int, str]) -> None:
    log = Path(tempfile.mkstemp(prefix="vos_campaign_llm_", suffix=".jsonl")[1])
    try:
        r = run_cli_turn(_GRASP, tool_script=_script(seed), extra_env={"VECTOR_FAKE_LLM_LOG": str(log)},
                         extra_args=["--embodiment", "go2_piper", "--headless"], timeout_sec=280)
        observations = [json.loads(line)["obs"] for line in log.read_text(encoding="utf-8").splitlines()]
        campaign[seed] = _diagnosis(r.verdict, observations)
    except AssertionError as exc:  # no verdict at all: the turn itself failed
        campaign[seed] = f"no verdict: {str(exc)[:300]}"
        raise
    finally:
        log.unlink(missing_ok=True)
    assert not campaign[seed], f"seed {seed}: {campaign[seed]}"
