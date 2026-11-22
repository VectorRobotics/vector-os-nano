# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E11 perception + spatial memory (WP11) through bare vector-cli on go2_piper, no VLM key.

1. '看看周围有什么' -> look (grounding-dino over the world vocabulary on one RGB-D
   snapshot) -> verify localized(...) of the green bottle, blue bottle and red can at
   0.05 m of ground truth, GROUNDED. '去绿色的瓶子那里' and 'go to the green bottle'
   drive to the SAME scene-graph instance.
2. A new vector-cli in the same HOME restores the objects from the world-keyed
   file (none at (0, 0)) and drives to the green bottle from memory; the memory alone
   never grounds localized() this turn (it was not perceived).
3. '用 RynnBrain 找一下绿色瓶子' routes detect to the RynnBrain backend (an external
   service: skipped when it is down) and its box agrees with grounding-dino's.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import yaml

from tests.e2e_repl import LAUNCH, LAUNCH_STEPS
from tests.harness.pty_cli import run_repl_session

_LOOK, _ZH, _EN = "看看周围有什么", "去绿色的瓶子那里", "go to the green bottle"
_THREE = ("localized('green bottle', 0.05) and localized('blue bottle', 0.05) "
          "and localized('red can', 0.05)")
_GO = {"name": "navigate_to_object"}
_SESSION1 = {"steps": LAUNCH_STEPS + [
    {"when": _LOOK, "tool_calls": [{"name": "look", "input": {}}]},
    {"when": "'world_x'", "tool_calls": [{"name": "verify", "input": {"expr": _THREE}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": _ZH, "tool_calls": [{**_GO, "input": {"object": "绿色的瓶子"}}]},
    {"when": "'object_id'", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": _EN, "tool_calls": [{**_GO, "input": {"object": "green bottle"}}]},
    {"when": "'object_id'", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}
_RECALL = "再去绿色的瓶子那里"
_SESSION2 = {"steps": LAUNCH_STEPS + [
    {"when": _RECALL, "tool_calls": [{**_GO, "input": {"object": "绿色的瓶子"}}]},
    {"when": "'object_id'", "tool_calls": [
        {"name": "verify", "input": {"expr": "localized('green bottle')"}}]},
    {"when": "", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}
_GT = {"green bottle": (10.88, 3.00), "blue bottle": (10.90, 2.78), "red can": (10.90, 3.22)}


def _turn(text: str, cmd: str, nxt: str | None = None) -> str:
    i = text.find(cmd)
    assert i != -1, f"{cmd!r} never typed:\n{text[-3000:]}"
    out = text[i:]
    out = out[: out.find(nxt)] if nxt and nxt in out else out
    return " ".join(out.split())


def _observations(log: Path) -> list[str]:
    return [json.loads(line)["obs"] for line in log.read_text(encoding="utf-8").splitlines()]


def _ids(obs: list[str]) -> list[str]:
    return [m for o in obs for m in re.findall(r"'object_id': '(obj_[0-9a-f]+)'", o)]


@pytest.mark.sim
@pytest.mark.cli_main
def test_e11_look_localizes_remembers_and_resolves_zh_en_to_one_instance(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    state = {"HOME": str(home), "VECTOR_HOME": str(home / ".vector")}
    log1 = tmp_path / "llm1.jsonl"
    r = run_repl_session([LAUNCH, _LOOK, _ZH, _EN], tool_script=_SESSION1, answers=["y", "y"],
                         extra_env={**state, "VECTOR_FAKE_LLM_LOG": str(log1)},
                         timeout_sec=480)
    text = r.text
    assert not r.timed_out, text[-3000:]
    look = _turn(text, _LOOK, _ZH)
    assert f"look → verify {_THREE} ✓ (actor=CAUSED)" in look, look[-3000:]
    assert "verdict GROUNDED verified=True" in look, look[-3000:]
    obs = _observations(log1)
    # the start line the model relays carries the perception capability line
    started = next(o for o in obs if "Started go2_piper simulation" in o)
    assert re.search(r"perception: detector=grounding-dino@\w+ \(cached\), segmenter=edgetam \(cached\), "
                     r"self_mask=yes, vlm=none \(failed: ValueError: no OPENROUTER_API_KEY set for scene VLM; set the key in the environment\), "
                     r"rynnbrain=(up|down)", started), started
    ids = _ids(obs)
    assert len(ids) == 2 and ids[0] == ids[1], f"zh and en names went to different instances: {ids}"

    # The world-keyed file a restart reads: every object localized, none at (0, 0).
    sg_file = home / ".vector" / "memory" / "apartment" / "scene_graph.yaml"
    objs = (yaml.safe_load(sg_file.read_text()) or {}).get("objects") or {}
    assert {o["category"] for o in objs.values()} >= set(_GT), objs
    for o in objs.values():
        assert o["x"] is not None and (o["x"], o["y"]) != (0.0, 0.0), o
        if o["category"] in _GT:
            assert math.dist((o["x"], o["y"]), _GT[o["category"]]) <= 0.05, o

    log2 = tmp_path / "llm2.jsonl"
    r = run_repl_session([LAUNCH, _RECALL], tool_script=_SESSION2, answers=["y"],
                         extra_env={**state, "VECTOR_FAKE_LLM_LOG": str(log2)},
                         timeout_sec=300)
    text = r.text
    assert not r.timed_out, text[-3000:]
    obs = _observations(log2)
    assert _ids(obs) == [ids[0]], "the restarted CLI must reach the remembered instance"
    world = re.search(r"'object_world': \[([-0-9.]+), ([-0-9.]+)\]", "\n".join(obs))
    assert world and math.dist((float(world[1]), float(world[2])), _GT["green bottle"]) <= 0.05
    recall = _turn(text, _RECALL)
    # memory from an earlier session is not this turn's perception: no false green
    assert "verify localized('green bottle') ·" in recall, recall[-3000:]
    assert "verdict GROUNDED" not in recall, recall[-3000:]


_UNREADY = {"steps": LAUNCH_STEPS + [
    {"when": _LOOK, "tool_calls": [{"name": "look", "input": {}}]},
    {"when": "(perception_failed)", "is_error": True, "text": "看不了：检测模型没有就绪。", "tool_calls": []},
]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_e11_a_look_without_the_weights_fails_loud_with_the_prefetch_command(tmp_path) -> None:
    """perception:look-reports-crashed-detector-as-nothing: a detector that cannot run
    made look answer 'nothing localized'; it is a failed look with its cause now."""
    hf = tmp_path / "empty_hf_home"
    hf.mkdir()
    log = tmp_path / "llm.jsonl"
    r = run_repl_session([LAUNCH, _LOOK], tool_script=_UNREADY, answers=["y"],
                         extra_env={"HF_HOME": str(hf), "HF_HUB_OFFLINE": "1",
                                    "VECTOR_FAKE_LLM_LOG": str(log)}, timeout_sec=300)
    assert not r.timed_out, r.text[-3000:]
    look = _turn(r.text, _LOOK)
    assert "python scripts/fetch_assets.py --weights" in look, look[-3000:]
    assert "nothing localized" not in look and "verdict GROUNDED" not in look, look[-3000:]
    told = next(o for o in _observations(log) if "(perception_failed)" in o)
    assert told.startswith("Perception could not run for any of the "), told
    assert "the detector failed: IDEA-Research/grounding-dino-tiny@" in told, told
    assert "pre-fetch with: python scripts/fetch_assets.py --weights" in told and str(hf) not in told, told


_RYNN = "用 RynnBrain 找一下绿色瓶子"
_RYNN_SCRIPT = {"steps": LAUNCH_STEPS + [
    {"when": _RYNN, "tool_calls": [
        {"name": "detect", "input": {"query": "绿色瓶子", "backend": "rynnbrain"}}]},
    {"when": "rynnbrain localized", "tool_calls": [
        {"name": "detect", "input": {"query": "green bottle"}}]},
    {"when": "grounding-dino localized", "tool_calls": [{"name": "finish", "input": {}}],
     "stop_reason": "end_turn"},
]}


def _box(obs: str) -> list[float]:
    m = re.search(r"box=\[([-0-9., ]+)\]", obs)
    assert m, obs
    return [float(v) for v in m[1].split(",")]


@pytest.mark.sim
@pytest.mark.cli_main
def test_e11_detect_routes_to_rynnbrain_and_agrees_with_grounding_dino(tmp_path) -> None:
    from vector_os_nano.perception.rynnbrain import RynnBrainDetector

    if not RynnBrainDetector().available(timeout=2.0):
        pytest.skip("RynnBrain (an external service at :8786) is not running")
    log = tmp_path / "llm.jsonl"
    r = run_repl_session([LAUNCH, _RYNN], tool_script=_RYNN_SCRIPT,
                         extra_env={"VECTOR_FAKE_LLM_LOG": str(log)}, timeout_sec=300)
    assert not r.timed_out, r.text[-3000:]
    obs = _observations(log)
    rynn = next(o for o in obs if "rynnbrain localized" in o)
    gd = next(o for o in obs if "grounding-dino localized" in o)
    a, b = _box(rynn), _box(gd)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    assert ix / union >= 0.3, f"RynnBrain box {a} vs grounding-dino {b}"
