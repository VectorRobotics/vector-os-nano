# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""tests/harness/gt_audit.py — the e2e tier's independent ground-truth audit (A3).

Hand-built transcripts and dumps, and a stand-in REPL on the real PTY harness: the
audit never trusts the verdict, a claimed turn the dump denies is red, and so is a
claimed turn with no row or no ground truth. The expectation table is checked against
the world and robot manifests, so it cannot go stale."""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

from tests.harness import gt_audit, live_matrix, pty_cli

_E2E = gt_audit.TABLE.parent
_PAIR = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")


# --- the table against the manifests ---------------------------------------------

def _stale(row: dict, bodies: list[str], sentence: str) -> str:
    """'' when every name and number of *row* is in the manifests of *bodies*, else what is not."""
    from vector_os_nano.embodiments.config import load_embodiment_config
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    cfgs = [load_embodiment_config(b) for b in bodies]
    worlds = [load_world(c.world) for c in cfgs]
    spawns = [tuple(c.spawn.xy) for c in cfgs if c.spawn is not None]
    key, arg = gt_audit.expectation(row)
    if key == "held_lifted":
        ok = any(arg in w.pickables for w in worlds)
    elif key == "delivered":
        ok = any(arg in w.pickables and w.user is not None for w in worlds)
    elif key == "object_on":
        ok = any(arg[0] in w.pickables and arg[1] in w.surfaces for w in worlds)
    elif key == "base_in":
        ok = any(tuple(arg) in w.boxes.values() for w in worlds)
    else:  # base_at: a room point, the user, the spawn, a pair the sentence names, or spawn + from_spawn
        points = {tuple(p) for w in worlds for p in (*w.rooms.values(), *([w.user] if w.user else []))}
        points |= set(spawns) | {(float(x), float(y)) for x, y in _PAIR.findall(sentence)}
        if "from_spawn" in row:
            dx, dy = row["from_spawn"]
            points = {(x + dx, y + dy) for x, y in spawns}
        ok = tuple(map(float, arg)) in points
    return "" if ok else f"{sentence!r}: {row} names nothing in the manifests of {bodies}"


def test_every_row_of_the_table_names_what_the_manifests_hold() -> None:
    flows = gt_audit.load_table()
    stale = [_stale(row, f["bodies"], s) for f in flows.values() for s, row in f["turns"].items() if "none" not in row]
    assert not [s for s in stale if s], stale


def test_every_cell_of_the_live_matrix_names_what_the_manifests_hold() -> None:
    cells = [c for c in live_matrix.load_cells() if "not_yet" not in c]
    stale = [_stale(c["expect"], [c["body"]], c["task"]) for c in cells]
    assert not [s for s in stale if s], stale


@pytest.mark.parametrize("row, words", [
    ({"held_lifted": "pickable_bottle_purple"}, "names nothing"),          # no such pickable
    ({"base_in": [14.0, 0.0, 20.0, 5.5]}, "names nothing"),                # a box no room has
    ({"base_at": [11.0, 3.0]}, "names nothing"),                           # a point from nowhere
    ({"base_at": [11.0, 3.0], "from_spawn": [2.0, 0.0]}, "names nothing"),  # the wrong offset
])
def test_a_stale_row_is_caught(row, words) -> None:
    assert words in _stale(row, ["go2"], "往前走一米")
    assert _stale({"base_at": [11.0, 3.0], "from_spawn": [1.0, 0.0]}, ["go2"], "往前走一米") == ""


def test_the_flow_is_the_e2e_file_of_the_running_test(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e_repl/test_e7_so101_banana.py::test_e7_x (setup)")
    assert gt_audit.current_flow() == "e7_so101_banana"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/vcli/test_native_loop_recover_pty.py::test_x (call)")
    assert gt_audit.current_flow() == ""


def test_every_flow_that_starts_a_simulator_has_rows_and_every_row_a_flow() -> None:
    flows = set(gt_audit.load_table())
    files = {p.stem.removeprefix("test_"): p.read_text(encoding="utf-8") for p in _E2E.glob("test_*.py")}
    sims = {f for f, src in files.items() if re.search(r"start_simulation|--embodiment", src)}
    assert sims - flows == set(), f"flows with no rows in {gt_audit.TABLE.name}: {sorted(sims - flows)}"
    assert flows - set(files) == set(), f"rows for flows that do not exist: {sorted(flows - set(files))}"


@pytest.mark.parametrize("row", [{"none": " "}, {"none": "x", "held_lifted": "b"}, {"teleported": "b"},
                                 {"held_lifted": "b", "delivered": "b"}, {"held_lifted": "b", "from_spawn": [1, 0]}])
def test_a_bad_row_fails_loud(tmp_path, row) -> None:
    p = tmp_path / "t.yaml"
    p.write_text(json.dumps({"schema": 1, "flows": {"f": {"bodies": ["so101"], "turns": {"s": row}}}}))
    with pytest.raises(ValueError):
        gt_audit.load_table(p)


# --- the audit, on hand-built dumps ------------------------------------------------------

_TABLE = {"standin": {"bodies": ["so101"], "turns": {
    "启动": {"none": "a launch claims nothing"},
    "抓起香蕉": {"held_lifted": "banana"},
    "去厨房": {"base_in": [14.0, 0.0, 20.0, 5.0]},
    "回到初始位置": {"none": "the arm going home"},
}}}
_YES = {"evidence": "GROUNDED", "verified": True, "grounded": "1/1", "goal_check": "proven",
        "line": "verdict GROUNDED verified=True (1/1 grounded) goal=proven"}
_NO = {**_YES, "evidence": "FAILED", "verified": False, "goal_check": "failed",
       "line": "verdict FAILED verified=False (0/1 grounded) goal=failed"}


def _dump(path: Path, *turns: tuple[str, float, bool, tuple | None]) -> Path:
    """One start/end pair per (goal, banana z at the end, welded at the end, base xy at the end)."""
    lines = []
    for n, (goal, z, welded, xy) in enumerate(turns, 1):
        for phase, (zz, w, b) in (("start", (0.02, False, (10.0, 3.0))), ("end", (z, welded, xy or (10.0, 3.0)))):
            lines.append({"phase": phase, "turn": n, "goal": goal, "surfaces": {}, "user": None, "base": [*b, 0.3, 0.0],
                          "objects": {"banana": {"pos": [0.3, 0.0, zz], "welded": w}}})
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    return path


def _audit(tmp_path, turns, dump_turns, *, dump=True):
    d = _dump(tmp_path / "x.gt.jsonl", *dump_turns) if dump else tmp_path / "missing.gt.jsonl"
    return {r["sentence"]: r for r in gt_audit.audit("standin", turns, d, _TABLE)}


def test_a_verified_pick_while_the_dump_shows_the_banana_on_the_table_is_false_grounded(tmp_path) -> None:
    r = _audit(tmp_path, [("抓起香蕉", _YES)], [("抓起香蕉", 0.02, False, None)])["抓起香蕉"]
    assert (r["grade"], r["outcome"]) == (False, "FALSE_GROUNDED")
    ok = _audit(tmp_path, [("抓起香蕉", _YES)], [("抓起香蕉", 0.12, True, None)])["抓起香蕉"]
    assert (ok["grade"], ok["outcome"]) == (True, "agree")


def test_a_verified_room_while_the_base_stopped_outside_its_box_is_false_grounded(tmp_path) -> None:
    r = _audit(tmp_path, [("去厨房", _YES)], [("去厨房", 0.02, False, (13.9, 3.0))])["去厨房"]  # at the door
    assert (r["grade"], r["outcome"]) == (False, "FALSE_GROUNDED")
    ok = _audit(tmp_path, [("去厨房", _YES)], [("去厨房", 0.02, False, (14.4, 3.0))])["去厨房"]
    assert ok["outcome"] == "agree"


def test_a_proven_goal_is_a_claim_even_when_the_turn_did_not_verify(tmp_path) -> None:
    proven = {**_NO, "goal_check": "proven"}
    r = _audit(tmp_path, [("抓起香蕉", proven)], [("抓起香蕉", 0.02, False, None)])["抓起香蕉"]
    assert r["claimed"] and r["outcome"] == "FALSE_GROUNDED"
    reject = _audit(tmp_path, [("抓起香蕉", _NO)], [("抓起香蕉", 0.12, True, None)])["抓起香蕉"]
    assert (reject["claimed"], reject["outcome"]) == (False, "FALSE_REJECT")


def test_a_claimed_turn_with_no_row_is_unaudited_and_one_not_claimed_is_listed(tmp_path) -> None:
    recs = _audit(tmp_path, [("挥挥手", _YES), ("看看", None)], [("挥挥手", 0.02, False, None)])
    assert recs["挥挥手"]["outcome"] == "UNAUDITED" and "gt_expect.yaml" in recs["挥挥手"]["note"]
    assert (recs["看看"]["outcome"], recs["看看"]["note"]) == ("not_gradable", "no row (not claimed)")
    unknown = gt_audit.audit("a_new_flow", [("抓起香蕉", _YES)], tmp_path / "x", _TABLE)
    assert unknown[0]["outcome"] == "UNAUDITED"


@pytest.mark.parametrize("dump", ["missing", "empty", "other turn"])
def test_a_claimed_turn_with_no_ground_truth_is_red_never_nothing_to_grade(tmp_path, dump) -> None:
    if dump == "missing":
        recs = _audit(tmp_path, [("抓起香蕉", _YES)], [], dump=False)
    elif dump == "empty":
        (tmp_path / "x.gt.jsonl").write_text("")
        recs = {r["sentence"]: r for r in gt_audit.audit("standin", [("抓起香蕉", _YES)], tmp_path / "x.gt.jsonl", _TABLE)}
    else:
        recs = _audit(tmp_path, [("抓起香蕉", _YES)], [("去厨房", 0.12, True, None)])
    assert recs["抓起香蕉"]["outcome"] == "NO_GT", recs
    # not claimed and no ground truth: listed, not red
    assert _audit(tmp_path, [("抓起香蕉", _NO)], [], dump=False)["抓起香蕉"]["outcome"] == "ungraded"


def test_the_same_sentence_twice_is_graded_on_its_own_turn(tmp_path) -> None:
    recs = gt_audit.audit("standin", [("抓起香蕉", _NO), ("抓起香蕉", _YES)],
                          _dump(tmp_path / "x.gt.jsonl", ("抓起香蕉", 0.12, True, None), ("抓起香蕉", 0.02, False, None)),
                          _TABLE)
    assert [(r["nth"], r["grade"], r["outcome"]) for r in recs] == [(1, True, "FALSE_REJECT"), (2, False, "FALSE_GROUNDED")]


def test_check_fails_the_flow_with_the_sentence_the_verdict_and_the_grade(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gt_audit, "load_table", lambda: _TABLE)
    monkeypatch.setenv(gt_audit.AUDIT_DIR_ENV, str(tmp_path / "run"))
    dump = _dump(tmp_path / "s.gt.jsonl", ("抓起香蕉", 0.02, False, None))
    text = "vector> 启动\n  ▸ Started so101 simulation, ready in 1.0s\nvector> 抓起香蕉\n  " + _YES["line"] + "\n"
    with pytest.raises(AssertionError) as e:
        gt_audit.check("standin", [("启动", None), ("抓起香蕉", _YES)], text, dump)
    msg = str(e.value)
    assert "FALSE_GROUNDED standin '抓起香蕉'" in msg and _YES["line"] in msg and "grade False" in msg, msg
    assert (tmp_path / "s.transcript.txt").read_text(encoding="utf-8") == text  # beside its dump
    # recorded in the run's folder, wherever the dump is (the live matrix keeps its own)
    assert len((tmp_path / "run" / "turns.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    # a session that started no simulator (the dev world) is not audited at all
    assert gt_audit.check("standin", [("挥挥手", _YES)], "vector> 挥挥手\n", dump) == []


def test_the_run_report_counts_every_outcome_and_lists_the_reasons(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(gt_audit, "load_table", lambda: _TABLE)
    monkeypatch.setenv(gt_audit.AUDIT_DIR_ENV, str(tmp_path))
    dump = _dump(tmp_path / "s.gt.jsonl", ("抓起香蕉", 0.12, True, None), ("回到初始位置", 0.12, True, None))
    text = "  ▸ Started so101 simulation, ready in 1.0s\n"
    gt_audit.check("standin", [("启动", None), ("抓起香蕉", _YES), ("回到初始位置", _YES)], text, dump)
    assert gt_audit.main([str(tmp_path)]) == 0
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert (rep["turns"], rep["graded"], len(rep["agree"])) == (3, 1, 1)
    assert {r["sentence"]: r["note"] for r in rep["not_gradable"]} == {
        "启动": "a launch claims nothing", "回到初始位置": "the arm going home"}
    out = capsys.readouterr().out
    assert "gt_audit: 3 turns audited, 1 graded (1 agree), FALSE GROUNDED 0, FALSE REJECT 0" in out, out
    with pytest.raises(AssertionError):
        gt_audit.check("standin", [("抓起香蕉", _YES)], text, _dump(tmp_path / "t.gt.jsonl", ("抓起香蕉", 0.02, False, None)))
    assert gt_audit.main([str(tmp_path)]) == 1  # a FALSE GROUNDED anywhere in the run is red
    assert "FALSE GROUNDED 1" in capsys.readouterr().out


# --- the harness hook, through the real PTY driver and a stand-in REPL -----------------------

_STANDIN = textwrap.dedent('''
    import json, os, sys
    dump, lift = os.environ["VECTOR_GT_DUMP"], float(os.environ["STANDIN_LIFT"])
    sim = os.environ.get("STANDIN_SIM", "ok")
    turn = 0
    def mark(phase, goal, z):
        with open(dump, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"phase": phase, "turn": turn, "goal": goal, "base": None, "surfaces": {}, "user": None,
                                 "objects": {"banana": {"pos": [0.3, 0.0, z], "welded": z > 0.02}}}) + "\\n")
    print("  Model: standin-model | Provider: standin")
    while True:
        print("vector> ", end="", flush=True)
        line = sys.stdin.readline().strip()
        if not line or line == "quit":
            break
        if line == "启动":
            print("  ▸ Started so101 simulation, ready in 1.0s" if sim == "ok" else "  ▸ Failed to start so101 simulation")
            continue
        turn += 1
        mark("start", line, 0.02)
        mark("end", line, 0.02 + lift)
        print("  verdict GROUNDED verified=True (1/1 grounded) goal=proven", flush=True)
''')


@pytest.fixture
def standin(tmp_path, monkeypatch):
    script = tmp_path / "standin.py"
    script.write_text(_STANDIN, encoding="utf-8")
    fake = tmp_path / "fake_python"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pty_cli.sys, "executable", str(fake))
    monkeypatch.setattr(gt_audit, "load_table", lambda: _TABLE)
    monkeypatch.setenv(gt_audit.AUDIT_DIR_ENV, str(tmp_path / "audit"))
    monkeypatch.setattr(gt_audit, "current_flow", lambda: "standin")  # as if tests/e2e_repl/test_standin.py

    def run(lines, out_dir=None, **env):
        return pty_cli.run_repl_session(lines, timeout_sec=20.0, cwd=tmp_path, out_dir=out_dir, extra_env=env)
    return run


def test_every_repl_session_dumps_and_an_e2e_flow_is_audited(standin, tmp_path) -> None:
    r = standin(["启动", "抓起香蕉"], STANDIN_LIFT="0.1", VECTOR_GT_DUMP="/dev/null")  # a caller's dump never wins
    assert not r.timed_out, r.text
    assert Path(r.dump).parent == tmp_path / "audit" and Path(r.dump).name.startswith("standin-")
    assert [json.loads(x)["phase"] for x in Path(r.dump).read_text().splitlines()] == ["start", "end"]
    assert [(a["sentence"], a["outcome"]) for a in r.audit] == [("启动", "not_gradable"), ("抓起香蕉", "agree")]
    # a caller's own folder (a live-matrix cell) keeps the dump; the run's folder still records the turns
    cell = standin(["启动", "抓起香蕉"], out_dir=tmp_path / "cell", STANDIN_LIFT="0.1")
    assert Path(cell.dump).parent == tmp_path / "cell"
    assert len((tmp_path / "audit" / "turns.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_a_flow_the_dump_denies_is_red_inside_the_harness(standin) -> None:
    with pytest.raises(AssertionError, match=r"FALSE_GROUNDED standin '抓起香蕉' \(#1\): verdict 'verdict GROUNDED"):
        standin(["启动", "抓起香蕉"], STANDIN_LIFT="0")


def test_a_session_that_starts_no_simulator_is_not_audited(standin) -> None:
    r = standin(["挥挥手"], STANDIN_LIFT="0")  # verified, no row, a false dump: no sim, so no audit
    assert r.audit == () and "verdict GROUNDED" in r.text


def test_a_gated_line_is_never_typed_when_what_it_waits_on_never_shows(standin) -> None:
    gated = (re.compile(r"Started so101 simulation, ready in"), "抓起香蕉")
    r = standin(["启动", gated], STANDIN_LIFT="0", STANDIN_SIM="fail")
    assert not r.timed_out and "Failed to start" in r.text, r.text
    assert "vector> 抓起香蕉" not in r.text, r.text
    ok = standin(["启动", gated], STANDIN_LIFT="0.1")
    assert "vector> 抓起香蕉" in ok.text and ok.audit[-1]["outcome"] == "agree", ok.text
