# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""live_matrix — the live matrix instrument (acceptance A10 of docs/plan-release-1.0.md).

The gate drives the product with a scripted model, which never makes a wrong first
move. This drives it with the REAL one: for every cell of ``live_matrix.yaml`` (body
x backend x task) and every repetition, one bare ``vector-cli`` REPL on a PTY
(tests/harness/pty_cli.run_repl_session), the launch sentence, then the task
sentence, with ``VECTOR_GT_DUMP`` set. Each repetition records the verdict the REPL
printed for the task turn, the tools the model chose (from the session file), the
wall time and the transcript, and is graded INDEPENDENTLY by scripts/gt_grade.py
(run as its own process: it shares no code with the product). The verdict is never
trusted: a turn that verified while the grade says nothing happened is FALSE
GROUNDED, a turn the grade says happened that did not verify is FALSE REJECT.

    /data/vos-analysis/bin/with-sim-slot .venv/bin/python -m tests.harness.live_matrix [--n 2] [--cells ID ...] [--out DIR]

Writes ``live_matrix.json`` and ``live_matrix.md`` to DIR (default
``$VECTOR_HOME/runs/live-matrix-<stamp>``), each repetition's transcript, GT dump and
session under ``DIR/<cell>/rep<k>/``. Billed: two model-driven turns per repetition.
Exit 1 when any FALSE GROUNDED exists, 2 when nothing ran, else 0. A cell that cannot
start (no key for the model, the launch did not bring up its body) is NOT-RUN with
its reason, never a product failure and never silently skipped; a ``not_yet`` cell
is listed with its reason and not run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from tests.harness.pty_cli import PROMPT, run_repl_session
from vector_os_nano.schemas import paths

MATRIX = Path(__file__).with_name("live_matrix.yaml")
GT_GRADE = Path(__file__).resolve().parents[2] / "scripts" / "gt_grade.py"
KINDS = ("navigate", "grasp", "fetch", "place", "home")
_FIELDS = ("id", "body", "backend", "launch", "task", "kind", "expect", "n")


def _grade_args(key: str, arg: Any) -> list[str]:
    """scripts/gt_grade.py flags for one ``expect`` entry (its output field is *key*)."""
    if key in ("base_in", "base_at"):
        return ["--box" if key == "base_in" else "--at", *(str(float(v)) for v in arg)]
    if key in ("held_lifted", "delivered"):
        return ["--object", str(arg)]
    if key == "object_on":
        return ["--object", str(arg[0]), "--surface", str(arg[1])]
    raise ValueError(f"unknown expect {key!r}; one of base_in, base_at, held_lifted, delivered, object_on")


def load_cells(path: Path = MATRIX) -> list[dict[str, Any]]:
    """The matrix's cells, each checked (fail loud: a bad cell names itself)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, this runner reads 1")
    cells = data["cells"]
    for c in cells:
        missing = [f for f in _FIELDS if f not in c]
        if missing:
            raise ValueError(f"cell {c.get('id')!r}: missing {missing}")
        if c["kind"] not in KINDS:
            raise ValueError(f"cell {c['id']!r}: kind {c['kind']!r}, one of {KINDS}")
        if "not_yet" in c:
            if not str(c["not_yet"]).strip():
                raise ValueError(f"cell {c['id']!r}: not_yet needs a reason")
            continue
        if len(c["expect"]) != 1:
            raise ValueError(f"cell {c['id']!r}: expect holds exactly one grade, got {c['expect']}")
        ((key, arg),) = c["expect"].items()
        _grade_args(key, arg)
    if len({c["id"] for c in cells}) != len(cells):
        raise ValueError("duplicate cell ids")
    return cells


# --- reading one repetition -------------------------------------------------

_VERDICT = re.compile(r"verdict (\w+) verified=(True|False) \((\d+)/(\d+) grounded\).*? goal=(\w+)")
_READY = re.compile(r"Started (\w+) simulation, ready in")
_START_FAILED = re.compile(r"(Failed to start .*|Cannot start .*)")


def _split(text: str, task: str) -> tuple[str, str]:
    """(launch part, task turn) of a transcript; the task turn is '' if never typed."""
    i = text.rfind(f"{PROMPT} {task}")
    return (text, "") if i == -1 else (text[:i], text[i:])


def launch_problem(text: str, body: str, task: str) -> str:
    """'' when the launch sentence brought up *body*, else why the cell did not start."""
    launch, _ = _split(text, task)
    ready = _READY.search(launch)
    if ready and ready.group(1) == body:
        return ""
    if ready:
        return f"the launch started {ready.group(1)}, the cell asks for {body}"
    failed = _START_FAILED.search(launch)
    if failed:
        return failed.group(1).strip()[:300]
    tail = " / ".join(ln.strip() for ln in launch.splitlines() if ln.strip())[-300:]
    return f"no 'Started {body} simulation, ready' line after the launch sentence; it ended: {tail}"


def parse_verdict(text: str, task: str) -> dict[str, Any] | None:
    """The verdict line the REPL printed for the task turn (the last one in it), or None."""
    hits = list(_VERDICT.finditer(_split(text, task)[1]))
    if not hits:
        return None
    m = hits[-1]
    return {"evidence": m[1], "verified": m[2] == "True", "grounded": f"{m[3]}/{m[4]}", "goal_check": m[5]}


def _ts(entry: dict[str, Any]) -> float:
    return datetime.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%SZ").timestamp()


def session_facts(path: Path, task: str) -> dict[str, Any]:
    """From the session file: the task turn's tools (``name!`` = returned an error),
    its seconds, and the model round trips of the whole session (the billed calls)."""
    if not path.is_file():
        return {"tools": [], "turn_s": None, "model_calls": 0}
    entries = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    calls = sum(e.get("type") == "assistant" for e in entries)
    starts = [i for i, e in enumerate(entries) if e.get("type") == "user" and e.get("content") == task]
    if not starts:
        return {"tools": [], "turn_s": None, "model_calls": calls}
    turn = entries[starts[-1]:]
    turn = turn[: next((i for i, e in enumerate(turn) if i and e.get("type") == "user"), len(turn))]
    turn = [e for e in turn if e.get("type") != "meta"]
    errors = {r.get("tool_use_id") for e in turn if e.get("type") == "tool_result"
              for r in e.get("results") or () if r.get("is_error")}
    tools = [b["name"] + ("!" if b.get("id") in errors else "")
             for e in turn if e.get("type") == "assistant" for b in e.get("tool_use") or ()]
    return {"tools": tools, "turn_s": _ts(turn[-1]) - _ts(turn[0]), "model_calls": calls}


def gt_grade(dump: Path, expect: dict[str, Any], task: str) -> tuple[bool | None, str]:
    """(the independent grade of the task turn, '' or why it could not be graded)."""
    ((key, arg),) = expect.items()
    if not dump.is_file():
        return None, "no GT dump (no sim turn was recorded)"
    r = subprocess.run([sys.executable, str(GT_GRADE), "--dump", str(dump), *_grade_args(key, arg)],
                       capture_output=True, text=True, timeout=60, check=False)
    if r.returncode:
        return None, f"gt_grade failed: {r.stderr.strip()[-300:]}"
    rows = [row for row in map(json.loads, r.stdout.splitlines()) if row["goal"] == task]
    if not rows:
        return None, "no finished task turn in the GT dump"
    return bool(rows[-1][key]), ""


def judge(cell: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """Grade one repetition from its files; never from the verdict alone."""
    text = Path(rec["transcript"]).read_text(encoding="utf-8")
    out = dict(rec)
    problem = launch_problem(text, cell["body"], cell["task"])
    if problem:
        return {**out, "status": "not_run", "reason": problem}
    verdict = parse_verdict(text, cell["task"])
    grade, why = gt_grade(Path(rec["dump"]), cell["expect"], cell["task"])
    verified = bool(verdict and verdict["verified"])
    if grade is None:
        outcome = "ungraded"
    elif verified != grade:
        outcome = "FALSE_GROUNDED" if verified else "FALSE_REJECT"
    else:
        outcome = "agree"
    return {**out, "status": "ran", "verdict": verdict, "grade": grade, "grade_note": why, "outcome": outcome,
            **session_facts(Path(rec["session"]), cell["task"])}


def run_rep(cell: dict[str, Any], rep: int, out: Path, *, tool_script: dict[str, Any] | None = None) -> dict[str, Any]:
    """One bare REPL session for *cell*: the real model unless *tool_script* is given."""
    d = out / cell["id"] / f"rep{rep}"
    d.mkdir(parents=True, exist_ok=True)
    home = d / "vector_home"  # this session's own state directory: its session file survives
    home.mkdir(exist_ok=True)
    if (paths.home() / "assets").is_dir():
        (home / "assets").symlink_to(paths.home() / "assets")
    dump = d / "gt.jsonl"
    dump.unlink(missing_ok=True)
    t0 = time.monotonic()
    r = run_repl_session(
        [cell["launch"], cell["task"]], tool_script=tool_script, live=tool_script is None,
        answers=["a"] * 20, timeout_sec=float(cell.get("timeout_s", 600)),
        # never the external nav stack on a shared box, whatever the repo .env says
        # ... and a live run is never a scripted one: a caller's test-seam variable does not reach the child
        extra_env={"VECTOR_HOME": str(home), "VECTOR_GT_DUMP": str(dump), "VECTOR_NAV_STACK": "",
                   **({"VECTOR_FAKE_LLM_TOOLS": ""} if tool_script is None else {})},
    )
    rec = {"cell": cell["id"], "rep": rep, "wall_s": round(time.monotonic() - t0, 1), "timed_out": r.timed_out,
           "exit_code": r.exit_code, "transcript": str(d / "transcript.txt"), "dump": str(dump),
           "session": str(d / "session.jsonl")}
    Path(rec["transcript"]).write_text(r.text, encoding="utf-8")
    sessions = sorted((home / "sessions").glob("*.jsonl"))
    if sessions:
        shutil.copy(sessions[-1], rec["session"])
    shutil.rmtree(home, ignore_errors=True)
    return rec


# --- the report ---------------------------------------------------------------

def summarize(cells: list[dict[str, Any]], reps: list[dict[str, Any]], *, skipped: str = "") -> dict[str, Any]:
    """Per cell: success / verified / goal proven k of N run, the false GROUNDED and
    false rejects with their transcripts, what did not run and why."""
    rows = []
    for c in cells:
        mine = [r for r in reps if r["cell"] == c["id"]]
        ran = [r for r in mine if r["status"] == "ran"]
        base = {k: c[k] for k in ("id", "body", "backend", "kind", "launch", "task", "expect")}
        if "not_yet" in c or skipped:
            rows.append({**base, "status": "not_yet" if "not_yet" in c else "not_run",
                         "reason": c.get("not_yet") or skipped, "reps": []})
            continue

        by = {o: [r["transcript"] for r in ran if r["outcome"] == o]
              for o in ("FALSE_GROUNDED", "FALSE_REJECT", "ungraded")}
        not_run = [f"rep{r['rep']}: {r['reason']}" for r in mine if r["status"] == "not_run"]
        rows.append({**base, "status": "ran" if ran else "not_run", "reason": "; ".join(not_run),
                     "n": len(mine), "n_run": len(ran),
                     "success": sum(r["grade"] is True for r in ran),
                     "verified": sum(bool(r["verdict"] and r["verdict"]["verified"]) for r in ran),
                     "goal_proven": sum(bool(r["verdict"] and r["verdict"]["goal_check"] == "proven") for r in ran),
                     "false_grounded": by["FALSE_GROUNDED"], "false_reject": by["FALSE_REJECT"],
                     "ungraded": by["ungraded"], "not_run": not_run, "reps": mine})
    return {"schema": 1, "cells": rows,
            "false_grounded": sum(len(r.get("false_grounded", [])) for r in rows),
            "false_reject": sum(len(r.get("false_reject", [])) for r in rows),
            "reps_run": sum(r.get("n_run", 0) for r in rows),
            "model_calls": sum(r.get("model_calls", 0) for r in reps),
            "wall_s": round(sum(r["wall_s"] for r in reps), 1)}


def exit_code(summary: dict[str, Any]) -> int:
    if summary["false_grounded"]:
        return 1
    return 0 if summary["reps_run"] else 2


def markdown(summary: dict[str, Any]) -> str:
    out = ["| cell | body | backend | kind | success | verified | goal proven | FALSE GROUNDED | FALSE REJECT | not run |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in summary["cells"]:
        if c["status"] != "ran":
            out.append(f"| {c['id']} | {c['body']} | {c['backend']} | {c['kind']} | {c['status']}: {c['reason']} | | | | | |")
            continue
        k = c["n_run"]
        out.append(f"| {c['id']} | {c['body']} | {c['backend']} | {c['kind']} | {c['success']}/{k} | {c['verified']}/{k} "
                   f"| {c['goal_proven']}/{k} | {len(c['false_grounded'])} | {len(c['false_reject'])} | {len(c['not_run'])} |")
    out += ["", f"FALSE GROUNDED {summary['false_grounded']}, FALSE REJECT {summary['false_reject']}; "
            f"{summary['reps_run']} repetitions run, {summary['model_calls']} model calls, {summary['wall_s']} s.", "",
            "| cell | rep | verdict | goal | grade | outcome | tools (! = error) | s | transcript |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in summary["cells"]:
        for r in c["reps"]:
            if r["status"] != "ran":
                out.append(f"| {c['id']} | {r['rep']} | not run: {r['reason']} |||||| {r['transcript']} |")
                continue
            v = r["verdict"] or {"evidence": "none", "verified": False, "goal_check": "-"}
            out.append(f"| {c['id']} | {r['rep']} | {v['evidence']} verified={v['verified']} | {v['goal_check']} "
                       f"| {r['grade']}{' (' + r['grade_note'] + ')' if r['grade_note'] else ''} | {r['outcome']} "
                       f"| {' '.join(r['tools'])} | {r['wall_s']} | {r['transcript']} |")
    return "\n".join(out) + "\n"


def missing_key() -> str:
    """'' when the real model's key resolves (the repo .env loaded as the child loads
    it), else the instruction the product gives for it."""
    from vector_os_nano.vcli.config import resolve_credentials

    creds = resolve_credentials()
    return "" if creds.api_key else creds.instruction()


def run(cells: list[dict[str, Any]], out: Path, n: int | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    runnable = [c for c in cells if "not_yet" not in c]
    skipped = missing_key() if runnable else ""
    reps = [] if skipped else [judge(c, run_rep(c, k, out)) for c in runnable for k in range(n or c["n"])]
    summary = summarize(cells, reps, skipped=skipped)
    (out / "live_matrix.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "live_matrix.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, help="repetitions per cell (default: each cell's n)")
    ap.add_argument("--cells", nargs="*", help="cell ids (default: all)")
    ap.add_argument("--out", type=Path, help="output directory")
    a = ap.parse_args(argv)
    cells = load_cells()
    if a.cells:
        unknown = set(a.cells) - {c["id"] for c in cells}
        if unknown:
            ap.error(f"unknown cells {sorted(unknown)}; one of {[c['id'] for c in cells]}")
        cells = [c for c in cells if c["id"] in a.cells]
    out = a.out or paths.home() / "runs" / f"live-matrix-{time.strftime('%Y%m%d-%H%M%S')}"
    summary = run(cells, out, a.n)
    print(markdown(summary))
    print(f"[live_matrix] {out}/live_matrix.json")
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
