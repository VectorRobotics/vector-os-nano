# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""gt_audit — join a session's verdicts, its ground-truth dump and the independent grade.

The ONE place the outcome rules live (acceptance A3 and A10 of docs/plan-release-1.0.md),
for both of its users:

- the e2e audit: every bare-REPL session of ``tests/e2e_repl`` writes a ``VECTOR_GT_DUMP``
  (tests/harness/pty_cli). After a session that started a simulator, each typed turn is
  looked up in the expectation table ``tests/e2e_repl/gt_expect.yaml`` and graded by
  ``scripts/gt_grade.py``, run as its own isolated process (it shares no code with the
  product). Red, failing the flow at once: FALSE GROUNDED (the product claimed the turn,
  the grade denies it), UNAUDITED (a claimed turn with no row), NO_GT (a claimed turn
  with an expectation and no ground truth for it). FALSE REJECT (the grade says it
  happened, the product did not claim it) is reported.
- the live matrix (tests/harness/live_matrix.py): the same grade and outcome for its task turn.

A turn is claimed when it verified or its task contract was proven: a count of tasks done
reads ``goal_check == proven``, never ``verified`` alone (charter 3.5).

    python -m tests.harness.gt_audit <audit dir>   # the run's report (scripts/gate.sh e2e)
"""
from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from tests.harness.pty_cli import PROMPT
from vector_os_nano.schemas import paths

GT_GRADE = Path(__file__).resolve().parents[2] / "scripts" / "gt_grade.py"
_PY = sys.executable  # isolated (-I): the grade imports nothing but the stdlib
TABLE = Path(__file__).resolve().parents[1] / "e2e_repl" / "gt_expect.yaml"
AUDIT_DIR_ENV = "VECTOR_GT_AUDIT_DIR"  # scripts/gate.sh e2e: one folder for the run
GRADES = ("base_in", "base_at", "base_turned", "held_lifted", "delivered", "object_on")
RED = ("FALSE_GROUNDED", "UNAUDITED", "NO_GT")
READY = re.compile(r"Started\s+(\w+)\s+simulation,\s+ready\s+in")
_VERDICT = re.compile(r"verdict (\w+) verified=(True|False) \((\d+)/(\d+) grounded\).*? goal=(\w+)")


# --- one expectation ---------------------------------------------------------

def expectation(row: dict[str, Any]) -> tuple[str, Any]:
    """The one grade of a row: (key, arg). ``from_spawn`` only says where a base_at
    point comes from (tests/unit/test_gt_audit.py checks it against the manifests)."""
    keys = [k for k in row if k != "from_spawn"]
    if len(keys) != 1:
        raise ValueError(f"an expectation holds exactly one grade, got {row}")
    key = keys[0]
    if "from_spawn" in row and key != "base_at":
        raise ValueError(f"from_spawn goes with base_at only, got {row}")
    return key, row[key]


def grade_args(row: dict[str, Any]) -> list[str]:
    """scripts/gt_grade.py flags for one expectation (fail loud on anything else)."""
    key, arg = expectation(row)
    if key in ("base_in", "base_at"):
        return ["--box" if key == "base_in" else "--at", *(str(float(v)) for v in arg)]
    if key == "base_turned":
        angle = float(arg)
        if not 5 < abs(angle) <= 180:
            raise ValueError("base_turned needs 5 < abs(degrees) <= 180")
        return ["--turn-deg", str(angle)]
    if key in ("held_lifted", "delivered"):
        return ["--object", str(arg)]
    if key == "object_on":
        return ["--object", str(arg[0]), "--surface", str(arg[1])]
    raise ValueError(f"unknown expect {key!r}; one of {', '.join(GRADES)}")


def grade(dump: Path, row: dict[str, Any], goal: str, nth: int = 1) -> tuple[bool | None, str]:
    """(the independent grade of the *nth* turn of *goal* in *dump*, '' or why there is none)."""
    if not dump.is_file() or not dump.stat().st_size:
        return None, "no GT dump (no sim turn was recorded)"
    r = subprocess.run([_PY, "-I", str(GT_GRADE), "--dump", str(dump), *grade_args(row),
                        "--goal", goal, "--nth", str(nth)],
                       capture_output=True, text=True, timeout=60, check=False)
    if r.returncode:
        return None, f"gt_grade failed: {r.stderr.strip()[-300:]}"
    rows = [json.loads(ln) for ln in r.stdout.splitlines()]
    if not rows:
        return None, f"no finished turn {nth} of {goal!r} in the GT dump"
    return bool(rows[0][expectation(row)[0]]), ""


# --- reading a transcript -----------------------------------------------------

def split_turns(text: str, sentences: list[str]) -> list[tuple[str, str]]:
    """(sentence, its turn's transcript) for each typed sentence in order, from its prompt
    line to the next one's, up to the first sentence the REPL never got."""
    marks: list[tuple[str, int]] = []
    at = 0
    for s in sentences:
        i = text.find(f"{PROMPT} {s}", at)
        if i == -1:
            break
        marks.append((s, i))
        at = i + 1
    ends = [i for _s, i in marks[1:]] + [len(text)]
    return [(s, text[i:end]) for (s, i), end in zip(marks, ends)]


def parse_verdict(turn: str) -> dict[str, Any] | None:
    """The verdict line the REPL printed in *turn* (the last one), or None."""
    hits = list(_VERDICT.finditer(turn))
    if not hits:
        return None
    m = hits[-1]
    return {"evidence": m[1], "verified": m[2] == "True", "grounded": f"{m[3]}/{m[4]}",
            "goal_check": m[5], "line": m[0]}


def json_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """The same fields from a ``VECTOR_VERDICT {json}`` report (``vector-cli -p --json``)."""
    v = {"evidence": report.get("evidence", ""), "verified": bool(report.get("verified")),
         "grounded": f"{report.get('n_grounded', 0)}/{report.get('n_steps', 0)}",
         "goal_check": report.get("goal_check", "unchecked")}
    return {**v, "line": "VECTOR_VERDICT " + " ".join(f"{k}={x}" for k, x in v.items())}


# --- the outcome rules ----------------------------------------------------------

def claimed(verdict: dict[str, Any] | None) -> bool:
    """The product says the task happened: the turn verified, or its contract was proven."""
    return bool(verdict) and (verdict["verified"] or verdict["goal_check"] == "proven")


def outcome(verdict: dict[str, Any] | None, graded: bool | None) -> str:
    if graded is None:
        return "ungraded"
    if claimed(verdict) != graded:
        return "FALSE_REJECT" if graded else "FALSE_GROUNDED"
    return "agree"


# --- the e2e audit ------------------------------------------------------------------

def current_flow() -> str:
    """The e2e flow of the running test (tests/e2e_repl/test_<flow>.py), else ''."""
    path = os.environ.get("PYTEST_CURRENT_TEST", "").split("::")[0]
    return Path(path).stem.removeprefix("test_") if path.startswith("tests/e2e_repl/") else ""


def out_dir(where: str | Path | None = None) -> Path:
    d = Path(where or os.environ.get(AUDIT_DIR_ENV) or paths.home() / "runs" / "gt_audit")
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_dump(where: str | Path | None = None) -> Path:
    """A fresh, empty file for one session's ``VECTOR_GT_DUMP``."""
    fd, path = tempfile.mkstemp(prefix=f"{current_flow() or 'session'}-", suffix=".gt.jsonl", dir=out_dir(where))
    os.close(fd)
    return Path(path)


@functools.cache
def load_table(path: Path = TABLE) -> dict[str, Any]:
    """flow -> {body, turns: {sentence: expectation | {none: reason}}}, each row checked."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, this audit reads 1")
    for flow, f in data["flows"].items():
        for sentence, row in f["turns"].items():
            if "none" in row:
                if len(row) != 1 or not str(row["none"]).strip():
                    raise ValueError(f"{flow} / {sentence!r}: none takes a reason and nothing else")
            else:
                grade_args(row)
    return data["flows"]


def audit(flow: str, turns: list[tuple[str, dict[str, Any] | None]], dump: Path,
          table: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every typed turn of one simulator session of *flow*, (sentence, verdict) in order,
    graded against its row: one record each, with its outcome."""
    rows = (load_table() if table is None else table).get(flow, {}).get("turns", {})
    seen: Counter[str] = Counter()
    out = []
    for sentence, verdict in turns:
        seen[sentence] += 1
        row = rows.get(sentence)
        rec: dict[str, Any] = {"flow": flow, "sentence": sentence, "nth": seen[sentence],
                               "verdict": verdict["line"] if verdict else None, "claimed": claimed(verdict),
                               "expect": row, "grade": None, "note": "", "dump": str(dump)}
        if row is None:
            rec["outcome"] = "UNAUDITED" if rec["claimed"] else "not_gradable"
            rec["note"] = f"add a row or a reason to {TABLE.name}" if rec["claimed"] else "no row (not claimed)"
        elif "none" in row:
            rec["outcome"], rec["note"] = "not_gradable", row["none"]
        else:
            rec["grade"], rec["note"] = grade(dump, row, sentence, seen[sentence])
            unproven = rec["grade"] is None and rec["claimed"]
            rec["outcome"] = "NO_GT" if unproven else outcome(verdict, rec["grade"])
        out.append(rec)
    return out


def why(rec: dict[str, Any]) -> str:
    return (f"{rec['outcome']} {rec['flow']} {rec['sentence']!r} (#{rec['nth']}): verdict {rec['verdict']!r}; "
            f"expect {rec['expect']}; grade {rec['grade']}{' (' + rec['note'] + ')' if rec['note'] else ''}; "
            f"dump {rec['dump']}")


def check(flow: str, turns: list[tuple[str, dict[str, Any] | None]], text: str, dump: Path) -> list[dict[str, Any]]:
    """The e2e hook: audit one session of *flow* (nothing when it started no simulator),
    append its records to the run's audit file, and fail the flow on a red outcome."""
    if not READY.search(text):
        return []
    dump.with_name(dump.name.replace(".gt.jsonl", ".transcript.txt")).write_text(text, encoding="utf-8")
    recs = audit(flow, turns, dump)
    with open(out_dir() / "turns.jsonl", "a", encoding="utf-8") as fh:  # the run's, wherever the dump is
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    red = [r for r in recs if r["outcome"] in RED]
    assert not red, "the independent GT audit denies this flow:\n" + "\n".join(map(why, red))
    return recs


# --- the run's report -----------------------------------------------------------------

def report(audit_dir: Path) -> dict[str, Any]:
    """Every audited turn of the run, by outcome; written to ``<audit_dir>/report.json``."""
    f = audit_dir / "turns.jsonl"
    recs = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []
    by = {o: [r for r in recs if r["outcome"] == o]
          for o in ("agree", *RED, "FALSE_REJECT", "ungraded", "not_gradable")}
    rep = {"schema": 1, "turns": len(recs), "graded": sum(r["grade"] is not None for r in recs),
           **{o.lower(): v for o, v in by.items()}}
    (audit_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def line(rep: dict[str, Any]) -> str:
    return (f"gt_audit: {rep['turns']} turns audited, {rep['graded']} graded ({len(rep['agree'])} agree), "
            f"FALSE GROUNDED {len(rep['false_grounded'])}, FALSE REJECT {len(rep['false_reject'])}, "
            f"unaudited {len(rep['unaudited'])}, no ground truth {len(rep['no_gt'])}, "
            f"not gradable {len(rep['not_gradable']) + len(rep['ungraded'])}")


def main(argv: list[str] | None = None) -> int:
    audit_dir = Path((argv or sys.argv[1:])[0])
    rep = report(audit_dir)
    print(f"[gate] {line(rep)} ({audit_dir}/report.json)")
    for r in (*rep["false_grounded"], *rep["unaudited"], *rep["no_gt"], *rep["false_reject"]):
        print(f"[gate]   {why(r)}")
    return 1 if any(rep[o.lower()] for o in RED) else 0


if __name__ == "__main__":
    sys.exit(main())
