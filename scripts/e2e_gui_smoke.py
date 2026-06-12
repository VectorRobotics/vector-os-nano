# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""One-command E2E GUI smoke for the habitat world (campaign #4 batch 3).

Drives the REAL vector-cli in a tmux session against the live apartment
scene, sends the standard NL instruction set, screenshots the habitat
viewer after every step, and grades each step DETERMINISTICALLY from the
CLI transcript (verified PASS / honest failure / silent-wrong). Screenshots
are evidence for human review — no pixel grading.

Usage:
    .venv/bin/python scripts/e2e_gui_smoke.py [--keep] [--session NAME]

Outputs artifacts/smoke-<ts>/{step_N.png, transcript.log, summary.json};
exit 0 = every step ok or honest, 1 = at least one silent-wrong/timeout,
2 = environment preflight failed.

Grading contract (mirrors the kernel's honesty mechanisms):
- "ok"     — the step did what was asked AND the verify evidence agrees.
- "honest" — the step failed LOUD with a diagnosable cause (bad_params,
             moved_short, lateral_unsupported...). Acceptable in smoke:
             LLM binding varies; silent wrongness is the only red.
- "fail"   — neither pattern seen (false-PASS family / timeout / crash).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- the instruction set (batch-2 GUI closure + campaign #3 regressions) ---
STEPS: "list[dict]" = [
    {
        "name": "navigate_kitchen",
        "command": "走到厨房",
        "ok": [r"Outcome: \[PASS\]"],
        "honest": [r"requires a label", r"bad_params", r"no_path"],
        "timeout": 240,
    },
    {
        "name": "kitchen_again_already_there",
        "command": "走到厨房",
        "ok": [r"already satisfied pre-exec", r"already_there"],
        "honest": [r"requires a label", r"bad_params"],
        "timeout": 240,
    },
    {
        "name": "walk_20m",
        "command": "往前走20米",
        "ok": [r"Outcome: \[PASS\]"],
        "honest": [r"moved [\d.]+m of the requested", r"bad_params",
                   r"blocked or unsupported"],
        "timeout": 240,
    },
    {
        "name": "walk_left_nonholonomic",
        "command": "往左走一米",
        # the EXPECTED outcome on a non-holonomic base IS the loud refusal
        "ok": [r"lateral", r"moved [\d.]+m of the requested",
               r"blocked or unsupported"],
        "honest": [r"bad_params", r"requires a", r"Outcome: \[FAIL\]"],
        "timeout": 240,
    },
]

_OUTCOME_RE = re.compile(r"Outcome: \[(PASS|FAIL)\]")


# ---------------------------------------------------------------------------
# Pure grading helpers (unit-tested in tests/unit/vcli/test_smoke_grading.py)
# ---------------------------------------------------------------------------

def count_outcomes(transcript: str) -> int:
    """Number of completed VGG turns in the transcript so far."""
    return len(_OUTCOME_RE.findall(transcript))


def new_segment(prev: str, cur: str) -> str:
    """The transcript portion appended since *prev* (resync-safe)."""
    if cur.startswith(prev):
        return cur[len(prev):]
    return cur  # pane scrolled past — grade the full visible text


def grade_segment(segment: str, ok: "list[str]", honest: "list[str]") -> str:
    """'ok' | 'honest' | 'fail' — ok patterns win over honest ones."""
    for pat in ok:
        if re.search(pat, segment):
            return "ok"
    for pat in honest:
        if re.search(pat, segment):
            return "honest"
    return "fail"


# ---------------------------------------------------------------------------
# tmux / X11 plumbing
# ---------------------------------------------------------------------------

def _sh(cmd: "list[str]", **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout


def _pane(session: str) -> str:
    return _sh(["tmux", "capture-pane", "-t", session, "-p", "-S", "-2000"])


def _habitat_window_id() -> str:
    tree = _sh(["xwininfo", "-root", "-tree"])
    ids = re.findall(r"(0x[0-9a-f]+) \"Vector Habitat", tree)
    return ids[-1] if ids else ""


def _screenshot(win: str, dest: Path) -> bool:
    if not win:
        return False
    return subprocess.run(["import", "-window", win, str(dest)],
                          capture_output=True).returncode == 0


def _openrouter_key() -> str:
    env_file = REPO / ".env"
    if env_file.exists():
        m = re.search(r"^OPENROUTER_API_KEY=[\"']?([^\"'\n]+)",
                      env_file.read_text(), re.M)
        if m:
            return m.group(1)
    return os.environ.get("OPENROUTER_API_KEY", "")


def preflight(session: str) -> "str | None":
    """Return an error string, or None when the environment is usable."""
    if not os.environ.get("DISPLAY"):
        return "no DISPLAY — GUI smoke needs the desktop session"
    for tool in ("tmux", "xwininfo", "import"):
        if subprocess.run(["which", tool], capture_output=True).returncode:
            return f"missing tool: {tool}"
    if not _openrouter_key():
        return "no OPENROUTER_API_KEY in repo .env or environment"
    if _sh(["tmux", "ls"]).find(session + ":") >= 0:
        return f"tmux session {session!r} already exists (use --session)"
    return None


def boot_cli(session: str, log_path: Path) -> None:
    env_assigns = (
        "DEEPSEEK_API_KEY=\"$VECTOR_SMOKE_KEY\" "
        "DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1 "
        "DEEPSEEK_MODEL=deepseek/deepseek-chat "
    )
    inner = (
        f"cd {REPO} && {env_assigns}.venv/bin/python -m "
        f"vector_os_nano.vcli.cli --scenario apartment 2>&1 | tee {log_path}"
    )
    # the key reaches the inner shell via the environment, never argv
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50",
         inner],
        env={**os.environ, "VECTOR_SMOKE_KEY": _openrouter_key()},
        check=True,
    )


def wait_for(predicate, timeout: float, interval: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(session: str, keep: bool) -> int:
    err = preflight(session)
    if err:
        print(f"PREFLIGHT FAIL: {err}")
        return 2

    art = REPO / "artifacts" / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    art.mkdir(parents=True, exist_ok=True)
    log_path = art / "transcript.log"
    results: "list[dict]" = []
    rc = 0

    boot_cli(session, log_path)
    try:
        if not wait_for(lambda: "vector>" in _pane(session), 120):
            print("FAIL: CLI prompt never appeared")
            return 1
        if not wait_for(_habitat_window_id, 60):
            print("FAIL: habitat viewer window never appeared")
            return 1
        win = _habitat_window_id()

        for i, step in enumerate(STEPS, 1):
            before = _pane(session)
            base_outcomes = count_outcomes(before)
            subprocess.run(["tmux", "send-keys", "-t", session,
                            step["command"], "Enter"], check=True)
            done = wait_for(
                lambda: count_outcomes(_pane(session)) > base_outcomes,
                step["timeout"],
            )
            after = _pane(session)
            shot = art / f"step{i}_{step['name']}.png"
            _screenshot(win, shot)
            segment = new_segment(before, after)
            verdict = ("fail" if not done
                       else grade_segment(segment, step["ok"], step["honest"]))
            results.append({
                "step": step["name"], "command": step["command"],
                "verdict": verdict, "screenshot": shot.name,
                "timed_out": not done,
            })
            print(f"[{i}/{len(STEPS)}] {step['name']}: {verdict.upper()}")
            if verdict == "fail":
                rc = 1

        subprocess.run(["tmux", "send-keys", "-t", session, "quit", "Enter"])
        wait_for(lambda: _sh(["tmux", "ls"]).find(session + ":") < 0, 20)
    finally:
        if not keep:
            subprocess.run(["tmux", "kill-session", "-t", session],
                           capture_output=True)
        leaked = [ln for ln in _sh(["pgrep", "-af", "server.py"]).splitlines()
                  if "habitat" in ln and "conda" in ln]
        if leaked:
            print(f"WARNING: leaked habitat server(s): {leaked}")
            rc = rc or 1

    (art / "summary.json").write_text(json.dumps(
        {"results": results, "exit_code": rc}, indent=2, ensure_ascii=False))
    print(f"\nartifacts: {art}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="vnano-smoke")
    ap.add_argument("--keep", action="store_true",
                    help="leave the tmux session running on exit")
    args = ap.parse_args()
    return run(args.session, args.keep)


if __name__ == "__main__":
    sys.exit(main())
