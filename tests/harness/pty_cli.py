# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""pty_cli — drive the REAL vector-cli through a stdlib PTY and read its verdict.

R2a PART C: this is the project's #1-failure fix made concrete. Instead of a
``~/sandbox`` script that pokes the engine directly (bypassing the product), this
harness spawns the ACTUAL entrypoint::

    python -m vector_os_nano.vcli.cli -p <prompt> --json

under a stdlib ``pty`` (so the child sees a TTY, exactly like real use), waits for
the fixed ``VECTOR_VERDICT {<json>}`` sentinel line on the child's stdout, parses
the JSON into a ``VerdictReport``-shaped dict, captures the child's exit code, and
asserts the machine invariant: exit 0 iff the turn verified or was a chat turn (no
action, no verify: a ``NO_TRACE`` verdict without an error).

Deterministic, network-free runs use the ``VECTOR_FAKE_LLM_TOOLS`` seam: pass a
``tool_script`` dict (written to a temp file the child reads) that a
``FakeToolScriptBackend`` replays as the model's native tool_use turns, so every
tool, GoalVerifier, evidence gate and verdict of the product runs with no live LLM.

STDLIB ONLY — uses ``pty`` + ``subprocess`` + ``os``; deliberately NO ``pexpect``
(a new dependency would be a CEO gate this build avoids).
"""
from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vector_os_nano.schemas import paths

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The fixed sentinel the child prints on stdout (see vcli.verdict.VERDICT_SENTINEL).
_SENTINEL = "VECTOR_VERDICT"

# Generous default — a sync dev-world turn (file_write + verify) is sub-second,
# but allow headroom for cold imports.
_DEFAULT_TIMEOUT_SEC = 90.0

# What a PTY child leaves of its test's pytest-timeout budget: the process-group
# kill, the fixture's own work after the session, and the test body.
_TEARDOWN_SEC = 20.0


def _session_deadline(timeout_sec: float) -> float:
    """The one deadline of a PTY child (run_cli_turn, run_repl_session): *timeout_sec*
    from now, cut to end inside the pytest-timeout budget of the test running it.

    scripts/gate.sh exports PYTEST_TIMEOUT and pytest-timeout reads it: a stalled
    child then returns its transcript instead of pytest-timeout killing the test with
    nothing to read (S3aV-08: an E5 hang left no transcript at 300 s of 420). The cut
    leaves _TEARDOWN_SEC, but never takes the child below a few seconds or past half
    the budget (a budget of 20 s or less once left it zero or negative).
    """
    budget = float(os.environ.get("PYTEST_TIMEOUT") or 0)
    if budget:
        timeout_sec = min(timeout_sec, max(budget - _TEARDOWN_SEC, min(3.0, budget / 2)))
    return _now() + timeout_sec


@dataclass(frozen=True)
class CliTurnResult:
    """The outcome of one real cli.main turn driven through the PTY."""

    verdict: dict[str, Any]   # parsed VECTOR_VERDICT payload
    exit_code: int
    raw_output: str           # full child output (for debugging)

    @property
    def verified(self) -> bool:
        return bool(self.verdict.get("verified", False))

    @property
    def evidence(self) -> str:
        return str(self.verdict.get("evidence", ""))

    @property
    def goal(self) -> str:
        return str(self.verdict.get("goal", ""))


def _write_tool_script(tool_script: dict[str, Any]) -> str:
    """Write a canned NATIVE tool-use script to a temp file; return its path.

    The VECTOR_FAKE_LLM_TOOLS seam — a ``{"turns": [...]}`` (open-loop) or
    ``{"steps": [...]}`` (reactive) JSON object the child's ``FakeToolScriptBackend``
    replays as the model's native tool_use turns.
    """
    fd, path = tempfile.mkstemp(prefix="vector_tool_script_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(tool_script, fh, ensure_ascii=False)
    return path


def _now() -> float:
    return time.monotonic()


def _child_env(*, live: bool, home_prefix: str) -> tuple[dict[str, str], str]:
    """The env every PTY child gets: repo on PYTHONPATH, its own temp HOME, and —
    unless *live* — no route to a real LLM/VLM or the external nav stack
    (placeholder Anthropic key; blank DeepSeek/OpenRouter keys and VECTOR_NAV_STACK,
    so the child's ``load_dotenv`` cannot fill them from the repo-root ``.env``).
    """
    env = dict(os.environ)
    # The child must import `tests` (the fake backends live under tests/harness).
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if not live:
        env.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
        env["DEEPSEEK_API_KEY"] = ""
        env["OPENROUTER_API_KEY"] = ""
        # No external CMU nav stack either: every --sim-go2 child would start it
        # (and rviz2) in its own sessions, outliving the child on a shared box.
        env["VECTOR_NAV_STACK"] = ""
    # Per-run HOME and VECTOR_HOME: a session / history / memory from a prior run must
    # never leak into this one, and the developer's real state directory stays untouched.
    # Model weights are a read-only shared cache, not run state: keep finding them
    # (perception loads its pinned weights from the HF cache).
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    home = tempfile.mkdtemp(prefix=home_prefix)
    env["HOME"] = home
    env["VECTOR_HOME"] = str(Path(home) / ".vector")
    (Path(home) / ".vector").mkdir()
    # So are the robot assets scripts/fetch_assets.py provisioned (linked, never written).
    provisioned = paths.home() / "assets"
    if provisioned.is_dir():
        (Path(home) / ".vector" / "assets").symlink_to(provisioned)
    return env, home


def _spawn(argv: list[str], env: dict[str, str], cwd: str | Path) -> tuple[subprocess.Popen, int]:
    """Start *argv* on a fresh PTY as the leader of its OWN session + process group.

    The PTY is the child's controlling terminal (so a written ``\\x03`` is a real
    Ctrl-C), and :func:`_kill_group` can take down the child and everything it
    spawned without touching any other process on the machine.
    """
    master_fd, slave_fd = pty.openpty()
    # A wide, fixed terminal: Rich never wraps a verdict line mid-token.
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 220, 0, 0))
    try:
        proc = subprocess.Popen(
            argv, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=str(cwd), env=env, close_fds=True, start_new_session=True,
            preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
        )
    finally:
        os.close(slave_fd)  # parent keeps only the master end
    return proc, master_fd


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group (it leads its own), then reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass  # group already gone
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


def _read_chunk(master_fd: int, wait_sec: float) -> bytes | None:
    """One read from the PTY master: bytes, b'' on timeout, None on EOF."""
    r, _, _ = select.select([master_fd], [], [], max(0.0, wait_sec))
    if master_fd not in r:
        return b""
    try:
        data = os.read(master_fd, 65536)
    except OSError:
        return None  # slave closed (EIO on Linux)
    return data or None


def _drain(master_fd: int, deadline: float, proc: subprocess.Popen) -> tuple[str, bool]:
    """Read the PTY until EOF, *deadline*, or the child has exited and gone quiet.

    A grandchild that inherited the PTY (a ROS node the sim started) can hold it
    open after the child exits, so EOF alone may never come; the caller then kills
    the group. Returns (output, reached_eof).
    """
    chunks: list[bytes] = []
    quiet_since = _now()
    while _now() < deadline:
        data = _read_chunk(master_fd, min(0.5, deadline - _now()))
        if data is None:
            return b"".join(chunks).decode("utf-8", errors="replace"), True
        if data:
            chunks.append(data)
            quiet_since = _now()
        elif proc.poll() is not None and _now() - quiet_since > 1.0:
            break
    return b"".join(chunks).decode("utf-8", errors="replace"), False


def _find_verdict_line(output: str) -> dict[str, Any] | None:
    """Scan child output for the fixed VECTOR_VERDICT sentinel line, parse its JSON.

    The sentinel is unique, so even if Rich/banner noise leaks onto the same PTY
    (stdout+stderr share the line discipline) the verdict is unambiguous. Returns
    the parsed payload, or None if no sentinel line is present.
    """
    for raw in output.splitlines():
        line = raw.strip("\r\n").lstrip("\r")
        idx = line.find(_SENTINEL + " ")
        if idx == -1:
            continue
        payload = line[idx + len(_SENTINEL) + 1 :].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


def run_cli_turn(
    prompt: str,
    *,
    tool_script: dict[str, Any] | None = None,
    sim: bool = False,
    sim_go2: bool = False,
    live: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    cwd: str | Path | None = None,
) -> CliTurnResult:
    """Run ONE real cli.main turn through a PTY and return its parsed verdict.

    Args:
        prompt:     the user turn text (passed as ``-p <prompt>``).
        tool_script: the model's native tool_use turns, replayed through the
                    ``VECTOR_FAKE_LLM_TOOLS`` seam (deterministic, no live LLM).
        sim/sim_go2: pass ``--sim`` / ``--sim-go2`` (heavy — caller serializes).
        live:       run against the REAL LLM provider (no fake seam). When True the
                    harness does NOT inject the placeholder ``ANTHROPIC_API_KEY`` —
                    the child inherits the real env and ``resolve_credentials``
                    loads the repo-root ``.env`` from ``cwd`` (the live provider key
                    reaches the child). Scripted callers leave this False so a fake
                    key keeps deterministic runs network-free.
        extra_env / extra_args: escape hatches for additional env / CLI flags.
        timeout_sec: max wait for the child to finish, cut to end inside
                    ``PYTEST_TIMEOUT`` when that is set (:func:`_session_deadline`).
        cwd:        working dir for the child (defaults to repo root).

    Asserts the machine invariant (exit 0 iff verified or a chat turn) before returning.
    Raises AssertionError if no VECTOR_VERDICT line was emitted.
    """
    argv = [
        sys.executable, "-m", "vector_os_nano.vcli.cli",
        "-p", prompt, "--json", "--no-permission",
    ]
    if sim:
        argv.append("--sim")
    if sim_go2:
        argv.append("--sim-go2")
    if extra_args:
        argv += list(extra_args)

    env, home_dir = _child_env(live=live, home_prefix="vector_pty_home_")
    script_path = _write_tool_script(tool_script) if tool_script is not None else None
    if extra_env:
        env.update(extra_env)
    if script_path is not None:
        # The real written path wins over any placeholder in extra_env.
        env["VECTOR_FAKE_LLM_TOOLS"] = script_path

    # ONE deadline for the whole turn: drain to EOF, then give the child whatever is
    # left to exit; past the deadline the whole process group is killed.
    deadline = _session_deadline(timeout_sec)
    proc, master_fd = _spawn(argv, env, cwd or _REPO_ROOT)
    try:
        output, _eof = _drain(master_fd, deadline, proc)
        try:
            proc.wait(timeout=max(0.1, deadline - _now()))
        except subprocess.TimeoutExpired:
            pass
        exit_code = proc.returncode if proc.returncode is not None else 1
    finally:
        _kill_group(proc)  # also reaps anything the child left in its group
        os.close(master_fd)
        if script_path is not None:
            try:
                os.unlink(script_path)
            except OSError:
                pass
        shutil.rmtree(home_dir, ignore_errors=True)

    verdict = _find_verdict_line(output)
    assert verdict is not None, (
        "no VECTOR_VERDICT line emitted by cli.main "
        f"(exit={exit_code}{', deadline reached' if _now() >= deadline else ''}). Raw output:\n{output}"
    )
    # The machine invariant the whole instrument rests on: exit 0 iff the turn
    # verified, or it was a chat turn (no action, no claim: NO_TRACE, no error).
    chat = verdict.get("evidence") == "NO_TRACE" and not verdict.get("error") and not verdict.get("acted")
    assert (bool(verdict.get("verified", False)) or chat) == (exit_code == 0), (
        f"verified={verdict.get('verified')} chat={chat} but exit_code={exit_code} "
        f"(must agree). verdict={verdict}"
    )
    return CliTurnResult(verdict=verdict, exit_code=exit_code, raw_output=output)


# ---------------------------------------------------------------------------
# REPL session driver — drive the INTERACTIVE REPL (bare vector-cli) by NL.
# ---------------------------------------------------------------------------
#
# The owner's ONLY acceptance interface is bare ``vector-cli`` + natural language.
# The REPL renders a HUMAN verdict line ("verdict GROUNDED verified=True (1/1
# grounded)"), not the JSON sentinel, so this driver returns the transcript and the
# caller asserts on the rendered text. It is event-driven, not wall-clock scripted:
# each line is typed when the REPL is back at its prompt, permission prompts are
# answered as they appear, and after ``quit`` the transcript is read to EOF.

PROMPT = "vector>"
PERMISSION_PROMPT = "Allow?"  # cli.ask_permission -> rich Prompt "Allow? [y/n/a] (y): "
_QUIET_SEC = 0.75  # no output for this long = the child is waiting for input
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]|\x1b[=>78]|\r")


def strip_ansi(text: str) -> str:
    """Drop terminal escapes and CRs so styled spans collapse to visible text."""
    return _ANSI.sub("", text)


@dataclass(frozen=True)
class ReplSessionResult:
    """The transcript + exit code of one interactive REPL session driven by NL."""

    transcript: str           # raw PTY bytes, decoded
    exit_code: int
    timed_out: bool = False   # a step hit its deadline; the group was killed

    @property
    def text(self) -> str:
        return strip_ansi(self.transcript)


_PROGRESS_KEY = re.compile(r"(\S+) (\d+)s")


def _has_run(text: str, key: str) -> bool:
    """Has the action of *key* ('navigate 3s') been running for at least that long in
    *text*: any of its progress frames ('navigate 4s · 1.2 m to go') at n >= N s?"""
    m = _PROGRESS_KEY.fullmatch(key)
    if m is None:
        raise ValueError(f"a progress key is '<action> <N>s' (e.g. 'navigate 3s'), got {key!r}")
    frames = re.findall(rf"(?<!\S){re.escape(m[1])} (\d+)s\b", text)
    return any(int(n) >= int(m[2]) for n in frames)


def _is_echo(after_prompt: str, typed: str) -> bool:
    """Is this prompt line just the REPL redrawing what we typed (not a new prompt)?

    Only the prompt's own line counts: an async turn may print the typed goal on
    later lines after a genuinely fresh prompt.
    """
    line = after_prompt.split("\n", 1)[0].strip()
    return bool(typed and line) and (line.startswith(typed) or typed.startswith(line))


class _Pty:
    """The parent side of one REPL child: accumulate output, type into it."""

    def __init__(self, master_fd: int) -> None:
        self.fd = master_fd
        self.buf = bytearray()
        self.eof = False
        self.last_data = _now()
        self.mark = 0  # visible-text offset of the last thing we typed

    def pump(self, wait_sec: float) -> None:
        data = _read_chunk(self.fd, wait_sec)
        if data is None:
            self.eof = True
        elif data:
            self.buf += data
            self.last_data = _now()

    def text(self) -> str:
        return strip_ansi(self.buf.decode("utf-8", errors="replace"))

    def send(self, text: str) -> None:
        self.mark = len(self.text())
        raw = text if len(text) == 1 and ord(text) < 32 else text + "\n"
        os.write(self.fd, raw.encode("utf-8"))

    def wait_input(self, deadline: float, typed: str) -> str:
        """Pump until the REPL wants input: 'prompt' | 'permission' | 'eof' | 'timeout'.

        Only output after the last send counts, the PTY echo of *typed* is not a
        fresh prompt, and the child must have gone quiet (a running turn animates).
        """
        while True:
            if self.eof:
                return "eof"
            if _now() >= deadline:
                return "timeout"
            self.pump(0.1)
            if _now() - self.last_data < _QUIET_SEC:
                continue
            new = self.text()[self.mark:]
            perm, prompt = new.rfind(PERMISSION_PROMPT), new.rfind(PROMPT)
            if perm != -1 and perm > prompt:
                return "permission"
            if prompt != -1 and not _is_echo(new[prompt + len(PROMPT):], typed):
                return "prompt"


def run_repl_session(
    lines: list[str | tuple[float | str, str]],
    *,
    tool_script: dict[str, Any] | None = None,
    answers: list[str] | None = None,
    live: bool = False,
    sim: bool = False,
    sim_go2: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    timeout_sec: float = 240.0,
    cwd: str | Path | None = None,
) -> ReplSessionResult:
    """Drive the REAL interactive ``vector-cli`` REPL under a PTY; return the transcript.

    The child is bare ``python -m vector_os_nano.vcli.cli`` — no flags unless the
    caller asks (``sim``/``sim_go2`` add the launch flag + ``--headless``;
    ``extra_args`` is the escape hatch). The LLM is the only thing replaced, via the
    ``VECTOR_FAKE_LLM_TOOLS`` env seam (an open-loop ``{"turns"}`` or a reactive
    ``{"steps"}`` script); ``live=True`` uses the real provider from ``.env``.

    Args:
        lines:   a ``str`` is typed once the REPL is idle at its prompt; a
                 ``(delay_sec, text)`` tuple is typed ``delay_sec`` after the previous
                 line WITHOUT waiting (mid-turn input; ``"\x03"`` is a Ctrl-C);
                 a ``("<action> <N>s", text)`` tuple is typed once that action has
                 been running for at least N seconds: any of its progress frames
                 ``<action> <n>s`` with n >= N after the previous line (keyed on
                 observed progress, not a clock; one skipped frame never stalls it).
                 ``quit`` is appended unless the last line already exits.
        answers: replies to permission prompts, in order; a prompt with no answer
                 left is refused with ``n`` (the safe default a user would pick).
        timeout_sec: ONE deadline for the whole session (boot, every turn, the
                 exit), cut to end inside ``PYTEST_TIMEOUT`` when that is set
                 (:func:`_session_deadline`). On expiry the process group is killed
                 and ``timed_out`` is set.
    """
    argv = [sys.executable, "-m", "vector_os_nano.vcli.cli"]
    if sim:
        argv.append("--sim")
    if sim_go2:
        argv.append("--sim-go2")
    if (sim or sim_go2) and "--headless" not in (extra_args or []):
        argv.append("--headless")
    argv += list(extra_args or [])

    env, home_dir = _child_env(live=live, home_prefix="vector_pty_repl_home_")
    env.setdefault("TERM", "xterm-256color")
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"  # no cursor-position round-trips on a PTY
    script_path = _write_tool_script(tool_script) if tool_script is not None else None
    if script_path is not None:
        env["VECTOR_FAKE_LLM_TOOLS"] = script_path
    if extra_env:
        env.update(extra_env)

    steps: list[str | tuple[float, str]] = list(lines)
    last = steps[-1] if steps else ""
    if not (isinstance(last, str) and last.strip().lower() in ("quit", "exit", "q", "/quit", "/exit")):
        steps.append("quit")
    pending = list(answers or [])
    timed_out = False
    proc, master_fd = _spawn(argv, env, cwd or _REPO_ROOT)
    tty = _Pty(master_fd)
    deadline = _session_deadline(timeout_sec)
    try:
        typed = ""
        for step in steps:
            if isinstance(step, tuple):
                delay, text = step
                if isinstance(delay, str):  # keyed on progress: the action ran >= N s
                    while not _has_run(tty.text()[tty.mark:], delay) and _now() < deadline and not tty.eof:
                        tty.pump(0.1)
                    if not _has_run(tty.text()[tty.mark:], delay):  # it never got there
                        timed_out = not tty.eof
                        break
                else:
                    until = min(_now() + delay, deadline)
                    while _now() < until and not tty.eof:
                        tty.pump(until - _now())
                tty.send(text)
                typed = text
                continue
            while True:
                state = tty.wait_input(deadline, typed)
                if state != "permission":
                    break
                tty.send(pending.pop(0) if pending else "n")
                typed = ""
            if state in ("eof", "timeout"):
                timed_out = state == "timeout"
                break
            tty.send(step)
            typed = step
        # Read to EOF after quit (the verdict of a last async turn is not lost), or
        # until the child has exited and gone quiet (a grandchild may hold the PTY).
        while not tty.eof and _now() < deadline:
            tty.pump(min(0.5, deadline - _now()))
            if proc.poll() is not None and _now() - tty.last_data > 1.0:
                break
        try:
            proc.wait(timeout=max(0.1, deadline - _now()))
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        _kill_group(proc)
        os.close(master_fd)
        if script_path is not None:
            try:
                os.unlink(script_path)
            except OSError:
                pass
        shutil.rmtree(home_dir, ignore_errors=True)

    exit_code = proc.returncode if proc.returncode is not None else 1
    transcript = tty.buf.decode("utf-8", errors="replace")
    return ReplSessionResult(transcript=transcript, exit_code=exit_code, timed_out=timed_out)
