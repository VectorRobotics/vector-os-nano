# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E1 + E3a — bare ``vector-cli`` + NL launches go2+Piper and grounds a perception grasp.

The owner's acceptance face, driven on a PTY with NO flags (the D97 failure mode
was this test hand-building the rig instead of using the product launcher):

    vector> 启动 go2 仿真 带机械臂 无窗口
      ▸ sim start go2_piper ok
      ▸ Started go2_piper simulation, ready in …            (E1: within 90 s)
    vector> 拿起绿色的瓶子
      ▸ perception_grasp → verify holding_object('pickable_bottle_green') ✓
      verdict GROUNDED verified=True                         (E3a)

Only the network LLM is replaced (``VECTOR_FAKE_LLM_TOOLS`` replays the model's tool
calls). Everything else is the product: SimSession (MuJoCoGo2 + MuJoCoPiper +
MuJoCoPiperGripper), rebind_world, the native producer, perception_grasp (colour
resolve -> rendered depth -> IK -> weld, never ground truth) and the untouched
verify spine. The driver is reactive: it waits for each line the REPL prints
instead of sleeping fixed delays.

The grasp itself is physics (measured ~0.83 reliable, D94/D95): a red here is
first a grasp miss — read the transcript before blaming the launcher.
"""
from __future__ import annotations

import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_REPO = Path(__file__).resolve().parents[2]
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[=>()][0-9A-B]?")

_SCRIPT = {"turns": [
    # turn 1 (legacy tool_use path: "仿真" bypasses the planner)
    {"tool_calls": [{"name": "start_simulation",
                     "input": {"embodiment": "go2_piper", "gui": False}}]},
    {"text": "go2_piper is up."},
    # turn 2 (native producer): perceive + grasp, then prove possession by GT
    {"tool_calls": [{"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
    {"tool_calls": [{"name": "verify",
                     "input": {"expr": "holding_object('pickable_bottle_green')"}}]},
    {"text": "holding the green bottle."},
]}


class _Repl:
    """Bare vector-cli on a PTY; ``expect`` waits for printed text, then moves on."""

    def __init__(self, env: dict[str, str]) -> None:
        self._master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vector_os_nano.vcli.cli"],
            stdin=slave, stdout=slave, stderr=slave, cwd=str(_REPO), env=env,
            close_fds=True, start_new_session=True,  # own pgid -> killpg at teardown
        )
        os.close(slave)
        self._buf: list[bytes] = []
        self._pos = 0
        self._stop = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([self._master], [], [], 0.2)
            if r:
                try:
                    data = os.read(self._master, 65536)
                except OSError:
                    return
                if not data:
                    return
                self._buf.append(data)

    def text(self) -> str:
        raw = b"".join(self._buf).decode("utf-8", errors="replace")
        return " ".join(_ANSI.sub("", raw).split())

    def expect(self, pattern: str, timeout: float) -> re.Match:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            m = re.compile(pattern).search(self.text(), self._pos)
            if m:
                self._pos = m.end()
                return m
            if self.proc.poll() is not None:
                break
            time.sleep(0.3)
        t = self.text()
        raise AssertionError(f"timed out waiting for {pattern!r}; transcript:\n{t[:3000]}\n...\n{t[-3000:]}")

    def send(self, line: str) -> None:
        os.write(self._master, (line + "\n").encode("utf-8"))

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.proc.pid, sig)
                    self.proc.wait(timeout=10)
                    break
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    continue
        self._stop.set()
        os.close(self._master)


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_bare_cli_nl_launch_go2_piper_then_grasp_grounds(tmp_path) -> None:
    script = tmp_path / "script.json"
    script.write_text(json.dumps(_SCRIPT, ensure_ascii=False))
    env = {k: v for k, v in os.environ.items()
           if k not in ("VECTOR_NATIVE_LOOP", "VECTOR_NATIVE_FIRST", "VECTOR_REPL_NATIVE")}
    env.update(
        # Explicitly empty (not absent): the CLI's load_dotenv would otherwise pull it
        # from a repo .env and start the optional TARE stack, which E1/E3a must not
        # depend on (its bridge also follows a machine-global /tmp nav flag).
        VECTOR_NAV_STACK="",
        PYTHONPATH=os.pathsep.join([str(_REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        HOME=tempfile.mkdtemp(prefix="vector_e1_home_", dir=tmp_path),
        TERM="xterm-256color", COLUMNS="220", PROMPT_TOOLKIT_NO_CPR="1",
        MUJOCO_GL=env.get("MUJOCO_GL", "egl"),
        VECTOR_FAKE_LLM_TOOLS=str(script),
    )
    env.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

    repl = _Repl(env)
    try:
        repl.expect(r"vector>", 90)
        repl.send("启动 go2 仿真 带机械臂 无窗口")
        t0 = time.monotonic()
        repl.expect(r"sim start go2_piper ok", 90)
        ready = repl.expect(r"Started go2_piper simulation, ready in [0-9.]+s: MuJoCoGo2 \+ MuJoCoPiper", 30)
        assert time.monotonic() - t0 < 90, "E1: the ready line must arrive within 90 s"
        line = repl.text()[ready.start():ready.start() + 300]
        assert "gait=" in line and "explore: unavailable (VECTOR_NAV_STACK unset)" in line

        repl.expect(r"vector>", 60)
        repl.send("拿起绿色的瓶子")
        repl.expect(r"perception_grasp → verify holding_object\('pickable_bottle_green'\)", 300)
        verdict = repl.expect(r"verdict (\w+) verified=(\w+)", 60)
        assert verdict.group(1, 2) == ("GROUNDED", "True"), repl.text()[-3000:]
    finally:
        repl.close()
