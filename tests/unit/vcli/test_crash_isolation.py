# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""A MuJoCo segfault loses at most the current REPL turn (WP06 crash isolation).

* the transcript and the scene graph are flushed between turns, not only at exit;
* faulthandler writes the C-level crash trace next to the transcript.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from vector_os_nano.vcli.runtime import Runtime, flush
from vector_os_nano.vcli.session import create_session, load_session

_REPO = Path(__file__).resolve().parents[3]


def test_flush_saves_transcript_and_scene_graph(tmp_path) -> None:
    session = create_session(directory=tmp_path)
    session.append_user("走到坐标 (9.0,3.0)")
    saved = []
    agent = SimpleNamespace(_spatial_memory=SimpleNamespace(save=lambda: saved.append(1)))
    flush(Runtime(engine=None, session=session, app_state={"agent": agent}))
    assert load_session(session.session_id, directory=tmp_path).to_messages()[0]["content"]
    assert saved == [1]
    flush(Runtime(engine=None, session=session, app_state={}))  # no sim: transcript only, never raises


def test_segfault_trace_lands_next_to_the_transcript(tmp_path) -> None:
    code = (
        "import faulthandler, sys\n"
        "from types import SimpleNamespace\n"
        "from pathlib import Path\n"
        "from vector_os_nano.vcli.repl import _enable_crash_log\n"
        f"_enable_crash_log(SimpleNamespace(_path=Path({str(tmp_path / 's1.jsonl')!r})))\n"
        "faulthandler._sigsegv()\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=_REPO, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode != 0
    log = (tmp_path / "s1.crash.log").read_text()
    assert "Fatal Python error: Segmentation fault" in log, log
