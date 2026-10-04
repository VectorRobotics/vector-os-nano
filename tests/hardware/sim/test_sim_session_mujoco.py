# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""WP03 — the in-process SimSession on REAL MuJoCo (headless EGL).

* E6: the switch cycle go2_piper -> go2 -> so101 -> go2_piper through the NL
  start_simulation tool, in ONE process (the realloc spike: MuJoCo 3.x reallocates
  worlds fine), returns threads and child processes to baseline.
* A dead physics thread before readiness is an error, never "Started".
* No VECTOR_NAV_STACK + a tare manifest: the launch succeeds and says explore is
  unavailable.
* Perception frames always have the size their intrinsics claim, even after another
  caller rendered the same camera at a different size.
* The go2_piper scene compiles from a relocated copy of the package.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from vector_os_nano.vcli.tools.base import CategorizedToolRegistry, ToolContext  # noqa: E402
from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool  # noqa: E402

pytestmark = pytest.mark.sim

_REPO = Path(__file__).resolve().parents[3]


def _ctx(app: dict[str, Any]) -> ToolContext:
    return ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                       abort=threading.Event(), app_state=app)


def _children() -> set[int]:
    """PIDs whose parent is this process (spawned subprocess groups show up here)."""
    me, kids = os.getpid(), set()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                fields = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(fields[1]) == me:
                kids.add(int(d))
    return kids


def _start(app: dict[str, Any], eid: str) -> Any:
    return SimStartTool().execute({"embodiment": eid, "gui": False}, _ctx(app))


def test_switch_cycle_returns_threads_and_children_to_baseline(monkeypatch) -> None:
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper

    monkeypatch.delenv("VECTOR_NAV_STACK", raising=False)
    monkeypatch.setenv("VECTOR_SIM_WITH_ARM", "1")  # must NOT pick the arm any more
    spawned: list[Any] = []
    real_popen = subprocess.Popen

    def _spy(argv: Any, *a: Any, **k: Any) -> Any:
        spawned.append(argv)
        return real_popen(argv, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", _spy)
    app: dict[str, Any] = {"registry": CategorizedToolRegistry()}
    threads0, kids0 = threading.active_count(), _children()

    for eid in ("go2_piper", "go2", "so101", "go2_piper"):
        res = _start(app, eid)
        assert not res.is_error and f"Started {eid} simulation, ready in" in res.content, res.content
        agent = app["sim_session"].agent
        if eid == "go2_piper":
            # In-process rig, never the ROS2 proxies / launch_explore.sh subprocess.
            assert type(agent._base) is MuJoCoGo2 and type(agent._arm) is MuJoCoPiper
            assert agent._base._mj.model.nu == 19
            # One perception at the manifest size; a 640x480 render of the same
            # camera by another caller (look) must not change what it gets.
            agent._base.get_camera_frame()
            p = agent._perception
            assert p.get_color_frame().shape == (240, 320, 3)
            assert p.get_depth_frame().shape == (240, 320)
            assert (p.get_intrinsics().width, p.get_intrinsics().height) == (320, 240)
        if eid == "go2":
            assert agent._arm is None and agent._base._mj.model.nu == 12  # env ignored
            assert "explore: coverage (VECTOR_NAV_STACK unset)" in res.content

    stop = SimStopTool().execute({}, _ctx(app))
    assert "Simulation stopped" in stop.content and app["sim_session"] is None
    assert threading.active_count() == threads0
    assert _children() == kids0
    assert spawned == [], f"the in-process sim spawned subprocesses: {spawned}"


def test_frames_render_from_any_thread_after_start(monkeypatch) -> None:
    """The ready probe renders on the starting (REPL) thread; skills then render
    from the vgg-executor thread and the nav bridge's camera timer. A GL context
    bound to the thread that last rendered (EGL_BAD_ACCESS) must not lock them out.
    """
    from concurrent.futures import ThreadPoolExecutor

    from vector_os_nano.sim.session import SimSession, SimSpec

    monkeypatch.setenv("VECTOR_NAV_STACK", "")
    threads0 = threading.active_count()
    s = SimSession.start(SimSpec("go2_piper", gui=False))
    try:
        p, base = s.agent._perception, s.agent._base
        assert p.get_color_frame().shape == (240, 320, 3)  # the starting thread

        def grab(_: int) -> tuple:
            return (p.get_color_frame().shape, p.get_depth_frame().shape,
                    base.get_camera_frame().shape)

        want = ((240, 320, 3), (240, 320), (480, 640, 3))
        out: list[tuple] = []
        t = threading.Thread(target=lambda: out.append(grab(0)), name="vgg-executor")
        t.start()
        t.join()
        assert out == [want]  # another thread, while the starting thread lives on
        with ThreadPoolExecutor(4) as ex:  # concurrent callers serialize, none fails
            assert list(ex.map(grab, range(8))) == [want] * 8
        assert p.get_color_frame().shape == (240, 320, 3)  # and back on the start thread
    finally:
        s.stop()
    assert threading.active_count() == threads0


def test_interpreter_exit_stops_the_session_cleanly() -> None:
    """The bare REPL leaves an NL-launched session to atexit, which runs after the
    interpreter has already retired the render thread."""
    code = (
        "import vector_os_nano.sim.session as S\n"
        "orig = S.SimSession.stop\n"
        "def stop(self):\n"
        "    r = orig(self); print('STOP:', r, flush=True); return r\n"
        "S.SimSession.stop = stop\n"
        "s = S.SimSession.start(S.SimSpec('go2', gui=False))\n"
        "s.agent._base.get_camera_frame()\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=_REPO, capture_output=True,
                         text=True, env={**os.environ, "VECTOR_NAV_STACK": ""}, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "disconnected" in out.stdout and "FAILED" not in out.stdout, out.stdout


def test_dead_physics_thread_is_an_error_never_started(monkeypatch) -> None:
    from vector_os_nano.sim.runtime import SimRuntime

    monkeypatch.setattr(SimRuntime, "_loop", lambda self: None)  # thread dies at once
    app: dict[str, Any] = {"registry": CategorizedToolRegistry()}
    threads0 = threading.active_count()
    res = _start(app, "go2")
    assert res.is_error and "Started" not in res.content, res.content
    assert "physics is not stepping" in res.content
    assert app.get("sim_session") is None
    assert threading.active_count() == threads0


def test_go2_piper_scene_compiles_from_a_relocated_checkout(tmp_path) -> None:
    pkg = tmp_path / "vector_os_nano"
    shutil.copytree(_REPO / "vector_os_nano", pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "scene_*.xml"))
    shutil.copytree(_REPO / "config", tmp_path / "config")  # world manifests (grasp welds)
    code = (
        "import mujoco, vector_os_nano as v\n"
        "from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml as b\n"
        "p = b(with_arm=True)\n"
        "m = mujoco.MjModel.from_xml_path(str(p))\n"
        "print(v.__file__, p, m.nu)\n"
    )
    cache = tmp_path / "cache"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "XDG_CACHE_HOME": str(cache)},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    mod_file, scene, nu = out.stdout.split()
    # the package is the relocated copy; the generated scene lives in the user
    # cache (never the package) and points at the relocated meshes
    assert mod_file.startswith(str(pkg)) and scene.startswith(str(cache))
    assert nu == "19"
    text = Path(scene).read_text()
    assert str(_REPO) not in text, "scene still points into the original checkout"
    assert str(pkg) in text
