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
            assert "explore: unavailable (VECTOR_NAV_STACK unset)" in res.content

    stop = SimStopTool().execute({}, _ctx(app))
    assert "Simulation stopped" in stop.content and app["sim_session"] is None
    assert threading.active_count() == threads0
    assert _children() == kids0
    assert spawned == [], f"the in-process sim spawned subprocesses: {spawned}"


def test_dead_physics_thread_is_an_error_never_started(monkeypatch) -> None:
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    monkeypatch.setattr(MuJoCoGo2, "_physics_loop", lambda self: None)  # thread dies at once
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
    code = (
        "import mujoco, vector_os_nano as v\n"
        "from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml as b\n"
        "p = b(with_arm=True)\n"
        "m = mujoco.MjModel.from_xml_path(str(p))\n"
        "print(v.__file__, p, m.nu)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)}, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    mod_file, scene, nu = out.stdout.split()
    assert mod_file.startswith(str(tmp_path)) and scene.startswith(str(tmp_path))
    assert nu == "19"
    assert str(_REPO) not in Path(scene).read_text(), "scene still points into the original checkout"
