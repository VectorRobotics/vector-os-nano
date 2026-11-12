# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""WP03 — SimSession / start_simulation / bind_body, hermetic (no MuJoCo).

The real-physics half (switch cycle, dead physics thread, renderer sizes, the
relocated checkout) lives in tests/hardware/sim/test_sim_session_mujoco.py.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.embodiments import config as emb_config
from vector_os_nano.embodiments import registry
from vector_os_nano.sim import session as sess
from vector_os_nano.sim.session import SimSession, SimSpec, SimStartError
from vector_os_nano.vcli.runtime import bind_body, build_registry
from vector_os_nano.vcli.tools.base import ToolContext
from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool


def _ctx(app: dict[str, Any]) -> ToolContext:
    return ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                       abort=threading.Event(), app_state=app)


# ---------------------------------------------------------------------------
# Registry + manifests
# ---------------------------------------------------------------------------


def test_registry_lists_every_manifest_and_each_loads() -> None:
    ids = registry.embodiment_ids()
    assert {"go2", "go2_piper", "so101", "g1"} <= set(ids)
    for eid in ids:
        cfg = registry.load(eid)
        assert cfg.id == eid and cfg.driver in sess._BUILDERS, (eid, cfg.driver)


def test_go2_piper_manifest_carries_the_arm_and_the_grasp_camera_size() -> None:
    cfg = registry.load("go2_piper")
    assert cfg.arm is not None and cfg.arm.dof == 6 and cfg.arm.gripper.finger_bodies
    assert cfg.model.path.endswith("go2_piper/go2_piper.xml")
    assert sess._camera_size(cfg) == (320, 240)
    assert registry.load("go2").arm is None
    assert registry.load("go2").nav["explorer"] == "tare"


def test_start_simulation_enum_is_generated_from_the_registry() -> None:
    prop = SimStartTool.input_schema["properties"]["embodiment"]
    assert prop["enum"] == registry.embodiment_ids()
    assert "embodiment" in SimStartTool.input_schema["required"]
    # The deleted fork is gone from the schema the model sees.
    assert set(SimStartTool.input_schema["properties"]) == {"embodiment", "gui"}


def test_unknown_embodiment_fails_loud_with_the_valid_set() -> None:
    res = SimStartTool().execute({"embodiment": "drone"}, _ctx({}))
    assert res.is_error and "drone" in res.content and "go2_piper" in res.content


def test_gui_must_be_a_real_bool(monkeypatch) -> None:
    def no_start(spec: SimSpec, **_k: Any) -> Any:
        raise SimStartError("not started", f"gui={spec.gui}")

    monkeypatch.setattr(SimSession, "start", staticmethod(no_start))
    res = SimStartTool().execute({"embodiment": "go2", "gui": "false"}, _ctx({}))
    assert res.is_error and "gui must be true or false" in res.content
    # Omitted, the default still opens the viewer (level71 passes gui=False explicitly).
    assert "gui=True" in SimStartTool().execute({"embodiment": "go2"}, _ctx({})).content


def test_corrupt_manifest_is_error_naming_the_key(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "robot.yaml").write_text(
        "id: broken\ndisplay_name: x\ndriver: mujoco_go2\n"
        "model: {path: a.xml, root_body: b}\nstance: {}\n"
        "policy: {ref: none}\ncapabilities: {}\n"  # no 'spawn'
    )
    monkeypatch.setattr(emb_config, "_EMBODIMENTS_DIR", tmp_path)
    res = SimStartTool().execute({"embodiment": "broken"}, _ctx({}))
    assert res.is_error
    assert "spawn" in res.content and "Started" not in res.content


def test_unknown_driver_is_error_naming_the_field(tmp_path, monkeypatch) -> None:
    src = (Path(emb_config.__file__).parent / "go2" / "robot.yaml").read_text()
    (tmp_path / "odd").mkdir()
    (tmp_path / "odd" / "robot.yaml").write_text(
        src.replace("id: go2\n", "id: odd\n").replace("driver: mujoco_go2", "driver: warp_drive")
    )
    monkeypatch.setattr(emb_config, "_EMBODIMENTS_DIR", tmp_path)
    with pytest.raises(SimStartError, match="driver='warp_drive'"):
        SimSession.start(SimSpec("odd", gui=False))


# ---------------------------------------------------------------------------
# Readiness + teardown
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, step: float) -> None:
        self.t, self.step = 0.0, step

    def get_sim_time(self) -> float:
        self.t += self.step
        return self.t


def _agent(base: Any = None, arm: Any = None, frame: Any = 1) -> Any:
    import numpy as np

    perception = SimpleNamespace(get_color_frame=lambda: np.full((24, 32, 3), frame, np.uint8))
    return SimpleNamespace(_base=base, _arm=arm, _perception=perception)


def test_ready_probe_fails_fast_when_the_clock_is_dead(monkeypatch) -> None:
    import time

    monkeypatch.setattr(sess, "_STALL_S", 0.3)
    cfg = registry.load("go2")
    t0 = time.monotonic()
    with pytest.raises(SimStartError, match="physics is not stepping"):
        sess._probe_ready(_agent(base=_Clock(0.0)), cfg, deadline=time.monotonic() + 60)
    assert time.monotonic() - t0 < 5  # a dead thread is detected, not waited out


def test_ready_probe_deadline_holds_for_a_creeping_clock() -> None:
    import time

    cfg = registry.load("go2")
    t0 = time.monotonic()
    with pytest.raises(SimStartError, match="physics is not stepping"):
        # Moves every poll (never "stalled") but would need ~250 s to reach 0.5 s.
        sess._probe_ready(_agent(base=_Clock(1e-4)), cfg, deadline=time.monotonic() + 1)
    assert time.monotonic() - t0 < 3


def test_ready_probe_never_renders_on_the_calling_thread() -> None:
    """A GL context stays bound to the live thread that last rendered with it, so a
    probe frame grabbed on the REPL thread would lock every other thread out."""
    import numpy as np

    seen: list[threading.Thread] = []

    def frame() -> Any:
        seen.append(threading.current_thread())
        return np.ones((24, 32, 3), np.uint8)

    agent = SimpleNamespace(_base=None, _arm=None,
                            _perception=SimpleNamespace(get_color_frame=frame))
    assert sess._probe_ready(agent, registry.load("go2"), deadline=1e18)["camera"] == "32x24"
    assert seen and seen[0] is not threading.current_thread() and not seen[0].is_alive()


def test_ready_probe_rejects_black_frame_and_nan_arm() -> None:
    import time

    cfg = registry.load("go2_piper")
    dl = time.monotonic() + 30
    with pytest.raises(SimStartError, match="black"):
        sess._probe_ready(_agent(base=_Clock(0.1), frame=0), cfg, deadline=dl)
    nan_arm = SimpleNamespace(get_joint_positions=lambda: [0.0, float("nan")])
    with pytest.raises(SimStartError, match="not finite"):
        sess._probe_ready(_agent(base=_Clock(0.1), arm=nan_arm), cfg, deadline=dl)
    ok = sess._probe_ready(_agent(base=_Clock(0.1), arm=SimpleNamespace(
        get_joint_positions=lambda: [0.0] * 6)), cfg, deadline=dl)
    assert ok["camera"] == "32x24" and ok["arm"] == "6-DoF"


def _fake_builder(cleanups: list[str], *, fail: bool = False):
    def build(session: SimSession, _key: Any) -> Any:
        session.own("device closed", lambda: cleanups.append("closed"))
        if fail:
            raise RuntimeError("boom")
        return _agent(base=_Clock(0.2), arm=SimpleNamespace(get_joint_positions=lambda: [0.0]))
    return build


class _Atexit:
    """Records the session's atexit hooks (CPython's _ncallbacks never shrinks)."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def register(self, fn: Any) -> None:
        self.hooks.append(fn)

    def unregister(self, fn: Any) -> None:
        self.hooks = [h for h in self.hooks if h != fn]


def test_failed_start_tears_down_what_it_built(monkeypatch) -> None:
    closed: list[str] = []
    hooks = _Atexit()
    monkeypatch.setattr(sess, "atexit", hooks)
    monkeypatch.setitem(sess._BUILDERS, "mujoco_go2", _fake_builder(closed, fail=True))
    with pytest.raises(SimStartError, match="boom"):
        SimSession.start(SimSpec("go2", gui=False))
    assert closed == ["closed"] and not sess._LIVE and not hooks.hooks


def test_stop_is_idempotent_and_unregisters_atexit(monkeypatch) -> None:
    closed: list[str] = []
    hooks = _Atexit()
    monkeypatch.setattr(sess, "atexit", hooks)
    monkeypatch.setitem(sess._BUILDERS, "mujoco_go2", _fake_builder(closed))
    s = SimSession.start(SimSpec("go2", gui=False))
    assert s in sess._LIVE and hooks.hooks == [s.stop]
    assert "device closed" in s.stop()
    assert s.stop() == "already stopped" and closed == ["closed"]
    assert s not in sess._LIVE and not hooks.hooks


# ---------------------------------------------------------------------------
# start/stop_simulation: switch semantics
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, spec: SimSpec) -> None:
        self.spec, self.agent, self.stopped = spec, None, 0

    def stop(self) -> str:
        self.stopped += 1
        return "fake stopped"

    def summary(self) -> str:
        return "ready in 0.0s: fake"


def test_switch_stops_the_previous_session_first(monkeypatch) -> None:
    started: list[SimSpec] = []

    def fake_start(spec: SimSpec, **_: Any) -> _FakeSession:
        assert all(s.stopped for s in app_sessions), "previous session still running"
        started.append(spec)
        s = _FakeSession(spec)
        app_sessions.append(s)
        return s

    app_sessions: list[_FakeSession] = []
    monkeypatch.setattr(SimSession, "start", staticmethod(fake_start))
    app: dict[str, Any] = {}
    tool = SimStartTool()
    for eid in ("go2_piper", "go2", "so101", "go2_piper"):
        res = tool.execute({"embodiment": eid, "gui": False}, _ctx(app))
        assert not res.is_error and f"Started {eid}" in res.content
        assert app["sim_session"].spec.embodiment == eid
    assert [s.stopped for s in app_sessions] == [1, 1, 1, 0]
    # Same spec again: no restart.
    res = tool.execute({"embodiment": "go2_piper", "gui": False}, _ctx(app))
    assert "already running" in res.content and len(started) == 4
    # Stop rebinds to no session; a second stop is a no-op message.
    assert "fake stopped" in SimStopTool().execute({}, _ctx(app)).content
    assert app["sim_session"] is None and app["agent"] is None
    assert SimStopTool().execute({}, _ctx(app)).content == "No simulation is running."


def test_switch_to_a_typo_never_stops_the_running_session(monkeypatch) -> None:
    """A bad embodiment id is rejected BEFORE the live sim is torn down (tools-worlds verifier)."""
    sessions: list[_FakeSession] = []

    def fake_start(spec: SimSpec, **_: Any) -> _FakeSession:
        sessions.append(_FakeSession(spec))
        return sessions[-1]

    monkeypatch.setattr(SimSession, "start", staticmethod(fake_start))
    app: dict[str, Any] = {}
    tool = SimStartTool()
    assert not tool.execute({"embodiment": "go2_piper", "gui": False}, _ctx(app)).is_error
    res = tool.execute({"embodiment": "go2-pipr", "gui": False}, _ctx(app))
    assert res.is_error and "go2_piper" in res.content  # names the valid set
    assert sessions[0].stopped == 0 and app["sim_session"] is sessions[0]


# ---------------------------------------------------------------------------
# bind_body — the ONE re-init path
# ---------------------------------------------------------------------------


class _Engine:
    """Records every bind; ``fail`` makes the next bind raise (engine.bind is atomic)."""

    def __init__(self) -> None:
        self.bound: list[tuple[Any, str]] = []
        self.fail: Exception | None = None

    def bind(self, agent: Any, world: Any) -> None:
        if self.fail is not None:
            raise self.fail
        self.bound.append((agent, world.name))


def _app(**extra: Any) -> dict[str, Any]:
    app = {"registry": build_registry(), "engine": _Engine()}
    app.update(extra)
    return app


def _robot_session() -> Any:
    from vector_os_nano.core.agent import Agent

    base = SimpleNamespace(get_position=lambda: [0.0, 0.0, 0.3], get_heading=lambda: 0.0)
    agent = Agent(base=base, config={}, skills=sess._skill_pack("go2"))
    return SimpleNamespace(agent=agent, spec=SimSpec("go2"))


def _skill_tools(reg: Any) -> list[str]:
    return [n for n in reg.list_tools() if getattr(reg.get(n), "_is_skill_wrapper", False)]


def test_bind_body_binds_the_engine_and_adds_the_skill_tools() -> None:
    app = _app()
    s = _robot_session()
    bind_body(app, s)
    assert app["engine"].bound == [(s.agent, "robot")]
    assert _skill_tools(app["registry"])
    assert app["agent"] is s.agent and app["sim_session"] is s


def test_bind_body_to_none_binds_the_dev_world_and_drops_the_skill_tools() -> None:
    app = _app()
    bind_body(app, _robot_session())
    bind_body(app, None)
    assert app["engine"].bound[-1] == (None, "dev")
    assert not _skill_tools(app["registry"])
    assert app["agent"] is None and app["sim_session"] is None


def test_a_started_body_the_engine_cannot_bind_is_stopped(monkeypatch) -> None:
    """The start tool never leaves a running sim the REPL is not bound to."""
    sessions: list[_FakeSession] = []

    def fake_start(spec: SimSpec, **_: Any) -> _FakeSession:
        sessions.append(_FakeSession(spec))
        return sessions[-1]

    monkeypatch.setattr(SimSession, "start", staticmethod(fake_start))
    app = _app()
    app["engine"].fail = RuntimeError("no ground truth")
    res = SimStartTool().execute({"embodiment": "go2", "gui": False}, _ctx(app))
    assert res.is_error and "binding it failed: no ground truth" in res.content
    assert sessions[0].stopped == 1 and app.get("sim_session") is None


def test_a_failed_bind_changes_nothing() -> None:
    """legacy-cognitive:B19: the engine is bound first; a failure leaves the REPL whole."""
    app = _app()
    app["engine"].fail = RuntimeError("cannot bind")
    with pytest.raises(RuntimeError):
        bind_body(app, _robot_session())
    assert "agent" not in app and not _skill_tools(app["registry"])


# ---------------------------------------------------------------------------
# CLI: fail loud, and stop the session before os._exit
# ---------------------------------------------------------------------------


def test_sim_flag_failure_exits_2_instead_of_dev_world(monkeypatch) -> None:
    from vector_os_nano.vcli import cli, runtime

    def boom(*_a: Any, **_k: Any) -> Any:
        raise SimStartError("not ready", "camera frame is empty/black")

    monkeypatch.setattr(SimSession, "start", staticmethod(boom))
    with pytest.raises(SimStartError):
        runtime.start_sim("go2", gui=False)
    monkeypatch.setattr(cli, "_maybe_reexec_under_mjpython", lambda args: None)
    monkeypatch.setattr(cli, "_setup_logging", lambda verbose: None)
    with pytest.raises(SystemExit) as ei:
        cli.main(["--sim-go2", "--headless"])
    assert ei.value.code == 2


def test_cli_aliases_map_onto_embodiments() -> None:
    from vector_os_nano.vcli import cli

    assert cli._requested_embodiment(cli.parse_args(["--sim-go2"])) == "go2"
    assert cli._requested_embodiment(cli.parse_args(["--sim"])) == "so101"
    assert cli._requested_embodiment(cli.parse_args(["--embodiment", "go2_piper"])) == "go2_piper"
    assert cli._requested_embodiment(cli.parse_args([])) is None


def test_print_path_stops_live_sessions_before_os_exit(monkeypatch) -> None:
    from vector_os_nano.vcli import cli

    s = _FakeSession(SimSpec("go2"))
    monkeypatch.setattr(sess, "_LIVE", [s])
    monkeypatch.setattr(cli, "run_one_turn", lambda args: 0)
    monkeypatch.setattr(cli, "_setup_logging", lambda verbose: None)  # no global side effect

    hard_exits: list[int] = []

    def fake_exit(code: int) -> None:
        hard_exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(cli.os, "_exit", fake_exit)
    with pytest.raises(SystemExit):
        cli.main(["-p", "hi", "--sim-go2", "--headless"])
    assert s.stopped == 1 and hard_exits == [0]

    # A session the turn itself launched (NL start_simulation, no flag) gets the
    # same stop + hard exit; a turn with no sim keeps the plain sys.exit.
    nl = _FakeSession(SimSpec("go2_piper"))
    monkeypatch.setattr(sess, "_LIVE", [nl])
    with pytest.raises(SystemExit):
        cli.main(["-p", "启动 go2 仿真"])
    assert nl.stopped == 1 and hard_exits == [0, 0]
    monkeypatch.setattr(sess, "_LIVE", [])
    with pytest.raises(SystemExit):
        cli.main(["-p", "hi"])
    assert hard_exits == [0, 0]


def test_go2_scene_never_silently_drops_the_arm(monkeypatch, tmp_path) -> None:
    from vector_os_nano.hardware.sim import mujoco_go2

    monkeypatch.setattr(mujoco_go2, "_GO2_PIPER_XML", tmp_path / "missing.xml")
    with pytest.raises(FileNotFoundError, match="missing.xml"):
        mujoco_go2._build_room_scene_xml(with_arm=True)


def test_arm_choice_comes_from_the_ctor_not_the_env(monkeypatch) -> None:
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    monkeypatch.setenv("VECTOR_SIM_WITH_ARM", "1")
    assert MuJoCoGo2()._with_arm is False
    assert MuJoCoGo2(with_arm=True)._with_arm is True


def test_nav_teardown_kills_every_group_in_its_session_and_nothing_else() -> None:
    """launch_nav_only.sh runs nodes under ``set -m`` (own pgid each): killing only the
    script's group orphaned them (seen live: rviz2/terrainAnalysis outliving the REPL)."""
    import subprocess
    import time

    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    script = subprocess.Popen(
        ["bash", "-c", "set -m; sleep 60 & sleep 60 & wait"],
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while len(sess._session_pgids({script.pid})) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(sess._session_pgids({script.pid})) == 3  # bash + 2 job groups
        sess._kill_sessions({script.pid}, grace_s=2.0)
        script.wait(timeout=5)
        assert sess._session_pgids({script.pid}) == set()
        assert bystander.poll() is None, "a process outside our session was killed"
    finally:
        bystander.kill()
        bystander.wait()
