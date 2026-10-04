# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimSession — the ONE in-process simulation behind NL ``start_simulation``.

``start_simulation(embodiment=...)`` and the ``--sim`` / ``--sim-go2`` /
``--embodiment`` flags all call :meth:`SimSession.start`. The body is the proven
in-process rig (MuJoCoGo2 + MuJoCoPiper + MuJoCoPiperGripper, every measured
grasp number came from it), chosen by the embodiment manifest
(``embodiments/<id>/robot.yaml``) — never by an env var.

* ``start`` fails loud with :class:`SimStartError` (reason + detail) and tears
  down whatever it built; it returns only after the readiness probe passed
  (sim time advancing, a non-black camera frame, finite arm joints).
* ``stop`` is idempotent: it disconnects the devices (renderers close with the
  base), kills only the process groups this session spawned and unregisters its
  atexit hook.
* :func:`rebind_world` is the ONE path that points the REPL at a session (or at
  no session): skill tools, tool categories, world, system prompt and VGG.
"""
from __future__ import annotations

import atexit
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from vector_os_nano.core.run_dir import run_path

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]

# Readiness: physics must advance this much sim time, and a clock that does not
# move at all for _STALL_S of wall time is a dead physics thread.
_READY_SIM_S = 0.5
_STALL_S = 3.0

# Frame-source methods a base must expose for Go2GraspPerception to localize.
_GO2_PERCEPTION_FRAME_METHODS = ("get_camera_frame", "get_depth_frame", "get_camera_pose")


@dataclass(frozen=True)
class SimSpec:
    """What to run: an embodiment id from the registry, with or without a window."""

    embodiment: str
    gui: bool = True


class SimStartError(RuntimeError):
    """A simulation could not be started or never became ready."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


_LIVE: list["SimSession"] = []


def session_for(agent: Any) -> "SimSession | None":
    """The live session that owns *agent*, or None (real hardware / no sim)."""
    if agent is None:
        return None
    return next((s for s in _LIVE if s.agent is agent), None)


def mark_turn(agent: Any, phase: str, goal: str = "") -> None:
    """Turn boundary hook for the producer: no-op without a live session. Never raises."""
    s = session_for(agent)
    if s is None:
        return
    try:
        s.mark_turn(phase, goal)
    except Exception as exc:  # noqa: BLE001 — a GT dump must never break a turn
        logger.warning("GT turn mark (%s) failed: %s", phase, exc)


def _append_gt_dump(path: str, s: "SimSession", phase: str, goal: str) -> None:
    """One JSON line of GT: base pose, every object's pose + weld, the world's
    surfaces and user location (everything tools/gt_grade.py needs, no oracle code)."""
    import json

    gt, world = s.gt, s.world
    welds = gt.weld_active()
    line = {
        "phase": phase, "turn": s._turns, "goal": goal, "embodiment": s.config.id,
        "world": s.config.world, "sim_time": gt.sim_time, "base": gt.base_pose(),
        "objects": {n: {"pos": p, "welded": welds.get(n, False)} for n, p in gt.object_positions().items()},
        "surfaces": {k: {"box": v.box, "height": v.height} for k, v in (world.surfaces if world else {}).items()},
        "user": world.user if world else None,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def stop_all() -> int:
    """Stop every live session (used before ``os._exit``, which skips atexit).

    Returns how many were live.
    """
    live = list(_LIVE)
    for s in live:
        s.stop()
    return len(live)


def is_sim_agent(agent: Any) -> bool:
    """True iff *agent* is the body of a live SimSession (no real-world consequence)."""
    return agent is not None and any(s.agent is agent for s in _LIVE)


class SimSession:
    """One running simulation: the Agent plus everything it owns."""

    def __init__(self, spec: SimSpec, config: Any) -> None:
        self.spec = spec
        self.config = config
        self.agent: Any = None
        self.notes: list[str] = []
        self.ready: dict[str, Any] = {}
        self.startup_s: float = 0.0
        self._cleanups: list[tuple[str, Callable[[], Any]]] = []
        self._stopped = False
        self.gt: Any = None  # GroundTruth of the running sim (oracles + graders only)
        self.world: Any = None  # the WorldManifest the body stands in
        self._turn: tuple[float, float] = (-math.inf, -math.inf)  # (sim, wall) turn start
        self._turns = 0

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def validate(spec: SimSpec):
        """Resolve the manifest + driver without starting anything (fail loud, valid set).

        Callers that replace a running session validate FIRST, so a typo never
        tears down the sim that is already up.
        """
        from vector_os_nano.embodiments import registry
        from vector_os_nano.embodiments.config import EmbodimentConfigError

        try:
            cfg = registry.load(spec.embodiment)
        except EmbodimentConfigError as exc:
            raise SimStartError("bad embodiment manifest", str(exc)) from exc
        builder = _BUILDERS.get(cfg.driver)
        if builder is None:
            raise SimStartError(
                "bad embodiment manifest",
                f"{cfg.id}/robot.yaml.driver={cfg.driver!r}; valid: {sorted(_BUILDERS)}",
            )
        return cfg, builder

    @classmethod
    def start(
        cls, spec: SimSpec, *, vlm_api_key: str | None = None, deadline_s: float = 90.0,
    ) -> "SimSession":
        cfg, builder = cls.validate(spec)
        session = cls(spec, cfg)
        t0 = time.monotonic()
        try:
            session.agent = builder(session, vlm_api_key)
            session.ready = _probe_ready(session.agent, cfg, deadline=t0 + deadline_s)
            session._bind_ground_truth()
        except BaseException as exc:
            session.stop()
            if isinstance(exc, (SimStartError, KeyboardInterrupt, SystemExit)):
                raise
            raise SimStartError("start failed", f"{type(exc).__name__}: {exc}") from exc
        session.startup_s = time.monotonic() - t0
        _LIVE.append(session)
        atexit.register(session.stop)
        return session

    def stop(self) -> str:
        """Tear down everything this session owns (LIFO). Idempotent."""
        if self._stopped:
            return "already stopped"
        self._stopped = True
        parts: list[str] = []
        while self._cleanups:
            what, fn = self._cleanups.pop()
            try:
                fn()
                parts.append(what)
            except Exception as exc:  # noqa: BLE001 — keep tearing down the rest
                parts.append(f"{what} FAILED ({exc})")
        atexit.unregister(self.stop)
        if self in _LIVE:
            _LIVE.remove(self)
        return "; ".join(parts) or "nothing to stop"

    # -- ground truth (oracles / graders; skills never see it) ---------------

    def _bind_ground_truth(self) -> None:
        import mujoco as mj

        from vector_os_nano.sim.ground_truth import GroundTruth
        from vector_os_nano.vcli.worlds.world_manifest import load_world

        agent = self.agent
        rt = getattr(getattr(agent, "_base", None), "runtime", None) or getattr(
            getattr(agent, "_arm", None), "runtime", None)
        if rt is None:  # no MjData owner: no oracles, every verify grades RAN
            self.notes.append("ground truth: unavailable (no SimRuntime on the base or the arm)")
            return
        root = mj.mj_name2id(rt.model, mj.mjtObj.mjOBJ_BODY, self.config.model.root_body)
        self.gt = GroundTruth(rt, robot_root_bid=int(root))
        self.world = load_world(self.config.world) if self.config.world else None
        # Skills resolve object names through the world manifest by id (never GT).
        cfg = getattr(agent, "_config", None)
        if isinstance(cfg, dict) and self.config.world:
            cfg["world"] = self.config.world

    def oracles(self) -> dict[str, Any]:
        """The GT verify namespace of this session (vcli/worlds/sim_oracles)."""
        from vector_os_nano.vcli.worlds.sim_oracles import build_namespace

        return build_namespace(
            self.gt, self.world, self.config,
            belief=lambda: getattr(self.agent, "_spatial_memory", None),
            turn_start=lambda: self._turn,
        )

    def mark_turn(self, phase: str, goal: str = "") -> None:
        """A REPL turn starts / ends: record its start (kept_holding, localized) and,
        with ``VECTOR_GT_DUMP=<file>``, append one GT line for the independent grader
        (tools/gt_grade.py)."""
        if self.gt is None:
            return
        if phase == "start":
            self._turns += 1
            self._turn = (self.gt.sim_time, time.time())
        path = os.environ.get("VECTOR_GT_DUMP", "")
        if path:
            _append_gt_dump(path, self, phase, goal)

    def own(self, what: str, cleanup: Callable[[], Any]) -> None:
        """Register a teardown step (run LIFO by stop)."""
        self._cleanups.append((what, cleanup))

    def summary(self) -> str:
        """One line: 'ready in Xs: <hardware>, <probe>, N skills; <notes>'."""
        agent = self.agent
        hw = " + ".join(
            type(d).__name__ for d in (getattr(agent, "_base", None), getattr(agent, "_arm", None)) if d
        )
        n = len(agent._skill_registry.list_skills()) if agent is not None else 0
        probe = ", ".join(f"{k}={v}" for k, v in self.ready.items())
        line = f"ready in {self.startup_s:.1f}s: {hw}, {probe}, {n} skills"
        return "; ".join([line, *self.notes])


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def _probe_ready(agent: Any, cfg: Any, *, deadline: float) -> dict[str, Any]:
    """Prove the sim is live, or raise SimStartError naming what is not."""
    import numpy as np

    base = getattr(agent, "_base", None)
    arm = getattr(agent, "_arm", None)
    report: dict[str, Any] = {}

    clock = getattr(base, "get_sim_time", None)
    runtime = getattr(base, "runtime", None)
    if callable(clock) and runtime is not None and not runtime.threaded:
        # Caller-stepped (VECTOR_SIM_LOCKSTEP / macOS pump): advance it here.
        t0 = clock()
        if not base.wait_sim(_READY_SIM_S) or clock() - t0 < _READY_SIM_S - 1e-6:
            raise SimStartError("not ready", "caller-stepped physics did not advance")
        report["sim_time"] = f"{clock():.1f}s"
    elif callable(clock):
        t0 = last = clock()
        moved_at = time.monotonic()
        while (now := clock()) - t0 < _READY_SIM_S:
            if now != last:
                last, moved_at = now, time.monotonic()
            if time.monotonic() - moved_at > _STALL_S or time.monotonic() > deadline:
                raise SimStartError(
                    "not ready",
                    f"sim time advanced {now - t0:.2f}s (< {_READY_SIM_S}s): physics is not stepping",
                )
            time.sleep(0.05)
        report["sim_time"] = f"{clock():.1f}s"
    if getattr(base, "gait_backend", ""):
        report["gait"] = base.gait_backend

    if cfg.capabilities.camera:
        perception = getattr(agent, "_perception", None)
        grab = getattr(perception, "get_color_frame", None) or getattr(base, "get_camera_frame", None)
        frame = None
        if callable(grab):
            # On a thread that then exits: a GL context stays bound to the live
            # thread that last rendered with it, and this is the REPL thread.
            with ThreadPoolExecutor(1) as ex:
                frame = ex.submit(grab).result()
        if frame is None or not np.asarray(frame).any():
            raise SimStartError("not ready", "camera frame is empty/black")
        h, w = np.asarray(frame).shape[:2]
        report["camera"] = f"{w}x{h}"

    if arm is not None:
        q = list(arm.get_joint_positions())
        if not q or not all(math.isfinite(v) for v in q):
            raise SimStartError("not ready", f"arm joints not finite: {q}")
        report["arm"] = f"{len(q)}-DoF"
    return report


# ---------------------------------------------------------------------------
# Builders (one per transitional manifest ``driver``; WP17 makes it one)
# ---------------------------------------------------------------------------


def _camera_size(cfg: Any) -> tuple[int, int]:
    """(width, height) of the manifest's camera sensor; fail loud if unset."""
    from vector_os_nano.embodiments.config import EmbodimentConfigError

    for s in cfg.sensors:
        if s.role == "camera":
            try:
                return int(s.params["width"]), int(s.params["height"])
            except KeyError as exc:
                raise EmbodimentConfigError(
                    f"{cfg.id}/robot.yaml: camera sensor params missing {exc}"
                ) from exc
    raise EmbodimentConfigError(f"{cfg.id}/robot.yaml: no sensor with role 'camera'")


def _user_config() -> dict[str, Any]:
    from vector_os_nano.core.config import load_config

    path = _REPO / "config" / "user.yaml"
    return load_config(str(path)) if path.exists() else {}


def _build_go2_perception(base: Any, *, width: int, height: int) -> Any:
    """Go2GraspPerception over *base* at the manifest size, or None without a camera.

    Built ONCE per session; ``register_manipulation_skills`` reuses it, so the
    grasp and look/localize paths can never run at different sizes.
    """
    if base is None:
        return None
    for meth in _GO2_PERCEPTION_FRAME_METHODS:
        if not callable(getattr(base, meth, None)):
            logger.debug("[sim] base lacks %s — perception None", meth)
            return None
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception

    return Go2GraspPerception(base, width=width, height=height)


def _build_mujoco_go2(session: SimSession, vlm_api_key: str | None) -> Any:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.hardware.sim.mujoco_go2 import _ROOM_XML, MuJoCoGo2
    from vector_os_nano.skills.go2 import get_go2_skills

    cfg, spec = session.config, session.spec
    width, height = _camera_size(cfg)
    base = MuJoCoGo2(
        gui=spec.gui, room=True, backend=str(cfg.policy.spec.get("backend", "auto")),
        with_arm=cfg.arm is not None,
    )
    session.own(f"{base.name} disconnected", base.disconnect)
    base.connect()
    base.stand()

    arm = gripper = None
    if cfg.arm is not None:
        from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
        from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper

        arm = MuJoCoPiper(base, cfg.arm)
        session.own(f"{arm.name} disconnected", arm.disconnect)
        arm.connect()
        gripper = MuJoCoPiperGripper(base, cfg.arm.gripper)
        session.own("gripper disconnected", gripper.disconnect)
        gripper.connect()

    user_cfg = _user_config()
    api_key = vlm_api_key or user_cfg.get("llm", {}).get("api_key") or os.environ.get(
        "OPENROUTER_API_KEY", ""
    )
    agent = Agent(base=base, arm=arm, gripper=gripper, llm_api_key=api_key, config=user_cfg)
    for skill in get_go2_skills():
        agent._skill_registry.register(skill)
    agent._perception = _build_go2_perception(base, width=width, height=height)
    agent._calibration = None
    if arm is not None:
        from vector_os_nano.skills.manipulation_setup import register_manipulation_skills

        register_manipulation_skills(agent)

    if api_key:
        try:
            from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

            agent._vlm = Go2VLMPerception(config={"api_key": api_key})
        except Exception as exc:  # noqa: BLE001 — VLM is optional
            logger.info("[sim] Go2VLMPerception unavailable: %s", exc)
            agent._vlm = None

    # Spatial memory is keyed by WORLD (the room scene), shared by every body
    # that stands in it, never mixed across worlds.
    sg_path = Path.home() / ".vector_os_nano" / _ROOM_XML.stem / "scene_graph.yaml"
    sg_path.parent.mkdir(parents=True, exist_ok=True)
    agent._spatial_memory = SceneGraph(persist_path=str(sg_path))
    agent._spatial_memory.load()

    if cfg.nav.get("explorer") == "tare":
        if not os.environ.get("VECTOR_NAV_STACK"):
            session.notes.append("explore: coverage (VECTOR_NAV_STACK unset)")
        else:
            try:
                session.own("nav stack stopped", _launch_ros2_stack(base, gui=spec.gui))
                session.notes.append("explore: tare (nav stack launched)")
            except Exception as exc:  # noqa: BLE001 — optional capability
                session.notes.append(f"explore: coverage (TARE unavailable: {exc})")
    return agent


def _build_mujoco_arm(session: SimSession, vlm_api_key: str | None) -> Any:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.skills.pick import SIM_PICK_CONFIG

    arm = MuJoCoArm(gui=session.spec.gui)
    session.own(f"{arm.name} disconnected", arm.disconnect)
    arm.connect()
    return Agent(
        arm=arm,
        gripper=MuJoCoGripper(arm),
        perception=MuJoCoPerception(arm),
        config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
    )


def _build_mujoco_g1(session: SimSession, vlm_api_key: str | None) -> Any:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_g1 import MuJoCoG1
    from vector_os_nano.perception.g1_head_perception import G1HeadPerception

    width, height = _camera_size(session.config)
    g1 = MuJoCoG1(gui=session.spec.gui, room=True)
    session.own("MuJoCoG1 disconnected", g1.disconnect)
    g1.connect()
    agent = Agent(base=g1)
    # Read-only head-camera perception for the detector capability (no arm).
    agent._perception = G1HeadPerception(g1, width=width, height=height)
    return agent


_BUILDERS: dict[str, Callable[[SimSession, str | None], Any]] = {
    "mujoco_go2": _build_mujoco_go2,
    "mujoco_arm": _build_mujoco_arm,
    "mujoco_g1": _build_mujoco_g1,
}


# ---------------------------------------------------------------------------
# Optional TARE capability: bridge + CMU nav stack (moved from cli; WP13 rewrites)
# ---------------------------------------------------------------------------


def _session_pgids(sids: set[int]) -> set[int]:
    """Process groups of every live process in the given sessions.

    The nav script runs its nodes under ``set -m`` (one process group each), so
    killing the script's own group orphans them; they all keep the script's SID.
    """
    pgids: set[int] = set()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                f = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(f[3]) in sids and f[0] != "Z":
                pgids.add(int(f[2]))
    return pgids


def _kill_sessions(sids: set[int], grace_s: float = 5.0) -> None:
    """SIGTERM, then SIGKILL, every process group in sessions we created."""
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for g in _session_pgids(sids):
            try:
                os.killpg(g, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + grace_s
        while _session_pgids(sids) and time.monotonic() < deadline:
            time.sleep(0.1)


def _launch_ros2_stack(go2: Any, *, gui: bool) -> Callable[[], None]:
    """Start the ROS2 bridge (thread) + nav stack (its own session).

    Returns the cleanup the session owns: stops the monitor and kills every
    process group in the sessions this call created (nothing else), closes the log.
    """
    import importlib.util
    import subprocess
    import sys
    import threading

    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError("ROS2 (rclpy) not available") from exc
    if not rclpy.ok():
        rclpy.init()
    spec = importlib.util.spec_from_file_location(
        "_vnav_bridge", _REPO / "scripts" / "go2_vnav_bridge.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vnav_bridge"] = mod
    spec.loader.exec_module(mod)
    node = mod.Go2VNavBridge(go2)

    def _spin() -> None:
        try:
            rclpy.spin(node)
        except Exception:  # noqa: BLE001
            pass

    spin_t = threading.Thread(target=_spin, daemon=True, name="vnav_bridge_spin")
    spin_t.start()

    argv = [str(_REPO / "scripts" / "launch_nav_only.sh")] + ([] if gui else ["--no-rviz"])
    log_path = Path(run_path("vector_nav_only.log"))
    if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
        log_path.write_text("")  # bound disk use
    log_fh = open(log_path, "a")
    sids: set[int] = set()  # every session we started (each Popen leads one)

    def _spawn() -> Any:
        # stdin from /dev/null: a node must never hold the REPL's terminal open.
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log_fh,
                             stderr=subprocess.STDOUT, start_new_session=True)
        sids.add(p.pid)
        return p

    proc_ref = [_spawn()]
    stop_evt = threading.Event()

    def _monitor() -> None:
        while not stop_evt.wait(5.0):
            if proc_ref[0].poll() is not None:
                logger.warning("nav stack exited (%s), restarting", proc_ref[0].returncode)
                proc_ref[0] = _spawn()

    monitor = threading.Thread(target=_monitor, daemon=True, name="nav_stack_monitor")
    monitor.start()

    def _cleanup() -> None:
        stop_evt.set()
        monitor.join(timeout=6.0)
        _kill_sessions(sids)
        proc_ref[0].poll()  # reap
        log_fh.close()
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        spin_t.join(timeout=2.0)

    return _cleanup


# ---------------------------------------------------------------------------
# The ONE world-rebind path
# ---------------------------------------------------------------------------


def rebind_world(app_state: dict[str, Any], session: SimSession | None) -> None:
    """Point the REPL at *session* (or at none): tools, world, prompt and VGG.

    Used by start/stop_simulation (the CLI startup builds the same state). An
    active playground scenario keeps its world; otherwise the agent picks it.
    """
    from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
    from vector_os_nano.vcli.prompt import build_system_prompt
    from vector_os_nano.vcli.robot_context import RobotContextProvider
    from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
    from vector_os_nano.vcli.worlds import resolve_world

    agent = session.agent if session is not None else None
    base = getattr(agent, "_base", None)
    arm = getattr(agent, "_arm", None)
    sg = getattr(agent, "_spatial_memory", None)
    skill_registry = getattr(agent, "_skill_registry", None)
    app_state.update(sim_session=session, agent=agent, scene_graph=sg,
                     skill_registry=skill_registry)

    registry = app_state.get("registry")
    if registry is not None:
        for name in list(registry.list_tools()):
            if getattr(registry.get(name), "_is_skill_wrapper", False):
                registry.unregister(name)
        if agent is not None:
            for t in wrap_skills(agent):
                registry.register(t, category="robot")
        for cat in ("robot", "diag", "system"):
            if agent is not None:
                registry.enable_category(cat)
            else:
                registry.disable_category(cat)

    world = app_state.get("world") if app_state.get("scenario") else resolve_world(agent)
    app_state["world"] = world

    engine = app_state.get("engine")
    if engine is None:
        return
    provider = RobotContextProvider(base=base, scene_graph=sg, arm=arm)
    app_state["robot_ctx_provider"] = provider
    engine._system_prompt = DynamicSystemPrompt(
        build_system_prompt(agent=agent, cwd=Path.cwd(), robot_context=provider, world=world),
        provider,
    )
    # The native path reads engine._vgg_agent: never leave it on a torn-down body,
    # even if init_vgg below bails out early.
    engine._vgg_agent = agent
    try:
        engine.init_vgg(
            agent=agent,
            skill_registry=skill_registry,
            on_vgg_step=app_state.get("vgg_step_callback"),
            on_vgg_step_view=app_state.get("vgg_step_view_callback"),
            world=world,
            tool_permission_resolver=app_state.get("tool_permission_resolver"),
            persist_dir=(Path.home() / ".vector") if not world.is_robot() else None,
        )
    except Exception as exc:  # noqa: BLE001 — the REPL keeps the tool_use path
        logger.warning("init_vgg after sim rebind failed: %s", exc)
