# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimSession — the ONE in-process simulation behind NL ``start_simulation``.

``start_simulation(embodiment=...)`` and the ``--sim`` / ``--sim-go2`` /
``--embodiment`` flags all call :meth:`SimSession.start`. The body is built from the
embodiment manifest (``embodiments/<id>/robot.yaml``) — never from an env var: a
legged body on the generic LeggedBase + Lidar (:func:`legged_body`), its arm on the
generic Arm / Gripper / Camera. The world side spawns it (:func:`standing_state`) and
stands it up again on ``/reset`` (:func:`respawn`).

* ``start`` fails loud with :class:`SimStartError` (reason + detail) and tears
  down whatever it built; it returns only after the readiness probe passed
  (sim time advancing, a non-black camera frame, finite arm joints).
* ``stop`` is idempotent: it disconnects the devices, closes the runtime (the
  renderers and the viewer close with the world), kills only the process groups this
  session spawned and unregisters its atexit hook.
* ``vcli.runtime.bind_body`` is the ONE path that points the REPL at a session (or
  at no session): the engine binding (agent + world) and the skill tools.
"""
from __future__ import annotations

import atexit
import itertools
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
# One runtime tick (sim s) of a body without a controller of its own (the generic arm).
_TICK_S = 0.005


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
_UIDS = itertools.count(1)


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
    surfaces and user location (everything scripts/gt_grade.py needs, no oracle code)."""
    import json

    gt, world = s.gt, s.world
    held = gt.attachments()
    line = {
        "phase": phase, "turn": s._turns, "goal": goal, "embodiment": s.config.id,
        "world": s.config.world, "sim_time": gt.sim_time, "base": gt.base_pose(),
        "objects": {n: {"pos": p, "welded": n in held} for n, p in gt.object_positions().items()},
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


def sim_id(agent: Any) -> str:
    """The embodiment id of the live SimSession whose body is *agent*, else ''."""
    return getattr(getattr(session_for(agent), "spec", None), "embodiment", "")


class SimSession:
    """One running simulation: the Agent plus everything it owns."""

    def __init__(self, spec: SimSpec, config: Any) -> None:
        self.spec = spec
        self.config = config
        self.uid = next(_UIDS)  # stamps every causation baseline taken in this session
        self.agent: Any = None
        self.notes: list[str] = []
        self.ready: dict[str, Any] = {}
        self.startup_s: float = 0.0
        self._cleanups: list[tuple[str, Callable[[], Any]]] = []
        self._stopped = False
        self.runtime: Any = None  # the SimRuntime the builder made (world side only)
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
        from vector_os_nano.embodiments.assets import AssetsMissing, asset_dir

        try:
            asset_dir(cfg)
        except AssetsMissing as exc:
            raise SimStartError("assets not provisioned", str(exc)) from exc
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
            session.ready = _probe_ready(session.agent, cfg, session.runtime, deadline=t0 + deadline_s)
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
        from vector_os_nano.sim.ground_truth import GroundTruth
        from vector_os_nano.vcli.worlds.world_manifest import load_world

        agent = self.agent
        if self.runtime is None:  # no world owner: no oracles, every verify grades RAN
            self.notes.append("ground truth: unavailable (the builder made no SimRuntime)")
            return
        self.gt = GroundTruth(self.runtime)
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
        (scripts/gt_grade.py)."""
        if self.gt is None:
            return
        if phase == "start":
            self._turns += 1
            self._turn = (self.gt.sim_time, time.time())
        path = os.environ.get("VECTOR_GT_DUMP", "")
        if path:
            _append_gt_dump(path, self, phase, goal)

    def reset_pose(self) -> None:
        """Stand the legged body up where it is (sim-only recovery: ``/reset``)."""
        if self.config.base is None or self.runtime is None:
            raise RuntimeError(f"{self.config.id} has no legged base to stand up")
        respawn(self.runtime, self.agent._base, self.config)

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


def _probe_ready(agent: Any, cfg: Any, runtime: Any, *, deadline: float) -> dict[str, Any]:
    """Prove the sim is live, or raise SimStartError naming what is not."""
    import numpy as np

    base = getattr(agent, "_base", None)
    arm = getattr(agent, "_arm", None)
    report: dict[str, Any] = {}

    if runtime is not None and not runtime.threaded:
        # Caller-stepped (VECTOR_SIM_LOCKSTEP / macOS pump): advance it here, through
        # the base when it has one (a policy that ticks outside the runtime, G1's).
        t0 = runtime.sim_time
        advance = getattr(base, "wait_sim", None) or runtime.wait_sim
        if not advance(_READY_SIM_S) or runtime.sim_time - t0 < _READY_SIM_S - 1e-6:
            raise SimStartError("not ready", "caller-stepped physics did not advance")
        report["sim_time"] = f"{runtime.sim_time:.1f}s"
    elif runtime is not None:
        t0 = last = runtime.sim_time
        moved_at = time.monotonic()
        while (now := runtime.sim_time) - t0 < _READY_SIM_S:
            if now != last:
                last, moved_at = now, time.monotonic()
            if time.monotonic() - moved_at > _STALL_S or time.monotonic() > deadline:
                raise SimStartError(
                    "not ready",
                    f"sim time advanced {now - t0:.2f}s (< {_READY_SIM_S}s): physics is not stepping",
                )
            time.sleep(0.05)
        report["sim_time"] = f"{runtime.sim_time:.1f}s"
    if getattr(base, "gait_backend", ""):
        report["gait"] = base.gait_backend

    if cfg.capabilities.camera:
        perception = getattr(agent, "_perception", None)
        capture = getattr(perception, "capture", None)
        grab = (lambda: capture().rgb) if callable(capture) else getattr(perception, "get_color_frame", None)
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


def _user_config() -> dict[str, Any]:
    from vector_os_nano.core.config import load_config

    path = _REPO / "config" / "user.yaml"
    return load_config(str(path)) if path.exists() else {}


def _devices(cfg: Any, rt: Any, *, wait_mount: Callable[[], Any] | None = None) -> tuple[Any, Any, Any]:
    """``(arm, gripper, camera)`` of *cfg* on *rt*: the generic drivers, None where
    the manifest has no such block."""
    from vector_os_nano.sim.drivers.arm import Arm
    from vector_os_nano.sim.drivers.camera import Camera
    from vector_os_nano.sim.drivers.gripper import Gripper
    from vector_os_nano.vcli.cognitive.abort import is_abort_requested

    from vector_os_nano.sim.scene.compose import check_fingers

    camera = Camera(cfg.camera, rt) if cfg.camera is not None else None
    if cfg.arm is None:
        return None, None, camera
    arm = Arm(cfg.arm, rt, abort=is_abort_requested, wait_mount=wait_mount)
    if cfg.arm.gripper is None:
        return arm, None, camera
    check_fingers(rt.world.scene, cfg.arm.gripper.finger_bodies)
    return arm, Gripper(cfg.arm.gripper, rt), camera


def _perception(camera: Any) -> Any:
    """The ONE perception over the manifest's camera: built once per session and shared by
    the grasp and look/localize skills, so they always see the same camera size."""
    from vector_os_nano.perception.camera_perception import CameraPerception

    return CameraPerception(camera) if camera is not None else None


def _perception_note(agent: Any, vlm_why: str = "", *, segmenter: bool = True) -> str:
    """The startup capability line: which models perception will use, or why not."""
    from vector_os_nano.perception.rynnbrain import RynnBrainDetector

    import importlib.util

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        detector = seg = "unavailable (torch/transformers not installed: uv pip install -e '.[all]')"
    else:
        from vector_os_nano.perception.grounding_dino import GroundingDinoDetector
        from vector_os_nano.perception.tracker import EdgeTAMTracker
        from vector_os_nano.perception.weights import status

        # ponytail: the device the detector WILL use, without importing torch at sim start
        # (that inits CUDA and spawns ldconfig); the load logs the real one ("[GDINO] loading").
        visible = os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
        device = "cuda" if visible and Path("/proc/driver/nvidia/version").exists() else "cpu"
        detector = f"grounding-dino@{device} ({status(GroundingDinoDetector.WEIGHTS)})"
        seg = f"edgetam ({status(EdgeTAMTracker.WEIGHTS)})"
    perception, vlm = getattr(agent, "_perception", None), getattr(agent, "_vlm", None)
    vlm_s = f"none ({vlm_why})" if vlm is None else f"{'local' if vlm._local else 'openrouter'}:{vlm._model}"
    return (
        f"perception: detector={detector}, "
        + (f"segmenter={seg}, self_mask={'yes' if callable(getattr(perception, 'capture', None)) else 'no'}, "
           if segmenter else "")
        + f"vlm={vlm_s}, rynnbrain={'up' if RynnBrainDetector().available() else 'down'}"
    )


def _skill_pack(embodiment: str) -> list:
    """The skills a body is offered: arm skills only on an arm, base skills only on a
    base, a stop on every body (the table S5 replaces with capability binding)."""
    from vector_os_nano.skills import HomeSkill, get_piper_skills, get_so101_skills
    from vector_os_nano.skills.go2 import get_go2_skills
    from vector_os_nano.skills.go2.stop import StopSkill

    return {
        "so101": get_so101_skills,
        "piper": lambda: [StopSkill(), HomeSkill(), *get_piper_skills()],
        "go2": get_go2_skills,
        "go2_piper": lambda: [*get_go2_skills(), *get_piper_skills()],
        "g1": lambda: [StopSkill()],  # its locomotion pack is S4's
    }[embodiment]()


def standing_state(scene: Any, cfg: Any, q: Any, xy: Any, heading: float) -> Any:
    """*q* with the body standing: its root at (*xy*, ``base.height``) facing *heading*,
    its base joints in the stance, its arm (if any) at home."""
    import numpy as np

    q = np.array(q, dtype=np.float64)
    q[scene.free_q(scene.robot_root)] = [xy[0], xy[1], cfg.base.height,
                                         math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    for name, angle in cfg.base.stance.items():
        q[scene.joint_q(scene.joint(name))] = angle
    for name, angle in zip(cfg.arm.joints, cfg.arm.home) if cfg.arm else ():
        q[scene.joint_q(scene.joint(name))] = angle
    return q


def legged_body(cfg: Any, scene: Any, *, backend: str = "mujoco", gui: bool = False) -> tuple[Any, Any, Any]:
    """``(runtime, base, lidar)``: the legged body of *cfg* loaded on *backend* in
    *scene*, spawned standing at its spawn, with ONE SimRuntime ticking at its policy's
    rate (not started yet), the generic LeggedBase connected and its manifest's Lidar
    (None without one). A manifest joint the compiled scene lacks fails first, naming
    the manifest file and its key."""
    from vector_os_nano.embodiments.config import EmbodimentConfigError
    from vector_os_nano.sim import backends
    from vector_os_nano.sim.drivers.base import LeggedBase
    from vector_os_nano.sim.drivers.lidar import Lidar
    from vector_os_nano.sim.runtime import SimRuntime
    from vector_os_nano.vcli.cognitive.abort import is_abort_requested

    for key, names in (("base.joints", cfg.base.joints), ("arm.joints", cfg.arm.joints if cfg.arm else ())):
        missing = [n for n in names if n not in scene.joint_ids]
        if missing:
            raise EmbodimentConfigError(f"{cfg.id}/robot.yaml: {key} {missing} are not joints of "
                                        f"{cfg.model.path}; joints: {sorted(scene.joint_ids)}")
    world = backends.get(backend).load(scene, gui=gui)
    rt = SimRuntime(world, substeps=round(1.0 / (cfg.base.policy.tick_hz * scene.timestep)),
                    threaded=not world.caller_thread_only, name=cfg.id)
    try:
        spawn = standing_state(scene, cfg, scene.model.qpos0, cfg.spawn.xy, cfg.spawn.heading)
        rt.call(lambda w: w.reset(spawn))
        lidar = Lidar(cfg.lidar, rt) if cfg.lidar is not None else None
        base = LeggedBase(cfg.base, rt, lidar=lidar, abort=is_abort_requested)
        base.connect()
    except BaseException:
        rt.close()
        raise
    return rt, base, lidar


def respawn(rt: Any, base: Any, cfg: Any, xy: Any = None, heading: float | None = None) -> None:
    """Stand the body up again (sim-only recovery: the ``/reset`` command) at *xy*
    facing *heading*, by default where it is: ONE step-boundary job writes the standing
    state (``world.reset``: velocities and torques zeroed), points the arm's servos at
    home and restarts the base."""
    scene = rt.world.scene

    def job(w: Any) -> None:
        q = w.state()[0]
        x, y, z, qw, qx, qy, qz = q[scene.free_q(scene.robot_root)]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        w.reset(standing_state(scene, cfg, q, (x, y) if xy is None else xy, yaw if heading is None else heading))
        if cfg.arm is not None:  # the arm's servos hold it at home
            w.set_targets([scene.joint(j) for j in cfg.arm.joints], cfg.arm.home)
        base.respawned()

    rt.call(job)


def _build_legged(session: SimSession, vlm_api_key: str | None) -> Any:
    """A legged body (go2, go2_piper): the apartment room with the body attached, ONE
    SimRuntime, the generic LeggedBase + Lidar, and the generic Arm / Gripper / Camera
    of its manifest."""
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.hardware.sim.scene_builder import room_scene
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    cfg, spec = session.config, session.spec
    world = load_world(cfg.world)
    rt, base, lidar = legged_body(cfg, room_scene(cfg, world), gui=spec.gui)
    session.own(f"{cfg.id} runtime closed", rt.close)
    session.own("base disconnected", base.disconnect)
    rt.start()
    base.set_posture("stand")
    session.runtime = rt
    arm, gripper, camera = _devices(cfg, rt, wait_mount=base.wait_settled)

    user_cfg = _user_config()
    api_key = vlm_api_key or user_cfg.get("llm", {}).get("api_key") or os.environ.get(
        "OPENROUTER_API_KEY", ""
    )
    agent = Agent(base=base, arm=arm, gripper=gripper, config=user_cfg, skills=_skill_pack(cfg.id))
    agent._perception = _perception(camera)
    agent._calibration = None

    # The scene VLM is optional naming sugar: an OpenRouter key OR a local server
    # (VECTOR_VLM_URL, e.g. Ollama). Discovery never needs it.
    vlm_why = "no OPENROUTER_API_KEY or VECTOR_VLM_URL"
    if api_key or os.environ.get("VECTOR_VLM_URL"):
        try:
            from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

            agent._vlm = Go2VLMPerception(config={
                "api_key": api_key, "rooms": sorted(world.rooms),
                **(user_cfg.get("vlm") or {}),
            })
        except Exception as exc:  # noqa: BLE001 — VLM is optional; the capability line says why
            vlm_why = f"failed: {type(exc).__name__}: {exc}"
            agent._vlm = None

    # Spatial memory is keyed by WORLD (shared by every body that stands in it,
    # never mixed across worlds) under $HOME (a benchmark trial's own HOME isolates
    # it). The world's rooms are seeded from its manifest, fixed at their centres.
    sg_path = Path.home() / ".vector_os_nano" / "worlds" / cfg.world / "scene_graph.yaml"
    sg_path.parent.mkdir(parents=True, exist_ok=True)
    agent._spatial_memory = SceneGraph(persist_path=str(sg_path), resolve=world.resolve)
    agent._spatial_memory.load()
    agent._spatial_memory.load_layout(str(world.path))

    if cfg.base.nav.explorer == "tare":
        if not os.environ.get("VECTOR_NAV_STACK"):
            session.notes.append("explore: coverage (VECTOR_NAV_STACK unset)")
        else:
            try:
                session.own("nav stack stopped", _launch_ros2_stack(
                    base, camera, lidar, session.reset_pose, gui=spec.gui))
                session.notes.append("explore: tare (nav stack launched)")
            except Exception as exc:  # noqa: BLE001 — optional capability
                session.notes.append(f"explore: coverage (TARE unavailable: {exc})")
    session.notes.append(_perception_note(agent, vlm_why))  # last: the explore note stays up front
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
    session.runtime = arm.runtime
    return Agent(
        arm=arm,
        gripper=MuJoCoGripper(arm),
        perception=MuJoCoPerception(arm),
        config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
        skills=_skill_pack(session.config.id),
    )


def _build_mujoco_g1(session: SimSession, vlm_api_key: str | None) -> Any:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_g1 import MuJoCoG1
    from vector_os_nano.perception.g1_head_perception import G1HeadPerception

    cam = session.config.camera
    g1 = MuJoCoG1(gui=session.spec.gui, room=True)
    session.own("MuJoCoG1 disconnected", g1.disconnect)
    g1.connect()
    session.runtime = g1.runtime
    agent = Agent(base=g1, skills=_skill_pack(session.config.id))
    # Read-only head-camera perception for the detector capability (no arm).
    agent._perception = G1HeadPerception(g1, width=cam.width, height=cam.height)
    session.notes.append(_perception_note(agent, "g1 has no scene VLM", segmenter=False))
    return agent


def _build_generic(session: SimSession, vlm_api_key: str | None) -> Any:
    """A body the generic drivers run whole (a fixed arm): its world's scene composed with
    the manifest (sim/scene/compose.py), loaded on MuJoCo, ONE SimRuntime with no
    controller (the servos hold the arm)."""
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.sim import backends
    from vector_os_nano.sim.runtime import SimRuntime
    from vector_os_nano.sim.scene.compose import compose
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    cfg = session.config
    world = load_world(cfg.world)
    if world.scene is None or world.spawn is None:
        raise SimStartError("bad world manifest", f"{world.path}: a fixed body needs the world's scene and spawn")
    scene = compose(world.scene, cfg, spawn=world.spawn, pickables=tuple(world.pickables))
    sim = backends.get("mujoco").load(scene, gui=session.spec.gui)
    rt = SimRuntime(sim, substeps=max(1, round(_TICK_S / scene.timestep)), threaded=not sim.caller_thread_only,
                    name=cfg.id)
    session.own(f"{cfg.id} runtime closed", rt.close)
    rt.start()
    session.runtime = rt
    arm, gripper, camera = _devices(cfg, rt)
    agent = Agent(arm=arm, gripper=gripper, config=_user_config(), skills=_skill_pack(cfg.id))
    agent._perception = _perception(camera)
    session.notes.append(_perception_note(agent, "a fixed arm has no scene VLM"))
    return agent


_BUILDERS: dict[str, Callable[[SimSession, str | None], Any]] = {
    "generic": _build_generic,
    "legged": _build_legged,
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


def _launch_ros2_stack(base: Any, camera: Any, lidar: Any, reset: Callable[[], None], *,
                       gui: bool) -> Callable[[], None]:
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
    node = mod.Go2VNavBridge(base, camera, lidar, reset)

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
