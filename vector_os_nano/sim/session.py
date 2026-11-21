# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimSession — the ONE in-process simulation behind NL ``start_simulation``.

CLI flags and the natural-language tool call SimSession.start with one selected
robot, backend and world. Manifests supply devices, controllers and reset poses.

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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable
from vector_os_nano.core.run_dir import run_path
from vector_os_nano.schemas import paths
from vector_os_nano.schemas.world import WorldManifest

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

    embodiment: str = ""
    gui: bool | None = None
    backend: str | None = None
    world: str | None = None

    def resolved(self) -> "SimSpec":
        """Explicit choices, then user defaults, then the robot's default world."""
        from vector_os_nano.embodiments import registry
        from vector_os_nano.schemas.user import load

        try:
            defaults = load(paths.config_file()).defaults
            if not isinstance(self.embodiment, str):
                raise ValueError("embodiment must be a string")
            embodiment = self.embodiment or defaults.embodiment
            gui = defaults.gui if self.gui is None else self.gui
            backend = defaults.backend if self.backend is None else self.backend
            world = self.world if self.world is not None else defaults.world
            if type(gui) is not bool:
                raise ValueError("gui must be true or false")
            for key, value in (("embodiment", embodiment), ("backend", backend), ("world", world)):
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{key} must be a non-empty string")
            cfg = registry.load(embodiment)
            return SimSpec(embodiment, gui, backend, world or cfg.world)
        except ValueError as exc:
            raise SimStartError("bad simulation selection", str(exc)) from exc


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
    grip = s.config.arm.gripper if s.config.arm else None
    pure_contact = grip is not None and not grip.assist
    line = {
        "schema": 1, "backend": s.runtime.world.name, "session": s.uid,
        "phase": phase, "turn": s._turns, "goal": goal, "embodiment": s.config.id,
        "world": world.id, "sim_time": gt.sim_time, "base": gt.base_pose(),
        "arm": ({"joints": list(s.config.arm.joints), "q": gt.joint_positions(s.config.arm.joints),
                 "home": list(s.config.arm.home)} if s.config.arm else None),
        "joint_groups": {n: {"joints": g.joints, "q": gt.joint_positions(g.joints)}
                         for n, g in s.config.joint_groups.items()},
        "grasp_mode": 'contact' if pure_contact else 'assisted',
        "objects": {n: {"pos": p, "welded": n in held,
                        "contact_held": bool(pure_contact and gt.contact_held(n, grip.weld_body))}
                    for n, p in gt.object_positions().items()},
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
        """Resolve the manifest + builder without starting anything (fail loud, valid set).

        Callers that replace a running session validate FIRST, so a typo never
        tears down the sim that is already up.
        """
        from vector_os_nano.embodiments import registry
        from vector_os_nano.schemas.robot import EmbodimentConfigError

        spec = spec.resolved()
        if not spec.gui:
            import sys

            os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

        try:
            cfg = registry.load(spec.embodiment)
        except EmbodimentConfigError as exc:
            raise SimStartError("bad embodiment manifest", str(exc)) from exc
        from vector_os_nano.embodiments.assets import AssetsMissing, asset_dir
        from vector_os_nano.schemas.world import load_world
        from vector_os_nano.sim import backends

        try:
            asset_dir(cfg)
        except AssetsMissing as exc:
            raise SimStartError("assets not provisioned", str(exc)) from exc
        try:
            world = load_world(spec.world)
            reason = backends.get(spec.backend).supports(cfg, world)
        except (KeyError, ValueError) as exc:
            raise SimStartError("bad simulation selection", str(exc)) from exc
        if reason:
            raise SimStartError("unsupported simulation", reason)
        return cfg, _build

    @classmethod
    def start(
        cls, spec: SimSpec, *, vlm_api_key: str | None = None, deadline_s: float = 90.0,
        abort: Callable[[], bool] = lambda: False,
        cancel: Callable[[], bool] | None = None,
    ) -> "SimSession":
        t0 = time.monotonic()
        cancelled = cancel or abort
        _check_start(t0 + deadline_s, cancelled)
        paths.configure_cuda_cache()
        spec = spec.resolved()
        cfg, builder = cls.validate(spec)
        session = cls(spec, cfg)
        session.abort = abort
        try:
            _check_start(t0 + deadline_s, cancelled)
            session.agent = builder(session, vlm_api_key)
            session.ready = _probe_ready(session.agent, cfg, session.runtime,
                                         deadline=t0 + deadline_s, abort=cancelled)
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
        from vector_os_nano.schemas.world import load_world

        agent = self.agent
        if self.runtime is None:  # no world owner: no oracles, every verify grades RAN
            self.notes.append("ground truth: unavailable (the builder made no SimRuntime)")
            return
        if isinstance(self.agent._config, dict):
            self.agent._config["joint_groups"] = {n: {"postures": g.postures} for n, g in self.config.joint_groups.items()}
            self.agent._config["skills"] = {**self.agent._config.get("skills", {}), **self.config.skills}
            if self.config.arm is not None:
                self.agent._config["arm"] = {"postures": self.config.arm.postures}
        self.gt = GroundTruth(self.runtime)
        self.world = load_world(self.spec.world or self.config.world)
        agent._services["object_world"] = self.world
        # Skills resolve object names through the world manifest by id (never GT).
        cfg = getattr(agent, "_config", None)
        if isinstance(cfg, dict) and self.config.world:
            cfg["world"] = self.world.id

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
        respawn(self.runtime, self.agent._base, self.config, floor_z=self.world.spawn[2])

    def own(self, what: str, cleanup: Callable[[], Any]) -> None:
        """Register a teardown step (run LIFO by stop)."""
        self._cleanups.append((what, cleanup))

    def metadata(self) -> dict[str, Any]:
        """Session provenance only; no evaluator state is exposed to tools."""
        from importlib.metadata import version

        backend = self.runtime.world.name if self.runtime is not None else self.spec.backend
        return {"embodiment": self.config.id, "backend": backend, "session": self.uid,
                "world": self.world.id if self.world else self.spec.world,
                "backend_version": version(backend) if backend else "",
                "perception_kind": type(self.agent._perception).__name__ if getattr(self.agent, "_perception", None) else "none"}

    def summary(self) -> str:
        """One line: 'ready in Xs: <hardware>, <probe>, N skills; <notes>'."""
        agent = self.agent
        hw = " + ".join(
            type(d).__name__ for d in (getattr(agent, "_base", None), getattr(agent, "_arm", None)) if d
        )
        n = len(agent._skill_registry.list_skills()) if agent is not None else 0
        probe = ", ".join(f"{k}={v}" for k, v in self.ready.items())
        caps = ",".join(f.name for f in fields(self.config.capabilities) if getattr(self.config.capabilities, f.name))
        policy = self.config.base.policy.on(self.spec.backend)[0] if self.config.base else "position_actuators"
        line = (f"ready in {self.startup_s:.1f}s: embodiment={self.spec.embodiment}, "
                f"backend={self.spec.backend}, world={self.spec.world}, policy={policy}, "
                f"capabilities=[{caps}]; {hw}, {probe}, {n} skills")
        return "; ".join([line, *self.notes])


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def _check_start(deadline: float, abort: Callable[[], bool]) -> None:
    if abort():
        raise SimStartError("start cancelled", "stop requested")
    if time.monotonic() > deadline:
        raise SimStartError("not ready", "startup deadline exceeded")


def _probe_ready(agent: Any, cfg: Any, runtime: Any, *, deadline: float,
                 abort: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Prove the sim is live, or raise SimStartError naming what is not."""
    import numpy as np

    base = getattr(agent, "_base", None)
    arm = getattr(agent, "_arm", None)
    report: dict[str, Any] = {}
    _check_start(deadline, abort)
    if runtime is None:
        raise SimStartError("not ready", "no physics runtime")

    if runtime is not None and not runtime.threaded:
        # Caller-stepped (VECTOR_SIM_LOCKSTEP / macOS pump): advance it here.
        t0 = runtime.sim_time
        from types import SimpleNamespace

        if (not runtime.wait_sim(_READY_SIM_S, SimpleNamespace(is_set=abort),
                                 wall_timeout=max(0.0, deadline - time.monotonic()))
                or runtime.sim_time - t0 < _READY_SIM_S - 1e-6):
            _check_start(deadline, abort)
            raise SimStartError("not ready", "caller-stepped physics did not advance")
        report["sim_time"] = f"{runtime.sim_time:.1f}s"
    elif runtime is not None:
        t0 = last = runtime.sim_time
        moved_at = time.monotonic()
        while (now := runtime.sim_time) - t0 < _READY_SIM_S:
            if abort():
                raise SimStartError("start cancelled", "stop requested")
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
                if callable(capture):
                    sample = ex.submit(capture).result()
                    frame = sample.rgb
                    if cfg.capabilities.depth:
                        depth = sample.depth
                        if (depth is None or np.shape(depth) != np.shape(frame)[:2]
                                or not np.isfinite(depth).all() or not (np.asarray(depth) > 0).any()):
                            raise SimStartError("not ready", "depth frame is missing or invalid")
                        report["depth"] = f"{np.shape(depth)[1]}x{np.shape(depth)[0]}"
                elif cfg.capabilities.depth:
                    raise SimStartError("not ready", "declared depth sensor has no capture method")
                else:
                    frame = ex.submit(grab).result()
        if frame is None or not np.asarray(frame).any():
            raise SimStartError("not ready", "camera frame is empty/black")
        h, w = np.asarray(frame).shape[:2]
        report["camera"] = f"{w}x{h}"

    if cfg.capabilities.has_arm:
        if arm is None:
            raise SimStartError("not ready", "declared arm is missing")
        q = list(arm.get_joint_positions())
        if not q or not all(math.isfinite(v) for v in q):
            raise SimStartError("not ready", f"arm joints not finite: {q}")
        report["arm"] = f"{len(q)}-DoF"
    if cfg.capabilities.lidar:
        lidar = agent._services.get("lidar")
        scan = lidar.scan() if lidar is not None else None
        ranges = np.asarray(scan.ranges) if scan is not None else np.array([])
        if not ranges.size or np.isnan(ranges).any() or (ranges < 0).any():
            raise SimStartError("not ready", "lidar scan is missing or invalid")
        report["lidar"] = f"{ranges.size} beams"
    _check_start(deadline, abort)
    return report


# ---------------------------------------------------------------------------
# Builders: a body with a ``base:`` block is legged, any other a fixed arm
# ---------------------------------------------------------------------------


def _user_config() -> dict[str, Any]:
    from vector_os_nano.core.config import load_config

    path = _REPO / "config" / "user.yaml"
    return load_config(str(path)) if path.exists() else {}


def _scene_vlm_config(user_cfg: dict[str, Any], vlm_api_key: str | None) -> tuple[str, dict[str, Any], str]:
    """The scene VLM's key, its config/user.yaml ``vlm:`` block and one notice.

    The key comes from the environment like every provider's: a key left in
    config/user.yaml (``llm.api_key``, ``vlm.api_key``) is never read nor sent, and the
    notice names the variable to set instead ('' when the file holds none).
    """
    vlm_cfg = dict(user_cfg.get("vlm") or {})
    stray = [f"{block}.api_key" for block, d in (("llm", user_cfg.get("llm")), ("vlm", vlm_cfg))
             if isinstance(d, dict) and d.get("api_key")]
    vlm_cfg.pop("api_key", None)
    note = (f"config/user.yaml {' and '.join(stray)} ignored (keys come only from the environment): "
            "set OPENROUTER_API_KEY") if stray else ""
    return vlm_api_key or os.environ.get("OPENROUTER_API_KEY", ""), vlm_cfg, note


def _perception(camera: Any, cameras: dict | None = None) -> Any:
    """The ONE perception over the manifest's camera: built once per session and shared by
    the grasp and look/localize skills, so they always see the same camera size."""
    from vector_os_nano.perception.camera_perception import CameraPerception

    return CameraPerception(camera, cameras=cameras) if camera is not None else None


def _perception_note(agent: Any, vlm_why: str = "") -> str:
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
        f"perception: detector={detector}, segmenter={seg}, "
        f"self_mask={'yes' if callable(getattr(perception, 'capture', None)) else 'no'}, "
        f"vlm={vlm_s}, rynnbrain={'up' if RynnBrainDetector().available() else 'down'}"
    )


def standing_state(scene: Any, cfg: Any, q: Any, xy: Any, heading: float, *, floor_z: float = 0.0) -> Any:
    """*q* with the body standing: its root at (*xy*, ``base.height``) facing *heading*,
    its base joints in the stance, its arm (if any) at home."""
    import numpy as np

    q = np.array(q, dtype=np.float64)
    q[scene.free_q(scene.robot_root)] = [xy[0], xy[1], floor_z + cfg.base.height,
                                         math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    for name, angle in cfg.base.stance.items():
        q[scene.joint_q(scene.joint(name))] = angle
    for name, angle in zip(cfg.arm.joints, cfg.arm.home) if cfg.arm else ():
        q[scene.joint_q(scene.joint(name))] = angle
    return q


def legged_body(cfg: Any, scene: Any, *, manifest: WorldManifest, backend: str = "mujoco", gui: bool = False,
                abort: Callable[[], bool] = lambda: False) -> tuple[Any, Any, Any]:
    """``(runtime, base, lidar)``: the legged body of *cfg* loaded on *backend* in
    *scene*, spawned standing at its spawn, with ONE SimRuntime ticking at its policy's
    rate (not started yet), the generic LeggedBase connected and its manifest's Lidar
    (None without one). A manifest joint the compiled scene lacks fails first, naming
    the manifest file and its key."""
    from vector_os_nano.embodiments.assets import policy_root
    from vector_os_nano.schemas.robot import EmbodimentConfigError
    from vector_os_nano.sim import backends
    from vector_os_nano.sim.drivers.base import LeggedBase
    from vector_os_nano.sim.drivers.lidar import Lidar
    from vector_os_nano.sim.runtime import SimRuntime

    for key, names in (("base.joints", cfg.base.joints), ("arm.joints", cfg.arm.joints if cfg.arm else ())):
        missing = [n for n in names if n not in scene.joint_ids]
        if missing:
            raise EmbodimentConfigError(f"{cfg.id}/robot.yaml: {key} {missing} are not joints of "
                                        f"{cfg.model.file}; joints: {sorted(scene.joint_ids)}")
    world = backends.get(backend).load(scene, gui=gui)
    rt = SimRuntime(world, substeps=round(1.0 / (cfg.base.policy.tick_hz * scene.timestep)),
                    threaded=not world.caller_thread_only, name=cfg.id)
    try:
        x, y, z, heading = manifest.spawn
        spawn = standing_state(scene, cfg, scene.model.qpos0, (x, y), heading, floor_z=z)
        rt.call(lambda w: w.reset(spawn))
        lidar = Lidar(cfg.lidar, rt) if cfg.lidar is not None else None
        base = LeggedBase(cfg.base, rt, folder=policy_root(cfg), lidar=lidar, abort=abort)
        base.connect()
    except BaseException:
        rt.close()
        raise
    return rt, base, lidar


def respawn(rt: Any, base: Any, cfg: Any, xy: Any = None, heading: float | None = None, *, floor_z: float = 0.0) -> None:
    """Stand the body up again (sim-only recovery: the ``/reset`` command) at *xy*
    facing *heading*, by default where it is: ONE step-boundary job writes the standing
    state (``world.reset``: velocities and torques zeroed), points the arm's servos at
    home and restarts the base."""
    scene = rt.world.scene

    def job(w: Any) -> None:
        q = w.state()[0]
        x, y, z, qw, qx, qy, qz = q[scene.free_q(scene.robot_root)]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        w.reset(standing_state(scene, cfg, q, (x, y) if xy is None else xy, yaw if heading is None else heading, floor_z=floor_z))
        if cfg.arm is not None:  # the arm's servos hold it at home
            w.set_targets([scene.joint(j) for j in cfg.arm.joints], cfg.arm.home)
        _initial_joints(w, cfg)
        base.respawned()

    rt.call(job)


def _initial_joints(world: Any, cfg: Any) -> None:
    """Reset declared scalar joints and their position servos, including passive mimics."""
    from vector_os_nano.sim.scene import reader

    initial = getattr(cfg, 'initial_joints', {})
    if not initial:
        return
    scene = world.scene
    q, _ = world.state()
    joints = {j.name: j for j in reader.joints(scene.model)}
    for name, value in initial.items():
        j = joints[name]
        if j.type not in (reader.JNT_HINGE, reader.JNT_SLIDE) or (j.range and not j.range[0] <= value <= j.range[1]):
            raise ValueError(f'{cfg.path}.initial_joints.{name}: outside scalar joint limits')
        q[j.qadr] = value
    world.reset(q)
    driven = [a.joint for a in reader.actuators(scene.model) if a.kind == 'position' and scene.model.joint(a.joint).name in initial]
    world.set_targets(driven, [q[scene.joint_q(j)] for j in driven])


def _build(session: SimSession, vlm_api_key: str | None) -> Any:
    """Compose the selected world, start its runtime, and bind manifest components."""
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.skills import get_skills
    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.sim import backends
    from vector_os_nano.sim.drivers.lidar import Lidar
    from vector_os_nano.sim.runtime import SimRuntime
    from vector_os_nano.sim.scene.compose import compose
    from vector_os_nano.schemas.world import load_world

    cfg, spec = session.config, session.spec
    world = session.world = load_world(spec.world)
    scene = compose(cfg, world)
    base = lidar = None
    if cfg.base is not None:
        rt, base, lidar = legged_body(cfg, scene, manifest=world, backend=spec.backend, gui=spec.gui,
                                     abort=session.abort)
    else:
        sim = backends.get(spec.backend).load(scene, gui=spec.gui)
        rt = SimRuntime(sim, substeps=max(1, round(_TICK_S / scene.timestep)),
                        threaded=not sim.caller_thread_only, name=cfg.id)
    session.own(f"{cfg.id} runtime closed", rt.close)
    if base is not None:
        session.own("base disconnected", base.disconnect)
    elif cfg.lidar is not None:
        lidar = Lidar(cfg.lidar, rt)
    session.runtime = rt
    rt.call(lambda w: _initial_joints(w, cfg))
    from vector_os_nano.sim.devices import devices, joint_groups

    groups = joint_groups(cfg, rt, abort=session.abort)
    rt.start()
    if base is not None:
        base.set_posture("stand")
    arm, gripper, camera = devices(cfg, rt, wait_mount=base.wait_settled if base else None,
                                     abort=session.abort)

    user_cfg = _user_config()
    api_key, vlm_cfg, key_note = _scene_vlm_config(user_cfg, vlm_api_key)
    agent = Agent(base=base, arm=arm, gripper=gripper, config=user_cfg, skills=get_skills(cfg.capabilities))
    agent._services["joint_groups"] = groups
    agent._services["lidar"] = lidar
    agent._services["layout"] = {"rooms": world.rooms, "boxes": world.boxes}
    from vector_os_nano.schemas.robot import CameraSpec
    from vector_os_nano.sim.drivers.camera import Camera
    agent._perception = _perception(camera, {s.name: Camera(s, rt) for s in cfg.sensors if isinstance(s, CameraSpec)})
    agent._calibration = None

    # The scene VLM is optional naming sugar: an OpenRouter key OR a local server
    # (VECTOR_VLM_URL, e.g. Ollama). Discovery never needs it.
    vlm_why = "no OPENROUTER_API_KEY or VECTOR_VLM_URL"
    if api_key or os.environ.get("VECTOR_VLM_URL"):
        try:
            from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

            agent._vlm = Go2VLMPerception(config={
                "api_key": api_key, "rooms": sorted(world.rooms),
                **vlm_cfg,
            })
        except Exception as exc:  # noqa: BLE001 — VLM is optional; the capability line says why
            vlm_why = f"failed: {type(exc).__name__}: {exc}"
            agent._vlm = None

    # Spatial memory is keyed by WORLD (shared by every body that stands in it,
    # never mixed across worlds) under $VECTOR_HOME/memory (a benchmark trial's own
    # VECTOR_HOME isolates it). The world's rooms are seeded from its manifest, fixed at
    # their centres. Memory an older release kept in ~/.vector_os_nano is moved in once.
    moved = paths.migrate_old_memory()
    sg_path = paths.memory(world.id) / paths.SCENE_GRAPH
    sg_path.parent.mkdir(parents=True, exist_ok=True)
    agent._spatial_memory = SceneGraph(persist_path=str(sg_path), resolve=world.resolve)
    agent._spatial_memory.load()
    agent._spatial_memory.load_layout(str(world.path))

    if cfg.base is not None and cfg.base.nav.explorer == "tare":
        if not os.environ.get("VECTOR_NAV_STACK"):
            session.notes.append("explore: coverage (VECTOR_NAV_STACK unset)")
        else:
            try:
                session.own("nav stack stopped", _launch_ros2_stack(
                    base, camera, lidar, session.reset_pose, gui=spec.gui,
                    terrain_map=sg_path.with_name(paths.TERRAIN_MAP)))
                session.notes.append("explore: tare (nav stack launched)")
            except Exception as exc:  # noqa: BLE001 — optional capability
                session.notes.append(f"explore: coverage (TARE unavailable: {exc})")
    session.notes.append(_perception_note(agent, vlm_why))  # after it: the explore note stays up front
    if key_note:
        session.notes.append(key_note)
    if moved:
        session.notes.append(moved)
    return agent


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
                       gui: bool, terrain_map: Path) -> Callable[[], None]:
    """Start the ROS2 bridge (thread, persisting the world's *terrain_map*) + nav stack (its own session).

    Returns the cleanup the session owns: stops the monitor and kills every
    process group in the sessions this call created (nothing else), closes the log.
    """
    import subprocess
    import threading

    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError("ROS2 (rclpy) not available") from exc
    if not rclpy.ok():
        rclpy.init()
    from vector_os_nano.sim.nav.ros2.bridge import NavigationBridge
    node = NavigationBridge(base, camera, lidar, reset, terrain_map=terrain_map)

    def _spin() -> None:
        try:
            rclpy.spin(node)
        except Exception:  # noqa: BLE001
            pass

    spin_t = threading.Thread(target=_spin, daemon=True, name="vnav_bridge_spin")
    spin_t.start()

    argv = ["bash", str(Path(__file__).parent / "nav/ros2/launch_nav_only.sh")] + ([] if gui else ["--no-rviz"])
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
