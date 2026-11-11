# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimRuntime — the ONE owner of a :class:`~vector_os_nano.interfaces.physics.PhysicsWorld`.

Every thread that touches the simulation goes through this object, so physics
stepping, mutations and reads can never race (the R12 / reset_pose segfault
class: ``mj_forward`` or a qpos write on live data while another thread steps).

* **Stepping.** One control tick = ``controller(world)`` then
  ``world.step(substeps)``, always under ``lock``. In the default mode one
  daemon thread runs ticks paced against an ABSOLUTE wall-clock schedule
  (``next += period``; a lag over ``_MAX_LAG_S`` is dropped, so catch-up is
  bounded). ``VECTOR_SIM_LOCKSTEP=1`` (or ``threaded=False``) starts no thread:
  physics advances only through :meth:`step` / :meth:`wait_sim` on the caller's
  thread, so a scripted run is deterministic.
* **Mutations.** :meth:`submit` runs ``fn(world)`` at a step boundary and
  returns a ``Future`` with its result. The stepper drains the queue before each
  tick (and every few ms while paused); with no daemon it runs inline.
* **Reads.** :meth:`snapshot` is the latest :class:`SimSnapshot` (``world.state()``
  and sim time of one instant), republished after every tick and every mutation;
  lock-free readers slice ``q`` / ``qd`` by the scene's canonical addresses.
  :meth:`render` renders from a private ``mj_copyData`` copy taken at a step
  boundary, so RGB, depth, segmentation and camera pose share one ``sim_time``.
* **Attachments.** Every change of ``world.attached()`` is logged with its sim
  time (:attr:`attach_log`), diffed at every publish.
* **One render owner.** Every offscreen renderer is created, used and closed on
  one worker thread: a GL (EGL/GLX) context stays bound to the live thread that
  last used it, and any other thread then fails with EGL_BAD_ACCESS. Renderers
  are cached per (kind, width, height), so a frame always has the size it was
  asked for and its intrinsics are derived from that shape.
* **Sim time.** :meth:`wait_sim` waits for SIM seconds, so a timed motion means
  the same thing at any real-time ratio.
* **Actor epoch.** :data:`ACTOR_EPOCH` marks when commanded motion belongs to a
  producer-dispatched tool (see :class:`ActorEpoch`).
"""
from __future__ import annotations

import collections
import contextlib
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from vector_os_nano.interfaces.physics import PhysicsWorld

logger = logging.getLogger(__name__)

LOCKSTEP_ENV = "VECTOR_SIM_LOCKSTEP"

# A tick that runs later than this behind schedule drops the backlog instead of
# sprinting to catch up (bounded catch-up after a stall, e.g. a GC pause).
_MAX_LAG_S = 0.1
# How often a parked (paused) daemon services the mutation queue.
_PARKED_POLL_S = 0.005
# wait_sim's default wall-clock safety cap: physics may run this much slower
# than real time before a wait gives up (a dead physics thread never hangs it).
_MIN_REALTIME = 0.05
_MIN_WALL_CAP_S = 5.0
_RT_LOG_EVERY_S = 60.0

Controller = Callable[[PhysicsWorld], None]


def lockstep_requested() -> bool:
    """True when ``VECTOR_SIM_LOCKSTEP=1`` asks for caller-stepped physics."""
    return os.environ.get(LOCKSTEP_ENV, "") == "1"


@dataclass(frozen=True)
class SimSnapshot:
    """State of one instant (``world.state()``): ``q`` / ``qd`` are copies in the
    canonical layout, never live views."""

    sim_time: float
    step: int
    q: np.ndarray
    qd: np.ndarray


@dataclass(frozen=True)
class RGBDFrame:
    """One camera frame; every field comes from the same ``sim_time``.

    ``K`` is derived from the camera's ``fovy`` and the ACTUAL image shape.
    ``depth`` is metres (float32), ``seg`` the per-pixel geom id (-1 = none).
    """

    rgb: np.ndarray | None
    depth: np.ndarray | None
    seg: np.ndarray | None
    K: Any
    cam_pos: np.ndarray
    cam_mat: np.ndarray
    sim_time: float
    camera: str


class ActorEpoch:
    """When commanded motion is the ACTOR's: only while a producer-dispatched tool runs.

    A producer (the native runner, the legacy executor) wraps each tool it dispatches
    in :meth:`tool`; the bodies' commanded-motion counters (``cmd_motion`` /
    ``ctrl_motion``, read by the actor-causation grader) accumulate only while one is
    open. A TARE/bridge follower or a background explore driving the base BETWEEN
    tools is therefore never counted as the actor causing a step.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open = 0

    @contextlib.contextmanager
    def tool(self):
        with self._lock:
            self._open += 1
        try:
            yield
        finally:
            with self._lock:
                self._open -= 1

    @property
    def active(self) -> bool:
        return self._open > 0


# ponytail: one epoch per process (one SimSession at a time); per-runtime if two sims
# ever share a process.
ACTOR_EPOCH = ActorEpoch()


class SimRuntime:
    """Owns one ``world``: one stepper, step-boundary mutations, snapshots."""

    def __init__(
        self,
        world: PhysicsWorld,
        *,
        controller: Controller | None = None,
        substeps: int = 1,
        threaded: bool = True,
        paced: bool = True,
        name: str = "sim",
    ) -> None:
        if lockstep_requested():
            threaded, paced = False, False
        self.world = world
        self.substeps = max(1, int(substeps))
        self.tick_dt = float(world.scene.timestep) * self.substeps
        self.threaded = threaded
        self.paced = paced
        self.name = name
        # Pacing target: sim seconds per wall second. Tests force <1 to prove
        # sim-time skills do not depend on it.
        self.rt_factor = 1.0
        self.lock = threading.RLock()
        self._controller: Controller | None = controller
        self._queue: collections.deque[tuple[Callable[[PhysicsWorld], Any], Future]] = collections.deque()
        self._wake = threading.Event()
        self._published = threading.Condition()
        self._pauses = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stepper: int | None = None  # ident of the thread stepping right now
        self._steps = 0
        # Attachment transition log: (sim_time, child body, parent body or None for a
        # detach) per change of world.attached(), appended at every publish
        # (GroundTruth.weld_events reads it).
        self.attach_log: collections.deque[tuple[float, int, int | None]] = collections.deque(maxlen=4096)
        self._attached = world.attached()
        self._snap = self._take_snapshot()
        self._render_pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            1, thread_name_prefix=f"{name}_render"
        )
        self._renderers: dict[tuple[str, int, int], Any] = {}
        self._render_data: Any = None  # render-thread-owned scratch MjData
        self._rt_mark = (time.monotonic(), world.time())
        self._rt_ratio = 0.0
        self._pace_t = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def set_controller(self, fn: Controller | None) -> None:
        """Install the per-tick controller ``fn(world)`` (runs before each world.step)."""
        with self.lock:
            self._controller = fn

    def start(self) -> None:
        """Start the physics daemon (no-op when caller-stepped)."""
        if not self.threaded:
            return
        if self._thread is not None:
            raise RuntimeError(f"{self.name}: physics thread already started")
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"{self.name}_physics")
        self._thread.start()

    def close(self) -> None:
        """Stop physics, fail pending mutations, close every renderer on its thread, then
        the world. Idempotent."""
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
            if t.is_alive():  # keep the handle: a live stepper must never be forgotten
                logger.error("%s: physics thread did not stop within 5 s", self.name)
                return
            self._thread = None
        self._fail_pending()
        pool, self._render_pool = self._render_pool, None
        if pool is not None:
            try:  # queued behind any in-flight render; closes each context on its thread
                pool.submit(self._close_renderers).result()
            except RuntimeError:  # atexit: interpreter shutdown already retired that thread
                self._close_renderers()
            pool.shutdown(wait=True)
        self.world.close()
        with self._published:
            self._published.notify_all()

    @property
    def closed(self) -> bool:
        return self._stop.is_set()

    @property
    def daemon_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        return self._snap.sim_time

    def snapshot(self) -> SimSnapshot:
        """The latest published state (republished every tick and every mutation)."""
        return self._snap

    def rt_ratio(self) -> float:
        """Measured sim-seconds per wall-second over the last measurement window."""
        now, sim = time.monotonic(), self._snap.sim_time
        t0, s0 = self._rt_mark
        if now - t0 >= 1.0:
            self._rt_ratio = (sim - s0) / (now - t0)
            self._rt_mark = (now, sim)
        return self._rt_ratio

    # ------------------------------------------------------------------
    # mutations
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[[PhysicsWorld], Any]) -> Future:
        """Run ``fn(world)`` at a step boundary; the Future carries its result.

        Never wait on the Future while holding :attr:`lock` from another thread.
        """
        fut: Future = Future()
        if self._stop.is_set():
            fut.set_exception(RuntimeError(f"{self.name}: runtime closed"))
            return fut
        if not self.daemon_alive or threading.get_ident() == self._stepper:
            with self.lock:
                self._run([(fn, fut)])
            return fut
        self._queue.append((fn, fut))
        self._wake.set()
        if self._stop.is_set():  # close() raced this append: never leave it hanging
            self._fail_pending()
        return fut

    def _fail_pending(self) -> None:
        while True:
            try:
                _fn, fut = self._queue.popleft()  # atomic: each job is failed exactly once
            except IndexError:
                return
            if fut.set_running_or_notify_cancel():
                fut.set_exception(RuntimeError(f"{self.name}: runtime closed"))

    def call(self, fn: Callable[[PhysicsWorld], Any]) -> Any:
        """``submit(fn).result()``: a read or mutation at a step boundary."""
        return self.submit(fn).result()

    @contextlib.contextmanager
    def paused(self):
        """Park the daemon at a step boundary for the duration (reference-counted)."""
        with self.lock:
            self._pauses += 1
        try:
            yield self
        finally:
            with self.lock:
                self._pauses -= 1
            self._wake.set()

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def step(self, ticks: int = 1, controller: Controller | None = None) -> None:
        """Advance *ticks* control ticks on the CALLER's thread, unpaced (daemon parked).

        ``controller`` overrides the installed one for these ticks only (e.g. a
        posture ramp).
        """
        with self.paused():
            prev, self._stepper = self._stepper, threading.get_ident()
            try:
                for _ in range(int(ticks)):
                    if self._stop.is_set():
                        return
                    with self.lock:
                        self._tick(controller)
            finally:
                self._stepper = prev

    def wait_sim(
        self,
        seconds: float,
        cancel: Any = None,
        wall_timeout: float | None = None,
    ) -> bool:
        """Block until SIM time has advanced by *seconds*.

        Caller-stepped mode steps the physics here. Returns False when *cancel*
        (anything with ``is_set()``) fires, the runtime closes, or the wall-clock
        safety cap passes (physics slower than ``_MIN_REALTIME`` or dead).
        """
        target = self.sim_time + max(0.0, float(seconds))
        cap = wall_timeout if wall_timeout is not None else max(
            _MIN_WALL_CAP_S, float(seconds) / _MIN_REALTIME
        )
        deadline = time.monotonic() + cap
        if not self.daemon_alive:
            with self.paused():
                prev, self._stepper = self._stepper, threading.get_ident()
                try:
                    while self.sim_time < target - 1e-9:
                        if (self._stop.is_set() or (cancel is not None and cancel.is_set())
                                or time.monotonic() > deadline):
                            return False
                        with self.lock:
                            self._tick(None)
                        if self.paced:
                            self._pace()
                finally:
                    self._stepper = prev
            return True
        with self._published:
            while self.sim_time < target - 1e-9:
                if (self._stop.is_set() or (cancel is not None and cancel.is_set())
                        or time.monotonic() > deadline):
                    return False
                self._published.wait(timeout=0.05)
        return True

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def render(
        self,
        camera: str,
        width: int,
        height: int,
        kinds: Iterable[str] = ("rgb", "depth"),
    ) -> RGBDFrame:
        """Render *kinds* (rgb / depth / seg) of *camera* from ONE step-boundary copy."""
        from vector_os_nano.perception.depth_projection import mujoco_intrinsics

        kinds = tuple(kinds)
        w, h = int(width), int(height)

        model = self.world.model

        def job() -> RGBDFrame:
            import mujoco as mj

            d = self._render_data
            if d is None:
                d = self._render_data = mj.MjData(model)
            self.call(lambda w: mj.mj_copyData(d, w.model, w.data))
            # Recompute every derived quantity from the copied state, so pose and
            # pixels are of exactly d.time (live xpos lags qpos by one mj_step).
            mj.mj_forward(model, d)
            cam = model.cam(camera).id
            out: dict[str, np.ndarray] = {}
            for kind in kinds:
                r = self._renderer(kind, w, h)
                r.update_scene(d, camera=cam)
                out[kind] = r.render().copy()
            seg = out.get("seg")
            return RGBDFrame(
                rgb=out.get("rgb"),
                depth=out["depth"].astype(np.float32) if "depth" in out else None,
                seg=seg[:, :, 0] if seg is not None and seg.ndim == 3 else seg,
                K=mujoco_intrinsics(w, h, vfov_deg=float(model.cam_fovy[cam])),
                cam_pos=d.cam_xpos[cam].copy(),
                cam_mat=d.cam_xmat[cam].copy(),
                sim_time=float(d.time),
                camera=camera,
            )

        pool = self._render_pool
        if pool is None:
            raise RuntimeError(f"{self.name}: runtime closed")
        return pool.submit(job).result()

    def warm_renderers(self, kinds: Iterable[str], sizes: Iterable[tuple[int, int]]) -> None:
        """Create the renderers for every (kind, size) now, on the render thread.

        Needed before a passive viewer opens under MUJOCO_GL=egl: an EGL context
        created AFTER the viewer's GLFW/GLX init can never be made current (probed
        on this NVIDIA box; contexts created before it keep working).
        """
        def job() -> None:
            for w, h in sizes:
                for kind in kinds:
                    self._renderer(kind, int(w), int(h))

        self._render_pool.submit(job).result()

    def _renderer(self, kind: str, width: int, height: int) -> Any:
        """The cached offscreen renderer for (kind, width, height); render thread only."""
        key = (kind, width, height)
        r = self._renderers.get(key)
        if r is None:
            import mujoco as mj

            try:
                r = mj.Renderer(self.world.model, height, width)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{self.name}: cannot create a {width}x{height} {kind} renderer ({exc}); "
                    f"with a viewer window open under MUJOCO_GL=egl only the renderers "
                    f"created before it work: {sorted(self._renderers)}"
                ) from exc
            if kind == "seg":
                r.enable_segmentation_rendering()
            else:
                r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
            if kind == "rgb":
                r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = True
            elif kind == "depth":
                r.enable_depth_rendering()
            self._renderers[key] = r
        return r

    def _close_renderers(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._renderers.clear()

    # ------------------------------------------------------------------
    # internals (callers hold self.lock)
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> SimSnapshot:
        q, qd = self.world.state()
        return SimSnapshot(self.world.time(), self._steps, q, qd)

    def _publish(self) -> None:
        self._snap = self._take_snapshot()
        now = self.world.attached()
        if now != self._attached:
            t, was = self._snap.sim_time, self._attached
            for child in sorted(was.keys() | now.keys()):
                if was.get(child) != now.get(child):
                    self.attach_log.append((t, child, now.get(child)))
            self._attached = now
        with self._published:
            self._published.notify_all()

    def _run(self, jobs: list[tuple[Callable[[PhysicsWorld], Any], Future]]) -> None:
        """Apply mutations, publish the new state, THEN resolve their futures."""
        done: list[tuple[Future, bool, Any]] = []
        for fn, fut in jobs:
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                done.append((fut, True, fn(self.world)))
            except BaseException as exc:  # noqa: BLE001 — delivered to the submitter
                done.append((fut, False, exc))
        if done:
            self._publish()
        for fut, ok, val in done:
            fut.set_result(val) if ok else fut.set_exception(val)

    def _drain(self) -> None:
        jobs = []
        while self._queue:
            jobs.append(self._queue.popleft())
        if jobs:
            self._run(jobs)

    def _tick(self, controller: Controller | None) -> None:
        self._drain()
        ctrl = controller or self._controller
        if ctrl is not None:
            ctrl(self.world)
        self.world.step(self.substeps)
        self._steps += self.substeps
        self._publish()

    def _pace(self) -> None:
        """Sleep to the absolute schedule; drop a backlog over _MAX_LAG_S."""
        now = time.perf_counter()
        if self._pace_t == 0.0 or now - self._pace_t > _MAX_LAG_S:
            self._pace_t = now
        self._pace_t += self.tick_dt / max(1e-6, self.rt_factor)
        ahead = self._pace_t - time.perf_counter()
        if ahead > 0:
            time.sleep(ahead)

    def _loop(self) -> None:
        self._stepper = threading.get_ident()
        last_log = time.monotonic()
        while not self._stop.is_set():
            with self.lock:
                stepped = self._pauses == 0
                try:
                    if stepped:
                        self._tick(None)
                    else:
                        self._drain()
                except BaseException:  # fail loud: a dead stepper must never be silent
                    logger.exception("%s: physics stepper died at sim_time=%.3f",
                                     self.name, self.sim_time)
                    # ponytail: a submit racing this exact instant can still wait
                    # forever; later submits see daemon_alive False and run inline.
                    self._fail_pending()
                    raise
            if not stepped:
                self._wake.wait(_PARKED_POLL_S)
                self._wake.clear()
                self._pace_t = 0.0
                continue
            if self.paced:
                self._pace()
            if time.monotonic() - last_log > _RT_LOG_EVERY_S:
                last_log = time.monotonic()
                logger.info("%s: real-time ratio %.2fx", self.name, self.rt_ratio())
