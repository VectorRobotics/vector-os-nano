# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""LeggedBase — the ONE legged-base driver (``interfaces.BaseProtocol``) for every body
and backend.

Built from the manifest's ``base`` block and a ``SimRuntime`` (plus the body's lidar,
when it has one). Its controller runs every runtime tick: it reads one instant of the
state through the scene's canonical addresses and writes joint torques through
``world.set_torque``, which the world holds for the tick's physics steps.

* A posture (``stand`` = the stance, or one of the manifest's ``postures``) is a PD
  ramp to its joint angles, then a PD hold that lasts until a non-zero velocity command.
  ``connect`` and a body reset hold the stance. A velocity command out of another
  posture first ramps the body back to its stance (``_RISE_S``), then hands over; a
  walk or a navigate out of one waits for that rise before it starts (a held posture
  is not a fall). A policy that ``balances`` (a learned walking policy) owns the
  stance: ``stand``, ``connect`` and a body reset hand the body to it at zero command.
* Otherwise the manifest's control policy (``sim/policies``) gets a ``PolicyObs`` and
  returns the torques; braking and the stance at zero command are its own.
* Fallen: the root below the manifest's ``fall_z`` or its up axis tilted past
  ``fall_tilt_deg`` (a body on its side can keep its root above ``fall_z``).
* Velocity commands are clipped to the manifest's limits. A walk or a navigate holds
  skill control: another thread's ``set_velocity`` is ignored until it ends, and the
  command is zeroed on any exit. ``stop`` zeroes it from any thread, always; one that
  lands during the rise before a walk ends that walk unmoved, and one that lands once a
  navigate holds control (its rise included) ends the navigate.
* ``cmd_motion`` counts the actor's commanded |vx| + |vy| + |vyaw| (causation grading):
  only writes that pass the gate while a producer-dispatched tool runs (ACTOR_EPOCH).

Timed motions count SIM seconds, and the thread that holds skill control holds the sim
clock between its waits (``SimRuntime.wait_sim(hold=True)``): its commands land on the
same ticks at any CPU load. The base never steps physics except for a posture, which
runs its ramp on the caller's thread with the stepper parked. It never moves the world:
the world side spawns and resets the body (``sim/session.py``) and tells it
(:meth:`LeggedBase.respawned`).
"""
from __future__ import annotations

import contextlib
import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vector_os_nano.schemas.robot import BaseSpec
from vector_os_nano.interfaces.policy import PolicyObs
from vector_os_nano.sim import policies
from vector_os_nano.sim.nav import vgraph as vg
from vector_os_nano.sim.nav.follow import NavResult, follow
from vector_os_nano.sim.runtime import ACTOR_EPOCH, SimRuntime
from vector_os_nano.sim.scene import reader

logger = logging.getLogger(__name__)

_ZERO = (0.0, 0.0, 0.0)
_POSTURE_HOLD_S = 0.5    # a posture ramp is followed by this much hold before it reports
_POSTURE_TOL = 0.15      # rad: every joint this close to the posture = reached
_WALK_SETTLE_S = 0.2     # a walk waits this long after its zero command
_RISE_S = 1.0            # a velocity command out of a posture first ramps to the stance this long


class _Hold:
    """A posture: from the joints of its first tick, a tanh ramp to *target* over
    *duration* sim seconds, then *target* (held, or with *rise* handed to the policy)."""

    __slots__ = ("target", "duration", "rise", "ramp", "q0", "k")

    def __init__(self, target: np.ndarray, duration: float, dt: float, rise: bool = False) -> None:
        self.target, self.duration, self.rise = target, duration, rise
        self.ramp = max(1, int(duration / dt)) if duration > 0 else 0
        self.q0: np.ndarray | None = None
        self.k = 0

    def q_des(self, q: np.ndarray, dt: float) -> np.ndarray:
        if self.q0 is None:
            self.q0 = q
        k, self.k = self.k, self.k + 1
        if k >= self.ramp:
            return self.target
        return self.q0 + float(np.tanh((k + 1) * dt / (self.duration / 3.0))) * (self.target - self.q0)


class _Abort:
    def __init__(self, abort: Callable[[], bool]) -> None:
        self.is_set = abort


class LeggedBase:
    """Args:
        spec: the manifest ``base`` block (names of the compiled scene).
        runtime: the SimRuntime that owns the world the body stands in; its tick is
            the policy's (``1 / base.policy.tick_hz``).
        folder: where the files the policy names resolve (the robot's asset folder).
        lidar: the body's ``Lidar``, or None.
        abort: polled during a walk; True ends it like a stop (the run's stop signal).
    """

    def __init__(self, spec: BaseSpec, runtime: SimRuntime, *, folder: Path, lidar: Any = None,
                 abort: Callable[[], bool] = lambda: False) -> None:
        scene = runtime.world.scene
        if abs(runtime.tick_dt * spec.policy.tick_hz - 1.0) > 1e-9:
            raise ValueError(f"LeggedBase: the runtime ticks every {runtime.tick_dt} s; the policy at "
                             f"{spec.policy.tick_hz} Hz needs {1.0 / spec.policy.tick_hz} s")
        self._spec = spec
        self._rt = runtime
        self._scene = scene
        self._lidar = lidar
        self._cancel = _Abort(abort)
        self._joints = [scene.joint(n) for n in spec.joints]
        self._qa = [scene.joint_q(j) for j in self._joints]
        self._da = [scene.joint_qd(j) for j in self._joints]
        self._fq, self._fd = scene.free_q(scene.robot_root), scene.free_qd(scene.robot_root)
        self._stance = np.array([spec.stance[j] for j in spec.joints], dtype=np.float64)
        self._limit = spec.torque_scale * reader.motor_limits(scene, self._joints)
        ref, params = spec.policy.on(runtime.world.name)
        self.gait_backend = ref
        self._policy = policies.make(ref, spec, params, self._limit, folder)
        self._cos_fall = math.cos(math.radians(spec.fall_tilt_deg))
        self._lock = threading.Lock()   # the command, the posture and the control owners
        self._cmd = _ZERO
        self._hold: _Hold | None = None
        self._owners: list[int] = []    # threads holding skill control, latest last
        self._stops = 0                 # stop() calls so far: a control block sees one land
        self._cmd_motion = 0.0
        self._origin: float | None = None  # sim time of the spawn: PolicyObs.time counts from it
        self._handover = True           # the policy starts fresh at its next tick
        self._ticked = _ZERO            # the command the policy last saw
        self._connected = False
        self._navigation: dict = {"state": "idle", "planner": "visibility_graph", "map_source": "authored_scene", "path": []}
        self._nav_stack = None

    # -- the BaseProtocol surface ----------------------------------------------

    @property
    def limits(self) -> tuple[float, float, float]:
        return self._spec.limits

    @property
    def holonomic(self) -> bool:
        return self._spec.holonomic

    def connect(self) -> None:
        """Take over the body as it stands (spawned in its stance): hold the stance."""
        if self._connected:
            raise RuntimeError("LeggedBase: already connected (disconnect first)")
        self.respawned()
        self._rt.set_controller(self._tick)
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self._connected = False
            self._rt.set_controller(None)

    def respawned(self) -> None:
        """The world side placed the body anew (its spawn, or a reset inside the same
        step-boundary job): drop the command, hold the stance, restart the policy and its
        clock at the next tick."""
        with self._lock:
            self._cmd = _ZERO
            self._hold = None if self._policy.balances else _Hold(self._stance, 0.0, self._rt.tick_dt)
            self._origin = None
            self._handover = True

    def stop(self) -> None:
        with self._lock:
            self._cmd = _ZERO
            self._stops += 1

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        self._command(vx, vy, vyaw)

    def _command(self, vx: float, vy: float, vyaw: float, stops: int | None = None) -> bool:
        """set_velocity; with *stops* (the stop count a control block began with), False
        and nothing set when a stop has landed since."""
        lx, ly, lyaw = self._spec.limits
        cmd = (float(np.clip(vx, -lx, lx)), float(np.clip(vy, -ly, ly)), float(np.clip(vyaw, -lyaw, lyaw)))
        with self._lock:
            if self._owners and self._owners[-1] != threading.get_ident():
                return False  # another thread's walk or navigate holds control
            if stops is not None and stops != self._stops:
                return False
            self._cmd = cmd
            if cmd != _ZERO and self._hold is not None and not self._hold.rise:
                stood = self._hold.target is self._stance
                self._hold = None if stood else _Hold(self._stance, _RISE_S, self._rt.tick_dt, rise=True)
            if ACTOR_EPOCH.active:  # a stop (0, 0, 0) adds nothing
                self._cmd_motion += abs(cmd[0]) + abs(cmd[1]) + abs(cmd[2])
        return True

    def walk(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0, duration: float = 1.0,
             *, velocity_fn: Callable[[], tuple[float, float, float] | None] | None = None,
             cancel: Any = None) -> bool:
        if velocity_fn is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError('feedback motion requires a finite positive deadline')
        stopped = _Abort(lambda: self._cancel.is_set() or (cancel is not None and cancel.is_set()))
        with self._control() as stops:
            if stopped.is_set():
                return False
            fell = self._fallen()  # a body that starts fallen has fallen
            if velocity_fn is None:
                if not self._command(vx, vy, vyaw, stops):
                    return False  # stopped during the rise: it never starts
                ran = self.wait_sim(duration, cancel=stopped)
            else:
                ran, deadline = False, self._rt.sim_time + duration
                while not fell and not stopped.is_set() and self._rt.sim_time < deadline-1e-9:
                    with self._lock:
                        if stops != self._stops:
                            break
                    command = velocity_fn()
                    if command is None:
                        ran = not stopped.is_set() and self._command(*_ZERO, stops)
                        break
                    if len(command) != 3 or not all(math.isfinite(value) for value in command):
                        raise ValueError('feedback controller must return three finite velocities or None')
                    if not self._command(*command, stops):
                        break
                    if not self.wait_sim(min(self._rt.tick_dt, deadline-self._rt.sim_time), cancel=stopped):
                        break
                    fell = self._fallen()
            self.set_velocity(0.0, 0.0, 0.0)
            self.wait_sim(_WALK_SETTLE_S, cancel=self._cancel)
            return bool(ran and not fell and not self._fallen())

    def set_posture(self, name: str, duration: float = 2.0) -> bool:
        """The stance of a policy that balances is that policy at zero command: out of a
        held posture the body is ramped there first, then handed over; True iff it is up
        (not fallen) after *duration* and the hold time."""
        if name == "stand":
            target = self._stance
        else:
            try:
                q = self._spec.postures[name]
            except KeyError:
                raise KeyError(f"no posture {name!r}; postures: {['stand', *sorted(self._spec.postures)]}") from None
            target = np.array([q[j] for j in self._spec.joints], dtype=np.float64)
        if not self._connected:
            raise RuntimeError("LeggedBase: not connected")
        dt = self._rt.tick_dt
        hold = _Hold(target, float(duration), dt, rise=name == "stand" and self._policy.balances)
        with self._lock:
            self._cmd = _ZERO
            self._hold = None if hold.rise and self._hold is None else hold
        self._rt.step(hold.ramp + int(_POSTURE_HOLD_S / dt))
        if hold.rise:
            return not self._fallen()
        q = self._rt.snapshot().q[self._qa]
        return bool(np.abs(q - target).max() < _POSTURE_TOL)

    def get_position(self) -> list[float]:
        return [float(v) for v in self._rt.snapshot().q[self._fq][:3]]

    def get_heading(self) -> float:
        w, x, y, z = self._rt.snapshot().q[self._fq][3:7]
        return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def get_velocity(self) -> list[float]:
        return [float(v) for v in self._rt.snapshot().qd[self._fd][:3]]

    def get_odometry(self) -> Any:
        from vector_os_nano.core.types import Odometry

        s = self._rt.snapshot()
        q, v = s.q[self._fq], s.qd[self._fd]
        w, x, y, z = (float(c) for c in q[3:7])
        # the WORLD z rate: the body-frame angular velocity through the third row of R
        vyaw = 2.0 * (x * z - w * y) * v[3] + 2.0 * (y * z + w * x) * v[4] + (1.0 - 2.0 * (x * x + y * y)) * v[5]
        return Odometry(
            timestamp=s.sim_time, x=float(q[0]), y=float(q[1]), z=float(q[2]), qx=x, qy=y, qz=z, qw=w,
            vx=float(v[0]), vy=float(v[1]), vz=float(v[2]), vyaw=float(vyaw),
        )

    def get_lidar_scan(self) -> Any:
        return None if self._lidar is None else self._lidar.scan()

    def get_sim_time(self) -> float:
        return self._rt.sim_time

    def wait_sim(self, seconds: float, cancel: Any = None) -> bool:
        """The thread that holds skill control holds the clock while it waits (the
        stepper parks at the end of the wait), so its next command lands on the next
        tick whatever the CPU load: a walk or a navigate runs as it does caller-stepped."""
        with self._lock:
            hold = bool(self._owners) and self._owners[-1] == threading.get_ident()
        return self._rt.wait_sim(seconds, cancel=cancel, hold=hold)

    def cmd_motion(self) -> float:
        return float(self._cmd_motion)

    def navigation_state(self) -> dict:
        """A detached, read-only view of the selected planner's path and result."""
        import copy
        return copy.deepcopy(self._navigation)

    def navigate_to(self, x: float, y: float, tol: float | None = None, speed: float | None = None,
                    timeout: float | None = None, cancel: Any = None, on_progress: Any = None) -> NavResult:
        """Walk to world (x, y) around the scene's static obstacles (the shared follower).

        Plans on the visibility graph of the scene at this instant, snapping a goal inside
        furniture to the nearest free point, and streams velocity in sim time while
        holding skill control. ``timeout`` is sim seconds (None: a budget from the
        geodesic). ``bool(result)`` is True only on arrival; ``result.reason`` says why
        not. The inflation is the manifest's ``base.nav.body_radius`` plus how far a
        mounted arm or a held object reaches past the body. A ``stop()`` once the
        navigate holds control (the rise out of a posture included) ends it 'aborted',
        like *cancel* (None: the run's stop signal).
        """
        if self._nav_stack is not None:
            outer = self._cancel if cancel is None else cancel
            with self._control() as stops:
                return self._nav_stack.navigate((x, y), nav=self._spec.nav, fallen=self._fallen,
                    cancel=_Abort(lambda: self._stops != stops or outer.is_set()),
                    tol=tol, speed=speed, timeout=timeout, on_progress=on_progress,
                    command=lambda *values: self._command(*values, stops=stops))
        q = self._rt.snapshot().q
        obstacles = vg.scene_obstacles(self._scene, q)
        reach = vg.scene_overhang(self._scene, q, set(self._joints), self._rt.attachments())
        nav = self._spec.nav
        outer = self._cancel if cancel is None else cancel
        with self._control() as stops:
            stopped = _Abort(lambda: self._stops != stops or outer.is_set())
            self._navigation = {"planner": "visibility_graph", "map_source": "authored_scene",
                                "state": "planning", "goal": [x, y], "path": []}
            def planned(path):
                self._navigation = {**self._navigation, "state": "following", "path": [list(p) for p in path]}
            def progress(remaining, elapsed):
                self._navigation = {**self._navigation, "remaining_m": remaining, "elapsed_s": elapsed}
                if on_progress is not None:
                    on_progress(remaining, elapsed)
            try:
                res = follow(self, (x, y), obstacles, nav=nav, fallen=self._fallen,
                             inflation=nav.body_radius + reach, tol=tol, speed=speed, timeout=timeout,
                             cancel=stopped, on_progress=progress, on_plan=planned)
            except BaseException:
                self._navigation = {**self._navigation, "state": "error"}
                raise
            self._navigation = {**self._navigation, "state": res.reason, "result": dict(res),
                                "remaining_m": res['dist'], "elapsed_s": res['elapsed_s']}
        logger.info("navigate_to(%.2f, %.2f): %s", x, y, res)
        return res

    def explore_tare(self, *, cancel, timeout=120., on_progress=None):
        """Give the installed exploration stack the same cancellable control lease."""
        if self._nav_stack is None:
            raise RuntimeError('NAV STACK is not connected')
        with self._control() as stops:
            return self._nav_stack.explore(nav=self._spec.nav, fallen=self._fallen,
                cancel=_Abort(lambda: self._stops != stops or self._cancel.is_set() or cancel.is_set()),
                timeout=timeout, on_progress=on_progress,
                command=lambda *values: self._command(*values, stops=stops))

    # -- beyond the protocol: the world side and the mount ----------------------------

    def wait_settled(self, timeout: float = 3.0) -> bool:
        """Wait (sim time) until a stop has braked out (an arm mounted on the body plans
        against the pose it ends in); False on the timeout."""
        deadline = self._rt.sim_time + timeout
        while not self._settled():
            if self._rt.sim_time >= deadline or not self._rt.wait_sim(0.05):
                return False
        return True

    # -- internals -------------------------------------------------------------------

    def _fallen(self) -> bool:
        """The root below ``fall_z``, or its up axis tilted past ``fall_tilt_deg``."""
        q = self._rt.snapshot().q[self._fq]
        return bool(q[2] < self._spec.fall_z or 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2) < self._cos_fall)

    def _settled(self) -> bool:
        with self._lock:
            if self._hold is not None:
                return not self._hold.rise
            pending = self._cmd == _ZERO and self._ticked != _ZERO  # a brake about to start
        return not (pending or self._policy.braking)

    def _rising(self) -> bool:
        with self._lock:
            return self._hold is not None and self._hold.rise

    @contextlib.contextmanager
    def _control(self):
        """Skill control for the block (nested and concurrent holders stack: the latest
        one commands), out of the stance: a held posture first rises to it, so a walk or a
        navigate measures its start (and its fall) on the body standing. The command is
        zeroed on any exit, KeyboardInterrupt included. Yields the stop count at entry."""
        me = threading.get_ident()
        with self._lock:
            self._owners.append(me)
            stops = self._stops
            if self._hold is not None and self._hold.target is not self._stance:
                self._hold = _Hold(self._stance, _RISE_S, self._rt.tick_dt, rise=True)
        try:
            self.wait_sim(0.0)  # park the clock here: the motion starts on the next tick
            while self._rising() and self.wait_sim(self._rt.tick_dt, cancel=self._cancel):
                pass
            yield stops
        finally:
            with self._lock:
                self._owners.remove(me)
                self._cmd = _ZERO
            self._rt.release_clock()

    def _tick(self, w: Any) -> None:
        """The runtime controller: one tick, before the world steps."""
        q, qd = w.state()
        jq, jd = q[self._qa], qd[self._da]
        with self._lock:
            if self._hold is not None and self._hold.rise and self._hold.k > self._hold.ramp:
                self._hold = None  # risen to the stance: the policy takes the command
            cmd, hold = self._cmd, self._hold
            if self._origin is None:
                self._origin = w.time()
        s = self._spec
        if hold is not None:
            tau = policies.pd(hold.q_des(jq, self._rt.tick_dt), jq, jd, s.kp, s.kd, self._limit)
            self._handover = True
        else:
            fq, fd = q[self._fq], qd[self._fd]
            obs = PolicyObs(time=w.time() - self._origin, base_pos=fq[:3], base_quat=fq[3:7],
                            base_lin_vel=fd[:3], base_ang_vel=fd[3:6], joint_pos=jq, joint_vel=jd,
                            command=np.array(cmd))
            if self._handover:
                self._policy.reset(obs)
                self._handover = False
            tau = self._policy.torques(obs)
            self._ticked = cmd
        w.set_torque(self._joints, tau)
