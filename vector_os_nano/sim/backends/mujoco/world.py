# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCoWorld: ``interfaces.PhysicsWorld`` on the compiled model itself.

The world owns a private copy of the compiled ``MjModel`` (attach writes the weld
anchor into ``eq_data``) and one ``MjData``. Commands live in ``data.ctrl``, so
``mj_step`` holds them across steps by construction. Renders run on one worker
thread (a GL context is bound to its thread) and rays on a scratch copy, each from a
state copied under the lock, so neither holds up the stepper. With ``gui`` a passive
viewer shows the world and follows a mobile robot (a free robot root). ``model`` /
``data`` are for MuJoCo-internal code only (the drivers under hardware/sim, on the
runtime's thread).
"""
from __future__ import annotations

import copy
import os
import sys
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import mujoco as mj
import numpy as np

from vector_os_nano.interfaces.physics import CONTACT_DISTANCE_M, CompiledScene, ContactLoad, RGBDFrame
from vector_os_nano.sim.scene import reader

_KINDS = ("rgb", "depth", "seg")
_VIEWER_SYNC_S = 1.0 / 60.0
_TRACK_Z = 0.3  # the viewer looks at a mobile robot's root, this high above the floor


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """The contract's K: from the vertical fovy and the actual shape."""
    fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


class MuJoCoWorld:
    name = "mujoco"

    def __init__(self, scene: CompiledScene, *, gui: bool = False) -> None:
        self.scene = scene
        self.model = m = copy.deepcopy(scene.model)
        self.data = d = mj.MjData(m)
        self._loads_valid = False
        self._loads_cache = None
        # contacts() runs its own collision pass on a copy whose margins reach
        # CONTACT_DISTANCE_M: at margin 0 MuJoCo emits no separated point, and a margin
        # on the stepped model makes force (gap does not cancel it), moving rest poses.
        self._probe = p = copy.deepcopy(m)
        p.geom_margin += CONTACT_DISTANCE_M
        p.pair_margin += CONTACT_DISTANCE_M
        p.opt.o_margin += CONTACT_DISTANCE_M  # what every contact uses under <flag override="enable"/>
        self._probe_data = mj.MjData(p)
        self._probe_valid = False
        self._lock = threading.Lock()
        acts = reader.actuators(m)
        self._position = {a.joint: a for a in acts if a.kind == "position"}
        self._motor = {a.joint: a for a in acts if a.kind == "motor"}
        self._welds = {
            (int(m.eq_obj1id[i]), int(m.eq_obj2id[i])): i
            for i in range(m.neq) if m.eq_type[i] == mj.mjtEq.mjEQ_WELD
        }
        self._welds = {pair: self._welds[pair] for pair in scene.grasp_pairs}
        mj.mj_forward(m, d)
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(1, thread_name_prefix="mujoco_render")
        self._renderers: dict[tuple[str, int, int], Any] = {}
        self._render_data: Any = None
        self._ray_lock = threading.Lock()
        self._ray_data = mj.MjData(m)
        self._viewer: Any = None
        self._synced = 0.0
        self.caller_thread_only = False
        self._recording = None
        # A mobile robot (a free root) is followed by the viewer camera.
        root = scene.robot_root
        free = root >= 0 and m.body_jntnum[root] and m.jnt_type[m.body_jntadr[root]] == mj.mjtJoint.mjJNT_FREE
        self._track = root if free else -1
        if gui:
            import mujoco.viewer  # noqa: PLC0415

            if os.environ.get("MUJOCO_GL", "").lower() == "egl":
                # EGL offscreen contexts must exist before the viewer's GLFW/GLX init.
                # ponytail: the camera sizes the bodies render (640x480, 320x240); another
                # size with the window open fails loud (_renderer).
                self.warm_renderers(_KINDS, ((320, 240), (640, 480)))
            self._viewer = v = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
            self.caller_thread_only = sys.platform == "darwin"  # Apple GLFW: main thread only
            if v is not None and self._track >= 0:
                v.cam.type = mj.mjtCamera.mjCAMERA_FREE
                v.cam.lookat[:] = [*d.xpos[self._track][:2], _TRACK_Z]
                # Look over indoor walls while following the robot across rooms.
                v.cam.distance, v.cam.elevation, v.cam.azimuth = 5.5, -70.0, -90.0
        if directory := os.environ.get('VECTOR_ROLLOUT_DIR'):
            from vector_os_nano.sim.backends.mujoco.recording import Recording

            try:
                self._recording = Recording(directory, scene)
                self._recording.state(m, d)
                self._recording.capture_due(self)
            except BaseException:
                self.close()
                raise

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        viewer, self._viewer = self._viewer, None
        if viewer is not None:
            viewer.close()
        pool, self._pool = self._pool, None
        if pool is not None:
            try:  # queued behind any in-flight render; closes each context on its thread
                pool.submit(self._close_renderers).result()
            except RuntimeError:  # atexit: interpreter shutdown already retired that thread
                self._close_renderers()
            pool.shutdown(wait=True)
        if self._recording is not None:
            self._recording.close()

    def time(self) -> float:
        return float(self.data.time)

    def step(self, n: int = 1) -> None:
        if self._recording is not None and self._recording.cameras and int(n) > 1:
            for _ in range(int(n)):
                self.step()
            return
        with self._lock:
            if self._recording is None:
                mj.mj_step(self.model, self.data, nstep=int(n))
                if int(n) > 0:
                    self._loads_valid, self._loads_cache = True, None
            else:
                for _ in range(int(n)):
                    mj.mj_step(self.model, self.data)
                    self._loads_valid, self._loads_cache = True, None
                    self._recording.state(self.model, self.data, self._contact_loads())
            if self._viewer is not None and _time.monotonic() - self._synced > _VIEWER_SYNC_S:
                if self._track >= 0:
                    self._viewer.cam.lookat[:] = [*self.data.xpos[self._track][:2], _TRACK_Z]
                self._viewer.sync()
                self._synced = _time.monotonic()
        if self._recording is not None:
            self._recording.capture_due(self)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self.data.qpos.copy(), self.data.qvel.copy()

    def reset(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        m, d = self.model, self.data
        q = np.asarray(q, dtype=np.float64).reshape(m.nq)
        qd = np.zeros(m.nv) if qd is None else np.asarray(qd, dtype=np.float64).reshape(m.nv)
        if self._recording is not None:
            self._recording.finish_videos()
        with self._lock:
            d.qpos[:], d.qvel[:] = q, qd
            for a in self._motor.values():
                d.ctrl[a.id] = 0.0
            d.qacc_warmstart[:] = 0.0
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            mj.mj_forward(m, d)
            self._loads_valid = False
            self._loads_cache = None
            if self._recording is not None:
                self._recording.state(m, d)
        if self._recording is not None:
            self._recording.capture_due(self)

    # -- actuation -----------------------------------------------------------

    def _actuators(self, table: dict, joints: Sequence[int], what: str) -> list:
        try:
            return [table[int(j)] for j in joints]
        except KeyError as exc:
            have = sorted(self.scene.model.joint(j).name for j in table)
            raise KeyError(f"joint {exc.args[0]} has no {what} actuator; joints with one: {have}") from None

    def set_targets(self, joints: Sequence[int], q: Sequence[float]) -> None:
        acts = self._actuators(self._position, joints, "position")
        q = np.asarray(q, dtype=np.float64).reshape(len(acts))
        with self._lock:
            for a, v in zip(acts, q):
                self.data.ctrl[a.id] = a.gear * v

    def targets(self, joints: Sequence[int]) -> np.ndarray:
        acts = self._actuators(self._position, joints, "position")
        return np.array([self.data.ctrl[a.id] / a.gear for a in acts])

    def set_torque(self, joints: Sequence[int], tau: Sequence[float]) -> None:
        acts = self._actuators(self._motor, joints, "motor")
        tau = np.asarray(tau, dtype=np.float64).reshape(len(acts))
        with self._lock:
            for a, t in zip(acts, tau):
                c = t / (a.gear * a.gain)
                self.data.ctrl[a.id] = c if a.ctrlrange is None else min(max(c, a.ctrlrange[0]), a.ctrlrange[1])

    # -- contact and attachment ----------------------------------------------

    def contacts(self, body: int) -> frozenset[int]:
        m, d = self._probe, self._probe_data
        with self._lock:
            # Both fingers query the same pose; a reset can change it at the same time.
            if not (self._probe_valid and np.array_equal(d.qpos, self.data.qpos)
                    and np.array_equal(d.mocap_pos, self.data.mocap_pos)
                    and np.array_equal(d.mocap_quat, self.data.mocap_quat)):
                self._probe_valid = False
                d.qpos[:] = self.data.qpos
                d.mocap_pos[:], d.mocap_quat[:] = self.data.mocap_pos, self.data.mocap_quat
                mj.mj_kinematics(m, d)
                mj.mj_collision(m, d)
                self._probe_valid = True
            n = int(d.ncon)
            near = d.contact.dist[:n] <= CONTACT_DISTANCE_M
            b1 = m.geom_bodyid[d.contact.geom1[:n][near]]
            b2 = m.geom_bodyid[d.contact.geom2[:n][near]]
        return frozenset(int(b) for b in np.concatenate([b2[b1 == body], b1[b2 == body]]))

    def contact_loads(self) -> tuple[ContactLoad, ...]:
        with self._lock:
            return self._contact_loads()

    def _contact_loads(self) -> tuple[ContactLoad, ...]:
        if not self._loads_valid:
            return ()
        if self._loads_cache is None:
            m, d, out, wrench = self.model, self.data, [], np.empty(6)
            for i, c in enumerate(d.contact):
                if c.geom1 < 0 or c.geom2 < 0:
                    raise NotImplementedError('contact loads require rigid geom contacts')
                mj.mj_contactForce(m, d, i, wrench)
                force = c.frame.reshape(3, 3).T @ wrench[:3]
                out.append(ContactLoad(int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]),
                                       tuple(c.pos), tuple(force), float(c.dist), tuple(c.frame[:3])))
            self._loads_cache = tuple(out)
        return self._loads_cache

    def attach(self, parent: int, child: int) -> None:
        pair = (int(parent), int(child))
        if pair not in self._welds:
            names = [(self.scene.body_name(p), self.scene.body_name(c)) for p, c in self.scene.grasp_pairs]
            raise KeyError(f"({parent}, {child}) is not a grasp pair; grasp pairs: {names}")
        held = self.attached().get(pair[1])
        if held is not None and held != pair[0]:
            raise ValueError(f"body {child} is already attached to body {held}; detach it first")
        m, d, i = self.model, self.data, self._welds[pair]
        with self._lock:
            # The engine's own body poses: as of the last step's kinematics (one physics
            # step behind q after a step, current after reset). Today's go2_piper grasp
            # anchor, kept bit for bit (S1 golden trace).
            pos, quat = d.xpos, d.xquat
            r1, r2 = np.empty(9), np.empty(9)
            mj.mju_quat2Mat(r1, quat[pair[0]])
            mj.mju_quat2Mat(r2, quat[pair[1]])
            R1, R2 = r1.reshape(3, 3), r2.reshape(3, 3)
            rel = np.empty(4)
            mj.mju_mat2Quat(rel, (R1.T @ R2).ravel())
            m.eq_data[i, :3] = 0.0
            m.eq_data[i, 3:6] = R1.T @ (pos[pair[1]] - pos[pair[0]])
            m.eq_data[i, 6:10] = rel
            d.eq_active[i] = 1

    def detach(self, child: int) -> None:
        with self._lock:
            for (_, c), i in self._welds.items():
                if c == child:
                    self.data.eq_active[i] = 0

    def attached(self) -> dict[int, int]:
        return {c: p for (p, c), i in self._welds.items() if self.data.eq_active[i]}

    # -- sensing ---------------------------------------------------------------

    def raycast(
        self, origins: np.ndarray, directions: np.ndarray, exclude_body: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        v = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        dist = np.full(len(o), -1.0)
        geom = np.full(len(o), -1, dtype=np.int32)
        uniq, inv = np.unique(o, axis=0, return_inverse=True)
        with self._ray_lock:
            d = self._ray_data
            with self._lock:
                d.qpos[:] = self.data.qpos
                d.mocap_pos[:], d.mocap_quat[:] = self.data.mocap_pos, self.data.mocap_quat
            mj.mj_kinematics(self.model, d)  # geom poses of exactly this q
            for k, origin in enumerate(uniq):  # one multi-ray per origin (a lidar has one)
                idx = np.flatnonzero(inv.ravel() == k)
                g, dd = np.empty(len(idx), dtype=np.int32), np.empty(len(idx))
                mj.mj_multiRay(self.model, d, origin, v[idx].ravel(), None, 1, int(exclude_body),
                               g, dd, None, len(idx), mj.mjMAXVAL)
                dist[idx], geom[idx] = dd, g
        hit = (dist >= 0) & (geom >= 0)
        body = np.where(hit, self.model.geom_bodyid[np.maximum(geom, 0)], -1)
        return np.where(hit, dist, -1.0), body.astype(np.int64)

    def render(
        self, camera: str, width: int, height: int, kinds: tuple[str, ...] = ("rgb", "depth"),
    ) -> RGBDFrame:
        kinds = tuple(kinds)
        bad = [k for k in kinds if k not in _KINDS]
        if bad:
            raise ValueError(f"unknown render kinds {bad}; kinds: {list(_KINDS)}")
        if camera not in self.scene.cameras:
            raise KeyError(f"no camera {camera!r}; cameras: {sorted(self.scene.cameras)}")
        cam, w, h = self.scene.cameras[camera], int(width), int(height)
        pool = self._pool
        if pool is None:
            raise RuntimeError("MuJoCoWorld is closed")
        return pool.submit(self._render_job, camera, cam, w, h, kinds).result()

    def _render_job(self, camera: str, cam: int, w: int, h: int, kinds: tuple[str, ...], record_frame=True) -> RGBDFrame:
        m = self.model
        d = self._render_data
        if d is None:
            d = self._render_data = mj.MjData(m)
        with self._lock:
            mj.mj_copyData(d, m, self.data)
        mj.mj_forward(m, d)  # pose and pixels of exactly d.time
        out: dict[str, np.ndarray] = {}
        for kind in kinds:
            r = self._renderer(kind, w, h)
            r.update_scene(d, camera=cam)
            out[kind] = r.render().copy()
        depth = out.get("depth")
        if depth is not None:
            depth = depth.astype(np.float32)
            depth[depth >= 0.999 * m.vis.map.zfar * m.stat.extent] = 0.0  # nothing there
        seg = out.get("seg")
        if seg is not None:
            ids, types = seg[:, :, 0], seg[:, :, 1]
            is_geom = (types == mj.mjtObj.mjOBJ_GEOM) & (ids >= 0)
            seg = np.where(is_geom, m.geom_bodyid[np.where(is_geom, ids, 0)], -1).astype(np.int32)
        frame = RGBDFrame(
            rgb=out.get("rgb"), depth=depth, seg=seg, K=intrinsics(w, h, float(m.cam_fovy[cam])),
            cam_pos=d.cam_xpos[cam].copy(), cam_mat=d.cam_xmat[cam].reshape(3, 3).copy(),
            sim_time=float(d.time), camera=camera,
        )
        if self._recording is not None and record_frame:
            self._recording.frame(frame)
        return frame

    def warm_renderers(self, kinds: Sequence[str], sizes: Sequence[tuple[int, int]]) -> None:
        """Create the renderers for every (kind, (width, height)) now, on the render thread.

        Needed before a passive viewer opens under MUJOCO_GL=egl: an EGL context created
        AFTER the viewer's GLFW/GLX init can never be made current (probed on an NVIDIA
        box; contexts created before it keep working).
        """
        def job() -> None:
            for w, h in sizes:
                for kind in kinds:
                    self._renderer(kind, int(w), int(h))

        self._pool.submit(job).result()

    def _renderer(self, kind: str, w: int, h: int) -> Any:
        """One colour/depth context and one non-MSAA ID context per size; render thread only."""
        key = ("rgb" if kind == "depth" else kind, w, h)
        r = self._renderers.get(key)
        if r is None:
            samples = self.model.vis.quality.offsamples
            try:
                # Multisample colour resolves can invent IDs at segmentation edges.
                if kind == "seg":
                    self.model.vis.quality.offsamples = 0
                r = mj.Renderer(self.model, h, w)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"cannot create a {w}x{h} {kind} renderer ({exc}); with a viewer window open "
                    f"under MUJOCO_GL=egl only the renderers created before it work: {sorted(self._renderers)}"
                ) from exc
            finally:
                self.model.vis.quality.offsamples = samples
            if kind == "seg":
                r.enable_segmentation_rendering()
            else:
                r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
            if kind != "seg":
                r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = True
            self._renderers[key] = r
        if kind == "depth":
            r.enable_depth_rendering()
        elif kind == "rgb":
            r.disable_depth_rendering()
        return r

    def _close_renderers(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # -- test seams (conformance only) ----------------------------------------

    def _link_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """The engine's own body-frame poses ``(pos (nbody, 3), quat (nbody, 4))``."""
        d = mj.MjData(self.model)
        with self._lock:
            d.qpos[:] = self.data.qpos
        mj.mj_kinematics(self.model, d)
        return d.xpos.copy(), d.xquat.copy()

    def _inertials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The engine's own ``(mass (nbody,), com (nbody, 3), inertia (nbody, 3, 3))``,
        COM and inertia tensor in the body frame."""
        m = self.model
        R = np.empty((m.nbody, 9))
        for b in range(m.nbody):
            mj.mju_quat2Mat(R[b], m.body_iquat[b])
        R = R.reshape(-1, 3, 3)
        inertia = np.einsum("bij,bj,bkj->bik", R, m.body_inertia, R)
        return m.body_mass.copy(), m.body_ipos.copy(), inertia
