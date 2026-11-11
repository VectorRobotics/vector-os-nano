# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Every binding sentence of charter section 3.1, on every backend (scenes/contract.xml)."""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from tests.conformance.rig import qexp, qmat, qmul, quat_angle
from vector_os_nano.interfaces import CONTACT_DISTANCE_M, PhysicsWorld, RGBDFrame
from vector_os_nano.sim.scene import canonical

pytestmark = pytest.mark.sim

G = 9.81


def _id(scene, name: str) -> int:
    return scene.body(name)


def _mass_matrix_diag(scene, joint: int) -> float:
    """M[dof, dof] at qpos0 from the compiled model (the inertia the joint drives)."""
    import mujoco as mj

    m = scene.model
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mj.mj_fullM(m, d, M)
    return float(M[scene.joint_qd(joint), scene.joint_qd(joint)])


# -- the world object -----------------------------------------------------------


def test_world_is_a_physics_world(backend, load, contract_scene):
    w = load(backend, contract_scene)
    assert isinstance(w, PhysicsWorld)
    assert w.name == backend and w.scene is contract_scene
    assert w.caller_thread_only is False  # headless: a stepper thread may drive it
    t0 = w.time()
    w.step(7)
    assert w.time() == pytest.approx(t0 + 7 * contract_scene.timestep, abs=1e-12)


def test_close_is_idempotent(backend, contract_scene):
    from vector_os_nano.sim import backends

    w = backends.get(backend).load(contract_scene, gui=False)
    w.close()
    w.close()


# -- state layout and reset round trip ----------------------------------------------


def test_state_layout_and_reset_round_trip(backend, load, contract_scene):
    m = contract_scene.model
    w = load(backend, contract_scene)
    q0, qd0 = w.state()
    assert q0.shape == (m.nq,) and qd0.shape == (m.nv,)
    np.testing.assert_allclose(q0, m.qpos0, atol=1e-9)  # loaded at the model pose
    rng = np.random.default_rng(1)
    q = m.qpos0.copy()
    for name in ("die", "parcel"):
        s = contract_scene.free_q(_id(contract_scene, name))
        quat = rng.normal(size=4)
        q[s] = np.concatenate([q[s][:3] + rng.uniform(-0.1, 0.1, 3) + [0, 0, 0.5], quat / np.linalg.norm(quat)])
    for name in ("spin", "servo", "weak", "rail"):
        q[contract_scene.joint_q(contract_scene.joint(name))] = rng.uniform(-1, 1)
    qd = rng.uniform(-1, 1, m.nv)
    w.reset(q, qd)
    q1, qd1 = w.state()
    np.testing.assert_allclose(q1, q, atol=1e-9)
    np.testing.assert_allclose(qd1, qd, atol=1e-9)
    w.reset(q)
    np.testing.assert_array_equal(w.state()[1], np.zeros(m.nv))  # qd omitted = zero
    # copies, not views
    q1[:] = 0
    np.testing.assert_allclose(w.state()[0], q, atol=1e-9)


def test_free_body_velocity_is_world_linear_and_body_angular(backend, load, contract_scene):
    """Free fall with spin: position integrates the WORLD linear velocity, orientation
    integrates the BODY angular velocity (a cube: torque-free, so omega is constant)."""
    sc = contract_scene
    die = _id(sc, "die")
    w = load(backend, sc)
    q, qd = sc.model.qpos0.copy(), np.zeros(sc.model.nv)
    rot = qexp(np.array([math.pi / 2, 0, 0]))  # body z = world -y
    q[sc.free_q(die)] = np.concatenate([[0.5, -1.0, 2.0], rot])
    v, omega = np.array([0.3, -0.2, 0.1]), np.array([0.0, 0.0, 2.0])
    qd[sc.free_qd(die)] = np.concatenate([v, omega])
    w.reset(q, qd)
    n, dt = 20, sc.timestep
    w.step(n)
    q1, qd1 = w.state()
    a = np.array([0, 0, -G])
    expect_p = q[sc.free_q(die)][:3] + v * n * dt + a * dt * dt * n * (n + 1) / 2
    np.testing.assert_allclose(q1[sc.free_q(die)][:3], expect_p, atol=1e-6)
    np.testing.assert_allclose(qd1[sc.free_qd(die)][:3], v + a * n * dt, atol=1e-6)
    np.testing.assert_allclose(qd1[sc.free_qd(die)][3:], omega, atol=1e-6)
    expect_q = qmul(rot, qexp(omega * n * dt))
    assert quat_angle(q1[sc.free_q(die)][3:], expect_q)[0] < 1e-6
    # and the layout is the body FRAME: canonical FK of q places the body there
    pos, _ = canonical.body_poses(sc, q1)
    np.testing.assert_allclose(pos[die], q1[sc.free_q(die)][:3], atol=1e-12)


def test_free_body_linear_velocity_is_the_frame_origin_s(backend, load, contract_scene):
    """The wand's COM sits 5 cm off its frame origin: spinning, the origin's velocity
    differs from the COM's by omega x r (0.15 m/s here), which the layout must carry."""
    sc = contract_scene
    wand = sc.body("wand")
    w = load(backend, sc)
    q, qd = sc.model.qpos0.copy(), np.zeros(sc.model.nv)
    v0, omega, r = np.array([0.2, 0.0, 0.5]), np.array([0.0, 0.0, 3.0]), np.array([0.05, 0.0, 0.0])
    qd[sc.free_qd(wand)] = np.concatenate([v0, omega])
    w.reset(q, qd)
    n, dt = 25, sc.timestep
    w.step(n)
    q1, qd1 = w.state()
    R = qmat(qexp(omega * n * dt))  # starts at identity: body z = world z
    g = np.array([0.0, 0.0, -G])
    com = q[sc.free_q(wand)][:3] + r + (v0 + np.cross(omega, r)) * n * dt + g * dt * dt * n * (n + 1) / 2
    np.testing.assert_allclose(q1[sc.free_q(wand)][:3], com - R @ r, atol=1e-4)
    v_com = v0 + np.cross(omega, r) + g * n * dt
    np.testing.assert_allclose(qd1[sc.free_qd(wand)][:3], v_com - np.cross(omega, R @ r), atol=2e-3)
    np.testing.assert_allclose(qd1[sc.free_qd(wand)][3:], omega, atol=1e-6)


# -- actuation -------------------------------------------------------------------


def test_torque_is_held_for_all_n_steps_and_clipped(backend, load, contract_scene):
    sc = contract_scene
    spin = sc.joint("spin")
    inertia = _mass_matrix_diag(sc, spin)
    w = load(backend, sc)
    w.set_torque([spin], [2.0])  # ctrlrange is +-0.5: clipped
    w.step(50)
    expect = 0.5 / inertia * 50 * sc.timestep
    qd = w.state()[1][sc.joint_qd(spin)]
    assert qd == pytest.approx(expect, rel=1e-6), "torque not held for all n steps, or not clipped"
    w.step(25)
    w.step(25)  # still held across calls, until replaced
    assert w.state()[1][sc.joint_qd(spin)] == pytest.approx(2 * expect, rel=1e-6)
    w.set_torque([spin], [-0.25])
    w.step(50)
    assert w.state()[1][sc.joint_qd(spin)] == pytest.approx(2 * expect - 0.25 / inertia * 50 * sc.timestep, rel=1e-6)


def test_reset_clears_torques_and_keeps_targets_and_attachments(backend, load, contract_scene):
    sc = contract_scene
    spin, servo = sc.joint("spin"), sc.joint("servo")
    w = load(backend, sc)
    w.set_torque([spin], [0.5])
    w.set_targets([servo], [0.1])
    w.attach(sc.body("carrier"), sc.body("parcel"))
    t = w.time()
    w.reset(sc.model.qpos0)
    assert w.time() == t
    w.step(50)
    assert abs(w.state()[1][sc.joint_qd(spin)]) < 1e-12
    assert w.targets([servo])[0] == pytest.approx(0.1)
    assert w.attached() == {sc.body("parcel"): sc.body("carrier")}


def test_actuator_kind_mismatch_is_a_key_error(backend, load, contract_scene):
    sc = contract_scene
    w = load(backend, sc)
    with pytest.raises(KeyError, match="servo"):  # names the joints that have one
        w.set_targets([sc.joint("spin")], [0.0])
    with pytest.raises(KeyError):
        w.targets([sc.joint("rail")])
    with pytest.raises(KeyError, match="spin"):
        w.set_torque([sc.joint("servo")], [0.0])
    with pytest.raises(KeyError):
        w.set_torque([sc.joint("die")], [0.0])  # a free joint has no actuator


def test_targets_read_back_and_use_the_model_gain(backend, load, contract_scene):
    """The servo holds a 1 kg mass at 0.3 m against gravity: the droop is m g l cos(q) / kp
    with the MODEL's kp (500), evaluated at every step."""
    sc = contract_scene
    servo = sc.joint("servo")
    w = load(backend, sc)
    w.set_targets([servo], [0.0])
    assert w.targets([servo])[0] == 0.0
    w.step(1500)
    q = w.state()[0][sc.joint_q(servo)]
    tau_g, kp = 1.0 * G * 0.3, 500.0
    assert q == pytest.approx(tau_g * math.cos(q) / kp, abs=2e-4)
    # the model's control range: a target beyond it is read back as written, tracked clipped
    w.set_targets([servo], [1.0])
    assert w.targets([servo])[0] == 1.0
    w.step(1500)
    q = w.state()[0][sc.joint_q(servo)]
    assert q == pytest.approx(0.3 + tau_g * math.cos(q) / kp, abs=5e-4)


def test_servo_force_is_limited_by_the_model(backend, load, contract_scene):
    """forcerange 1.5 N m < m g l = 2.94 N m: the arm sags until m g l cos(q) = 1.5."""
    sc = contract_scene
    weak = sc.joint("weak")
    w = load(backend, sc)
    w.set_targets([weak], [0.0])
    w.step(3000)
    q, qd = w.state()
    assert abs(qd[sc.joint_qd(weak)]) < 1e-3
    assert q[sc.joint_q(weak)] == pytest.approx(math.acos(1.5 / (G * 0.3)), abs=2e-3)


def test_frictionloss_is_a_deadband_then_coulomb_drag(backend, load, contract_scene):
    """The drag hinge (vertical axis, frictionloss 0.3 N m, nothing else): a torque below
    the loss holds it, one above it accelerates at (tau - loss) / I, and a spinning joint
    decelerates at loss / I. MuJoCo's frictionloss is a soft constraint: it creeps a few mrad."""
    sc = contract_scene
    drag = sc.joint("drag")
    a, v = sc.joint_q(drag), sc.joint_qd(drag)
    loss, inertia, T = 0.3, _mass_matrix_diag(sc, drag), 0.5
    n = int(T / sc.timestep)
    w = load(backend, sc)
    w.set_torque([drag], [0.2])
    w.step(n)
    q, qd = w.state()
    assert abs(q[a]) < 0.02 and abs(qd[v]) < 0.05, f"below the loss it moved {q[a]:.3f} rad"
    w.reset(sc.model.qpos0)
    w.set_torque([drag], [0.5])
    w.step(n)
    assert w.state()[1][v] == pytest.approx((0.5 - loss) / inertia * T, rel=1e-3)
    w.set_torque([drag], [0.0])
    qd0 = np.zeros(sc.model.nv)
    qd0[v] = 5.0
    w.reset(sc.model.qpos0, qd0)
    w.step(50)
    assert w.state()[1][v] == pytest.approx(5.0 - loss / inertia * 50 * sc.timestep, rel=1e-3)


# -- contacts -------------------------------------------------------------------


def test_contacts_threshold_and_freshness_after_reset(backend, load, contract_scene):
    sc = contract_scene
    die = _id(sc, "die")
    w = load(backend, sc)
    q = sc.model.qpos0.copy()
    s = sc.free_q(die)

    def put(z: float) -> frozenset[int]:
        q[s] = [0.5, -1.0, z, 1, 0, 0, 0]
        w.reset(q)
        return w.contacts(die)

    assert 0 in put(0.03 - 0.0005)        # 0.5 mm into the floor (world body 0)
    assert put(0.03 + 0.005) == frozenset()  # 5 mm gap: none, and nothing stale
    assert 0 in put(0.03 - 0.0005)
    assert die in w.contacts(0)           # symmetric
    w.step(200)                           # it rests: still in contact
    assert 0 in w.contacts(die)
    assert put(0.03 + 2 * CONTACT_DISTANCE_M) == frozenset()


@pytest.mark.parametrize("gap, touching", [(0.2e-3, True), (0.5e-3, True), (0.9e-3, True), (1.5e-3, False)])
def test_default_margin_gap_within_the_threshold_is_a_contact(backend, load, contract_scene, gap, touching):
    """The die has the default margin 0, where MuJoCo alone emits no separated point;
    the charter's <= CONTACT_DISTANCE_M still holds on every backend."""
    sc = contract_scene
    die = _id(sc, "die")
    w = load(backend, sc)
    q = sc.model.qpos0.copy()
    q[sc.free_q(die)] = [0.5, -1.0, 0.03 + gap, 1, 0, 0, 0]
    w.reset(q)
    assert w.contacts(die) == (frozenset({0}) if touching else frozenset()), f"{gap * 1000:.1f} mm"


@pytest.mark.parametrize("gap, touching", [
    (0.9 * CONTACT_DISTANCE_M, True), (1.5 * CONTACT_DISTANCE_M, False), (0.005, False)])
def test_contact_threshold_is_the_contract_s_not_the_engine_s(backend, load, contract_scene, gap, touching):
    """The 1 m slab: both engines report separated points out to ~1 cm here (MuJoCo: its
    margin; Bullet: its breaking threshold), so only the <= CONTACT_DISTANCE_M filter decides."""
    sc = contract_scene
    slab = sc.body("slab")
    w = load(backend, sc)
    q = sc.model.qpos0.copy()
    q[sc.free_q(slab)] = [-3.0, 0.0, 0.1 + gap, 1, 0, 0, 0]
    w.reset(q)
    assert w.contacts(slab) == (frozenset({0}) if touching else frozenset()), f"{gap * 1000:.1f} mm"


@pytest.mark.parametrize("gap, touching", [(0.5e-3, True), (1.5e-3, False)])
def test_mujoco_explicit_pair_reaches_the_threshold_and_steps_unchanged(load, tmp_path, gap, touching):
    """A ``<pair>`` contact (margin 0) reads the charter's threshold too, and widening
    contact detection leaves the stepped model's margins (its physics) as compiled."""
    from vector_os_nano.sim.scene import compiled_scene

    xml = tmp_path / "pair.xml"
    xml.write_text("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 0.1"/>
      <body name="ball"><freejoint/><geom name="ball" type="sphere" size="0.05" contype="0" conaffinity="0"/></body>
    </worldbody><contact><pair geom1="floor" geom2="ball"/></contact></mujoco>""")
    sc = compiled_scene(xml)
    w = load("mujoco", sc)
    ball = sc.body("ball")
    q = sc.model.qpos0.copy()
    q[sc.free_q(ball)] = [0.0, 0.0, 0.05 + gap, 1, 0, 0, 0]
    w.reset(q)
    assert w.contacts(ball) == (frozenset({0}) if touching else frozenset())
    for f in ("geom_margin", "geom_gap", "pair_margin", "pair_gap"):
        assert np.array_equal(getattr(w.model, f), getattr(sc.model, f)), f


# -- attach ------------------------------------------------------------------------


def test_attach_only_declared_pairs(backend, load, contract_scene):
    sc = contract_scene
    w = load(backend, sc)
    with pytest.raises(KeyError, match="grasp pairs.*carrier.*parcel"):
        w.attach(sc.body("carrier"), sc.body("die"))  # a compiled weld, but not a declared pair
    with pytest.raises(KeyError):
        w.attach(sc.body("parcel"), sc.body("carrier"))  # a declared pair reversed
    assert w.attached() == {}


def test_attach_holds_the_current_relative_pose(backend, load, contract_scene):
    sc = contract_scene
    carrier, parcel, rail = sc.body("carrier"), sc.body("parcel"), sc.joint("rail")
    w = load(backend, sc)

    def rel(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = canonical.body_poses(sc, q)
        return qmat(quat[carrier]).T @ (pos[parcel] - pos[carrier]), quat[parcel]

    w.attach(carrier, parcel)
    assert w.attached() == {parcel: carrier}
    p0, r0 = rel(w.state()[0])
    mass = 2.0 + 0.2
    drift = tilt = 0.0
    dt = sc.timestep
    for k in range(int(1.5 / dt)):
        t = k * dt
        accel = 2.0 if t < 0.5 else 10.0 * math.sin(2 * math.pi * 5 * (t - 0.5))  # push, then a 5 Hz shake
        w.set_torque([rail], [mass * accel])
        w.step(1)
        p, r = rel(w.state()[0])
        drift = max(drift, float(np.linalg.norm(p - p0)))
        tilt = max(tilt, float(quat_angle(r, r0)[0]))  # the carrier only translates
    q = w.state()[0]
    assert q[sc.joint_q(rail)] > 0.2  # it really moved
    assert drift < 1e-3, f"attached child drifted {drift * 1000:.2f} mm"
    assert tilt < 1e-2, f"attached child turned {tilt:.4f} rad"
    with pytest.raises(ValueError, match="already attached"):
        w.attach(sc.body("arm"), parcel)
    w.detach(parcel)
    assert w.attached() == {}
    w.detach(parcel)  # detaching a free child is a no-op
    z0 = w.state()[0][sc.free_q(parcel)][2]
    w.set_torque([rail], [0.0])
    w.step(100)
    assert w.state()[0][sc.free_q(parcel)][2] < z0 - 0.02  # released: it falls


# -- raycast ---------------------------------------------------------------------


def test_raycast_distance_body_exclusion_and_miss(backend, load, contract_scene):
    sc = contract_scene
    w = load(backend, sc)
    o = np.array([[0.0, -1.5, 0.5], [0.0, -1.5, 0.5], [0.0, -1.5, 0.5]])
    d = np.array([[1.0, 0, 0], [0, 0, 1.0], [math.cos(0.01), math.sin(0.01), 0]])
    dist, body = w.raycast(o, d)
    assert dist[0] == pytest.approx(0.95, abs=1e-6) and body[0] == sc.body("occluder")
    assert dist[1] == -1.0 and body[1] == -1  # straight up: nothing there
    assert dist[2] == pytest.approx(0.95 / math.cos(0.01), abs=1e-6) and body[2] == sc.body("occluder")
    dist, body = w.raycast(o, d, exclude_body=sc.body("occluder"))
    assert dist[0] == pytest.approx(1.95, abs=1e-6) and body[0] == sc.body("backstop")
    assert body[1] == -1


# -- render ------------------------------------------------------------------------


def _uv(frame: RGBDFrame, p: np.ndarray) -> tuple[float, float]:
    """World point -> continuous (u, v) through K and cam_mat (columns right, up, backward);
    pixel (row, col) covers [col, col + 1) x [row, row + 1)."""
    c = frame.cam_mat.T @ (p - frame.cam_pos)
    return frame.K[0, 2] + frame.K[0, 0] * c[0] / -c[2], frame.K[1, 2] - frame.K[1, 1] * c[1] / -c[2]


def _project(frame: RGBDFrame, p: np.ndarray) -> tuple[int, int]:
    u, v = _uv(frame, p)
    return int(v), int(u)


def test_pixel_centres_follow_K(backend, load, contract_scene):
    """The marker's pixel centroid lands where K puts its centre (a half-pixel shift fails)."""
    f = load(backend, contract_scene).render("cam", 640, 480, ("seg",))
    rows, cols = np.nonzero(f.seg == contract_scene.body("marker"))
    u, v = _uv(f, np.array([1.5, 0.3, 0.7]))
    assert abs(cols.mean() + 0.5 - u) < 0.25 and abs(rows.mean() + 0.5 - v) < 0.25


@pytest.mark.parametrize("size", [(320, 240), (100, 60)])
def test_render_shape_intrinsics_depth_and_segmentation(backend, load, contract_scene, size):
    sc = contract_scene
    w, h = size
    world = load(backend, sc)
    f = world.render("cam", w, h, ("rgb", "depth", "seg"))
    assert f.rgb.shape == (h, w, 3) and f.rgb.dtype == np.uint8
    assert f.depth.shape == (h, w) and f.depth.dtype == np.float32
    assert f.seg.shape == (h, w) and f.seg.dtype == np.int32
    fy = (h / 2) / math.tan(math.radians(60) / 2)
    np.testing.assert_allclose(f.K, [[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(f.cam_pos, [0, 0, 0.5], atol=1e-9)
    np.testing.assert_allclose(f.cam_mat, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], atol=1e-9)  # right, up, backward
    assert f.camera == "cam" and f.sim_time == world.time()
    # z-depth along the optical axis: the wall face is a plane at x = 1.95 everywhere
    r, c = _project(f, np.array([1.95, 0.0, 0.5]))
    assert f.depth[r, c] == pytest.approx(1.95, abs=2e-3)
    r, c = _project(f, np.array([1.95, -0.5, 0.15]))  # off-axis: still 1.95, not the ray length
    assert f.depth[r, c] == pytest.approx(1.95, abs=2e-3)
    assert f.seg[r, c] == sc.body("wall")
    assert f.seg[0, 0] == -1 and f.depth[0, 0] == 0.0  # sky: no body, invalid depth
    # canonical BODY ids (the wall has two geoms, so later geom ids are not body ids)
    assert sc.model.body_geomadr[sc.body("marker")] != sc.body("marker")
    for name, p in (("marker", [1.5, 0.3, 0.7]), ("ghost", [1.5, -0.3, 0.3])):
        r, c = _project(f, np.array(p))
        assert f.seg[r, c] == sc.body(name), name
    r, c = _project(f, np.array([1.5, 0.0, 0.8]))
    assert f.seg[r, c] == sc.body("wall")  # the invisible collision box is not rendered


def test_render_kinds_and_errors(backend, load, contract_scene):
    world = load(backend, contract_scene)
    f = world.render("cam", 64, 48, ("depth",))
    assert f.rgb is None and f.seg is None and f.depth.shape == (48, 64)
    with pytest.raises(KeyError, match="cam"):
        world.render("nope", 64, 48)
    with pytest.raises(ValueError):
        world.render("cam", 64, 48, ("normals",))


def test_render_from_a_second_thread_while_stepping(backend, load, contract_scene):
    world = load(backend, contract_scene)
    frames: list[RGBDFrame] = []
    errors: list[BaseException] = []

    def shoot() -> None:
        try:
            for _ in range(8):
                frames.append(world.render("cam", 64, 48, ("rgb", "depth", "seg")))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    world.set_torque([contract_scene.joint("spin")], [0.5])
    t0 = world.time()
    th = threading.Thread(target=shoot)
    th.start()
    while th.is_alive():
        world.step(10)
    th.join()
    assert not errors, errors
    times = [f.sim_time for f in frames]
    assert len(frames) == 8 and times == sorted(times)
    assert t0 <= times[0] and times[-1] <= world.time()
    for f in frames:
        assert f.rgb.shape == (48, 64, 3) and f.seg.shape == (48, 64)
        assert (f.seg == contract_scene.body("wall")).sum() > 100
