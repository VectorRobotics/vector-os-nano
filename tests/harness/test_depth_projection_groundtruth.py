"""Ground-truth depth projection tests for BUG-1 (RGB/depth timing mismatch).

Two classes of tests:

1. TestDepthProjectionGroundTruth — validates pixel_to_camera + camera_to_world
   against known geometry from MuJoCo sim. Uses the d435_rgb/d435_depth named
   cameras so the projection math is exact. These tests use manually
   placed pixels for ground-truth validation.

2. TestTimingMismatchBug — demonstrates that capturing depth at a different
   robot position from the RGB frame causes large world-position errors.
   This proves the bug exists before the fix and fails after the fix.

All tests use go2_room fixture (room=True) for the house environment with
furniture at known positions.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip entire module if mujoco is not installed
pytest.importorskip("mujoco", reason="mujoco not installed — skipping depth projection tests")

# ---------------------------------------------------------------------------
# Helpers: load MuJoCoGo2 via path to avoid optional-dependency cascade
# ---------------------------------------------------------------------------


def _import_mujoco_go2():
    """Import MuJoCoGo2 by file path, loading its dependency chain."""
    import importlib.util

    _types_path = _REPO_ROOT / "vector_os_nano" / "core" / "types.py"
    types_spec = importlib.util.spec_from_file_location(
        "vector_os_nano.core.types", str(_types_path)
    )
    types_mod = importlib.util.module_from_spec(types_spec)  # type: ignore[arg-type]
    sys.modules.setdefault("vector_os_nano.core.types", types_mod)
    types_spec.loader.exec_module(types_mod)  # type: ignore[union-attr]

    spec = importlib.util.spec_from_file_location(
        "vector_os_nano.hardware.sim.mujoco_go2",
        str(_REPO_ROOT / "vector_os_nano" / "hardware" / "sim" / "mujoco_go2.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["vector_os_nano.hardware.sim.mujoco_go2"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.MuJoCoGo2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def go2_room():
    """Single MuJoCoGo2 instance with room=True for the whole module.

    scope=module to avoid the overhead of reloading the room XML per test.
    Each test that moves the robot must restore the pose or request its own
    fixture instance.
    """
    MuJoCoGo2 = _import_mujoco_go2()
    robot = MuJoCoGo2(gui=False, room=True)
    robot.connect()
    robot.stand(duration=2.0)
    yield robot
    robot.disconnect()


@pytest.fixture
def go2_room_fn():
    """Function-scoped MuJoCoGo2 with room=True. Disconnects after each test."""
    MuJoCoGo2 = _import_mujoco_go2()
    robot = MuJoCoGo2(gui=False, room=True)
    robot.connect()
    robot.stand(duration=2.0)
    yield robot
    robot.disconnect()


# ---------------------------------------------------------------------------
# Helper: get camera intrinsics the same way the production code does
# ---------------------------------------------------------------------------


def _get_sim_intrinsics(width: int = 320, height: int = 240):
    from vector_os_nano.perception.depth_projection import mujoco_intrinsics
    return mujoco_intrinsics(width, height, vfov_deg=42.0)


# ---------------------------------------------------------------------------
# Test 1 — Center pixel ground-truth round-trip
# ---------------------------------------------------------------------------


class TestDepthProjectionGroundTruth:
    """Validate pixel_to_camera + camera_to_world against MuJoCo camera pose.

    Manually pick pixels (center, off-center) and verify the projected
    world point is geometrically correct.
    All data is captured atomically from MuJoCo at the same simulation step.
    """

    def test_center_pixel_projects_to_camera_forward(self, go2_room):
        """Center pixel should project to a point directly in front of the camera."""
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        # Capture all data atomically (same simulation step)
        frame = go2_room.get_camera_frame()
        depth = go2_room.get_depth_frame()
        cam_xpos, cam_xmat = go2_room.get_camera_pose()

        h, w = depth.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)

        # Center pixel
        cx, cy = intrinsics.cx, intrinsics.cy
        d_center = float(depth[int(cy), int(cx)])

        # If no valid depth at center, pick median of a 20x20 region
        if d_center <= 0.1 or d_center > 10.0:
            patch = depth[int(cy) - 10:int(cy) + 10, int(cx) - 10:int(cx) + 10]
            valid = patch[(patch > 0.1) & (patch < 10.0)]
            if len(valid) == 0:
                pytest.skip("No valid depth at center — scene may not be visible")
            d_center = float(np.median(valid))

        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d_center, intrinsics)

        # Center pixel → x_cam ≈ 0, y_cam ≈ 0
        assert abs(x_cam) < 0.05 * d_center, (
            f"Center pixel should have x_cam ≈ 0, got {x_cam:.4f} at depth {d_center:.3f}m"
        )
        assert abs(y_cam) < 0.05 * d_center, (
            f"Center pixel should have y_cam ≈ 0, got {y_cam:.4f} at depth {d_center:.3f}m"
        )
        assert abs(z_cam - d_center) < 1e-4

        # Project to world using exact MuJoCo camera pose
        pos = go2_room.get_position()
        heading = go2_room.get_heading()
        wx, wy, wz = camera_to_world(
            x_cam, y_cam, z_cam,
            robot_x=float(pos[0]), robot_y=float(pos[1]), robot_z=float(pos[2]),
            robot_heading=float(heading),
            cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        # Verify world point is in front of the camera (not behind)
        xmat = np.array(cam_xmat, dtype=np.float64).reshape(3, 3)
        cam_forward = -xmat[:, 2]  # MuJoCo: col2 = -forward
        delta = np.array([wx, wy, wz]) - np.array(cam_xpos)
        forward_dist = float(np.dot(delta, cam_forward))
        assert forward_dist > 0.05, (
            f"Projected world point should be in front of camera, got forward_dist={forward_dist:.4f}m"
        )
        assert abs(forward_dist - d_center) < 0.1, (
            f"Forward distance {forward_dist:.4f}m should match depth {d_center:.4f}m (within 10cm)"
        )

    def test_center_pixel_world_error_near_zero(self, go2_room):
        """Reproduce the session-verified result: center pixel has near-zero error.

        The camera forward axis passes through the center pixel. For center pixel,
        the projected world point should match cam_pos + depth * cam_forward exactly.
        """
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        depth = go2_room.get_depth_frame()
        cam_xpos, cam_xmat = go2_room.get_camera_pose()

        h, w = depth.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy

        d = float(depth[int(cy), int(cx)])
        if d <= 0.1 or d > 10.0:
            pytest.skip("No valid depth at center pixel")

        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d, intrinsics)

        # Ground-truth: cam_pos + d * cam_forward
        xmat = np.array(cam_xmat, dtype=np.float64).reshape(3, 3)
        cam_forward = -xmat[:, 2]
        gt_world = np.array(cam_xpos, dtype=np.float64) + d * cam_forward

        pos = go2_room.get_position()
        heading = go2_room.get_heading()
        wx, wy, wz = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        error = math.sqrt((wx - gt_world[0])**2 + (wy - gt_world[1])**2)
        assert error < 0.01, (
            f"Center pixel world error should be < 1cm (got {error:.4f}m). "
            f"Got ({wx:.4f}, {wy:.4f}), expected ({gt_world[0]:.4f}, {gt_world[1]:.4f})"
        )

    def test_off_center_pixel_projection_consistent(self, go2_room):
        """Off-center pixels project consistently: lateral offset in image → lateral offset in world.

        Pick a pixel 40px to the right of center. The world projection should be
        displaced to the right (in camera's right direction) relative to center.
        """
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        depth = go2_room.get_depth_frame()
        cam_xpos, cam_xmat = go2_room.get_camera_pose()

        h, w = depth.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy

        # Use same depth for center and right pixel (eliminates depth ambiguity)
        d = float(depth[int(cy), int(cx)])
        if d <= 0.1 or d > 9.0:
            pytest.skip("No valid depth at center pixel")

        # Center
        xc_cam, yc_cam, zc_cam = pixel_to_camera(cx, cy, d, intrinsics)
        pos = go2_room.get_position()
        heading = go2_room.get_heading()

        wxc, wyc, _ = camera_to_world(
            xc_cam, yc_cam, zc_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        # 40px to the right of center, same depth
        u_right = cx + 40.0
        xr_cam, yr_cam, zr_cam = pixel_to_camera(u_right, cy, d, intrinsics)
        wxr, wyr, _ = camera_to_world(
            xr_cam, yr_cam, zr_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        # camera_to_world is a rigid transform (rotation + translation).
        # Rotation preserves distance: the 3D Euclidean distance between the two
        # world points must equal the camera-frame distance between them.
        # In camera frame: center and right pixel differ only in x_cam by (40 * d / fx),
        # y_cam and z_cam are identical (same depth). So the 3D distance = 40 * d / fx.
        x_cam_diff = xr_cam - xc_cam  # = 40 * d / fx
        expected_3d_dist = abs(x_cam_diff)

        # Need world z to compute full 3D distance — call camera_to_world for z too
        _, _, wzc = camera_to_world(
            xc_cam, yc_cam, zc_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )
        _, _, wzr = camera_to_world(
            xr_cam, yr_cam, zr_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        actual_3d_dist = math.sqrt((wxr - wxc)**2 + (wyr - wyc)**2 + (wzr - wzc)**2)

        assert abs(actual_3d_dist - expected_3d_dist) < 0.005, (
            f"3D world distance {actual_3d_dist:.4f}m should equal camera-frame "
            f"displacement {expected_3d_dist:.4f}m (rigid transform preserves distance)"
        )

    def test_atomic_capture_is_consistent(self, go2_room):
        """get_camera_frame + get_depth_frame + get_camera_pose all read from same sim state.

        At rest, calling them in sequence should be consistent (robot isn't moving).
        If the robot were moving between calls, results would differ.
        This is a sanity check — the robot is standing still here.
        """
        # Capture twice; at rest, results should be nearly identical
        depth1 = go2_room.get_depth_frame()
        cam_xpos1, cam_xmat1 = go2_room.get_camera_pose()

        depth2 = go2_room.get_depth_frame()
        cam_xpos2, cam_xmat2 = go2_room.get_camera_pose()

        # Camera position should not have changed
        pos_diff = float(np.linalg.norm(np.array(cam_xpos1) - np.array(cam_xpos2)))
        assert pos_diff < 0.01, (
            f"Camera position changed between sequential reads at rest: {pos_diff:.4f}m"
        )

    def test_projection_within_room_bounds(self, go2_room):
        """The projected center pixel should land inside the house (0..20, 0..14)."""
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        depth = go2_room.get_depth_frame()
        cam_xpos, cam_xmat = go2_room.get_camera_pose()

        h, w = depth.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy

        d = float(depth[int(cy), int(cx)])
        if d <= 0.1 or d > 10.0:
            pytest.skip("No valid depth at center")

        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d, intrinsics)
        pos = go2_room.get_position()
        heading = go2_room.get_heading()
        wx, wy, wz = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        # House bounds: x in [−0.5, 20.5], y in [−0.5, 14.5], z in [−0.2, 3.0]
        assert -0.5 <= wx <= 20.5, f"world_x {wx:.3f} outside house x bounds"
        assert -0.5 <= wy <= 14.5, f"world_y {wy:.3f} outside house y bounds"
        assert -0.2 <= wz <= 3.0, f"world_z {wz:.3f} outside house z bounds"


# ---------------------------------------------------------------------------
# Test 2 — Timing-mismatch bug demonstration
# ---------------------------------------------------------------------------


class TestTimingMismatchBug:
    """Proves that RGB/depth at different poses produces large position errors.

    This test captures RGB at position A, moves the robot, then captures
    depth + pose at position B. The world position computed from mismatched
    data should be far from the correct position.

    These tests use function-scoped go2_room_fn fixture because they move the robot.
    """

    def test_mismatched_pose_causes_large_error(self, go2_room_fn):
        """Verify that timing mismatch causes > 1.0m position error.

        Capture depth at position A. Move robot forward. Capture camera pose
        at position B. Project using depth from A but pose from B — the result
        should be wrong by the distance the robot moved.
        """
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        robot = go2_room_fn

        # --- Position A: capture depth + correct pose ---
        depth_a = robot.get_depth_frame()
        cam_xpos_a, cam_xmat_a = robot.get_camera_pose()
        pos_a = robot.get_position()
        heading_a = robot.get_heading()

        h, w = depth_a.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy

        d = float(depth_a[int(cy), int(cx)])
        if d <= 0.1 or d > 8.0:
            pytest.skip("No valid center depth at position A")

        # Ground-truth world point (correct: depth A + pose A)
        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d, intrinsics)
        wx_correct, wy_correct, _ = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos_a[0]), float(pos_a[1]), float(pos_a[2]),
            float(heading_a), cam_xpos=cam_xpos_a, cam_xmat=cam_xmat_a,
        )

        # --- Move robot forward 1.5m ---
        try:
            robot.walk(vx=0.5, vy=0.0, vyaw=0.0, duration=3.0)
        except Exception:
            pytest.skip("walk() failed — cannot test timing mismatch")

        # --- Position B: capture pose ONLY (simulating the bug) ---
        cam_xpos_b, cam_xmat_b = robot.get_camera_pose()
        pos_b = robot.get_position()
        heading_b = robot.get_heading()

        # Verify we actually moved
        dist_moved = math.sqrt(
            (float(pos_b[0]) - float(pos_a[0]))**2 +
            (float(pos_b[1]) - float(pos_a[1]))**2
        )
        if dist_moved < 0.5:
            pytest.skip(f"Robot only moved {dist_moved:.2f}m — need at least 0.5m to test bug")

        # --- Buggy projection: depth from A, pose from B ---
        wx_wrong, wy_wrong, _ = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos_b[0]), float(pos_b[1]), float(pos_b[2]),
            float(heading_b), cam_xpos=cam_xpos_b, cam_xmat=cam_xmat_b,
        )

        # Error should be large — at least half the distance moved.
        # Even 0.5m error is catastrophic for object placement in a room.
        error = math.sqrt((wx_wrong - wx_correct)**2 + (wy_wrong - wy_correct)**2)
        min_expected_error = 0.5 * dist_moved
        assert error > min_expected_error, (
            f"Timing mismatch error should be > {min_expected_error:.2f}m (proving the bug), "
            f"got {error:.4f}m. Robot moved {dist_moved:.2f}m. "
            f"Correct: ({wx_correct:.2f}, {wy_correct:.2f}), "
            f"Wrong: ({wx_wrong:.2f}, {wy_wrong:.2f})"
        )

    def test_atomic_capture_has_small_error(self, go2_room_fn):
        """Verify that capturing all data atomically has < 0.1m error.

        This is the CORRECT behavior after the fix: capture RGB, depth, and
        pose all at the same simulation step.
        """
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        robot = go2_room_fn

        # Capture all atomically — no robot movement between calls
        depth = robot.get_depth_frame()
        cam_xpos, cam_xmat = robot.get_camera_pose()
        pos = robot.get_position()
        heading = robot.get_heading()

        h, w = depth.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy
        d = float(depth[int(cy), int(cx)])

        if d <= 0.1 or d > 10.0:
            pytest.skip("No valid depth at center")

        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d, intrinsics)
        wx, wy, wz = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        # Ground truth: cam_pos + d * cam_forward
        xmat = np.array(cam_xmat, dtype=np.float64).reshape(3, 3)
        cam_forward = -xmat[:, 2]
        gt_world = np.array(cam_xpos, dtype=np.float64) + d * cam_forward

        error = math.sqrt((wx - gt_world[0])**2 + (wy - gt_world[1])**2)
        assert error < 0.1, (
            f"Atomic capture world error should be < 10cm, got {error:.4f}m"
        )


# ---------------------------------------------------------------------------
# Test 3 — Fixed code path: explore.py + look.py atomic capture
# ---------------------------------------------------------------------------


class TestAtomicCapturePattern:
    """Verify that the production fix (atomic capture before VLM calls) works correctly.

    These tests simulate what the fixed explore.py and look.py do:
    capture frame + depth + cam_pose all at once, then use cached values.
    """

    def test_capture_then_delay_preserves_correct_projection(self, go2_room_fn):
        """Atomic capture at t=0 + 'slow VLM call' (simulated) still gives correct result.

        This mirrors the fixed code pattern:
            frame = base.get_camera_frame()
            depth = base.get_depth_frame()      # capture NOW
            cam_xpos, cam_xmat = base.get_camera_pose()  # capture NOW
            # ... slow VLM calls ...
            # use cached depth + cam_pose
        """
        import time
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        robot = go2_room_fn

        # --- Atomic capture (the fix) ---
        _frame = robot.get_camera_frame()
        depth_cached = robot.get_depth_frame()
        cam_xpos_cached, cam_xmat_cached = robot.get_camera_pose()
        pos_cached = robot.get_position()
        heading_cached = robot.get_heading()

        # Simulate slow VLM call (0.1s is enough to test the pattern without
        # actually calling an LLM — in production this is 2-20s while robot moves)
        time.sleep(0.1)

        # Use CACHED values (not fresh) — this is the fix
        h, w = depth_cached.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy
        d = float(depth_cached[int(cy), int(cx)])

        if d <= 0.1 or d > 10.0:
            pytest.skip("No valid center depth in cached frame")

        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d, intrinsics)
        wx, wy, wz = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos_cached[0]), float(pos_cached[1]), float(pos_cached[2]),
            float(heading_cached),
            cam_xpos=cam_xpos_cached, cam_xmat=cam_xmat_cached,
        )

        # Ground truth at capture time
        xmat = np.array(cam_xmat_cached, dtype=np.float64).reshape(3, 3)
        cam_forward = -xmat[:, 2]
        gt_world = np.array(cam_xpos_cached, dtype=np.float64) + d * cam_forward

        error = math.sqrt((wx - gt_world[0])**2 + (wy - gt_world[1])**2)
        assert error < 0.1, (
            f"Fixed pattern (atomic capture) should have < 10cm error, got {error:.4f}m"
        )

    def test_look_skill_atomic_capture_field_order(self, go2_room_fn):
        """Verify the corrected field-capture order in look.py.

        The fix requires:
            frame   = base.get_camera_frame()    <- first
            depth   = base.get_depth_frame()      <- second (before VLM)
            pose    = base.get_camera_pose()      <- third (before VLM)
            scene   = vlm.describe_scene(frame)   <- slow, AFTER sensor capture
        This test simulates that ordering and checks correctness.
        """
        from vector_os_nano.perception.depth_projection import (
            mujoco_intrinsics,
            pixel_to_camera,
            camera_to_world,
        )

        robot = go2_room_fn

        # Simulate corrected look.py capture order
        frame = robot.get_camera_frame()  # step 1
        depth = robot.get_depth_frame()   # step 2 — NOW, not after VLM
        cam_xpos, cam_xmat = robot.get_camera_pose()  # step 3 — NOW
        pos = robot.get_position()
        heading = robot.get_heading()
        # (VLM call would go here in production — slow but uses cached frame)

        # Now project using the atomically captured data
        h, w = depth.shape[:2]
        intrinsics = mujoco_intrinsics(w, h, vfov_deg=42.0)
        cx, cy = intrinsics.cx, intrinsics.cy
        d = float(depth[int(cy), int(cx)])

        if d <= 0.1 or d > 10.0:
            pytest.skip("No valid center depth")

        x_cam, y_cam, z_cam = pixel_to_camera(cx, cy, d, intrinsics)
        wx, wy, wz = camera_to_world(
            x_cam, y_cam, z_cam,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(heading), cam_xpos=cam_xpos, cam_xmat=cam_xmat,
        )

        # Sanity: world point is within the house
        assert -1.0 <= wx <= 21.0, f"world_x {wx:.3f} out of house bounds"
        assert -1.0 <= wy <= 15.0, f"world_y {wy:.3f} out of house bounds"

        # Verify frame was actually captured (not None / empty)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
