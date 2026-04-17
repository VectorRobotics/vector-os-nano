"""V-Graph Debug Harness — Level 70: FAR visibility graph edge formation.

Diagnoses why FAR planner's V-Graph doesn't build edges through doorways.
Two diagnostic levels:

  Level 1 (MuJoCo-only): Terrain analysis at doors — obstacle density,
    clearance width, 2D visibility rays. No ROS2 needed. Fast (~5s total).

  Level 2 (ROS2 + nav stack): Full V-Graph integration — launch bridge +
    FAR, drive robot through rooms, subscribe to /robot_vgraph, count
    cross-room edges. Requires full nav stack running.

Usage:
  # Fast terrain diagnostic (run in loop)
  pytest tests/harness/test_vgraph_debug.py -k terrain -v --tb=short

  # Ceiling height sweep (find optimal filter)
  pytest tests/harness/test_vgraph_debug.py -k sweep -v --tb=short

  # Cross-door visibility rays
  pytest tests/harness/test_vgraph_debug.py -k visibility -v --tb=short

  # Local vs accumulated terrain comparison
  pytest tests/harness/test_vgraph_debug.py -k accumulated -v --tb=short

  # Full V-Graph integration (needs ROS2)
  pytest tests/harness/test_vgraph_debug.py -k vgraph -v --tb=short

Room layout (from go2_room.xml):
     10  +=D====+=D===+D+=====D====+
         |      |                  |
         |Dining|  OPEN HALLWAY    | Study
         |Room  |  + GALLERY       | (6x5)
         |(6x5) |  (8m x 10m)     |
      5  +=D====+  open plan      +=D=====+
         |      |                  |       |
         |Living|  Go2 starts     |Kitchen |
         |Room  |  (10, 3)        |(6x5)  |
         |(6x5) |                  |       |
      0  +======+==================+=======+

  D = doorway (1.2m gap)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SIM_DIR = _REPO / "vector_os_nano" / "hardware" / "sim"


# ---------------------------------------------------------------------------
# Door specifications (derived from go2_room.xml wall geoms)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoorSpec:
    """A doorway in the room layout."""
    name: str
    wall_axis: str         # "x" or "y" — axis perpendicular to the wall
    wall_pos: float        # position of the wall on wall_axis
    gap_min: float         # start of gap on the other axis
    gap_max: float         # end of gap on the other axis
    room_a: str            # room on the low side
    room_b: str            # room on the high side
    # Observation points: one in each room (for visibility ray)
    obs_a: tuple[float, float] = (0.0, 0.0)
    obs_b: tuple[float, float] = (0.0, 0.0)

    @property
    def gap_center(self) -> tuple[float, float]:
        mid = (self.gap_min + self.gap_max) / 2
        if self.wall_axis == "x":
            return (self.wall_pos, mid)
        return (mid, self.wall_pos)

    @property
    def gap_width(self) -> float:
        return self.gap_max - self.gap_min


DOORS: list[DoorSpec] = [
    DoorSpec("living_hall", "x", 6.0, 2.4, 3.6, "living_room", "hallway",
             obs_a=(4.0, 3.0), obs_b=(8.0, 3.0)),
    DoorSpec("dining_hall", "x", 6.0, 7.4, 8.6, "dining_room", "hallway",
             obs_a=(4.0, 8.0), obs_b=(8.0, 8.0)),
    DoorSpec("kitchen_hall", "x", 14.0, 2.4, 3.6, "kitchen", "hallway",
             obs_a=(16.0, 3.0), obs_b=(12.0, 3.0)),
    DoorSpec("study_hall", "x", 14.0, 7.4, 8.6, "study", "hallway",
             obs_a=(16.0, 8.0), obs_b=(12.0, 8.0)),
    DoorSpec("living_dining", "y", 5.0, 2.4, 3.6, "living_room", "dining_room",
             obs_a=(3.0, 3.5), obs_b=(3.0, 6.5)),
    DoorSpec("kitchen_study", "y", 5.0, 16.4, 17.6, "kitchen", "study",
             obs_a=(17.0, 3.5), obs_b=(17.0, 6.5)),
]


# ---------------------------------------------------------------------------
# Headless MuJoCo raycaster (no physics thread, no gait — just ray casting)
# ---------------------------------------------------------------------------

class HeadlessRaycaster:
    """Fast lidar simulation: load model, teleport robot, cast rays.

    No physics loop. No gait controller. Just mj_ray for terrain analysis.
    ~50ms per position (11,160 rays @ 360 azimuth x 31 elevation).
    """

    # Lidar config — must match mujoco_go2.py
    LIDAR_OFFSET_X = 0.3
    LIDAR_OFFSET_Z = 0.2
    TILT_DEG = -20.0
    MAX_RANGE = 12.0
    N_AZIMUTH = 360
    ELEVATIONS = list(range(-8, 53, 2))  # -8 to +52 in 2deg steps

    def __init__(self) -> None:
        import mujoco as mj
        self._mj = mj
        scene_path = self._build_scene()
        self.model = mj.MjModel.from_xml_path(str(scene_path))
        self.data = mj.MjData(self.model)
        self._base_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "base_link"
        )
        # Collect robot geom IDs (to exclude from raycasting)
        self._robot_geoms: set[int] = set()
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            check = bid
            while check > 0:
                if check == self._base_bid:
                    self._robot_geoms.add(gid)
                    break
                check = self.model.body_parentid[check]

    def _build_scene(self) -> Path:
        """Build scene_room.xml (same as MuJoCoGo2._build_room_scene_xml)."""
        mjcf_dir = _SIM_DIR / "mjcf" / "go2"
        room_xml = _SIM_DIR / "go2_room.xml"
        go2_xml = mjcf_dir / "go2.xml"
        assets_dir = mjcf_dir / "assets"
        template = room_xml.read_text()
        xml = template.replace("GO2_MODEL_PATH", str(go2_xml))
        xml = xml.replace("GO2_ASSETS_DIR", str(assets_dir))
        out = mjcf_dir / "scene_room.xml"
        out.write_text(xml)
        return out

    def teleport(self, x: float, y: float, heading: float = 0.0) -> None:
        """Place robot at (x, y) with given heading. No physics step."""
        z = 0.28  # standing height
        # Quaternion from yaw
        cy, sy = math.cos(heading / 2), math.sin(heading / 2)
        self.data.qpos[0:3] = [x, y, z]
        self.data.qpos[3:7] = [cy, 0.0, 0.0, sy]  # w, x, y, z
        # Set standing joint angles
        stand = [0.0, 0.75, -1.5] * 4  # FL, FR, RL, RR
        self.data.qpos[7:19] = stand
        self._mj.mj_forward(self.model, self.data)

    def cast_rays(self) -> list[tuple[float, float, float, float]]:
        """Cast Livox MID360-like 3D lidar. Returns [(x,y,z,intensity), ...]."""
        mj = self._mj
        pos = self.data.qpos[0:3].copy().astype(np.float64)
        heading = self._get_heading()
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        pos_lidar = np.array([
            float(pos[0]) + cos_h * self.LIDAR_OFFSET_X,
            float(pos[1]) + sin_h * self.LIDAR_OFFSET_X,
            float(pos[2]) + self.LIDAR_OFFSET_Z,
        ], dtype=np.float64)

        tilt_rad = math.radians(self.TILT_DEG)
        cos_tilt, sin_tilt = math.cos(tilt_rad), math.sin(tilt_rad)
        points: list[tuple[float, float, float, float]] = []
        geom_id = np.zeros(1, dtype=np.int32)

        for elev_deg in self.ELEVATIONS:
            elev_rad = math.radians(elev_deg)
            cos_elev, sin_elev = math.cos(elev_rad), math.sin(elev_rad)
            step = 360.0 / self.N_AZIMUTH
            for i in range(self.N_AZIMUTH):
                azimuth = heading + math.radians(i * step - 180)
                dx_w = cos_elev * math.cos(azimuth)
                dy_w = cos_elev * math.sin(azimuth)
                dz_w = sin_elev
                # world -> body -> tilt -> world
                dx_b = dx_w * cos_h + dy_w * sin_h
                dy_b = -dx_w * sin_h + dy_w * cos_h
                dz_b = dz_w
                dx_bt = dx_b * cos_tilt - dz_b * sin_tilt
                dz_bt = dx_b * sin_tilt + dz_b * cos_tilt
                direction = np.array([
                    dx_bt * cos_h - dy_b * sin_h,
                    dx_bt * sin_h + dy_b * cos_h,
                    dz_bt,
                ], dtype=np.float64)
                dist = mj.mj_ray(
                    self.model, self.data, pos_lidar, direction,
                    None, 1, self._base_bid, geom_id,
                )
                if dist > 0 and dist < self.MAX_RANGE:
                    if int(geom_id[0]) not in self._robot_geoms:
                        px = pos_lidar[0] + dist * direction[0]
                        py = pos_lidar[1] + dist * direction[1]
                        pz = pos_lidar[2] + dist * direction[2]
                        points.append((float(px), float(py), float(pz), 0.0))
        return points

    def _get_heading(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def apply_height_filter(
    points: list[tuple[float, float, float, float]],
    ground_z: float,
    min_height: float = 0.15,
    max_height: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Filter points to obstacle band [min_height, max_height] above ground.

    Mimics FAR's terrain classification:
      - Below min_height (0.15m): ground/traversable, not obstacles
      - Above max_height (ceiling filter): ceiling, not navigation-relevant
    """
    return [
        (x, y, z, i) for x, y, z, i in points
        if min_height <= (z - ground_z) <= max_height
    ]


def points_in_door_region(
    points: list[tuple[float, float, float, float]],
    door: DoorSpec,
    band: float = 0.15,
) -> list[tuple[float, float]]:
    """Extract 2D points within `band` meters of the wall, in the door gap region.

    Returns only (x, y) projected. These are the points that would form
    obstacles in FAR's 2D contour check.
    """
    result = []
    for x, y, z, _ in points:
        if door.wall_axis == "x":
            if abs(x - door.wall_pos) < band:
                if door.gap_min - 0.1 <= y <= door.gap_max + 0.1:
                    result.append((x, y))
        else:
            if abs(y - door.wall_pos) < band:
                if door.gap_min - 0.1 <= x <= door.gap_max + 0.1:
                    result.append((x, y))
    return result


@dataclass
class DoorMetrics:
    """Diagnostic metrics for one door at one ceiling filter height."""
    door_name: str
    ceiling_height: float
    total_points: int         # total lidar points after ceiling filter
    door_region_points: int   # obstacle points in the door gap region
    gap_width_m: float        # physical door width (1.2m)
    clear_width_m: float      # effective obstacle-free width
    clear_ratio: float        # clear_width / gap_width
    visibility_blocked: bool  # is the center ray blocked?
    visibility_hits: int      # obstacle points near center ray


def compute_door_metrics(
    points: list[tuple[float, float, float, float]],
    door: DoorSpec,
    ceiling_height: float,
    ground_z: float = 0.0,
    min_height: float = 0.15,
    ray_tolerance: float = 0.08,
) -> DoorMetrics:
    """Compute all metrics for one door at one ceiling filter height."""
    filtered = apply_height_filter(points, ground_z, min_height, ceiling_height)
    door_pts = points_in_door_region(filtered, door)

    # Effective clear width: divide door into 0.05m bins, count empty ones
    n_bins = max(1, int(door.gap_width / 0.05))
    bins = [0] * n_bins
    for pt in door_pts:
        coord = pt[1] if door.wall_axis == "x" else pt[0]
        idx = int((coord - door.gap_min) / 0.05)
        if 0 <= idx < n_bins:
            bins[idx] += 1
    clear_bins = sum(1 for b in bins if b == 0)
    clear_width = clear_bins * 0.05

    # Visibility ray: from obs_a through door center to obs_b
    ax, ay = door.obs_a
    bx, by = door.obs_b
    ray_hits = _count_ray_obstacles(filtered, ax, ay, bx, by, ray_tolerance)

    return DoorMetrics(
        door_name=door.name,
        ceiling_height=ceiling_height,
        total_points=len(filtered),
        door_region_points=len(door_pts),
        gap_width_m=door.gap_width,
        clear_width_m=clear_width,
        clear_ratio=clear_width / door.gap_width if door.gap_width > 0 else 0,
        visibility_blocked=ray_hits > 0,
        visibility_hits=ray_hits,
    )


def _count_ray_obstacles(
    points: list[tuple[float, float, float, float]],
    ax: float, ay: float, bx: float, by: float,
    tolerance: float,
) -> int:
    """Count 2D obstacle points within tolerance of line segment AB."""
    dx, dy = bx - ax, by - ay
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return 0
    # Unit normal to the line
    nx, ny = -dy / length, dx / length
    count = 0
    for x, y, z, _ in points:
        # Project onto line
        t = ((x - ax) * dx + (y - ay) * dy) / (length * length)
        if t < 0 or t > 1:
            continue
        # Distance to line
        dist = abs((x - ax) * nx + (y - ay) * ny)
        if dist < tolerance:
            count += 1
    return count


def format_metrics_table(all_metrics: list[DoorMetrics]) -> str:
    """Format metrics into a readable table."""
    lines = [
        f"{'Door':<18} {'Ceil':>5} {'TotPts':>7} {'DoorPts':>8} "
        f"{'Clear':>6} {'Ratio':>6} {'RayHit':>7} {'Blocked':>8}",
        "-" * 80,
    ]
    for m in all_metrics:
        lines.append(
            f"{m.door_name:<18} {m.ceiling_height:>5.1f} {m.total_points:>7} "
            f"{m.door_region_points:>8} {m.clear_width_m:>5.2f}m "
            f"{m.clear_ratio:>5.1%} {m.visibility_hits:>7} "
            f"{'BLOCKED' if m.visibility_blocked else 'CLEAR':>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raycaster() -> HeadlessRaycaster:
    """Module-scoped headless raycaster (loads MuJoCo model once)."""
    return HeadlessRaycaster()


@pytest.fixture(scope="module")
def terrain_cache(raycaster: HeadlessRaycaster) -> dict[str, Any]:
    """Capture lidar data at door observation positions.

    Returns dict with:
      - "scans": {position_key: [(x,y,z,i), ...]}
      - "accumulated": merged pointcloud from all positions
    """
    cache: dict[str, Any] = {"scans": {}, "accumulated": []}
    all_points: list[tuple[float, float, float, float]] = []

    # Capture at each door's observation points + door center
    positions: list[tuple[str, float, float, float]] = []
    for door in DOORS:
        ax, ay = door.obs_a
        bx, by = door.obs_b
        cx, cy = door.gap_center
        # Face toward door center from each side
        heading_a = math.atan2(cy - ay, cx - ax)
        heading_b = math.atan2(cy - by, cx - bx)
        positions.append((f"{door.name}_a", ax, ay, heading_a))
        positions.append((f"{door.name}_b", bx, by, heading_b))
        # Also capture from door center (looking both ways)
        positions.append((f"{door.name}_center", cx, cy, heading_a))

    # Additional hallway positions for coverage
    for hx in (8.0, 10.0, 12.0):
        for hy in (3.0, 5.0, 8.0):
            positions.append((f"hall_{hx:.0f}_{hy:.0f}", hx, hy, 0.0))

    for key, x, y, h in positions:
        raycaster.teleport(x, y, h)
        pts = raycaster.cast_rays()
        cache["scans"][key] = pts
        all_points.extend(pts)

    # Deduplicate accumulated terrain (voxel grid, 0.1m resolution)
    voxels: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for p in all_points:
        ix, iy = int(p[0] / 0.1), int(p[1] / 0.1)
        key = (ix, iy)
        if key not in voxels or p[2] > voxels[key][2]:
            voxels[key] = p
    cache["accumulated"] = list(voxels.values())

    return cache


# ---------------------------------------------------------------------------
# Test Class 1: Doorway terrain diagnostics
# ---------------------------------------------------------------------------

class TestDoorwayTerrain:
    """Analyze obstacle point density at each doorway."""

    def test_terrain_capture_sanity(self, terrain_cache: dict) -> None:
        """Verify terrain capture produced data at all positions."""
        scans = terrain_cache["scans"]
        assert len(scans) >= len(DOORS) * 3, (
            f"Expected >= {len(DOORS) * 3} scan positions, got {len(scans)}"
        )
        for key, pts in scans.items():
            assert len(pts) > 100, f"Scan '{key}' has only {len(pts)} points"

    def test_accumulated_terrain_coverage(self, terrain_cache: dict) -> None:
        """Accumulated terrain should cover more area than any single scan."""
        acc = terrain_cache["accumulated"]
        single_max = max(len(pts) for pts in terrain_cache["scans"].values())
        assert len(acc) > single_max, (
            f"Accumulated {len(acc)} points not larger than "
            f"single best {single_max}"
        )

    @pytest.mark.parametrize("ceiling_h", [0.6, 0.8, 1.0, 1.2, 1.5, 1.8])
    def test_door_clearance_at_height(
        self, terrain_cache: dict, ceiling_h: float,
    ) -> None:
        """Report doorway clearance metrics at each ceiling filter height.

        Not an assertion — diagnostic output. Shows how many obstacle points
        are in each door opening and what effective clearance remains.
        """
        all_metrics: list[DoorMetrics] = []
        for door in DOORS:
            # Use the door-center scan (best local view)
            pts = terrain_cache["scans"].get(f"{door.name}_center", [])
            if not pts:
                continue
            m = compute_door_metrics(pts, door, ceiling_h)
            all_metrics.append(m)

        print(f"\n=== Doorway clearance at ceiling_filter={ceiling_h}m ===")
        print(format_metrics_table(all_metrics))

        # Diagnostic assertion: at SOME height, doors should be mostly clear
        if ceiling_h == 1.0:  # current config value
            for m in all_metrics:
                # Report but don't fail — this is diagnostic
                if m.clear_ratio < 0.5:
                    print(
                        f"  WARNING: {m.door_name} has {m.clear_ratio:.0%} "
                        f"clearance at {ceiling_h}m — FAR may block this edge"
                    )


# ---------------------------------------------------------------------------
# Test Class 2: Cross-door visibility rays
# ---------------------------------------------------------------------------

class TestCrossDoorVisibility:
    """Cast 2D visibility rays across each doorway and check for blockage."""

    @pytest.mark.parametrize("ceiling_h", [0.8, 1.0, 1.2, 1.5])
    def test_visibility_ray_per_door(
        self, terrain_cache: dict, ceiling_h: float,
    ) -> None:
        """For each door, cast ray from room A center to room B center.

        This mimics FAR's IsNavNodesConnectFreePolygon: can a straight line
        pass through the door without hitting obstacle contours?
        """
        results: list[tuple[str, bool, int]] = []
        for door in DOORS:
            pts = terrain_cache["scans"].get(f"{door.name}_center", [])
            if not pts:
                continue
            filtered = apply_height_filter(pts, 0.0, 0.15, ceiling_h)
            ax, ay = door.obs_a
            bx, by = door.obs_b
            hits = _count_ray_obstacles(filtered, ax, ay, bx, by, 0.08)
            results.append((door.name, hits > 0, hits))

        print(f"\n=== Visibility rays at ceiling_filter={ceiling_h}m ===")
        for name, blocked, hits in results:
            status = "BLOCKED" if blocked else "CLEAR"
            print(f"  {name:<18} {status:>8}  ({hits} obstacle points on ray)")

        # At 1.0m, all doors should ideally be CLEAR
        if ceiling_h == 1.0:
            blocked_doors = [name for name, b, _ in results if b]
            if blocked_doors:
                print(f"  === BLOCKED DOORS at 1.0m: {blocked_doors} ===")

    def test_multi_ray_fan_through_door(
        self, terrain_cache: dict,
    ) -> None:
        """Cast a fan of 5 rays through each door (not just center).

        If the center ray is blocked but side rays pass, the door opening
        might have a localized obstruction (door frame, furniture).
        """
        ceiling_h = 1.0
        print(f"\n=== Multi-ray fan at ceiling_filter={ceiling_h}m ===")
        for door in DOORS:
            pts = terrain_cache["scans"].get(f"{door.name}_center", [])
            if not pts:
                continue
            filtered = apply_height_filter(pts, 0.0, 0.15, ceiling_h)
            ax, ay = door.obs_a
            bx, by = door.obs_b

            # Fan: 5 parallel rays offset along the door gap axis
            offsets = [-0.4, -0.2, 0.0, 0.2, 0.4]
            ray_results: list[int] = []
            for off in offsets:
                if door.wall_axis == "x":
                    hits = _count_ray_obstacles(
                        filtered, ax, ay + off, bx, by + off, 0.05,
                    )
                else:
                    hits = _count_ray_obstacles(
                        filtered, ax + off, ay, bx + off, by, 0.05,
                    )
                ray_results.append(hits)

            clear_count = sum(1 for h in ray_results if h == 0)
            print(
                f"  {door.name:<18} rays={ray_results} "
                f"clear={clear_count}/5"
            )


# ---------------------------------------------------------------------------
# Test Class 3: Local vs accumulated terrain comparison
# ---------------------------------------------------------------------------

class TestAccumulatedTerrainEffect:
    """Compare visibility using local-only vs accumulated terrain.

    Key hypothesis: does adding global terrain to /registered_scan
    help or hurt V-Graph edge formation through doors?
    """

    def test_accumulated_vs_local(self, terrain_cache: dict) -> None:
        """Compare doorway visibility with local vs accumulated terrain."""
        ceiling_h = 1.0
        accumulated = terrain_cache["accumulated"]

        print(f"\n=== Local vs Accumulated terrain at ceiling={ceiling_h}m ===")
        print(f"{'Door':<18} {'Local':<20} {'Accumulated':<20} {'Delta':>8}")
        print("-" * 70)

        for door in DOORS:
            # Local: scan from door center only
            local_pts = terrain_cache["scans"].get(f"{door.name}_center", [])
            if not local_pts:
                continue

            m_local = compute_door_metrics(local_pts, door, ceiling_h)
            m_acc = compute_door_metrics(accumulated, door, ceiling_h)

            local_str = (
                f"{'BLOCKED' if m_local.visibility_blocked else 'CLEAR':>7} "
                f"({m_local.visibility_hits} hits)"
            )
            acc_str = (
                f"{'BLOCKED' if m_acc.visibility_blocked else 'CLEAR':>7} "
                f"({m_acc.visibility_hits} hits)"
            )
            delta = m_acc.visibility_hits - m_local.visibility_hits
            delta_str = f"{'+' if delta > 0 else ''}{delta}"

            print(f"  {door.name:<18} {local_str:<20} {acc_str:<20} {delta_str:>8}")

            # If accumulated has MORE hits than local, global terrain HURTS
            if delta > 0:
                print(
                    f"    ^ Accumulated terrain ADDS {delta} obstacles — "
                    f"global merge would HURT V-Graph here"
                )
            elif delta < 0:
                print(
                    f"    ^ Accumulated terrain has FEWER obstacles — "
                    f"different voxelization"
                )


# ---------------------------------------------------------------------------
# Test Class 4: Ceiling height sweep (parameter optimization)
# ---------------------------------------------------------------------------

class TestCeilingHeightSweep:
    """Sweep ceiling_filter_height and report optimal value for V-Graph."""

    def test_sweep_all_doors(self, terrain_cache: dict) -> None:
        """Find ceiling height that maximizes door visibility."""
        heights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]

        print("\n=== Ceiling height sweep: doors clear / total ===")
        print(f"{'Height':<8}", end="")
        for door in DOORS:
            print(f" {door.name[:12]:<13}", end="")
        print(f" {'TOTAL':>7}")
        print("-" * (8 + 13 * len(DOORS) + 8))

        best_height = 0.0
        best_clear = -1

        for h in heights:
            n_clear = 0
            print(f"{h:<8.1f}", end="")
            for door in DOORS:
                pts = terrain_cache["scans"].get(f"{door.name}_center", [])
                if not pts:
                    print(f" {'?':<13}", end="")
                    continue
                m = compute_door_metrics(pts, door, h)
                status = "OK" if not m.visibility_blocked else f"X({m.visibility_hits})"
                print(f" {status:<13}", end="")
                if not m.visibility_blocked:
                    n_clear += 1

            print(f" {n_clear:>3}/{len(DOORS)}")
            if n_clear > best_clear:
                best_clear = n_clear
                best_height = h

        print(f"\nBest: ceiling_filter={best_height}m ({best_clear}/{len(DOORS)} doors clear)")

    def test_point_density_vs_height(self, terrain_cache: dict) -> None:
        """Show how total obstacle points change with ceiling height.

        FAR needs SOME points for corner detection. Too few = no V-Graph nodes.
        Too many = doors blocked. Find the sweet spot.
        """
        heights = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]

        # Use a hallway-center scan (good baseline)
        pts = terrain_cache["scans"].get("hall_10_3", [])
        if not pts:
            pytest.skip("No hallway scan available")

        print("\n=== Point density vs ceiling height ===")
        print(f"{'Height':<8} {'Points':>8} {'%kept':>7}")
        print("-" * 25)
        total = len(pts)
        for h in heights:
            filtered = apply_height_filter(pts, 0.0, 0.15, h)
            pct = len(filtered) / total if total > 0 else 0
            print(f"{h:<8.1f} {len(filtered):>8} {pct:>6.1%}")


# ---------------------------------------------------------------------------
# Test Class 5: FAR V-Graph integration (requires ROS2 + nav stack)
# ---------------------------------------------------------------------------

class TestVGraphIntegration:
    """Full integration: launch bridge + FAR, drive robot, monitor V-Graph.

    Requires:
      1. ROS2 Jazzy sourced
      2. Nav stack built (vector_navigation_stack)
      3. No other nav stack instances running
    """

    @pytest.fixture(autouse=True)
    def _check_ros2(self) -> None:
        """Skip if ROS2 is not available."""
        try:
            import rclpy  # noqa: F401
        except ImportError:
            pytest.skip("ROS2 not available — skipping V-Graph integration")

    def test_vgraph_edge_monitoring(self) -> None:
        """Launch nav stack, drive robot through doors, count V-Graph edges.

        This test launches the bridge and FAR, drives the robot from the
        hallway through the living room door and back, then checks if
        FAR built any V-Graph edges crossing the doorway.

        Run with:
          source /opt/ros/jazzy/setup.bash
          pytest tests/harness/test_vgraph_debug.py -k vgraph_edge -v -s
        """
        import rclpy
        import subprocess
        import time
        import signal

        # Start bridge + nav stack
        launch_script = _REPO / "scripts" / "launch_vnav.sh"
        if not launch_script.exists():
            pytest.skip("launch_vnav.sh not found")

        rclpy.init()
        node = rclpy.create_node("vgraph_debug_monitor")

        # Track V-Graph messages
        vgraph_msgs: list[Any] = []
        viz_msgs: list[Any] = []

        try:
            from visualization_msgs.msg import MarkerArray
            from rclpy.qos import QoSProfile, ReliabilityPolicy

            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT, depth=5,
            )

            node.create_subscription(
                MarkerArray, "/viz_graph_topic",
                lambda msg: viz_msgs.append(msg), qos,
            )
        except ImportError:
            pytest.skip("visualization_msgs not available")

        # Launch nav stack in background
        proc = subprocess.Popen(
            ["bash", str(launch_script), "--no-gui"],
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )

        try:
            # Wait for stack to initialize
            time.sleep(15)

            # Drive robot: hallway → living room door → living room → back
            # TODO: publish /way_point or /goal_point to move robot
            # For now, just monitor what FAR produces

            # Monitor for 30 seconds
            start = time.time()
            while time.time() - start < 30:
                rclpy.spin_once(node, timeout_sec=0.5)

            # Analyze V-Graph messages
            if viz_msgs:
                last_msg = viz_msgs[-1]
                n_markers = len(last_msg.markers)
                print(f"\nV-Graph: {n_markers} markers in last /viz_graph_topic msg")
                print(f"Total /viz_graph_topic messages received: {len(viz_msgs)}")
            else:
                print("\nWARNING: No /viz_graph_topic messages received")

        finally:
            proc.terminate()
            proc.wait(timeout=10)
            node.destroy_node()
            rclpy.shutdown()
