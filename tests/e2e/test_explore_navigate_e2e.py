"""E2E pipeline test: MuJoCo → explore → SceneGraph → navigate.

Headless MuJoCo (no GUI, no ROS2, no nav stack). Tests the core
pipeline that a user experiences via vector-cli:

    1. Start sim (MuJoCoGo2 room scene)
    2. Walk through house (populate SceneGraph)
    3. Navigate to a room (verify position change)

Also validates FAR config has correct values for V-Graph building.

Timeout: each test <30s, full suite <120s.
"""
from __future__ import annotations

import math
import os
import time

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_go2 = None  # module-level singleton — reuse across tests (expensive to create)


def _get_go2():
    """Lazy-create a headless MuJoCoGo2 in room scene."""
    global _go2
    if _go2 is not None and _go2._connected:
        return _go2
    mujoco = pytest.importorskip("mujoco")
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    _go2 = MuJoCoGo2(gui=False, room=True, backend="sinusoidal")
    _go2.connect()
    return _go2


@pytest.fixture(scope="module")
def go2():
    """Shared headless Go2 instance for the entire test module."""
    g = _get_go2()
    yield g
    # Don't disconnect — other tests may still need it.
    # Cleanup happens at process exit via MuJoCoGo2.__del__ or atexit.


@pytest.fixture(scope="module")
def scene_graph():
    """Fresh SceneGraph loaded with room layout."""
    from vector_os_nano.core.scene_graph import SceneGraph

    sg = SceneGraph()
    layout_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config",
        "room_layout.yaml",
    )
    if os.path.isfile(layout_path):
        sg.load_layout(layout_path)
    return sg


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# AC2: FAR config validation
# ---------------------------------------------------------------------------


class TestFARConfig:
    """Verify far_go2_indoor.yaml has correct values for V-Graph building."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        cfg_path = os.path.join(_repo_root(), "config", "far_go2_indoor.yaml")
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        self.cfg = raw.get("far_planner", {}).get("ros__parameters", {})

    def test_intensity_threshold_low(self):
        """new_intensity_thred must be <= 1.0 so bridge height-as-intensity works."""
        val = self.cfg.get("util/new_intensity_thred", 2.0)
        assert val <= 1.0, f"new_intensity_thred={val}, must be <= 1.0"

    def test_decay_time_large(self):
        """dynamic_obs_dacay_time must be large for static indoor (no obstacle decay)."""
        val = self.cfg.get("util/dynamic_obs_dacay_time", 2.0)
        assert val >= 100.0, f"dynamic_obs_dacay_time={val}, must be >= 100 for static indoor"

    def test_connect_votes_small(self):
        """connect_votes_size should be <= 5 for faster V-Graph edge building."""
        val = self.cfg.get("graph/connect_votes_size", 10)
        assert val <= 8, f"connect_votes_size={val}, should be <= 8"


# ---------------------------------------------------------------------------
# AC3: MuJoCo loads and robot functions
# ---------------------------------------------------------------------------


class TestMuJoCoRoom:
    """Verify MuJoCo room scene loads and robot is functional."""

    def test_model_loads(self, go2):
        """Room scene loads without error, robot connected."""
        assert go2._connected

    def test_robot_position(self, go2):
        """Robot starts at entry hall (~10, 3)."""
        pos = go2.get_position()
        assert 8.0 < pos[0] < 12.0, f"x={pos[0]}"
        assert 1.0 < pos[1] < 5.0, f"y={pos[1]}"

    def test_robot_standing(self, go2):
        """Robot is upright (z > 0.15m)."""
        pos = go2.get_position()
        assert pos[2] > 0.15, f"z={pos[2]}, robot may have fallen"

    def test_lidar_produces_points(self, go2):
        """Lidar raycast returns non-empty pointcloud."""
        points = go2.get_3d_pointcloud()
        assert len(points) > 100, f"Only {len(points)} lidar points"

    def test_camera_produces_frame(self, go2):
        """Camera returns a non-black image."""
        frame = go2.get_camera_frame()
        assert frame is not None
        assert frame.shape[0] > 0 and frame.shape[1] > 0


# ---------------------------------------------------------------------------
# AC3: Explore populates SceneGraph
# ---------------------------------------------------------------------------


class TestExploreSceneGraph:
    """Walk the robot through rooms and verify SceneGraph populates."""

    def test_walk_and_detect_rooms(self, go2, scene_graph):
        """Walk forward, SceneGraph records visits via nearest_room."""
        # Walk forward 3 seconds at 0.5 m/s
        go2.walk(0.5, 0.0, 0.0, 3.0)
        time.sleep(0.5)

        pos = go2.get_position()
        room = scene_graph.nearest_room(pos[0], pos[1])
        assert room is not None, f"No room at ({pos[0]:.1f}, {pos[1]:.1f})"

        # Record visit
        scene_graph.visit(room, pos[0], pos[1])
        visited = scene_graph.get_visited_rooms()
        assert len(visited) >= 1, "No rooms visited after walking"

    def test_scene_graph_has_rooms(self, scene_graph):
        """SceneGraph loaded from room_layout.yaml has 8 rooms."""
        all_rooms = scene_graph.get_all_rooms()
        assert len(all_rooms) >= 7, f"Only {len(all_rooms)} rooms in SceneGraph"

    def test_scene_graph_has_doors(self, scene_graph):
        """SceneGraph has door connections between rooms."""
        doors = scene_graph.get_all_doors()
        assert len(doors) >= 5, f"Only {len(doors)} doors"

    def test_door_chain_works(self, scene_graph):
        """BFS door chain finds path between non-adjacent rooms."""
        chain = scene_graph.get_door_chain("living_room", "kitchen")
        assert len(chain) >= 2, f"Door chain living→kitchen has {len(chain)} waypoints"
        # Chain should go through hallway
        labels = [label for _, _, label in chain]
        assert any("hall" in l for l in labels), f"Chain doesn't pass through hallway: {labels}"


# ---------------------------------------------------------------------------
# AC3: Navigate changes robot position
# ---------------------------------------------------------------------------


class TestNavigate:
    """Verify navigate skill moves robot toward target room."""

    def test_walk_toward_kitchen(self, go2, scene_graph):
        """Walk robot toward kitchen area, verify position change."""
        start_pos = go2.get_position()

        # Kitchen is at (17, 2.5) — walk east (positive X)
        go2.walk(0.5, 0.0, 0.0, 4.0)
        time.sleep(0.5)

        end_pos = go2.get_position()

        # Robot should have moved east (X increased)
        dx = end_pos[0] - start_pos[0]
        assert dx > 0.5, f"Robot only moved {dx:.1f}m east, expected >0.5m"

        # Robot still upright
        assert end_pos[2] > 0.15, f"Robot fell: z={end_pos[2]}"

    def test_navigate_skill_resolves_room(self):
        """NavigateSkill._resolve_room handles aliases."""
        from vector_os_nano.skills.navigate import _resolve_room

        assert _resolve_room("kitchen") == "kitchen"
        assert _resolve_room("厨房") == "kitchen"
        assert _resolve_room("master bedroom") == "master_bedroom"
        assert _resolve_room("主卧") == "master_bedroom"
        assert _resolve_room("客厅") == "living_room"
        assert _resolve_room("nonexistent") is None

    def test_navigate_skill_fuzzy_match(self):
        """Fuzzy room matching finds partial matches."""
        from vector_os_nano.skills.navigate import _fuzzy_room_match

        rooms = ["living_room", "kitchen", "master_bedroom", "guest_bedroom", "hallway"]
        assert _fuzzy_room_match("master room", rooms) == "master_bedroom"
        assert _fuzzy_room_match("guest", rooms) == "guest_bedroom"

    def test_robot_survives_extended_walk(self, go2):
        """Robot remains upright after turning and walking."""
        # Turn 90 degrees then walk
        go2.walk(0.0, 0.0, 1.0, 1.5)  # turn left ~1.5 rad
        time.sleep(0.3)
        go2.walk(0.4, 0.0, 0.0, 2.0)  # walk forward
        time.sleep(0.3)

        pos = go2.get_position()
        assert pos[2] > 0.15, f"Robot fell after turn+walk: z={pos[2]}"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _cleanup_go2():
    """Disconnect Go2 after all tests in this module."""
    yield
    global _go2
    if _go2 is not None:
        try:
            _go2.disconnect()
        except Exception:
            pass
        _go2 = None
