"""Level 30: Terrain map persistence harness.

Tests verify:
  - TerrainAccumulator merges pointclouds into 2D voxel grid
  - Save/load roundtrip preserves data
  - File size is reasonable (< 5MB)
  - Bridge wiring: accumulate, auto-save, startup replay
  - /clear_memory deletes terrain file
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nav_debug_helpers import read_bridge_source

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BRIDGE = os.path.join(_REPO, "scripts", "go2_vnav_bridge.py")


def _load_accumulator():
    """Import TerrainAccumulator from bridge without importing ROS2."""
    src = open(_BRIDGE).read()
    start = src.find("class TerrainAccumulator")
    end = src.find("\nclass Go2VNavBridge")
    ns = {}
    exec(src[start:end], ns)
    return ns["TerrainAccumulator"]


# ===================================================================
# Part 1: TerrainAccumulator behavioral tests
# ===================================================================

class TestTerrainAccumulatorBasic:

    def test_add_merges_to_single_voxel(self):
        """Two points in same 0.1m voxel merge to one entry."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        t.add([(1.0, 2.0, 0.5, 0.1), (1.05, 2.05, 0.6, 0.2)])
        assert t.size == 1

    def test_add_keeps_max_z(self):
        """Voxel stores max z across all added points."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        t.add([(1.0, 2.0, 0.3, 0.1)])
        t.add([(1.0, 2.0, 0.8, 0.1)])
        t.add([(1.0, 2.0, 0.5, 0.1)])
        pts = t.to_pointcloud()
        assert len(pts) == 1
        assert pts[0][2] == pytest.approx(0.8, abs=0.01)

    def test_add_filters_z_range(self):
        """Points outside z_min/z_max are rejected."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1, z_min=0.0, z_max=1.0)
        t.add([(1.0, 2.0, -1.0, 0.1)])  # below z_min
        t.add([(1.0, 2.0, 5.0, 0.1)])   # above z_max
        assert t.size == 0

    def test_grid_resolution(self):
        """Points 0.15m apart in different voxels at 0.1m resolution."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        t.add([(0.0, 0.0, 0.5, 0.1), (0.15, 0.0, 0.5, 0.1)])
        assert t.size == 2

    def test_point_count_tracks_raw_input(self):
        """point_count tracks total raw points, not unique voxels."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        t.add([(1.0, 2.0, 0.5, 0.1)] * 100)
        assert t.point_count == 100
        assert t.size == 1


class TestTerrainSaveLoad:

    def test_save_load_roundtrip(self):
        """Save → load → data matches."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        for i in range(50):
            t.add([(float(i) * 0.2, float(i) * 0.3, 0.5, 0.1)])
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            assert t.save(path)
            t2 = Acc()
            assert t2.load(path)
            assert t2.size == t.size
            pts1 = sorted(t.to_pointcloud())
            pts2 = sorted(t2.to_pointcloud())
            for p1, p2 in zip(pts1, pts2):
                assert p1[0] == pytest.approx(p2[0], abs=0.01)
                assert p1[1] == pytest.approx(p2[1], abs=0.01)
                assert p1[2] == pytest.approx(p2[2], abs=0.01)
        finally:
            os.unlink(path)

    def test_save_empty_returns_false(self):
        """Cannot save empty accumulator."""
        Acc = _load_accumulator()
        t = Acc()
        assert not t.save("/tmp/test_empty.npz")

    def test_load_missing_returns_false(self):
        """Loading nonexistent file returns False."""
        Acc = _load_accumulator()
        t = Acc()
        assert not t.load("/tmp/nonexistent_terrain.npz")

    def test_file_size_reasonable(self):
        """Terrain map for 20x30m house at 0.1m should be < 5MB."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        # Simulate dense scan: 200x300 = 60k voxels
        for ix in range(200):
            for iy in range(0, 300, 3):  # every 3rd = 20k voxels
                t.add([(ix * 0.1, iy * 0.1, 0.5, 0.1)])
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            t.save(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            assert size_mb < 5.0, f"Terrain file {size_mb:.1f}MB exceeds 5MB limit"
        finally:
            os.unlink(path)

    def test_to_pointcloud_format(self):
        """to_pointcloud returns (x, y, z, intensity) tuples."""
        Acc = _load_accumulator()
        t = Acc(voxel_size=0.1)
        t.add([(5.0, 10.0, 1.2, 0.3)])
        pts = t.to_pointcloud()
        assert len(pts) == 1
        x, y, z, intensity = pts[0]
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert isinstance(intensity, float)


# ===================================================================
# Part 2: Bridge wiring (static analysis)
# ===================================================================

class TestBridgeTerrainWiring:

    def test_bridge_has_terrain_accumulator(self):
        """Bridge creates TerrainAccumulator instance."""
        src = read_bridge_source()
        assert "TerrainAccumulator" in src
        assert "_terrain_acc" in src

    def test_bridge_has_terrain_map_path(self):
        """Bridge configures terrain map save path."""
        src = read_bridge_source()
        assert "terrain_map.npz" in src
        assert "_terrain_map_path" in src

    def test_accumulate_in_publish_pointcloud(self):
        """_publish_pointcloud calls _terrain_acc.add when nav enabled."""
        src = read_bridge_source()
        pc_start = src.find("def _publish_pointcloud")
        pc_end = src.find("\n    def ", pc_start + 1)
        pc_body = src[pc_start:pc_end]
        assert "_terrain_acc.add" in pc_body, (
            "_publish_pointcloud must call _terrain_acc.add()"
        )
        assert "_nav_enabled" in pc_body, (
            "Terrain accumulation should only happen when nav is enabled"
        )

    def test_auto_save_timer(self):
        """Bridge has auto-save terrain timer."""
        src = read_bridge_source()
        assert "_auto_save_terrain" in src
        assert "save_terrain" in src

    def test_save_terrain_method(self):
        """Bridge has save_terrain() method."""
        src = read_bridge_source()
        assert "def save_terrain" in src

    def test_terrain_replay_on_startup(self):
        """Bridge loads terrain and replays on startup."""
        src = read_bridge_source()
        assert "_replay_terrain" in src
        assert "terrain_replay" in src

    def test_replay_publishes_pointcloud(self):
        """Replay method publishes PointCloud2."""
        src = read_bridge_source()
        replay_start = src.find("def _replay_terrain")
        if replay_start < 0:
            pytest.fail("_replay_terrain method not found")
        replay_end = src.find("\n    def ", replay_start + 1)
        replay_body = src[replay_start:replay_end]
        assert "PointCloud2" in replay_body or "_pc_pub.publish" in replay_body

    def test_replay_frame_is_map(self):
        """Replay publishes in 'map' frame — set by _build_terrain_pc2 helper."""
        src = read_bridge_source()
        # _replay_terrain delegates frame setup to _build_terrain_pc2
        helper_start = src.find("def _build_terrain_pc2")
        if helper_start < 0:
            pytest.fail("_build_terrain_pc2 not found")
        helper_end = src.find("\n    def ", helper_start + 1)
        helper_body = src[helper_start:helper_end]
        assert '"map"' in helper_body


class TestClearMemoryTerrain:

    def test_clear_memory_deletes_terrain(self):
        """CLI /clear_memory handler should delete terrain_map.npz."""
        cli_path = os.path.join(
            _REPO, "vector_os_nano", "vcli", "cli.py"
        )
        with open(cli_path) as f:
            src = f.read()
        assert "terrain_map" in src, (
            "/clear_memory handler must delete terrain_map.npz"
        )
