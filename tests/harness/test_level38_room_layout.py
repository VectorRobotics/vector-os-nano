# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""L38: Room layout loading + position-based room detection (sim mode).

Tests that SceneGraph.load_layout() properly seeds rooms and doors from
config/worlds/apartment.yaml, enabling instant nearest_room() without VLM.
"""
from __future__ import annotations

import os
import textwrap

import pytest

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.scene_nodes import RoomNode


@pytest.fixture
def layout_file(tmp_path):
    """Create a minimal room layout YAML."""
    content = textwrap.dedent("""\
        rooms:
          kitchen:    [17.0, 2.5]
          hallway:    [10.0, 5.0]
          study:      [17.0, 7.5]
          living_room: [3.0, 2.5]
        doors:
          kitchen-hallway: [13.5, 3.0]
          study-hallway:   [13.5, 8.0]
    """)
    p = tmp_path / "room_layout.yaml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def full_layout_file(tmp_path):
    """Full 8-room layout matching go2_room.xml."""
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "worlds", "apartment.yaml",
    )
    if os.path.isfile(src):
        import shutil
        dst = tmp_path / "room_layout.yaml"
        shutil.copy(src, dst)
        return str(dst)
    pytest.skip("config/worlds/apartment.yaml not found")


class TestLoadLayout:
    """SceneGraph.load_layout() seeds rooms from YAML."""

    def test_loads_rooms(self, layout_file):
        sg = SceneGraph()
        count = sg.load_layout(layout_file)
        assert count == 4

    def test_rooms_have_positions(self, layout_file):
        sg = SceneGraph()
        sg.load_layout(layout_file)
        kitchen = sg.get_room("kitchen")
        assert kitchen is not None
        assert abs(kitchen.center_x - 17.0) < 0.1
        assert abs(kitchen.center_y - 2.5) < 0.1

    def test_rooms_have_high_visit_count(self, layout_file):
        """Seeded rooms should have visit_count >= 3 (trusted by navigate)."""
        sg = SceneGraph()
        sg.load_layout(layout_file)
        for room in sg.get_all_rooms():
            assert room.visit_count >= 3, f"{room.room_id} visit_count too low"

    def test_loads_doors(self, layout_file):
        sg = SceneGraph()
        sg.load_layout(layout_file)
        door = sg.get_door("kitchen", "hallway")
        assert door is not None
        assert abs(door[0] - 13.5) < 0.1
        assert abs(door[1] - 3.0) < 0.1

    def test_door_chain_works_after_load(self, layout_file):
        sg = SceneGraph()
        sg.load_layout(layout_file)
        chain = sg.get_door_chain("kitchen", "study")
        assert len(chain) > 0, "Should find path kitchen → hallway → study"

    def test_nearest_room_works_after_load(self, layout_file):
        sg = SceneGraph()
        sg.load_layout(layout_file)
        assert sg.nearest_room(17.0, 2.5) == "kitchen"
        assert sg.nearest_room(10.0, 5.0) == "hallway"
        assert sg.nearest_room(3.0, 2.5) == "living_room"

    def test_returns_zero_for_missing_file(self):
        sg = SceneGraph()
        assert sg.load_layout("/nonexistent/path.yaml") == 0

    def test_returns_zero_for_bad_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("not: a: valid: [layout")
        sg = SceneGraph()
        assert sg.load_layout(str(p)) == 0

    def test_full_layout_8_rooms(self, full_layout_file):
        sg = SceneGraph()
        count = sg.load_layout(full_layout_file)
        assert count == 8

    def test_full_layout_all_rooms_navigable(self, full_layout_file):
        """Every room should be reachable from every other room via doors."""
        sg = SceneGraph()
        sg.load_layout(full_layout_file)
        rooms = [r.room_id for r in sg.get_all_rooms()]
        for src in rooms:
            for dst in rooms:
                if src != dst:
                    chain = sg.get_door_chain(src, dst)
                    assert len(chain) > 0, f"No path from {src} to {dst}"


class TestExploreWithLayout:
    """Explore uses position-based room detection when layout is loaded."""

    def test_no_vlm_room_discovery_in_explore(self):
        """explore.py should NOT have VLM room discovery logic."""
        import inspect
        from vector_os_nano.skills.go2 import explore
        source = inspect.getsource(explore._exploration_loop)
        assert "_vlm_discover" not in source, "VLM discovery should be removed"
        assert "_VLM_ROOM_INTERVAL" not in source, "VLM interval should be removed"
        assert "_NEW_ROOM_DIST" not in source, "VLM distance check should be removed"

    def test_explore_uses_nearest_room(self):
        """explore.py should use SceneGraph.nearest_room() for room detection."""
        import inspect
        from vector_os_nano.skills.go2 import explore
        source = inspect.getsource(explore._exploration_loop)
        assert "nearest_room" in source


class TestConfigFileExists:
    """Verify config/worlds/apartment.yaml is present and valid."""

    def test_config_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "worlds", "apartment.yaml",
        )
        assert os.path.isfile(path), "config/worlds/apartment.yaml must exist for sim"

    def test_config_has_8_rooms(self):
        import yaml
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "worlds", "apartment.yaml",
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data.get("rooms", {})) == 8

    def test_config_has_7_doors(self):
        import yaml
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "worlds", "apartment.yaml",
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data.get("doors", {})) == 7, "Should have 7 doors (each room to hallway)"
