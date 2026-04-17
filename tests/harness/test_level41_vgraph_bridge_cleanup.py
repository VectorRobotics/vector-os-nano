"""L41: Bridge cleanup — /terrain_map and /terrain_map_ext must be removed.

RED phase: these tests assert the ABSENCE of bad code that still exists in
go2_vnav_bridge.py. All 6 tests must FAIL until Task 2 (revert) is applied.

Invariants enforced:
  - No _terrain_map_pub / _terrain_map_ext_pub publisher declarations
  - No _sync_terrain_to_far method
  - No _publish_accumulated_terrain method
  - _replay_terrain publishes ONLY to _pc_pub (/registered_scan)
  - No /terrain_map or /terrain_map_ext topic strings in __init__
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nav_debug_helpers import read_bridge_source


# ---------------------------------------------------------------------------
# Helpers (copied from test_level36_terrain_replay.py)
# ---------------------------------------------------------------------------

def _get_method_body(source: str, method_name: str) -> str:
    """Return source slice for a method in Go2VNavBridge."""
    start = source.find(f"    def {method_name}(")
    if start < 0:
        raise AssertionError(f"Method {method_name!r} not found in bridge source")
    end = source.find("\n    def ", start + 1)
    return source[start:end] if end > 0 else source[start:]


def _get_init_body(source: str) -> str:
    """Return source slice for Go2VNavBridge.__init__."""
    cls_start = source.find("class Go2VNavBridge")
    body = source[cls_start:]
    start = body.find("    def __init__(")
    end = body.find("\n    def ", start + 1)
    return body[start:end] if end > 0 else body[start:]


# ---------------------------------------------------------------------------
# Tests — all must FAIL (RED) until the bridge cleanup is applied
# ---------------------------------------------------------------------------

class TestVGraphBridgeCleanup:

    def test_bridge_has_no_terrain_map_publisher_declaration(self):
        """After cleanup, __init__ must NOT declare _terrain_map_pub."""
        src = read_bridge_source()
        init = _get_init_body(src)
        assert "_terrain_map_pub" not in init, (
            "_terrain_map_pub must be removed from __init__ — "
            "bridge must not publish to /terrain_map"
        )

    def test_bridge_has_no_terrain_map_ext_publisher_declaration(self):
        """After cleanup, __init__ must NOT declare _terrain_map_ext_pub."""
        src = read_bridge_source()
        init = _get_init_body(src)
        assert "_terrain_map_ext_pub" not in init, (
            "_terrain_map_ext_pub must be removed from __init__ — "
            "bridge must not publish to /terrain_map_ext"
        )

    def test_bridge_has_no_sync_terrain_to_far_method(self):
        """After cleanup, _sync_terrain_to_far must not exist anywhere in bridge."""
        src = read_bridge_source()
        assert "def _sync_terrain_to_far" not in src, (
            "_sync_terrain_to_far must be removed — "
            "it duplicates terrainAnalysis publisher with wrong intensity semantics"
        )

    def test_bridge_has_no_publish_accumulated_terrain_method(self):
        """After cleanup, _publish_accumulated_terrain must not exist in bridge."""
        src = read_bridge_source()
        assert "def _publish_accumulated_terrain" not in src, (
            "_publish_accumulated_terrain must be removed — "
            "it is a legacy wrapper around the bad _sync_terrain_to_far"
        )

    def test_replay_terrain_publishes_only_registered_scan(self):
        """_replay_terrain must publish ONLY to _pc_pub (/registered_scan)."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        assert "_pc_pub.publish" in body, (
            "_replay_terrain must still publish to _pc_pub (/registered_scan)"
        )
        assert "_terrain_map_pub.publish" not in body, (
            "_replay_terrain must NOT publish to _terrain_map_pub"
        )
        assert "_terrain_map_ext_pub.publish" not in body, (
            "_replay_terrain must NOT publish to _terrain_map_ext_pub"
        )

    def test_no_terrain_map_topic_string_in_init(self):
        """After cleanup, /terrain_map and /terrain_map_ext must not appear in __init__."""
        src = read_bridge_source()
        init = _get_init_body(src)
        assert '"/terrain_map"' not in init, (
            '"/terrain_map" topic string must be removed from __init__'
        )
        assert '"/terrain_map_ext"' not in init, (
            '"/terrain_map_ext" topic string must be removed from __init__'
        )
