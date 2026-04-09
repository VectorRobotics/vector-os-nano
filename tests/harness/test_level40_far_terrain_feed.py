"""L40: FAR terrain feed — periodic accumulated terrain publishing.

Validates that the bridge periodically publishes accumulated terrain
directly to /terrain_map and /terrain_map_ext during exploration,
bypassing terrainAnalysis's ~3.5m range filter so FAR gets full
terrain coverage for V-Graph building.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

BRIDGE_PATH = _REPO / "scripts" / "go2_vnav_bridge.py"


def _read_bridge_source() -> str:
    return BRIDGE_PATH.read_text()


# ---------------------------------------------------------------------------
# L40-1: _build_terrain_pc2 helper exists
# ---------------------------------------------------------------------------


def test_build_terrain_pc2_method_exists() -> None:
    """Bridge must have _build_terrain_pc2() to avoid PointCloud2 building duplication."""
    source = _read_bridge_source()
    assert "def _build_terrain_pc2" in source, \
        "Bridge must have _build_terrain_pc2() method"


def test_build_terrain_pc2_sets_map_frame() -> None:
    """_build_terrain_pc2 must set frame_id='map'."""
    source = _read_bridge_source()
    idx = source.find("def _build_terrain_pc2")
    assert idx > 0
    method_body = source[idx:idx + 800]
    assert '"map"' in method_body or "'map'" in method_body, \
        "_build_terrain_pc2 must set frame_id='map'"


def test_build_terrain_pc2_has_intensity_field() -> None:
    """_build_terrain_pc2 must include intensity in PointField list."""
    source = _read_bridge_source()
    idx = source.find("def _build_terrain_pc2")
    assert idx > 0
    method_body = source[idx:idx + 500]
    assert "intensity" in method_body, \
        "_build_terrain_pc2 must include intensity field"


# ---------------------------------------------------------------------------
# L40-2: _publish_accumulated_terrain method
# ---------------------------------------------------------------------------


def test_publish_accumulated_terrain_exists() -> None:
    """Bridge must have _publish_accumulated_terrain() for periodic FAR feeding."""
    source = _read_bridge_source()
    assert "def _publish_accumulated_terrain" in source


def test_publish_accumulated_terrain_publishes_to_terrain_map() -> None:
    """_publish_accumulated_terrain must publish to /terrain_map."""
    source = _read_bridge_source()
    idx = source.find("def _publish_accumulated_terrain")
    assert idx > 0
    method_body = source[idx:idx + 500]
    assert "_terrain_map_pub" in method_body, \
        "Must publish to /terrain_map"


def test_publish_accumulated_terrain_publishes_to_terrain_map_ext() -> None:
    """_publish_accumulated_terrain must publish to /terrain_map_ext."""
    source = _read_bridge_source()
    idx = source.find("def _publish_accumulated_terrain")
    assert idx > 0
    method_body = source[idx:idx + 500]
    assert "_terrain_map_ext_pub" in method_body, \
        "Must publish to /terrain_map_ext"


def test_publish_accumulated_terrain_uses_accumulator() -> None:
    """_publish_accumulated_terrain must read from _terrain_acc."""
    source = _read_bridge_source()
    idx = source.find("def _publish_accumulated_terrain")
    assert idx > 0
    method_body = source[idx:idx + 500]
    assert "_terrain_acc" in method_body, \
        "Must use terrain accumulator data"


# ---------------------------------------------------------------------------
# L40-3: _auto_save_terrain triggers periodic FAR sync
# ---------------------------------------------------------------------------


def test_auto_save_terrain_calls_publish_accumulated() -> None:
    """_auto_save_terrain must periodically call _publish_accumulated_terrain."""
    source = _read_bridge_source()
    idx = source.find("def _auto_save_terrain")
    assert idx > 0
    # Find end of method (next def or end of indentation)
    next_def = source.find("\n    def ", idx + 10)
    method_body = source[idx:next_def] if next_def > 0 else source[idx:idx + 500]
    assert "_publish_accumulated_terrain" in method_body, \
        "_auto_save_terrain must call _publish_accumulated_terrain"


def test_auto_save_terrain_has_interval_guard() -> None:
    """_auto_save_terrain must NOT publish every 30s — needs interval (e.g., every 60s)."""
    source = _read_bridge_source()
    idx = source.find("def _auto_save_terrain")
    assert idx > 0
    next_def = source.find("\n    def ", idx + 10)
    method_body = source[idx:next_def] if next_def > 0 else source[idx:idx + 500]
    # Should have some modulo or counter check to avoid publishing every 30s
    assert "%" in method_body or "count" in method_body.lower(), \
        "_auto_save_terrain should have interval guard (not publish every 30s)"


# ---------------------------------------------------------------------------
# L40-4: _replay_terrain uses _build_terrain_pc2 (no duplication)
# ---------------------------------------------------------------------------


def test_replay_terrain_uses_build_helper() -> None:
    """_replay_terrain should use _build_terrain_pc2 to avoid code duplication."""
    source = _read_bridge_source()
    idx = source.find("def _replay_terrain")
    assert idx > 0
    next_def = source.find("\n    def ", idx + 10)
    method_body = source[idx:next_def] if next_def > 0 else source[idx:idx + 800]
    assert "_build_terrain_pc2" in method_body, \
        "_replay_terrain should use _build_terrain_pc2 helper"


# ---------------------------------------------------------------------------
# L40-5: Terrain replay publishes to all 3 topics
# ---------------------------------------------------------------------------


def test_replay_terrain_publishes_registered_scan() -> None:
    """_replay_terrain must publish to /registered_scan."""
    source = _read_bridge_source()
    idx = source.find("def _replay_terrain")
    assert idx > 0
    next_def = source.find("\n    def ", idx + 10)
    method_body = source[idx:next_def] if next_def > 0 else source[idx:idx + 800]
    assert "_pc_pub" in method_body, \
        "_replay_terrain must publish to /registered_scan via _pc_pub"


def test_replay_terrain_publishes_terrain_map() -> None:
    """_replay_terrain must publish to /terrain_map."""
    source = _read_bridge_source()
    idx = source.find("def _replay_terrain")
    assert idx > 0
    next_def = source.find("\n    def ", idx + 10)
    method_body = source[idx:next_def] if next_def > 0 else source[idx:idx + 800]
    assert "_terrain_map_pub" in method_body


def test_replay_terrain_publishes_terrain_map_ext() -> None:
    """_replay_terrain must publish to /terrain_map_ext."""
    source = _read_bridge_source()
    idx = source.find("def _replay_terrain")
    assert idx > 0
    next_def = source.find("\n    def ", idx + 10)
    method_body = source[idx:next_def] if next_def > 0 else source[idx:idx + 800]
    assert "_terrain_map_ext_pub" in method_body
