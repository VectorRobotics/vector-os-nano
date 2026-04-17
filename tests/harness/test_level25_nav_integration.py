"""Level-25 nav integration tests — source-inspection harness.

These tests verify the structural contract between Go2ROS2Proxy and
NavigateSkill without requiring a live ROS2 / MuJoCo environment.

All checks are performed via inspect / AST / source-text analysis so they
run instantly in CI with no external dependencies.
"""
from __future__ import annotations

import inspect
import importlib
import re
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_proxy() -> type:
    """Import and return the Go2ROS2Proxy class."""
    mod = importlib.import_module(
        "vector_os_nano.hardware.sim.go2_ros2_proxy"
    )
    return mod.Go2ROS2Proxy


def _proxy_source() -> str:
    """Return full source text of go2_ros2_proxy.py."""
    mod = importlib.import_module(
        "vector_os_nano.hardware.sim.go2_ros2_proxy"
    )
    return inspect.getsource(mod)


def _navigate_skill_source() -> str:
    """Return full source text of skills/navigate.py."""
    mod = importlib.import_module("vector_os_nano.skills.navigate")
    return inspect.getsource(mod)


def _navigate_module():
    """Return the skills/navigate module."""
    return importlib.import_module("vector_os_nano.skills.navigate")


# ---------------------------------------------------------------------------
# Proxy structural tests
# ---------------------------------------------------------------------------

def test_proxy_has_navigate_to() -> None:
    """Go2ROS2Proxy exposes a navigate_to(x, y, timeout) method."""
    proxy_cls = _load_proxy()
    assert hasattr(proxy_cls, "navigate_to"), (
        "Go2ROS2Proxy is missing navigate_to method"
    )
    method = getattr(proxy_cls, "navigate_to")
    assert callable(method), "navigate_to must be callable"

    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    assert "x" in params, "navigate_to must accept 'x' parameter"
    assert "y" in params, "navigate_to must accept 'y' parameter"
    assert "timeout" in params, "navigate_to must accept 'timeout' parameter"


def test_proxy_has_waypoint_publisher() -> None:
    """Proxy creates a /way_point publisher (direct to localPlanner)."""
    src = _proxy_source()
    assert "/way_point" in src, (
        "go2_ros2_proxy.py must reference '/way_point' topic"
    )
    assert "waypoint_pub" in src, (
        "go2_ros2_proxy.py must have a _waypoint_pub attribute"
    )
    assert "PointStamped" in src, (
        "go2_ros2_proxy.py must import/use PointStamped for /goal_point"
    )


def test_proxy_navigate_creates_nav_flag() -> None:
    """navigate_to() references the /tmp/vector_nav_active flag file."""
    src = _proxy_source()
    assert "/tmp/vector_nav_active" in src, (
        "navigate_to must create /tmp/vector_nav_active flag"
    )
    # Should also open/write the flag
    assert re.search(r'open\(.*vector_nav_active', src), (
        "navigate_to must open /tmp/vector_nav_active for writing"
    )


def test_proxy_has_cancel_navigation() -> None:
    """Go2ROS2Proxy exposes cancel_navigation() method."""
    proxy_cls = _load_proxy()
    assert hasattr(proxy_cls, "cancel_navigation"), (
        "Go2ROS2Proxy is missing cancel_navigation method"
    )


def test_proxy_has_stop_navigation() -> None:
    """Go2ROS2Proxy exposes stop_navigation() method that removes nav flag."""
    proxy_cls = _load_proxy()
    assert hasattr(proxy_cls, "stop_navigation"), (
        "Go2ROS2Proxy is missing stop_navigation method"
    )
    src = _proxy_source()
    # stop_navigation must remove the flag (os.remove)
    assert re.search(r'os\.remove.*vector_nav_active', src), (
        "stop_navigation must call os.remove('/tmp/vector_nav_active')"
    )


def test_proxy_navigate_stores_nav_goal() -> None:
    """navigate_to() assigns to self._nav_goal for RViz visualization."""
    src = _proxy_source()
    # Within navigate_to the goal should be stored
    assert "_nav_goal" in src, (
        "navigate_to must store goal in self._nav_goal"
    )


# ---------------------------------------------------------------------------
# NavigateSkill structural tests
# ---------------------------------------------------------------------------

def test_navigate_skill_uses_proxy() -> None:
    """NavigateSkill.execute() checks hasattr(base, 'navigate_to') for Mode 0.

    We extract the execute() method source and verify both the hasattr check
    and the nav services check appear there, with proxy mode first.
    """
    nav_mod = _navigate_module()
    navigate_skill_cls = nav_mod.NavigateSkill
    execute_src = inspect.getsource(navigate_skill_cls.execute)

    assert "hasattr" in execute_src, (
        "NavigateSkill.execute must use hasattr() to check for navigate_to"
    )
    assert "navigate_to" in execute_src, (
        "NavigateSkill.execute must reference navigate_to"
    )
    assert 'context.services.get("nav")' in execute_src, (
        "NavigateSkill.execute must retain NavStackClient mode check"
    )

    # Proxy mode (Mode 0) must appear before NavStackClient mode (Mode 1)
    proxy_mode_pos = execute_src.find('hasattr(context.base, "navigate_to")')
    nav_stack_pos = execute_src.find('context.services.get("nav")')
    assert proxy_mode_pos != -1, (
        "NavigateSkill.execute must check hasattr(context.base, 'navigate_to')"
    )
    assert proxy_mode_pos < nav_stack_pos, (
        "Proxy mode (Mode 0) must appear before NavStackClient mode (Mode 1) in execute()"
    )


def test_navigate_skill_proxy_calls_navigate_to() -> None:
    """NavigateSkill._navigate_with_proxy() calls context.base.navigate_to()."""
    src = _navigate_skill_source()
    assert "_navigate_with_proxy" in src, (
        "NavigateSkill must have _navigate_with_proxy helper method"
    )
    # The proxy method must call base.navigate_to
    assert "context.base.navigate_to(" in src, (
        "_navigate_with_proxy must call context.base.navigate_to()"
    )


def test_navigate_skill_proxy_falls_back_to_dead_reckoning() -> None:
    """_navigate_with_proxy() falls back to dead-reckoning when proxy returns False."""
    src = _navigate_skill_source()
    # Should call _dead_reckoning as fallback inside _navigate_with_proxy
    # Check that dead_reckoning is referenced after the proxy nav_result check
    proxy_method_match = re.search(
        r'def _navigate_with_proxy.*?def _navigate_with_nav_stack',
        src,
        re.DOTALL,
    )
    assert proxy_method_match is not None, (
        "_navigate_with_proxy method must exist before _navigate_with_nav_stack"
    )
    proxy_block = proxy_method_match.group(0)
    assert "_dead_reckoning" in proxy_block, (
        "_navigate_with_proxy must call self._dead_reckoning() as fallback"
    )


def test_navigate_skill_room_aliases() -> None:
    """All 8 canonical rooms + key Chinese aliases resolve correctly."""
    nav_mod = _navigate_module()
    resolve = nav_mod._resolve_room

    # English canonical names
    canonical_rooms = [
        "living_room",
        "dining_room",
        "kitchen",
        "study",
        "master_bedroom",
        "guest_bedroom",
        "bathroom",
        "hallway",
    ]
    for room in canonical_rooms:
        result = resolve(room)
        assert result == room, (
            f"_resolve_room('{room}') should return '{room}', got '{result}'"
        )

    # English aliases
    english_aliases = {
        "living room": "living_room",
        "dining room": "dining_room",
        "kitchen": "kitchen",
        "study": "study",
        "master bedroom": "master_bedroom",
        "bedroom": "master_bedroom",
        "guest bedroom": "guest_bedroom",
        "bathroom": "bathroom",
        "hallway": "hallway",
    }
    for alias, expected in english_aliases.items():
        result = resolve(alias)
        assert result == expected, (
            f"_resolve_room('{alias}') should return '{expected}', got '{result}'"
        )

    # Chinese aliases
    chinese_aliases = {
        "客厅": "living_room",
        "餐厅": "dining_room",
        "厨房": "kitchen",
        "书房": "study",
        "主卧": "master_bedroom",
        "客卧": "guest_bedroom",
        "卫生间": "bathroom",
        "走廊": "hallway",
    }
    for alias, expected in chinese_aliases.items():
        result = resolve(alias)
        assert result == expected, (
            f"_resolve_room('{alias}') should return '{expected}', got '{result}'"
        )


def test_arrival_threshold() -> None:
    """navigate_to() arrival distance is between 0.5 m and 1.0 m.

    After nav.yaml refactor, _ARRIVAL_DIST is set via _nav("arrival_radius", 0.8).
    Accept both: literal type annotation AND _nav() call pattern.
    """
    src = _proxy_source()
    # Pattern 1 (legacy): _ARRIVAL_DIST: float = 0.8
    match = re.search(r'_ARRIVAL_DIST\s*(?::\s*float\s*)?=\s*([\d.]+)', src)
    if match:
        arrival_dist = float(match.group(1))
    else:
        # Pattern 2 (current): _ARRIVAL_DIST = _nav("arrival_radius", 0.8)
        # or _ARRIVAL = _nav("arrival_radius", 0.8)
        match = re.search(
            r'_ARRIVAL(?:_DIST)?\s*=\s*_nav\s*\([^,]+,\s*([\d.]+)\)',
            src,
        )
        assert match is not None, (
            "navigate_to must define _ARRIVAL_DIST or _ARRIVAL threshold "
            "(literal or _nav() call)"
        )
        arrival_dist = float(match.group(1))
    assert 0.5 <= arrival_dist <= 1.0, (
        f"arrival distance={arrival_dist} must be in [0.5, 1.0] m"
    )


def test_navigate_mode_result_label() -> None:
    """Successful proxy navigation reports mode='proxy_nav_stack' in result_data."""
    src = _navigate_skill_source()
    assert "proxy_nav_stack" in src, (
        "_navigate_with_proxy must label its result mode as 'proxy_nav_stack'"
    )


def test_navigate_skill_cancels_exploration_on_start() -> None:
    """NavigateSkill.execute() cancels background exploration before navigating."""
    src = _navigate_skill_source()
    assert "cancel_exploration" in src, (
        "NavigateSkill must call cancel_exploration() when exploration is active"
    )
    assert "is_exploring" in src, (
        "NavigateSkill must check is_exploring() before cancelling"
    )


def test_navigate_skill_ensures_nav_flag() -> None:
    """NavigateSkill.execute() ensures /tmp/vector_nav_active exists before navigating."""
    src = _navigate_skill_source()
    assert "vector_nav_active" in src, (
        "NavigateSkill must reference /tmp/vector_nav_active"
    )


# ---------------------------------------------------------------------------
# P1a: TARE-only stop (Beta)
# ---------------------------------------------------------------------------

def _explore_source() -> str:
    """Return full source text of skills/go2/explore.py."""
    mod = importlib.import_module("vector_os_nano.skills.go2.explore")
    return inspect.getsource(mod)


def _explore_module():
    """Return the skills/go2/explore module."""
    return importlib.import_module("vector_os_nano.skills.go2.explore")


def test_stop_tare_only_exists() -> None:
    """explore.py exposes a stop_tare_only() function."""
    mod = _explore_module()
    assert hasattr(mod, "stop_tare_only"), (
        "explore.py must expose stop_tare_only() — kills TARE only, keeps FAR alive"
    )
    func = getattr(mod, "stop_tare_only")
    assert callable(func), "stop_tare_only must be callable"


def test_cancel_exploration_preserves_nav() -> None:
    """cancel_exploration() calls stop_tare_only, NOT os.killpg on nav_explore_proc.

    Verifies that cancel_exploration no longer kills the full nav stack process
    group — it should only stop TARE via stop_tare_only().
    """
    src = _explore_source()

    # Must call stop_tare_only
    assert "stop_tare_only" in src, (
        "cancel_exploration must call stop_tare_only() instead of killing the nav stack"
    )

    # cancel_exploration source block must NOT kill nav_explore_proc process group
    # Extract just the cancel_exploration function body
    cancel_match = re.search(
        r'def cancel_exploration\(\).*?(?=\ndef |\Z)',
        src,
        re.DOTALL,
    )
    assert cancel_match is not None, "cancel_exploration function not found in source"
    cancel_src = cancel_match.group(0)

    assert "killpg" not in cancel_src, (
        "cancel_exploration must NOT call os.killpg — use stop_tare_only() instead"
    )
    assert "_nav_explore_proc" not in cancel_src, (
        "cancel_exploration must NOT reference _nav_explore_proc — "
        "FAR + localPlanner must remain running"
    )


def test_is_nav_stack_running_exists() -> None:
    """explore.py exposes is_nav_stack_running() function using pgrep localPlanner."""
    mod = _explore_module()
    assert hasattr(mod, "is_nav_stack_running"), (
        "explore.py must expose is_nav_stack_running() "
        "— checks whether localPlanner is alive"
    )
    func = getattr(mod, "is_nav_stack_running")
    assert callable(func), "is_nav_stack_running must be callable"

    src = _explore_source()
    assert "localPlanner" in src, (
        "is_nav_stack_running must check for 'localPlanner' via pgrep"
    )
    assert "pgrep" in src, (
        "is_nav_stack_running must use pgrep to check process status"
    )


# ---------------------------------------------------------------------------
# P1b: SceneGraph stores room positions (Beta)
# ---------------------------------------------------------------------------

def test_scene_graph_has_room_positions() -> None:
    """SceneGraph.visit() stores robot position as room center coordinates.

    Verifies that after calling visit(room, x, y) the SceneGraph returns
    the stored coordinates via get_room() and get_location(), and that
    NavigateSkill uses _get_room_center_from_memory() to prefer these
    explored positions over hardcoded _ROOM_CENTERS values.
    """
    from vector_os_nano.core.scene_graph import SceneGraph

    sg = SceneGraph()
    sg.visit("kitchen", 16.5, 2.8)

    # get_room() API
    room_node = sg.get_room("kitchen")
    assert room_node is not None, "SceneGraph must store room after visit()"
    assert room_node.center_x == 16.5, (
        f"RoomNode.center_x should be 16.5, got {room_node.center_x}"
    )
    assert room_node.center_y == 2.8, (
        f"RoomNode.center_y should be 2.8, got {room_node.center_y}"
    )
    assert room_node.visit_count >= 1, "visit_count should be >= 1 after visit()"

    # get_location() backward-compat API
    loc = sg.get_location("kitchen")
    assert loc is not None, "get_location() must return result for visited room"
    assert loc.x == 16.5, f"LocationRecord.x should be 16.5, got {loc.x}"
    assert loc.y == 2.8, f"LocationRecord.y should be 2.8, got {loc.y}"

    # NavigateSkill uses _get_room_center_from_memory
    nav_src = _navigate_skill_source()
    assert "_get_room_center_from_memory" in nav_src, (
        "NavigateSkill must call _get_room_center_from_memory() "
        "to prefer SceneGraph positions"
    )
    assert "spatial_memory" in nav_src, (
        "NavigateSkill must check context.services.get('spatial_memory')"
    )
