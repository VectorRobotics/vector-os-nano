"""Level 4 — End-to-end patrol: real MuJoCo + real VLM API.

Runs a patrol of 2 rooms (hallway then kitchen) to keep API cost low.
Each test that calls the VLM is marked with pytest.mark.timeout(120) to
prevent runaway wall-clock time in CI.

Cost estimate: ~$0.05–$0.10 per full run (2–3 API calls per test, 2 tests).

Requirements:
  - mujoco must be installed
  - OPENROUTER_API_KEY must be set (env) or present in config/user.yaml
  - sinusoidal backend (no convex_mpc dependency)

Skips:
  - Entire module skips if mujoco is unavailable (pytest.importorskip in conftest)
  - Individual tests skip if no API key is found
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root on sys.path (conftest may not run before module-level code)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# API key helper (mirrors test_level0_vlm_api.py)
# ---------------------------------------------------------------------------

_API_KEY: str | None = None


def _get_api_key() -> str:
    """Load OpenRouter API key from environment or config/user.yaml."""
    global _API_KEY
    if _API_KEY is not None:
        return _API_KEY

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            import yaml

            cfg_path = _REPO_ROOT / "config" / "user.yaml"
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh)
            key = cfg.get("llm", {}).get("api_key", "")
        except Exception:
            pass

    _API_KEY = key
    return key


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLevel4Patrol:
    """L4: End-to-end patrol — real MuJoCo + real GPT-4o vision API.

    NOTE: v2.0 NavigateSkill delegates to base.go_to_waypoint() /
    base.navigate_to() — methods that exist on Go2ROS2Proxy but NOT on
    the plain MuJoCoGo2 used here. These tests therefore need a running
    ROS2 nav stack + proxy, not just raw MuJoCo. Skipped until either
    (a) the test migrates to the proxy fixture, or (b) MuJoCoGo2 grows
    a navigate_to stub for harness use.
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _require_api_key(self):
        """Skip the test if no API key is available."""
        if not _get_api_key():
            pytest.skip("No OPENROUTER_API_KEY available")
        pytest.skip(
            "L4 patrol tests need ROS2 proxy for base.go_to_waypoint — "
            "MuJoCoGo2 doesn't have it. Run manually with nav stack for AC6."
        )

    @pytest.fixture
    def go2(self):
        """Create a headless Go2 sim with room=True, sinusoidal backend."""
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

        robot = MuJoCoGo2(gui=False, room=True, backend="sinusoidal")
        robot.connect()
        robot.stand()
        yield robot
        robot.disconnect()

    @pytest.fixture
    def vlm(self):
        """Instantiate the VLM perception wrapper with the real API key."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

        key = _get_api_key()
        return Go2VLMPerception(config={"api_key": key})

    @pytest.fixture
    def spatial_memory(self):
        """Fresh SceneGraph pre-seeded with room_layout.yaml static positions.

        NavigateSkill resolves rooms via SceneGraph.get_room() (v2.0).
        Using the legacy SpatialMemory class leaves the room table empty
        and NavigateSkill returns "Room position unknown. Explore more."
        """
        import os
        from vector_os_nano.core.scene_graph import SceneGraph

        sg = SceneGraph()
        layout = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "room_layout.yaml",
        )
        if os.path.exists(layout):
            sg.load_layout(layout)
        return sg

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.timeout(120)
    def test_navigate_and_look(self, go2, vlm, spatial_memory):
        """Navigate to hallway and describe the scene with the real VLM."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.skills.go2.look import LookSkill
        from vector_os_nano.skills.navigate import NavigateSkill

        ctx = SkillContext(
            bases={"default": go2},
            services={"vlm": vlm, "spatial_memory": spatial_memory},
        )

        # Navigate to hallway (starting point — minimal movement)
        nav = NavigateSkill()
        nav_result = nav.execute({"room": "hallway"}, ctx)
        assert nav_result.success, (
            f"NavigateSkill failed: {nav_result.error_message}"
        )

        # Capture and describe the scene
        look = LookSkill()
        look_result = look.execute({}, ctx)
        assert look_result.success, (
            f"LookSkill failed: {look_result.error_message}"
        )
        summary = look_result.result_data.get("summary", "")
        assert summary, "LookSkill returned empty summary"
        assert len(summary) > 5, f"Summary too short: {summary!r}"

    @pytest.mark.timeout(120)
    def test_two_room_patrol(self, go2, vlm, spatial_memory):
        """Patrol 2 rooms (hallway + kitchen): navigate, look, record observations."""
        from vector_os_nano.core.skill import SkillContext, SkillRegistry
        from vector_os_nano.skills.go2.look import LookSkill
        from vector_os_nano.skills.navigate import NavigateSkill

        nav = NavigateSkill()
        look = LookSkill()

        registry = SkillRegistry()
        registry.register(nav)
        registry.register(look)

        rooms_to_visit = ["hallway", "kitchen"]
        observations: dict[str, str] = {}

        for room in rooms_to_visit:
            ctx = SkillContext(
                bases={"default": go2},
                services={
                    "vlm": vlm,
                    "spatial_memory": spatial_memory,
                    "skill_registry": registry,
                },
            )

            nav_result = nav.execute({"room": room}, ctx)
            if nav_result.success:
                look_result = look.execute({}, ctx)
                if look_result.success:
                    obs = look_result.result_data.get("summary", "")
                    if obs:
                        observations[room] = obs

        assert len(observations) >= 1, (
            f"Expected at least 1 room observation, got 0. "
            f"Navigation results may have failed."
        )

        visited = spatial_memory.get_visited_rooms()
        assert len(visited) >= 1, (
            f"SpatialMemory should record at least 1 visited room, got: {visited}"
        )

    @pytest.mark.timeout(60)
    def test_vlm_cost_under_budget(self, go2, vlm):
        """A single VLM call on a synthetic frame stays under the $1 budget."""
        # Take one frame from the live sim and call describe_scene once.
        # This verifies cost tracking works and a single call is inexpensive.
        frame = go2.get_camera_frame()
        assert frame is not None, "get_camera_frame() returned None"

        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

        key = _get_api_key()
        isolated_vlm = Go2VLMPerception(config={"api_key": key})
        isolated_vlm.describe_scene(frame)

        assert isolated_vlm.cumulative_cost_usd < 1.0, (
            f"Single VLM call cost ${isolated_vlm.cumulative_cost_usd:.4f} — "
            f"exceeds $1.00 budget"
        )

    @pytest.mark.timeout(120)
    def test_spatial_memory_persists_observations(self, go2, vlm, spatial_memory):
        """After looking in a room, spatial memory records the observation."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.skills.go2.look import LookSkill
        from vector_os_nano.skills.navigate import NavigateSkill

        nav = NavigateSkill()
        look = LookSkill()

        ctx = SkillContext(
            bases={"default": go2},
            services={"vlm": vlm, "spatial_memory": spatial_memory},
        )

        nav_result = nav.execute({"room": "hallway"}, ctx)
        if not nav_result.success:
            pytest.skip("Navigation to hallway failed — cannot test memory persistence")

        look_result = look.execute({}, ctx)
        assert look_result.success, f"LookSkill failed: {look_result.error_message}"

        # Memory should have at least one visited room entry after the look
        visited = spatial_memory.get_visited_rooms()
        assert len(visited) >= 1, (
            f"SpatialMemory.get_visited_rooms() is empty after look. "
            f"Expected at least hallway to be recorded."
        )
