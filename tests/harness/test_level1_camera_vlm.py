"""Level 1 — Camera-to-VLM pipeline verification.

Tests verify that MuJoCo renders a valid camera frame, the frame can be
encoded, and GPT-4o via OpenRouter parses it correctly.

Pipeline under test:
    MuJoCoGo2.get_camera_frame() -> Go2VLMPerception -> SceneDescription / RoomIdentification

Prerequisites:
- mujoco installed (harness conftest enforces this)
- OPENROUTER_API_KEY set, or config/user.yaml contains llm.api_key

All tests in this class are marked @pytest.mark.slow (real API calls).
They are skipped automatically when no API key is present.

Estimated API cost: ~$0.05 per full run (4 API calls with MuJoCo frames).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure repo root is importable (mirrors conftest.py pattern)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TIMEOUT_S: float = 30.0
_MAX_COST_PER_CALL_USD: float = 0.50


# ---------------------------------------------------------------------------
# API key helper (mirrors test_level0_vlm_api pattern)
# ---------------------------------------------------------------------------

_API_KEY: str | None = None


def _get_api_key() -> str:
    """Load OpenRouter API key from config or environment."""
    global _API_KEY
    if _API_KEY is not None:
        return _API_KEY

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            import yaml  # type: ignore[import]
            cfg_path = _REPO_ROOT / "config" / "user.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            key = cfg.get("llm", {}).get("api_key", "")
        except Exception:
            pass

    _API_KEY = key
    return key


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _import_mujoco_go2():
    """Import MuJoCoGo2 by direct file path, avoiding package cascade.

    Mirrors the pattern in conftest.py — avoids importing httpx and other
    optional deps that may not be present in the harness environment.
    Loads vector_os_nano.core.types first since mujoco_go2 requires it.
    """
    import importlib.util

    module_path = (
        _REPO_ROOT / "vector_os_nano" / "hardware" / "sim" / "mujoco_go2.py"
    )
    types_path = _REPO_ROOT / "vector_os_nano" / "core" / "types.py"

    types_spec = importlib.util.spec_from_file_location(
        "vector_os_nano.core.types", str(types_path)
    )
    types_mod = importlib.util.module_from_spec(types_spec)  # type: ignore[arg-type]
    sys.modules.setdefault("vector_os_nano.core.types", types_mod)
    types_spec.loader.exec_module(types_mod)  # type: ignore[union-attr]

    spec = importlib.util.spec_from_file_location(
        "vector_os_nano.hardware.sim.mujoco_go2", str(module_path)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["vector_os_nano.hardware.sim.mujoco_go2"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.MuJoCoGo2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLevel1CameraVLM:
    """L1: MuJoCo camera frame -> GPT-4o vision pipeline."""

    @pytest.fixture(autouse=True)
    def _require_api_key(self):
        """Skip entire class if no API key is available."""
        key = _get_api_key()
        if not key:
            pytest.skip("No OPENROUTER_API_KEY available — skipping L1 camera/VLM tests")

    @pytest.fixture
    def go2(self):
        """Create headless Go2 sim with room scene, stand, yield, disconnect."""
        MuJoCoGo2 = _import_mujoco_go2()
        robot = MuJoCoGo2(gui=False, room=True, backend="sinusoidal")
        robot.connect()
        robot.stand()
        yield robot
        robot.disconnect()

    @pytest.fixture
    def vlm(self):
        """Create Go2VLMPerception with the loaded API key."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        key = _get_api_key()
        return Go2VLMPerception(config={"api_key": key})

    # ------------------------------------------------------------------
    # Camera frame shape tests (no API cost)
    # ------------------------------------------------------------------

    def test_camera_frame_shape(self, go2):
        """get_camera_frame() returns a (480, 640, 3) uint8 RGB array."""
        frame = go2.get_camera_frame()

        assert frame is not None, "get_camera_frame() returned None"
        assert isinstance(frame, np.ndarray), (
            f"Expected np.ndarray, got {type(frame)}"
        )
        assert frame.shape == (480, 640, 3), (
            f"Expected shape (480, 640, 3), got {frame.shape}"
        )
        assert frame.dtype == np.uint8, (
            f"Expected dtype uint8, got {frame.dtype}"
        )

    def test_camera_frame_not_black(self, go2):
        """get_camera_frame() returns a non-black image (MuJoCo rendered something)."""
        frame = go2.get_camera_frame()

        mean_brightness = float(frame.mean())
        assert mean_brightness > 0.0, (
            "Camera frame is entirely black — MuJoCo may not have rendered the scene"
        )

    # ------------------------------------------------------------------
    # Full VLM pipeline tests (real API calls)
    # ------------------------------------------------------------------

    def test_describe_scene_from_mujoco(self, go2, vlm):
        """VLM can describe a MuJoCo-rendered scene and returns a populated SceneDescription."""
        from vector_os_nano.perception.vlm_go2 import SceneDescription

        frame = go2.get_camera_frame()
        scene = vlm.describe_scene(frame)

        assert isinstance(scene, SceneDescription), (
            f"Expected SceneDescription, got {type(scene)}"
        )
        assert scene.summary, "SceneDescription.summary is empty"
        assert scene.room_type, "SceneDescription.room_type is empty"
        assert isinstance(scene.objects, list), (
            f"SceneDescription.objects must be a list, got {type(scene.objects)}"
        )

    def test_identify_room_from_mujoco(self, go2, vlm):
        """VLM identifies a room from a MuJoCo-rendered frame."""
        from vector_os_nano.perception.vlm_go2 import RoomIdentification

        frame = go2.get_camera_frame()
        room = vlm.identify_room(frame)

        assert isinstance(room, RoomIdentification), (
            f"Expected RoomIdentification, got {type(room)}"
        )
        assert room.room, "RoomIdentification.room is empty"
        assert 0.0 <= room.confidence <= 1.0, (
            f"Confidence {room.confidence} is out of [0.0, 1.0] range"
        )
        assert room.reasoning, "RoomIdentification.reasoning is empty"

    def test_find_objects_from_mujoco(self, go2, vlm):
        """VLM find_objects() returns a list from a MuJoCo scene."""
        from vector_os_nano.perception.vlm_go2 import DetectedObject

        frame = go2.get_camera_frame()
        objects = vlm.find_objects(frame)

        assert isinstance(objects, list), (
            f"find_objects() must return a list, got {type(objects)}"
        )
        # Each element must be a DetectedObject with valid confidence range
        for obj in objects:
            assert isinstance(obj, DetectedObject), (
                f"find_objects() element is not DetectedObject: {type(obj)}"
            )
            assert 0.0 <= obj.confidence <= 1.0, (
                f"DetectedObject confidence {obj.confidence} out of range"
            )

    def test_cost_stays_low(self, go2, vlm):
        """Single describe_scene call costs less than $0.50."""
        frame = go2.get_camera_frame()
        vlm.describe_scene(frame)

        cost = vlm.cumulative_cost_usd
        assert cost < _MAX_COST_PER_CALL_USD, (
            f"Single describe_scene call cost ${cost:.4f} "
            f"(limit ${_MAX_COST_PER_CALL_USD:.2f})"
        )
