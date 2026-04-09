"""Test Phase 3 primitives extension and auto-observe concept.

Level 61 — TDD harness.

Tests:
  - PrimitiveContext.object_memory field exists and defaults to None
  - world.last_seen() delegates to ObjectMemory
  - world.certainty() delegates to ObjectMemory
  - world.find_object() delegates to ObjectMemory
  - world.objects_in_room() delegates to ObjectMemory
  - world.room_coverage() delegates to SceneGraph.get_room_coverage()
  - Graceful degradation when context/object_memory is None
  - Auto-observe concept: VLM mock wiring, VLM failure is non-blocking
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from vector_os_nano.vcli.cognitive.object_memory import ObjectMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_populated_om() -> ObjectMemory:
    """Return an ObjectMemory pre-populated with two objects."""
    om = ObjectMemory()
    om.update("obj1", "cup", "kitchen", 5.0, 3.0, 0.9)
    om.update("obj2", "book", "bedroom", 2.0, 1.0, 0.7)
    return om


# ---------------------------------------------------------------------------
# TestPrimitiveContext — field existence
# ---------------------------------------------------------------------------


class TestPrimitiveContextObjectMemoryField:
    """PrimitiveContext must have object_memory field defaulting to None."""

    def test_object_memory_field_defaults_to_none(self):
        from vector_os_nano.vcli.primitives import PrimitiveContext
        ctx = PrimitiveContext()
        assert ctx.object_memory is None

    def test_object_memory_field_accepts_instance(self):
        from vector_os_nano.vcli.primitives import PrimitiveContext
        om = ObjectMemory()
        ctx = PrimitiveContext(object_memory=om)
        assert ctx.object_memory is om

    def test_object_memory_field_accepts_none_explicitly(self):
        from vector_os_nano.vcli.primitives import PrimitiveContext
        ctx = PrimitiveContext(object_memory=None)
        assert ctx.object_memory is None

    def test_existing_fields_unaffected(self):
        """Adding object_memory must not break existing field defaults."""
        from vector_os_nano.vcli.primitives import PrimitiveContext
        ctx = PrimitiveContext()
        assert ctx.base is None
        assert ctx.scene_graph is None
        assert ctx.vlm is None
        assert ctx.nav_client is None
        assert ctx.skill_registry is None


# ---------------------------------------------------------------------------
# TestPrimitivesWorld — fixture
# ---------------------------------------------------------------------------


class TestPrimitivesWorld:
    """Test new world.py primitive functions."""

    @pytest.fixture
    def setup_primitives(self):
        """Set up primitives with mock context holding ObjectMemory + SceneGraph."""
        from vector_os_nano.vcli.primitives import world, PrimitiveContext
        om = _make_populated_om()
        mock_sg = MagicMock()
        mock_sg.get_room_coverage.return_value = 0.65
        ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=om)
        old_ctx = world._ctx
        world._ctx = ctx
        yield om, mock_sg
        world._ctx = old_ctx  # restore

    # ------------------------------------------------------------------
    # last_seen
    # ------------------------------------------------------------------

    def test_last_seen_found(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import last_seen
        result = last_seen("cup")
        assert result is not None
        assert result["room"] == "kitchen"

    def test_last_seen_returns_dict_keys(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import last_seen
        result = last_seen("cup")
        assert "room" in result
        assert "position" in result
        assert "seconds_ago" in result
        assert "confidence" in result

    def test_last_seen_not_found(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import last_seen
        assert last_seen("laptop") is None

    def test_last_seen_no_context(self):
        from vector_os_nano.vcli.primitives import world
        old = world._ctx
        world._ctx = None
        try:
            from vector_os_nano.vcli.primitives.world import last_seen
            assert last_seen("cup") is None
        finally:
            world._ctx = old

    def test_last_seen_no_object_memory(self):
        from vector_os_nano.vcli.primitives import world, PrimitiveContext
        old = world._ctx
        mock_sg = MagicMock()
        world._ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=None)
        try:
            from vector_os_nano.vcli.primitives.world import last_seen
            assert last_seen("cup") is None
        finally:
            world._ctx = old

    # ------------------------------------------------------------------
    # certainty
    # ------------------------------------------------------------------

    def test_certainty_cup_in_kitchen(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import certainty
        val = certainty("cup在kitchen")
        assert val > 0.5

    def test_certainty_english_format(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import certainty
        val = certainty("cup in kitchen")
        assert val > 0.5

    def test_certainty_missing_object(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import certainty
        assert certainty("laptop在kitchen") == 0.0

    def test_certainty_no_context(self):
        from vector_os_nano.vcli.primitives import world
        old = world._ctx
        world._ctx = None
        try:
            from vector_os_nano.vcli.primitives.world import certainty
            assert certainty("cup在kitchen") == 0.0
        finally:
            world._ctx = old

    def test_certainty_no_object_memory(self):
        from vector_os_nano.vcli.primitives import world, PrimitiveContext
        old = world._ctx
        mock_sg = MagicMock()
        world._ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=None)
        try:
            from vector_os_nano.vcli.primitives.world import certainty
            assert certainty("cup在kitchen") == 0.0
        finally:
            world._ctx = old

    # ------------------------------------------------------------------
    # find_object
    # ------------------------------------------------------------------

    def test_find_object_found(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import find_object
        results = find_object("cup")
        assert len(results) == 1
        assert results[0]["category"] == "cup"

    def test_find_object_not_found(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import find_object
        results = find_object("laptop")
        assert results == []

    def test_find_object_no_context(self):
        from vector_os_nano.vcli.primitives import world
        old = world._ctx
        world._ctx = None
        try:
            from vector_os_nano.vcli.primitives.world import find_object
            assert find_object("cup") == []
        finally:
            world._ctx = old

    def test_find_object_no_object_memory(self):
        from vector_os_nano.vcli.primitives import world, PrimitiveContext
        old = world._ctx
        mock_sg = MagicMock()
        world._ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=None)
        try:
            from vector_os_nano.vcli.primitives.world import find_object
            assert find_object("cup") == []
        finally:
            world._ctx = old

    def test_find_object_result_has_required_keys(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import find_object
        results = find_object("cup")
        assert len(results) == 1
        r = results[0]
        assert "category" in r
        assert "room" in r

    # ------------------------------------------------------------------
    # objects_in_room
    # ------------------------------------------------------------------

    def test_objects_in_room_kitchen(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import objects_in_room
        results = objects_in_room("kitchen")
        assert len(results) == 1
        assert results[0]["category"] == "cup"

    def test_objects_in_room_bedroom(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import objects_in_room
        results = objects_in_room("bedroom")
        assert len(results) == 1
        assert results[0]["category"] == "book"

    def test_objects_in_room_empty_room(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import objects_in_room
        results = objects_in_room("garage")
        assert results == []

    def test_objects_in_room_no_context(self):
        from vector_os_nano.vcli.primitives import world
        old = world._ctx
        world._ctx = None
        try:
            from vector_os_nano.vcli.primitives.world import objects_in_room
            assert objects_in_room("kitchen") == []
        finally:
            world._ctx = old

    def test_objects_in_room_no_object_memory(self):
        from vector_os_nano.vcli.primitives import world, PrimitiveContext
        old = world._ctx
        mock_sg = MagicMock()
        world._ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=None)
        try:
            from vector_os_nano.vcli.primitives.world import objects_in_room
            assert objects_in_room("kitchen") == []
        finally:
            world._ctx = old

    # ------------------------------------------------------------------
    # room_coverage
    # ------------------------------------------------------------------

    def test_room_coverage_delegates_to_scene_graph(self, setup_primitives):
        from vector_os_nano.vcli.primitives.world import room_coverage
        val = room_coverage("kitchen")
        assert val == pytest.approx(0.65)

    def test_room_coverage_no_context_raises_or_returns_zero(self):
        """When no SceneGraph, room_coverage raises RuntimeError (no context)."""
        from vector_os_nano.vcli.primitives import world
        old = world._ctx
        world._ctx = None
        try:
            from vector_os_nano.vcli.primitives.world import room_coverage
            # _require_scene_graph() raises RuntimeError when _ctx is None
            with pytest.raises(RuntimeError):
                room_coverage("kitchen")
        finally:
            world._ctx = old

    def test_room_coverage_scene_graph_raises_returns_zero(self, setup_primitives):
        """If get_room_coverage() raises, returns 0.0."""
        from vector_os_nano.vcli.primitives.world import room_coverage
        om, mock_sg = setup_primitives
        mock_sg.get_room_coverage.side_effect = ValueError("unknown room")
        val = room_coverage("nonexistent")
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestAutoObserveConcept — mocked VLM + hook logic
# ---------------------------------------------------------------------------


class TestAutoObserveConcept:
    """Test that auto-observe logic works with mocks (no MuJoCo)."""

    def test_vlm_describe_and_detect(self):
        """Simulate what auto-observe would do: describe + find objects."""
        import numpy as np
        mock_vlm = MagicMock()

        desc_mock = MagicMock()
        desc_mock.summary = "A kitchen with a cup on the counter"
        mock_vlm.describe_scene.return_value = desc_mock

        cup_obj = MagicMock()
        cup_obj.name = "cup"
        cup_obj.confidence = 0.9
        counter_obj = MagicMock()
        counter_obj.name = "counter"
        counter_obj.confidence = 0.7
        mock_vlm.find_objects.return_value = [cup_obj, counter_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        desc = mock_vlm.describe_scene(frame)
        objs = mock_vlm.find_objects(frame)
        assert "kitchen" in str(desc.summary)
        assert len(objs) == 2

    def test_auto_observe_vlm_failure_non_blocking(self):
        """VLM failure must be caught — exploration must not crash."""
        mock_vlm = MagicMock()
        mock_vlm.describe_scene.side_effect = RuntimeError("VLM unavailable")
        caught = False
        try:
            mock_vlm.describe_scene(None)
        except RuntimeError:
            caught = True
        assert caught
        # The hook wraps in try/except so exploration continues regardless

    def test_auto_observe_no_vlm_skips(self):
        """When agent._vlm is None, auto-observe is skipped."""
        agent = MagicMock()
        agent._vlm = None
        _vlm = getattr(agent, "_vlm", None)
        assert _vlm is None  # skip condition confirmed

    def test_observe_with_viewpoint_called_on_success(self):
        """Mock scene_graph.observe_with_viewpoint is called with correct args."""
        mock_sg = MagicMock()
        mock_vlm = MagicMock()
        import numpy as np

        desc_mock = MagicMock()
        desc_mock.summary = "living room view"
        mock_vlm.describe_scene.return_value = desc_mock

        sofa_obj = MagicMock()
        sofa_obj.name = "sofa"
        sofa_obj.confidence = 0.85
        mock_vlm.find_objects.return_value = [sofa_obj]
        mock_sg.should_add_viewpoint.return_value = True

        # Simulate the hook logic
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        current_room = "living_room"
        x, y, heading = 1.0, 2.0, 0.0

        if mock_vlm is not None and mock_sg.should_add_viewpoint(current_room, x, y):
            try:
                frame_cap = frame
                desc_result = mock_vlm.describe_scene(frame_cap)
                obj_result = mock_vlm.find_objects(frame_cap)
                scene_summary = str(getattr(desc_result, "summary", ""))
                detected = [
                    {
                        "category": str(getattr(o, "name", "")),
                        "confidence": float(getattr(o, "confidence", 0.5)),
                    }
                    for o in (obj_result or [])
                ]
                object_names = [d["category"] for d in detected if d["category"]]
                mock_sg.observe_with_viewpoint(
                    room=current_room, x=x, y=y, heading=heading,
                    objects=object_names,
                    description=scene_summary,
                    detected_objects=detected,
                )
            except Exception:
                pass  # non-blocking

        mock_sg.observe_with_viewpoint.assert_called_once_with(
            room="living_room",
            x=1.0, y=2.0, heading=0.0,
            objects=["sofa"],
            description="living room view",
            detected_objects=[{"category": "sofa", "confidence": 0.85}],
        )

    def test_observe_with_viewpoint_not_called_when_no_new_viewpoint(self):
        """Hook skips if should_add_viewpoint returns False."""
        mock_sg = MagicMock()
        mock_sg.should_add_viewpoint.return_value = False
        mock_vlm = MagicMock()

        if mock_vlm is not None and mock_sg.should_add_viewpoint("kitchen", 0.0, 0.0):
            mock_sg.observe_with_viewpoint()

        mock_sg.observe_with_viewpoint.assert_not_called()

    def test_auto_observe_extracts_object_names(self):
        """Object names are correctly extracted from find_objects results."""
        mock_vlm = MagicMock()

        chair_obj = MagicMock()
        chair_obj.name = "chair"
        chair_obj.confidence = 0.8

        empty_obj = MagicMock()
        empty_obj.name = ""
        empty_obj.confidence = 0.3

        table_obj = MagicMock()
        table_obj.name = "table"
        table_obj.confidence = 0.75

        mock_vlm.find_objects.return_value = [chair_obj, empty_obj, table_obj]
        obj_result = mock_vlm.find_objects(None)
        detected = [
            {
                "category": str(getattr(o, "name", "")),
                "confidence": float(getattr(o, "confidence", 0.5)),
            }
            for o in (obj_result or [])
        ]
        object_names = [d["category"] for d in detected if d["category"]]
        assert "chair" in object_names
        assert "table" in object_names
        assert "" not in object_names

    def test_auto_observe_agent_without_vlm_attr(self):
        """Agent objects without _vlm attribute do not trigger hook."""
        class MinimalAgent:
            pass

        agent = MinimalAgent()
        _vlm = getattr(agent, "_vlm", None) if hasattr(agent, "_vlm") else None
        assert _vlm is None


# ---------------------------------------------------------------------------
# TestInitPrimitivesWithObjectMemory — integration
# ---------------------------------------------------------------------------


class TestInitPrimitivesWithObjectMemory:
    """init_primitives() propagates object_memory to world._ctx."""

    def test_init_primitives_sets_object_memory_on_world(self):
        from vector_os_nano.vcli.primitives import init_primitives, PrimitiveContext
        from vector_os_nano.vcli.primitives import world
        om = ObjectMemory()
        mock_sg = MagicMock()
        ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=om)
        init_primitives(ctx)
        assert world._ctx is ctx
        assert world._ctx.object_memory is om

    def test_world_functions_work_after_init_primitives(self):
        from vector_os_nano.vcli.primitives import init_primitives, PrimitiveContext
        from vector_os_nano.vcli.primitives.world import last_seen, find_object
        om = ObjectMemory()
        om.update("x1", "lamp", "hallway", 0.0, 0.0, 0.9)
        mock_sg = MagicMock()
        mock_sg.get_room_coverage.return_value = 0.3
        ctx = PrimitiveContext(scene_graph=mock_sg, object_memory=om)
        init_primitives(ctx)
        assert last_seen("lamp") is not None
        assert last_seen("lamp")["room"] == "hallway"
        results = find_object("lamp")
        assert len(results) == 1
