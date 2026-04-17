"""Level 57 — ObjectMemory TDD tests.

All tests use mocked SceneGraph — no MuJoCo instances needed.
Run with: pytest tests/harness/test_level57_object_memory.py -v
"""
from __future__ import annotations

import math
import threading
import time
from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli.cognitive.object_memory import ObjectMemory, TrackedObject


# ---------------------------------------------------------------------------
# Helpers — build a mock SceneGraph
# ---------------------------------------------------------------------------


def _make_mock_scene_graph(rooms_objects: dict[str, list[dict]]) -> MagicMock:
    """Create a MagicMock SceneGraph with given rooms and objects.

    Args:
        rooms_objects: {room_id: [{"object_id", "category", "x", "y", "confidence"}]}
    """
    sg = MagicMock()

    room_nodes = []
    for room_id in rooms_objects:
        rn = MagicMock()
        rn.room_id = room_id
        room_nodes.append(rn)

    sg.get_all_rooms.return_value = room_nodes

    def find_objects_in_room(room_id: str):
        objs = rooms_objects.get(room_id, [])
        result = []
        for o in objs:
            node = MagicMock()
            node.object_id = o["object_id"]
            node.category = o["category"]
            node.x = o.get("x", 0.0)
            node.y = o.get("y", 0.0)
            node.confidence = o.get("confidence", 0.9)
            node.room_id = room_id
            result.append(node)
        return result

    sg.find_objects_in_room.side_effect = find_objects_in_room
    return sg


# ---------------------------------------------------------------------------
# 1. TrackedObject is frozen (immutable)
# ---------------------------------------------------------------------------


class TestTrackedObjectFrozen:
    def test_tracked_object_frozen(self):
        obj = TrackedObject(
            object_id="abc",
            category="cup",
            room_id="kitchen",
            x=1.0,
            y=2.0,
            last_seen=time.time(),
            base_confidence=0.9,
            observation_count=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            obj.category = "mug"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Effective confidence decay
# ---------------------------------------------------------------------------


class TestEffectiveConfidence:
    def test_effective_confidence_immediate(self):
        mem = ObjectMemory(decay_lambda=0.001)
        now = time.time()
        obj = TrackedObject(
            object_id="x",
            category="cup",
            room_id="kitchen",
            x=0.0,
            y=0.0,
            last_seen=now,
            base_confidence=0.9,
            observation_count=1,
        )
        result = mem.effective_confidence(obj)
        # elapsed ~ 0 → result ≈ 0.9
        assert abs(result - 0.9) < 0.01

    def test_effective_confidence_10min(self):
        mem = ObjectMemory(decay_lambda=0.001)
        past = time.time() - 600  # 10 minutes ago
        obj = TrackedObject(
            object_id="x",
            category="cup",
            room_id="kitchen",
            x=0.0,
            y=0.0,
            last_seen=past,
            base_confidence=1.0,
            observation_count=1,
        )
        result = mem.effective_confidence(obj)
        expected = math.exp(-0.001 * 600)
        assert abs(result - expected) < 0.01

    def test_effective_confidence_30min(self):
        mem = ObjectMemory(decay_lambda=0.001)
        past = time.time() - 1800  # 30 minutes ago
        obj = TrackedObject(
            object_id="x",
            category="cup",
            room_id="kitchen",
            x=0.0,
            y=0.0,
            last_seen=past,
            base_confidence=1.0,
            observation_count=1,
        )
        result = mem.effective_confidence(obj)
        expected = math.exp(-0.001 * 1800)
        assert abs(result - 0.165) < 0.01
        assert abs(result - expected) < 0.001

    def test_effective_confidence_zero_base(self):
        mem = ObjectMemory(decay_lambda=0.001)
        obj = TrackedObject(
            object_id="x",
            category="cup",
            room_id="kitchen",
            x=0.0,
            y=0.0,
            last_seen=time.time(),
            base_confidence=0.0,
            observation_count=1,
        )
        assert mem.effective_confidence(obj) == 0.0


# ---------------------------------------------------------------------------
# 3. sync_from_scene_graph
# ---------------------------------------------------------------------------


class TestSyncFromSceneGraph:
    def test_sync_from_scene_graph(self):
        sg = _make_mock_scene_graph({
            "kitchen": [
                {"object_id": "obj1", "category": "cup", "x": 1.0, "y": 2.0, "confidence": 0.9},
                {"object_id": "obj2", "category": "plate", "x": 1.5, "y": 2.5, "confidence": 0.8},
            ],
            "bedroom": [
                {"object_id": "obj3", "category": "pillow", "x": 5.0, "y": 6.0, "confidence": 0.7},
            ],
        })
        mem = ObjectMemory()
        count = mem.sync_from_scene_graph(sg)
        assert count == 3
        assert "obj1" in mem._objects
        assert "obj2" in mem._objects
        assert "obj3" in mem._objects

    def test_sync_increments_observation_count(self):
        sg = _make_mock_scene_graph({
            "kitchen": [
                {"object_id": "obj1", "category": "cup", "x": 1.0, "y": 2.0, "confidence": 0.9},
            ],
        })
        mem = ObjectMemory()
        mem.sync_from_scene_graph(sg)
        assert mem._objects["obj1"].observation_count == 1

        # second sync — same object
        mem.sync_from_scene_graph(sg)
        assert mem._objects["obj1"].observation_count == 2

    def test_sync_updates_last_seen(self):
        sg = _make_mock_scene_graph({
            "kitchen": [
                {"object_id": "obj1", "category": "cup", "x": 1.0, "y": 2.0, "confidence": 0.9},
            ],
        })
        mem = ObjectMemory()
        mem.sync_from_scene_graph(sg)
        t1 = mem._objects["obj1"].last_seen

        time.sleep(0.05)
        mem.sync_from_scene_graph(sg)
        t2 = mem._objects["obj1"].last_seen
        assert t2 > t1

    def test_sync_adds_new_objects(self):
        sg1 = _make_mock_scene_graph({
            "kitchen": [
                {"object_id": "obj1", "category": "cup", "x": 1.0, "y": 2.0, "confidence": 0.9},
            ],
        })
        sg2 = _make_mock_scene_graph({
            "kitchen": [
                {"object_id": "obj1", "category": "cup", "x": 1.0, "y": 2.0, "confidence": 0.9},
                {"object_id": "obj_new", "category": "mug", "x": 2.0, "y": 3.0, "confidence": 0.8},
            ],
        })
        mem = ObjectMemory()
        mem.sync_from_scene_graph(sg1)
        assert "obj_new" not in mem._objects

        mem.sync_from_scene_graph(sg2)
        assert "obj_new" in mem._objects
        assert mem._objects["obj_new"].observation_count == 1

    def test_sync_returns_zero_for_empty_graph(self):
        sg = _make_mock_scene_graph({})
        mem = ObjectMemory()
        assert mem.sync_from_scene_graph(sg) == 0


# ---------------------------------------------------------------------------
# 4. last_seen
# ---------------------------------------------------------------------------


class TestLastSeen:
    def _mem_with_cup(self, seconds_ago: float = 300) -> ObjectMemory:
        mem = ObjectMemory(decay_lambda=0.001)
        past = time.time() - seconds_ago
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1",
            category="cup",
            room_id="kitchen",
            x=1.0,
            y=2.0,
            last_seen=past,
            base_confidence=0.9,
            observation_count=3,
        )
        return mem

    def test_last_seen_found(self):
        mem = self._mem_with_cup(300)
        result = mem.last_seen("cup")
        assert result is not None
        assert result["room"] == "kitchen"
        assert result["position"] == (1.0, 2.0)
        assert abs(result["seconds_ago"] - 300) < 2
        assert 0.0 < result["confidence"] < 0.9

    def test_last_seen_not_found(self):
        mem = self._mem_with_cup(300)
        assert mem.last_seen("laptop") is None

    def test_last_seen_substring_match(self):
        mem = self._mem_with_cup(300)
        # "cu" is substring of "cup"
        result = mem.last_seen("cu")
        assert result is not None
        assert result["room"] == "kitchen"

    def test_last_seen_case_insensitive(self):
        mem = self._mem_with_cup(300)
        result = mem.last_seen("CUP")
        assert result is not None

    def test_last_seen_returns_most_recent(self):
        """If two cups, returns the one seen more recently."""
        mem = ObjectMemory(decay_lambda=0.001)
        now = time.time()
        mem._objects["obj_old"] = TrackedObject(
            object_id="obj_old",
            category="cup",
            room_id="bedroom",
            x=5.0,
            y=6.0,
            last_seen=now - 1000,
            base_confidence=0.9,
            observation_count=1,
        )
        mem._objects["obj_new"] = TrackedObject(
            object_id="obj_new",
            category="cup",
            room_id="kitchen",
            x=1.0,
            y=2.0,
            last_seen=now - 100,
            base_confidence=0.9,
            observation_count=1,
        )
        result = mem.last_seen("cup")
        assert result["room"] == "kitchen"


# ---------------------------------------------------------------------------
# 5. certainty
# ---------------------------------------------------------------------------


class TestCertainty:
    def _mem_with_cup_in_kitchen(self) -> ObjectMemory:
        mem = ObjectMemory(decay_lambda=0.001)
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1",
            category="cup",
            room_id="kitchen",
            x=1.0,
            y=2.0,
            last_seen=time.time(),
            base_confidence=0.9,
            observation_count=1,
        )
        return mem

    def test_certainty_chinese(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("cup在kitchen")
        assert abs(result - 0.9) < 0.05

    def test_certainty_english(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("cup in kitchen")
        assert abs(result - 0.9) < 0.05

    def test_certainty_not_found(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("laptop在kitchen")
        assert result == 0.0

    def test_certainty_wrong_room(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("cup在bedroom")
        assert result == 0.0

    def test_certainty_bad_format(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("hello world")
        assert result == 0.0

    def test_certainty_empty_string(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("")
        assert result == 0.0

    def test_certainty_case_insensitive_category(self):
        mem = self._mem_with_cup_in_kitchen()
        result = mem.certainty("CUP在kitchen")
        assert abs(result - 0.9) < 0.05


# ---------------------------------------------------------------------------
# 6. objects_in_room
# ---------------------------------------------------------------------------


class TestObjectsInRoom:
    def test_objects_in_room(self):
        mem = ObjectMemory(decay_lambda=0.001)
        now = time.time()
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1", category="cup", room_id="kitchen",
            x=1.0, y=2.0, last_seen=now, base_confidence=0.9, observation_count=1,
        )
        mem._objects["obj2"] = TrackedObject(
            object_id="obj2", category="plate", room_id="kitchen",
            x=2.0, y=3.0, last_seen=now, base_confidence=0.8, observation_count=1,
        )
        mem._objects["obj3"] = TrackedObject(
            object_id="obj3", category="pillow", room_id="bedroom",
            x=5.0, y=6.0, last_seen=now, base_confidence=0.7, observation_count=1,
        )
        result = mem.objects_in_room("kitchen")
        assert len(result) == 2
        ids = {r["object_id"] for r in result}
        assert ids == {"obj1", "obj2"}
        # sorted by confidence descending
        assert result[0]["confidence"] >= result[1]["confidence"]

    def test_objects_in_room_filters_low_confidence(self):
        """Very old object should be filtered (confidence < 0.01)."""
        mem = ObjectMemory(decay_lambda=0.1)  # fast decay
        very_old = time.time() - 10000  # confidence near 0
        mem._objects["old_obj"] = TrackedObject(
            object_id="old_obj", category="cup", room_id="kitchen",
            x=1.0, y=2.0, last_seen=very_old, base_confidence=0.9, observation_count=1,
        )
        result = mem.objects_in_room("kitchen")
        assert len(result) == 0

    def test_objects_in_room_empty(self):
        mem = ObjectMemory()
        result = mem.objects_in_room("nonexistent_room")
        assert result == []

    def test_objects_in_room_dict_keys(self):
        mem = ObjectMemory()
        now = time.time()
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1", category="cup", room_id="kitchen",
            x=1.0, y=2.0, last_seen=now, base_confidence=0.9, observation_count=2,
        )
        result = mem.objects_in_room("kitchen")
        assert len(result) == 1
        d = result[0]
        assert "object_id" in d
        assert "category" in d
        assert "x" in d
        assert "y" in d
        assert "confidence" in d
        assert "seconds_ago" in d


# ---------------------------------------------------------------------------
# 7. find_object
# ---------------------------------------------------------------------------


class TestFindObject:
    def test_find_object(self):
        mem = ObjectMemory(decay_lambda=0.001)
        now = time.time()
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1", category="cup", room_id="kitchen",
            x=1.0, y=2.0, last_seen=now, base_confidence=0.9, observation_count=1,
        )
        mem._objects["obj2"] = TrackedObject(
            object_id="obj2", category="coffee_cup", room_id="bedroom",
            x=5.0, y=6.0, last_seen=now - 100, base_confidence=0.8, observation_count=1,
        )
        result = mem.find_object("cup")
        assert len(result) == 2
        # sorted confidence descending
        assert result[0]["confidence"] >= result[1]["confidence"]

    def test_find_object_not_found(self):
        mem = ObjectMemory()
        result = mem.find_object("laptop")
        assert result == []

    def test_find_object_dict_keys(self):
        mem = ObjectMemory()
        now = time.time()
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1", category="cup", room_id="kitchen",
            x=1.0, y=2.0, last_seen=now, base_confidence=0.9, observation_count=1,
        )
        result = mem.find_object("cup")
        d = result[0]
        assert "object_id" in d
        assert "category" in d
        assert "room" in d
        assert "x" in d
        assert "y" in d
        assert "confidence" in d
        assert "seconds_ago" in d


# ---------------------------------------------------------------------------
# 8. update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_new(self):
        mem = ObjectMemory()
        mem.update("obj1", "cup", "kitchen", 1.0, 2.0, confidence=0.9)
        assert "obj1" in mem._objects
        obj = mem._objects["obj1"]
        assert obj.category == "cup"
        assert obj.room_id == "kitchen"
        assert obj.x == 1.0
        assert obj.y == 2.0
        assert obj.base_confidence == 0.9
        assert obj.observation_count == 1

    def test_update_existing_increments_count(self):
        mem = ObjectMemory()
        mem.update("obj1", "cup", "kitchen", 1.0, 2.0, confidence=0.9)
        mem.update("obj1", "cup", "kitchen", 1.1, 2.1, confidence=0.85)
        obj = mem._objects["obj1"]
        assert obj.observation_count == 2
        assert obj.x == 1.1  # position updated
        assert obj.base_confidence == 0.85

    def test_update_refreshes_last_seen(self):
        mem = ObjectMemory()
        mem.update("obj1", "cup", "kitchen", 1.0, 2.0)
        t1 = mem._objects["obj1"].last_seen
        time.sleep(0.05)
        mem.update("obj1", "cup", "kitchen", 1.0, 2.0)
        t2 = mem._objects["obj1"].last_seen
        assert t2 > t1

    def test_update_default_confidence(self):
        mem = ObjectMemory()
        mem.update("obj1", "cup", "kitchen", 1.0, 2.0)
        assert mem._objects["obj1"].base_confidence == 0.9


# ---------------------------------------------------------------------------
# 9. to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestSerializationRoundtrip:
    def test_to_dict_from_dict_roundtrip(self):
        mem = ObjectMemory(decay_lambda=0.002)
        ts = time.time() - 300
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1", category="cup", room_id="kitchen",
            x=1.0, y=2.0, last_seen=ts, base_confidence=0.9, observation_count=3,
        )
        mem._objects["obj2"] = TrackedObject(
            object_id="obj2", category="plate", room_id="bedroom",
            x=5.0, y=6.0, last_seen=ts - 100, base_confidence=0.7, observation_count=1,
        )
        serialized = mem.to_dict()
        mem2 = ObjectMemory.from_dict(serialized, decay_lambda=0.002)

        assert set(mem2._objects.keys()) == {"obj1", "obj2"}
        obj1 = mem2._objects["obj1"]
        assert obj1.category == "cup"
        assert obj1.room_id == "kitchen"
        assert obj1.x == 1.0
        assert obj1.y == 2.0
        assert abs(obj1.last_seen - ts) < 0.001  # timestamp preserved
        assert obj1.base_confidence == 0.9
        assert obj1.observation_count == 3

    def test_from_dict_preserves_timestamps(self):
        """Deserialization must NOT reset last_seen to now()."""
        mem = ObjectMemory()
        old_ts = time.time() - 9999  # very old
        mem._objects["obj1"] = TrackedObject(
            object_id="obj1", category="cup", room_id="kitchen",
            x=0.0, y=0.0, last_seen=old_ts, base_confidence=0.9, observation_count=1,
        )
        data = mem.to_dict()
        mem2 = ObjectMemory.from_dict(data)
        assert abs(mem2._objects["obj1"].last_seen - old_ts) < 0.001

    def test_to_dict_empty(self):
        mem = ObjectMemory()
        assert mem.to_dict() == []

    def test_from_dict_empty(self):
        mem = ObjectMemory.from_dict([])
        assert len(mem._objects) == 0


# ---------------------------------------------------------------------------
# 10. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_thread_safety_concurrent_reads(self):
        mem = ObjectMemory()
        now = time.time()
        for i in range(20):
            mem._objects[f"obj{i}"] = TrackedObject(
                object_id=f"obj{i}", category="cup", room_id="kitchen",
                x=float(i), y=0.0, last_seen=now, base_confidence=0.9, observation_count=1,
            )

        errors = []

        def read_worker():
            try:
                for _ in range(100):
                    mem.objects_in_room("kitchen")
                    mem.find_object("cup")
                    mem.last_seen("cup")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

    def test_thread_safety_concurrent_updates(self):
        mem = ObjectMemory()
        errors = []

        def write_worker(worker_id: int):
            try:
                for i in range(20):
                    mem.update(f"obj_{worker_id}_{i}", "cup", "kitchen", float(i), 0.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_worker, args=(tid,)) for tid in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        assert len(mem._objects) == 100  # 5 workers * 20 objects
