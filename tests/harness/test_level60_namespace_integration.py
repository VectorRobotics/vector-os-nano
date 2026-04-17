"""Test Phase 3 namespace extension — GoalVerifier can use new functions."""
import time
import pytest
from unittest.mock import MagicMock
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.object_memory import ObjectMemory


class TestNamespaceExtension:
    """GoalVerifier with Phase 3 functions."""

    @pytest.fixture
    def object_memory(self):
        om = ObjectMemory()
        om.update("obj1", "cup", "kitchen", 5.0, 3.0, 0.9)
        om.update("obj2", "chair", "kitchen", 5.5, 3.5, 0.8)
        om.update("obj3", "book", "bedroom", 2.0, 1.0, 0.7)
        return om

    @pytest.fixture
    def namespace(self, object_memory):
        ns = {
            "last_seen": object_memory.last_seen,
            "certainty": object_memory.certainty,
            "objects_in_room": object_memory.objects_in_room,
            "find_object": object_memory.find_object,
            "room_coverage": lambda room_id: 0.6 if room_id == "kitchen" else 0.0,
            "predict_navigation": lambda target: {"reachable": target == "kitchen", "door_count": 1, "estimated_steps": 2, "rooms_on_path": ["hallway", "kitchen"], "confidence": 1.0},
            # Keep existing functions
            "nearest_room": lambda: "hallway",
            "get_visited_rooms": lambda: ["hallway", "kitchen"],
            "query_rooms": lambda: [{"id": "hallway"}, {"id": "kitchen"}],
            "world_stats": lambda: {"rooms": 2, "objects": 3, "visited_rooms": 2},
            "describe_scene": lambda: "",
            "detect_objects": lambda query="": [],
        }
        return ns

    def test_verify_last_seen_found(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("last_seen('cup') is not None") is True

    def test_verify_last_seen_not_found(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("last_seen('laptop') is not None") is False

    def test_verify_certainty_above_threshold(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("certainty('cup在kitchen') > 0.5") is True

    def test_verify_certainty_wrong_room(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("certainty('cup在bedroom') > 0.5") is False

    def test_verify_objects_in_room(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("len(objects_in_room('kitchen')) >= 2") is True

    def test_verify_find_object(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("len(find_object('cup')) > 0") is True

    def test_verify_room_coverage(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("room_coverage('kitchen') > 0.5") is True

    def test_verify_predict_navigation_reachable(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("predict_navigation('kitchen')['reachable']") is True

    def test_verify_predict_navigation_unreachable(self, namespace):
        v = GoalVerifier(namespace)
        assert v.verify("predict_navigation('garage')['reachable']") is False

    def test_verify_combined_expression(self, namespace):
        """Compound expression using multiple Phase 3 functions."""
        v = GoalVerifier(namespace)
        assert v.verify("certainty('cup在kitchen') > 0.3 and room_coverage('kitchen') > 0.3") is True


class TestDecomposerWhitelist:
    """GoalDecomposer accepts new verify functions."""

    def test_new_functions_in_whitelist(self):
        new_fns = {"last_seen", "certainty", "objects_in_room", "find_object", "room_coverage", "predict_navigation"}
        assert new_fns.issubset(GoalDecomposer.VERIFY_FUNCTIONS)

    def test_new_functions_have_signatures(self):
        new_fns = {"last_seen", "certainty", "objects_in_room", "find_object", "room_coverage", "predict_navigation"}
        for fn in new_fns:
            assert fn in GoalDecomposer._VERIFY_FN_SIGNATURES, f"Missing signature for {fn}"

    def test_validate_verify_accepts_new_functions(self):
        d = GoalDecomposer(backend=MagicMock())
        assert d._validate_verify("last_seen('cup') is not None") is not None
        assert d._validate_verify("certainty('cup在kitchen') > 0.5") is not None
        assert d._validate_verify("len(objects_in_room('kitchen')) > 0") is not None
        assert d._validate_verify("len(find_object('cup')) > 0") is not None
        assert d._validate_verify("room_coverage('kitchen') > 0.5") is not None
        assert d._validate_verify("predict_navigation('kitchen')['reachable']") is not None
