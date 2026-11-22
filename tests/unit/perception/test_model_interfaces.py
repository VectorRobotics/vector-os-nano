"""Model adapters expose the public contracts without loading their weights."""
import pytest


def test_model_rows_build_lazily_and_reject_unknown_names():
    from vector_os_nano.interfaces import Detector, Segmenter
    from vector_os_nano.perception.models import build
    from vector_os_nano.schemas.user import DETECTORS, SEGMENTERS
    for name in DETECTORS:
        assert isinstance(build(DETECTORS, name), Detector)
    for name in SEGMENTERS:
        assert isinstance(build(SEGMENTERS, name), Segmenter)
    for table in (DETECTORS, SEGMENTERS):
        with pytest.raises(ValueError, match='valid:'):
            build(table, 'misspelled-model')


def test_camera_and_routed_detect_use_the_same_model_rows(monkeypatch):
    import numpy as np
    from vector_os_nano.interfaces import Detection
    from vector_os_nano.schemas import user as models
    from vector_os_nano.perception.camera_perception import CameraPerception
    from vector_os_nano.perception.detector_capability import DetectorCapability
    from vector_os_nano.perception import rynnbrain
    expected = [Detection('cup', (1., 2., 3., 4.), .9)]
    seen = []
    class Detector:
        def detect(self, rgb, query):
            seen.append((rgb, query))
            return expected
    monkeypatch.setattr(rynnbrain, 'RynnBrainDetector', Detector)
    monkeypatch.setitem(models.DETECTORS, 'test-model', models.DETECTORS['rynnbrain'])
    rgb = np.zeros((8, 8, 3), np.uint8)
    perception = CameraPerception(None, detector_name='test-model')
    assert perception.detect_in(rgb, 'cup') == expected
    result = DetectorCapability().invoke({'rgb': rgb, 'query': 'cup', 'backend': 'test-model'}, None)
    assert result.success and result.output['backend'] == 'test-model'
    assert len(seen) == 2 and all(pixels is rgb and query == 'cup' for pixels, query in seen)


@pytest.mark.parametrize('name', ['grounding-dino', 'rynnbrain'])
def test_detector_adapters_obey_the_public_contract(name):
    from vector_os_nano.interfaces import Detection, Detector
    from vector_os_nano.perception.grounding_dino import GroundingDinoDetector
    from vector_os_nano.perception.rynnbrain import RynnBrainDetector
    model = GroundingDinoDetector() if name == 'grounding-dino' else RynnBrainDetector()
    assert isinstance(model, Detector)
    sample = Detection('cup', (1., 2., 30., 40.), .8)
    assert Detection.from_dict(sample.to_dict()) == sample


def test_edgetam_implements_the_public_segmenter_contract():
    from vector_os_nano.interfaces import Segmenter
    from vector_os_nano.perception.tracker import EdgeTAMTracker
    assert isinstance(EdgeTAMTracker(), Segmenter)
