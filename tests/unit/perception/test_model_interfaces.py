"""Model adapters expose the public contracts without loading their weights."""
import pytest


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
