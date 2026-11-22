"""The shipped config reader selects models for both camera and routed perception."""
import numpy as np
import pytest

from vector_os_nano.schemas import user


def config(tmp_path, monkeypatch, body):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path))
    path = tmp_path/'config.yaml'
    path.write_text('schema: 1\nperception:\n' + body)
    return path


def test_config_reaches_camera_routed_detector_and_scene_vlm(tmp_path, monkeypatch):
    from vector_os_nano.interfaces import Detection
    from vector_os_nano.perception import rynnbrain
    from vector_os_nano.perception.detector_capability import DetectorCapability
    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
    from vector_os_nano.sim.session import _perception
    path = config(tmp_path, monkeypatch, '  detector: rynnbrain\n  segmenter: edgetam\n'
                  '  scene_vlm: {model: local-vision, base_url: http://localhost:11434/v1, api_key_env: null}\n')
    settings = user.load(path).perception
    assert settings.detector == 'rynnbrain' and settings.segmenter == 'edgetam'
    seen = []
    class Detector:
        def detect(self, rgb, query):
            seen.append(query)
            return [Detection('cup', (1., 2., 3., 4.), .9)]
    monkeypatch.setattr(rynnbrain, 'RynnBrainDetector', Detector)
    camera = _perception(object())
    rgb = np.zeros((8, 8, 3), np.uint8)
    assert camera.detect_in(rgb, 'cup')
    result = DetectorCapability(perception=camera).invoke({'rgb': rgb, 'query': 'cup'}, None)
    assert result.success and result.output['backend'] == 'rynnbrain'
    direct = DetectorCapability().invoke({'rgb': rgb, 'query': 'cup'}, None)
    assert direct.success and direct.output['backend'] == 'rynnbrain'
    assert seen == ['cup', 'cup', 'cup']
    vlm = Go2VLMPerception()
    assert vlm._base_url == 'http://localhost:11434/v1'
    assert vlm._model == 'local-vision' and vlm._local


@pytest.mark.parametrize('name', ['VECTOR_VLM_URL', 'VECTOR_VLM_MODEL'])
def test_retired_selection_vars_refuse_to_switch_an_existing_endpoint(tmp_path, monkeypatch, name):
    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
    config(tmp_path, monkeypatch, '  scene_vlm: {api_key_env: null}\n')
    monkeypatch.setenv(name, 'old-local-selection')
    with pytest.raises(ValueError, match='config.yaml'):
        Go2VLMPerception()


@pytest.mark.parametrize('body,key', [
    ('  detector: missing\n', 'detector'), ('  segmenter: missing\n', 'segmenter'),
    ('  detector: [rynnbrain]\n', 'detector'), ('  extra: true\n', 'extra'),
    ('  scene_vlm: true\n', 'scene_vlm'), ('  scene_vlm: {model: 3}\n', 'model'),
    ('  scene_vlm: {base_url: file:///tmp/model}\n', 'base_url'),
    ('  scene_vlm: {base_url: "http://[invalid/v1"}\n', 'base_url'),
    ('  scene_vlm: {base_url: "https://user:password@example.com/v1"}\n', 'base_url'),
    ('  scene_vlm: {api_key_env: 3}\n', 'api_key_env'),
    ('  scene_vlm: {api_key: sk-PRIVATE}\n', 'api_key'),
])
def test_bad_models_and_inline_credentials_are_refused(tmp_path, monkeypatch, body, key):
    path = config(tmp_path, monkeypatch, body)
    with pytest.raises(user.UserConfigError) as error:
        user.load(path)
    assert key in str(error.value) and 'sk-PRIVATE' not in str(error.value)


def test_scene_vlm_can_be_disabled_and_credentials_use_named_env(tmp_path, monkeypatch):
    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
    path = config(tmp_path, monkeypatch, '  scene_vlm: null\n')
    assert user.load(path).perception.scene_vlm is None
    with pytest.raises(ValueError, match='disabled'):
        Go2VLMPerception()
    config(tmp_path, monkeypatch, '  scene_vlm: {api_key_env: MY_VISION_KEY}\n')
    monkeypatch.delenv('MY_VISION_KEY', raising=False)
    with pytest.raises(ValueError, match='MY_VISION_KEY'):
        Go2VLMPerception()
    monkeypatch.setenv('MY_VISION_KEY', 'fixture-key')
    vlm = Go2VLMPerception()
    assert vlm._api_key == 'fixture-key' and not vlm._local
