# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Simulation uses public user config and never reads inline model credentials."""
import pytest

from vector_os_nano.schemas.user import UserConfigError
from vector_os_nano.sim.session import _user_config


@pytest.mark.parametrize('block', ['llm', 'perception:\n  scene_vlm'])
def test_inline_keys_are_refused_before_session_construction(tmp_path, monkeypatch, block):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path))
    indent = '    ' if block.startswith('perception') else '  '
    (tmp_path/'config.yaml').write_text(f'schema: 1\n{block}:\n{indent}api_key: sk-PRIVATE\n')
    with pytest.raises(UserConfigError, match='looks like a secret') as error:
        _user_config()
    assert 'sk-PRIVATE' not in str(error.value)


def test_session_reads_canonical_model_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path))
    (tmp_path/'config.yaml').write_text('schema: 1\nperception:\n  detector: rynnbrain\n'
                                      '  scene_vlm: {api_key_env: MY_VISION_KEY, model: chosen-model}\n')
    cfg = _user_config()['perception']
    assert cfg['detector'] == 'rynnbrain'
    assert cfg['scene_vlm']['model'] == 'chosen-model'
    assert cfg['scene_vlm']['api_key_env'] == 'MY_VISION_KEY'
    assert 'api_key' not in cfg['scene_vlm']
