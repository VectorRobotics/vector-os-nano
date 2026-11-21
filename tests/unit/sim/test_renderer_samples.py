"""Segmentation IDs must not pass through multisample colour averaging."""
from types import SimpleNamespace

import pytest

from vector_os_nano.sim.backends.mujoco import world as backend


@pytest.mark.parametrize("kind, expected", [("seg", 0), ("rgb", 4), ("depth", 4)])
@pytest.mark.parametrize("fails", [False, True])
def test_renderer_samples_are_local_and_restored(monkeypatch, kind, expected, fails):
    quality = SimpleNamespace(offsamples=4)
    world = SimpleNamespace(model=SimpleNamespace(vis=SimpleNamespace(quality=quality)), _renderers={})
    calls = []
    renderer = SimpleNamespace(scene=SimpleNamespace(flags={}),
                               enable_segmentation_rendering=lambda: None,
                               enable_depth_rendering=lambda: None,
                               disable_depth_rendering=lambda: None)

    def create(model, height, width):
        calls.append((model.vis.quality.offsamples, height, width))
        if fails:
            raise RuntimeError("context unavailable")
        return renderer

    monkeypatch.setattr(backend.mj, "Renderer", create)
    if fails:
        with pytest.raises(RuntimeError, match="context unavailable"):
            backend.MuJoCoWorld._renderer(world, kind, 320, 240)
        assert not world._renderers
    else:
        assert backend.MuJoCoWorld._renderer(world, kind, 320, 240) is renderer
        assert backend.MuJoCoWorld._renderer(world, kind, 320, 240) is renderer
    assert calls == [(expected, 240, 320)]
    assert quality.offsamples == 4


def test_rgb_depth_share_assets_but_segmentation_keeps_its_own_context(monkeypatch):
    from unittest.mock import Mock

    world = SimpleNamespace(model=SimpleNamespace(vis=SimpleNamespace(
        quality=SimpleNamespace(offsamples=4))), _renderers={})
    create = Mock(side_effect=lambda *args: Mock(scene=SimpleNamespace(flags={})))
    monkeypatch.setattr(backend.mj, 'Renderer', create)
    rgb = backend.MuJoCoWorld._renderer(world, 'rgb', 320, 240)
    assert backend.MuJoCoWorld._renderer(world, 'depth', 320, 240) is rgb
    rgb.enable_depth_rendering.assert_called_once()
    assert backend.MuJoCoWorld._renderer(world, 'rgb', 320, 240) is rgb
    assert rgb.disable_depth_rendering.call_count == 2
    seg = backend.MuJoCoWorld._renderer(world, 'seg', 320, 240)
    assert seg is not rgb and create.call_count == 2
    backend.MuJoCoWorld._close_renderers(world)
    rgb.close.assert_called_once()
    seg.close.assert_called_once()
