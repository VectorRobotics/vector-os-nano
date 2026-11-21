"""Placement geometry is explicit, finite, and constrained by its owner manifest."""
from copy import deepcopy
from dataclasses import replace
import pytest
from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.robot import EmbodimentConfigError
from vector_os_nano.schemas.world import WorldManifestError, parse_world


@pytest.mark.parametrize('key,value', [('hover_height', .01), ('release_height', -1),
    ('settle_duration', float('nan')), ('durations', [1, 0, 1]), ('keep_orientation', 'yes'), ('invented', 1)])
def test_bad_place_config_rejected_before_start(key, value):
    cfg = registry.load('piper'); skills = deepcopy(cfg.skills)
    skills['place'][key] = value
    with pytest.raises(EmbodimentConfigError):
        registry.validate(replace(cfg, skills=skills))


@pytest.mark.parametrize('point', [[2, .5], [0, float('nan')], [True, .5], [0]])
def test_bad_surface_place_point_is_not_a_navigation_target(tmp_path, point):
    raw = dict(schema=1, scene='scene.xml', spawn=dict(xy=[0, 0], z=0, heading=0),
               surfaces={'table': dict(body='table', box=[0, 0, 1, 1], height=.4, place_point=point)})
    with pytest.raises(WorldManifestError):
        parse_world(raw, 'invalid', tmp_path)


@pytest.mark.parametrize('fault', ['undeclared', 'assisted', 'range', 'step', 'fraction', 'load', 'nonfinite', 'boolean'])
def test_support_feedback_configuration_fails_closed(fault):
    cfg = registry.load('piper')
    gripper = replace(cfg.arm.gripper, force_feedback=True, assist=False)
    if fault == 'undeclared': gripper = replace(gripper, force_feedback=False)
    if fault == 'assisted': gripper = replace(gripper, assist=True)
    support = dict(max_descent=.02, step=.002, duration=.25, dwell=.5,
                   minimum_load=.2, unload_fraction=.25, max_force=8.)
    if fault == 'range': support['max_descent'] = .05
    if fault == 'step': support['step'] = .03
    if fault == 'fraction': support['unload_fraction'] = 1.
    if fault == 'load': support['minimum_load'] = 8.
    if fault == 'nonfinite': support['duration'] = float('nan')
    if fault == 'boolean': support['dwell'] = True
    skills = deepcopy(cfg.skills); skills['place']['support'] = support
    with pytest.raises(EmbodimentConfigError):
        registry.validate(replace(cfg, arm=replace(cfg.arm, gripper=gripper), skills=skills))
