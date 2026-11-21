# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Robot-specific grasp geometry and poses are validated before any motion."""
import copy

import pytest
import yaml
from vector_os_nano.embodiments.registry import validate

from vector_os_nano.schemas.robot import _EMBODIMENTS_DIR, EmbodimentConfigError, parse_embodiment_config


def raw():
    r = yaml.safe_load((_EMBODIMENTS_DIR / "piper" / "robot.yaml").read_text())
    r["arm"]["postures"] = {"view": [0., 1.2, 0., 0., 0., 0.]}
    r["skills"] = {"grasp": {"pre_grasp_height": .05, "grasp_z_above": .01,
        "lift_height": .08, "approach_back": .1, "durations": [4., 1.5, 2., 2.],
        "reach_m": .7, "view_duration": 1.2, "hold_duration": 2.}}
    return r


def test_grasp_data_and_named_poses_survive_loading():
    cfg = validate(parse_embodiment_config(raw()))
    assert cfg.arm.postures["view"] == (0., 1.2, 0., 0., 0., 0.)
    assert cfg.skills["grasp"]["pre_grasp_height"] == .05
    assert cfg.skills["grasp"]["durations"] == (4., 1.5, 2., 2.)


@pytest.mark.parametrize("edit,key", [
    (lambda r: r["arm"]["postures"].update(view=[0.]), "arm.postures.view"),
    (lambda r: r["arm"]["postures"].update(view=[float('nan')]*6), "arm.postures.view"),
    (lambda r: r["skills"]["grasp"].update(pre_grasp_height=float('nan')), "skills.grasp.pre_grasp_height"),
    (lambda r: r["skills"]["grasp"].update(durations=[4., 0., 2., 2.]), "skills.grasp.durations"),
    (lambda r: r["skills"]["grasp"].update(reach_m=-1.), "skills.grasp.reach_m"),
    (lambda r: r["skills"]["grasp"].update(typo=1.), "skills.grasp"),
    (lambda r: r["skills"].update(unknown={}), "skills"),
    (lambda r: r.pop("arm"), "skills.grasp"),
])
def test_invalid_grasp_data_names_its_key(edit, key):
    r = copy.deepcopy(raw())
    edit(r)
    with pytest.raises(EmbodimentConfigError) as exc:
        validate(parse_embodiment_config(r, ctx="variant/robot.yaml"), ctx="variant/robot.yaml")
    assert "variant/robot.yaml" in str(exc.value) and key in str(exc.value)


def test_schema_leaves_owner_blocks_open_but_start_checks_installed_owners():
    r = raw()
    r['skills'] = {'external': {'its_own_key': 42}}
    cfg = parse_embodiment_config(r)
    assert cfg.skills == r['skills']
    with pytest.raises(EmbodimentConfigError, match='unknown owner.*external.*valid'):
        validate(cfg)
