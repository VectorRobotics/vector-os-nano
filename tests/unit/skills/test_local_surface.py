"""Moving supports use current pixels, with explicit authored orientation and geometry."""
from pathlib import Path
import math
from unittest.mock import Mock

import numpy as np
import pytest

from vector_os_nano.schemas.world import Surface, SurfaceAnchor, WorldManifestError, parse_world
from vector_os_nano.skills import place
from tests.unit.skills.test_place import context
from tests.unit.skills.test_drawer import config
from vector_os_nano.schemas.robot import parse_embodiment_config, EmbodimentConfigError
from vector_os_nano.embodiments.registry import validate


def surface():
    return Surface('bench', 'drawer', (-.2, -.2, .2, .3), -.07, place_point=(.1, .2),
        local_anchor=SurfaceAnchor('drawer knob', (0., 0., 0.), -math.pi/2, .06), ceiling=.1)


@pytest.mark.parametrize('fault', [None, 'missing_anchor', 'missing_ceiling', 'missing_point', 'world_body', 'reversed_height', 'nan_yaw', 'zero_extent'])
def test_local_surface_manifest_requires_complete_finite_geometry(fault):
    row=dict(body='drawer', box=[-.2, -.2, .2, .3], height=-.07, ceiling=.1, place_point=[.1,.2],
             local_anchor=dict(query='drawer knob', point=[0,0,0], yaw=0, max_extent=.06))
    if fault in ('missing_anchor','missing_ceiling','missing_point'):
        row.pop({'missing_anchor':'local_anchor','missing_ceiling':'ceiling','missing_point':'place_point'}[fault])
    if fault=='world_body':row['body']='world'
    if fault=='reversed_height':row['ceiling']=-.1
    if fault=='nan_yaw':row['local_anchor']['yaw']=math.nan
    if fault=='zero_extent':row['local_anchor']['max_extent']=0
    raw=dict(schema=1,scene='scene.xml',spawn=dict(xy=[0,0],z=0,heading=0),surfaces={'bin':row})
    if fault:
        with pytest.raises(WorldManifestError):parse_world(raw,'local',Path('world.yaml'))
    else:
        result=parse_world(raw,'local',Path('world.yaml')).surfaces['bin']
        assert result.local_anchor.query=='drawer knob' and result.ceiling==.1


def test_landmark_translation_is_fresh_and_does_not_mutate_the_manifest(monkeypatch):
    ctx=context();ctx.perception_sources['head']=Mock();s=surface()
    observation={'visible_bounds':[[1.99,2.99,.59],[2.01,3.01,.61]],'frame':{'sim_time':12.}}
    perceive=Mock(return_value=(None,observation,None));monkeypatch.setattr(place,'perceive',perceive)
    result,seen,error=place.localize(s,ctx,{'camera':'head'},ctx.services['cancel'])
    assert error is None and seen is observation
    np.testing.assert_allclose([*result.place_point,result.height],[2.2,2.9,.53])
    assert s.place_point==(.1,.2) and s.height==-.07
    assert perceive.call_args.kwargs['camera']=='head'


def test_place_reobserves_after_approaching_and_uses_the_new_location(monkeypatch):
    ctx=context();ctx.perception_sources['head']=Mock();ctx.services['object_world'].surfaces['bench']=surface()
    observations=[{'visible_bounds':[[x-.01,-.01,.49],[x+.01,.01,.51]]} for x in (0.,.1)]
    perceive=Mock(side_effect=[(None,o,None) for o in observations]);monkeypatch.setattr(place,'perceive',perceive)
    result=place.PlaceSkill().execute({'surface':'bench'},ctx)
    assert result.success and perceive.call_count==2
    np.testing.assert_allclose(result.result_data['placed_at'],[.3,-.1,.43])
    assert result.result_data['observation']==observations[-1]
    np.testing.assert_allclose(ctx.arm.move_joints.call_args_list[0].args[0][:2],[.3,-.1])


@pytest.mark.parametrize('bounds', [None, [[0,0,0],[.2,.02,.02]], [[0,0,0],[.01,.01,math.nan]]])
def test_missing_or_implausible_anchor_never_releases(bounds,monkeypatch):
    ctx=context();ctx.services['object_world'].surfaces['bench']=surface()
    ctx.perception_sources['head']=Mock() if bounds is not None else None
    monkeypatch.setattr(place,'perceive',lambda *a,**kw:(None,{'visible_bounds':bounds},None))
    assert not place.PlaceSkill().execute({'surface':'bench'},ctx).success
    ctx.gripper.open.assert_not_called();ctx.arm.move_joints.assert_not_called()


@pytest.mark.parametrize('camera',['head','missing',12])
def test_placement_camera_is_a_declared_rgbd_sensor(camera):
    raw=config();raw['skills']['place']['camera']=camera
    if camera=='head':assert validate(parse_embodiment_config(raw)).skills['place']['camera']=='head'
    else:
        with pytest.raises(EmbodimentConfigError,match='place.camera'):validate(parse_embodiment_config(raw))


@pytest.mark.parametrize('lost_on', [1,2])
def test_slip_during_landmark_observation_stops_before_placement(monkeypatch,lost_on):
    ctx=context();ctx.perception_sources['head']=Mock();ctx.services['object_world'].surfaces['bench']=surface()
    observations=0
    def perceive(*a,**kw):
        nonlocal observations
        observations+=1
        ctx.gripper.is_holding.return_value=observations<lost_on
        return None,{'visible_bounds':[[0,0,.5],[.02,.02,.52]]},None
    monkeypatch.setattr(place,'perceive',perceive)
    reach=Mock(return_value=([],None));monkeypatch.setattr(place,'reach',reach)
    assert place.PlaceSkill().execute({'surface':'bench'},ctx).diagnosis_code=='not_holding'
    assert reach.call_count==lost_on-1
    ctx.arm.move_joints.assert_not_called();ctx.gripper.open.assert_not_called()
