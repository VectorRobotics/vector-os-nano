"""Exact furniture references and independent joint-state predicates."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vector_os_nano.schemas.world import Articulation, load_world
from vector_os_nano.vcli.worlds.sim_oracles import build_namespace


def world():
    drawer = Articulation('cabinet/drawer', 'drawer_slide', 0., -.27, .003,
                          ('storage drawer', '收纳抽屉'))
    return replace(load_world('workbench'), articulations={'drawer': drawer})


@pytest.mark.parametrize('text,expected', [('drawer', 'drawer'), ('the storage drawer', 'drawer'),
    ('收纳抽屉', 'drawer'), ('not the drawer', None), ('drawer and cup', None), (None, None)])
def test_whole_articulation_reference(text, expected):
    assert world().resolve_articulation(text) == expected


def test_ambiguous_furniture_is_not_bound_and_is_not_pickable():
    w = world()
    w = replace(w, articulations={**w.articulations, 'other': w.articulations['drawer']})
    assert w.resolve_articulation('storage drawer') is None
    assert w.resolve('drawer') is None


@pytest.mark.parametrize('q,opened,closed', [(0., False, True), (-.002, False, True),
    (-.004, False, False), (-.269, False, False), (-.27, True, False), (-.28, True, False),
    (float('nan'), False, False)])
def test_oracles_read_only_independent_state(q, opened, closed):
    class GT:
        has_base = False
        def joint_positions(self, names):
            assert names == ('drawer_slide',)
            return [q]
    ns = build_namespace(GT(), world(), SimpleNamespace(arm=None, camera=None))
    assert ns['articulation_open']('storage drawer') is opened
    assert ns['articulation_closed']('drawer') is closed
    assert not ns['articulation_closed']('not drawer')
    assert ns['articulation_closed'].__oracle__.transition == 'rise'


@pytest.mark.parametrize('name', ['articulation_open', 'articulation_closed'])
def test_vacuous_drawer_verification_never_grounds(name):
    from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
    oracles = frozenset({name})
    assert classify_verify_expr(f"{name}('drawer')", oracles) == 'GROUNDED'
    for expr in (f"{name}('drawer') < 2", f"{name}('drawer') == False", f"not {name}('drawer')"):
        assert classify_verify_expr(expr, oracles) == 'RAN'
