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


@pytest.mark.parametrize('attack', ['none', 'no_command', 'no_jaw_command', 'unchanged', 'wrong_drawer', 'nonfinite'])
def test_drawer_causation_requires_its_truth_transition_and_actuation(attack):
    from vector_os_nano.vcli.cognitive.actor_causation import capture, grade
    state = {'q': 0., 'command': 0., 'jaw': 0.}
    gt = SimpleNamespace(has_base=False, attachments=lambda: {}, joint_positions=lambda _: [state['q']])
    agent = SimpleNamespace(_base=SimpleNamespace(cmd_motion=lambda: state['command']),
                            _gripper=SimpleNamespace(ctrl_motion=lambda: state['jaw']))
    session = SimpleNamespace(uid='drawer-test', gt=gt, world=world())
    before = capture(agent, session)
    state.update(q=-.28, command=1., jaw=.08)
    if attack == 'no_command': state['command'] = 0.
    if attack == 'no_jaw_command': state['jaw'] = 0.
    if attack == 'unchanged': state['q'] = 0.
    if attack == 'nonfinite': state['q'] = float('nan')
    after = capture(agent, session)
    if attack == 'wrong_drawer': after = replace(after, articulations={'other': -.28})
    ns = build_namespace(gt, session.world, SimpleNamespace(arm=None, camera=None))
    assert grade(before, after, "articulation_open('storage drawer')", ns, world=session.world).value == ('CAUSED' if attack == 'none' else 'UNCAUSED')


def test_drawer_verify_hint_binds_furniture_without_making_it_pickable(monkeypatch):
    from vector_os_nano.vcli import native_runner
    from vector_os_nano.skills.drawer import OpenDrawerSkill
    ns = build_namespace(SimpleNamespace(has_base=False), world(), SimpleNamespace(arm=None, camera=None))
    runner = SimpleNamespace(_world=None, _agent=None, _oracles=ns, _oracle_names=frozenset(ns))
    monkeypatch.setattr(native_runner, 'sim_session', lambda *_: SimpleNamespace(world=world()))
    hint = native_runner.NativeStepRunner._bound_hint
    assert hint(runner, OpenDrawerSkill(), {'query': 'storage drawer'}) == "articulation_open('storage drawer')"
    assert hint(runner, OpenDrawerSkill(), {'query': 'not the drawer'}) is None


@pytest.mark.parametrize('text,predicate', [('open the storage drawer', 'articulation_open'),
    ('打开收纳抽屉', 'articulation_open'), ('close the drawer', 'articulation_closed'),
    ('把收纳抽屉关好', 'articulation_closed'), ('do not open the drawer', None),
    ('open the drawer halfway', None), ('open drawer and close drawer', None),
    ('open the gripper', None), ('close drawer if it is empty', None)])
def test_only_whole_named_drawer_goals_compile(monkeypatch, text, predicate):
    from vector_os_nano.vcli.cognitive import object_goal as goals
    monkeypatch.setattr(goals, '_manifest', lambda _: world())
    names = frozenset({'articulation_open', 'articulation_closed'})
    expected = () if predicate is None else (f"{predicate}('drawer')",)
    assert goals.task_contract(text, names, 'test') == expected
    if predicate:
        assert goals.backs_contract(expected[0], f"{predicate}('storage drawer')", 'test')
        assert not goals.backs_contract(expected[0], f"{predicate}('other drawer')", 'test')
        assert goals.task_contract(text, frozenset(), 'test') == ()
