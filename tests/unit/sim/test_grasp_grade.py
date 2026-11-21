"""Independent recorded contact loads reject false possession and hidden force cancellation."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scripts.grade_grasp import contact_support


@pytest.mark.parametrize('attack,failed', [('none', None), ('same_side', 'sustained_opposing_loads'),
    ('gap', 'sustained_opposing_loads'), ('proximity', 'sustained_opposing_loads'),
    ('no_weight_support', 'load_support'), ('excess_force', 'finger_force_limit'),
    ('force_cancellation', 'finger_force_limit'), ('short', 'sustained_opposing_loads'),
    ('wrong_object', 'sustained_opposing_loads'), ('empty', 'sustained_opposing_loads'),
    ('negative_time', 'invalid'), ('future_time', 'invalid'), ('between_steps', 'invalid')])
def test_contact_loads_must_sustain_and_support_the_named_object(attack, failed):
    model = NS(opt=NS(timestep=.1, gravity=np.array([0, 0, -9.81])),
               body_mass=np.array([0, 1, 1, .1]), body=lambda n: NS(name=f'body{n}'))
    times = np.arange(21)*.1
    states = {'sim_time': times[:, None]}
    contacts = dict(sim_time=np.repeat(times, 2)[:, None],
        body_a=np.tile([1, 2], len(times))[:, None], body_b=np.full((len(times)*2, 1), 3),
        normal_on_b=np.tile([[1., 0, 0], [-1., 0, 0]], (len(times), 1)),
        force_on_b=np.tile([[1., 0, .4905], [-1., 0, .4905]], (len(times), 1)))
    if attack == 'same_side':
        contacts['normal_on_b'][1::2, 0] = 1
        contacts['force_on_b'][1::2, 0] = 1
    if attack == 'gap':
        contacts = {k: np.delete(v, 20, axis=0) for k, v in contacts.items()}
    if attack == 'proximity': contacts['force_on_b'] *= .01
    if attack == 'no_weight_support': contacts['force_on_b'][:, 2] = 0
    if attack == 'excess_force': contacts['force_on_b'][0, 0] = 8
    if attack == 'short':
        states['sim_time'] = times[:-1, None]
        contacts = {k: v[:-2] for k, v in contacts.items()}
    if attack == 'wrong_object': contacts['body_b'][:] = 4
    if attack == 'empty': contacts = {k: v[:0] for k, v in contacts.items()}
    if attack in ('negative_time', 'future_time', 'between_steps'):
        contacts['sim_time'][0] = {'negative_time': -.1, 'future_time': 1e9, 'between_steps': .01}[attack]
        with pytest.raises(ValueError, match='physics steps'):
            contact_support(model, states, contacts, 3, [1, 2])
        return
    if attack == 'force_cancellation':
        for k in contacts:
            contacts[k] = np.concatenate([contacts[k], contacts[k][[0, 0]]])
        contacts['force_on_b'][-2:] = [[0, 100, 0], [0, -100, 0]]
    checks, _ = contact_support(model, states, contacts, 3, [1, 2])
    assert all(checks.values()) if failed is None else checks[failed] is False
