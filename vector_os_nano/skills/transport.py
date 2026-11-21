"""Prepare a declared carrying posture before public base-motion skills."""
import threading
from vector_os_nano.schemas.robot import EmbodimentConfigError, _keys, _require
from vector_os_nano.skills.grasp_config import _positive
from vector_os_nano.skills.grasp import _cancelled, _move
from vector_os_nano.skills.grasp_perception import _fail
from vector_os_nano.skills.place_support import read
from vector_os_nano.perception.base import failure_cause


def parse(raw, cfg, ctx):
    if cfg.base is None or cfg.arm is None or cfg.arm.gripper is None:
        raise EmbodimentConfigError(f'{ctx}: requires a base, arm and gripper')
    row = _keys(raw, ctx, {'posture', 'duration', 'minimum_load', 'max_force'})
    posture = _require(row, 'posture', ctx)
    if not isinstance(posture, str) or posture not in cfg.arm.postures:
        raise EmbodimentConfigError(f'{ctx}.posture: expected a configured arm posture')
    result = dict(posture=posture, duration=_positive(_require(row, 'duration', ctx), f'{ctx}.duration'))
    if {'minimum_load', 'max_force'} & row.keys():
        result.update({k: _positive(_require(row, k, ctx), f'{ctx}.{k}') for k in ('minimum_load', 'max_force')})
        if not cfg.arm.gripper.force_feedback or result['minimum_load'] >= result['max_force']:
            raise EmbodimentConfigError(f'{ctx}: force limits require declared feedback and minimum_load < max_force')
    return result


def prepare(context):
    token = context.services.get('cancel') or threading.Event()
    if token.is_set():
        return _cancelled(token)
    arm, grip = context.arm, context.gripper
    try:
        if arm is None or grip is None or not grip.is_holding():
            return None
        from vector_os_nano.skills.go2.explore import cancel_exploration
        cancel_exploration()
        context.base.stop()
        cfg = context.config.get('skills', {}).get('transport')
        if cfg is None:
            return _fail('not_configured', 'Holding an object: configure a transport posture before moving the base.')
        before = None
        if 'minimum_load' in cfg:
            before = read(grip, cfg)
            if before[0] < cfg['minimum_load']:
                return _fail('carry_not_ready', 'Fingertips do not carry a suspended load; keep the base stopped and lift clear or finish placing.')
        error = _move(arm, context.config['arm']['postures'][cfg['posture']], cfg['duration'], 'carry posture', token)
        if error:
            return error
        if not grip.is_holding():
            return _fail('not_holding', 'Grasp lost during carry preparation; base remains stopped.')
        if before is not None and read(grip, cfg, before[1])[0] < cfg['minimum_load']:
            return _fail('carry_not_ready', 'Load was not maintained in the carrying posture; base remains stopped.')
    except Exception as exc:
        return _cancelled(token) or _fail('device_failed', f'Cannot confirm carrying readiness: {failure_cause(exc)}; base remains stopped.')
    return None
