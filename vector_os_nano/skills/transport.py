"""Prepare a declared carrying posture before public base-motion skills."""
import math
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


class CarryGuard:
    """A cancellation token that latches loss of a carried load; never releases it."""

    def __init__(self, context, cancel=None):
        self.context = context
        self.cancel = cancel if cancel is not None else context.services.get('cancel') or threading.Event()
        self.cfg = context.config.get('skills', {}).get('transport', {})
        self.error = None
        self.last_clock = self.last_sensor = -math.inf
        try:
            self.loaded = context.gripper is not None and context.gripper.is_holding()
        except Exception as exc:
            self._fail('device_failed', failure_cause(exc))

    def _fail(self, code, message):
        self.error = _fail(code, message + '; base stopped, gripper state preserved.')
        self.context.base.stop()

    def is_set(self):
        if self.error is not None or self.cancel.is_set():
            return True
        if not self.loaded:
            return False
        try:
            now = self.context.base.get_sim_time()
            if not math.isfinite(now) or now < self.last_clock:
                raise ValueError('carrying monitor clock is nonfinite or moved backwards')
            if now-self.last_clock < .02:
                return False
            self.last_clock = now
            if not self.context.gripper.is_holding():
                self._fail('not_holding', 'Grasp lost during base motion')
            elif 'minimum_load' in self.cfg:
                load, self.last_sensor = read(self.context.gripper, self.cfg, self.last_sensor)
                if load < self.cfg['minimum_load']:
                    self._fail('carry_not_ready', 'Suspended load lost during base motion')
        except Exception as exc:
            self._fail('device_failed', failure_cause(exc))
        return self.error is not None

    def failure(self):
        self.is_set()
        return self.error or _cancelled(self.cancel)
