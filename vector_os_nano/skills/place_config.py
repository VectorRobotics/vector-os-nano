"""Place-owned manifest parameters; no morphology defaults in the skill."""
from vector_os_nano.schemas.robot import EmbodimentConfigError, _keys, _require, _floats
from vector_os_nano.skills.grasp_config import _positive


def parse(raw, cfg, ctx):
    if cfg.arm is None or cfg.arm.gripper is None:
        raise EmbodimentConfigError(f'{ctx}: requires an arm with a gripper')
    numbers = {'hover_height', 'release_height', 'retreat_height', 'settle_duration', 'reach_m', 'position_tolerance'}
    row = _keys(raw, ctx, numbers | {'durations', 'keep_orientation', 'base', 'support'})
    result = {k: _positive(_require(row, k, ctx), f'{ctx}.{k}', zero=k == 'release_height') for k in numbers}
    if result['hover_height'] <= result['release_height']:
        raise EmbodimentConfigError(f'{ctx}: hover_height must exceed release_height')
    result['durations'] = _floats(_require(row, 'durations', ctx), 3, f'{ctx}.durations')
    if any(d <= 0 for d in result['durations']):
        raise EmbodimentConfigError(f'{ctx}.durations: all motion durations must be positive')
    keep = _require(row, 'keep_orientation', ctx)
    if not isinstance(keep, bool):
        raise EmbodimentConfigError(f'{ctx}.keep_orientation: expected a boolean')
    result['keep_orientation'] = keep
    if 'support' in row:
        keys = {'max_descent', 'step', 'duration', 'dwell', 'minimum_load', 'unload_fraction', 'max_force'}
        support = _keys(row['support'], f'{ctx}.support', keys)
        values = {k: _positive(_require(support, k, ctx), f'{ctx}.support.{k}') for k in keys}
        if (not cfg.arm.gripper.force_feedback or cfg.arm.gripper.assist or
                values['step'] > values['max_descent'] or values['unload_fraction'] >= 1 or
                values['minimum_load'] >= values['max_force'] or
                values['max_descent'] > min(result['release_height'], result['hover_height']-result['release_height'])):
            raise EmbodimentConfigError(f'{ctx}.support: requires unassisted force sensing and bounded load/descent limits')
        result['support'] = values
    if 'base' in row:
        bctx = f'{ctx}.base'
        if cfg.base is None:
            raise EmbodimentConfigError(f'{bctx}: requires a base')
        positive = {'standoff', 'yaw_speed', 'yaw_deadband', 'step_duration', 'approach_steps'}
        b = _keys(row['base'], bctx, positive | {'lateral_offset', 'carry_posture'})
        values = {k: _positive(_require(b, k, bctx), f'{bctx}.{k}') for k in positive}
        if not values['approach_steps'].is_integer():
            raise EmbodimentConfigError(f'{bctx}.approach_steps: expected a positive integer')
        values['approach_steps'] = int(values['approach_steps'])
        values['lateral_offset'] = _floats([_require(b, 'lateral_offset', bctx)], 1, f'{bctx}.lateral_offset')[0]
        posture = _require(b, 'carry_posture', bctx)
        if not isinstance(posture, str) or posture not in cfg.arm.postures:
            raise EmbodimentConfigError(f'{bctx}.carry_posture: expected a configured arm posture')
        values['carry_posture'] = posture
        result['base'] = values
    return result
