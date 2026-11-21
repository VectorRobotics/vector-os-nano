# SPDX-License-Identifier: Apache-2.0
"""Move a configured head or torso group through the shared joint servo."""
from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.base import failure_cause


@skill(requires=('joint_groups',), aliases=['调整躯干', '转头', 'joint posture'])
class JointPostureSkill:
    name = 'joint_posture'
    description = 'Move a named joint group to one of its configured postures. Read robot_status for available groups and poses.'
    parameters = {
        'group': {'type': 'string', 'required': True, 'description': 'Configured joint group name.'},
        'posture': {'type': 'string', 'required': True, 'description': 'Configured posture name.'},
    }
    verify_hint = 'joint_group_at({group}, {posture})'
    preconditions = []
    postconditions = []
    effects = {}
    failure_modes = ['unknown_group', 'unknown_posture', 'cancelled', 'move_failed', 'device_failed']

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        def fail(code, message):
            return SkillResult(False, {'diagnosis': code}, message, diagnosis_code=code)
        name, pose = params.get('group'), params.get('posture')
        groups = context.services.get('joint_groups', {})
        if not isinstance(name, str) or name not in groups:
            return fail('unknown_group', f'Unknown joint group; available: {sorted(groups)}')
        postures = context.config.get('joint_groups', {}).get(name, {}).get('postures', {})
        if not isinstance(pose, str) or pose not in postures:
            return fail('unknown_posture', f'Unknown posture for {name}; available: {sorted(postures)}')
        token = context.services.get('cancel')
        if token is not None and token.is_set():
            return fail('cancelled', 'Posture cancelled; maintaining the current hold.')
        try:
            group = groups[name]
            reached = group.move_joints(list(postures[pose]))
            if token is not None and token.is_set():
                group.stop()
                return fail('cancelled', 'Posture cancelled; maintaining the current hold.')
            if not reached:
                return fail('move_failed', f'{name} did not reach {pose}; inspect clearance before retrying.')
            return SkillResult(True, {'group': name, 'posture': pose, 'joint_values': group.get_joint_positions()})
        except Exception as exc:
            return fail('device_failed', f'Joint controller failed: {failure_cause(exc)}')
