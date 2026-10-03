import numpy as np
from .go2_robot_data import PinGo2Model
from .gait import Gait
from dataclasses import dataclass

# --------------------------------------------------------------------------------
# Leg Controller Setting
# --------------------------------------------------------------------------------

KP_SWING = np.diag([400, 400, 400])
KD_SWING = np.diag([75, 75, 75])

# Mapping from leg name to index in the mask
LEG_INDEX = {
    "FL": 0,
    "FR": 1,
    "RL": 2,
    "RR": 3,
}

# Mapping from leg name to the joint torque slice in (C*dq + g)
JOINT_SLICES = {
    "FL": slice(6, 9),
    "FR": slice(9, 12),
    "RL": slice(12, 15),
    "RR": slice(15, 18),
}

@dataclass
class LegOutput:
    tau: np.ndarray       # shape (3,)
    pos_des: np.ndarray   # shape (3,)
    pos_now: np.ndarray   # shape (3,)
    vel_des: np.ndarray   # shape (3,)
    vel_now: np.ndarray   # shape (3,)


class LegController():
        
    def __init__(self):
            self.last_mask = np.array([2, 2, 2, 2])

    def compute_leg_torque(
        self,
        leg: str,
        go2: PinGo2Model,
        gait: Gait,
        contact_force: np.ndarray,
        current_time: float,
    ):
        # Extract Parameters
        leg_idx = LEG_INDEX[leg]
        joint_slice = JOINT_SLICES[leg]

        J_foot_world = go2.compute_3x3_foot_Jacobian_world(leg)      # (3x3)
        J_full_foot_world = go2.compute_full_foot_Jacobian_world(leg)  # (3x18)
        g, C, M = go2.compute_dynamcis_terms()

        current_mask = gait.compute_current_mask(current_time)
        tau_cmd = np.zeros((3, 1))

        # Initialize desired to current
        foot_pos_des, foot_vel_des = go2.get_single_foot_state_in_world(leg)
        foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

        # Detect takeoff transition
        if self.last_mask[leg_idx] != current_mask[leg_idx] and current_mask[leg_idx] == 0:
            # This leg just took off
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(go2, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)

        # Swing vs stance
        if current_mask[leg_idx] == 0:  # Swing phase
            takeoff_time = getattr(self, f"{leg}_takeoff_time")
            traj = getattr(self, f"{leg}_traj")

            time_since_takeoff = current_time - takeoff_time
            foot_pos_des, foot_vel_des, foot_acl_des = traj(time_since_takeoff)
            foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

            pos_error = foot_pos_des - foot_pos_now
            vel_error = foot_vel_des - foot_vel_now

            Lambda = np.linalg.inv(
                J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T
            )  # (3x3)
            Jdot_dq = go2.compute_Jdot_dq_world(leg)

            # Feedforward term (3x1)
            f_ff = Lambda @ (foot_acl_des - Jdot_dq)

            # PD + feedforward in Cartesian space
            force = KP_SWING @ pos_error + KD_SWING @ vel_error + f_ff  # (3x1)

            # Map to joint torques + add (C*dq + g) leg segment
            tau_cmd = J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[joint_slice]

        else:  # Stance phase
            tau_cmd = J_foot_world.T @ -contact_force

        # Update mask memory
        self.last_mask[leg_idx] = current_mask[leg_idx]

        return LegOutput(
            tau=tau_cmd.reshape(3,),
            pos_des=foot_pos_des,
            pos_now=foot_pos_now,
            vel_des=foot_vel_des,
            vel_now=foot_vel_now,
        )