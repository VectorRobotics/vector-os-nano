from pathlib import Path
import pinocchio as pin 
import numpy as np
from numpy import cos, sin

# --------------------------------------------------------------------------------
# Model Setting
# --------------------------------------------------------------------------------

# vector-os-nano vendoring: kinematics-only URDF shipped next to the package
# (upstream loaded meshes via RobotWrapper; the visual/collision models were never used).
URDF_PATH = Path(__file__).resolve().parent / "assets" / "go2_description.urdf"

class ConfigurationState:

    def __init__(self):

        # Initial generalized positions
        self.base_pos = np.array([0.0, 0.0, 0.27])
        self.base_quad = np.array([0.0, 0.0, 0.0, 1.0])
        self.FL_joint_angle = np.array([0.0, 0.9, -1.8])
        self.FR_joint_angle =  np.array([0.0, 0.9, -1.8])
        self.RL_joint_angle = np.array([0.0, 0.9, -1.8])
        self.RR_joint_angle = np.array([0.0, 0.9, -1.8])

        # Initial generalized velocities
        self.base_vel = np.array([0.0, 0.0, 0.0])
        self.base_ang_vel = np.array([0.0, 0.0, 0.0])
        self.FL_joint_vel = np.array([0.0, 0.0, 0.0])
        self.FR_joint_vel = np.array([0.0, 0.0, 0.0])
        self.RL_joint_vel = np.array([0.0, 0.0, 0.0])
        self.RR_joint_vel = np.array([0.0, 0.0, 0.0])

    def get_q(self):
        #Generalized position: (19x1)
        q = np.concatenate([self.base_pos, self.base_quad, 
                            self.FL_joint_angle, self.FR_joint_angle,
                            self.RL_joint_angle, self.RR_joint_angle])
        return q
    
    def get_dq(self):
        #Generalized velocity: (18x1)
        dq = np.concatenate([self.base_vel, self.base_ang_vel, 
                            self.FL_joint_vel, self.FR_joint_vel,
                            self.RL_joint_vel, self.RR_joint_vel])
        return dq
    
    def update_q(self, q):
        # base pose
        self.base_pos  = q[0:3]  # [x, y, z]
        self.base_quad = q[3:7]  # quaternion [x, y, z, w]

        # joint angles: FL, FR, RL, RR each [hip, thigh, calf]
        j = q[7:19]
        self.FL_joint_angle = j[0:3]
        self.FR_joint_angle = j[3:6]
        self.RL_joint_angle = j[6:9]
        self.RR_joint_angle = j[9:12]

    def update_dq(self, v):

        # base twist
        self.base_vel     = v[0:3]      # [vx, vy, vz]
        self.base_ang_vel = v[3:6]      # [wx, wy, wz]

        # joint velocities: FL, FR, RL, RR each [hip, thigh, calf]
        jv = v[6:18]
        self.FL_joint_vel = jv[0:3]
        self.FR_joint_vel = jv[3:6]
        self.RL_joint_vel = jv[6:9]
        self.RR_joint_vel = jv[9:12]
    
    def compute_euler_angle_world(self):
        # 1) raw roll, pitch, yaw in [-pi, pi]
        qx, qy, qz, qw = self.base_quad
        q_eig = pin.Quaternion(qw, qx, qy, qz)
        R = q_eig.toRotationMatrix()                        # returns 3x3 matrix from base -> world
        rpy = pin.rpy.matrixToRpy(R)                        # returns Euler ZYX
        roll, pitch, yaw_meas = np.array(rpy).reshape(3,)

        # 2) initialize unwrap state on first call
        if not hasattr(self, "_yaw_unwrap_initialized"):
            self._yaw_unwrap_initialized = True
            self._yaw_prev_meas = yaw_meas
            self._yaw_cont = yaw_meas
        else:
            # 3) unwrap: keep smallest change between steps
            yaw_delta = (yaw_meas - self._yaw_prev_meas + np.pi) % (2 * np.pi) - np.pi
            self._yaw_cont += yaw_delta
            self._yaw_prev_meas = yaw_meas

        # 4) return roll, pitch, continuous yaw
        return np.array([roll, pitch, self._yaw_cont])
    
    def update_with_euler_angle(self, roll, pitch, yaw):

        cr,sr = np.cos(roll/2), np.sin(roll/2)
        cp,sp = np.cos(pitch/2), np.sin(pitch/2)
        cy,sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        self.base_quad = np.array([qx, qy, qz, qw])

class PinGo2Model:

    def __init__(self):

        # Build robot (free-flyer)
        self.model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
        # Initial data containers
        self.data = self.model.createData()

        # Initial configuration
        self.current_config = ConfigurationState()
        self.q_init = self.current_config.get_q()
        self.dq_init = self.current_config.get_dq()

        # Forward kinematics / frame placements at q_init
        pin.forwardKinematics(self.model, self.data, self.q_init)
        pin.updateFramePlacements(self.model, self.data)

        self.base_id = self.model.getFrameId("base")

        self.FL_foot_id = self.model.getFrameId("FL_foot_joint")
        self.FR_foot_id = self.model.getFrameId("FR_foot_joint")
        self.RL_foot_id = self.model.getFrameId("RL_foot_joint")
        self.RR_foot_id = self.model.getFrameId("RR_foot_joint")

        self.FL_hip_id = self.model.getFrameId("FL_thigh_joint")
        self.FR_hip_id = self.model.getFrameId("FR_thigh_joint")
        self.RL_hip_id = self.model.getFrameId("RL_thigh_joint")
        self.RR_hip_id = self.model.getFrameId("RR_thigh_joint")

        oMb = self.data.oMf[self.base_id]
        oMh1 = self.data.oMf[self.FL_hip_id]
        oMh2 = self.data.oMf[self.FR_hip_id]
        oMh3 = self.data.oMf[self.RL_hip_id]
        oMh4 = self.data.oMf[self.RR_hip_id]

        bMh1 = oMb.actInv(oMh1)
        bMh2 = oMb.actInv(oMh2)
        bMh3 = oMb.actInv(oMh3)
        bMh4 = oMb.actInv(oMh4)

        self.FL_hip_offset = bMh1.translation.copy()
        self.FR_hip_offset = bMh2.translation.copy()
        self.RL_hip_offset = bMh3.translation.copy()
        self.RR_hip_offset = bMh4.translation.copy()

        self.update_model(self.q_init, self.dq_init)

        self.x_pos_des_world = []
        self.y_pos_des_world = []
        self.x_vel_des_world = []
        self.y_vel_des_world = []
        self.yaw_rate_des_world = []

    def get_hip_offset(self, leg: str):
        name = f"{leg.upper()}_hip_offset"
        return getattr(self, name)
    
    def compute_com_x_vec(self):

        # This function return the 6-DOF 12 states centroidal x-vector
        pos_com_world = self.pos_com_world
        rpy_com_world = self.current_config.compute_euler_angle_world()
        vel_com_world = self.vel_com_world
        rpy_rate_body = self.current_config.base_ang_vel
        
        R = self.R_body_to_world
        omega_world = R @ rpy_rate_body

        x_vec = np.concatenate([pos_com_world, rpy_com_world, 
                                vel_com_world, omega_world])
        
        # numpy>=2: flat (12,) — the upstream (12,1) made x_vec[5] an array and
        # broke scalar-slot assignment every tick (tricky-bugs Case 2).
        return x_vec.reshape(-1)

    def update_model(self, q, dq):
        self.current_config.update_q(q)
        self.current_config.update_dq(dq)
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data) 
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        self.oMb = self.data.oMf[self.base_id]
        self.oMf1 = self.data.oMf[self.FL_foot_id]
        self.oMf2 = self.data.oMf[self.FR_foot_id]
        self.oMf3 = self.data.oMf[self.RL_foot_id]
        self.oMf4 = self.data.oMf[self.RR_foot_id]
        self.pos_com_world = self.data.com[0]
        self.vel_com_world = self.data.vcom[0]

        yaw = self.current_config.compute_euler_angle_world()[2]
        R_bw = np.array(self.oMb.rotation)

        self.R_body_to_world = R_bw
        self.R_world_to_body = R_bw.T

        self.R_z = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw),  cos(yaw), 0],
            [0,             0,            1]
        ])

    def update_model_simplified(self, q, dq):

        roll = q[3]
        pitch = q[4]
        yaw = q[5]

        cr,sr = np.cos(roll/2), np.sin(roll/2)
        cp,sp = np.cos(pitch/2), np.sin(pitch/2)
        cy,sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        q_full = np.concatenate([
            q[0:3],                # base position
            [qx, qy, qz, qw],      # base quaternion
            np.zeros(12)           # 12 leg joint angles
        ])

        dq_full = np.concatenate([
            dq[0:6],              
            np.zeros(12)
        ])

        self.update_model(q_full, dq_full)

    def get_foot_placement_in_world(self):

        FL_placement = self.oMf1.translation.copy()
        FR_placement = self.oMf2.translation.copy()
        RL_placement = self.oMf3.translation.copy()
        RR_placement = self.oMf4.translation.copy()

        return FL_placement, FR_placement, RL_placement, RR_placement
    
    def get_foot_lever_world(self):

        pos_com_world = self.pos_com_world    
        FL_placement = self.oMf1.translation - pos_com_world
        FR_placement = self.oMf2.translation - pos_com_world
        RL_placement = self.oMf3.translation - pos_com_world
        RR_placement = self.oMf4.translation - pos_com_world

        return FL_placement, FR_placement, RL_placement, RR_placement
    
    def get_single_foot_state_in_world(self, leg: str):

        foot_id = getattr(self, f"{leg}_foot_id")

        # position in world (assumes updateFramePlacements already called)
        oMf = self.data.oMf[foot_id]
        foot_pos_world = oMf.translation.copy()  # (3,)

        # 6D spatial velocity in LOCAL_WORLD_ALIGNED (axes = world)
        v6 = pin.getFrameVelocity(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        foot_vel_world = np.array(v6.linear).copy()  # (3,)

        return foot_pos_world, foot_vel_world
    
    
    def compute_3x3_foot_Jacobian_world(self, leg: str):

        foot_id = getattr(self, f"{leg}_foot_id")  # e.g. "FL_foot_id"
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3,:]

        joint_ids  = [self.model.getJointId(f"{leg}_hip_joint"), 
                      self.model.getJointId(f"{leg}_thigh_joint"), 
                      self.model.getJointId(f"{leg}_calf_joint")]

        vcols = [self.model.joints[jid].idx_v for jid in joint_ids]

        J_leg_pos_world = J_pos_world[:, vcols] 

        return J_leg_pos_world
    

    def compute_3x3_foot_Jacobian_body(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")  # e.g. "FL_foot_id"

        # 6xnv Jacobian, expressed in WORLD (because of LOCAL_WORLD_ALIGNED)
        J_world = pin.getFrameJacobian(
            self.model, self.data, foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_pos_world = J_world[0:3, :]          # (3 x nv)

        # Base placement in world: oMb
        oMb = self.data.oMf[self.base_id]      # SE3 of base in world
        R_wb = oMb.rotation                    # R_WB

        # Rotate Jacobian into BODY (base) frame
        J_pos_body = R_wb.T @ J_pos_world      # (3 x nv)

        # Pick the 3 leg joints you care about
        joint_ids = [
            self.model.getJointId(f"{leg}_hip_joint"),
            self.model.getJointId(f"{leg}_thigh_joint"),
            self.model.getJointId(f"{leg}_calf_joint"),
        ]
        vcols = [self.model.joints[jid].idx_v for jid in joint_ids]

        J_leg_pos_body = J_pos_body[:, vcols]  # (3 x 3)

        return J_leg_pos_body
    
    def compute_Jdot_dq_world(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")

        # Make sure these were already computed in update_model:
        # pin.computeJointJacobians(self.model, self.data, q)
        # pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)

        Jdot = pin.getFrameJacobianTimeVariation(
            self.model, self.data, foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        Jdot_dq = Jdot[0:3, :] @ self.current_config.get_dq()  # or store dq as self.dq
        return np.asarray(Jdot_dq).reshape(3,)

    
    def compute_full_foot_Jacobian_world(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")  # e.g. "FL_foot_id"

        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3,:]

        return J_pos_world
    
    def compute_dynamcis_terms(self):
        g = self.data.g           # gravity torque term (18 x 1)
        C = self.data.C           # Coriolis matrix (18 x 18)
        M = self.data.M           # joint-space inertia matrix (18 x 18)

        return g, C, M

    def run_simulation(self, u_vec):

        N_input = u_vec.shape[1] # Sequence of input given
        assert N_input == self.dynamics_N, f"Expected {N_input=} to equal {self.dynamics_N=}"

        x_traj = np.zeros((12, N_input+1))
        x_init = self.compute_com_x_vec()
        x_traj[:, [0]] = x_init

        for i in range(N_input):
            u_i   = u_vec[:, i].reshape(-1, 1)
            x_traj[:, i+1] = (self.Ad @ x_traj[:, [i]] + self.Bd[i] @ u_i + self.gd).flatten()

        return x_init, x_traj





