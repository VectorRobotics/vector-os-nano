import numpy as np
from .go2_robot_data import PinGo2Model
from .gait import Gait
from numpy import cos, sin
from scipy.signal import cont2discrete
from scipy.linalg import expm

class ComTraj:

    def __init__(self, go2: PinGo2Model):
        self.dummy_go2 = PinGo2Model()
        x_vec = go2.compute_com_x_vec().reshape(-1)
        self.pos_des_world = x_vec[0:3]

    def compute_x_ref_vec(self):
        refs = [
            self.pos_traj_world,
            self.rpy_traj_world,
            self.vel_traj_world,
            self.omega_traj_world,
        ]
        # stack into shape (12, N)
        N = min(r.shape[1] for r in refs)
        ref_traj = np.vstack([r[:, :N] for r in refs])
        return ref_traj      

    def generate_traj(self,
                        go2: PinGo2Model,
                        gait: Gait,
                        time_now: float,
                        x_vel_des_body: float,
                        y_vel_des_body: float,
                        z_pos_des_body: float,
                        yaw_rate_des_body: float,
                        time_step: float):
        
        self.initial_x_vec= go2.compute_com_x_vec()
        initial_pos = self.initial_x_vec[0:3]
        self.m = go2.data.Ig.mass
        self.I_com_world = go2.data.Ig.inertia
        x0, y0, z0 = initial_pos
        yaw = self.initial_x_vec[5]
        self.dummy_go2.yaw_rate_des_world = yaw_rate_des_body
        time_horizon = gait.gait_period


        max_pos_error = 0.1   # define the threshold for error of position
        
        #clamp desired world COM to stay near current position
        if self.pos_des_world[0] - x0 > max_pos_error:
            self.pos_des_world[0] = x0 + max_pos_error
        if x0 - self.pos_des_world[0] > max_pos_error:
            self.pos_des_world[0] = x0 - max_pos_error

        if self.pos_des_world[1] - y0 > max_pos_error:
            self.pos_des_world[1] = y0 + max_pos_error
        if y0 - self.pos_des_world[1] > max_pos_error:
            self.pos_des_world[1] = y0 - max_pos_error

        self.pos_des_world[2] = z_pos_des_body

        go2.x_pos_des_world = self.pos_des_world[0]
        go2.y_pos_des_world = self.pos_des_world[1]
        
        # 3) Time horizon
        self.N = int(time_horizon / time_step)
        N = self.N
        t_vec = (np.arange(N) + 1) * time_step      # [dt, 2dt, ..., N*dt]
        self.time = t_vec

        # Rotation
        R_z = go2.R_z
        vel_desired_world = R_z @ np.array([x_vel_des_body, y_vel_des_body, 0.0])
        go2.x_vel_des_world = vel_desired_world[0]
        go2.y_vel_des_world = vel_desired_world[1]

        self.time = t_vec
        # 5) Allocate trajectory arrays (3 x N)
        self.pos_traj_world     = np.zeros((3, N))
        self.vel_traj_world     = np.zeros((3, N))
        self.rpy_traj_world     = np.zeros((3, N))
        self.omega_traj_world = np.zeros((3, N))

        # Position in world: p(t) = p_des + v_world * t
        # Broadcasting: (3,1) + (3,1)*(1,N) -> (3,N)
        self.pos_traj_world[:, :] = (
            self.pos_des_world.reshape(3, 1) + (vel_desired_world.reshape(3, 1) * t_vec.reshape(1, N))
        )


        # Linear velocity in world: constant over horizon
        self.vel_traj_world[:, :] = vel_desired_world.reshape(3, 1)

        # RPY in world:
        # Keep roll, pitch constant; integrate yaw with desired yaw rate
        self.rpy_traj_world[0, :] = 0.0
        self.rpy_traj_world[1, :] = 0.0
        self.rpy_traj_world[2, :] = yaw + yaw_rate_des_body * t_vec

        # RPY rates in BODY frame: only yaw rate non-zero
        self.omega_traj_world[0, :] = 0.0
        self.omega_traj_world[1, :] = 0.0
        self.omega_traj_world[2, :] = yaw_rate_des_body
        go2.yaw_rate_des_world = yaw_rate_des_body

        self.contact_table = gait.compute_contact_table(time_now, time_step, N)

        r_fl_traj_world = np.zeros((3,N))
        r_fr_traj_world = np.zeros((3,N))
        r_rl_traj_world = np.zeros((3,N))
        r_rr_traj_world = np.zeros((3,N))

        [r_fl_next_td_world, r_fr_next_td_world, r_rl_next_td_world, r_rr_next_td_world] = go2.get_foot_lever_world()

        mask_previous = np.array([2,2,2,2])
        q = np.zeros(6)
        dq = np.zeros(6)

        for i in range(N):
            current_mask = gait.compute_current_mask(time_now + i * time_step)

            q[0:3] = self.pos_traj_world[:,i]
            q[3:6] = self.rpy_traj_world[:,i]
            # pinocchio needs body frame velocity
            R = go2.R_world_to_body
            v_world = self.vel_traj_world[:,i]
            w_world = self.omega_traj_world[:,i]
            w_body = R @ w_world
            v_body = R @ v_world
            dq[0:3] = v_body
            dq[3:6] = w_body # assume rpy rate = omega here
            self.dummy_go2.update_model_simplified(q, dq)

            p_base_traj_world = self.dummy_go2.current_config.base_pos

            ## Front-left foot
            if current_mask[0] != mask_previous[0] and current_mask[0] == 0:
                # Takes off
                pos_fl_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "FL") # This returns the next touchdown position in world coordinate
                r_fl_next_td_world = pos_fl_next_td_world - p_base_traj_world

                r_fl_traj_world[:,i] = np.array([0,0,0])

            if current_mask[0] != mask_previous[0] and current_mask[0] == 1:
                # Touch down
                r_fl_traj_world[:,i] = r_fl_next_td_world # Update the touchdown position 

            if current_mask[0] == mask_previous[0]:
                # No change from last time step
                r_fl_traj_world[:,i] = r_fl_traj_world[:,i-1] # No change, reuse last value 


            ## Front-right foot
            if current_mask[1] != mask_previous[1] and current_mask[1] == 0:
                # Takes off
                pos_fr_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "FR") # This returns the next touchdown position in world coordinate
                r_fr_next_td_world = pos_fr_next_td_world - p_base_traj_world

                r_fr_traj_world[:,i] = np.array([0,0,0])

            if current_mask[1] != mask_previous[1] and current_mask[1] == 1:
                # Touch down
                r_fr_traj_world[:,i] = r_fr_next_td_world # Update the touchdown position 

            if current_mask[1] == mask_previous[1]:
                # No change from last time step
                r_fr_traj_world[:,i] = r_fr_traj_world[:,i-1] # No change, reuse last value 

            ## Rear-left foot
            if current_mask[2] != mask_previous[2] and current_mask[2] == 0:
                # Takes off
                pos_rl_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "RL") # This returns the next touchdown position in world coordinate
                r_rl_next_td_world = pos_rl_next_td_world - p_base_traj_world

                r_rl_traj_world[:,i] = np.array([0,0,0])
            elif current_mask[2] != mask_previous[2] and current_mask[2] == 1:
                # Touch down
                r_rl_traj_world[:,i] = r_rl_next_td_world # Update the touchdown position 

            elif current_mask[2] == mask_previous[2]:
                # No change from last time step
                r_rl_traj_world[:,i] = r_rl_traj_world[:,i-1] # No change, reuse last value 


            ## Rear-right foot
            if current_mask[3] != mask_previous[3] and current_mask[3] == 0:
                # Takes off
                pos_rr_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "RR") # This returns the next touchdown position in world coordinate
                r_rr_next_td_world = pos_rr_next_td_world - p_base_traj_world

                r_rr_traj_world[:,i] = np.array([0,0,0])
            elif current_mask[3] != mask_previous[3] and current_mask[3] == 1:
                # Touch down
                r_rr_traj_world[:,i] = r_rr_next_td_world # Update the touchdown position 

            elif current_mask[3] == mask_previous[3]:
                # No change from last time step
                r_rr_traj_world[:,i] = r_rr_traj_world[:,i-1] # No change, reuse last value 


            mask_previous = current_mask

        # Save
        self.r_fl_foot_world = r_fl_traj_world
        self.r_fr_foot_world = r_fr_traj_world
        self.r_rl_foot_world = r_rl_traj_world
        self.r_rr_foot_world = r_rr_traj_world

        # Update the traj dynamics
        self._continuousDynamics(go2)
        self._discreteDynamics(time_step)

    def _skew(self,vector):

        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _continuousDynamics(self, go2):
        
        m = self.m
        I_com_world = self.I_com_world

        yaw_avg = np.average(self.rpy_traj_world[2, :])
        
        R_z = np.array([
            [cos(yaw_avg), -sin(yaw_avg), 0],
            [sin(yaw_avg),  cos(yaw_avg), 0],
            [0,             0,            1]
        ])

        self.Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), R_z.T           ],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        ])

        self.Bc = np.zeros((self.N, 12, 12))
        for i in range(self.N):

            # foot lever arms in WORLD frame (from COM to foot)
            r1_world = self.r_fl_foot_world[:, i]
            r2_world = self.r_fr_foot_world[:, i]
            r3_world = self.r_rl_foot_world[:, i]
            r4_world = self.r_rr_foot_world[:, i]

            skew_r1 = self._skew(r1_world)
            skew_r2 = self._skew(r2_world)
            skew_r3 = self._skew(r3_world)
            skew_r4 = self._skew(r4_world)

            I_inv = np.linalg.inv(I_com_world)

            self.Bc[i] = np.block([
                [np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3))],
                [np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3))],
                [(1/m) * np.eye(3), (1/m) * np.eye(3), (1/m) * np.eye(3), (1/m) * np.eye(3)],
                [I_inv @ skew_r1,   I_inv @ skew_r2,   I_inv @ skew_r3,   I_inv @ skew_r4],
            ])

        # Gravity Vector
        self.gc = np.array([
            0, 0, 0, 
            0, 0, 0, 
            0, 0, -9.81, 
            0, 0, 0 
            ])

    def _discreteDynamics(self, dt: float):
        N = self.N
        m = self.m
        I_inv = np.linalg.inv(self.I_com_world)

        # Use your same yaw-avg rotation for rpy integration
        yaw_avg = float(np.average(self.rpy_traj_world[2, :]))
        cy, sy = np.cos(yaw_avg), np.sin(yaw_avg)
        RzT = np.array([[cy, sy, 0.0],
                        [-sy, cy, 0.0],
                        [0.0, 0.0, 1.0]], dtype=float)

        # ---- Ad ----
        Ad = np.eye(12, dtype=float)
        Ad[0:3, 6:9] = dt * np.eye(3)        # p <- v
        Ad[3:6, 9:12] = dt * RzT             # rpy <- omega (approx)
        self.Ad = Ad

        # ---- gd ---- (gravity only affects position/velocity)
        g = np.array([0.0, 0.0, -9.81], dtype=float)
        gd = np.zeros((12, 1), dtype=float)
        gd[0:3, 0] = 0.5 * g * dt * dt       # p += 1/2 g dt^2
        gd[6:9, 0] = g * dt                  # v += g dt
        self.gd = gd

        # ---- Bd ----
        Bd = np.zeros((N, 12, 12), dtype=float)

        Bp = (0.5 * dt * dt / m) * np.eye(3) # p from forces
        Bv = (dt / m) * np.eye(3)            # v from forces

        for i in range(N):
            r1 = self.r_fl_foot_world[:, i]
            r2 = self.r_fr_foot_world[:, i]
            r3 = self.r_rl_foot_world[:, i]
            r4 = self.r_rr_foot_world[:, i]

            W1 = I_inv @ self._skew(r1)
            W2 = I_inv @ self._skew(r2)
            W3 = I_inv @ self._skew(r3)
            W4 = I_inv @ self._skew(r4)

            Bi = Bd[i]

            # p block
            Bi[0:3, 0:3]   = Bp
            Bi[0:3, 3:6]   = Bp
            Bi[0:3, 6:9]   = Bp
            Bi[0:3, 9:12]  = Bp

            # v block
            Bi[6:9, 0:3]   = Bv
            Bi[6:9, 3:6]   = Bv
            Bi[6:9, 6:9]   = Bv
            Bi[6:9, 9:12]  = Bv

            # omega block
            Bi[9:12, 0:3]  = dt * W1
            Bi[9:12, 3:6]  = dt * W2
            Bi[9:12, 6:9]  = dt * W3
            Bi[9:12, 9:12] = dt * W4

            # rpy also gets input through omega integration (2nd order term)
            Bi[3:6, 0:3]   = 0.5 * dt * dt * (RzT @ W1)
            Bi[3:6, 3:6]   = 0.5 * dt * dt * (RzT @ W2)
            Bi[3:6, 6:9]   = 0.5 * dt * dt * (RzT @ W3)
            Bi[3:6, 9:12]  = 0.5 * dt * dt * (RzT @ W4)

        self.Bd = Bd