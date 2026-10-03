import casadi as ca
import numpy as np
import scipy.sparse as sp
from .com_trajectory import ComTraj
from .go2_robot_data import PinGo2Model
import time

# --------------------------------------------------------------------------------
# Model Predictive Control Setting
# --------------------------------------------------------------------------------

COST_MATRIX_Q = np.diag([1, 1, 50,  10, 20, 1,  2, 2, 1,  1, 1, 1])     # State cost weight matrix
COST_MATRIX_R = np.diag([1e-5] * 12)                                    # Input cost weight matrix

MU = 0.8    # Friction coefficient
NX = 12     # State size (6-DOF 12 states)
NU = 12     # Input size (4 x 3D force)

# Solver Option
OPTS = {
    'warm_start_primal': True,
    'warm_start_dual': True,

    "osqp": {
        "eps_abs": 1e-4,
        "eps_rel": 1e-4,
        "max_iter": 1000,
        "polish": False,
        "verbose": False,
        'adaptive_rho': True,
        "check_termination": 10,
        'adaptive_rho_interval': 25,
        "scaling": 5,
        "scaled_termination": True
    }
}

SOLVER_NAME: str = "osqp"

class CentroidalMPC:
    def __init__(self, go2:PinGo2Model, traj: ComTraj):
        self.Q = COST_MATRIX_Q 
        self.R = COST_MATRIX_R 
        self.nvars = traj.N * NX + traj.N * NU    # Total number of decision variables                          
        self.solve_time: float = 0 
        self.N = traj.N

        # Below we compute the constant elements of the MPC controller
        # This heavily reduces the time needed for update between each iteration

        # 1. Pre-compute Constant Helper Matrices (For Dynamics)
        self.I_block = ca.DM.eye(self.N * NX)
        ones_N_minus_1 = np.ones(self.N - 1)
        S_scipy = sp.kron(sp.diags([ones_N_minus_1], [-1]), sp.eye(NX))
        self.S_block = self._scipy_to_casadi(S_scipy)

        # 2. Pre-compute Constant Friction Matrix (Static Part)
        # The friction inequality constraints of the constraint matrix A is constant
        self.A_ineq_static = self._precompute_friction_matrix(traj)

        # 3. Create CasADi Function for Dynamics (Parametric Part)
        # The dynamics equality constraints of the constraint matrix A changes 
        # between iteration because of time-varying dynamics
        self.dyn_builder = self._create_dynamics_function()

        # 4. Initialize Solver
        self._build_sparse_matrix(traj, verbose=True)

    def solve_QP(self, go2:PinGo2Model, traj: ComTraj, verbose: bool = False):
        """
        # This method is called in every MPC iteration
        """
        t0 = time.perf_counter()

        # 1) Update the QP (time-varying elements including reference trajectory, contact table, dynamics)
        [g, A, lba, uba] = self._update_sparse_matrix(traj)        # update the QP
        [lbx, ubx] = self._compute_bounds(traj)                    # update box constraints
        t1 = time.perf_counter()

        # 2) Solve the QP
        qp_args = {
        'h': self.H_const,
        'g': g,
        'a': A,
        'lba': lba, 
        'uba': uba, 
        'lbx': lbx, 
        'ubx': ubx
        }
        
        # 3) Warm start if this is not the first time solving
        if hasattr(self, 'x_prev') and self.x_prev is not None:
            qp_args['x0'] = self.x_prev            # Primal guess
            qp_args['lam_x0'] = self.lam_x_prev    # Dual guess (simple bounds)
            qp_args['lam_a0'] = self.lam_a_prev    # Dual guess (linear constraints)

        # 4) Solve the QP
        sol = self.solver(**qp_args)
        t2 = time.perf_counter()

        # 5) Document time spent
        t_compute = t1 - t0         # Time spent on updating the QP
        t_solve   = t2 - t1         # Time spent on solving the QP
        self.update_time = t_compute * 1e3
        self.solve_time = t_solve * 1e3

        # 6) Save the solution for warm start next time
        self.x_prev = sol["x"]              # Warm start primal
        self.lam_x_prev = sol["lam_x"]      # Warm start dual (bounds)
        self.lam_a_prev = sol["lam_a"]      # Warm start dual (constraints)

        # 7) Print Summary
        stats = self.solver.stats()
        if verbose:
            stats = self.solver.stats()
            print(f"[QP SOLVER] update matrix takes {t_compute*1e3:.3f} ms")
            print(f"[QP SOLVER] solver takes {t_solve*1e3:.3f} ms")
            print(f"[QP SOLVER] total time = {(t_compute + t_solve)*1e3:.3f} ms  ({1.0/(t_compute + t_solve):.1f} Hz)")
            print(f"[QP SOLVER] status: {stats.get('return_status')}")
        return sol

    def _compute_bounds(self, traj: ComTraj):
        """
        This method returns box constriant bounds
        """

        fz_min = 10     # Prevent slipping
        N = traj.N      
        nvars = self.nvars
        start_u = N*12

        # 1) Start with infinities once
        lbx_np = np.full((nvars, 1), -np.inf, dtype=float)
        ubx_np = np.full((nvars, 1),  np.inf, dtype=float)

        # 2) Precompute force indices for all (leg, axis, k)
        # Layout of forces per timestep: [FLx, FLy, FLz, FRx, FRy, FRz, RLx, RLy, RLz, RRx, RRy, RRz]
        force_block = (np.arange(12)[:, None] + 12*np.arange(N)[None, :])  # (12, N)
        force_idx   = start_u + force_block                                # (12, N)

        # 3) Contact mask
        contact = np.asarray(traj.contact_table, dtype=bool)  # (4, N); True=stance, False=swing

        # Indices for each leg's (x,y,z) rows within the 12 rows
        leg_rows = np.array([[0, 1, 2],
                            [3, 4, 5],
                            [6, 7, 8],
                            [9,10,11]])

        # 4) Swing legs → all three components = 0
        swing = ~contact
        # expand swing to the three axes:
        swing_xyz = np.repeat(swing[:, None, :], 3, axis=1)

        # map to (12, N) mask
        mask_12N = np.zeros((12, N), dtype=bool)
        mask_12N[leg_rows.reshape(-1), :] = swing_xyz.reshape(12, N)

        swing_idx = force_idx[mask_12N]
        lbx_np[swing_idx, 0] = 0.0
        ubx_np[swing_idx, 0] = 0.0

        # 5) Stance legs → fz >= fz_min (only the z row of each leg: rows 2,5,8,11)
        fz_rows = np.array([2, 5, 8, 11])
        stance_idx_2d = force_idx[fz_rows[:, None], np.arange(N)[None, :]]
        stance_mask   = contact
        stance_idx    = stance_idx_2d[stance_mask]

        # keep the tighter lower bound if any
        lbx_np[stance_idx, 0] = np.maximum(lbx_np[stance_idx, 0], fz_min)

        # 6) Convert to CasADi
        lbx = ca.DM(lbx_np)
        ubx = ca.DM(ubx_np)

        return lbx, ubx
    
    def _build_sparse_matrix(self, traj: ComTraj, verbose: bool = False):
        """
        Builds the initial QP solver with the correct sparsity structure.
        """
        # 1) Build Hessian H
        rows, cols, vals = [], [], []
        for k in range(self.N):
            base = k*NX
            for i in range(NX):
                if self.Q[i,i] != 0:
                    rows.append(base+i)
                    cols.append(base+i)
                    vals.append(2*self.Q[i,i])
        
        for k in range(self.N):
            base = self.N*NX + k*NU
            for i in range(NU):
                if self.R[i,i] != 0:
                    rows.append(base+i)
                    cols.append(base+i)
                    vals.append(2*self.R[i,i])
        
        self.H_const = ca.DM.triplet(rows, cols, ca.DM(vals), self.nvars, self.nvars)
        self.H_sp = self.H_const.sparsity()

        # 2) Build Initial A Matrix
        Ad_dm = ca.DM(traj.Ad)
        Bd_stacked_np = traj.Bd.reshape(self.N * NX, NU)
        Bd_seq_dm = ca.DM(Bd_stacked_np)
        
        A_init = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)
        self.A_sp = A_init.sparsity()

        # 3) Create Solver
        qp = {'h': self.H_sp, 'a': self.A_sp}
        self.solver = ca.conic('S', SOLVER_NAME, qp, OPTS)

        # 4) Print Summary
        nH_rows, nH_cols = self.H_const.size()
        nA_rows, nA_cols = A_init.size()
        H_sp = self.H_sp
        A_sp = self.A_sp
        nH_nnz = H_sp.nnz()
        nA_nnz = A_sp.nnz()
        nH_tot = nH_rows * nH_cols
        nA_tot = nA_rows * nA_cols

        if verbose:
            print("\n[QP Init] ===== MPC QP Structure =====")
            print(f"  H: {nH_rows:4d} x {nH_cols:<4d} | nnz = {nH_nnz:6d} | dens = {nH_nnz / nH_tot:7.4f}")
            print(f"  A: {nA_rows:4d} x {nA_cols:<4d} | nnz = {nA_nnz:6d} | dens = {nA_nnz / nA_tot:7.4f}")
            print(f"  vars: {nH_cols:d} | constr: {nA_rows:d} | horizon N = {self.N}")
            print("[QP Init] ✓ Initialization complete.\n")




    def _update_sparse_matrix(self, traj: ComTraj):
        # 1) Get current model
        # Ad is constant (12, 12)
        Ad_dm = ca.DM(traj.Ad) 
        
        # Bd is Time-Varying (N, 12, 12)
        Bd_stacked_np = traj.Bd.reshape(self.N * NX, NU)
        Bd_seq_dm = ca.DM(Bd_stacked_np)

        # 2) Assemble A
        A_dm = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)

        # 3) Compute g vector (Linear Cost)
        Q_mat = ca.DM(self.Q)
        x_ref_np = traj.compute_x_ref_vec() 
        x_ref_dm = ca.DM(x_ref_np)
        gx_mat = -2 * (Q_mat @ x_ref_dm)
        g_x = ca.vec(gx_mat)
        g = ca.vertcat(g_x, ca.DM.zeros(self.N*NU, 1))

        # 4) Compute Bounds
        # Dynamics Equality Bounds
        x0 = ca.DM(traj.initial_x_vec)              # Initial condition
        gd = ca.DM(traj.gd)                         # Discrete gravity Vector
        beq_first = Ad_dm @ x0 + gd                 # Initial condition constraint
        beq_rest  = ca.repmat(gd, self.N-1, 1)
        beq = ca.vertcat(beq_first, beq_rest)

        # Friction Inequality Bounds
        n_ineq = 4 * 4 * self.N
        l_ineq = -ca.inf * ca.DM.ones(n_ineq, 1)
        
        # Set Stance legs to <= 0.0
        u_ineq_np = np.inf * np.ones(n_ineq)
        ct = traj.contact_table
        
        idx = 0
        for k in range(self.N):
            for leg in range(4):
                if ct[leg, k] == 1: 
                    # STANCE: Enforce friction pyramid <= 0
                    u_ineq_np[idx:idx+4] = 0.0
                idx += 4
        
        u_ineq = ca.DM(u_ineq_np)

        # Final Stack
        lb = ca.vertcat(beq, l_ineq)
        ub = ca.vertcat(beq, u_ineq)

        return g, A_dm, lb, ub
     
    def _assemble_A_matrix(self, Ad, Bd):
        """
        Combines Dynamics (Parametric) and Friction (Static)
        """
        # 1) Generate Block Diagonals for Dynamics
        big_minus_Ad, big_minus_Bd = self.dyn_builder(Ad, Bd)
        
        # 2) Shift Ad to lower diagonal
        term_Ad = self.S_block @ big_minus_Ad
        
        # 3) Create Dynamics LHS
        A_eq = ca.horzcat(self.I_block + term_Ad, big_minus_Bd)
        
        # 4) Stack with Static Friction
        A_total = ca.vertcat(A_eq, self.A_ineq_static)
        
        return A_total
    
    def _create_dynamics_function(self):
        Ad_sym = ca.SX.sym('Ad', NX, NX)
        Bd_seq_sym = ca.SX.sym('Bd_seq', self.N * NX, NU)
        
        list_Ad = [-Ad_sym] * self.N
        list_Bd = []

        for k in range(self.N):
            idx_start = k * NX
            idx_end   = (k + 1) * NX
            Bk = Bd_seq_sym[idx_start:idx_end, :]
            list_Bd.append(-Bk)
        
        big_Ad = ca.diagcat(*list_Ad)
        big_Bd = ca.diagcat(*list_Bd) # Diagcat puts the slices on the diagonal
        
        return ca.Function('dyn_builder', [Ad_sym, Bd_seq_sym], [big_Ad, big_Bd])


    def _precompute_friction_matrix(self, traj):
        """
        Builds the Constant Friction Cone Matrix.
        """
        rows, cols, vals = [], [], []
        baseU = self.N * NX 
        r0 = 0
        
        for k in range(self.N):
            uk0 = baseU + k * NU
            for leg in range(4):
                fx, fy, fz = 3*leg, 3*leg+1, 3*leg+2
                
                # Pyramid Face 1: fx - mu*fz
                rows.extend([r0, r0])
                cols.extend([uk0+fx, uk0+fz])
                vals.extend([1.0, -MU])
                r0+=1
                # Pyramid Face 2: -fx - mu*fz
                rows.extend([r0, r0])
                cols.extend([uk0+fx, uk0+fz])
                vals.extend([-1.0, -MU])
                r0+=1
                # Pyramid Face 3: fy - mu*fz
                rows.extend([r0, r0])
                cols.extend([uk0+fy, uk0+fz])
                vals.extend([1.0, -MU])
                r0+=1
                # Pyramid Face 4: -fy - mu*fz
                rows.extend([r0, r0])
                cols.extend([uk0+fy, uk0+fz])
                vals.extend([-1.0, -MU])
                r0+=1

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(r0, self.nvars))
        return self._scipy_to_casadi(A_sp)

    @staticmethod
    def _scipy_to_casadi(M):
        M = M.tocsc()
        return ca.DM(ca.Sparsity(M.shape[0], M.shape[1], M.indptr, M.indices), M.data)