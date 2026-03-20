"""Test Pinocchio IK solver — no hardware needed."""
import numpy as np

print("=" * 50)
print("Test: Pinocchio FK/IK Solver")
print("=" * 50)

from vector_os.hardware.so101.ik_solver import IKSolver

solver = IKSolver()
print("IK solver loaded (URDF parsed)")

# Home joint positions
home = [-0.014, -1.238, 0.562, 0.858, 0.311]

# Test FK: where is the gripper at home?
pos, rot = solver.fk(home)
print(f"\nFK at home position:")
print(f"  Gripper XYZ: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
print(f"  = [{pos[0]*100:.1f}, {pos[1]*100:.1f}, {pos[2]*100:.1f}] cm")

# Test FK gripper tip
tip_pos, tip_rot = solver.fk_gripper_tip(home)
print(f"\nFK gripper tip:")
print(f"  Tip XYZ: [{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}] m")

# Test IK: can we reach a point in front of the robot?
target = (0.20, 0.0, 0.15)
print(f"\nIK to target {target}:")
solution, error = solver.ik_position(target, home)
if solution is not None:
    print(f"  Solution: {[round(j, 3) for j in solution]}")
    print(f"  Residual error: {error*1000:.1f} mm")
    # Verify with FK
    verify_pos, _ = solver.fk(solution)
    actual_error = np.linalg.norm(np.array(target) - verify_pos)
    print(f"  FK verify error: {actual_error*1000:.1f} mm")
else:
    print(f"  No solution (error: {error*1000:.1f} mm)")

# Test multiple targets across workspace
print(f"\nWorkspace reachability test:")
targets = [
    (0.15, 0.0, 0.10, "center near"),
    (0.25, 0.0, 0.10, "center far"),
    (0.20, 0.10, 0.10, "left"),
    (0.20, -0.10, 0.10, "right"),
    (0.20, 0.0, 0.20, "high"),
    (0.20, 0.0, 0.02, "low (table)"),
    (0.50, 0.0, 0.10, "very far (unreachable?)"),
]

for x, y, z, label in targets:
    sol, err = solver.ik_position((x, y, z), home)
    if sol is not None:
        vp, _ = solver.fk(sol)
        actual = np.linalg.norm(np.array([x, y, z]) - vp)
        status = "OK" if actual < 0.005 else f"WARN ({actual*1000:.0f}mm)"
        print(f"  {label:20s} ({x:.2f},{y:.2f},{z:.2f}) -> {status}")
    else:
        print(f"  {label:20s} ({x:.2f},{y:.2f},{z:.2f}) -> UNREACHABLE")

# Test trajectory interpolation
print(f"\nTrajectory interpolation:")
traj = solver.interpolate_trajectory(home, [0.0]*5, num_steps=10, duration_sec=2.0)
print(f"  {len(traj)} waypoints over 2.0s")
print(f"  First: t={traj[0]['time_from_start']:.2f}s joints={[round(j,2) for j in traj[0]['positions']]}")
print(f"  Last:  t={traj[-1]['time_from_start']:.2f}s joints={[round(j,2) for j in traj[-1]['positions']]}")

print()
print("=" * 50)
print("IK TEST COMPLETE")
print("=" * 50)
