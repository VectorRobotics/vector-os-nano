# Real-Sim Plan — high-fidelity sim with REAL physics, controlled entirely via vector-cli

Status: **NEXT (campaign #8), owner-set 2026-06-13.** This is the current
phase plan (Tier 4). The completed G1 gait/navigate/obstacle-planner work
(campaigns #5/#6/#7) is the foundation; this phase makes it run in a
high-fidelity sim with real physics, not the habitat kinematic glide.

## The owner's goal (verbatim intent, 2026-06-13)

Let G1 or Go2 run REAL VLN / navigation-stack / SysNav / or simple control
**inside a high-fidelity sim**, where the robot:
1. moves with **real local motion** (a walking gait — NOT translation/glide);
2. has **real physics + collision** (no pass-through / 穿模);
3. does **obstacle avoidance**;
4. produces **real sensor data**;
5. **autonomously explores, then goes to the target object's point** (a real
   closed VLN/exploration loop, not a "lazy" shortcut to the coordinate);
6. is controlled **ENTIRELY through vector-cli** — one CLI does all control,
   and ALL acceptance testing is done after opening that CLI (not via
   standalone python harnesses). This is the product goal.

## The architecture truth that forces a decision (CEO gate, do FIRST)

The two sims we have are fundamentally different and neither alone meets the
goal:

| | habitat-sim 0.3.3 (current "高保真" world) | MuJoCo (G1 gait, campaigns #5-#7) |
|---|---|---|
| rendering | photoreal | basic |
| locomotion | navmesh KINEMATIC (`try_step` slides along walls) | REAL rigid-body gait (policy walks) |
| collision/physics | none for the agent — **glides, passes through** | real contact dynamics |
| sensors | photoreal RGB + equirect pano + GT depth | camera render; no lidar yet |

So "G1 glides / passes through" in habitat is INHERENT to habitat's kinematic
backend, not a bug to patch. Real gait + collision + obstacle physics needs a
PHYSICS sim. The R1 of campaign #8 MUST be a PROBE + judge-panel workflow that
decides the substrate. Candidate architectures (workflow scores them):

- **A. MuJoCo-as-substrate**: make MuJoCo the high-fidelity world — add camera
  + depth/lidar sensors to MuJoCo scenes, run the gait + collision + obstacle
  avoidance + SysNav/nav-stack consumption there. Reuses campaigns #5-#7
  directly (real gait + the visibility-graph planner). Loses photoreal.
- **B. Habitat 3.0 / Bullet physics**: un-pin habitat to a physics-enabled
  build so the agent has rigid-body dynamics + collision IN the photoreal
  world. Big migration; risk the pinned-version invariants.
- **C. Isaac Sim** (ADR-005, the M4 fallback): RTX lidar + PhysX + photoreal.
  Heavy (Docker/GPU), but purpose-built for exactly this. 
- **D. Co-sim**: MuJoCo physics stepped under habitat rendering. Highest
  complexity; tightest coupling risk.

Bias for the decision: real physics/collision is non-negotiable (owner point
#2); reuse the campaign #5-#7 gait + planner where possible; everything must
reduce to vector-cli control (owner point #6); honest verify (collision means
geodesic ≠ euclidean, obstacle clearance is a real constraint — rule 5).

## Batch shape (the loop will self-author each round; this is the frame)

- **R1 PROBE (CEO gate)**: workflow judge-panel over substrates A/B/C/D →
  DQ executive summary to the owner. Build nothing irreversible until the
  owner picks. Spike the top candidate in `~/sandbox/` (real gait + collision
  + a sensor frame) as evidence.
- **then**: on the chosen substrate — wire real local-motion + collision into
  the world; expose camera/lidar sensors; run obstacle avoidance with real
  contact; the autonomous explore→find-object→go-to-object VLN loop; ALL
  driven from vector-cli (`--scenario <world>` then NL turns). Each batch
  closes with a vector-cli + screenshot acceptance (owner watches the CLI).

## Carry-over from campaigns #5-#7 (reusable, on feat/playground-vln)

- `hardware/sim/mujoco_g1.py` — G1 real policy gait (set_velocity/walk/stop/
  navigate_to/geodesic_distance), real-physics odometry, on-demand chase-cam.
- `hardware/sim/g1_vgraph.py` — pure visibility-graph obstacle planner
  (plan_path / path_length, the honest geodesic).
- `vcli/g1_runtime.py` + catalog `g1_flat` — boot a G1 agent via vector-cli.
- Campaign #7 batch 1 (planner) is DONE; batch 2/3 (navigate_to_avoiding +
  obstacle scene + GUI) fold INTO this phase's chosen substrate, since the
  obstacle scene + collision are exactly what changes.

## Hard constraints (unchanged)

vector-cli is the ONLY acceptance surface (owner). Real physics/collision is
mandatory. Verify only stricter (rule 5). DQ-4 (merge to master) still awaits
the owner. No PR/merge from the loop. New external sim deps (Isaac, habitat
re-pin) are CEO gates → decision-queue.
