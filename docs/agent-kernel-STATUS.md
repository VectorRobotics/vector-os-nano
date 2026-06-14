# Verified Agent Kernel — STATUS (resume anchor)

One-page "where are we / what's next". Read this first when resuming; durable design is
[ARCHITECTURE.md](ARCHITECTURE.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md).

- Branch: `feat/playground-vln` (off `master` @ PR #13 merge; gait-fix branch merged as PR #13).
  Campaign: third world — photoreal/semantic nav world (VLN-first, SysNav revival rides along);
  M0 DONE: ADR-009 (Proposed) recommends pinned habitat-sim 0.3.3 conda-py3.9 subprocess +
  bridge — spike-verified on the Linux box (photoreal RGB, equirect pano+depth, 819K-pt
  unprojected cloud, GT odom). owner APPROVED 2026-06-10; HM3D agreement (DQ-1) still pending — interim scenes are license-free (test-scenes/HSSD/Replica).
  M1 DONE (backend-agnostic, ungated): `Scenario` carries additive `sim_backend`/`scene_ref`;
  a non-MJCF world registers/resolves through `WorldRegistry` with the engine surface intact
  (tests/vcli/test_scenario_backend_seam.py). M2 (the habitat world itself) awaits the gate.
- Last updated: 2026-06-14 (campaign #9 R5 — depth-at-bbox reliable recognise→navigate, g1 headless 3/3).
- Scope guard: this is **vector-os-nano only** — not the UniLab go2arm-grasp work.

## Current state (2026-06-11)

- **Campaign #2 (全栈居住世界, ~/.vector-nano-loop/) — N0 SHIPPED: composed ReplicaCAD
  dataset scenes + the `house` preset.** `Scenario.scene_dataset_config` (additive field):
  when set, `scene_ref` is a dataset scene-instance NAME, not a file; the repo side
  resolves the dataset config + the AUTHORED navmesh (`scenes.py` —
  `resolve_scene_dataset_config`/`preflight_scene_instance`/`dataset_navmesh_path`;
  bare habitat_sim ignores the dataset's `navmesh_instances` mapping, spike-verified)
  and the server takes `--dataset-config`/`--navmesh` plus `--agent-radius/height`
  with a `recompute_navmesh` fallback. `--scenario house` = ReplicaCAD apt_1
  (CC-BY-4.0; ~49 m² navigable, 120 placed objects; 5 hand-labeled world-frame rooms).
  LIVE acceptance (`~/sandbox/live_test_house.py`): all room-pair geodesics finite,
  every room center snaps inside its box, cross-house navigate tv_corner→entryway
  10.6 m with geodesic remaining 0.098 (< 0.5 criterion), photoreal furnished render;
  GUI window present in the X11 tree (visual quality = owner check). Scene data stays
  OUTSIDE the repo (`~/sandbox/habitat-spike/data`; downloader:
  `python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset`).
  **N1 SHIPPED: the streaming cmd_vel boundary.** The server runs a 50 Hz
  integration thread (single pose/velocity authority; `try_step`-constrained;
  0.6 s deadman; the agent object is now a render puppet synced on the op
  thread; `try_step` concurrent with renders spike-verified). Extra socket
  connections are STREAM channels (restricted op set {ping, get_state,
  set_velocity} that never touches the sim) — `HabitatBase.set_velocity` is
  non-blocking and state reads never queue behind a pano render (live: max
  gap 29 ms under pano load). `HabitatSysnavBridge` adds a 50 Hz
  `/state_estimation` timer (with twist) + a `/cmd_vel` subscriber on a
  SEPARATE fast node with `spin_in_background()` per-node single-threaded
  executors — rclpy's MultiThreadedExecutor was MEASURED capping a 50 Hz
  timer at ~29 Hz (pano load irrelevant), STE hits 50.0. LIVE acceptance
  (`~/sandbox/live_test_stream.py`, all PASS): 50 msgs/s odom, /cmd_vel at
  10 Hz drives 1.0 m, deadman stops a stale stream, wall-slide never leaves
  the navmesh.
  **N2 SHIPPED: the REAL nav stack drives the house.** Chain (no oracle
  planning anywhere): habitat feed → terrain_analysis → localPlanner (G1
  profile — head-mounted sensor offsets 0/0, 0.5 m footprint) → pathFollower
  → /navigation_cmd_vel → streaming set_velocity. Feed additions (N2):
  /state_estimation is the SENSOR pose (base + 1.2 m eye height, CMU
  convention — may also fix the M4 "object z frame-shifted" follow-up,
  SysNav re-verify deferred to N4) + TF map→sensor per fast tick + /speed
  keep-alive (no /joy — the Go2 lesson) + /navigation_cmd_vel TwistStamped
  subscription + optional scan ceiling filter. THE find: habitat equirect
  DEPTH is cubemap-FACE z-depth, not euclidean ray distance (tricky-bugs
  Case 5) — flat floors rippled ±14 cm and the planner saw obstacles
  everywhere; `unproject_equirect_depth` now divides by the dominant ray
  axis component (face-center/face-seam pinned by tests). Launcher:
  scripts/launch_habitat_nav.sh (local-planner chain; FAR/TARE out of N2).
  LIVE (`~/sandbox/live_test_nav_house.py` ALL PASS): kitchen→dining-south
  6.05 m cross-room sensor navigation in 11 s, geodesic remaining
  0.49 < 0.5, final pose on-mesh. Goal-picking lessons live in the harness
  comments (room centers can BE furniture; pathFollower halts at the path
  END — stopDisThre 0.4 — so near-furniture goals end ~0.6 m short).
  **N3 SHIPPED: the G1 is VISIBLE — third-person chase view.**
  `scripts/build_g1_glb.py` composes the Menagerie unitree_g1 (BSD-3,
  local checkout) visual meshes at the "home" keyframe into ONE rigid
  y-up GLB (offline, sandbox venv with mujoco+trimesh; output lands under
  the habitat data root with an `.object_config.json` + LICENSE copy —
  never in git). The server loads it as a KINEMATIC rigid object glued to
  the pose authority (`_place_body`, +π/2 yaw for the +x→-z forward
  remap); egocentric renders (256 rgb, equirect pano) teleport-hide it —
  the eye sensors sit inside the head mesh and 0.3.3 has no per-sensor
  masking. Viewer gains `--viewer-mode first|chase` (chase = behind/above,
  pitched down; runtime default CHASE via `resolve_habitat_viewer`, env
  `VECTOR_HABITAT_VIEWER` overrides) and a `viewer_frame` op (base64 PNG,
  `body=False` discriminator) consumed by `HabitatBase.viewer_frame_png`.
  LIVE (`~/sandbox/live_test_body.py` ALL PASS): body occupies 1.33% of
  the chase frame (with/without diff), ego pano byte-stable with the body
  present (no self-occlusion), pano min depth 0.755 (not inside a mesh),
  60.8% frame change after navigate; X11 window check PASS in chase mode.
  Artifacts /tmp/n3_chase_*.png; visual quality = owner check.
  **N4 part A SHIPPED: the navigate skill drives the REAL nav stack + the
  SysNav z-frame finding is RESOLVED.** `NavigateToPointSkill` now selects
  transport: when `base._nav_feed` (attached by `wire_sysnav_feed`) reports
  a live pathFollower, navigation publishes `/way_point` and monitors
  EUCLIDEAN progress only (`HabitatSysnavBridge.navigate_to` — 5 s grace,
  stall/timeout honest failures; the navmesh oracle stays verify-only);
  sim-oracle `base.navigate_to` is the fallback, and
  `result_data.transport` records provenance. LIVE
  (`~/sandbox/live_test_nav_skill.py` ALL PASS): stack-down → sim_oracle;
  cross-room via nav_stack in 12 s (verify geodesic 0.43); goal inside a
  counter stalls honestly (success=False, diagnosis=stall). SysNav
  re-verify on the FIXED cloud (M4 harness, apartment_1): object nodes
  flow and heights are now physically plausible — floor z=-1.54 reference:
  sofa 0.55 m, light 2.63 m (ceiling), picture 1.49 m (wall); the old
  sofa-at-floor-level (-1.57) z-shift is gone (cubemap-depth + sensor-pose
  fixes).
  **N4 part B-1: full-stack VLN 4/5 + THREE cross-world contamination
  guards (the live run exposed a kernel bug family).** LIVE
  (`~/sandbox/live_test_vln_fullstack.py` ALL PASS): real LLM + SysNav GPU
  perception (sofa/chair/desk… in the live world model) + REAL nav stack
  simultaneously; EN coord 18 s and ZH two-waypoint chain 125 s
  verified-done with every navigation transport=nav_stack; relative motion
  ✓; 走到游泳池旁边 fails honestly. The guards (all three were live
  failures first): (1) goal_decomposer strategy-name AFFINITY — an
  unambiguous prefix ('navigate') resolves to the known strategy (M5
  finding closed); (2) TEMPLATE-instantiation guard — a matched experience
  template carrying strategies unknown in THIS world is rejected, plan
  fresh (instantiate() used to bypass validation entirely);
  (3) StrategySelector guards — the go2 keyword ladder no longer routes to
  skills this registry doesn't have (M5 'Skill not found: look' was the
  same hole) and a stats override never promotes a phantom; the engine
  fast path excludes navigate_to (cannot bind x/y or the geodesic verify).
  **N4 COMPLETE — full-stack VLN 5/5.** Final pieces: (a) semantic
  STANDOFF — label goals floor tol at 1.5 and verify with
  `at_position(x, y, 1.6)` (EUCLIDEAN: an object's centre sits OFF the
  navmesh, which distorts geodesic); (b) `backfill_target_params`
  (verify_strengthen) — coords the planner bound only into the verify are
  deterministically copied into missing navigate params (the same-coords
  contract, repaired kernel-side); (c) coordinate-goal verify CALIBRATED
  to the real controller: < 0.8 (five live runs showed the follower parks
  within ~0.4 m of the planned path END and plans end early near clutter —
  sub-half-metre was oracle fiction). FINAL RUN all-AS-EXPECTED: EN coord
  16 s, ZH two-waypoint chain 50 s, 走到sofa那里 32 s — every navigation
  transport=nav_stack — relative motion, and 走到游泳池旁边 honest
  failure. Traces ~/sandbox/n4_vln/.
  **N5 part A SHIPPED: one-click setup + QUICKSTART.**
  `scripts/setup_house_world.sh` — idempotent, SELF-CHECKING (each step
  verifies its result; a provisioned box prints all OK and exits 0,
  `--check` audits without changes, a bare box reports actionable [MISS]
  items and exits 1 — both paths tested): habitat conda env + the numpy
  1.26 re-pin + ReplicaCAD (CC-BY-4.0, license-noted) + Menagerie clone +
  G1 GLB build (isolated build venv) + repo venv check + status-only
  lines for the optional siblings (ROS2/nav stack/SysNav — PolyForm-NC
  stays get-it-yourself). `QUICKSTART.md` (root, operator-facing):
  prerequisites → setup → launch → optional nav-stack/SysNav layers →
  honest troubleshooting (numpy pin, clearance stopping, DISPLAY).
  **N5 COMPLETE — CAMPAIGN #2 (全栈居住世界) CLOSED, all milestones
  N0-N5 shipped.** Part B: (a) clean-environment drill — a FRESH conda
  env + fresh data root provisioned end-to-end by the setup script in
  1m24s real (fresh habitat env, ReplicaCAD download, G1 GLB build —
  all [OK]; the drill also caught and fixed two script bugs: conda being
  a shell function invisible to scripts, and a silent no-output path
  when the habitat env is missing); (b) pure-NL flow live ALL PASS
  (`~/sandbox/live_test_n5_nl_flow.py`): bare session →
  start_simulation with NO scenario boots the HOUSE world (NL default
  switched from apartment — the flagship is the multi-room world),
  robot_status knows it, one real-LLM navigation verified-done,
  stop_simulation clean; (c) hobbyist scan (fill #5): LeRobot (HF) is
  the de-facto hobbyist entry point but is manipulation/learning-
  centric (datasets, policies, LIBERO/Meta-World, GR00T) — NOT a
  navigation/VLN sandbox; our photoreal-house + real-nav-stack +
  deterministic-verify NL loop is complementary positioning, not
  competition. Campaign scorecard in ~/.vector-nano-loop/journal.md;
  master merge is the owner's gate (decision queue).
- **Go2 explore gait (飘/瘸腿): FIXED, owner-confirmed live.** Root cause was two-clock skew
  (physics ~0.65× real-time vs wall-tick velocity ramps in the nav bridge) — full case in
  [tricky-bugs.md](tricky-bugs.md) Case 1. Fix `d7e158b`: `_follow_path` ramps + wall-escape
  state machine integrate against sim-dt (`hardware/sim/sim_clock.py` + `MuJoCoGo2.get_sim_time()`).
  Escalation to `/clock`+`use_sim_time` (plan B) NOT needed. Env-gated diagnostics
  (`VECTOR_PHYS_LOG`/`VECTOR_MPC_LOG`/`VECTOR_CMDVEL_LOG` in `mujoco_go2.py`) kept — zero cost
  when off; remove when no longer useful.
- **Arm touchstone: hardened end-to-end** (perception, grounding, decompose target binding,
  real-time timeouts, singular/plural intent, long chains incl. foreach grab-everything,
  place/handover, honest+fast failure, grounded verify). Single-skill AND long-chain NL control
  work live on real deepseek, 中/英.
- **Suite + venv reality:** see CLAUDE.md "Build / test" (venv is `.venv` now; 4 documented
  environmental reds; sim tests open real GL windows on a desktop).

## North star (restated 2026-06-05)

**Vector OS Nano = natural language controls everything**, via a built-in agent that
decomposes NL → plans → executes → verifies → replans (a grounded CLOSED loop). VGG is that
engine; verify is the moat (deterministic, never LLM-graded). Robots are the end; the dev/
macOS path is a means. Generalize across embodiments (arm, go2, future) — never one-off patches.

## Shipped (condensed; details in `git log` + tricky-bugs.md)

- **Phases A–C.2:** kernel/world decoupling (robot-free boot); dev world acts + StrategyStats
  (`80916f4`/`f5b9eb4`, hardening `8e961f8`, e2e `bee46f7`); capability seam + cross-capability
  routing (`62fcfc1`/`2a7c942`).
- **Arm + Stages 0–2** (`aebd61e`, `cdbfada`): SO-101 NL control; window-by-default sim
  (`--headless` opt-out, mjpython auto re-exec); Blackboard + `${step.path}` binding +
  `result_data`; registry-derived decompose vocab (no split-brain).
- **Playground v1 + ADR-008 seam prelude;** **Stage 4** control-flow IR (`foreach` expand from a
  producing step via Blackboard, obs-driven replan hook) + **Go2 second embodiment**;
  **Stage 5** unified controller (every turn — chat included — is a verified trace;
  `VECTOR_LEGACY_TURN=1` one-release fallback; stage5 plan deleted as fully shipped).
- **Live-hardening I–VII** (real CLI + deepseek): retry-strategy fix, REPL log quiet, meta-input
  routing, decompose JSON robustness, mjpython locate fix, **P0 segfault** sync-exec gate
  (`118f886`), RobotWorld sim-oracle grounding (`75cbdba`), LLM-side target binding (`4a2edf7`,
  `f53bd04`), sim grasp z_offset (`c953d72`) → headless NL grasp chain works end-to-end.
- **R2 round (real GUI run findings):** R2-2 skill-declared `typical_duration_sec` timeout floor ·
  R2-3 singular/plural intent + unbound→nearest (sim-gated) · R2-5 sim motor auto-allow
  (ALL-semantics safety) · R2-6 ROS2-absent logs at DEBUG · R2-7 detect/foreach `name`+`label`
  contract, pick fail-fast on absent target, detect honest-empty, grounded place verify
  (`not holding_object()`), target-aware `holding_object(target)` + `picked_object` recorded ·
  R2-8 session compaction orphaned-tool-message 400 fix · go2 base verify grounding (`2edeb25`).
- **Phase E:** W1.1 learning-tier evidence gate (`85d59a2`) · W1.2 fail-loud world-registration
  preflight (`7afe2c0`) · W1.3 scene_graph TextLLM adapter (`c3103ad`) · W1.4 playground
  step-primitives wired live → **Wave 1 complete**; W2.4 typed `failure_class` into replan
  (`14ae43c`) · **W2.3 ObjectMemory re-query-freshest** (2026-06-10, implemented inline after
  the harness-flake warning: readers re-query the live SceneGraph TTL-gated, decay cache as
  fallback, byte-identical without a ref; 46 level57 + 49 level60/61 green).
- **2026-06-10 third-world campaign session (feat/playground-vln, 13 commits, autonomous loop):**
  G0 Linux-clone bootstrap (.venv on the verified stack, sim.sh hardened) · M0 ADR-009
  simulator selection (habitat-sim recommended, 2 hardware spikes: photoreal RGB, equirect
  pano+depth, 819K-pt unprojected cloud, GT odom) · M1 backend-aware Scenario + non-MJCF world
  registration proven · FALSE-SUCCESS class killed (#2b `verify_strengthen` named-target verify;
  #3 kernel `step_output()` per-step self-verify) · Phase E Wave 2 complete (W2.1 run-registry/
  daemon/CLI ops + W2.2 RUN_ID watchdog /proc sweep) · Stage 3 referring-expression grounding
  (`Objects (live)` in world_context + exact-name binding guidance) · W3.1 WorldBlueprint ·
  W3.3 first slice (protocol-based provider resolution, `_base` seam) · hygiene (dead examples/
  deleted, doc truth-ups). Suite 1140→1186 green.
- **This branch (gait fix, merged PR #13):** engine sync-exec gated on mjpython only — Linux REPL responsive (`fcc6b20`) ·
  go2 gated diagnostics + `get_sim_time()` (`6a39f6c`) · venv reconcile `.venv-nano`→`.venv`
  (`13a9429`) · **the gait sim-dt fix** (`d7e158b`) · docs: tricky-bugs casebook + STATUS condense
  (this commit).

## OPEN — prioritized backlog

-5. **CAMPAIGN #8 — IN PROGRESS (owner-set 2026-06-13): high-fidelity sim with
   REAL physics, controlled entirely via vector-cli.** Full goal + the
   architecture decision in **docs/realsim-plan.md** (READ IT FIRST next
   session). One-line: G1/Go2 must run real VLN/nav-stack/SysNav/control in a
   high-fidelity sim with REAL local-motion gait (not the habitat glide),
   real physics + collision (no pass-through), obstacle avoidance, real
   sensors, autonomous explore→go-to-object — ALL driven from vector-cli (the
   only acceptance surface).
   - **R0 SHIPPED (2026-06-13): the existing G1 MuJoCo work is now CLI-testable
     with a LIVE window.** `vector-cli --scenario g1_flat` opens a real MuJoCo
     viewer; NL `往前走`/`走到坐标` drives the real policy gait and the owner
     WATCHES it walk (acceptance: tmux CLI + screenshots, `往前走两米` → walk_skill
     [PASS]). `G1MuJoCoBase(gui=...)` + `boot_g1_agent(gui=...)` +
     `cli._maybe_init_g1_agent` (gui=not --headless). **Owner reported "卡":
     root-caused (NOT sim fidelity — physics is 26x real-time headless) to the
     passive viewer's render thread STARVING the background control thread to
     ~0.4x. Fix = PUMP mode: with a window open the control loop runs on the
     CALLER thread (no daemon), holding 1.0x (GUI walk 1.28m == headless). See
     tricky-bugs Case 13.** Render rate-capped to 30 FPS.
   - **R1 PROBE done → DQ-10 PENDING owner.** Substrate judge-panel (workflow
     wdimxq6pn, 5 sonnet agents) ranks **A. MuJoCo-as-world 22 ≫ D co-sim 10 /
     B habitat-Bullet 9 / C Isaac 9**. Recommend A: real gait+collision+vcli
     satisfied in-process, ZERO new deps, reuses all #5-#7; sensors ALREADY in
     repo (sensors/lidar360.py MuJoCoLivox360, sensors/pano360.py MuJoCoPano360
     — just unwired). Spike (~/sandbox/g1-substrate-spike/) proves real
     collision (G1 blocked at box, 345N, no pass-through) + RGB/depth frames,
     zero deps. The ONE risk: MuJoCo non-photoreal → VLN object-recog domain
     gap (req #5) — mitigate with GT-label scaffolding, upgrade perception (or
     co-sim) later. AWAITING OWNER (DQ-10); build nothing on the substrate
     until picked.
   - **R3 SHIPPED (substrate-AGNOSTIC big build, not gated by DQ-10 — physics/
     collision/lidar are MuJoCo under both top candidates A & D):** the
     `g1_room` scenario — a closed MJCF room (walls + 3 obstacle boxes + 3
     labeled targets, REAL collision) built via MjSpec from the flat scene
     (hardware/sim/g1_room.py). `G1MuJoCoBase(room=True)`: enumerates routing
     polygons from the compiled model (g1_room.obstacles_from_model — keeps
     g1_vgraph pure), wires the in-repo MuJoCoLivox360 lidar stepped on the
     control thread (Case 12/13, get_lidar_scan snapshot), and `navigate_to`
     now ROUTES AROUND obstacles via g1_vgraph waypoints (geodesic =
     visibility-graph path length, the same planner execution walks — rule 5).
     `vector-cli --scenario g1_room` boots it. Headless check: 7 obstacles
     enumerated, lidar 5760 pts, navigate to target_red reached routing around
     the centre obstacle (geodesic 4.52m vs straight 3.70m). Flat g1_flat
     behaviour unchanged (no obstacles → euclidean geodesic, direct drive).
     **Deferred to post-DQ-10: photoreal-RGB VLN object recognition (req #5).**
   - **R4 REVIEW (no code):** AUDIT confirmed all 3 commits real, modules +
     scenarios present, suite green, journal truthful, no spinning, docs
     bounded (7 canonical + 3 root). Req status: #1 gait ✓, #2 collision ✓,
     #3 avoidance ✓, #4 lidar ✓ (RGB/depth deferred), #6 vcli ✓; #5 explore→
     go-to-target is NEXT. **R5 = substrate-agnostic req-#5 loop**: lidar
     occupancy grid → frontier autonomous explore → GoToTargetSkill (NL label
     → g1_room.target_position → obstacle-aware navigate). Photoreal RGB
     recognition stays gated by DQ-10; explore must be real lidar-map driven
     (no hardcoded waypoints).
   - **R5 SHIPPED (go-to-labeled-target + occupancy foundation):** (1)
     hardware/sim/occupancy.py — a pure-numpy ray-march OccupancyGrid (lidar
     hits → OCCUPIED, ray cells → FREE, sticky obstacles) with coverage() +
     frontiers() + nearest_frontier_world() (the mapping foundation R6's
     explorer consumes). (2) NavigateToPointSkill now resolves a `label`
     against `base.list_targets()` (g1_room GT targets) when the world model
     has no semantic match — '去蓝色目标'/'go to blue' drives to the labeled
     target with obstacle-aware nav, WITHOUT photoreal recognition (the
     DQ-10-gated half stays deferred); zh/en color aliases. Semantic-perception
     path unchanged when present. The 'go to the target object's point' half of
     req #5.
   - **R6 SHIPPED (autonomous frontier exploration — req #5 'explore' half):**
     occupancy wired into G1 (observe() integrates the lidar on the CALLER
     thread — snapshot reads, off the gait hot path); ExploreSkill loops
     nearest-frontier (min-dist filtered) → obstacle-aware navigate → until a
     coverage target or no frontier (real lidar-map driven, NO hardcoded
     waypoints). Fixed the lidar self-hit (geom_group mask → rays see the
     ENVIRONMENT not the G1 body) + added include_misses (free-ray to max range
     so open space is mapped, not just where a wall was struck). Headless:
     coverage 0.61→0.94 over 3 autonomously-chosen frontiers. GUI: vector-cli
     --scenario g1_room, 探索房间 → explore ran 19.9s [PASS], G1 autonomously
     roamed. Fixed a latent pump-mode negative-sleep bug (tricky-bugs Case 14).
     req #5 STRUCTURE complete (explore + go-to-target); photoreal RGB
     object-recognition still awaits DQ-10.
   - **R7 — req #5 FULL CLOSED LOOP verified end-to-end (ZERO new code):** the
     VGG planner composes the existing skills into a real VLN flow — '探索房间并
     走到红色目标' decomposes to [explore_room → navigate_to_red_target] and ran
     2/2 [PASS] via vector-cli (explore verified by the honest
     coverage > start_coverage predicate; robot ended at the red target). The
     owner's req #5 ('autonomously explore, then go to the target's point') is
     demonstrated as ONE CLI command, all real-physics/sensor-driven. The only
     remaining piece is photoreal RGB object RECOGNITION (find an UNKNOWN-label
     target by vision) — gated by DQ-10.
   - **R8 REVIEW → campaign #8 at a NATURAL PAUSE POINT (loop paused).** AUDIT
     clean (all commits real, modules/scenario/skill registered, suite green,
     no spinning, docs bounded). The owner's 6 requirements are ALL
     structurally GREEN (gait/collision/avoidance/lidar+occupancy/explore→go-to
     /vcli-only), with req #5 demonstrated as ONE CLI command (R7). The only
     substantive remaining work — photoreal RGB object RECOGNITION — is
     hard-gated by **DQ-10** (MuJoCo-as-world keeps basic rendering; co-sim/
     photoreal enables real VLM recognition). Continuing would be low-value
     hardening or faking gated work, so the loop PAUSED with a phase-completion
     summary to the owner. **Resume by deciding DQ-10 (unblocks recognition) or
     /loop again for hardening (explore robustness / multi-room / nav-stack
     costmap).** DQ-4 (merge to master) MERGED 2026-06-14.
   - **DQ-10 APPROVED (A = MuJoCo-as-world) by owner 2026-06-13** → visual
     recognition unblocked, loop resumed.
   - **R9 SHIPPED (visual object recognition — first piece):** a pelvis-mounted
     first-person forward camera wired into G1 (g1_room adds a fixed HEAD_CAM;
     rendered on the control thread via the on-demand mechanism — Case 12, off
     the gait hot path; get_camera_frame() returns RGB). perception/
     color_targets.py: detect_targets(rgb) finds red/blue/green boxes by colour
     segmentation (pure numpy, deterministic, no new deps — robust on MuJoCo's
     basic render for saturated colours). Two real-frame gotchas fixed: env
     geoms are in geom-group ENV_GEOM_GROUP (lidar mask) which the offscreen
     Renderer hid → enable all groups in the camera MjvOption; MjSpec cameras
     take a quat (xyaxes silently no-ops). Verified headless: G1 sees the red
     object centre-frame and detect_targets returns it (screenshot
     g1_R9_firstperson_detect.png). Suite 1599 passed. NEXT R10: full vision
     closed loop ('找到并走到红色物体' → explore+recognise → estimate seen
     target's world pose → go to it; optional VLM). R12 forced REVIEW.
   - **R10 SHIPPED — req #5 vision closed loop COMPLETE:** VisionSeekSkill
     (skills/vision_seek.py) finds & approaches a colour target by RECOGNITION
     (camera), not GT coords: loop get_camera_frame → detect_targets → decide
     (_seek_action: not-seen→scan, off-centre→turn, centred→forward) →
     progress-stall arrival (the small low box clips up close so pixel area is
     unreliable; the honest signal is the robot stops making net progress at
     the recognised target). Honest (rule 5): never seen → fails, never GT
     fallback; the at_position(target,1.6) verify is the ground-truth judge.
     Fixed a pump-mode camera DEADLOCK (get_camera_frame renders directly when
     the caller thread IS the control thread). Obstacles recoloured grey + one
     moved off-axis so recognition is unambiguous and a target has clear LOS.
     GUI: vector-cli --scenario g1_room, '找到并走到红色物体' → vision_seek
     [PASS] at_position(3.7,0,1.6) in 9.5s (screenshot g1_R10_vision_seek.png);
     headless servoed to 0.62m of target_red. **req #5 is now FULLY closed on
     real physics + real sensors + real recognition, all from vector-cli.**
     Suite 1605 passed.
   - **R11 SHIPPED — grand VLN capstone:** ExploreAndSeekSkill
     (skills/explore_seek.py) composes ExploreSkill + VisionSeekSkill
     deterministically so '探索房间找到红色物体并走过去' runs as ONE step (the
     LLM's 3-way split dropped the colour param). GUI: that command →
     explore_and_seek [PASS] at_position(3.7,0,1.6) in 47.6s (screenshot
     g1_R11_grand_vln.png) — the G1 explored the room, recognised the red
     object by camera, and walked to it. **The owner's entire campaign #8 goal
     is achieved: real gait + collision + obstacle avoidance + lidar/camera
     sensors + autonomous explore + recognition-based go-to-object, the full
     VLN loop as one vector-cli command, on real physics.** Suite 1608 passed.
     NEXT R12: forced REVIEW + campaign-completion report; only DQ-4 (merge to
     master) remains, owner's call.
   - **R12 REVIEW → CAMPAIGN #8 COMPLETE, loop PAUSED.** AUDIT clean (all 12
     rounds' commits real, modules/skills registered, suite 1608 green, no
     spinning, docs bounded). Every owner requirement is GREEN and GUI-verified
     via vector-cli on real MuJoCo physics: ① real gait ② real collision (no
     pass-through) ③ obstacle avoidance ④ real sensors (lidar+occupancy+camera)
     ⑤ autonomous explore → recognition-based go-to-object ⑥ vector-cli-only —
     plus the grand one-command VLN (explore+recognise+approach, R11). Nothing
     substantive remains that isn't owner-gated, so the loop PAUSED with a
     completion report. **DQ-4 MERGED (owner-approved 2026-06-14): feat/playground-vln (88
     commits, campaigns #2-#8) --no-ff merged → master (origin/master
     3e82996). Ongoing dev continues on feat/playground-vln.** Resume with /loop for optional hardening (VLM
     recognition / multi-target / multi-room / nav-stack costmap).

   - **CAMPAIGN #9 R1 SHIPPED — track A: real VLM semantic recognition + furnished
     scene.** owner direction (2026-06-14): tracks 1+3+4 (A real Qwen-VL semantic
     recognition / B multi-room+nav-stack / C Go2 parity); NOT sim-to-real. R1 = A.
     New `g1_room_vlm` scenario: the room's 3 colour boxes are swapped for REAL
     Kenney furniture meshes (chair / sofa / potted plant) so a real VLM can ground
     an OBJECT CLASS, not a colour. `g1_room.build_room_model(furnished=True)`
     places the meshes via MjSpec (Y-up→Z-up quat; a throwaway-compile measure pass
     centres each mesh on its planned (x,y) and rests it on the floor — meshes have
     off-origin pivots); walls + obstacles + lidar + camera unchanged; the colour
     room is byte-for-byte the #8 scene. `perception/vlm_targets.VlmTargetDetector`
     wraps Qwen-VL (openrouter, via Go2VLMPerception._call_vlm; injectable for
     tests) with a grounding prompt → adapts the bbox to the SAME
     `{label,x_norm,y_norm,area_frac}` contract color_targets emits. `skills/
     vlm_seek.VlmSeekSkill` reuses the shared `vision_seek._seek_loop` (extracted)
     with the VLM detector. Honest (rule 5): the VLM is REALLY called; no GT
     fallback; at_position(obj,1.6) is the deterministic judge. Three control fixes
     the live VLM exposed (all in tricky-bugs): the ~2 s VLM latency vs the 0.4 s
     walk deadman made the gait stutter and the progress-stall fire metres short
     (→ per-action deadman: forward 3.0 s, turn/scan 0.5 s); a long turn at the
     forward duration over-rotated and flung the target off-screen (→ short turn
     step); intermittent detection + a noisy/oversized VLM bbox area caused both
     a false area-arrival (→ VLM arrive_area raised to 0.55, lean on collision-
     blocked progress-stall) and lost targets (→ last-bearing COAST across missed
     frames before search-scanning). `vlm_go2._parse_json_response` hardened with a
     balanced-brace extractor (unclosed ```json fences / trailing Qwen garbage).
     **GUI ACCEPTANCE PASSED:** vector-cli --scenario g1_room_vlm, natural-language
     「去找椅子，用相机识别走过去」 → routed to vlm_seek_skill → real Qwen-VL recognised
     the chair → walked to it → `VGG [PASS] find_and_navigate_to_chair | via
     vlm_seek | verify at_position(3.6,0.0,1.6)` (headless repro: arrived 0.50 m
     from the chair; first-person frames /tmp/c9r1_*.png show the recognised chair).
     Suite 1627 passed (3 known deepseek .env reds tolerated). NEXT: R2 — richer
     multi-object VLM scenes, or track C (Go2 into the room), or track B (multi-room
     + nav-stack). Every track independently shippable.

   - **CAMPAIGN #9 R2 SHIPPED — track C foundation: world-agnostic furnished room +
     Go2 VLM recognition (Go2 autonomous ARRIVAL deferred to R3).** The furnished-
     room builder is now embodiment-agnostic: `g1_room._add_box_statics` /
     `_add_furniture` / `_add_pelvis_head_cam` extracted, and
     `build_furnished_room_model(base_scene, recog_cam_body=...)` furnishes ANY
     flat scene with the SAME collidable chair/sofa/plant targets (g1 rooms stay
     byte-identical; one builder, two embodiments — rule #2/#7 in code).
     `MuJoCoGo2(furnished=True)` builds from go2 `scene_flat` via that builder,
     spawns at origin facing the targets, exposes `list_targets`, and gets a
     forward wide recognition camera (`RECOG_CAM` on base_link — Go2's stock d435
     is too low/narrow/down to frame furniture) that `get_camera_frame` uses in
     furnished mode with ALL geom groups enabled (group-3 furniture was hidden —
     the R9 bug again). `go2_room_vlm` scenario + `go2_runtime.boot_go2_agent`
     (light boot: sim + WalkSkill/TurnSkill/StopSkill + the SAME VlmSeekSkill, no
     ROS/nav stack) + cli `_maybe_init_go2_agent` dispatch. `vlm_seek` reads a
     per-embodiment `seek_step_duration` (g1 3.0 s async deadman; go2 1.2 s
     blocking walk). **VERIFIED:** Go2 RECOGNISES the furniture via the same
     VlmSeekSkill (real Qwen-VL, seen=True headless); vector-cli --scenario
     go2_room_vlm boots the furnished room (screenshot /tmp/c9r2_go2_gui_boot.png).
     Suite 1631 passed.
     **R2 follow-up (workflow-designed closed-loop heading control):** a design
     workflow (6 agents) chose an embodiment-scoped closed-loop heading P-controller
     for Go2: latch the target's ABSOLUTE world heading at detection
     (theta_goal = get_heading() − 0.80·x_norm), then each tick steer
     vyaw = clip(1.8·err, ±0.8) off the cheap get_heading() between slow VLM calls;
     turn if |err|>0.35 else walk-forward-with-trim; forward bursts capped 0.6 s.
     Gated behind a Go2-only `seek_heading_hold` attr (g1 ALSO has get_heading, so
     the gate is the ATTR — Case 19); progress-stall sampling is also embodiment-
     scoped (g1 all-tick orbit-arrival; Go2 forward-tick-only). g1 path byte-
     identical (unit-tested). This FIXED Go2's backward-wander (now approaches
     forward, ~1.3–2.3 m) but **did NOT achieve reliable arrival**.
     **KEY FINDING (tricky Case 21):** VLM visual-servoing arrival is INHERENTLY
     FLAKY for BOTH embodiments — slow (~2 s) + noisy + intermittent VLM detection
     × gait drift means run-to-run the robot sometimes arrives (g1 0.58 m) and
     sometimes stalls 2+ m short (same code, no change). R1's clean PASS was a real
     but non-robust sample. **→ R3 PIVOT: arrival should be VLM-RECOGNISE → drive to
     the recognised location via a reliable controller (navigate_to / track-B
     nav-stack waypoint), NOT pure visual servoing.** The recognition + the world-
     agnostic room are solid; the SERVOING is the wrong tool for the last metres.
     NEXT R3: recognise→navigate arrival (combines B+C, reliable) / track A multi-
     object / track B multi-room nav-stack.

   - **CAMPAIGN #9 R3 (PARTIAL) — recognise→navigate pipeline built + tested; honest
     position estimation needs depth-at-bbox (next round).** Built the Case-21 pivot
     scaffolding: `perception/target_locate.locate_from_bearing` (pure, 6 tests —
     recognition bearing + lidar range → world xy) + `skills/recognize_navigate.
     RecognizeNavigateSkill` (4 tests — VLM recognise → locate → reliable
     navigate_to, NOT visual servoing) + an 8 m furnished-room lidar (vs the
     colour room's 3 m explore range). **NOT wired into the CLI** — verified
     headless that it FAILS in the obstacle-furnished room: the lidar
     "nearest-hit-in-bearing" locates an intervening OBSTACLE, not the chair
     (lidar has range, not semantics — tricky Case 22); and far-target VLM
     acquisition is ~50%/frame. When the chair is recognised AND unobstructed the
     full chain works (one run located the chair 3.39 m and navigated to 0.62 m).
     **The honest fix = DEPTH-AT-BBOX** (depth at the recognised pixels → the
     chair's distance, skipping obstacles → navigate_to). Needs ONE embodiment
     with BOTH depth-at-bbox AND navigate_to: g1 has navigate_to but no depth cam;
     go2 has get_depth_frame but no navigate_to. NEXT (R4 REVIEW, then R5): add a
     g1 HEAD_CAM depth render (or a go2 navigate_to) → land recognise→navigate
     reliably on one robot, then the other. Suite green; g1/go2 seek unchanged.

   - **CAMPAIGN #9 R5 SHIPPED — depth-at-bbox → RELIABLE recognise→navigate on g1
     (headless 3/3).** g1 HEAD_CAM now renders DEPTH (`get_camera_observation` →
     atomic {rgb, depth, cam_pos, cam_mat, fovy}, one control-thread render — Case
     12; second depth Renderer). `target_locate.locate_from_depth` back-projects
     the NEAREST surface in the recognised bbox (20th-pctile — a thin chair lets a
     median see the wall behind) → the object's world (x,y), SEMANTIC (skips
     intervening obstacles, fixing Case 22). `RecognizeNavigateSkill` prefers
     depth-at-bbox (lidar fallback), confirms with TWO agreeing estimates (a lone
     bad VLM bbox won't repeat), and navigates to a 0.7 m STANDOFF in front of the
     surface (the surface can be in the planner's inflated wall zone → no path).
     REGISTERED in the g1 furnished CLI. **Headless 3/3 reliable arrival**
     (0.45/0.78/0.87 m, all by=depth — the Case-21 flakiness is CLOSED on g1).
     Suite 1648 green. **GUI caveat (R6):** the planner now routes to the single
     recognize_navigate skill (a sharpened description killed a flaky 2-step
     vlm_seek prefix), but one GUI run's VGG verify showed UNBOUND
     `at_position(x, y, 1.6)` (planner left x,y literal instead of binding the
     chair coords) → verify fail despite the deterministically-validated skill.
     R6: pin the verify-coord binding for recognize_navigate (vlm_seek bound it
     in R1 — LLM variance), confirm GUI arrival, then go2 navigate_to parity.
   - The owner saw earlier: G1 in habitat still glides/passes through (habitat
     is navmesh-KINEMATIC by design — real physics needs a different
     substrate). R1 = a PROBE + judge-panel workflow to pick the substrate
     (MuJoCo-as-world / habitat-3-Bullet / Isaac / co-sim) → DQ to the owner
     BEFORE building. Campaigns #5-#7 (real MuJoCo gait + the visibility-graph
     planner) are the reusable foundation.

-4. **CAMPAIGN #7 — batch 1 SHIPPED, batches 2-3 FOLD INTO campaign #8
   (2026-06-13).** Visibility-graph obstacle planner (hardware/sim/
   g1_vgraph.py, pure geometry, 12 offline tests; commit 1186255) + a
   legible demo (scripts/demo_g1_obstacle_plan.py). Batch 2/3
   (navigate_to_avoiding + obstacle scene + GUI) were superseded by the
   campaign #8 substrate decision — obstacle collision IS what changes, so
   they re-form on the chosen physics substrate.

-3. **CAMPAIGN #6 COMPLETE (2026-06-13, owner directive "做 G1 navigate";
   R1-R3, 3 green pushed commits, suite 1548→1558)**: G1 closed-loop
   waypoint navigation.
   - Batch 1 (R1): G1MuJoCoBase.navigate_to(x,y,tol,speed) — caller-thread
     face→walk→arrive controller driving the real policy; three-value
     contract from real get_position deltas. Gains MEASURED (a sandbox
     spike: vyaw 0.6→~0.29 rad/s, vx=0 turn stable). All 24 adversarial-
     workflow gait traps handled: tol floor (effective_tol), heading
     dead-band, capture mode near goal (no pivot-orbit), settle phase
     (authoritative arrival pose after the biped coasts), fall detection,
     mode-gated stall, NaN guard. moved_m (path) + net_m (displacement).
   - Batch 2 (R2): NavigateToPointSkill wired into boot_g1_agent
     (base-generic, zero skill change). Closed the flat-scene verify break
     the workflow scout flagged: geodesic_dist fail-safes to inf without a
     navmesh oracle → coordinate goals could never verify. Honest fix
     (not a loosening): G1MuJoCoBase.geodesic_distance returns euclidean —
     geodesic IS the straight line on an obstacle-free scene; the base
     reports its world's true geodesic (rule 5 intact).
   - Batch 3 (R3): on-demand chase-cam frames during a navigate +
     scripts/g1_gait_smoke.py navigate phase. GUI-verified: G1 walks a
     2.5m diagonal to the goal, distance monotonically 2.88→…→reached
     (remaining 0.23 < tol 0.3, net 2.55m, clean trajectory moved≈net no
     orbit), frames show stepping. Smoke 3/3 OK.
   - 11 navigate tests (contract/real-drive/turn-then-walk/tol-floor/NaN/
     wiring/verify-binds/viewer-approach). AWAITING OWNER: DQ-4 (merge to
     master — campaigns #2-#6 all on feat/playground-vln, still waiting).

-2. **CAMPAIGN #5 COMPLETE (2026-06-13, owner directive "G1 用网上现成
   步态"; R1-R6, 4 green pushed commits, suite 1530→1548)**: real G1
   humanoid gait via the unitree_rl_gym pretrained policy (BSD-3; DQ-9
   APPROVED — zero new pip deps, policy 144 KB + MJCF fetched by
   scripts/setup_g1_gait.sh into gitignored assets/g1_gait/, never
   vendored).
   - Batch 1 PROBE (R1): sandbox spike PASS; obs recipe 47 reverse-
     engineered; DQ-9 approved.
   - Batch 2 BUILD (R2-R3): G1MuJoCoBase (hardware/sim/mujoco_g1.py) —
     50 Hz policy control thread, BaseProtocol set_velocity (0.6 s
     deadman)/walk/stop/real-physics odometry, supports_holonomic=True
     (real lateral stepping; the habitat base refuses vy — two worlds,
     two truths). An adversarial workflow (19 agents) found+fixed 2
     CRITICALs in this code BEFORE they bit: cross-thread torn MjData
     reads (snapshot hand-off; tricky-bugs Case 12) and a navigate
     AttributeError on a walk-only base. Pacing batched to wall time.
   - Batch 3 wiring + GUI (R5-R6): catalog g1_flat scenario, "g1" in
     PlaygroundWorld._BASE_EMBODIMENTS, vcli/g1_runtime.boot_g1_agent
     (base-only registry), cli dispatch. WalkSkill needs ZERO change —
     drives the base + measures real moved_m. On-demand chase-camera
     render on the control thread (Case-12-safe); GUI-verified: chase
     frames during a walk show G1 STEPPING and advancing 1.1 m (not the
     rigid glide the owner flagged). scripts/g1_gait_smoke.py: 2/2 OK
     (forward 1.10 m, turn-walk 0.53 m, upright, valid frames).
   - 13 G1 tests (real-physics displacement/lateral/stop/deadman/pacing/
     odometry/torn-read/viewer), skipif-gated on assets. AWAITING OWNER:
     DQ-4 (merge to master — still waiting). Open polish: exact-1x pacing
     (desktop ~0.63x under governor); G1 navigate (flat scene, no navmesh).

-1. **CAMPAIGN #3 (重构 — 杜绝 false-PASS 家族) — batches 1/2/2.5/3 SHIPPED,
   batch 4 HANDED OVER (owner paused the loop 2026-06-12 for cleanup +
   direction review).** Spec: docs/design-review-2026-06-12-plan.md (26
   adversarially-confirmed findings). Shipped over rounds R1-R11 (11 green
   pushed commits, suite 1325→1458):
   - **Batch 1 — kernel invariants**: (I) pre-execution verify baseline
     (`StepRecord.pre_satisfied`, "(already satisfied pre-exec)" honesty);
     (II) evidence gate per-step exemptions, world-level `is_robot` bypass
     DELETED (PlaygroundWorld exempts nothing); (III) verify single source
     (`_VERIFY_MAP` deleted, skill `verify_template` is the only origin,
     `unverified: True` tags instead of silent 'True').
   - **Batch 2 — seed root-fix**: rooms are REGIONS (rect-aware tol +
     `visited()` w/ object-proximity fallback); three-value navigation
     contract `{reached, already_there, moved_m, elapsed_s}` at all three
     layers (lease-first, strict tol, geodesic arrival check kills
     through-wall PASS); walk/turn MEASURE motion (`moved_m`/`turned_rad`,
     moved_short fail-loud); loud unresolved-foreach (#10); navigate label
     backfill from `visited('<label>')`.
   - **Batch 2.5 — DQ-6**: Qwen VL 72B vision backbone via OpenRouter
     (qwen2.5-vl-72b-instruct; VECTOR_VLM_MODEL_OPENROUTER overrides);
     habitat agent wired for visual verify (get_camera_frame + _vlm).
   - **Batch 3 — contract single source**: deg→rad converted at the selector
     seam; unknown primitive params fail loud; enum/default reach the LLM
     schema; explicit `is_motor`/`confirm_exempt` (E-stop never gated);
     `supports_holonomic` checked BEFORE commanding; direction enum
     validation + zh aliases.
   - **GUI acceptance mode (owner directive)**: real vector-cli in tmux +
     window screenshots, now the standard. Verified on screen: full-height
     G1, kitchen drive, paced walk, SysNav green labels (16 objects),
     real-LLM turns binding REAL predicates (visited/step_output).
   - **CAMPAIGN #4 COMPLETE (2026-06-12, R1-R12, 8 green pushed commits,
     suite 1458→1530)**: ① robustness batch DONE (#11 dep-skip unified
     semantics, #15 rid pairing + timeout tiers, #16 ticketed navigation,
     #17 post-hoc timeout honesty + completed-step injection, #18
     target-aware replan inheritance, #20 serve-loop hardening; + Case 10
     cross-thread render crash found/fixed via GUI) ② LLM param
     robustness DONE (null-strip at parse seam Case 11, param_check
     re-ask pass, retry re-routing keeps bindings, first-error survives
     retries, timeout carries exec error) ③ E2E GUI smoke DONE
     (scripts/e2e_gui_smoke.py — FINAL RUN 4/4 ok exit 0 on house:
     kitchen verified-PASS, already_there, walk-20m PASS, lateral honest
     refusal) ④ N6 gait PROBE DONE → **DQ-8 PENDING owner**: repo already
     ships a real quadruped gait (mujoco_go2.py trot+MPC); decision = B1
     Go2 reuse (~1-2 rounds, recommended) vs B2 G1 humanoid policy
     (weeks). AWAITING OWNER: DQ-8 (gait route), DQ-4 (merge — owner said
     wait). Env watch: 2× transient X/GL suite crashes (level71 family).
   - **BATCH 1 (robustness) COMPLETE + GUI-verified (campaign #4 R5)**:
     #18 — replan param inheritance is target-aware: match by (strategy,
     sub_goal name); strategy-level fallback only when the prior tree bound
     that strategy ONCE; ambiguity stays empty → bad_params feeds the
     replan (the old "latest wins" sent BOTH navigates to sofa). GUI
     acceptance on the live apartment scene: ticketed far navigate PASS
     7.2 s with REAL motion across polls (HUD (-1.0,-0.2)→(-4.8,0.8),
     screenshots /tmp/r5_fix_mid*.png), repeat → already_there 0.0 s
     "(already satisfied pre-exec)", blocked walk → honest moved_short
     FAIL. R5 also caught + fixed a REAL #16 bug live (tricky-bugs Case
     10): the nav worker rendered off the op thread and the habitat
     process died quietly — worker now drives with allow_render=False,
     navigate_status animates on the op thread, and the bridge dead gate
     reports its true cause. **Batch 2 underway (R6)**: the R5 kitchen
     failure is ROOT-CAUSED and fixed — deepseek-chat emits every schema
     key with null/"" values; the poisoned keys defeated `setdefault` and
     `"x" in params`, so the visited()/coords backfill "fired" yet
     returned useless params (tricky-bugs Case 11). Fix: null/"" stripped
     at the decomposer parse seam (a null value IS a missing param — the
     whole pipeline sees honest missing-ness) + `backfill_target_params`
     treats null/"" as missing. **R7: the param-completeness pass is
     in** — `cognitive/param_check.py` compares each step's params
     against the skill's declared schema (required + enum, registry
     single source); the decomposer re-asks the LLM exactly once with
     the per-step missing/illegal lists + legal sets (corrections can
     only rebind the named steps' params); still-broken → bad_params
     gate. Remaining batch-2: GUI re-test (kitchen + 走20米) as the
     batch closer. Note: a wrong VALUE inside a legal range ("走20米"
     → distance=1.0) is only caught if the LLM omits/illegal-binds —
     range/intent checking stays out of scope (no LLM grading).
     DONE earlier: #19 (campaign #3 R6); **#16 + #17 (campaign #4 R3)**
     — ticketed navigation: server `navigate_start`/`navigate_status` run
     the drive on a motion worker (one in flight; `stop` cancels via
     `_nav_cancel`; walk/sync-navigate refuse while ticketed — two pose
     writers would race); `HabitatBase.navigate_to` is caller-blocking but
     polls the ticket every 0.25 s, so the bridge lock and op thread are
     free between polls and renders/panos interleave with the drive
     (legacy-server fallback to the sync op kept). Post-hoc timeout
     honesty: a timed-out step that EXECUTED still runs verify — verified
     → honest PASS with `result_data.timing_warning` (output captured, so
     bindings survive); unverified → still failure_class="timeout". The
     re-decompose context now names the previous attempt's verified steps
     ("plan only the remainder" — outputs stay available via ${step.path});
     **#15 (campaign #4 R2)** — request/response pairing: monotonic `rid`
     injected by the bridge and echoed by the server; STALE lines (late
     answers to timed-out requests) are discarded by rid — that is the
     resync path (a main-channel reconnect is impossible by design: closing
     it shuts the server down); torn JSON / future rid = bridge DEAD, loud.
     Per-op read-timeout tiers (walk: duration+30 s; navigate_to 300 s;
     pano/render/viewer_frame 120 s; default 60 s — the flat 60 s vs 67 s
     "walk 20 m" desync is gone); timeouts raise HabitatBridgeError, never
     bare socket.timeout. Reads moved off makefile() onto an internal
     buffered line reader (makefile refuses reads after one timeout and
     tears half-received lines). Server caps walk duration (120 s);
     **#11 + #20 (campaign #4 R1)** — dependency-failure skip with UNIFIED semantics
     (a failed step poisons transitive dependents with a skipped
     StepRecord, failure_class="dep_skipped" added to FAILURE_CLASSES;
     GoalExecutor.execute no longer aborts the whole tree, both paths share
     blocking_dependency/skipped_step_record; dead max_redecompose knob
     deleted from HarnessConfig — Layer-2 intentionally does not exist),
     and the habitat server main loop is now `_serve_main` with op
     localized per iteration (malformed first line no longer NameError-kills
     the server; tests stub habitat_sim and drive the real loop).
   Known env reds: 3 deepseek .env + level71 (now SEGFAULTS the suite on
   this desktop — canonical command carries --deselect).

0. ~~OWNER LIVE-TEST FINDING #0 (status/persona surface blind to the habitat world)~~ —
   **FIXED 2026-06-11 + NL sim startup shipped, live-verified.** (a) Persona is now
   backend-selected by `PlaygroundWorld.persona_blocks()`: habitat scenarios get
   `HABITAT_ROLE_PROMPT`/`HABITAT_TOOL_INSTRUCTIONS` ("the world is ALREADY RUNNING", no
   MuJoCo launch_explore.sh guidance). (b) `RobotContextProvider` takes `world`+`world_model`:
   the [Robot State] block carries `World: 'apartment' — habitat ... RUNNING` + live object
   count, and the go2 "Nav stack: stopped" lines no longer leak into non-MuJoCo worlds.
   (c) `robot_status` reports world/scenario/live-objects/SysNav state and falls back to
   `app_state["agent"]` mid-turn. (d) NL startup: `start_simulation(sim_type="habitat",
   scenario=...)` boots the world conversationally ("启动habitat模拟") via the new
   single-sourced `vcli/habitat_runtime.py` (same code path as `--scenario apartment`);
   `sysnav_perception(start/stop/status)` runs the perception pair ("启动sysnav") —
   fail-loud preflights, idempotent, torn down by stop_simulation too. IntentRouter routes
   habitat/sysnav phrases to the sim tools; the `system` category (robot_status) is enabled
   on NL sim start. Tests: tests/vcli/test_habitat_status_surface.py (36) + LIVE on the real
   conda subprocess: status surface verified AND the full sysnav tool chain (start → 4 real
   objects sofa/light/picture into the world model → status → stop) — owner re-test pending.
   **Owner finding #2 (2026-06-11, "我需要能看到的sim"): FIXED — live viewer window.** The
   pinned conda habitat build is the HEADLESS variant (no native window possible), so
   `server.py --gui` opens a live first-person OpenCV window (512² dedicated sensor rendered
   per-step during walk/navigate via the single-sensor draw API — the heavy equirect pair is
   NOT re-rendered; HighGUI confined to one viewer thread; user-closing the window never
   kills the sim). Plumbed end-to-end: `HabitatBridge(gui)`/`HabitatBase(gui)` →
   `habitat_runtime.resolve_habitat_gui` (env `VECTOR_HABITAT_GUI=0/1` override > tool `gui`
   param > DISPLAY-present default ON) → both entry paths. conda env addition:
   opencv-python==4.9.0.80 (numpy-1-compatible; NOTE pip first pulled cv2 4.13 and silently
   upgraded numpy→2.0.2 breaking habitat-sim — re-pinned numpy==1.26.4 + pillow==10.4.0; the
   .venv-sysnav numpy lesson now applies to the habitat-spike env too). LIVE-verified:
   X11 tree shows "Vector Habitat — apartment_1.glb" + motion driven (walk + navigate,
   reached=True). Owner re-test pending.

1. **M2 — the habitat third world: OWNER APPROVED DQ-2 (2026-06-10) — IN PROGRESS.**
   Part 1 SHIPPED: `playground/habitat/` server (standalone py3.9, conda subprocess, JSON/socket,
   navmesh `try_step` kinematics, geodesic/snap/semantic-objects oracle ops) + `HabitatBridge`
   (PORT handshake, fail-loud, watchdog-tagged) + `HabitatBase` (full BaseProtocol + narrow
   provider specs; vy honestly unsupported) — REAL subprocess e2e green on the Linux box
   (walk on skokloster navmesh, exact odom, geodesic consistency). Part 2 SHIPPED — M2 COMPLETE: `apartment` preset (embodiment `mobile`, license-free
   apartment_1, rooms honestly empty until HM3D-Semantics/DQ-1), `geodesic_dist` predicate
   (kernel oracle, fail-safe inf — the VLN success criterion), `--scenario apartment` boots
   the kinematic base via `_maybe_init_habitat_agent` (M1 `sim_backend` dispatch), and the
   acceptance e2e passes LIVE: registry world → real conda-subprocess base → real GoalVerifier
   sandbox evaluating at_position (discriminating) / geodesic_dist / facing. M3 muscle layer SHIPPED (2026-06-10): server `navigate_to` (shortest-path waypoint
   following, deterministic + bounded, honest stuck-stop) + always-on 256x256 egocentric RGB
   + `render` op (base64 PNG, lazy PIL in the conda env); `HabitatBase.navigate_to/render_rgb_png`.
   Live e2e: cross-apartment navigation verified by `geodesic_dist < 0.5` through the REAL
   sandbox + a real PNG frame. **M3 COMPLETE (2026-06-11)**: `NavigateToPointSkill` (base-generic, verify_hint =
   the VLN criterion) + mobile skill set (walk/turn/stop/navigate_to — the habitat agent's
   registry is rebuilt base-only per rule 3, no arm skills taught) + LIVE acceptance
   (tests/vcli/test_habitat_nl_slice.py, VECTOR_LIVE_LLM=1): 5 NL instructions 中/英 incl. a
   two-waypoint chain, real LLM + real conda subprocess + real harness, all verified-done,
   ≥4 hard-evidence, every verify a deterministic sandbox predicate (geodesic_dist /
   at_position / True) — never an LLM judge. M0-M3 COMPLETE. **M4 IN PROGRESS (owner directive 2026-06-11)** — Part A SHIPPED:
   habitat equirect color+depth pano op (512x1024, pose-synced) + `HabitatSysnavBridge`
   publishing the SysNav input triplet (/camera/image equirect RGB, /registered_scan
   WORLD-frame cloud unprojected from the same-frame depth — reuses the tested LidarSample
   PointCloud2 layout, /state_estimation GT odom). rclpy works IN the venv; the node
   roundtrip is integration-tested (discovery-then-drain pattern — single-threaded wait-set
   starvation documented in the test). Live: real pano pair + plausible world cloud verified.
   Part B: `tare_planner` BUILT in the sibling workspace (3m34s, msgs importable) and the
   CONSUMPTION path is green with REAL messages — stub /object_nodes_list →
   LiveSysnavBridge → WorldModel (sysnav_<id> canonical ids, VLM-confirmed confidence 1.0 —
   the v2.4 contract works unchanged; tests/integration/test_sysnav_consumption_path.py,
   skips where the workspace isn't sourced). **M4 COMPLETE (2026-06-11, DQ-3 approved)**: the FULL SysNav graph runs on the photoreal
   world — habitat pano → detection_node (YOLOE-26x TensorRT engine, exported on this GPU) →
   semantic_mapping_node (SAM2.1 base+) → /object_nodes_list → LiveSysnavBridge → WorldModel
   produced REAL labelled nodes (sofa/light/picture in apartment_1, conf 0.70 unverified tier).
   SysNav runs on ITS OWN venv (.venv-sysnav, numpy 1.26 — ROS cv_bridge is numpy-1-ABI;
   lesson: uv silently upgraded numpy when adding spacy/rerun — re-pin after every install
   batch). Harness: scripts/verify_habitat_sysnav.py. FOLLOW-UPS: object z-coords look
   frame-shifted (sofa z=-1.57) — audit the fusion pixel/extrinsic convention vs our equirect;
   old MuJoCo-era SysnavSimTool stays unregistered (superseded by the habitat runner).
   **M5 (2026-06-11): demo harness shipped + owner live-testing handover.**
   scripts/demo_third_world.py runs the full visible loop (semantic objects → NL → navigate →
   deterministic verify, artifacts under ~/sandbox/m5_demo/). The demo EXPOSED and we FIXED two
   real gaps: (1) Objects (live) now carries per-label best-confidence COORDS (semantic
   navigation was vacuous without them); (2) navigate-to-OBJECT false success killed — the
   navigate analogue of #2b: NavigateToPointSkill takes a `label` param resolved against the
   LIVE world model, unknown object FAILS LOUDLY with the known-object set (tested), guidance
   forbids inventing x/y. HabitatBridge.request is now thread-safe (demo drives nav + pano
   from two threads). Demo verdict PARTIAL (3/4: one run hit the LLM shortening strategy
   'navigate_to'→'navigate' — fail-loud worked; naming-affordance polish item). CLI handover:
   VECTOR_HABITAT_SYSNAV=1 wires the pano feed + /object_nodes_list consumer into the REPL
   agent (scripts/launch_sysnav_nodes.sh runs the perception pair). Owner testing live. Everything ungated that M2/M3 need is in place: seam (M1), grounding
   (referring expressions, step_output verify), ops (registry/daemon/watchdog), WorldBlueprint.
   On approval: conda-py3.9 habitat subprocess + socket bridge (go2-sim pattern), kinematic
   `BaseProtocol` over navmesh `VelocityControl`/`try_step`, oracle predicates
   (at_position/visited/object_visible/geodesic_dist), `vector-cli --scenario hm3d_house`.
2. **Phase E remainder:** `--daemon` goal-runner design (align with real need after M2) ·
   W3.2 capability factory + model-zoo bridge (strategic — do WITH the C.3 owner framing,
   not before it) · W3.3 remaining getattr-by-string sites (incremental, mechanical).
3. **Stage 3 grounding remainder:** referring-expression resolution SHIPPED 2026-06-10 —
   `world_context` now carries `Objects (live): ...` (world-model labels + property hints,
   sim-oracle body-name fallback, capped, fail-safe) and `_TARGET_BINDING_GUIDANCE` teaches
   resolving the user's reference (any language, attributes) to the EXACT listed name at
   plan/replan time; an unresolved binding fails loudly and replan re-binds from the fresh
   list (kernel stays deterministic — the LLM is the language component; combined with the
   #2b target-aware verify the wrong-object path cannot false-pass). STILL OPEN: the VLM
   `MuJoCoPerception`/`DetectSkill` real-perception path on no-oracle hardware.
   **Phase C.3/C.4** stays blocked behind that
   (phase-c plan superseded — see git history / campaign #8 in this doc).
4. **Owner-gated window checks (cannot verify headless):** R2-1 `--sim-go2` macOS in-process
   window (opens, no segfault, walk animates; frozen-when-idle is expected v1) · R2-4 single ^C
   aborts to prompt under mjpython, second ^C exits.
5. **Chores:** pin/vendor `convex_mpc` in `pyproject.toml` (a venv rebuild silently loses the
   numpy2 fixes — tricky-bugs Case 2) · cli.py 32 pre-existing ruff errors · version is still
   0.1.0 in pyproject/version.py while history speaks of v2.x (owner call) ·
   `ros2/nodes/agent_node.py` still calls the removed `agent.execute` (every /execute service
   call errors into the broad except) · hygiene 2026-06-10: `examples/` DELETED (all five
   taught the removed v0.1 `agent.execute` API and crashed on it; git history is the archive),
   `run_turn_unified` dark-launch docstring corrected (it is live), phase-d Stage-4/5 status
   headers truthed up, cli-tool-system diagram now shows the unified controller ·
   canary test for the
   private `mujoco.viewer._MJPYTHON` probe (`viewer_mode`) · remove the TEMP gated diagnostics
   when gait work is truly done. Linux clone (`~/Desktop/vector_os_nano`) reconciled 2026-06-10:
   synced to master, `.venv` rebuilt (mujoco 3.9 / numpy 2.4.6 / pin 4.0 + convex_mpc editable),
   `sim.sh` hardened (no silent system-python fallback). NOTE on Linux the known-red
   `test_sim_tool_lifecycle_dev_to_arm_to_dev` is a process-killing SEGFAULT
   (`mujoco.viewer.launch_passive` GL thread) — deselect it there; an intermittent
   at-exit GL-teardown segfault observed twice now (GLFW 'not initialized' warning, AFTER the
   summary prints — results complete both times; treat exit 139 with a complete tail as green).

## Run / verify

Tests: see CLAUDE.md "Build / test" (canonical command, expected environmental reds).
Quirk: go2 sim load rewrites `mjcf/go2/scene_room_piper.xml` abs paths — `git checkout` it
before committing. Live validation pattern: real CLI + deepseek headless first; GUI/timing/^C
behaviors are owner-window checks — never claim them verified headless.

## Pointers

- Rules + read order: [../CLAUDE.md](../CLAUDE.md)
- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Hidden bugs: [tricky-bugs.md](tricky-bugs.md)
- Plans: none active (campaign #8 COMPLETE; realsim-plan.md deleted per doc-governance, see git history). Open item: DQ-4 (merge feat/playground-vln → master).
- ADRs: [ADR-006](architecture-decisions/ADR-006-agent-kernel-world-plugin.md) ·
  [ADR-007](architecture-decisions/ADR-007-closed-loop-controller.md) ·
  [ADR-008](architecture-decisions/ADR-008-playground-parallel-track.md)
- Superseded docs live in git history (`git log --all -- <path>`). No working-tree archive.

## Autonomous loop

Owner-away iterations are campaign-driven and live OUTSIDE the repo:
`~/.vector-nano-loop/{constitution,campaign,journal,next-prompt,decision-queue}.md`.
Start with `/loop` + the constitution prompt (constitution.md is the fixed state machine;
campaign.md holds the current milestones). The standing-mission prompt that used to live
here is superseded — git history has it.
