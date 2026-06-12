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
- Last updated: 2026-06-11.
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

-1. **OWNER LIVE-TEST FINDINGS 2026-06-12 (post-campaign-#2, FIX FIRST next
   session).** Owner ran the pure-NL flow in a plain terminal (no ROS sourced):
   (a) **G1 body wrong in the chase view** — the robot renders TINY/misplaced
   relative to the furniture (screenshot
   ~/Pictures/Screenshots/'Screenshot from 2026-06-12 01-30-25.png': at
   pos (-2.5, 0.6) the G1 looks ~0.4 m tall next to a bar table; expected
   1.32 m). Hypotheses to discriminate: GLB unit/scale applied by habitat's
   object template loader; chase-cam FOV/offset making it look small; body
   placed at navmesh y (+0.12 float) — measure the rendered body height
   against a known scene object headless before touching anything.
   (b) **"启动sysnav" fails NL-start from an unsourced shell** — fail-loud
   worked (told the owner to source ROS+SysNav), but the N5 promise is NL
   start WITHOUT terminal rituals. Fix direction: run the consumer/feed in a
   subprocess that sources the overlays itself (launch_sysnav_nodes already
   does this for the GPU pair) or extend AMENT/PYTHONPATH programmatically
   at tool start; the bare `vector-cli` symlink launcher could also source
   ROS when present.
   (c) **Walking/NL motion did not work in the owner's session** — mode
   unknown (owner report, no transcript of the failing turn; the first
   `启动habitat模拟` also hit a transient `Error: Connection error.` from
   deepseek). Reproduce in a real CLI turn: NL walk + navigate after NL sim
   start — suspect surface: the production turn pipeline (permissions on
   motor tools / VGG rebuild after NL sim start / IntentRouter), since the
   scripted harness path (engine.vgg_* direct) passed.
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
   ([agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md)).
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
- Plans: [agent-kernel-phase-e-plan.md](agent-kernel-phase-e-plan.md) (CURRENT, Wave 2) ·
  [agent-kernel-phase-d-plan.md](agent-kernel-phase-d-plan.md) (Stage 3 remainder open) ·
  [agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md) (C.3/C.4 decisions open)
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
