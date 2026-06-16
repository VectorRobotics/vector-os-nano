# Verified Agent Kernel — STATUS (resume anchor)

One-page "where are we / what's next". Read this first when resuming; durable design is
[ARCHITECTURE.md](ARCHITECTURE.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md);
full round-by-round history is in `git log` + the loop journal (`~/.vector-nano-loop/`).

- Branch: `feat/playground-vln` (campaigns #2–#10 live here; #2–#8 merged to `master` via DQ-4 @ `3e82996`).
- **Campaign #11 (CURRENT, owner 2026-06-15) — 统一具身 × 能力矩阵.** Intermediate goal: in ONE vector-cli sim, freely switch G1/Go2, each running the full stack — locomotion + navigation stack + VLN + SysNav — any embodiment × any capability, one NL command end-to-end, all real-sim accepted (vector-cli is the only acceptance surface). Self-directing loop: each round self-authors its prompt and runs DESIGN(workflow)→BUILD(TDD)→TEST/ACCEPT(real sim+cli)→DOC+COMMIT→NEXT; explicit code/doc REVIEW + real sim/cli regression every 4th round. Spec: `~/.vector-nano-loop/campaign.md`. Milestones M0(audit+switch-seam design)→M1(switch+locomotion)→M2(nav stack both)→M3(VLN both)→M4(SysNav both)→M5(full-matrix generalize). **M0 DONE (R1, workflow wbikcsq4w)**: capability matrix (Locomotion both READY; Nav G1 READY/Go2 PARTIAL; VLN G1 READY/Go2 PARTIAL; SysNav both MISSING — feed side) + unified-switch-seam design → ADR-011 + CEO gate DQ-14 (PENDING). Key finding: the rebind already exists (`sim_tool.py:216-271`) but refuses when a base is connected — the seam = extract `_rebind_agent` + add `switch_embodiment` tool (boot-then-swap, capability-probe skill registration, kernel/BaseProtocol untouched). Verified claims vs code; correction: `SkillRegistry` has no `unregister` → rebuild fresh registry on boot. **M1 DONE (R2)**: runtime g1⇄go2 in-session hot-switch shipped. `SimStartTool._rebind_agent` extracted (single-source rebind, rule 3; drops stale skill-wrapper tools) + new `SwitchEmbodimentTool` (`vcli/tools/switch_tool.py`, category sim, intent triggers switch/切换/换成): boot-then-swap, fail-loud, no-op guard, g1 `prefer_daemon=True`. kernel/BaseProtocol untouched (rule 2/7). **Real-sim PASS** (`~/sandbox/c11_m1_switch_leak.py`, 6 boots g1⇄go2): walk every cycle (go2 ~0.72 m, g1 ~1.53 m — gait re-mounts on fresh base), ZERO gait threads survive teardown, renderers+model released, VRAM growth 0 MB, thread growth 0. Tool-path verified through `execute()` (go2→g1 swap + walk 1.53 m; no-op guard works). Tests: 9 new + suite green. (base classes: `MuJoCoGo2` / `G1MuJoCoBase`.) **M2 (R3) — Go2 VLN infrastructure SHIPPED (photoreal end-to-end PARTIAL).** Gave `MuJoCoGo2` `navigate_to`(→3-value dict)/`geodesic_distance` (reuse `g1_vgraph`, Go2 trot follows waypoints; `_GO2_NAV_FALL_Z=0.20` not G1's 0.4 — Go2 stands ~0.28-0.35) + wired `LaserScan.points` from the world-frame lidar cloud (rule-6 additive) so `recognize_navigate` can locate + `go2_runtime` registers `RecognizeNavigateSkill` by capability probe. **Real-sim PASS**: Go2 `navigate_to` drives to a coordinate (reached, 1.20 m, z=0.278 > FALL_Z — the #1 silent-fail fix validated); skill registered; photoreal Go2 camera renders fine standalone (1.5 s). 8 new TDD green. **OPEN (photoreal VLN end-to-end):** `recognize_navigate` on Go2 timed out the camera mid-loop (`camera failed: timed out`, 158 s) — NOT the nav code; a Blender-bridge timeout-under-load during the multi-iteration recognise→advance→re-range loop (camera works standalone). Queued for R4 debug (Hypothesis Loop). **R4 REVIEW done** (workflow w0pbiyrk5 + Hypothesis-Loop debug): code/doc audit applied — ARCHITECTURE §7 gains the campaign #11 embodiment-switch seam + go2 VLN-symmetry note; `LaserScan.points` documented live-only (rule-4); regression PASS (g1⇄go2 switch+walk, zero leak, after M2 additive changes). **Camera-timeout root cause:** instrumented run = 9 photoreal renders fast (~1 s) then the **10th Blender render hangs ~120 s** (server hang under sustained render-while-walking) AND deeper — `navigate_to` never fires: the chair is never *located* (lidar `points` populated=3879, but bearing→locate doesn't resolve the furniture target). Both = photoreal-VLN-end-to-end blockers → **queued M2-followup** (not safe quick-fixes; M2 nav infra stays validated). **Logged debt (R5):** extract shared world-agnostic `_nav_controller` (g1/go2 ~180-line nav dup, rule 7/3) gated on G1 nav regression; capability-probe test drive `boot_go2_agent`; navigate_to `waypoints`-key uniformity; remove leftover `VECTOR_CMDVEL_LOG` diag. **R5 DONE — shared `_nav_controller` extracted (rule 7/3 nav-dup debt paid).** New `hardware/sim/_nav_controller.py`: `drive_to_point` + `route_and_drive` (campaign-#6 controller VERBATIM) + frozen `NavConsts` + a ctrl-token factory; G1/Go2 now keep thin delegates (`_G1_NAV`/`_GO2_NAV` packs; G1 uses no-op nullcontext, Go2 passes its `_ctrl_token` CM + `_drive_for`). ~180 dup lines → one shared controller (can't drift). **G1-regression gate PASSED before touching Go2** (G1 navigate reached=True, behavior preserved); Go2 navigate PASS (re-run; the 0.31-vs-0.30 borderline is gait variance, not regression); M1 switch leak harness PASS (both walk, zero leak); 8 new controller TDD (incl. FALL_Z-injection + ctrl-token-order pins). kernel/BaseProtocol untouched. **M3 (R6) DONE — nav-stack capability on BOTH embodiments (in-process scope).** `go2_runtime` now capability-probe-registers `NavigateToPointSkill` (alongside `recognize_navigate`); both G1 and Go2 drive coordinate goals through the shared `_nav_controller` returning the same three-value dict (rule-3 single-source). Added a fail-loud guard in `NavigateToPointSkill` for a non-dict `navigate_to` return (the raw `Go2ROS2Proxy` bool, track-B). **Real-sim**: Go2 `navigate_to` skill drives to the coordinate (moved 1.13 m, geodesic-verified); G1 already shipped. Known Go2-trot precision: it stops ~2-3 mm past the arrival break, so `reached` straddles the exact tol boundary (base method straddles identically: 0.297/0.302/0.402) — reliable with ~3 cm tol margin; a Go2 capture-creep tuning follow-up, NOT loosened verify (rule 5). 9 TDD green. **FAR ROS2 nav stack = BLOCKED here** (nav-stack C++ nodes not installed — only build tree) → **DQ-15 CEO gate / track-B** (colcon rebuild + `Go2ROS2Proxy` bool→dict adapter). **M4 (R7) DONE — SysNav feed-source on BOTH embodiments (seam scope; live nodes track-C).** Added `get_pano()→{rgb,depth,pos,heading}` (HabitatBase-contract) to `MuJoCoG1` (pelvis) + `MuJoCoGo2` (base_link), backed by a new `MuJoCoPano360.render_rgbd()` (6 depth cube-faces stitched via the same LUT, nearest gather). `sysnav_tool` gate confirmed capability-based (`hasattr get_pano`) — message de-habitat-ified; `wire_sysnav_feed` already embodiment-agnostic. kernel/BaseProtocol untouched (rule 2/7). **Real-sim PASS**: both bases emit a non-empty 1920×640 equirect (g1 1.8M nonzero px depth≤28 m; go2 1.85M depth≤23 m), panos saved. pano GL renderers closed on disconnect (leak guard). **Live SysNav object_nodes = DQ-16 CEO gate / track-C** (the `~/Desktop/SysNav/.venv-sysnav` python interpreter is deleted + broken params path + unverified TRT engines — heavy re-provision). M5 still needs object detection → also gated on DQ-16. **NEXT R8 = REVIEW round** (full-matrix real sim+cli regression: g1⇄go2 switch + walk + navigate + VLN + get_pano + code/doc audit + full pytest). Then M5 (full-matrix generalization, partly DQ-16-gated). Deferred debt: go2-trot arrival tuning; `sysnav_runtime.py` extraction (optional, wiring already agnostic); capability-probe test→`boot_go2_agent`; `waypoints`-key uniformity; remove `VECTOR_CMDVEL_LOG`. M2-followup: photoreal-VLN locate + 10th-render hang. Deferred debt: go2-trot arrival tuning; capability-probe test→drive `boot_go2_agent`; navigate_to `waypoints`-key uniformity; remove `VECTOR_CMDVEL_LOG`. M2-followup: photoreal-VLN locate + 10th-render hang. (campaign #10 DQ-13 eye-in-hand grasp committed `9bd9a15` but BANKED — not in this goal; resume later.)
- (history) 2026-06-15 campaign #10 DQ-13: eye-in-hand wrist camera. Added downward `piper_wrist_rgb`/`piper_wrist_depth` on link6 (optical axis = link6 +z = world -Z at top-down) + overhead SCAN POSE + `get_grasp_observation`/`get_scan_pose`/`get_support_z` on MuJoCoGo2 + `_scan_then_observe` in recognize_pick (near-vertical ray -> no R18 overshoot). REAL-SIM verified: STEP 0 scan pose reachable from grasp standoff (x>=10.6); STEP 1a locate round-trip 0.0cm / worst bbox-centre 2.2cm (vs old forward-cam 20-30cm), all objects in-frame at fovy=58, scan_height=0.25. Tests green (860 + 3 new DQ-13). R20 (2026-06-15): hardened the scan path — `_scan_then_observe` now SWEEPS scan heights (0.25..0.12, highest reachable first; the live standing base settles lower/drifts so the reachable height varies); wrist fovy 58->80 (frame the y=+-0.15 object span at low reachable heights); optional `VECTOR_PICK_SINGLE_LABEL` to RENDER a named object (e.g. the can) — scene presentation, VLM still finds+locates it (rule 5). **Closed grasp NOT yet achieved — diagnosed the blocker (real sim, NOT a perception flaw):** the Go2 (~0.65 m body) COLLIDES with the pick table (front edge x=10.80) at any standoff close enough to reach objects at x=11.0; teleporting the dog to x=10.8 drove it into the table (contact explosion -> dog bounced to 10.375, can flung to floor z=0.033). Without disturbance objects rest correctly at z=0.24. NEXT (R21): fix the standoff geometry — either walk the dog gradually to a stable standoff (as the live flow does) or move the pickables to a reachable, non-colliding x (~10.85, sanctioned by the loop prompt) + aim `get_scan_pose` at them; then close the grasp (is_holding()=True headless + vector-cli "认出罐子抓起来" + screenshot). The core fix (eye-in-hand near-vertical locate) is verified; this is scene/standoff geometry.
- Scope guard: this is **vector-os-nano only** — not the UniLab go2arm-grasp work.

## North star

**Vector OS Nano = natural language controls everything**, via a built-in agent that
decomposes NL → plans → executes → verifies → replans (a grounded CLOSED loop). VGG is that
engine; verify is the moat (deterministic, never LLM-graded). Robots are the end; the dev/
macOS path is a means. Generalize across embodiments (arm, go2, g1) — never one-off patches.

## Current state / what's next

**Campaign #10 (NEXT, owner 2026-06-14) — HIGH-FIDELITY perception sim for real VLN + manipulation.**
The owner's goal: G1 doing autonomous navigation AND manipulation in a sim whose PERCEPTION is
real enough to drive genuine VLN — real camera images + multi-sensor data feeding real image
recognition / VLN, **not just physics**. Constraints/lessons:
- Isaac Sim = too heavy.
- **MuJoCo alone is a physics engine** — campaign #9 proved a VLM on MuJoCo's basic render hits a
  perception-fidelity ceiling (domain gap, noisy/flaky grounding). "If it's only MuJoCo, the
  existing go2 + NAVSTACK already covers nav" — so MuJoCo-VLM-render is NOT the differentiator.
- The real need is a **high-fidelity perception substrate** (photoreal/co-sim/other) good enough
  for real recognition + VLN + manipulation. **This re-opens the DQ-10 substrate decision = CEO
  gate (DQ-11).** Spike in `~/sandbox/` before any repo dependency.
- **R1 PROBE done** (workflow `waxcsgpd6`, 5-agent panel; ADR-010): top three are a paper tie
  (Genesis 54 / SAPIEN 53 / co-sim 53 of 60), so the deciders are **reuse + risk**, not score.
  **Recommended primary = co-sim** (keep MuJoCo physics unchanged → reuses all of #5–#9; bolt on a
  Blender/OptiX photoreal renderer = the *only* candidate with confirmed RTX 5080/Blackwell support;
  clean license). **SAPIEN/ManiSkill3 = strong fallback** (best paper scores + cleanest in-process
  seam, but rtx=2 Blackwell unknown + discards MuJoCo reuse). Genesis = track-don't-bet (closed-source
  1-month Nyx wheel). habitat re-pin REJECTED (suction-grasp is architectural; can't get photoreal +
  manipulable in one scene). Isaac REJECTED (owner-excluded + live Blackwell TiledCamera hang).
  co-sim's one real unknown = render latency.
- **R2 PROBE-spike done** (`~/sandbox/c10-substrate-spike/`, repo zero-dep): **co-sim's killer risk
  RESOLVED** — Blender Cycles+OptiX runs natively on the RTX 5080 Blackwell (`OPTIX: RTX 5080`), ~826 ms
  @32 samples (~1.2 Hz, persistent process avoids warmup) = loop-viable. Honest finding: on low-poly
  Kenney assets a renderer swap is only a *modest* grounding gain (Blender 2/3, MuJoCo 2/3, failing on
  *different* objects; grounding is stochastic/flaky, not resolution-bound). **The real lever is
  photoreal ASSETS + perception-pipeline robustness, not the engine** — which *strengthens* co-sim
  (keep all #5–#9 physics reuse; add photoreal assets incrementally; greenfield migration buys the
  same asset-bound ceiling).
- **R3 PROBE-spike done** — "asset fidelity is the lever" CONFIRMED: a photoreal CC0 armchair
  (PolyHaven 4K PBR) rendered in Blender OptiX grounds **chair 0.95, precise + correctly disambiguates
  "not a sofa"** through the same 160px pipeline that gave flaky 2/3 on toy meshes. **co-sim now
  validated end-to-end on the RTX 5080** (OptiX + ~1.2 Hz + confident photoreal grounding + full
  physics reuse + free CC0 assets) → recommendation firms to a confident **co-sim** (SAPIEN alt if its
  richer out-of-box ecosystem outweighs MuJoCo reuse). **DQ-11 RESOLVED (owner 2026-06-14,
  option "borrow don't depend"): self-built lightweight co-sim — MuJoCo physics (reuse #5–#9) +
  Blender Cycles/OptiX photoreal render behind a `PhotorealRenderer` world adapter (kernel untouched,
  rule 2), MuJoCo-state→frame over the #9 subprocess+socket scaffold. Fits our 24.04 / RTX 5080 /
  non-ROS vector-cli stack. Covers BOTH G1/Go2 photoreal VLN AND manipulation (Piper, #17/#19) — what
  the evaluated MATRiX (zsibot, MuJoCo+UE5) doesn't (quadruped-only, no public image, 22.04/Humble-only).
  MATRiX/UE5 = architecture validation, not a dependency; UE5-plugin = optional real-time upgrade only
  if Blender latency blocks. Heavy external deps remain CEO-gated.**
- **BUILD sequence (campaign.md):** (1) `PhotorealRenderer` seam: MuJoCo state → Blender subprocess →
  RGB frame, pose-aligned, behind `get_camera_observation()`, one frame end-to-end (TDD). (2) photoreal
  scene: PolyHaven CC0 PBR furniture + photoreal room (or 3DGS scan), VLM-grounding step-change vs #9.
  (3) G1/Go2 photoreal VLN (reuse recognise→navigate on photoreal frames). (4) manipulation: Piper
  grasp driven by photoreal perception. vector-cli only + screenshots; real VLM always called (rule 5).
- **BUILD R1 DONE** (`vector_os_nano/playground/photoreal/`): the co-sim wire plumbing, repo zero-dep.
  `camera.py` = pure MuJoCo→Blender camera transform (identity-of-columns; both engines share Z-up
  world + camera -Z-forward/+Y-up — TDD pins the axes so a refactor can't slip in drift). `bridge.py` =
  `BlenderBridge` client (clones the #9 subprocess+socket pattern; `blender_bin()`/`blender_available()`
  via `VECTOR_BLENDER`; PNG over base64). `server.py` = Blender-side Cycles/OptiX render server (GPL,
  subprocess-isolated — repo venv never imports `bpy`). 12 tests green (pure pose + fake-server protocol
  + Blender-gated e2e). End-to-end proven on the box: pose → bridge → **OPTIX RTX 5080** render (640×480
  @64 = 1085 ms) of a photoreal CC0 armchair → real Qwen-VL grounds **chair 0.95 + correctly rejects
  sofa** — same fidelity as R3, now through the production seam. Evidence: `~/sandbox/c10-substrate-spike/
  r1_e2e_armchair.png`. NOT yet done (R2): the `PhotorealRenderer` adapter that reads a live MuJoCo
  `(model,data)` (cam pose + body→asset scene spec) behind `get_camera_observation()`, + the photoreal
  multi-asset room. (Note: live qwen-VL hit a transient upstream 429 mid-round — called for real, failed
  loud, succeeded on retry; never faked.)
- **BUILD R2 DONE** (commit pending): the `PhotorealRenderer` world adapter + scene builder. `scene.py` =
  pure `build_room_scene_spec(placements, asset_map, ...)` → Blender scene dict that SUBSTITUTES photoreal
  CC0 assets for the toy meshes at the room's visible centres (unmapped labels skipped — partial libraries
  are valid). `camera.py` gains `look_at_camera_matrix` (build demo/test poses in MuJoCo convention).
  `renderer.py` = `PhotorealRenderer.observe(model, data)` reads the LIVE `g1_head_cam` pose
  (`cam_xpos`/`cam_xmat`/`cam_fovy`) and renders the photoreal frame from EXACTLY that pose → returns the
  `{rgb, cam_pos, cam_mat, fovy}` dict shape recognise→navigate already consumes. 18 playground tests green
  (pure scene + look-at + fake-bridge renderer asserting matrix_world == MuJoCo pose, no drift end-to-end;
  uses a tiny hand-built MJCF, no Blender/network). E2E on the box: a real MuJoCo first-person `g1_head_cam`
  pose → adapter → OPTIX 640×480@64 (~1.08 s) of the photoreal armchair room → real Qwen-VL **chair 0.95**.
  Evidence `~/sandbox/c10-substrate-spike/r2_room_photoreal.png`. Honest: the single-seat armchair also
  grounds as **sofa 0.9** (genuine VLM ambiguity on one upholstered seat — a distinct sofa asset
  disambiguates; asset-library expansion is later).
- **BUILD R3 DONE** (commit pending): photoreal wired into `mujoco_g1.get_camera_observation` as a HYBRID —
  rgb=Blender(OptiX), depth/pose=MuJoCo, one camera frame (the Blender render runs from the exact pose
  MuJoCo rendered the depth at, and over a socket so it's safe off the control thread, unlike MuJoCo's GL
  Renderer). Env-gated `photoreal` flag (default OFF → furnished-room behaviour byte-identical, rule 2/9);
  `boot_g1_agent` enables it on `g1_room_vlm` when `VECTOR_G1_PHOTOREAL` is set. Lazy `BlenderBridge`
  (injectable for tests), closed in `disconnect`; fails loud if requested with no Blender; assets from
  `VECTOR_PHOTOREAL_ASSETS` (heavy CC0 assets never vendored — unmapped targets skipped). `renderer.py`
  gained `render_from_pose`. 22 playground tests green + full suite 1670 passed, no g1 regression. **E2E
  through the REAL g1 base** (booted room+furnished+photoreal headless): `get_camera_observation` returns
  `{rgb(640×480 Blender), rgb_mujoco, depth, cam_pos, cam_mat, fovy}` and the real Qwen-VL grounds **chair
  0.9** on the photoreal frame. Evidence `~/sandbox/c10-substrate-spike/r3_g1_photoreal_obs.png` (+ the
  MuJoCo frame side-by-side).
- **BUILD R5 — vector-cli photoreal ACCEPTANCE PASSED** (owner-requested): live REPL,
  `vector-cli --scenario g1_room_vlm` + `VECTOR_G1_PHOTOREAL=1` (boot banner shows "[PHOTOREAL co-sim:
  Blender/OptiX]"). NL command `找到椅子并走过去` → VGG planner → `recognize_navigate` →
  **[PASS] verify `visited('chair')`, 1/1 verified (21.1s)** — the real VLM grounded the chair on the
  offscreen photoreal Blender frame, depth-at-bbox located it, `navigate_to` drove there, and the
  DETERMINISTIC verify confirmed arrival at the chair's GT coords (rule 5, no teleport). Live viewer
  screenshot shows the G1 standing AT the chair: `~/sandbox/c10-substrate-spike/r5_vcli_viewer.png`. The
  hybrid co-sim is confirmed in the real CLI: physics+viewer=MuJoCo, perception=photoreal Blender. Clean
  teardown (Blender subprocess closed on quit; exit segfault is the known tolerated GL-teardown 139).
  **builds #1–#3 DONE + CLI-accepted.** Remaining: scene-quality polish (per-asset yaw so the chair faces
  the camera, walls, sofa/plant CC0 assets to disambiguate) — optional; and **build #4 = Piper
  manipulation** (reuse #17/#19 MuJoCoPiper + PickTopDownSkill driven by photoreal perception).
- **BUILD R6 DONE — photoreal co-sim generalized to the Go2/Piper embodiment** (commit pending). Extracted
  the furnished-room co-sim wiring into a shared factory `playground/photoreal/cosim.py`
  (`furnished_room_renderer` + `furnished_room_asset_map`) so G1 and Go2 share ONE asset-mapping +
  bridge-lifecycle (rule 7, world-agnostic); refactored g1's `_ensure_photoreal_renderer` onto it (g1
  tests still green). `MuJoCoGo2` gained the same env-gated `photoreal` flag: `get_camera_frame` returns a
  Blender render from the resolved head cam's (`d435_rgb`/`RECOG_CAM`) LIVE pose; lazy bridge closed on
  disconnect; `go2_runtime` enables it on `VECTOR_G1_PHOTOREAL`/`VECTOR_GO2_PHOTOREAL`. 28 playground tests
  green + suite 1676 passed, no go2 regression. E2E through the real go2 base (headless furnished+photoreal):
  `get_camera_frame` → Blender 640×480 → real Qwen-VL **chair 0.9** (`r6_go2_photoreal.png`). This is the
  structural enabler for **build #4**: the Go2 carries BOTH the head camera and the Piper arm, so the pick
  skills' perception (autodetect/recognise) now grounds on photoreal frames.
- **BUILD R7 DONE — photoreal PICK-scene rendering** (commit pending). The pick scene is the apartment
  (table at x=11 + 3 graspable `pickable_*` cylinders: blue/green bottle, red can), a different world from
  the furnished VLM room. Added: `scene.py` `objects` primitive passthrough; `server.py` renders
  cylinder/box primitives (Blender, coloured — no mesh asset); `cosim.py` `build_pick_scene_spec` +
  generic `scene_renderer`; `MuJoCoGo2._ensure_photoreal_renderer` now branches (furnished→room,
  else→pick scene built from LIVE `pickable_*` body poses + colours/sizes read from the model, no
  hardcoding) via new `_pick_objects`/`_pick_table`. 29 playground tests green + suite 1676 passed (the
  3 deepseek reds + 1 habitat ticketed-drive TIMING flake that PASSES in isolation and is outside the
  changed files — not an R7 regression). E2E through the real go2 pick base (headless, photoreal):
  `_pick_objects` correctly reads the 3 live cylinders, `get_camera_frame` renders them photoreal, and
  the real Qwen-VL grounds **"red can" 0.9** (`r7_go2pick_photoreal.png`). Honest: generic "can"/"bottle"
  → `[]` — the simple primitives ground by COLOUR (the R2 toy-primitive flakiness); photoreal can/bottle
  MESH assets would harden it. This proves the photoreal PERCEPTION substrate for manipulation. NEXT (R8):
  close the loop — `vector-cli` go2_piper "认出红色的罐子→抓起来" with the grasp target PERCEPTION-derived
  (recognise→pick, not registered GT) + verify grasped + screenshot; optional photoreal object meshes.
- **First post-approval task:** prune the superseded MuJoCo-VLM-render perception code from #9 (kept
  for now — tested + interconnected; the world-agnostic builder / recognise→navigate / target_locate
  geometry are reused on the new substrate).

**Campaign #9 outcome (MuJoCo furnished-room VLN, feat/playground-vln) — sound engineering, capped by
the perception substrate:**
- World-agnostic furnished-room builder (one builder, two embodiments — g1 + go2; rule #2/#7).
- Real Qwen-VL furniture recognition on the MuJoCo render (both embodiments) — works, but flaky on
  MuJoCo's non-photoreal images (the ceiling above).
- Reliable arrival via **recognise → depth-at-bbox locate → navigate_to** (g1 headless 3/3) after
  visual-servoing proved flaky (tricky Cases 21–23).
- g1 integrated into vector-cli (`start_simulation(sim_type='g1')`, "启动g1"); live viewer shows the
  furnished room; in-REPL g1 walks via a daemon control thread (Cases 22/24). Suite 1648 green.

## Campaign history (one line each; details in git log + loop journal)

- **#2** full-stack居住世界: house world (ReplicaCAD) + 50Hz cmd_vel + real nav stack + full VLN 5/5.
- **#3** kernel refactor: false-PASS family killed (pre_satisfied / evidence-gate / single-source verify).
- **#4** robustness + LLM param-binding + E2E GUI smoke + N6 gait PROBE.
- **#5** G1 real gait (unitree_rl_gym pretrained policy, MuJoCo, zero new deps).
- **#6** G1 closed-loop waypoint navigate (obstacle-aware vgraph; euclidean geodesic on flat).
- **#7** visibility-graph obstacle planner (pure geometry); batches folded into #8.
- **#8** MuJoCo furnished-physics VLN (DQ-10=A): gait+collision+avoidance+lidar/occupancy/camera+
  explore+colour-recognition→go, one vector-cli command, GUI-verified. Merged master (DQ-4).
- **#9** real VLM semantic recognition + recognise→navigate + double-embodiment + g1 vector-cli
  integration (this campaign — see Current state).

## Run / verify

Tests: see CLAUDE.md "Build / test" (canonical chunked command, expected environmental reds:
3 deepseek `.env` + level71 GL segfault — always `--deselect` it; exit 139 + full summary = green).
Quirk: go2 sim load rewrites `mjcf/go2/scene_room_piper.xml` abs paths — `git checkout` it before
committing. Live validation: real CLI + deepseek headless first; GUI/timing behaviours are
owner-window checks (tmux real vector-cli + screenshot) — never claimed verified headless.

## Pointers

- Rules + read order: [../CLAUDE.md](../CLAUDE.md)
- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Hidden bugs: [tricky-bugs.md](tricky-bugs.md)
- Subsystems: [cli-tool-system.md](cli-tool-system.md) · [skill-protocol.md](skill-protocol.md) · [sim-dev-guide.md](sim-dev-guide.md)
- ADRs: [ADR-006](architecture-decisions/ADR-006-agent-kernel-world-plugin.md) ·
  [ADR-007](architecture-decisions/ADR-007-closed-loop-controller.md) ·
  [ADR-008](architecture-decisions/ADR-008-playground-parallel-track.md) ·
  [ADR-009](architecture-decisions/ADR-009-third-world-simulator-selection.md)
- Plans: none active. Open CEO gate: campaign #10 substrate selection (decision-queue.md).
- Superseded docs/STATUS detail live in git history (`git log --all -- <path>`). No working-tree archive.

## Autonomous loop

Owner-away iterations are campaign-driven and live OUTSIDE the repo:
`~/.vector-nano-loop/{constitution,campaign,journal,next-prompt,decision-queue}.md`.
Start with `/loop` + the constitution prompt (constitution.md = fixed state machine; campaign.md =
current milestones). Past campaign journals are archived as `journal-cN-DONE.md`.
