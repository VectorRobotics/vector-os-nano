# Tricky Bugs — hidden-bug casebook

**This doc records the IMPLICIT/hidden bugs hit during development** — the ones whose
symptom pointed away from the cause, that survived a green test suite, or that hid behind
"every component is correct in isolation". Append-only, newest case first. Keep entries
SHORT: dot points, key facts only (symptom → why hidden → root cause → fix → lesson).
Routine bugs do NOT belong here; git history covers those.

---

## Case 1 — Go2 explore gait instability (飘/瘸腿): two-clock skew (2026-06, fixed `d7e158b`)

- **Symptom:** during explore the gait went unstable/limping, step size over/undershoot.
  Worse on a loaded machine (GUI + RViz). Single `walk` skill commands looked fine.
- **Why hidden:** every component was correct in isolation. Ruled out BY DATA before the
  real cause: code regression (gait + bridge byte-identical to a known-good commit),
  duplicate cmd sources (cmd log: single MainThread @19 Hz, smooth), duplicate physics
  daemons (count=1), mujoco 3.1.6-vs-3.9, pinocchio 3.9-vs-4.0 (dynamics byte-identical),
  swallowed QP failures (2645/2645 ok), solver tolerance, velocity smoothing.
- **Root cause:** a CROSS-DOMAIN interaction invisible in either domain alone. The physics
  daemon ran compute-bound at ~0.65× real-time, while `go2_vnav_bridge._follow_path`
  ramped velocity by fixed per-tick increments on a 20 Hz WALL timer → the commanded
  velocity profile slewed ~1.5× faster in the gait's own (sim) time than it was tuned
  for → MPC gait destabilized.
- **Cracked by:** measuring before theorizing — tiny env-gated diagnostics
  (`VECTOR_PHYS_LOG` printed `sim/wall≈0.65x` + daemon count; `VECTOR_CMDVEL_LOG` proved
  one clean command source). The ratio number WAS the diagnosis.
- **Fix:** integrate every ramp/accumulator in `_follow_path` against actual sim-dt
  (`hardware/sim/sim_clock.sim_tick_dt` + `MuJoCoGo2.get_sim_time()`); the wall-escape
  state machine converted to sim time too (adversarial review caught the remaining mixed
  time bases). sim/wall=1 → byte-identical; no sim clock (real HW) → nominal fallback.
- **Lesson:** a wall-clock controller commanding a simulation must integrate by sim-dt.
  "Sim runs slower than real-time" silently changes the meaning of every per-tick
  constant — and no unit test catches it, because each side is correct alone.

## Case 2 — MPC "stands but won't walk": errors swallowed by `except: pass` (2026-06)

- **Symptom:** the dog stood up but never walked. No error, no warning, anywhere.
- **Why hidden:** the QP fallback in the per-tick control loop was a bare
  `except Exception: pass` — it ate the SAME exception every single tick, so a hard
  dependency break presented as silent physical misbehavior instead of a stack trace.
- **Root cause:** external `convex_mpc` was written for numpy<2; the rebuilt venv had
  numpy 2.x, which hard-errors on `(N,1)`-array→scalar-slot assignment. The solver threw
  on every tick → no torque ever computed → PD hold only.
- **Fix:** shape-only fixes at the source (`compute_com_x_vec` → `(12,)`,
  `compute_current_mask` → `(4,)`); the except clauses now count+log failures
  (`VECTOR_MPC_LOG`). NOTE: convex_mpc is still not pinned in `pyproject.toml`.
- **Lesson:** `except: pass` on a control path converts loud failures into silent wrong
  behavior. Always count/log swallowed exceptions — "0 failures" must be provable.

## Case 3 — Explore stack silently ran on the WRONG interpreter: dead PYTHONPATH (2026-06, fixed `13a9429`)

- **Symptom:** none, at first — explore "worked", but on system python3 (mujoco 3.6)
  instead of the repo venv (mujoco 3.9 / numpy 2.4.6 / pinocchio 4.0) where the MPC fix
  had been verified.
- **Why hidden:** the uv rebuild renamed `.venv-nano` → `.venv`; the launch scripts'
  PYTHONPATH pointed at the deleted dir. Python silently ignores nonexistent path
  entries and falls back to whatever system site-packages provide.
- **Root cause:** the venv path was hardcoded in 12+ scripts with no existence check and
  no single source.
- **Fix:** all launch/test scripts + `vector-sim` + `verify_pick_top_down.py` prefer
  `.venv` with a `.venv-nano` fallback; CLAUDE.md build line updated.
- **Lesson:** PYTHONPATH to a missing dir fails silent — when debugging "version
  mismatch" symptoms, print `module.__file__` to verify WHICH copy is actually loaded;
  single-source interpreter resolution.

## Case 4 — 50 Hz odom capped at 21 Hz: the executor, not the workload (2026-06, fixed `cd20c9f`)

- **Symptom:** the N1 `/state_estimation` timer (50 Hz) measured 21-25 Hz live, while
  the underlying stream-channel roundtrip averaged 0.24 ms — three orders of magnitude
  of headroom.
- **Why hidden:** the obvious suspect was the ~1 s pano callback blocking the timer
  (same node, default MutuallyExclusiveCallbackGroup). Moving the fast path to its own
  callback group changed nothing — the misdirection survived one "fix".
- **Root cause (measured, discriminating experiment):** rclpy's MultiThreadedExecutor
  itself caps a 50 Hz Python timer at ~29 Hz on this box — with or WITHOUT pano load.
  A SingleThreadedExecutor hits 50.0 Hz exactly.
- **Fix:** fast path (state timer + /cmd_vel sub) lives on its own NODE;
  `HabitatSysnavBridge.spin_in_background()` runs one SingleThreadedExecutor thread
  per node.
- **Lesson:** in rclpy, callback-rate problems are as likely the EXECUTOR as the
  callback. Benchmark the empty loop before blaming the workload, and prefer one
  STE per rate-critical node over one MTE for everything.

## Case 5 — habitat equirect DEPTH is cubemap-face z-depth, not ray distance (2026-06, N2)

- **Symptom:** the nav stack saw obstacles EVERYWHERE in the house world —
  /free_paths width 0, robot frozen — while terrain_map data flowed normally.
- **Why hidden:** the cloud had passed M4 acceptance ("plausible world cloud",
  real SysNav objects out of it). The warp is smooth and looks right to the eye;
  only quantitative flatness checks exposed a ±14 cm ripple on a FLAT floor.
  (A first diagnosis bug compounded it: reading terrain_map intensity at the
  wrong byte offset — PCL PointXYZI is 32-byte with intensity at 16 — produced
  a fake "all 1.0" lead. Parse PointCloud2 by field offsets, never by column.)
- **Root cause (discriminating experiment):** rendering the EMPTY stage and
  comparing against the analytic ray-to-floor distance: every downward angle
  returned the constant camera-to-floor HEIGHT (1.319) instead of
  height/sin(elevation). habitat's EquirectangularSensor depth is the
  perpendicular z-depth of whichever CUBEMAP FACE the ray samples.
- **Fix:** `unproject_equirect_depth` converts euclidean = face_depth / |û·n̂|
  (the largest axis component of the unit ray in the sensor frame); pinned by
  a face-seam (45°, ×√2) and face-center (×1) test pair.
- **Lesson:** never trust a depth convention without an analytic ground-truth
  check (flat floor + trigonometry is free); "plausible-looking" point clouds
  hide smooth systematic warps that downstream consumers turn into hard
  failures.

## Case 6 — G1 body half-sunk in the floor: habitat re-centers render assets (2026-06-12)

- **Symptom:** the robot looked TINY/misplaced next to furniture in the chase
  view (~0.4 m apparent vs 1.32 m expected) — everything pointed at GLB
  unit/scale or camera FOV.
- **Why hidden:** the GLB itself was perfect (trimesh: 1.32 m tall, feet at
  y=0, scale 1:1) and N3 acceptance only checked "body occupies pixels", not
  WHERE. A half-sunk torso still occupies pixels.
- **Root cause (discriminating experiment):** add the object in an empty stage
  at translation (0,0,0) and read `root_scene_node.cumulative_bb`: min.y =
  −0.661. habitat's object template loader re-centers a render asset to its
  bounding-box CENTER — `translation` is the body's middle, not its feet, so
  gluing translation.y to the navmesh floor sinks the body half its height.
- **Fix:** measure once at body creation (`_body_y_off = -bb.min.y`, the bb is
  translation-independent) and lift `_place_body` by it.
- **Lesson:** for any engine, verify WHERE an asset's origin lands after
  import (place at origin, read the world bb) before trusting position math;
  "renders and moves" does not mean "stands on the floor".

## Case 7 — replan ghosts: the vocab taught actions nothing could execute (2026-06-12)

- **Symptom:** owner's `走到sofa` failed with "navigate_to requires a label OR
  numeric x and y" — which points at the LLM binding params wrong. The scripted
  harness passed the same goal.
- **Why hidden:** the FIRST decompose was always correct (label bound, loud
  object-not-found error). The garbage only appeared in REPLAN cascades, which
  the harness never exercised with an EMPTY world model: replans (1) chased a
  `scan_360` route taught by the derived vocab on `has_base` alone — but
  NOTHING in production ever calls `init_primitives`, so every base primitive
  raises 'No hardware connected'; (2) re-emitted navigate_to with the previous
  step's `{"label": "sofa"}` dropped.
- **Root cause:** teaching/routing decisions keyed on a static capability flag
  (`has_base`) instead of the EXECUTABLE truth (is the primitive layer wired?),
  plus no kernel guarantee that replanned steps keep prior param bindings.
- **Fix:** `primitives_ready()` gates vocab teaching, all StrategySelector
  primitive routes, and the engine preflight (registry-less legacy selectors
  byte-identical); `VGGHarness._inherit_replan_params` carries prior bindings
  into replanned same-strategy steps; an empty-world-model object goal now
  names the actual fix (start sysnav).
- **Lesson:** an action space must be derived from what can EXECUTE, not what
  the embodiment theoretically supports — and replans must never degrade
  params the previous plan had already bound. Test failure paths with the
  world in its EMPTIEST state, not just the happy fixture.

## Case 8 — "robot didn't walk": real motion, invisible in one blink (2026-06-12)

- **Symptom:** owner's `往前走` reported PASS in 0.1 s but the robot looked
  stationary — everything pointed at the motion pipeline (params, skill,
  base) being broken.
- **Why hidden:** the motion WAS real: params were correct and odom moved a
  full metre — every kernel-side check (verify, position delta, harness
  asserts) passed, because they all measure STATE, not its time course. The
  server's `walk` op integrated all kinematic steps back-to-back with no
  wall-time pacing, and the chase camera is agent-mounted, so a 1 m teleport
  moves the camera WITH the robot — between two frames almost nothing in the
  image changes.
- **Root cause:** `duration` was treated as an integration variable, not a
  wall-clock contract; nothing in the acceptance ever asserted elapsed time.
- **Fix:** `walk` paces each step to `duration` wall time; oracle
  `navigate_to` paces at `speed` m/s and emits viewer frames per step. Live
  check now asserts WALL TIME (1 m ≈ 3.3 s), not just displacement.
- **Lesson:** for any embodied action, verify the TRAJECTORY's time course,
  not just the end state — and remember that a tracking camera hides exactly
  the motion it tracks.

## Case 9 — "verification failed" that was really a silent enum coercion (2026-06-12)

- **Symptom:** GUI test `往左走一米` failed with "verification failed" after a
  full 3.3 s walk — pointing at the verify predicate or the navigation stack.
- **Why hidden:** the walk SUCCEEDED (motion evidence and all). The planner
  had emitted a direction value outside the enum ('左'); `_DIRECTION_MAP.get(
  direction, (1.0, 0.0))` silently coerced it to FORWARD, so the robot walked
  a metre the wrong way and only the position predicate caught it — two
  layers from the cause. The schema had also stripped enum/default, so the
  LLM never saw the legal value set it was violating.
- **Fix:** skills validate enum values and fail loud with the legal set
  (bad_params); zh aliases resolve natively; skill_wrapper passes
  enum/default through to the LLM schema.
- **Lesson:** `dict.get(key, default)` on an LLM-supplied enum is a silent
  contract rewrite — validate at the boundary and let the schema TEACH the
  legal set, or the failure surfaces two layers away dressed as something else.

## Case 10 — "protocol desync" that was really a cross-thread render killing the process (2026-06-12)

- **Symptom:** GUI session: navigate fails instantly with "bridge dead after
  protocol desync — restart the habitat world". Points hard at the (brand
  new) rid-pairing protocol code.
- **Why hidden:** three layers of misdirection. (1) The dead-GATE message
  said "desync" for EVERY death cause — the actual flip was the EOF branch
  (server closed the connection). (2) The server had died QUIETLY between
  turns (stderr → DEVNULL), minutes after the action that doomed it. (3) The
  doomed action was the previous round's #16 ticketed-navigate worker calling
  `_sync_agent()`/`emit_frame()` — habitat_sim agent/sensor access is
  OP-THREAD-ONLY (emit_frame's own docstring says so), and violating the
  confinement crashes natively, off-schedule, with no Python trace.
- **Fix:** the nav worker drives with `allow_render=False` (pathfinder math
  only — `try_step` concurrent with renders IS spike-verified safe); the op
  thread animates on each `navigate_status` poll (~4 FPS); the dead gate now
  reports its actual cause (`_dead_reason`).
- **Lesson:** when moving work onto a new thread, grep the moved code for
  thread-confinement contracts FIRST — the crash from violating one lands
  later, in another component, wearing that component's error message. And
  never let a failure gate flatten distinct causes into one string.

## Case 11 — a backfill that "didn't fire" had fired and returned poison (2026-06-12)

- **Symptom:** GUI: `走到厨房` fails with "navigate_to requires a label OR
  numeric x and y" although the plan's verify says `visited('kitchen')` and a
  deterministic backfill exists precisely to copy that label into params.
  Every reading says "the backfill didn't run on this path".
- **Why hidden:** it RAN — and returned the params unchanged-but-poisoned.
  deepseek-chat emits every schema key with null (`{"label": null}`), and the
  repair used `setdefault`, which keeps an existing key whatever its value;
  the coord guard used `"x" in params`, which grades null coords as bound.
  Reproducing the parse path with `{}` (the "obvious" empty case) PASSES —
  only the null-valued shape fails, and nothing logs the difference.
- **Fix:** null/"" values are stripped at the decomposer parse seam (a null
  value IS a missing param; the whole pipeline now sees honest missing-ness),
  and the backfill's own checks treat null/"" as missing.
- **Lesson:** "key present" and "value bound" are different predicates —
  `setdefault`/`in` encode the first and silently lie about the second. When
  an LLM fills a schema, normalize null-shaped output at the FIRST seam, and
  when a repro with the obvious input passes, reproduce with the model's
  ACTUAL output shape before concluding "didn't fire".

## Case 12 — torn cross-thread read of MuJoCo state under the GIL (2026-06-12)

- **Symptom:** none yet — caught by an adversarial workflow review of brand-new
  threading code before it shipped a bug. G1MuJoCoBase runs mj_step in a
  background control thread; get_position/get_odometry read self._data.qpos
  from the main thread with no lock.
- **Why hidden:** the GIL gives a FALSE sense of safety. `mujoco.mj_step` is a
  single native C call that writes the whole 37-float qpos array; the GIL does
  NOT interrupt it, but it also does NOT make a Python-side `self._data.qpos[0]`
  read atomic with respect to the C writes — a reader can observe qpos[0] from
  step N+1 and qpos[1] from step N (a pose that never physically existed). A
  denormalized quaternion is the tell.
- **Fix:** the control thread publishes a consistent (qpos[:7], qvel[:6]) COPY
  under a short lock once per policy batch; readers consume only that snapshot,
  never self._data. Lower contention than locking the whole mj_step, and it
  also closes a TOCTOU between the connected-check and the read (one locked
  region). Regression: a reader thread hammering get_odometry during a walk
  asserts every quaternion stays unit-norm.
- **Lesson:** "the GIL protects me" is wrong for C-extension state. Any object a
  C call mutates in bulk (numpy-backed sim state, ctypes buffers) needs explicit
  synchronization or a snapshot hand-off — the GIL only serializes BYTECODE.

## Case 13 — passive viewer's render thread starves the background control thread (2026-06-13)

- **Symptom:** owner opened the live G1 window via `vector-cli --scenario
  g1_flat`, said "渲染很卡" (very choppy) and hypothesized "仿真度太高" (sim
  fidelity too high). The gait also under-walked (`往前走` moved 0.30m of 1.0m
  → walk_skill FAIL).
- **Why hidden:** the symptom pointed straight at PHYSICS fidelity, the natural
  culprit for a heavy sim. But a headless bench proved the opposite: physics
  runs at **26x real-time** (step batch 0.76ms) — fidelity is nowhere near the
  bottleneck. Throttling `viewer.sync()` (5-8ms each) didn't help either: even
  at ~1 sync/sec the loop capped at 0.4x. The real cause: `mujoco.viewer.
  launch_passive` spawns an INTERNAL render thread; when the gait control loop
  runs on a *background daemon* thread (same scheduler priority), the render
  thread starves it to ~0.4x. The discriminator: the SAME loop on the MAIN
  thread holds 1.0x. Daemon-without-viewer = 1.0x; daemon-WITH-viewer = 0.4x.
- **Fix:** PUMP mode. With a window open, run NO daemon — the caller thread
  (REPL/skill) drives `_step_batch()` via `_advance()` during walk/turn/
  navigate, so the gait steps on the viewer-friendly main thread at 1.0x (GUI
  walk 1.28m == headless 1.28m). Headless keeps the proven daemon path
  unchanged (all gait tests use gui=False). Mirrors viewer_mode.py's
  MAIN_THREAD_PUMP concept (which existed only for macOS/mjpython before).
- **Lesson:** when a perf symptom names the obvious heavy component ("sim too
  detailed"), bench that component in isolation FIRST. Here physics was 26x
  headroom and the real thief was thread scheduling against an opaque library
  render thread — invisible until you measure daemon-thread vs main-thread with
  the viewer attached. Also: a high system load (a runaway SysNav node eating
  24 cores) was simultaneously inflating the lag — always check `loadavg`
  before trusting a perf measurement.

## Case 14 — pump-mode _advance can sleep a negative duration (2026-06-13)

- **Symptom:** during a long PUMP-mode action (ExploreSkill driving many
  navigate legs in the live window), the step crashed with "sleep length must
  be non-negative". Daemon/headless mode never hit it.
- **Why hidden:** G1MuJoCoBase._advance(seconds) paces with
  `time.sleep(min(remaining - 0.003, deadline - time.monotonic()))`. Once the
  loop runs slightly past `deadline`, `deadline - now` is NEGATIVE; if
  `remaining > 0.004` still held that tick, the min() picked the negative value
  → time.sleep(negative) raises. Only the PUMP path (window open) uses this
  branch — every prior GUI run (R0/R3/R5) was short enough to never straddle
  the deadline mid-tick, so it lay dormant until explore's many legs hit it.
- **Fix:** clamp to non-negative — `time.sleep(max(0.0, min(...)))`.
- **Lesson:** any `time.sleep` fed by a `deadline - now()` difference must be
  clamped ≥ 0; the deadline can pass between the guard and the call. A latent
  pump-only bug needs a long pump-mode action to surface — short ones mask it.

## Case 15 — vision-seek target acquisition from spawn is FOV-fragile (2026-06-13, FIXED 2026-06-14)

- **Symptom:** standalone `找到并走到红色物体` (VisionSeekSkill from the spawn
  pose) intermittently reports seen=False — the robot scans for the full
  budget without ever recognising the target — while the SAME skill succeeds
  (a) headless and (b) inside ExploreAndSeekSkill (explore-then-seek).
- **Why hidden:** the pelvis camera is pitched down ~12° so floor-level targets
  stay in frame DURING approach; but a far target straight ahead at spawn sits
  at/above the FOV's upper edge. If it isn't detected on the first frames, the
  acquisition strategy is turn-in-place scan, which sweeps azimuth but never
  changes pitch/range — so a far target outside the vertical FOV is never
  brought in. The explore-then-seek capstone works because explore first
  repositions the robot to where the target is in view. A diagnostic confirms
  the camera + detector themselves are fine (red detected after a 0.5 m
  forward step), so this is an ACQUISITION-strategy gap, not a render/GL bug.
- **Partial fix applied:** scan now steps forward every _SCAN_ADVANCE_EVERY
  ticks (sweep + advance) — helps but heading drifts during the turn-scan so
  acquisition of a specific far target is still unreliable.
- **FIX (2026-06-14):** reduced the camera down-pitch ~12°->6° + widened
  fovy 60->70 so a FAR target at spawn range sits near frame centre (not
  bottom-clipped) and its pixel area clears the detector; lowered
  detect_targets min_area_frac 0.004->0.0025 for margin. Arrival still works
  (progress-stall, not close-up detection). GUI cold-start '找到并走到红色物体'
  now PASSes at_position(3.7,0,1.6) in 9 s (acquires at spawn, no long scan).
- **Lesson:** a perceive-act loop's ARRIVAL was tuned hard, but ACQUISITION
  (first detection) was assumed — re-test the cold-start (target not initially
  in view), not just the mid-approach.

## Case 16 — slow perception stutters the gait → progress-stall fires metres short (2026-06-14)
- **Symptom:** VLM-seek "arrived" 9 s in, robot at 0.18 m, chair at 3.6 m. Verify
  (at_position 1.6) correctly FAILED. Looked like a bad arrival heuristic.
- **Cause:** the seek loop re-armed the walk deadman for 0.4 s (fine for the
  instant colour detector), but a real Qwen-VL call takes ~2 s. So the robot
  walked 0.4 s then STOOD STILL ~1.6 s every tick waiting for the VLM; over the
  8-tick progress-stall window it covered <0.6 m of genuine forward walking →
  "no net progress" → false arrival far from target.
- **Fix:** per-action deadman — FORWARD re-arms for step_duration (3.0 s for VLM,
  > the call latency, so motion stays continuous across the perceive); TURN/SCAN
  stay short (0.5 s). Colour seek unchanged (0.4 s, instant detector).
- **Lesson:** an actuation cadence tuned to an INSTANT sensor silently breaks when
  the sensor becomes SLOW — the deadman must outlast the perceive latency or the
  "stalled" signal measures the sensor's latency, not the robot's progress.

## Case 17 — a turn sized for forward over-rotates and flings the target off-screen (2026-06-14)
- **Symptom:** after Case 16's fix, VLM-seek detected the chair once (x_norm -0.2)
  then never again — robot spun in place and wandered off.
- **Cause:** the same long step_duration (3.0 s) was applied to TURN. A 3 s turn at
  0.5 rad/s = ~86° rotation to correct a ~10° bearing error — massive overshoot →
  target leaves the FOV → "not seen" → scan keeps turning the same way → lost.
- **Fix:** decouple — only forward uses the long duration; turn/scan use a short
  _TURN_STEP (0.5 s ≈ 14°), an incremental correction, then re-perceive.
- **Lesson:** one "step size" for both translate and rotate is wrong when their
  natural scales differ; a correction step must be sized to the error it corrects.

## Case 18 — a noisy VLM bbox is not a range proxy (false arrival) nor reliable per-frame (lost) (2026-06-14)
- **Symptom (a):** VLM-seek "arrived" at spawn — a 0.046 bbox at 3.6 m exceeded the
  colour detector's 0.04 arrive-area. **Symptom (b):** detection dropped on ~half
  the frames even with the chair dead-ahead (gait sway / motion blur), so the robot
  kept losing it and spinning to re-search.
- **Cause:** a VLM grounding box is noisy in BOTH size (occasionally oversized at
  range) and presence (intermittent), unlike the clean colour blob the loop was
  built around.
- **Fix (a):** raise the VLM arrive_area to 0.55 (target must nearly fill the frame)
  and lean on the PHYSICAL progress-stall (collision-blocked at the object) for
  arrival. **Fix (b):** a last-bearing COAST — keep heading toward the last seen
  bearing for up to _COAST_MISSES (4) missed frames before falling back to a
  search-scan, so an intermittently-detected target dead-ahead is not abandoned.
- **Related:** Qwen sometimes wraps the JSON in an unclosed ```json fence or appends
  trailing garbage (`{...}']`); _parse_json_response now extracts the first balanced
  {...} object as a last resort so those frames still yield a detection.
- **Lesson:** when a learned detector replaces a hand-crafted one, its FAILURE
  SHAPES differ — design the control loop for "noisy + intermittent + loosely
  bounded", not for the clean signal the geometry detector gave you.

## Case 19 — a Go2-acquisition fix (forward-only stall gate) silently regressed g1 arrival (2026-06-14)
- **Symptom:** tuning the shared `vision_seek._seek_loop` to make Go2 acquire a
  far target (pre-acquisition forward-bias) + not false-arrive mid-turn (count
  progress-stall ONLY on forward ticks) fixed nothing for Go2 AND broke g1: g1
  vlm_seek went from arriving 0.50 m (R1) to stopping 1.83 m short.
- **Cause:** g1 arrives by ORBITING the target at close range (turn-dominated
  near the chair) and the all-tick progress-stall caught that as "arrived". The
  forward-only gate required 8 consecutive FORWARD ticks with no progress, which
  an orbiting robot rarely produces → g1 never stalled → never arrived.
- **Fix:** reverted both experiments; restored R1 all-tick stall + sweep-scan. g1
  back to 0.58 m PASS. Go2 arrival left to R3 (it needs closed-loop heading
  control for the gait yaw-drift, not a stall-gate tweak — a different problem).
- **Lesson:** before tuning a SHARED control loop for a new embodiment, pin the
  old embodiment's success with a regression run FIRST — a change that helps the
  new case can quietly break the proven one (their arrival signatures differ:
  g1 orbits, go2 yaw-drifts). Embodiment-specific control belongs behind an
  embodiment-specific hint, not baked into the shared path.

## Case 20 — Go2's furnished-room camera saw only floor: env geoms hidden + cam too low (2026-06-14)
- **Symptom:** the VLM returned nothing for Go2 in the furnished room; the d435
  frame was bare floor + a sky band, no furniture.
- **Two compounding causes:** (1) the shared room tags walls/furniture with
  ENV_GEOM_GROUP=3, which Go2's `get_camera_frame` (default render option) HID —
  the same group-visibility trap as g1 R9; (2) Go2's stock d435 is at ~0.2 m,
  42° FOV, pitched 5° DOWN — it frames the floor, not furniture at range.
- **Fix:** (1) enable all geom groups in the furnished-mode renderer (scoped to
  furnished so the apartment view is byte-identical); (2) mount a forward, ~level
  (pitched up 8°), 75° recognition camera on base_link in the furnished build,
  used by get_camera_frame in furnished mode. Go2 then grounds the chair.
- **Lesson:** porting a perception capability to a new robot inherits BOTH the
  old render-group gotcha AND the new robot's sensor geometry — a dog's-eye
  sensor designed for ground/obstacles is not a furniture-recognition view.

## Case 21 — VLM visual-servoing arrival is inherently flaky (both embodiments) (2026-06-14)
- **Symptom:** the SAME vlm_seek code reaches the chair on one run (g1 0.58 m PASS)
  and stalls 2+ m short on the next (g1 2.38 m, go2 1.3-2.3 m) — no code change
  between runs. Looked like a regression each time; it was run-to-run VARIANCE.
- **Cause:** closing a control loop on a ~2 s, noisy, intermittent VLM bearing
  while the gait drifts/sways is fundamentally under-sampled — the robot acts on
  stale/jittery bearings, curves, and the progress-stall (the only arrival signal
  when the box is too small/clipped to area-arrive) fires wherever forward motion
  happens to lull. A closed-loop heading P-controller (Go2) fixed the egregious
  backward-WANDER (it now approaches forward) but not the flaky last-metres.
- **Fix (direction, R3):** do NOT visually-servo the last metres. VLM RECOGNISES
  the object → estimate/return its location → drive there with a RELIABLE
  controller (navigate_to / nav-stack waypoint, which uses odometry + a planner,
  not pixel bearings). Recognition is the perception win; arrival is a navigation
  problem with a proven solution already in the repo (the apartment go2 nav stack).
- **Lesson:** a slow, noisy sensor can be a great DETECTOR and a terrible
  CONTROLLER. Use the VLM to decide WHAT/WHERE, use odometry+planner to decide
  HOW to get there. Don't put a 0.5 Hz noisy signal in a tight servo loop.

## Case 22 — recognise→navigate via lidar picks an OBSTACLE, not the recognised object (2026-06-14)
- **Context:** the Case-21 pivot — replace flaky VLM visual-servoing with VLM
  RECOGNISE (bearing) → LIDAR locate (range at bearing) → reliable navigate_to.
- **Symptom:** g1 "go to the chair" navigated to (1.9, 0.4) / (2.1, -1.0) — the grey
  OBSTACLES — not the chair at (3.6, 0).
- **Cause:** the lidar gives RANGE but no SEMANTICS. "Nearest hit in the recognised
  bearing" returns whatever surface is closest along that ray — an obstacle
  between the robot and the chair, not the chair. The VLM knows WHICH object
  (its bbox is on the chair); the lidar can't associate its return with that.
- **Also:** acquisition of a far/small target is ~50% per frame (VLM intermittent);
  holding heading + re-querying helps, but a blind forward-advance fallback can
  overshoot a never-detected target.
- **Fix (next round):** DEPTH-AT-BBOX — read the depth at the recognised bbox
  centre (the chair's own pixels) → distance to the CHAIR (semantic, skips
  intervening obstacles) → project to world → navigate_to. Needs a depth source
  co-registered with the recognition camera: g1 needs a depth render on HEAD_CAM
  (it has none); go2 HAS get_depth_frame but lacks navigate_to. Landing the pivot
  reliably = give ONE embodiment BOTH (depth-at-bbox + navigate_to).
- **Lesson:** fusing a semantic detector with a geometric ranger needs the range
  sampled AT the detection (same pixels/bbox), not "nearest thing in that
  direction" — otherwise the geometry layer silently re-targets to clutter.

## Case 23 — depth-at-bbox median sees the wall BEHIND a thin object (2026-06-14)
- **Context:** R5 recognise→navigate locates the object by depth at the recognised
  bbox (the semantic fix for Case 22). First cut used the bbox-window MEDIAN depth.
- **Symptom:** the located point landed PAST the chair (~4.0-4.5 m vs chair 3.6 m),
  sometimes at the front wall — and a wall-adjacent goal is unreachable to the
  planner (inflation → no path → navigate_to returns remaining=inf).
- **Cause:** a thin object (chair legs/back) only partly fills its bbox; the gaps
  see THROUGH to the wall behind, so the window median is biased toward the far
  surface.
- **Fixes:** (1) use the NEAREST surface (20th-percentile depth) in the window —
  the object's own front face, not the background; (2) navigate to a STANDOFF
  point 0.7 m IN FRONT of the located surface, never onto it (the surface can sit
  in the planner's inflated wall/object zone). 3/3 reliable headless arrival after.
- **Lesson:** when fusing a 2-D detection box with a depth map, the box spans
  object AND background — take the NEAR depth, and treat the result as a surface
  to stand off from, not a waypoint to occupy.

## Case 24 — a PUMP-mode base booted on a worker thread is never pumped (frozen) (2026-06-14)
- **Symptom:** g1 launched via `--scenario` walks in the GUI; the SAME g1 launched
  in-REPL via `start_simulation(sim_type='g1')` boots "ok" but the gait is frozen
  and the viewer never updates (empty/static), the boot returns in ~1.7s.
- **Cause:** G1's PUMP mode (chosen when a viewer is open) runs NO control daemon —
  the gait advances only when the CALLER thread drives `_advance`/`_step_batch`.
  The `--scenario` path boots g1 on the main REPL thread, which then pumps it; the
  `start_simulation` tool runs on an engine WORKER thread, so after the tool
  returns nobody pumps that base → physics never steps → frozen + no viewer sync.
- **Fix:** `prefer_daemon` — when booted mid-REPL, force DAEMON mode even with a
  viewer: a dedicated control thread drives the gait and its `_step_batch` syncs
  the viewer (thread-agnostic; the proven headless-daemon path + viewer sync;
  ~0.4x under the passive viewer's render thread per Case 13, but it walks +
  renders). The startup `--scenario` path keeps PUMP (1.0x, main-thread driven).
- **Lesson:** a "caller-thread-pumped" execution model only works if the caller
  is a long-lived loop. Code launched by a one-shot tool call on a worker thread
  needs its OWN driver thread — don't assume someone will keep pumping it.

## c11 R10 — a swallowed VLM 429 reads as "object absent" + blind-walks (silent)
- **Symptom:** under upstream qwen-VL rate-limiting, `recognize_navigate` reported
  `not_found` and the robot wandered for the full `max_iters` — pointing AT the
  object/recognition, AWAY from the real cause (the endpoint was throttled).
- **Cause:** `_call_vlm` raised a generic `RuntimeError` on 429; `detect_targets`'
  broad `except Exception` swallowed it to `[]` — indistinguishable from "VLM saw
  nothing". The seek loop treated each 429 as a miss → after `_MISS_BEFORE_SWEEP`
  it WALKED, hammering the throttled endpoint with no backoff, then lied `not_found`.
- **Fix:** typed `VlmRateLimitError(RuntimeError)` (429-only; 529 keeps its 5xx
  retry path) + opt-in `detect_targets(raise_on_rate_limit=True)` re-raise +
  `recognize_navigate` consecutive-`rl_streak` cap (3, with backoff) → honest
  `diagnosis="vlm_unavailable"`, ZERO walks. The post-streak `continue` is
  load-bearing (without it a 429 falls into the miss branch and walks).
- **Lesson (module identity):** resolve the exception class from the LIVE module
  (`_vlm_go2.VlmRateLimitError`), not a cached `from … import` binding —
  `importlib.reload(vlm_go2)` (a wiring test does this) rebinds the class via
  shared module globals, so a stale binding silently fails `isinstance`/`except`
  and swallows a real 429. A typed exception only helps if both sides see the
  SAME class object.

## c11 R11 — "switch to go2 runs bash" blamed the wrong layer
- **Symptom (M5 R9):** NL "switch to go2" made the LLM run
  `bash("./scripts/launch_explore.sh &")` instead of `switch_embodiment`. STATUS
  hypothesized "the answer/ReAct path offers the planner raw bash / doesn't honor
  the intent category filter."
- **Reality (workflow wsv6u3ddo, empirically reproduced):** the engine answer
  path DOES honor the filter — `route('切到go2'/'switch to go2')=['robot','sim',
  'system']` excludes bash (category `code`) and offers `switch_embodiment`. The
  bash symptom had THREE unrelated causes: (1) Chinese `切到/切换到X` contains `到`,
  a `_MOTOR_PATTERN`, so `should_use_vgg`→True → the VGG decompose path where a
  `@tool` (not `@skill`) is unreachable (decompose vocab is SKILL-registry-only,
  rule 3); (2) `ROBOT_TOOL_INSTRUCTIONS` literally taught bash+launch_explore.sh
  for Go2 and never named `switch_embodiment`, so the LLM improvised shell from
  persona text even when bash was filtered out of the schema; (3) the M5 harness
  registered every tool with the default category → route filter returned ZERO
  tools → switch_embodiment unavailable → fell back to the prompt's bash text.
- **Fix:** route switch intents to the tool_use path (`_SYSTEM_BYPASS` += switch
  markers, before the motor check) + remove the persona's bash directive. Do NOT
  add `switch_embodiment` as a `@skill` (rule-3 split-brain) and do NOT disable
  bash for the planner (no operator-vs-planner split → would break dev bash).
- **Lesson:** before "fix" a path, REPRODUCE the route empirically. The cheap
  keyword router returning `None` (= all tools) for unanticipated phrasings, plus
  a persona that teaches the wrong tool, looked exactly like "the filter is
  broken." It wasn't — three separate layers each contributed. A symptom that
  names a mechanism ("offers raw bash") is a hypothesis, not a diagnosis.
