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
