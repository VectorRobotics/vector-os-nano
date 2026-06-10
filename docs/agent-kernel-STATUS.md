# Verified Agent Kernel — STATUS (resume anchor)

One-page "where are we / what's next". Read this first when resuming; durable design is
[ARCHITECTURE.md](ARCHITECTURE.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md).

- Branch: `feat/playground-vln` (off `master` @ PR #13 merge; gait-fix branch merged as PR #13).
  Campaign: third world — photoreal/semantic nav world (VLN-first, SysNav revival rides along);
  M0 DONE: ADR-009 (Proposed) recommends pinned habitat-sim 0.3.3 conda-py3.9 subprocess +
  bridge — spike-verified on the Linux box (photoreal RGB, equirect pano+depth, 819K-pt
  unprojected cloud, GT odom). GATED on owner approval (new external dependency) + HM3D
  agreement; interim scenes are license-free (test-scenes/HSSD/Replica).
  M1 DONE (backend-agnostic, ungated): `Scenario` carries additive `sim_backend`/`scene_ref`;
  a non-MJCF world registers/resolves through `WorldRegistry` with the engine surface intact
  (tests/vcli/test_scenario_backend_seam.py). M2 (the habitat world itself) awaits the gate.
- Last updated: 2026-06-10.
- Scope guard: this is **vector-os-nano only** — not the UniLab go2arm-grasp work.

## Current state (2026-06-09)

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
- **This branch:** engine sync-exec gated on mjpython only — Linux REPL responsive (`fcc6b20`) ·
  go2 gated diagnostics + `get_sim_time()` (`6a39f6c`) · venv reconcile `.venv-nano`→`.venv`
  (`13a9429`) · **the gait sim-dt fix** (`d7e158b`) · docs: tricky-bugs casebook + STATUS condense
  (this commit).

## OPEN — prioritized backlog

1. **Phase E Wave 2 (resume point): W2.1 mostly SHIPPED 2026-06-10** — `vcli/run_registry.py`
   (frozen RunEntry, atomic JSON under `~/.vector-os-nano/runs/`, dead-PID sweep),
   `vcli/daemon.py` (double-fork daemonize smoke-tested end-to-end on Linux, SIGTERM→SIGKILL
   `stop_run`, `tail_log`), CLI `--status/--stop/--log` early-exit ops (live-verified).
   REMAINING: `--daemon` CLI wiring needs a goal-runner design (what runs headless — a single
   NL goal through `run_turn_unified`?) — do together with **W2.2 (RUN_ID watchdog orphan
   sweep + /tmp flag retirement; coordinate with any parallel GUI session)**. macOS daemonize
   is owner-unverified. Then Wave 3. (W2.3 SHIPPED — see Shipped.)
   Plan: [agent-kernel-phase-e-plan.md](agent-kernel-phase-e-plan.md).
2. **Named-vs-generic grasp — DONE (2026-06-10).** (a) target-aware `holding_object()` +
   `picked_object` shipped earlier; (b) shipped now: `verify_strengthen.strengthen_target_verify`
   rewrites a bare `holding_object()` to `holding_object('<target>')` whenever the step's params
   bind a named target — applied at BOTH plan chokepoints (`_validate_sub_goal` for every LLM
   decompose/replan; the engine 1-step fast path, whose pick verify is no longer the `True`
   sentinel). Structural/stricter-only; `${...}` refs and `__` skipped. Generic grabs keep
   unbound→nearest + bare form.
3. **Detect-verify close-the-loop — DONE (2026-06-10).** Kernel `step_output(path)` is injected
   per-evaluation (executor binds it to the step's own `exec_output`; `GoalVerifier.evaluate`
   gained an `extra_ns` overlay — sandbox unchanged, never persists). `DetectSkill.verify_hint`
   is now `len(step_output('objects')) > 0` (was the false-passing query-less oracle call);
   `step_output` is unioned into every world's decompose allowlist (kernel-provided, like
   `answer`). Verify now consumes the step's own alias-aware observation (Rule 4).
4. **Stage 3 grounding remainder:** VLM `MuJoCoPerception`/`DetectSkill` perception path,
   referring-expression resolution ("the red cup" → object_id), ObjectMemory re-sync (=W2.3).
   **Phase C.3/C.4** (real specialized model in the robot world) stays blocked behind this
   ([agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md)).
5. **Owner-gated window checks (cannot verify headless):** R2-1 `--sim-go2` macOS in-process
   window (opens, no segfault, walk animates; frozen-when-idle is expected v1) · R2-4 single ^C
   aborts to prompt under mjpython, second ^C exits.
6. **Chores:** pin/vendor `convex_mpc` in `pyproject.toml` (a venv rebuild silently loses the
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
   at-exit GL-teardown segfault was also observed once (suite results were complete).

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

## Autonomous /loop prompt (the standing mission for owner-away iterations)

Run via `/loop <this prompt>` (no interval => self-paced). Mission-oriented + high-autonomy: each firing
advances the mission as far as it safely can, not a single tiny edit.

> **Mission: advance vector-os-nano toward a generalizable PHYSICAL agent for robots.** Iterate autonomously
> (owner away; auto-approve on; current work branch; ONLY this project, never UniLab). This is
> a mission, not a checklist: make natural language truly control a robot through a grounded CLOSED loop
> (understand -> decompose -> plan -> execute -> verify -> replan -> recover), generalizing across embodiments
> (arm, go2, future) AND tasks. Simulation is a MEANS; the end is a physical robot agent. Push the LLM through
> the whole cognitive layer (language, decomposition, planning, strategy/verify selection, recovery); keep
> grounding/verify/safety DETERMINISTIC — verify is the moat, never LLM-graded. Prefer fixes that remove an
> embodiment asymmetry or generalize a mechanism over one-off patches.
>
> Each iteration, ORIENT then act with judgment — you have wide latitude:
> - ORIENT: read `docs/agent-kernel-STATUS.md` (OPEN backlog + current state), `docs/ARCHITECTURE.md`,
>   `docs/tricky-bugs.md`, and memories `vector-os-nano-live-hardening` / `-language-layer` /
>   `workflow-model-tiering`. Optionally run the real cli + deepseek to feel current state and discover issues.
> - CHOOSE a meaningful objective — a bug class, a capability, an architectural improvement — that moves the
>   mission forward. You MAY pursue a FARTHER goal across several workflows/edits in one iteration; don't
>   artificially stop at one tiny change. Decompose it yourself and advance as far as you safely can.
> - BUILD: reproduce/diagnose first, then implement via focused dynamic **Workflows** (implement -> 2-3
>   adversarial reviewers -> critic), chaining as many as the objective needs. Pin agent models per
>   `workflow-model-tiering`. Write/extend tests for logic that matters; add evals where output quality matters.
> - VERIFY HONESTLY: keep the canonical suite green (`.venv/bin/python -m pytest tests/vcli
>   tests/unit/vcli -q`); validate behavior headless with the real cli + deepseek wherever possible. Some
>   things only reproduce in the owner's mjpython window (GUI render, real-time timing, Ctrl-C under mjpython)
>   — reason carefully, add what headless coverage you can, and CLEARLY hand the visual/timing confirmation to
>   the owner. Never claim a GUI-visual works unverified.
> - COMMIT + RECORD: self-review the real diff; green-then-commit in isolated, logically-scoped commits,
>   updating STATUS (+ ARCHITECTURE if structure/contracts changed; + tricky-bugs.md if a hidden bug was
>   cracked) and the relevant memory in the SAME commit (Doc Governance). Record what you did + what's next so
>   the next iteration resumes cleanly. **Do NOT push.**
>   `git checkout mjcf/go2/scene_room_piper.xml` if a go2 test dirtied it.
>
> DON'T interrupt the owner to ask — make reasonable decisions and proceed. Only stop/surface on a GENUINE
> blocker: the canonical suite goes red and you can't get it green (halt-on-red; salvage + commit partial
> green), something needs the owner's GUI/hardware confirmation, or an action would be destructive /
> irreversible / outward-facing (push, deploy, delete owner data). Otherwise keep advancing the mission,
> iteration after iteration, then schedule the next one.
>
> Current backlog: the **OPEN — prioritized backlog** section above (advance any; not exhaustive — discover more).
