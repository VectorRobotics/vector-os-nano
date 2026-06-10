# Verified Agent Kernel — Phase E Plan (dimos-informed hardening + substrate)

- Status: **IN PROGRESS.** Wave 1 complete (W1.1–W1.4) + W2.4 shipped on
  `feat/verified-agent-kernel` (merged via PR #11/#12); W2.3 shipped 2026-06-10 on
  `feat/playground-vln`. OPEN: W2.1 + W2.2 (pair — process-lifecycle coordination), Wave 3.
  Each item is independently shippable and keeps the canonical suite green.
- Date: 2026-06-08 (status updated 2026-06-10)
- Branch: `feat/verified-agent-kernel` (HEAD `25ef2ba`; canonical `tests/vcli tests/unit/vcli`
  baseline 863 passed / 4 skipped)
- Related: [ARCHITECTURE.md](ARCHITECTURE.md), [agent-kernel-STATUS.md](agent-kernel-STATUS.md),
  [agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md),
  [architecture-decisions/ADR-006-agent-kernel-world-plugin.md](architecture-decisions/ADR-006-agent-kernel-world-plugin.md),
  [architecture-decisions/ADR-007-closed-loop-controller.md](architecture-decisions/ADR-007-closed-loop-controller.md)
- Note: file:line references are against HEAD `25ef2ba`; treat as **approximate search hints**
  (symbol names are durable, line numbers rot). Stage 5 / Phase C remain the primary lines of
  work — this phase is a parallel hardening + substrate strand, sequenced to not collide.

## Why this phase exists

This phase is the actionable output of a source-level comparison between vector-os-nano (von)
and **dimos** (dimensionalOS/dimos), a much larger horizontal robotics SDK. The two solve
different halves of the robot-agent problem:

- **dimos = substrate breadth.** A ROS-replacement transport-agnostic pub/sub + RPC fabric,
  declarative whole-stack composition (Blueprints/autoconnect), a native no-ROS message
  ecosystem, a multiprocess module runtime with a real daemon/run/stop/status lifecycle, a
  full local model zoo, multi-sim and multi-embodiment breadth. Its agent is a **stock
  LangGraph ReAct loop** with **no per-step verification, no goal graph, no replan controller**
  — its only success signal is "the call returned without an exception before timeout".
- **von = reasoning depth.** A narrow single-process verified agentic kernel where the loop
  IS the product: NL decomposes into a frozen GoalTree, each step carries a deterministic
  AST-sandboxed verify predicate behind an evidence gate, with observation-driven replan, a
  measured strategy selector, and verified-trace template compilation.

The decisive asymmetry is verification: **dimos has no concept of "did the step achieve its
goal"; von does.** That is von's moat and must not be diluted. This phase therefore borrows
from dimos ONLY where it (a) hardens the verified loop, or (b) retires named von debt without
touching verify — and explicitly rejects the parts that would dilute the moat (see Anti-goals).

It also closes one gap that is purely internal to von and unrelated to dimos but surfaced by
the comparison: the learning loop (strategy bandit + template compilation) currently learns
from `step.success`, not from the deterministic evidence gate — so a VLM `visual_override`
"success" can train the bandit and compile a "verified" template. Closing that (W1.1) is the
single highest-ROI item.

## Guardrails (apply to every item)

1. North-star fit: strengthen the verified closed loop or fix named debt; never weaken the
   deterministic-verification moat, never widen the AST sandbox, never eval/exec model output.
2. Additive + green-bar: every increment is independently shippable, defaults to no-op where it
   injects a new seam, and keeps `tests/vcli tests/unit/vcli` green (baseline 863/4) with its
   own regressions.
3. Kernel stays thin: do NOT import dimos's distributed substrate wholesale. Borrow narrow
   patterns, not the LCM bus / forkserver pool / pickle-over-the-wire RPC.
4. Doc Governance: any item that changes structure/contracts updates `ARCHITECTURE.md` in the
   same commit; STATUS updates on every work commit.

## Anti-goals (deliberately NOT borrowed from dimos)

- The verification-free ReAct/MCP agent loop (`agents/mcp/mcp_client.py`, stock LangGraph
  `create_agent`) — it IS von's non-verified fallback (`should_use_vgg` -> `run_turn`).
- Implicit stream-name+type auto-wiring (a typo silently fails to connect) — keep von's
  explicit, machine-checkable `${step.output.path}` Blackboard binding.
- The distributed substrate wholesale (LCM bus, forkserver workers, Actor/RPCClient proxies,
  importlib hot-reload, ~2.6K LOC) — borrow only the narrow daemon/run-registry/watchdog.
- pickle-as-default-wire / `pickle.loads` of untrusted bytes — conflicts with von's
  no-eval/exec posture. Use a typed schema (Pydantic/JSON) if cross-process state is ever needed.
- Per-message-thread backpressure and similarity-only embedding recall with no predicate gate.

## Overview

| # | Item | Wave | Effort | Impact | Depends on |
|---|---|---|---|---|---|
| W1.1 | Gate bandit reward + template compilation on `evidence_passed`, not `step.success` | 1 | S | high | reuse existing `_evidence_ok` |
| W1.2 | Fail-loud world-registration pre-flight validator | 1 | S | med | — |
| W1.3 | Decouple `scene_graph` provider behind a `TextLLM` adapter | 1 | S | med | — |
| W1.4 | Wire `playground.build_step_primitives` into the live executor (or delete) | 1 | M | med | — |
| W2.1 | Daemon + on-disk run-registry + `vector status/stop/log` lifecycle | 2 | M | high | — |
| W2.2 | RUN_ID watchdog orphan sweep + retire `/tmp` flag-file IPC | 2 | L | med | W2.1 |
| W2.3 | ObjectMemory re-query-freshest read (kill sync-once-stale) | 2 | M | med | — |
| W2.4 | Typed `failure_class` into the replan context | 2 | M | med | — |
| W3.1 | Frozen `WorldBlueprint` value object the engine consumes (additive) | 3 | M | med | Stage 5 stable |
| W3.2 | Typed lazy Capability factory + bridge dimos's local model zoo | 3 | M-L | high (strategic) | W3.1 |
| W3.3 | Protocol-compliance to kill getattr-by-string on agent internals | 3 | L | med | — |

Cadence: ship all of Wave 1 first (small, independent, direct moat payoff); then W2.1+W2.2 as a
pair and W2.3/W2.4 independently; do Wave 3 after Stage 5 stabilizes the kernel/world seam.

---

## Wave 1 — moat hardening (do first)

### W1.1 — Gate bandit reward and template compilation on the evidence gate

**Goal.** "Verified" must mean verified through the learning loop too: non-deterministic
outcomes (VLM `visual_override`, `verify="True"` sentinels) must NOT train the strategy bandit
or compile into reusable templates.

**Current state (verified at HEAD).**
- The evidence gate already exists: `trace_store.evidence_passed(trace, is_robot=False)`
  (`vcli/cognitive/trace_store.py`, def ~215) correctly excludes `StepRecord.visual_override`
  and `answer_only`-bound sentinels; the engine already wraps it as `_evidence_ok(trace)`
  (`vcli/engine.py`, def ~1557) and uses it for the unified-turn `verified` flag (~1520/1547).
- But the learning tier bypasses it: the bandit records `success=step.success`
  (`vcli/cognitive/vgg_harness.py` ~443-446); the compiler filters on `t.success`
  (`vcli/cognitive/experience_compiler.py`, `compile` ~198); `_maybe_compile_experience` gates
  on `trace.success` (`vcli/engine.py` ~1045-1058).
- `StepRecord` fields (frozen, additive): `success`, `verify_result`, `visual_override=False`,
  `result_data`, `error` (`vcli/cognitive/types.py`). `StrategyStats.record(strategy_name,
  sub_goal_name, success, duration_sec)`.

**Design.**
- Distinguish granularity: the bandit records **per-step** (needs a per-step evidence notion);
  the compiler is **per-trace** (use the existing trace-level `evidence_passed`).
- Add `trace_store.step_evidence_ok(step, sub_goal, is_robot) -> bool` reusing the same
  internal sentinel + `_is_answer_only` logic as `evidence_passed`
  (`step.verify_result and not step.visual_override and not _is_sentinel(sub_goal.verify)
  and not _is_answer_only(sub_goal)`).
- `vgg_harness` ~443: pass `success=step_evidence_ok(step, sub_goal, is_robot)` to
  `_stats.record(...)`. **Control flow is unchanged** — the retry/continue gates (~452, ~521)
  keep using `step.success` (whether to retry is an execution question, not an evidence
  question).
- `_maybe_compile_experience` (engine ~1045): gate on `self._evidence_ok(trace)` so only
  evidence-backed traces enter `_successful_traces`; keep the compiler's `success` filter as
  defense-in-depth (or pass an `evidence_fn` into `compile`).

**Critical design risk (load-bearing).** `evidence_passed` is **lenient for the robot world**
(async motor skills legitimately use `verify="True"` sentinels; `is_robot=True` relaxes the
gate). A strict per-step evidence reward would record almost no evidence on the robot world and
**starve robot learning**. `step_evidence_ok` MUST mirror the same `is_robot` leniency — this
change is materially meaningful only for dev/playground worlds (which carry real predicates);
robot behavior stays as-is.

**Test.** (1) a `visual_override` trace is recorded as not-evidence in stats and not compiled;
(2) a `verify="True"` sentinel step earns no bandit reward; (3) a real-predicate pass still
earns reward and still compiles; (4) robot-world leniency is preserved (no learning starvation).
Reuse existing trace_store tests; add the four cases above.

**Acceptance.** Bandit rankings and the template library are no longer trainable by a VLM
override or a sentinel verify; dev/playground suite green; robot learning unaffected.

**Effort S / Impact high.**

### W1.2 — Fail-loud world-registration pre-flight validator

**Goal.** Surface single-source-vocab / route / verify-namespace drift at startup with an
actionable error, instead of as an opaque mid-plan step failure.

**Current state.** `init_vgg` (`vcli/engine.py` def ~269) derives the vocab but never checks
that the derived vocab, the selector route table, and the verify namespace are mutually
consistent. The `World` contract (`vcli/worlds/base.py`: `decompose_vocab`,
`derive_vocab_from_registry`, `build_verify_namespace`, def ~59-110) is the input.

**Design.** Add `engine._preflight_validate_world(vocab, selector, verify_ns)`, called after
vocab derivation and before the first `try_vgg`. Assert: every `DecomposeVocab.strategies`
name resolves to a real `StrategySelector` route; every `DecomposeVocab.verify_functions` name
exists in the verify namespace; every registered capability name is selector-routable; no name
collisions. On failure, raise a multi-line error listing the offending names and the valid set
(mirrors dimos `_verify_no_name_conflicts` / the `control_task_registry` "Unknown ...
Available: [...]" surface). Scope to statically-knowable names; lazily-provided ones warn
unless a strict flag is set. The dev world (`derive_vocab_from_registry()` false, supplies its
own vocab) is validated within its declared scope.

**Test.** A world with a `verify_function` that has no namespace provider, or a strategy with no
route, aborts at startup with a valid-set error; dev/robot/playground worlds pass silently.

**Acceptance.** No startup regression on the three worlds; injected drift fails loud at boot.

**Effort S / Impact med.**

### W1.3 — Decouple `scene_graph` provider behind a `TextLLM` adapter

**Goal.** Remove the hardcoded provider + private-attribute read from the kernel; make the
spatial-reasoning path mockable.

**Current state (verified).** `core/scene_graph.py:rank_rooms_for_goal` (def ~473) inlines
`vlm._api_key` (~517), `"openai/gpt-4o"` (~521), `https://openrouter.ai/...` (~527) and a raw
httpx POST — a textbook adapter break and a provider leak into `core/`.

**Design.** Define a narrow `TextLLM` Protocol (one method, e.g.
`complete_json(prompt: str, *, schema_hint=None) -> str`) injected into `SceneGraph` from the
world/backend. The default adapter is backed by the existing `create_backend` factory so it
reuses von's configured provider (`config/user.yaml`; default is now DeepSeek) instead of a
hardcoded openrouter/gpt-4o, and never reads `_api_key`. Replace the inline httpx block.

**Test.** Inject a fake `TextLLM` returning canned JSON; assert the room ranking parses
correctly; remove any live-network reliance.

**Acceptance.** Behavior-preserving; offline-testable; provider choice lives in the
world/backend, not in `core/`.

**Effort S / Impact med.**

### W1.4 — Wire `playground.build_step_primitives` into the live executor (or delete)

**Goal.** Remove a tested-but-dead "split-brain" so the playground verifies the same primitives
that actually run.

**Current state (verified).** `playground/world.py:build_step_primitives` (def ~157) returns
`{DETECT_STRATEGY: make_detect_producer(...)}` / `{ROOMS_STRATEGY: make_rooms_producer(...)}`
but the live `GoalExecutor` never calls it (it resolves primitives via four hardcoded importlib
module paths in `_resolve_primitive`). The producing-step primitives are exercised only by tests.

**Design (wire — recommended).** In `init_vgg`, when the world exposes `build_step_primitives`,
pass its dict to `GoalExecutor` as a new defaulted `world_primitives=None` map; `_resolve_primitive`
consults that map first, then falls back to the importlib paths. This makes the tabletop
detect/grasp producers the tests exercise the ones that actually run live.
**Alternative (delete).** If this path is judged not to go live soon, delete the method + its
tests to avoid false confidence. (Owner decision — see Open decisions.)

**Test.** A new executor-level test asserts the live path invokes the world primitive
(`DETECT_STRATEGY`); the existing foreach / "grab everything" tests guard tabletop routing.

**Acceptance.** The playground closed-loop demo runs the real producer; existing tests green.

**Effort M / Impact med.**

---

## Wave 2 — operability + freshness (orthogonal to VGG, low regression risk)

### W2.1 — Daemon + on-disk run-registry + `vector status/stop/log`

**Goal.** A closed-loop controller you can leave running, inspect, and stop gracefully — the
operability a long-running robot controller needs.

**Current state.** No run lifecycle; only an ad-hoc atexit + `killpg` SIGTERM/SIGKILL cleanup of
the sim subprocess (`vcli/cli.py` ~596-684). The VGG loop already runs on a daemon thread.

**Design (borrow dimos `run_registry.py` + `daemon.py`; pure operational layer).**
- New `vcli/run_registry.py`: `@dataclass(frozen=True) RunEntry{run_id, pid, started_at, world,
  scenario, log_dir, argv}`, persisted as JSON under `~/.vector-os-nano/runs/` with an atomic
  write (tmp + `os.replace`).
- New `vcli/daemon.py`: `vector run --daemon` double-forks, redirects stdio to a per-run JSONL
  log, installs SIGTERM/SIGINT handlers that stop the VGG loop and clean the registry.
- New CLI subcommands: `vector status`, `vector stop [run_id]` (SIGTERM -> SIGKILL escalation),
  `vector log -f` (tail/follow). The foreground REPL path is unchanged; `--daemon` is opt-in.

**Test.** Start `--daemon` -> registry entry exists -> `status` lists it -> `stop` cleans up ->
no residue. Cover macOS/Linux daemonize edge cases.

**Acceptance.** Background-run a world, query status, follow logs, stop gracefully.

**Effort M / Impact high.**

### W2.2 — RUN_ID watchdog orphan sweep + retire `/tmp` flag-file IPC

**Goal.** A crashed or stopped run leaves no orphaned sim/ROS/explore process or stale flag file.

**Current state.** Module-level singleton threads (`skills/go2/explore.py`: `_explore_thread`,
`_tare_proc`, `_nav_stack_proc`) and ~15 `/tmp` flag files (`vector_nav_active`, ... in
`hardware/sim/go2_ros2_proxy.py`).

**Design (borrow dimos `process_lifecycle.py`; incremental).**
- Tag every descendant process with a `VECTOR_RUN_ID` env var; a psutil watchdog sidecar (tied
  to the W2.1 run-registry) sweeps strays by tag even after SIGKILL.
- Replace same-process flag files with a ~30-line in-memory PubSub
  (`defaultdict[topic] -> callbacks`, borrowing dimos `pubsub/impl/memory.py`).
- KEEP the genuine cross-process boundary with the separate ROS2 process as ONE explicit,
  documented channel — do not rip out wholesale.

**Test.** `kill -9` a RUN_ID-tagged run -> watchdog sweeps the children; explore cancel/stop
signals still work after the same-process flags move to PubSub.

**Acceptance.** No orphan residue on abnormal exit; `/tmp` flag count materially reduced.

**Risk.** The `/tmp` flags are a contract with the ROS2 bridge — migrate incrementally,
same-process flags first.

**Effort L / Impact med.**

### W2.3 — ObjectMemory re-query-freshest read — SHIPPED 2026-06-10 (`de4869b`)

**Goal.** Eliminate sync-once-then-stale so verify/replan see the freshest object state.

**Current state (verified).** `vcli/cognitive/object_memory.py`: `sync_from_scene_graph`
(def ~81) is called explicitly then read as truth; `last_seen` (~130), `objects_in_room` (~203),
`find_object` (~232) read a stale cache.

**Design (borrow memory2 "re-query, never cache a stale snapshot" + `BackPressure.LATEST`; pure
read-side, stays synchronous).** In the three read methods, re-query the live SceneGraph for the
category/room at call time; fall back to the exp-decay cache only when the live store has
nothing; cap with a short TTL so hot loops don't thrash; keep decay as graceful degradation.

**Test.** After a SceneGraph change, the read methods return the new value, not the init-time
snapshot; with an empty SceneGraph they fall back to decay.

**Acceptance.** Fresh object state; no lock/perf regression.

**Effort M / Impact med.**

### W2.4 — Typed `failure_class` into the replan context

**Goal.** Let observation-driven replan branch on the failure CLASS instead of a stringified error.

**Current state (verified).** `StepRecord.error: str` (`vcli/cognitive/types.py`, frozen —
additive last-field is safe). The executor captures exceptions at
`vcli/cognitive/goal_executor.py` ~291 (`error=str(exc)`), with verify-fail (~320/337) and
`visual_override` (~389-393) paths. The replan context is built in `vgg_harness` ~422-458.

**Design (borrow dimos `serialize_exception` fidelity + `DEFAULT_RPC_TIMEOUTS` table).**
- Add `failure_class: str = ""` to `StepRecord` (additive, last field, defaulted): one of
  `timeout` / `verify_fail` / `exec_error` / `ik_fail` / `tool_error`.
- Classify at the executor failure sites; thread `failure_class` into the harness replan context
  so a replan can branch (e.g. timeout -> faster strategy; ik_fail -> alternate grasp pose).
- Optional: turn the post-hoc step timeout into a per-skill timeout table (long navigate vs
  instant query).
- Security: bound the captured detail to exception type + top frame, scrub paths; never write
  secrets/internal paths into the trace store (rule 10).

**Test.** Construct timeout / verify-fail / exec-error failures -> correct `failure_class` ->
replan picks a different strategy per class.

**Acceptance.** Replan can branch by failure class; zero extra model calls (deterministic
classification).

**Effort M / Impact med.**

---

## Wave 3 — structural / seam investments (after Stage 5 stabilizes the seam)

### W3.1 — Frozen `WorldBlueprint` value object the engine consumes (additive)

**Goal.** Make world+scenario configuration declarative, diffable, serializable, and replayable
the way GoalTree already is — without replacing the imperative registration.

**Current state.** The world seam is an imperative Protocol (`vcli/worlds/base.py` ~59-110)
called method-by-method in `init_vgg` (`vcli/engine.py` ~269-453); `RobotWorld` is an empty
Phase-A shim.

**Design (borrow dimos Blueprint replace-immutability).** New `vcli/worlds/blueprint.py`:
`@dataclass(frozen=True) WorldBlueprint{tools_factory, verify_namespace_factory,
decompose_vocab, persona_blocks, capability_factories, has_base}` with fluent
`.with_capability()/.disabled()` builders returning new instances. `World` gains an optional
`.blueprint()` (mirroring dimos `Module.blueprint()`); the engine can build-from-blueprint.
Fully additive: the imperative path stays until all worlds emit blueprints.

**Test.** dev/robot/playground each emit a blueprint; the engine's build-from-blueprint result
is byte-identical to the imperative path (parity).

**Acceptance.** Declarative composition of stack variants (dev / go2-base / go2+arm / tabletop);
no imperative-path regression.

**Risk.** Over-engineering if only two-three worlds ever exist; keep `.blueprint()` optional and
do this after Stage 5 stabilizes the seam.

**Effort M / Impact med.**

### W3.2 — Typed lazy Capability factory + bridge dimos's local model zoo (strategic)

**Goal.** Connect von's measured routing brain (the StrategySelector bandit) to a real model
zoo — the clearest "each project has the half the other is missing" opportunity in the
comparison.

**Current state (verified).** `vcli/cognitive/capabilities/registry.py:get(name) -> Any | None`
(~50); the zoo holds only `LLMChatCapability` (`capabilities/chat.py`). dimos ships a uniform
local-model zoo (CLIP / MobileCLIP / Moondream / Florence / Qwen / EdgeTAM behind a
`LocalModel -> HuggingFaceModel -> VlModel` ABC tier, Apache-2.0 — reusable with SPDX
attribution) but has zero routing intelligence (callers hardcode the model).

**Design.**
- Add a typed factory layer `create(name: CapabilityName) -> Capability` (`CapabilityName` a
  `Literal`) with `"module:attr"` lazy factories so heavy deps (a local VLM / detector) load
  only when that capability is actually composed into a world. The registry stays the single
  source; the factory gives a typed, statically-discoverable surface (feeds the W1.2 pre-flight
  validator + the single-source decompose vocab).
- Phase C.3 bridge: register dimos-style local models (detector / segmenter / VL / embedding) as
  read-only capabilities in the robot/playground worlds; the existing bandit routes by measured
  fit (`detect_*` accrues `("moondream","detect_*")` vs `("florence","detect_*")`). Verify still
  gates every capability result deterministically — a capability never self-certifies (already
  enforced; invoke-success-but-verify-false => `StepRecord.success=False`).
- Constraint: detection/recall outputs must still pass an AST/evidence predicate before driving
  a plan step (does not dilute the moat).

**Test.** Two mock detector capabilities compete on a `detect_*` pattern; after >=3 attempts the
bandit promotes the winner; the lazy factory does not import a model until it is composed.

**Acceptance.** The model zoo is non-empty; measured routing works; verify is unchanged; Phase
C.3 has a working on-ramp.

**Effort M-L / Impact high (strategic).**

### W3.3 — Protocol-compliance to kill getattr-by-string on agent internals

**Goal.** The kernel asks for a Protocol, not a named private attribute; an arm-only vs base
world fails loud with a clear "no provider met spec" error.

**Current state.** The kernel probes agent internals via `getattr(agent, '_base'/'_arm'/
'_perception')` duck-typing (`vcli/cognitive/goal_executor.py`, engine, skills);
`hardware/base.py` already defines `BaseProtocol`.

**Design (borrow dimos `spec_structural_compliance` + `_resolve_single_ref` fail-loud ambiguity
errors; in-process DI-by-Protocol, no bus).** Standardize narrow `runtime_checkable` Protocols
(Base/Arm/Perception surfaces for what the cognitive layer actually calls); resolve a capability
by `isinstance(Protocol)` with a fail-loud "no / multiple provider met spec X, candidates: [...]"
error. Do it incrementally, `_base` seam first, under the existing baseless-arm-world tests.

**Test.** A baseless arm world requesting a base capability gets a clear "no base provider"
error rather than an AttributeError / silent None.

**Acceptance.** The `_base` seam migrates first; behavior preserved, errors clearer.

**Effort L / Impact med.**

**Bonus (dimos skill resource arbitration).** dimos `@skill(uses=[...])` + a `CapabilityRegistry`
mutual-exclusion lock (refuses to start a skill whose required capability is held by another) +
`__skill_lifecycle__` (instant/background) is a safety pattern von lacks. Worth folding into the
W3.2 capability system to give "skill X holds the base; another motor skill can't run
concurrently" arbitration for the foreach / future-parallel execution path.

---

## Relationship to Stage 5 / Phase C

- **Stage 5 (unify the two planning paths).** W1.1 dovetails with it: the unified controller's
  `_unified_plan_turn` already reads `_evidence_ok` (S5.3 is partially dark-launched at HEAD,
  ahead of what STATUS records); W1.1 aligns the LEARNING tier to the same evidence gate, closing
  the loop end-to-end. W1.1/W1.2/W1.3 touch the same engine/cognitive paths and fit alongside the
  Stage 5 finish.
- **Phase C (model zoo).** W3.2 is the direct on-ramp to C.3 — it turns "capability seam exists
  but the zoo is empty" into "non-empty zoo + measured routing". W3.3 supports the RobotWorld
  migration that C.3 presumes (the ADR-006 seam is currently only half-realized for the robot
  path: robot verify bindings still live kernel-side in `engine._build_verifier_namespace`).

## Doc Governance

- This file is a Tier-4 plan with open decisions (below) — allowed while open.
- On the FIRST work commit of this phase: add a one-line pointer to this plan in
  `agent-kernel-STATUS.md` and update STATUS's "next" section.
- W3.1 changes the world contract -> update `ARCHITECTURE.md` Section 5 in the same commit.
  W2.4 adds a `StepRecord` field -> update the types description in the same commit. W2.1 adds a
  new kernel-adjacent operational surface -> note it in the module map.
- Delete this plan once Phase E fully ships (git history keeps it).

## Open decisions (owner sign-off before coding)

1. **W1.4 wire vs delete.** Wire `build_step_primitives` into the live executor (recommended,
   makes the playground demo real) or delete the dead path (smaller, if it won't go live soon)?
2. **W1.1 robot-world learning policy.** Confirm that the per-step evidence reward mirrors
   `evidence_passed`'s `is_robot` leniency (so robot learning is not starved), i.e. the change is
   meaningful only for dev/playground worlds with real predicates.
3. **W3.1 scope.** Build the `WorldBlueprint` now, or defer until there are >3 worlds (risk of
   ossifying the seam mid-Phase-C-migration)?
4. **W3.2 model sourcing.** Vendor/depend on dimos's Apache-2.0 model zoo (with SPDX attribution)
   vs wrap von's own perception models as capabilities vs both. Which models first
   (detector vs segmenter vs VL)?
5. **Commit cadence.** One commit per item (recommended) vs per-wave.

## Risk table

| Risk | Mitigation |
|---|---|
| W1.1 strict evidence starves robot learning | `step_evidence_ok` mirrors `evidence_passed` `is_robot` leniency; add a robot-leniency regression test |
| W1.4 wiring changes tabletop routing | Existing foreach/grab tests guard + a new executor-level test that the live path invokes the world primitive |
| W2.1 macOS/Linux daemonize differences | Thin, well-tested module; `--daemon` opt-in, foreground path unchanged |
| W2.2 `/tmp` flags are a ROS2-bridge contract | Incremental: keep the real cross-process boundary as one explicit documented channel; migrate same-process flags first |
| W3.1 over-engineering | `.blueprint()` optional, imperative path retained, do after Stage 5 |
| W3.2 heavy deps / unverified recall | Lazy load; recall still gated by an evidence predicate; read-only capabilities carry no side-effect risk |
| Global | Every increment keeps 863+ green, adds its own regression, never eval/exec model output, never leaks secrets/paths |
