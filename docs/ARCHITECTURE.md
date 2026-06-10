# Vector OS Nano — Architecture

- Status: Canonical durable design doc. Supersedes `agent-kernel.md` (removed; in git history).
- Scope: the orchestration layer — VectorEngine + the VGG cognitive layer + the CLI/MCP
  entry points. The OS *around* the models, in service of robots.
- For live "where are we / what's next", see [agent-kernel-STATUS.md](agent-kernel-STATUS.md).
- For the kernel/world decision, see
  [architecture-decisions/ADR-006-agent-kernel-world-plugin.md](architecture-decisions/ADR-006-agent-kernel-world-plugin.md);
  the closed-loop reframe is recorded in
  [ADR-007](architecture-decisions/ADR-007-closed-loop-controller.md).

---

## 1. Vision and north star

Vector OS Nano is not another model — it is an **agent-orchestration system built around
models, in service of robots**. The promise is simple: **natural language controls
everything**. A built-in agent **decomposes** an instruction into a plan, **routes** each
step to the right capability, **executes** the long chain, **verifies** every step against
a machine-checkable predicate, and **re-plans** on verified failure — with no fine-tuning.
The things models do unreliably on their own (planning, step verification, recovery) are
exactly the OS's job.

Unified large models, specialized small models, classical skills, and atomic actions are
orchestrated into one deployable whole that is cross-hardware, cross-model, and
cross-system. Developers build physical AI the way they write code: any model, any skill,
any robot — plug and play. We are not building the smartest brain; we make *every* brain
run trustworthily on the industrial floor.

**Robots are the end. The dev / macOS path is a means** — a hardware-free build/test loop
that proves cross-system deployability and exercises the orchestration logic without a
body attached.

**North star, in one line:** NL in → decompose → plan → execute long-chain → verify each
step → re-plan → evidence-gated done. The single-skill slice ships today; the long-chain
closed loop is being completed (see Section 7).

---

## 2. Kernel vs World plugin

The architecture has one load-bearing seam: a pure-Python **kernel** that never imports a
world, and **worlds** that register into it.

**The kernel is generic and hardware-free.** VectorEngine, the model backends, the entire
VGG cognitive layer (decompose / verify sandbox / execute / harness / selector / stats /
blackboard / capabilities), the general file/bash/glob/grep/web tools, session,
permissions, and the intent router are all pure Python with no robot or ROS2 dependency.
They run anywhere (macOS first).

**A world registers exactly four things** — and nothing else crosses the seam:

1. **Tools** — into the existing `CategorizedToolRegistry` under the world's own category
   (intent routing already filters by category).
2. **A verify / primitive namespace** — the dict `GoalVerifier` evaluates predicates
   against (robot predicates like `holding_object()`; dev predicates like file existence).
3. **A decompose vocabulary** — the strategy menu plus the verify-function names,
   signatures, descriptions, and few-shot examples the `GoalDecomposer` prompt teaches.
   Single-sourced from the world's skill/capability registry so the prompt the LLM reads
   and the validator allowlist can never drift apart.
4. **A persona / prompt block** — the role prompt and tool instructions for the domain.

A world thus contributes the **routable capabilities** for its domain — today skills and
atomic actions; as the heterogeneous model zoo lands, specialized models (detectors,
planners, VLA policies) register here too, and the StrategySelector routes a sub-goal to
the right one by measured fit. This is the "any model, any skill, any robot" seam.

| Stays in the kernel (generic) | Moves to a world plugin |
|---|---|
| VectorEngine, backends, session, permissions, intent-router mechanism | Robot/arm/ROS2 launch and embodiment wiring |
| VGG: verifier sandbox, executor, harness, selector, stats, code-executor, blackboard, topo-sort, retry | Robot/dev verify bindings (the predicate namespace) |
| `GoalDecomposer` mechanics (JSON parse, AST validation) | The world's *vocabulary* (strategies, verify funcs, examples) |
| General tools: file / bash / glob / grep / web | Robot / diag / sim tools + skill wrappers |
| `CategorizedToolRegistry` category mechanism | `skills/`, `hardware/`, `perception/`, `ros2/` |

**Worlds today:** `dev` (laptop, robot-free, the build/test means — ships in the kernel),
and `robot` embodiments — Go2 (has a mobile base) and SO-101 / Piper arm (no base). The
kernel is identical across all of them; only the four registered things change.

**The seam is the integration contract — and it is what makes parallel development
possible.** Because a world touches the kernel through exactly these four registrations
(plus the verified-loop observation surface in Section 5), the kernel and a world can be
built as **separate, parallel tracks** that meet only at the contract. This is how the
**playground** is developed: its own track (eventually its own repo) registering preset
scenes, embodiments, and tasks as worlds, while the agentic kernel evolves behind the same
versioned contract — neither track reaching across except through the four registrations
and the observation surface. (Decision: ADR-008.)

---

## 3. Block diagram

```
+----------------------------------------------------------------------+
|                              AGENT KERNEL                             |
|     (pure Python; no robot/ROS2 import; runs on macOS; generic)      |
|                                                                      |
|  VectorEngine     run_turn (tool-use loop) | vgg_decompose/execute   |
|  Backends         Anthropic | OpenAI-compat | local                  |
|  VGG (cognitive/) GoalDecomposer  -> GoalTree (frozen DAG)           |
|                   StrategySelector -> skill|primitive|code|tool|cap  |
|                   GoalExecutor (topo-sort, timeout, capture output)  |
|                   GoalVerifier (AST sandbox; evaluate -> (bool,raw)) |
|                   VGGHarness (plan-act-verify-replan loop)           |
|                   Blackboard (per-run observations; ${path} binding) |
|                   StrategyStats | ExperienceCompiler/TemplateLibrary |
|                   CapabilityRegistry (chat now; detectors/VLA later) |
|  General tools    file/bash/glob/grep/web                            |
|  Session | Permissions (7-layer) | IntentRouter                     |
+---------------------------------+------------------------------------+
                                  | a World registers 4 things:
                                  |  1 tools   2 verify namespace
                                  |  3 decompose vocab (from registry)
                                  |  4 persona / prompt
        +-------------------------+---------------------------+
        |                         |                           |
   DEV world                 ROBOT world                (future worlds /
   (laptop, no robot)        Go2 (has_base) | SO-101 arm   embodiments)
   build/test MEANS          (no base)
                             tools + verify + vocab + persona
                             skills / primitives / (models)

   The kernel NEVER imports a world; worlds register INTO the kernel.
```

Two entry points — the `vector-cli` REPL and the `vector-os-mcp` server — share this one
engine. Robots are the end; the dev path is the hardware-free means. The seam is also the
**track boundary**: the agentic kernel (this whole block) and the **playground** (a separate,
parallel-developed world track) are built independently and integrate only across it
(Section 5; ADR-008).

---

## 4. Planning flow (the closed loop)

The kernel is moving from an open-loop "compiler" (plan once, execute blind, discard
observations) to a grounded closed-loop "controller" (observe, bind, re-plan against
measured outcomes). The `[Stage N TODO]` markers below honestly mark what is aspirational
versus shipped — see Section 7 for stage status.

```
NL input
  |
  v
run_turn_unified (ONE closed-loop controller — both frontends call it)
  classify_intent  ==> IntentDecision{route, reason, complex}   [HINT, not a verify gate]
  |   (should_use_vgg is now a cheap shape PRE-CLASSIFIER feeding the controller,
  |    incl. the conversational-question guard; it no longer forks AROUND verify)
  |
  |-- tool_use route (chat / question / greeting):
  |     ReAct run_turn produces the answer (streaming, permissions, hooks, P0 stop)
  |       -> wrap as a 0-action answer-only GoalTree -> harness verify (trivially true)
  |       -> evidence gate EXEMPTS the answer-only step (an action step with no
  |          predicate still FAILS the gate — the moat is intact)
  |
  |-- vgg route (simple/alias or complex):
        build world_context  <--------------------------------------+
          [Stage 3 TODO: + structured perception / grounding]       |
          |                                                          |
          v                                                          |
        GoalDecompose (LLM; vocab single-sourced from registry)     |
          -> GoalTree (frozen DAG; foreach SHIPPED, until/if TODO)  |
          |   (1-step fast path stays deterministic — no LLM)        |
          v   for each sub-goal in topological order:               |
        StrategySelector -> executor_type (world-scoped, fail-loud) |
          |                                                          |
          v                                                          |
        execute --> capture output --> Blackboard --${path} bind----+  (-> next step params)
          |                                                          |
          v                                                          |
        GoalVerifier.evaluate -> (bool, value) -> StepRecord.result_data
          |                                                          |
   pass --+-- fail / new observation --> REPLAN --------------------+
          |        (fresh world_context + validation_notes;
          v         Stage 4 TODO: observation-driven, mid-tree)
        verified done (evidence-gated)
  |
  v
UnifiedTurnResult{text, trace, snapshot, intent, tool_calls, verified, usage}
```

Both frontends (`vector-cli`, `vector-os-mcp`) call `run_turn_unified`, so EVERY turn —
chat included — produces a verified-loop trace; the keyword gate is a routing **hint**,
not a fork in front of verify. `VECTOR_LEGACY_TURN=1` restores the pre-cut-over open
`run_turn` ReAct loop for one release as a fallback. The 1-step fast path stays
deterministic (no LLM call) for single skills and aliases. The VGG path is the closed
loop: each execution writes its output to the Blackboard; later steps bind to those
outputs via `${step.output.path}` references; verification returns both a gate boolean
and the raw value (recorded on the step); and every re-plan rebuilds `world_context` from
scratch so the planner sees the latest observations and the prior attempt's validation
notes.

---

## 5. Core invariants and contracts

These are the contracts the kernel guarantees. Anything that violates them is a bug.

- **Kernel/world seam.** The kernel never imports a world. A world crosses the seam by
  registering exactly four things (tools, verify namespace, decompose vocab, persona).
  No other coupling is permitted; robot specifics never leak into kernel code paths.

- **Single-source vocabulary.** The decompose vocabulary is derived from the world's
  skill/capability registry, so the prompt the LLM reads and the validator allowlist are
  the same set by construction (no split-brain). On failure to build a world vocabulary,
  fall back to a **neutral** vocabulary — never to another domain's defaults. (The
  historical GO2-default fallback on a baseless arm world is exactly the bug this
  invariant forbids.)

- **Verify is a deterministic predicate.** `GoalVerifier.evaluate()` returns
  `(bool, value)`: the boolean is the evidence gate, the value is the raw observation
  recorded on the step. Verification runs in an AST sandbox with restricted builtins and a
  hard timeout. The sandbox is **only ever stricter** than plain Python — never `eval` or
  `exec`, never an import escape. Verification is machine-checkable, not an LLM judge
  (an escalation ladder to a visual/VLM check exists for cases predicates cannot express;
  LLM judging is the last resort, not the default). A verify can additionally consume the
  CURRENT step's own structured output through the kernel-injected `step_output(path)`
  function (a per-evaluation namespace overlay; the executor binds it to that step's
  `exec_output`) — so e.g. a detect step verifies what IT observed, alias-aware, instead
  of re-querying a separate oracle that false-passes when the requested target is absent.
  The overlay never persists and the sandbox is not loosened.

- **Closed-loop observation flow.** Each step's output is written to the per-run
  Blackboard. Downstream parameters bind to upstream outputs via `${step.output.path}`
  references, resolved by pure dict/list traversal (no code execution). `StepRecord`
  carries `result_data` so observations — not just success/error strings — drive the next
  step and the next re-plan. `world_context` is rebuilt on every (re)decompose so the
  planner always reasons over the latest state.

- **Frozen-dataclass, additive-only.** Plan structures (`GoalTree`, `SubGoal`,
  `StepRecord`) are frozen dataclasses. Evolve them by adding fields, never by mutating in
  place — data flow stays inspectable and replayable.

- **World-agnostic mechanisms.** Decompose mechanics, the verify sandbox, the executor,
  the harness, the selector, and the stats bandit contain no domain knowledge. Domain
  knowledge lives only in the registered four things.

- **Fail-loud routing and validation.** An unresolved strategy surfaces a clear "skill X
  is not in this world's registry" error rather than silently degrading to a fallback.
  Hallucinated or dropped strategies are fed back into the re-plan context as validation
  notes so the next planning pass stops repeating the mistake.

- **The seam is a versioned public contract (the parallel-track boundary).** The kernel and
  a world (e.g. the playground) integrate ONLY through the four registrations (tools, verify
  namespace, decompose vocab, persona) plus the **verified-loop observation surface** — the
  kernel exposing each run's `GoalTree`, per-step `StepRecord` (`success` / `verify_result` /
  `result_data`), and replan `validation_notes` as structured, inspectable data a front-end
  can render. Neither track reaches across the seam by any other path. This contract is what
  lets the agentic kernel and the playground be built as separate, parallel tracks (ADR-008).
  Status: world selection now goes through `WorldRegistry` (`resolve_world` / `resolve_world_named`)
  and the engine MERGES the active world's `build_verify_namespace` additively — so a world (e.g.
  the playground) OWNS its predicates. The deterministic ARM sim-oracle predicates are now
  single-sourced in the kernel (`vcli/worlds/arm_sim_oracle.py`) and contributed by BOTH
  `PlaygroundWorld` AND `RobotWorld` (the latter binds them whenever a sim arm — one exposing
  `get_object_positions` — is connected), so detect/verify are grounded in the plain robot arm
  world, not just the playground. Remaining debt: the legacy scalar bindings (`get_position` /
  dev predicates / the non-sim perception stubs) still live in `engine._build_verifier_namespace`
  and merge additively; the real-hardware (no sim oracle) grounding path is deferred to Stage 3.

---

## 6. Conceptual module map

One line each. No line numbers (they rot — when in doubt, read the file). Paths are
relative to `vector_os_nano/`.

**Engine and backends**
- `vcli/engine.py` — VectorEngine: the tool-use loop, dispatch, VGG entry points, and the
  per-world verify-namespace binding.
- `vcli/backends/` — model adapters (`anthropic`, `openai_compat`) behind a common type.

**VGG cognitive layer** (`vcli/cognitive/`)
- `goal_decomposer.py` — NL -> GoalTree; JSON parse + AST validation; teaches the world's
  vocabulary in its prompt.
- `goal_verifier.py` — the AST predicate sandbox; `evaluate() -> (bool, value)`.
- `goal_executor.py` — topo-sort execution, timeout, output capture, executor-type dispatch
  (skill / primitive / code / tool / capability).
- `vgg_harness.py` — the plan-act-verify-replan loop; rebuilds `world_context` each pass.
- `strategy_selector.py` — chooses an executor type / strategy for a sub-goal, world-scoped.
- `strategy_stats.py` — measured per-strategy success rates; the bandit driving selection.
  Its per-step reward is EVIDENCE-GATED (W1.1): the learning tier records `step.success AND
  step_evidence_ok(...)`, not raw `step.success`, so a VLM visual-override / sentinel `verify="True"`
  "success" cannot train the bandit. Robot world bypasses the gate (reward collapses to `step.success`).
  Single chokepoint: `GoalExecutor._record_strategy_stats` (all record sites route through it).
- `blackboard.py` — per-run observation store; resolves `${step.output.path}` bindings.
- `vocab_from_registry.py` — `build_decompose_vocab`: single-sources the decompose
  vocabulary from the skill registry.
- `capabilities/` — the capability seam (`Capability` protocol + `CapabilityRegistry` +
  `LLMChatCapability`); the bridge to a heterogeneous model zoo.
- `trace_store.py` — save / load / replay of verified runs; the evidence gate and
  verify-as-eval signal. `evidence_passed` (per-trace) + `step_evidence_ok` (per-step analogue,
  W1.1) are the deterministic gate the LEARNING tier (bandit reward + template compilation) is
  measured against — never raw `step.success`.
- `template_library.py` — compiled reusable plan templates; backs the no-LLM fast path.
- `experience_compiler.py` — turns successful verified traces into templates (no
  fine-tuning). Compilation is EVIDENCE-GATED (W1.1): `engine._maybe_compile_experience`
  requires `trace.success AND _evidence_ok(trace)`, so only evidence-backed traces compile.
- `types.py` — frozen plan structures (`GoalTree`, `SubGoal`, `StepRecord`, `ForEachSpec`);
  `SubGoal.foreach` carries a control-flow loop the executor expands at runtime. A failed
  `StepRecord` carries a deterministic typed `failure_class` (W2.4: timeout/verify_fail/ik_fail/
  tool_error/exec_error) threaded into the replan context so the re-decompose adapts by class.
- `observation.py` — the verified-loop observation surface: a pure JSON-safe export view over the
  frozen types (`step_view` / `run_snapshot`) + plain-text renderers; what a front-end renders.

**Worlds** (`vcli/worlds/`)
- `base.py` — the `World` protocol (the four-thing contract).
- `blueprint.py` — frozen `WorldBlueprint` value object (W3.1): `blueprint_of(world)` derives a
  declarative, diffable snapshot of any world's seam contributions; `BlueprintWorld` adapts it
  back into a full `World` the engine consumes unchanged — parity with the imperative seam is
  structural, and the imperative path stays authoritative.
- `dev.py` — the robot-free dev/code world (default; build/test means).
- `robot.py` — robot embodiments (Go2 with a base; SO-101 / Piper arm without one).
- `registry.py` — `WorldRegistry`: world/scenario resolution (agent-driven `resolve_world` +
  named `resolve_world_named`); worlds self-register via lazy factories (the seam-as-contract entry).

**Playground track** (`playground/` — a separate, parallel-developed world track; ADR-008)
- `world.py` / `scenario.py` / `catalog.py` — embodiment-aware `PlaygroundWorld` + frozen `Scenario`
  + the preset catalog (arm: `tabletop`, `tabletop_tray`; quadruped: `go2_room`); registers into the
  kernel `WorldRegistry` via a lazy hook. `Scenario` is sim-backend-aware (additive
  `sim_backend`/`scene_ref` fields, default `"mujoco"`/`""`): an MJCF scenario loads `scene_xml`,
  a non-MJCF backend (e.g. habitat, ADR-009) dispatches on `scene_ref` — the kernel never
  imports a simulator from this data.
- `verify/` — sim-oracle verify predicates contributed across the seam. The ARM predicates
  (`holding_object`/`arm_at_home`/`placed_count`/`detect_objects`/`describe_scene`) are SINGLE-SOURCED
  in the kernel at `vcli/worlds/arm_sim_oracle.py` (so `RobotWorld` can reuse them without the kernel
  importing the playground); `playground/verify/arm_predicates.py` + `scene_predicates.py` are thin
  re-export shims. The Go2 base predicates (`at_position`/`facing`/`visited`) still live here.

**Tools, routing, prompt, session, permissions**
- `vcli/providers.py` — protocol-based provider resolution (W3.3): narrow `runtime_checkable`
  specs (`BaseStateProvider`/`BaseMotionProvider`) + fail-loud `resolve_provider`/`ensure_provider`
  ("no base provider met spec X: missing [...]") replacing getattr-by-string on agent internals;
  `_base` seam migrated first, the rest incremental.
- `vcli/tools/` — general tools (file/bash/glob/grep/web) + world-contributed tool wrappers.
- `vcli/intent_router.py` — category-filtered routing that trims the tool/context surface.
- `vcli/dynamic_prompt.py` — the composable system prompt rebuilt as world state changes.
- `vcli/prompt.py` — persona / role-prompt blocks.
- `vcli/session.py` — conversation state and JSONL transcript.
- `vcli/permissions.py` — the 7-layer permission system gating side-effecting tools.

---

## 7. Current state and roadmap

The work is staged from open-loop to closed-loop. **Live status (which stage, what is
committed) lives in [agent-kernel-STATUS.md](agent-kernel-STATUS.md)** — this section gives
the durable shape only.

**Two parallel tracks (ADR-008).** Work proceeds on two tracks that meet only at the seam
contract (Section 5): the **kernel track** (the closed-loop stages below — grounding,
control-flow IR, unify paths — the agentic moat, behind the contract) and the **playground
track** (a separate, parallel-developed catalog of preset scenes, embodiments, and tasks
registered as worlds, NL entry into a scene, and the view that renders the verified loop, in
front of the contract). A small **shared prelude is not parallelizable** and comes first:
define and harden the seam contract (verify namespace owned by the world; `resolve_world` -> a
world/scenario registry; the verified-loop observation surface). The flagship demo — a
*visible, self-correcting long chain* — is a cross-track rendezvous, needing the kernel
track's grounding + control-flow AND the playground track's view, so the tracks run in
parallel but meet at milestones.

**Shipped:**

- **Stage 0 — NL-first visible sim.** On macOS the CLI re-execs under `mjpython` so the
  arm sim opens a window by default; headless is the opt-out.
- **Stage 1 — close the loop.** The per-run Blackboard with safe `${step.output.path}`
  param-binding (resolved by pure dict/list traversal); `GoalVerifier.evaluate() ->
  (bool, raw)`; `StepRecord.result_data`; VGGHarness rebuilds `world_context` on every
  (re)decompose.
- **Stage 2 — single-source the vocab.** `vocab_from_registry.build_decompose_vocab`
  derives the decompose vocabulary from the skill registry (killing the GO2 split-brain);
  base primitives are gated on `has_base`; the StrategySelector is world-scoped; validation
  is fail-loud and feeds `GoalTree.validation_notes` back into re-plan.

**Remaining:**

- **Stage 3 — grounding.** Structured perception into `world_context`; referring-expression
  resolution; wire real `detect` / `describe`; arm predicates; ObjectMemory re-sync.
- **Stage 4 — control-flow IR.** `foreach` / `until` / `if` in the goal model plus
  observation-driven mid-tree re-plan — what makes "把所有东西抓一遍" (grab everything,
  one by one) work end-to-end.
- **Stage 5 — unify the paths.** Collapse the fast path and the VGG path into one
  closed-loop controller and drop the keyword intent gate.
- **Stage 6 (later) — learning loop v2 + model zoo (C.3).** A real specialized model
  registered in the robot world, routed by measured fit.

Prior phases (A–C) established the foundation: kernel/world decoupling (Phase A), the
differentiation tier wired and made real (Phase B — tool-backed execution, code-as-policy
sandbox, verify-as-eval, persistent stats, experience compilation), and the capability
seam plus cross-capability routing (Phase C.1/C.2). Phase C.3/C.4 are open and sequenced
after the closed-loop stages; see
[agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md) and
[agent-kernel-phase-d-plan.md](agent-kernel-phase-d-plan.md).

---

## 8. Honest positioning

Mainstream agent runtimes (Claude Code, Cursor, Devin, OpenHands, LangGraph) bottom out in
a ReAct-style tool loop where "verification" is the model self-assessing over whatever tool
feedback exists. The differentiator here is **deterministic per-step verification as a
first-class runtime primitive**, an inspectable self-revising goal graph, measured-strategy
selection, and verified-by-construction template compilation — unified in one
fine-tuning-free, domain-general runtime.

| Dimension | Mainstream agent CLIs | Vector OS Nano (VGG) |
|---|---|---|
| Verification | Model self-assessment + opportunistic tests | Machine-checkable predicate per sub-goal, AST-sandboxed, deterministic |
| Plan representation | Opaque chat chain / flat todo / static graph | LLM-generated, inspectable, replayable goal graph; re-plans against measured outcomes |
| Retry / strategy | Single-trajectory reflection | Ranked strategies for the *same* goal, picked by measured success rate |
| Cross-task learning | Human-authored skills/rules | Verified traces compiled into reusable templates; no fine-tuning |
| Cost / latency | Caching + compaction | Same, plus a deterministic single-skill fast path (no LLM call) |

**Do not overclaim.** Every pillar has clear prior art: **Voyager** (a verified, reusable
skill library without fine-tuning), **ToolGate / ProgPrompt** (pre/post-condition
contracts), **process reward models** (step-level verification), **LLM+P** (symbolic plan
verification), and **multi-armed bandits** (measured selection). The defensible
contribution is the **conjunction and engineering**: deterministic per-sub-goal predicate
contracts + measured-success strategy selection + verified-by-construction template
compilation, in one fine-tuning-free domain-general runtime aimed at the generator-verifier
gap that LLM-judge and self-refine pipelines cannot close. Any "no system unites all four
pillars" claim is a best-effort literature finding, not a proof — keep external claims
hedged.

**Soft spots, stated openly:**

- The LLM still authors the goal tree *and* the predicates, so verification bias re-enters
  at predicate-authoring time. Deterministic predicates remove subjectivity only within
  what the predicate can express. Mitigation: predicate libraries, human-reviewable
  predicates, and the escalation ladder (deterministic -> visual/VLM -> LLM judge as last
  resort).
- Model-zoo routing (route a sub-goal to a detector / planner / VLA policy by measured fit)
  is **forward-looking**. Today the kernel routes to skills, primitives, code, tools, and
  one chat capability. The capability seam exists; the heterogeneous fleet does not yet.
- LLM-authored strategy code is sandboxed only on the VGG `code` path. Any code-as-policy
  execution must be forced through the AST validator before it runs.

**Non-goals.** No foundation model and no competing on raw model intelligence. The
dev/macOS path is a means, not a product. No re-implementing industrial navigation on
macOS, no driving real hardware from macOS (Linux + ROS2 stays the high-fidelity backend),
and no fine-tuning — the kernel is orchestration-first and frozen-model by design.

**References (prior art):** ReAct; Reflexion; Tree-of-Thoughts; LATS; Plan-and-Solve;
process reward models ("Let's Verify Step by Step"); the generator-verifier gap (Stechly
and Kambhampati); LLM+P; Code-as-Policies; ProgPrompt; Voyager; ToolGate; FormalJudge; VeriGuard;
library learning (LILO); bandits for LLM selection.
