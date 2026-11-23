# Vector OS Nano — Architecture

- Status: Canonical durable design doc. High-level + durable; legacy implementation
  deep-dive removed (git history keeps it).
- Scope: the orchestration layer — the generic agent kernel, the honest-verify spine,
  the plug-and-play platform contracts, and the CLI/MCP entry points. The OS *around* the
  models, in service of robots.
- For live "where are we / what's next" (stages, current round), see
  [agent-kernel-STATUS.md](agent-kernel-STATUS.md).
- For the decision history (kernel/world seam, closed-loop reframe, native-producer ruling,
  embodiment-as-config, …) see [DECISIONS.md](DECISIONS.md).

---

## 1. Vision

Vector OS Nano is not another model — it is an **agent-orchestration runtime built around
models, in service of robots**. The four things models do *unreliably* on their own are
exactly the OS's job: **plan** the task, **route** each step to the right model AND skill,
**verify** every step against a machine-checkable predicate, and **recover** on verified
failure — no fine-tuning. It orchestrates general large models, specialized small models,
classical skills, and atomic actions into one deployable whole that is cross-hardware,
cross-model, cross-system. Further, it is a **programmable, plug-and-play platform**:
developers bring any model, skill, or robot and plug it in with no kernel edits. Robots are
the end; sim is the current means. **The authoritative goal statement is the North Star in
[CLAUDE.md](../CLAUDE.md) — read it first; this doc describes the architecture that serves it,
not a second copy of the goal.**

---

## 2. Kernel / World seam

The architecture has one load-bearing seam: a pure-Python **kernel** that never imports a
world, and **worlds** that register into it.

**The kernel is generic and hardware-free.** VectorEngine, the model backends, the entire
cognitive spine (decompose / verify sandbox / execute / harness / selector / stats /
blackboard / capabilities / trace-store), the native tool-use producer, the general
file/bash/glob/grep/web tools, session, permissions, and the intent router are all pure
Python with no robot or ROS2 dependency. They run anywhere (macOS first).

**A world registers exactly four things** — and nothing else crosses the seam
(`vcli/worlds/base.py`, the `World` protocol):

1. **Tools** — into the `CategorizedToolRegistry` under the world's own category (intent
   routing filters by category).
2. **A verify / primitive namespace** — the callables `GoalVerifier` evaluates predicates
   against (robot predicates like `holding_object()`; dev predicates like file existence).
3. **A decompose vocabulary** — the strategy menu plus verify-function names, signatures,
   and few-shot examples the `GoalDecomposer` prompt teaches. Single-sourced from the
   world's skill/capability registry so the LLM prompt and the validator allowlist can never
   drift apart.
4. **A persona / prompt block** — the role prompt and tool instructions for the domain.

A world thus contributes the **routable capabilities** for its domain; the kernel is
identical across all of them, only the four registered things change. This is the "any model,
any skill, any robot" seam.

| Stays in the kernel (generic) | Moves to a world plugin |
|---|---|
| VectorEngine, backends, session, permissions, intent-router mechanism | Robot/arm/ROS2 launch and embodiment wiring |
| Cognitive spine: verifier sandbox, executor, harness, selector, stats, blackboard, trace-store, native producer | The world's verify bindings (the predicate namespace) |
| `GoalDecomposer` mechanics (JSON parse, AST validation) | The world's *vocabulary* (strategies, verify funcs, examples) |
| General tools: file / bash / glob / grep / web | Robot / diag / sim tools + skill wrappers |
| `CategorizedToolRegistry` category mechanism | `skills/`, `hardware/`, `perception/`, `ros2/` |

**Worlds today:** `dev` (laptop, robot-free — the build/test means, ships in the kernel) and
`robot` (Go2 mobile base, SO-101 / Piper arm, Unitree G1 humanoid). A separate
parallel-developed **playground** track registers preset scenes/embodiments the same way.

---

## 3. The five plug-and-play contracts

The structural spine of the platform (CLAUDE.md Rule 11). Each is a way to bring your own
part across the seam — the **user provides data/code on the left, the system does the work
generically on the right**.

- **Embodiment** — *user provides:* a URDF + meshes + a `robot.yaml` manifest (spawn pose,
  joint/stance layout, sensor mounts, root body, gait/policy ref, capability profile).
  *system does:* stands the body up through ONE generic driver — no per-robot driver class,
  no morphology constants in Python. The schema is `embodiments/config.py`
  (`EmbodimentConfig` + frozen sub-specs, fail-loud loader); every body conforms **uniformly**
  to `hardware/base.py` `BaseProtocol` (no per-robot method drift). **Honest status**
  (branch `arch/plug-and-play`): S1 (config schema + BaseProtocol uniformity), S2 (drivers
  READ `robot.yaml` + generic `DofLayout` — no hardcoded qpos), S3a/S3b (one shared lidar +
  ONE `MjSpec.attach` scene builder) and S5a (capability gating single-sourced onto
  `resolve_capability_profile`) have landed; `go2`/`g1` ship as `robot.yaml` configs. STILL
  per-driver classes (one generic driver class is S4, a CEO-gated stage); the remaining work
  is the embodiment registry (S4) + policy plugin (S5) + capability planner-exposure (S6).
- **Policy** — *user provides:* a gait / control policy (RL / MPC / scripted) plus an
  obs/action spec. *system does:* drives the body with it; the policy lives separately,
  plugged in by spec (referenced from `policy.ref` in the manifest), not fused into the body.
- **Skill** — *user provides:* a `@skill` declaring what it `requires` (arm / base / camera);
  it may wrap an external **VLA / VLM** or a classical **grasp / nav** stack. *system does:*
  makes it callable by NL, routes to it by requirement match, and grades it like any step.
- **Capability** — *user provides:* an external model or stack (detector / planner / VLA)
  registered as a routable unit with a typed `(input → output)` contract
  (`vcli/cognitive/capabilities/`, the `Capability` protocol). *system does:* routes a sub-goal
  to it by measured fit and verifies the result — the capability never self-certifies.
- **Verify** — *user provides:* a world-side predicate reading **independent ground truth**.
  *system does:* uses it as the evidence gate for every step the actor took — this is how
  success is *proven*, not claimed.

---

## 4. The honest-verify moat

What makes "bring your own model" *trustworthy* — and what no sim wrapper has — is the
**honest-verify spine** (`trace_store` / `actor_causation` / `evidence_classifier`).
Every plugged-in model and skill is graded by a deterministic predicate that reads ground
truth **the actor cannot author**: the runtime trusts *evidence*, never a self-report. The
spine is the **frozen reference** of the system — the native producer hands it an
`ExecutionTrace` and the spine computes `verified` byte-for-byte unchanged; the producer never
computes it itself. Verification is **only ever stricter, never looser** (Rule 5): an
AST-sandbox predicate, no `eval`/`exec`, no import escape, with an escalation ladder to a
visual/VLM check only as a last resort. An action step with no predicate **fails** the gate.
This is the moat — the reason a heterogeneous fleet of bring-your-own models can run
trustworthily on the industrial floor.

---

## 5. Block diagram

```
+----------------------------------------------------------------------+
|                              AGENT KERNEL                             |
|     (pure Python; no robot/ROS2 import; runs on macOS; generic)      |
|                                                                      |
|  VectorEngine    tool-use loop | native producer | VGG entry points  |
|  Backends        Anthropic | OpenAI-compat | local                   |
|  Producer        native_loop.run_turn_native  (DEFAULT; model-driven |
|                  ReAct: skills + code tools + synthetic verify/finish)|
|  Cognitive spine GoalDecomposer -> GoalTree (frozen DAG)             |
|   (cognitive/)   StrategySelector -> skill|primitive|code|tool|cap   |
|                  GoalExecutor (topo-sort, timeout, capture output)   |
|                  GoalVerifier (AST sandbox; evaluate -> (bool, raw)) |
|                  VGGHarness (plan-act-verify-replan loop)            |
|                  Blackboard (per-run obs; ${step.output.path} bind)  |
|                  StrategyStats | ExperienceCompiler/TemplateLibrary  |
|                  CapabilityRegistry (chat + grounding-dino detector) |
|  ===== HONEST-VERIFY MOAT (frozen spine) ==========================  |
|   trace_store | actor_causation | evidence_classifier  (frozen)     |
|   grades every step vs INDEPENDENT GT; only ever stricter (R5)      |
|  General tools   file/bash/glob/grep/web                            |
|  Session | Permissions (8-layer) | IntentRouter                     |
+---------------------------------+------------------------------------+
                                  | a World registers 4 things:
                                  |  1 tools   2 verify namespace
                                  |  3 decompose vocab (from registry)
                                  |  4 persona / prompt
        +-------------------------+---------------------------+
        |                                                     |
   DEV world                                          ROBOT world
   (laptop, no robot)                          Go2 | G1 (has_base) | SO-101 arm
   build/test MEANS                            embodiments = robot.yaml configs
   chat capability                             skills/primitives + detect capability

   The kernel NEVER imports a world; worlds register INTO the kernel.
```

Two entry points — the `vector-cli` REPL and the `vector-os-mcp` server — share this one
engine. Bare `vector-cli` + NL runs the **native** producer by default
(`VECTOR_REPL_NATIVE=0` = reversible legacy hatch).

---

## 6. Invariants / contracts

The architectural form of the CLAUDE.md Rules. Anything that violates them is a bug.

- **Kernel/world seam.** The kernel never imports a world; a world crosses only by the four
  registrations. Robot specifics never leak into kernel code paths.
- **Single-source vocabulary.** The decompose vocab is derived from the world's
  skill/capability registry, so prompt and validator allowlist are one set by construction;
  on failure, fall back to a **neutral** vocab — never another domain's defaults.
- **Verify is a deterministic predicate.** `GoalVerifier.evaluate() -> (bool, value)`; the
  sandbox is only ever stricter than plain Python; machine-checkable, not an LLM judge.
- **Closed-loop observation.** Each step's output is written to the per-run Blackboard;
  downstream params bind via `${step.output.path}`; `world_context` is rebuilt on every
  (re)decompose so the planner reasons over the latest state.
- **Frozen-dataclass, additive-only.** Plan/config structures (`GoalTree`, `SubGoal`,
  `StepRecord`, `EmbodimentConfig`) are frozen; evolve by adding a field last with a default.
- **World-agnostic mechanisms.** Decompose, verify sandbox, executor, harness, selector, and
  the stats bandit hold no domain knowledge — it lives only in the registered four things.
- **Fail loud.** An unresolved strategy or a missing config key surfaces a clear error
  (with the valid set / offending key) and feeds the replan context — never a silent fallback.
- **Embodiment as config.** A morphology is a `robot.yaml` + assets read by one generic
  driver, never a bespoke per-robot class; every body conforms uniformly to `BaseProtocol`.
  If adding a robot needs a kernel/driver edit, the difference belongs in config.
- **Arm motion reports what happened.** `move_joints` returns True only when the joints
  settled within the manifest `track_tol` (a blocked arm returns False); `stop()` and an
  abort preempt it within one set-point period. Arm IK/FK always use the LIVE base pose.
  A sim grasp is physical first: the gripper welds an object only when every finger body
  touches it while it holds the jaws open, and `is_holding(name)` reads that live weld.

---

## 7. Conceptual module map

One terse line per package. No line numbers (they rot — read the file). Paths relative to
`vector_os_nano/`.

- `sim/nav/ros2/control.py` — optional session-owned external NAV STACK control:
  FAR goals or TARE exploration select one waypoint stream for the existing CMU
  localPlanner/pathFollower. A dedicated ROS executor caches commands; the skill
  thread applies them under the base control lease. Stale routes, paths or commands
  stop motion; cancellation disarms the lease. This mode has no authored-map planner
  fallback. `SimSession` owns the stack processes; `SessionViz` publishes sensors
  and command provenance. Arrival checks the original requested goal.
- `vcli/engine.py` + `vcli/backends/` — VectorEngine (tool-use loop, dispatch, VGG entry
  points, per-world verify-namespace binding) and the model adapters behind a common type.
- `vcli/native_loop.py` — the default model-driven producer: a native tool-use ReAct loop
  (world skills + code tools + synthetic `verify`/`finish`) assembling an `ExecutionTrace`;
  imports ONLY the honest spine, never the planner modules (firewall-tested).
- `vcli/cognitive/` — the cognitive spine **including the frozen honest-verify moat**:
  `goal_decomposer` · `goal_verifier` · `goal_executor` · `vgg_harness` · `strategy_selector`
  · `strategy_stats` · `blackboard` · `vocab_from_registry` · `template_library` ·
  `experience_compiler` · `types` (frozen plan structures) · `observation` (JSON-safe export
  surface) · the frozen moat: `trace_store` · `actor_causation` · `evidence_classifier`.
- `vcli/verdict.py` — the CLI-turn acceptance instrument (`VerdictReport`) that CONSUMES the
  spine (re-uses `evidence_passed`); used by `-p/--json` and the native producer. It is
  top-level `vcli`, NOT under `cognitive/`.
- `vcli/cognitive/capabilities/` — the Capability seam (`Capability` protocol +
  `CapabilityRegistry` + `LLMChatCapability`); the bridge to a heterogeneous model zoo.
- `vcli/worlds/` — `base.py` (the four-thing `World` protocol), `dev.py`, `robot.py`,
  `registry.py` (`WorldRegistry` resolution), `world_manifest.py` (loader + name resolver
  for `config/worlds/<id>.yaml`: rooms, surfaces, pickables, user location) and
  `sim_oracles.py` — the ONE module of ground-truth verify predicates, bound to the live
  SimSession's `sim/ground_truth.GroundTruth` (skills/perception never import either).
- `embodiments/` — **embodiment-as-config**: `config.py` (`EmbodimentConfig` +
  `CapabilityProfile` schema + fail-loud loader), per-robot `<id>/robot.yaml` manifests
  (`go2`, `g1`, `so101`, and `go2_piper` whose typed `arm` block — joint/actuator/site
  names, home, IK seeds, track tolerance, gripper finger/weld bodies — drives
  MuJoCoPiper + MuJoCoPiperGripper), and `capability_profile.py` — the ONE resolver (`resolve_capability_profile(agent)`)
  every "what can this body do" gate consults (Rule 11, no capability-by-code drift): the
  native producer's navigate gate, `worlds/robot._agent_has_camera`, and `engine._has_base`
  all single-source through it (gated flags runtime-authoritative; enrichment flags reconcile
  runtime-OR-declared for the runtime-attach, e.g. go2+Piper).
- `perception/` — perception backends incl. `detector_capability.py` (the grounding-dino
  open-vocabulary `detect`, registered by `RobotWorld` onto a camera embodiment).
- `skills/` — the `@skill` library (declares `requires` arm|base|camera) + SkillFlow routing.
- `hardware/` + `hardware/sim/` — `base.py` `BaseProtocol` (uniform across bodies) and the
  MuJoCo sim drivers (go2 / arm) every body's config describes.
- `vcli/tools/` · `intent_router.py` · `dynamic_prompt.py` · `prompt.py` · `session.py` ·
  `permissions.py` — general tools, category-filtered routing, the composable system prompt,
  persona blocks, conversation/transcript state, and the 8-layer permission gate.
- `mcp/` + `ros2/` + `integrations/` — the `vector-os-mcp` server entry point, the ROS2
  bridge (high-fidelity Linux backend), and external integrations.
