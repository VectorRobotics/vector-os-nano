# Vector OS Nano — Redesign Brief (input for a from-scratch restart)

**Date:** 2026-06-18 · **Status:** Tier-4 plan input (delete once the new spec exists).
**Source:** 6-agent adversarial review (3 code reviews + doc review + 2 direction critics, all opus,
read-only) + a full real test run. This is ANALYSIS to inform the owner's redesign — it is **not**
the new design. The owner decides the new design next session.

> The owner's call ("方向上出现错误了 / the direction is wrong") is **correct**. Six independent
> reviewers converged on the same root causes below — this is not one opinion, it is a consensus.

---

## 1. Test & quality reality (honest)

- Suite is **green by the documented tolerances**: 891 (unit/core+vcli) + 41 (playground) + 850
  (vcli) = **1782 passed**, only the 3 known deepseek `.env` reds. **But green ≠ working.**
- The recurring failure mode was **harness-vs-real-cli false-greens**: capabilities were marked PASS
  via ~55 out-of-repo `~/sandbox/*.py` scripts that hand-built the engine and bypassed `cli.main`,
  then failed in the real `vector-cli` (owner: "全是bug — no switch / no VLN / no photoreal"). 35
  test files construct the engine directly; only 3 touch `cli.main`. **83K test LOC certified broken
  features.** The green suite is therefore weak evidence, by construction.

## 2. Where the direction went wrong (the 3 root causes)

1. **The "verify-is-the-moat" thesis does not transfer to a real robot — the stated END.** Deterministic
   verify (`at_position`/`visited`/`holding_object`) anchors on **ground-truth coordinates the sim
   seeds at boot** (`g1_runtime.py:147`, `go2_runtime.py:110`). Strip the sim oracle (real hardware)
   and the predicates have nothing to anchor on. The moat is real in the hardware-free dev world and
   **theatrical in the robot world**. It is also *relaxed exactly where it matters* — `RobotWorld`
   exempts motor skills from the evidence gate, `visual_override` launders a failed predicate into
   `success=True`, and the robot reward path collapses to `step.success`. And the LLM **authors both
   the goal tree AND the predicates**, so verification bias re-enters at authoring time. The one real
   differentiator is overclaimed.

2. **The GT oracle is co-located with the planner's world state, separated only by a flag → leaks by
   construction.** GT coords live in the same `WorldModel` the planner reads; safety = "every reader
   remembered to call `get_perceived_objects()` not `get_objects()`." That discipline failed **3+
   times** (`_live_objects_line`, `WorldQueryTool`, `navigate_to(label)`), each whack-a-mole patched.
   A moat held by an exclusion predicate threaded through every reader is not a moat.

3. **A bespoke VGG planner is the wrong build for 2026, and it's buried under 12 campaigns of
   accretion.** `goal_decomposer` (1235) + `goal_executor` (1459) + strategy-selector + experience-
   compiler + template-library reinvent what frontier tool-use loops now do out of the box — for
   near-zero demonstrated payoff (the learning tier's real hit-rate is ~0; templates are mostly
   rejected). The differentiator was never the planner; it was verification — but the code budget went
   to the planner. Symptoms: 2 cognitive layers never collapsed (ReAct `run_turn` + VGG, with chat
   wrapped in a fake 0-action "verified" GoalTree — Stage 5 "unify the paths" still unfinished after
   12 campaigns); a 488-line **keyword router** (`intent_router.py`) that *is* the rule-1 anti-pattern
   it forbids; the CLI re-implements routing inline so "one controller, two frontends" is false (MCP
   differs); embodiment-switch is a `@tool` not a `@skill`, forcing a whole segmentation epicycle
   (ADR-012); embodiment boot is dispatched from **4+ divergent sites** (and `/scenario` boots
   nothing — the live rule-11 gap); 8 files blow the 800-line rule (`cli.py` 2224, `engine.py` 2205,
   `mujoco_go2.py` 1920); 28 `VECTOR_*` env flags gate core behavior (each a rule-11 violation).

## 3. KEEP — the small defensible core (carry into the redesign)

- **The AST-sandboxed deterministic predicate evaluator** (`goal_verifier.py`, ~260 lines: blocked
  nodes, dunder reject, safe-builtins-only, hard timeout, `evaluate()->(bool,value)`, never
  eval/exec). The one piece that genuinely delivers "deterministic, never LLM-graded."
- **Frozen, inspectable, replayable plan data model** (`GoalTree/SubGoal/StepRecord` + closed
  `FailureClass` enum) + **Blackboard `${step.output.path}` pure-traversal binding** (observations,
  not success/error strings, flow forward). Transfers to *any* agent loop.
- **Kernel/world seam as a CONCEPT** (ADR-006: a world registers tools + verify namespace + decompose
  vocab + persona; kernel never imports a world) + the clean `WorldRegistry`, `Scenario` DTO,
  `SkillWrapperTool`, and `vocab_from_registry` single-source derivation.
- **Honest sensing substrate** (sensor-derived, never GT): `target_locate` geometry (the
  recognise→depth/lidar-locate→navigate insight beats pixel servoing), the shared world-agnostic
  `_nav_controller`, MuJoCo physics for G1/Go2/arm, the top-down grasp geometry.
- **The subprocess-over-socket isolation pattern** for GPL/heavy/incompatible-interpreter sims
  (Blender, habitat py3.9) — quarantine technique, reusable regardless of which substrate stays.
- **`CLAUDE.md` constitution + `ARCHITECTURE.md` §8 (honest positioning) + `tricky-bugs.md`** — the
  governance and the empirical record that kept the docs bounded and named the soft spots honestly.

## 4. DISCARD — accretion to drop

The bespoke VGG planner/decomposer/strategy-selector/experience-compiler/template-library · the
488-line keyword `intent_router` · GT-seeded-into-the-world-model moat + `is_verify_anchor` filtering ·
the dual ReAct+VGG turn loops + the answer-only fake-GoalTree ceremony + `VECTOR_LEGACY_TURN` · the
CLI's inline per-frontend routing · `switch_embodiment`-as-`@tool` + segmentation pre-pass (ADR-012) ·
the 4+ embodiment-boot dispatch sites + the do-nothing `enter_scenario` boot + the legacy `_init_agent`
`--sim` path · N-variants-per-verb skills (4 seek / 4 pick / 3 place) · dead isaac/gazebo/pybullet
backends (~1100 LOC, still in the live enum) · one of the two photoreal substrates (habitat conda +
Blender co-sim) · the dual perception stacks (Moondream vs Qwen) · the two spatial stores (`scene_graph`
1011 + `world_model` 482) · the 28 `VECTOR_*` flags · the 2200-line god-files · `~/sandbox` harness
scripts as acceptance · the cross-session bandit + template learning tier.

## 5. The 3 strategic corrections the redesign must make at the root

1. **Make the moat honest by construction.** A deterministic verify must anchor on something the
   system does NOT control — real sensor/ROS2 state, or an *explicitly-labeled* sim-smoke check never
   confused with acceptance. The verify oracle must live in a store the planner/means **physically
   cannot read** (separate namespace), so the whole leak class disappears. Decide up front whether
   predicates come from a **human-reviewed/curated library** (defensible) or stay LLM-authored (then
   "deterministic moat" is overclaimed).
2. **Spend the code budget on verification, not a planner.** Lean on a frontier model's native
   tool-use/structured-output loop for decompose+replan. **One** turn path, **one** unified action
   space (everything callable is one kind of thing → routing + embodiment-switch contortions vanish),
   no fork-around-verify, no keyword router. A from-scratch core targeting the §3 keepers is likely
   **<10K LOC** vs today's 59K — and more trustworthy.
3. **Acceptance = the product on real (or honestly-labeled) ground, not a harness.** Bare `vector-cli`
   (rule 11) is the *ergonomic* surface, but "done" must mean a real robot / hardware-faithful run
   whose verification **survives removing the oracle** — never a hand-built engine script. Position the
   thesis as a **thin, trustworthy verification+orchestration LAYER on top of ROS2/Nav2/VLA on real
   hardware**, not a from-scratch nav/manip stack proven in a hand-rolled photoreal sim.

## 6. Open question for the owner (the deepest one)

Is the bet **"we make frontier-model + existing robot stacks (ROS2/Nav2/VLA/foundation policies)
trustworthy via deterministic verification"** — or a bespoke autonomy stack? Every reviewer leaned to
the former: keep verification as the differentiator, let the mature stacks do nav/manip/perception.
That choice should anchor the new design.

---
*Full per-finding detail (file:line) is in the review run; this brief is the synthesis. History of the
old direction lives in git + ADRs + `tricky-bugs.md`.*
