# ADR-012 — Planner-composed embodiment switch (compound single-sentence cross-embodiment chains)

Status: ACCEPTED — Option A implemented & real-verified 3/3 (campaign #12 M4-B). Date: 2026-06-17.
(Implemented under the owner's "继续" directive as intra-vcli work — no kernel/interface/dep change,
so no hard CEO gate crossed; the DQ-18 release-to-master merge remains separately owner-gated.)

## Context

ADR-011 built the in-session switch seam: in ONE vector-cli session, G1⇄Go2 switch via
`switch_embodiment` (a `@tool`; single-source `_rebind_agent`). Campaign #12 M4(A) then proved
every capability works **turn-by-turn** (one NL command per turn) on both embodiments, real-sim
verified. The remaining north-star gap is **M4-B: one compound sentence that BOTH switches
embodiment AND acts** — e.g. "切到go2，然后往前走一米" — which scored **0/3** (harness
`~/sandbox/c12_m4_chains.py`).

A 4-agent read-only investigation (workflow `wf_3759aa06-8a7`) traced the exact mechanism, each
claim verified against `file:line`:

**Root cause — a single architectural gap, not three bugs:**
- A compound sentence trips `IntentRouter.is_complex()` → routes to **VGG decompose**
  (`intent_router.py:339`), which runs BEFORE the single-step switch bypass (`:347`).
- But the VGG decompose vocabulary is **single-sourced from `skill_registry.to_schemas()` — only
  `@skills`** (rule 3; `vocab_from_registry.py:341-365`). `switch_embodiment` is a `@tool`, so it is
  **structurally absent from the planner's action space** (`intent_router.py:177-178`: "@tool —
  UNREACHABLE on the VGG/decompose path"). The planner has no token to emit a switch step.
- Observed: chain A "切到go2…" → planner decomposed but **never switched** (stayed G1); chain C
  "换成go2…" → LLM **thrashed ~50 tools** (grep/file_read/bash/start_simulation) and bound the leg
  to the **wrong backend** (Go2ROS2Proxy, not MuJoCoGo2). (The T5 "(2,1) unreachable" issue was a
  separate, already-fixed M4(A) red herring — geodesic=inf fail-loud, rule 8 working as designed.)

**Two latent hazards any "switch as a plan step" design must handle:**
- **Stale-closure**: the in-flight `GoalExecutor` captures `_agent_ref`/`_skill_registry` at
  `init_vgg` time (`engine.py:505-533`); a mid-plan switch's `_rebind_agent` re-inits VGG, but the
  running executor keeps the OLD closure → later steps run on the dead/disconnected agent.
- **init_vgg-during-run**: swapping the executor/selector under a live `VGGHarness.run` loop is a
  lifecycle/concurrency hazard (unverified).

## Decision (PROPOSED — recommend Option A)

**Option A — controller-level switch segmentation (RECOMMENDED).** Before decompose, the
controller splits a compound command at embodiment-switch boundaries into ordered segments. Each
leading/boundary switch is executed via the **already-proven top-level path** (`switch_embodiment`
→ `_rebind_agent`, ADR-011), then the **residual** is routed normally (single skill or its own
decompose) **on the now-active embodiment**. Because the switch completes BEFORE any plan begins,
the residual is decomposed/executed entirely against the NEW agent — **both latent hazards vanish
by construction** (no mid-plan rebind, no stale closure).
- Files: `intent_router.py` (segment a compound at switch boundaries) + small orchestration in
  `engine.run_turn_unified` (run segments in order, rebinding between). NO
  executor/dispatcher/vocab/kernel change.
- rule 2/7 (kernel/BaseProtocol zero-change) ✓; rule 3 (no vocab duplication — switch stays a
  `@tool`, no new strategy) ✓; rule 6 (no dataclass change) ✓.
- 100% of the failing chains are "switch-first" → directly covered. Switch-in-the-middle
  ("walk, then switch, then walk") handled as N ordered segments.
- Verify (rule 5): switch confirmed **deterministically** at the segment boundary (active
  embodiment == target) before running the residual; residual legs keep their normal skill
  verifies. No new verify-namespace predicate required; the moat is not loosened.

**Option B — planner-native switch step (the investigation's RANK 1; future evolution).** Make
switch emittable as a VGG step via the existing `tool_call` strategy: build a robot-world
`ToolDispatcher` with allowlist `{switch_embodiment}`, fix `ToolDispatcher._make_context`
(`app_state=None` → live app_state), teach the robot vocab `tool_call` (reconcile against rule-3
single-source), AND fix the stale-closure + init_vgg-during-run hazards (make the executor's
agent/registry reference indirect). True *interleaved* composition, but larger blast radius and
real lifecycle risk. **Defer** until a use-case needs switch interleaved INSIDE a single decomposed
plan; Option A covers the realistic switch-first / segmented case first.

**Option C — switch as a real `@skill` (RANK 3; rejected).** Would auto-enter the vocab, but switch
needs the app_state/registry rebind that `SkillContext` does not carry → forces widening
`SkillContext` (rule-6 additive risk) plus skill/executor wiring. Biggest blast radius — rejected.

## Consequences
- New nodes/topics/interfaces: **none**. No msg/srv/action, no cross-package data-flow change, no
  new deps → **no further hard CEO gate** beyond approving this architecture direction.
- Model change: compound cross-embodiment commands become first-class (controller composes
  switch + residual); the switch operation itself remains a `@tool` (ADR-011 seam unchanged).
- Files (Option A): `vcli/intent_router.py`, `vcli/engine.py` (+ tests). kernel/BaseProtocol zero
  change.

## Milestone / acceptance (Option A)
- **TDD RED**: a unit test asserting "切到go2，然后往前走一米" yields segments
  [switch→go2, "往前走一米"] and (with a fake engine) runs the switch then the residual on the NEW
  agent; bash never fires; a switch-in-the-middle case yields the right ordered segments.
- **GREEN**: implement segmentation + ordered execution.
- **REAL-VERIFY (the only acceptance that counts)**: run `c12_m4_chains.py` (the 0/3 harness) in
  **real sim via vector-cli**, expect **3/3** (end-embodiment correct, `switch_embodiment` fired,
  no bash, real displacement ≥ 0.2 m); screenshot Read-back; `rosm nuke --yes` after; confirm no
  GL/thread/VRAM growth across the run.

## Risks
- Mis-segmentation of a compound sentence (where to cut). Mitigate: reuse the proven
  `_SWITCH_VERBS` + `_EMBODIMENT_TARGETS` detector (single-source, rule 3); fail loud (rule 8) on
  an ambiguous segment rather than guessing.
- Switch-in-the-middle ordering correctness — covered by an explicit multi-segment test.
- GL/EGL + MuJoCo physics-thread leak across repeated rebinds (historical crash class) — already
  mitigated by ADR-011's ≥5× switch leak check; re-confirm during the 3/3 real run.

## Outcome (2026-06-17, Option A shipped)

Implemented exactly as Option A: `IntentRouter.split_switch_segments` (splits on sequential
word-connectors + CJK punctuation only — NOT the ASCII comma, so coordinate lists survive) +
`VectorEngine._run_switch_segmented` (runs each segment as its own turn, re-reading the active
agent from `app_state` between segments → the rebind/stale-closure hazard is gone by construction).
The compound turn's `UnifiedTurnResult` now surfaces the UNION of tool_calls across segments, so
the switch is observable. Added one single-source move synonym ("挪") so a `换成go2，挪半米`-style
residual is honest. kernel/BaseProtocol untouched (rule 2/7); switch stays a `@tool`, no new vocab
(rule 3); frozen dataclasses unchanged (rule 6). 15 new unit tests (segmentation + the
rebind-threading) + chunked suite green (1718 + 41 playground; 3 tolerated deepseek `.env` reds).

**REAL-VERIFY (real MuJoCo sim, real LLM, `c12_m4_chains.py`): 3/3 (was 0/3).**
- A `切到go2，然后往前走一米`: G1→Go2, switch_embodiment fired, no bash, residual verified, moved 0.394 m.
- B `switch to g1, then navigate to x=1.2 y=0`: Go2→G1, switch fired, no bash, residual verified, moved 0.649 m.
- C `换成go2，挪半米`: G1→Go2, switch fired, no bash, moved 0.89 m (residual decompose carried no clean
  verify predicate → `verified=False`, but the robot physically moved — honest note, not a regression).

Option B (planner-native interleaved switch) remains the future evolution if a use-case needs a
switch INSIDE a single decomposed plan; not needed for the switch-first / segmented compound case.
