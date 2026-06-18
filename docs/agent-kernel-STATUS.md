# Vector OS Nano — STATUS (resume anchor)

One-page "where are we / what's next". Read this first; durable design is [ARCHITECTURE.md](ARCHITECTURE.md);
hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md); full round-by-round history is in `git log` +
the loop journal (`~/.vector-nano-loop/`).

updated: 2026-06-18
phase:   **DIRECTION UNDER CEO REDESIGN REVIEW — a from-scratch redesign starts next session.**
goal:    (current direction) NL controls a robot via decompose→plan→execute→verify→replan; verify is the moat.

## Where we are
- The owner has called the current direction wrong and will **restart the design from scratch**. A
  6-agent adversarial review (2026-06-18) confirmed it and produced **[REDESIGN-BRIEF.md](REDESIGN-BRIEF.md)** —
  read that next: the small defensible core to KEEP, the accretion to DISCARD, and the 3 root
  corrections. **No new build work on the old direction; nothing is in flight.**
- **Merged to `master` @ `f282180`:** campaigns #2–#11 (DQ-17, owner-approved 2026-06-15) — unified
  embodiment × capability matrix, MuJoCo+Blender photoreal co-sim, G1/Go2/arm locomotion+nav+VLN.
- **Branch `feat/playground-vln`** carries campaign-#12 follow-ups on top of master (NOT yet merged):
  GT-fallback rule-5 moat fix (`10c6a60`), project rule 11 + bare-`vector-cli` self-sufficiency part 1
  (`d28f28b`). These are real fixes but belong to the old direction — the redesign decides what carries.
- Test suite green by documented tolerances (1782 passed; 3 deepseek `.env` reds). Caveat: green ≠
  working — the old direction's recurring failure was green-suite-but-broken-in-real-cli (see brief §1).

## Honest soft spots (the redesign must confront — detail in REDESIGN-BRIEF.md)
- The deterministic verify "moat" anchors on sim-seeded ground truth → does not transfer to real
  hardware, and leaked to the planner repeatedly (co-located oracle).
- Two cognitive layers never unified; a 488-line keyword router contradicts rule 1; routing is forked
  per-frontend; 8 files blow the 800-line rule; 28 `VECTOR_*` flags; harness-not-product acceptance.

## Owner-gated (if the OLD direction were continued — likely moot under redesign)
- DQ-18 merge `feat/playground-vln`→master · DQ-16 M2 SysNav (`~/Desktop/SysNav/.venv-sysnav`
  re-provision) · DQ-15 M3 FAR ROS2 (heavy colcon). Decide these in light of the redesign, not before it.

## Read order for the redesign session
[REDESIGN-BRIEF.md](REDESIGN-BRIEF.md) → [ARCHITECTURE.md](ARCHITECTURE.md) §8 (honest positioning) →
[tricky-bugs.md](tricky-bugs.md) (why the moat leaked / harness false-greens) → ADR-006 (kernel/world
seam), ADR-007 (closed-loop), ADR-010 (co-sim).
