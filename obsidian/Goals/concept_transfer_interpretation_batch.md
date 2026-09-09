---
id: GOAL-CONCEPT-TRANSFER-INTERPRETATION-BATCH
title: Concept Transfer Interpretation Batch
prefix: CTRANSFER
status: active
created: 2026-09-09
source_model: Qwen/Qwen2.5-0.5B-Instruct
target_model: microsoft/Phi-4-mini-instruct
---

# Goal: Concept Transfer Interpretation Batch

**Status**: `active`
**Target**: Run enough interpretation-test trials (`concept_transfer_test.py`,
four conditions: rotated / random-rotation / self round-trip / real text)
across all six concept-dictionary domains to see how consistent the
pattern found so far actually is, and what the failure modes look like in
aggregate, before writing up a conclusion from a single anecdotal run.

## The pattern established so far (single-run, not yet confirmed at scale)

- **Real text**: coherent, correct - confirms the harness works.
- **Self round-trip** (target model's own native layer-24 state,
  re-injected through the same mechanism, no translation): grammatically
  fluent but wrong topic - confirms the injection mechanism itself is not
  the problem.
- **Rotated** (calibrated cross-model translation, rescaled to match
  target's native vector magnitude): numerically stable across two runs,
  but still produces degenerate repetitive output, not yet coherent.
- **Random rotation** (negative control): crashed outright both times
  with a CUDA device-side assert (`probability tensor contains inf, nan,
  or element < 0`), with two different unseeded random matrices - a
  much more severe failure than the calibrated rotation's, suggesting
  the calibrated rotation is landing somewhere numerically sane (a
  learned region of Phi-4-mini's activation space) that an arbitrary
  random direction does not, even after matching for scale.

That last point is the real open question this batch is meant to answer:
is "calibrated survives, random crashes" a consistent, real distinction,
or did two runs just get lucky/unlucky on which side of a coin flip they
landed on? 50 single-run trials isn't enough to know.

## Why this runs differently from every other track

**This does not run through `main.py`'s autonomous idle loop** the way
ALIGNQ/XALIGNQ/XALIGNDS/XALIGNPHI do - not an oversight, a deliberate
choice. Once a CUDA device-side assert fires (which the random-rotation
condition has now triggered twice), the CUDA context is left corrupted
for the rest of that process; there is no way to recover and safely run
more trials in the same process afterward. Every other track's trial
handler runs in-process inside the long-lived `main.py` supervisor
precisely because it's never at risk of an unrecoverable crash - this one
is, so it needs process isolation per trial instead.

Run via `src/piper_geometry/ctransfer_batch.py` (`python3
src/piper_geometry/ctransfer_batch.py [num_trials]`, defaults to 50) -
launches `concept_transfer_test.py --domain <domain>` as a **fresh
subprocess per trial**, cycling randomly through all six domains, so one
trial's crash only costs that trial - the next one starts with a
genuinely clean CUDA context regardless. Logs a running summary
(success/crash count, per-domain breakdown) to
`obsidian/Experiments/ctransfer_batch_log.md`, plus each individual
trial's own full four-condition output via the same `CTRANSFER-*.md` note
format `concept_transfer_test.py` already writes.

## Success Criteria
- Confirms (or refutes) whether the calibrated-survives/random-crashes
  pattern holds consistently across domains and repeated runs, not just
  the two seen so far.
- Surfaces whether crash rate or output quality correlates with domain -
  is this pattern universal, or specific to physics-flavored content?
- Enough real examples of the rotated condition's output, across
  different domains, to judge whether "numerically stable but
  incoherent" is the whole story or whether some domains/passages
  produce noticeably better results than others.
- A journal entry once there's enough data to say something real about
  the pattern, rather than generalizing from n=2.
