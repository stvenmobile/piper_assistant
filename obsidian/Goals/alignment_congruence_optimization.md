---
id: GOAL-ALIGNMENT-CONGRUENCE-OPTIMIZATION
title: Alignment Congruence Optimization
prefix: ALIGNQ
status: active
created: 2026-09-07
target_model: Qwen/Qwen2.5-0.5B-Instruct
---

# Goal: Alignment Congruence Optimization

**Status**: `active`
**Target**: Assess and improve the quality of Procrustes rotations between
residual-stream layers before trusting them for concept-transfer testing,
by measuring two distinct things every trial rather than one:

1. **In-sample congruence** - how well an orthogonal rotation fits the
   calibration data it was built from (via the SVD singular values other
   Procrustes code in this repo currently discards). 1.0 means a rotation
   exists that fits the calibration set essentially exactly; near 0 means
   no orthogonal map fits well, regardless of calibration size.
2. **Held-out generalization** - cosine similarity and top-1
   nearest-neighbor accuracy on a fixed set of concepts never used for
   calibration, so numbers are comparable across trials.

A rotation can fit its calibration set closely and still fail to
generalize (or vice versa is impossible, but the gap between the two is
itself informative) - logging both across many trials is the point.

## What each trial varies
Implemented in `src/piper_geometry/congruence_optimizer.py`
(`CongruenceOptimizer.run_trial`), dispatched from
`PiperSupervisor._run_alignq_trial`:
- **Layer pair**: (6,10), (8,14), (10,16), (12,18) on the target model
- **Calibration set size**: 12 / 24 / 48 / 96 / 192 / 280 concepts, sampled
  from `data/checkpoints/concepts_dictionary.json` (280 is the pool's
  practical ceiling - 300 total concepts minus 16 reserved for held-out -
  not the 896-dim hidden size; the first 27 trials at 12-48 showed
  congruence pinned near 1.0 regardless of layer pair, the expected result
  when calibration size is far below the embedding dimension, so this was
  widened to see whether congruence trends down as size approaches what
  the dictionary can actually provide)
- **Centering**: whether to mean-subtract calibration vectors before
  computing the rotation (aligner.py doesn't; align_agents.py does - this
  was an unmeasured inconsistency between the two)

16 concepts are held out from the same dictionary, fixed by a constant
random seed, and reused unchanged across every trial regardless of
calibration size, so accuracy/cosine numbers stay comparable trial to
trial.

## Why this track exists
Prior same-model same-layer runs (Archive_Layer22_k4, 2026-08-27/28)
plateaued at a flat 0.264 cosine similarity with no exploration of layer
choice, calibration size, or centering - so it was never clear whether
0.264 reflects a real ceiling on what's alignable at that layer, or just
an under-explored corner of these three variables. This track is meant to
answer that before trusting any concept-transfer accuracy number test
built on top of a rotation whose fit was never actually checked.

## Success Criteria
- Enough completed trials across the (layer pair x calibration size x
  centering) grid to see which combinations produce high congruence *and*
  high held-out accuracy together, versus high congruence with poor
  generalization (overfitting) or the reverse.
- A follow-on decision, informed by that data, on whether to extend this
  same approach to genuinely different models (not just layers of one
  model) - the comparison Piper's broader research program is ultimately
  aimed at.
