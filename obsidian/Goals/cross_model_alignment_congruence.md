---
id: GOAL-CROSS-MODEL-ALIGNMENT-CONGRUENCE
title: Cross-Model Alignment Congruence
prefix: XALIGNQ
status: active
created: 2026-09-08
source_model: Qwen/Qwen2.5-0.5B-Instruct
target_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
---

# Goal: Cross-Model Alignment Congruence

**Status**: `active`
**Target**: Extend [[alignment_congruence_optimization]]'s validated
measurement approach (in-sample congruence + held-out cosine similarity/
accuracy) from two layers of one model to two layers of two genuinely
different models - the comparison this whole research program was
ultimately aimed at.

## Why now, and why this pair first

The same-model ALIGNQ track (289 trials, 2026-09-07/08) validated the
whole approach: congruence at small calibration sizes was structurally
inflated and meaningless, cosine similarity climbed from 0.61 to 0.96 as
calibration size grew toward the concept dictionary's ceiling, deeper
layer pairs generalized better than shallow ones, and centering made no
measurable difference for same-model layers. That gives a working recipe
to carry into the harder cross-model problem, rather than debugging the
measurement approach and the cross-model question at the same time.

Starting with the smallest reasonable model pair - **Qwen2.5-0.5B-Instruct
(24 layers) vs TinyLlama-1.1B-Chat-v1.0 (22 layers)** - deliberately,
before attempting anything heavier: two models resident on the Jetson
simultaneously uses meaningfully more memory than ALIGNQ's one, and this
pair was untested on this hardware before committing to it.

## What each trial varies

Implemented in `src/piper_geometry/congruence_optimizer.py`
(`CongruenceOptimizer(target_model_name=...)`, `CROSS_MODEL_PARAM_GRID`),
dispatched from `PiperSupervisor._run_xalignq_trial`:
- **Calibration set size**: 12 / 24 / 48 / 96 / 192 / 280 (same grid as ALIGNQ)
- **Centering**: True / False (same grid as ALIGNQ - worth re-testing
  here specifically, since two genuinely different models are more likely
  to have a real offset between their representation spaces than two
  layers of one model did)

**Layer choice is fixed, not swept, for this first batch**: Qwen layer 18
(75% depth) paired with TinyLlama layer 17 (~77% depth) - both picked at
comparably deep, abstract-representation territory rather than reusing
ALIGNQ's specific "shallow source / deep receiver" pairing, which was a
finding about one model's evolving residual stream and isn't obviously
the right lesson to carry into a two-separate-models comparison. An
approximate starting guess, revisitable once results are in - see
[[research-precedents]] and [[interpretability-precedents]] for the
literature this choice and the broader approach are grounded in.

## What "success" looks like here

Per [[research-precedents]], recent published work (vec2vec, NeurIPS
2025) achieved 0.92 cosine similarity translating between genuinely
different models with zero paired data - that's the realistic benchmark
for this track, not ALIGNQ's same-model 0.96. Getting meaningfully closer
to that number as calibration size grows would be a strong result;
falling well short even at calibration_size=280 would suggest either this
layer pairing is a poor match, or whole-vector rotation isn't the right
level to align at for genuinely different architectures (see
[[interpretability-precedents]]'s crosscoders section for the natural
next technique to reach for if so).

## Success Criteria
- Enough completed trials across (calibration size x centering) to see
  whether cosine similarity trends upward with calibration size the way
  it did for ALIGNQ, and whether centering matters here where it didn't
  for same-model.
- A comparison against ALIGNQ's same-model numbers at matching
  calibration sizes, to quantify how much harder cross-model alignment
  actually is in practice.
- A follow-on decision on whether to sweep layer choice, try a heavier
  model pair, or move toward a feature-level technique (crosscoders)
  instead of whole-vector rotation.
