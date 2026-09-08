---
id: GOAL-CROSS-MODEL-ALIGNMENT-CONGRUENCE-DEEPSEEK
title: Cross-Model Alignment Congruence (DeepSeek-R1-Distill-Qwen-1.5B)
prefix: XALIGNDS
status: paused
created: 2026-09-08
source_model: Qwen/Qwen2.5-0.5B-Instruct
target_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
---

# Goal: Cross-Model Alignment Congruence (DeepSeek-R1-Distill-Qwen-1.5B)

**Status**: `paused`

**Paused 2026-09-08**: 96 trials collected. Cosine similarity tracked
XALIGNQ's TinyLlama results closely (0.904 vs 0.912 at calibration_size=280),
confirming the calibration-size trend generalizes across target models -
but held-out accuracy was notably worse (8.9-9.7% vs TinyLlama's
25-25.6%) at matching calibration sizes despite similar cosine similarity,
a genuine and still-unexplained divergence worth remembering. Moving to
[[cross_model_alignment_congruence_phi4]] (prefix XALIGNPHI), a
meaningfully larger model, per explicit request to move on to Phi-4-mini
next. Reactivate to dig into the accuracy-vs-cosine-similarity divergence
here specifically, or to extend this pairing's own per-cell sample sizes,
if warranted later.

**Target**: Same measurement as [[cross_model_alignment_congruence]]
(congruence + held-out cosine similarity/accuracy), on a second, larger
model pairing - a moderate step up in size before attempting anything as
large as Phi-4-mini-instruct (~3.8B).

## Why this pairing, and why now

[[cross_model_alignment_congruence]] (Qwen2.5-0.5B-Instruct vs
TinyLlama-1.1B-Chat-v1.0, 63 trials) confirmed the same monotonic
calibration-size trend ALIGNQ found for same-model layers, and at
`calibration_size=280` reached 0.912 cosine similarity - close to
vec2vec's own published 0.92 benchmark (see [[research-precedents]]) for
genuinely cross-model translation, using nothing more than a single
closed-form Procrustes rotation. Memory on the Jetson stayed comfortable
throughout (~8.2/15.3GB with both models plus Ollama's own model
resident) - real headroom to try something larger.

DeepSeek-R1-Distill-Qwen-1.5B is built on the Qwen2.5-1.5B base
architecture (28 layers, ~3GB in fp16) - a deliberate moderate increase
over TinyLlama's 1.1B rather than jumping straight to Phi-4-mini-instruct,
to confirm memory scales as expected with one more data point before
attempting the largest candidate.

## What each trial varies

Implemented in `src/piper_geometry/congruence_optimizer.py`
(`CROSS_MODEL_CONFIGS["XALIGNDS"]`), dispatched from
`PiperSupervisor._run_cross_model_trial("XALIGNDS")` - the same shared
handler XALIGNQ uses, just looked up under this prefix:
- **Calibration set size**: 12 / 24 / 48 / 96 / 192 / 280 (same grid as ALIGNQ/XALIGNQ)
- **Centering**: True / False - XALIGNQ's own result here was unexpected
  (center=False outperformed center=True, the opposite of the original
  hypothesis that two different models would benefit *more* from
  centering than two layers of one model did) but was based on thin
  per-cell samples (2-10 trials); worth re-testing on a second pairing
  before treating that finding as anything more than preliminary.

**Layer choice is fixed**: Qwen layer 18 (75% of 24) paired with
DeepSeek-R1-Distill layer 21 (75% of 28) - both at the same relative
depth used for the TinyLlama pairing, for consistency across comparisons.

## Success Criteria
- Enough trials across (calibration size x centering) for a direct
  comparison against XALIGNQ's numbers at matching settings - does a
  larger, architecturally-similar-to-Qwen target model align better,
  worse, or about the same as TinyLlama did?
- Confirmation (or correction) of XALIGNQ's centering finding with a
  second, independent pairing.
- Continued confirmation that Jetson memory headroom scales as expected,
  informing whether Phi-4-mini-instruct is a reasonable next step.
