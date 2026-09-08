---
id: GOAL-CROSS-MODEL-ALIGNMENT-CONGRUENCE-PHI4
title: Cross-Model Alignment Congruence (Phi-4-mini-instruct)
prefix: XALIGNPHI
status: active
created: 2026-09-08
source_model: Qwen/Qwen2.5-0.5B-Instruct
target_model: microsoft/Phi-4-mini-instruct
---

# Goal: Cross-Model Alignment Congruence (Phi-4-mini-instruct)

**Status**: `active`
**Target**: Same measurement as [[cross_model_alignment_congruence]] and
[[cross_model_alignment_congruence_deepseek]] (congruence + held-out
cosine similarity/accuracy), on the largest model pairing attempted so
far - a genuine step up rather than another moderate increment.

## Why this pairing, and the real risk worth naming

Two smaller pairings are done:
- XALIGNQ (TinyLlama-1.1B, 63 trials): 0.912 cosine similarity at calibration_size=280.
- XALIGNDS (DeepSeek-R1-Distill-Qwen-1.5B, 96 trials): 0.904 cosine
  similarity at calibration_size=280 - nearly identical to TinyLlama
  despite the larger model, but held-out *accuracy* was notably worse
  (8.9-9.7% vs TinyLlama's 25-25.6%) at matching calibration sizes despite
  similar cosine similarity - a reminder that cosine similarity alone
  doesn't predict how usable a result actually is.

Both prior pairings ran comfortably within the Jetson's memory (8.2GB,
then 9.7GB of 15.3GB total, both with Qwen + Ollama's own model also
resident). Phi-4-mini-instruct is a real step up - ~3.8B params, ~7.6GB in
fp16 versus DeepSeek's ~3GB - and extrapolating the memory trend from the
first two pairings puts total usage around 14GB, close enough to the
15.3GB ceiling to be a genuine, unconfirmed risk rather than a formality.
Worth watching the first trial's memory closely; if it's too tight,
stopping Ollama's resident chat model during a run is the first thing to
try before giving up on this pairing.

## What each trial varies

Implemented in `src/piper_geometry/congruence_optimizer.py`
(`CROSS_MODEL_CONFIGS["XALIGNPHI"]`), dispatched from the same shared
`PiperSupervisor._run_cross_model_trial("XALIGNPHI")` handler every
cross-model track uses:
- **Calibration set size**: 12 / 24 / 48 / 96 / 192 / 280 (same grid as every prior track)
- **Centering**: True / False - both XALIGNQ and XALIGNDS leaned toward
  center=False being at least as good as center=True; a third independent
  pairing showing the same direction would make that a settled default
  rather than a two-pairing pattern

**Layer choice is fixed**: Qwen layer 18 (75% of 24) paired with Phi-4-mini
layer 24 (75% of 32) - same relative-depth convention as every prior pairing.

## Plan
Let this run 6-7 hours unattended before the first check-in, rather than
the shorter first looks the earlier pairings got - a larger model means
slower trials, so a longer window is needed to reach a comparable sample
size across the calibration grid.

## Success Criteria
- A direct three-way comparison against XALIGNQ and XALIGNDS at matching
  calibration sizes: does a genuinely larger, architecturally different
  model (Phi-4-mini-instruct is not Qwen-architecture-based, unlike
  DeepSeek-R1-Distill) align better, worse, or similarly?
- A third data point on the centering question.
- Confirmation of whether the extrapolated memory risk was real, and if
  so, whether it's a hard ceiling on this approach's largest model choice
  or something workarounds (freeing Ollama's model) can absorb.
- A decision on whether other same-size-range models (candidates
  discussed earlier: TinyLlama and DeepSeek already covered; others in
  the ~1-4B range) are worth trying before concluding this line of
  cross-model testing.
