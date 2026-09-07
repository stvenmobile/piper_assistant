---
id: GOAL-AUTOMATED-CONCEPT-CURATION
title: Autonomous Concept Curation and Interestingness Tagging
prefix: ACCIT
status: paused
---

# Goal: Autonomous Concept Curation and Interestingness Tagging

**Status**: `paused`
**Target**: Implement a deterministic scoring pipeline combining surprise, centrality, and entropy metrics to quantitatively rank and tag concepts of interest.

**Implementation note (2026-09-07)**: this track's trial handler
(`PiperSupervisor._run_accit_trial`) is currently a stub that logs a random
score - none of the objectives below are implemented yet. A duplicated
frontmatter delimiter also meant `status: active` was never actually read
by the code, so trials ran under a different track's label instead. Status
set to `paused` until the real pipeline exists; see Objectives for what
that requires (per-token loss capture doesn't exist anywhere in the
codebase yet, and the semantic vault currently has too few interlinked
concepts for centrality scoring to be meaningful).

## Objectives
1. **Prediction Error Metrics**: Compute rolling token perplexity and loss variance $Var(L)$ to flag cognitive surprise spikes where $Var(L) > 2.5\sigma$.
2. **Topology & Centrality Scoring**: Compute graph betweenness centrality $BC(v)$ over the semantic vault to prioritize bridge concepts.
3. **Composite Interestingness Formula**: Score concepts using the weighted function: 
   $$\text{Score}(c) = w_1 S(c) + w_2 C(c) + w_3 E(c)$$
4. **Autonomous Nomination**: Automatically generate an active Obsidian goal track when $\text{Score}(c) \ge 0.85$.

## Success Criteria
- Automated Python scoring pipeline parsing reasoning logs and computing composite interestingness scores.
- Zero false-positive goal generation during baseline testing against known trivial concept clusters.