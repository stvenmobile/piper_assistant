---
---
id: GOAL-AUTOMATED-CONCEPT-CURATION
title: Autonomous Concept Curation and Interestingness Tagging
prefix: ACCIT
status: active
---

# Goal: Autonomous Concept Curation and Interestingness Tagging

**Status**: `active`
**Target**: Implement a deterministic scoring pipeline combining surprise, centrality, and entropy metrics to quantitatively rank and tag concepts of interest.

## Objectives
1. **Prediction Error Metrics**: Compute rolling token perplexity and loss variance $Var(L)$ to flag cognitive surprise spikes where $Var(L) > 2.5\sigma$.
2. **Topology & Centrality Scoring**: Compute graph betweenness centrality $BC(v)$ over the semantic vault to prioritize bridge concepts.
3. **Composite Interestingness Formula**: Score concepts using the weighted function: 
   $$\text{Score}(c) = w_1 S(c) + w_2 C(c) + w_3 E(c)$$
4. **Autonomous Nomination**: Automatically generate an active Obsidian goal track when $\text{Score}(c) \ge 0.85$.

## Success Criteria
- Automated Python scoring pipeline parsing reasoning logs and computing composite interestingness scores.
- Zero false-positive goal generation during baseline testing against known trivial concept clusters.