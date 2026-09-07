---
id: GOAL-PHASE3-CLOSED-LOOP-SIGNALING
title: Closed-Loop Multi-Turn Non-Verbal Signaling Protocol
prefix: P3LOOP
status: concluded
---

# Goal: Closed-Loop Multi-Turn Non-Verbal Signaling Protocol

**Status**: `active`
**Target**: Transition from single-shot latent transfer (Phase 2) to bidirectional, multi-turn non-verbal communication exchanges between transmitter and receiver models.

## Objectives
1. **Bidirectional State Synchronization**: Implement feedback loops where the receiver acknowledges or queries clarification via latent token packets.
2. **Multi-Turn Continuity**: Maintain conceptual coherence across 3 to 5 continuous token exchanges without reverting to plain text.
3. **Adaptive Thresholding**: Dynamically adjust InfoNCE temperature and attention pooling weights based on conversational feedback signals.

## Success Criteria
- Successful multi-turn communication loop execution with $>85\%$ semantic reconstruction consistency across sequential exchanges.
- Integration into `src/piper_brain/signaling_game.py` for automated simulation testing during idle compute windows.