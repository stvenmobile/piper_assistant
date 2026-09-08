# Research Precedents: Feature Decomposition & Mechanistic Interpretability

Compiled 2026-09-08, as a follow-up to [[research-precedents]]. Prompted
by a specific question raised in discussion: is a "concept" inside a
model atomic (one whole unit, the way a Chinese character maps to one
idea) or compositional (built from smaller, reusable primitives, the way
English words are built from an alphabet of sounds)? This matters
directly for whether a shared "language" between AI models should be
built at the level of whole concept vectors (what ALIGNQ's Procrustes
rotation currently does) or at the level of more primitive, disentangled
building blocks underneath them.

## The short answer

Compositional, not atomic - and there's a specific, well-developed body
of technique for finding the primitives. Models pack far more concepts
than they have dimensions by overlapping them ("superposition"), which
means a raw concept vector is usually a tangled combination of many
smaller features rather than one clean unit. A separate line of work
(sparse autoencoders) exists specifically to untangle that combination
back into something closer to an "alphabet."

## Why linear rotation approaches like ALIGNQ's make sense at all

**The Linear Representation Hypothesis.**
[Overview](https://www.emergentmind.com/topics/linear-representation-hypothesis-lrh)

The hypothesis that meaningful concepts - sentiment, truthfulness,
language identity, even more abstract notions - are represented as
*linear directions* in a model's activation space: literally a vector you
can add or subtract (the classic word-embedding example: `king - man +
woman ≈ queen`). This is the underlying reason a simple linear/rotation
map (Procrustes) is a sensible tool to reach for at all - if concepts
weren't roughly linear, no rotation would be expected to preserve them
well between two spaces.

**Caveat worth keeping in mind**: this isn't universally true. Recent
work has found some features - days of the week, months of the year -
are represented as *circular* structures rather than straight-line
directions, evidence against a strict version of the hypothesis. Not
everything in these models is a flat line through the origin; some
concepts have real curved/manifold geometry. Worth remembering as a
limitation on how far a purely linear map can be expected to generalize.

## Why a raw concept vector is usually a tangle of several ideas

**Elhage, Olah et al. - "Toy Models of Superposition," Anthropic 2022.**
[transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html) /
[arXiv:2209.10652](https://arxiv.org/abs/2209.10652)

Models represent *more* features (concepts) than they have neurons or
dimensions available, by encoding many of them as overlapping,
non-orthogonal directions that share the same dimensions - "superposition."
This is why a single neuron often appears to respond to several unrelated
things at once (polysemanticity): it's not disorganization, it's a
deliberate-looking compression strategy that works because most concepts
are used rarely, so the interference between them is usually small.
The paper also finds superposed features organize into specific geometric
structures (digons, triangles, pentagons, tetrahedrons) - there's real,
discoverable structure to how the compression is arranged, not just noise.

**Practical implication for ALIGNQ**: a whole concept vector - the thing
`congruence_optimizer.py` currently rotates as a single unit - is
probably several entangled ideas at once, not one clean "character."
Aligning at that level works (the 289-trial results prove it does,
imperfectly), but it's aligning bundles, not primitives.

## The technique for untangling the bundle

**"Towards Monosemanticity" (2023) and "Scaling Monosemanticity:
Extracting Interpretable Features from Claude 3 Sonnet" (2024), Anthropic.**
[transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

Trains a sparse autoencoder (SAE) - an auxiliary model - to re-expand a
model's compressed, superposed activations into a much larger space where
individual features become monosemantic: each one reliably fires for one
specific, human-recognizable idea (the well-known example found in Claude
3 Sonnet: a feature that activates specifically for sycophantic praise).
This scaled successfully to a real production-sized model, not just toy
examples. This is, concretely, the "find the alphabet" project - the SAE
discovers a dictionary of more primitive building blocks that the raw,
tangled activation vector turns out to be a combination of.

## Is that alphabet shared across different models?

**Gurnee et al. - "Universal Neurons in GPT2 Language Models."**
[arXiv:2401.12181](https://arxiv.org/abs/2401.12181)

**"Sparse Autoencoders Reveal Universal Feature Spaces Across Large
Language Models."** [arXiv:2410.06981](https://arxiv.org/html/2410.06981)

About 1-5% of neurons are "universal" - consistently activating on the
same inputs across independently-trained models, causally important, and
often individually interpretable. SAE-derived features (the disentangled
kind, not raw neurons) show even broader cross-model universality. This
is direct empirical evidence, at a finer mechanistic grain than the
Platonic Representation Hypothesis' broad geometric argument, that at
least part of the "alphabet" genuinely is shared across models trained
independently of each other.

## The most directly actionable finding for cross-model ALIGNQ work

**Crosscoders / model diffing, Anthropic.**
[Diff tool announcement](https://www.anthropic.com/research/diff-tool) /
["Cross-Architecture Model Diffing with Crosscoders" (2026)](https://arxiv.org/html/2602.11729v1)

A crosscoder is a sparse-autoencoder variant trained on *two models at
once*, learning one shared feature dictionary that explains both models'
activations simultaneously - automatically surfacing which features are
common to both versus unique to one. The 2026 extension explicitly
applies this across genuinely different architectures with different
tokenizers and hidden dimensions - the paper's own example pairing is
**Llama vs. Qwen**, the exact kind of cross-model, cross-architecture
comparison this project is moving toward next.

This is a more powerful, finer-grained cousin of what `congruence_optimizer.py`
does: instead of rotating whole concept vectors between two models and
measuring how well they land, a crosscoder decomposes both models down to
primitive features first and directly identifies which ones correspond.
Not something to implement immediately - it's a heavier technique
requiring its own training run - but worth keeping as the natural next
step if whole-vector rotation turns out to plateau on the cross-model
work, or if the question shifts from "how well do these models align
overall" to "which specific concepts transfer and which don't."

## Bottom line

The "logographic vs. alphabetic" framing from the discussion that
prompted this note turns out to map onto real, current interpretability
research quite precisely: raw model representations are closer to a
compressed, overlapping tangle than clean atomic symbols (superposition),
there's a working technique for decomposing that tangle into something
alphabet-like (sparse autoencoders / monosemantic features), and there's
real evidence that at least part of that alphabet is shared across
independently-trained models (universal neurons, universal SAE features)
- with a purpose-built tool (crosscoders) for finding that overlap
directly between two specific models, already demonstrated on a Llama-vs-Qwen-shaped
comparison.
