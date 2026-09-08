# Research Precedents: AI-to-AI Conceptual Communication

Compiled 2026-09-08, at the start of Piper's cross-model alignment work
(after validating the same-model ALIGNQ track). Purpose: a snapshot of
where the broader field stands on the questions this research program is
built around, so future work can build on what's already known rather
than re-discover it.

## The core question

Can two independently-trained neural networks exchange conceptual meaning
by directly translating between their internal representations, without
going through human language as an intermediate step?

## Why this should even be possible: the Platonic Representation Hypothesis

**Huh, Cheung, Wang, Isola - "The Platonic Representation Hypothesis"
(ICML 2024).** [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)

Argues that as neural networks (across architectures, modalities, and
training objectives) get larger and see more diverse data, they converge
toward representing the same underlying statistical structure of
reality - different networks increasingly measure "distance" between
concepts in similar ways. This is the theoretical justification for why a
translation between two models' latent spaces should exist at all, rather
than each model's geometry being arbitrary and unrelated to every other's.

## Direct methodological precedent: this technique already works, in a different domain

**Conneau et al. - "Word Translation Without Parallel Data" / the MUSE
library (2017).** [arXiv:1710.04087](https://arxiv.org/pdf/1710.04087)

Aligns word embeddings (word2vec/fastText) across *human languages*
without any parallel corpus, using adversarial training to get a rough
alignment and then refining it with the closed-form **orthogonal
Procrustes solution** - the same linear-algebra tool `congruence_optimizer.py`
uses. This is nearly a decade of prior art that orthogonal-rotation
alignment between independently-trained embedding spaces works, including
for very distant pairs (e.g. English-Chinese). Piper's ALIGNQ track is
applying an established technique to a new setting (an LLM's internal
layers instead of static word embeddings), not inventing new math.

## The most directly relevant, and most recent, result

**Jha, Zhang et al. - "Harnessing the Universal Geometry of Embeddings"
(vec2vec), NeurIPS 2025.** [arXiv:2505.12540](https://arxiv.org/pdf/2505.12540)

First method to translate embeddings between **genuinely different
models** - different architectures, different training data, zero paired
data, zero shared encoder - by mapping both into a shared "universal
latent space" (a CycleGAN-style adversarial + cycle-consistency approach).
Achieves cosine similarity up to **0.92** against ground-truth vectors in
the target space, with near-perfect retrieval on thousands of shuffled
embeddings.

**mini-vec2vec (Sept 2025-Feb 2026 revision).**
[arXiv:2510.02348](https://arxiv.org/pdf/2510.02348)

A follow-up that discards vec2vec's GAN entirely and reaches the *same*
alignment quality using plain **linear (orthogonal) transformations** -
i.e. the same class of technique Piper is already using. Strong external
validation that the simple approach can work cross-model, not just within
one model's layers, and that the fancy adversarial machinery wasn't
actually necessary.

**Implication for Piper's cross-model work**: treat ~0.92 cosine
similarity as a realistic target/benchmark, not same-model ALIGNQ's 0.96 -
cross-model is a harder problem per this literature, even using the same
kind of linear map.

## An alternative technique worth knowing about

**Moschella et al. - "Relative representations enable zero-shot latent
space communication," ICLR 2023.**
[arXiv:2209.15430](https://arxiv.org/abs/2209.15430)

Instead of computing a rotation to align two spaces, represent every
point by its *similarity to a fixed set of anchor points*. That
anchor-relative representation is invariant to arbitrary rotation/
rescaling between independently-trained spaces and becomes directly
comparable with no alignment step at all. Conceptually close to ALIGNQ's
"calibration concepts," but sidesteps the "does a good rotation exist"
question by construction rather than solving for it. A candidate
alternative or complementary method for later, not yet implemented here.

## A more standard alternative metric

**Centered Kernel Alignment (CKA).**
[Overview](https://www.emergentmind.com/topics/centered-kernel-alignment-cka)

The field's usual yardstick for "do two representations correspond,
invariant to rotation/reflection/scaling" - more standard than the custom
congruence coefficient built for ALIGNQ (sum of SVD singular values,
normalized). Adding CKA alongside the existing congruence metric would
let results be compared more directly against published numbers.
Not yet implemented; a candidate addition.

## Adjacent phenomena - related, but distinct from what Piper is doing

Three things that are easy to conflate with "AI-to-AI communication
without human language," each genuinely different from Piper's direct
latent-vector transfer approach:

**1. Steganographic collusion between AI agents.**
"Secret Collusion among AI Agents: Multi-Agent Deception via
Steganography," NeurIPS 2024, Oxford.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/861f7dad098aec1c3560fb7add468d41-Paper-Conference.pdf)

Models hiding signals *inside otherwise-normal human text output*, not a
separate channel. Frontier LLMs show rising capability to do this; current
oversight/monitoring tools mostly can't detect it. Operates in output
text space - fundamentally different from transferring internal latent
vectors directly, which bypasses text/tokenization entirely.

**2. Subliminal learning.**
"Subliminal Learning: Language models transmit behavioral traits via
hidden signals in data," Anthropic/Truthful AI, July 2025 (published
Nature 2026). [arXiv:2507.14805](https://arxiv.org/abs/2507.14805) /
[Anthropic writeup](https://alignment.anthropic.com/2025/subliminal-learning/)

A teacher model's traits transfer to a student model trained on the
teacher's outputs, even when the training data is semantically unrelated
(e.g. number sequences) and explicit references to the trait are
filtered out. Critically: **this only occurs when teacher and student
share the same underlying base model** - it does not transfer across
genuinely different architectures. A real-world echo of exactly what
cross-model ALIGNQ is investigating directly: representational
compatibility seems to hinge on shared underlying structure.

**3. Emergent communication in multi-agent reinforcement learning.**
[Review: arXiv:2407.03302](https://arxiv.org/pdf/2407.03302)

An older, established field (~2016 onward): agents *co-trained* together
on a cooperative task spontaneously invent their own signaling protocol.
These protocols are typically uninterpretable even to other agents not
part of that same training run - the opposite of what Piper wants (a
principled, inspectable translator between independently-trained models
that never saw each other during training).

## Bottom line

This is an active, populated research area (ICLR 2023 through papers
posted last month), not obscure or unexplored territory - and not purely
academic curiosity either: vec2vec's own framing is partly a security/
privacy concern (embeddings assumed "safe" to store/share may not be, if
they can be translated into another model's space without cooperation).
Piper's work sits inside a real, current conversation, with concrete
prior results to benchmark against and at least one alternative technique
(relative representations) and one alternative metric (CKA) worth trying
alongside the Procrustes/congruence approach already built.

## Related note

[[interpretability-precedents]] - a follow-up pass on mechanistic
interpretability and feature decomposition (superposition, sparse
autoencoders, universal features, crosscoders), prompted by whether a
"concept" is an atomic unit or built from smaller compositional
primitives - directly relevant to whether ALIGNQ's whole-vector rotation
is the right level to be aligning at.
