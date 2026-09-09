"""
Piper Geometry: Concept Transfer Interpretation Test.

congruence_optimizer.py answers "does a translated vector land near the
right geometric target." This script answers a different question: does
the receiving model actually *behave* as if it understood translated
content, given nothing but that content to generate from? Those are not
the same thing - recognizing that a signal is meaningful and correctly
interpreting it are different capabilities (the same way hearing an
unfamiliar human language, you know it's language before you know what
it means, if ever).

Two design decisions worth recording, since both came from correcting a
real mistake in the first version of this script:

1. FULL-SEQUENCE translation, not isolated concept vectors. The first
   version translated four separate concept summaries (each a
   last-token vector from an unrelated short phrase) and injected them
   as four disconnected slots. That's a reasonable design for measuring
   alignment (congruence_optimizer.py's job), but it strips away all the
   context a real passage carries - a receiving model has nothing to
   work with beyond four floating, mutually-unrelated points. Here,
   every token of a real passage is translated and the whole sequence is
   injected together, preserving the relational structure between
   positions that generation actually depends on.

2. Injection at the model's OWN layer 24 via a forward hook, not via
   generate(inputs_embeds=...). inputs_embeds only ever supplies the
   model's *input* layer (before any of its transformer blocks run) -
   but the validated rotation maps onto Phi-4-mini's LAYER 24 hidden
   state, a representation that's already been through 24 layers of
   processing. Feeding that in as inputs_embeds forces a mid-stack
   snapshot through all 32 layers as if it were raw input text - a
   category mismatch, not a fair test of the translation. A forward hook
   on layer 24 overrides the residual stream at exactly the layer the
   rotation was calibrated for, and only for the initial prefill pass -
   every token generated afterward runs layers 25-32 normally, same as
   it would for any other input.

Four conditions, same passage, side by side:

1. ROTATED: every token of the passage extracted from the source model
   at its calibrated layer, translated via the validated cross-model
   rotation, injected into the target model at its own matching layer.
2. RANDOM ROTATION (negative control): the same source vectors, mapped
   through a random semi-orthogonal matrix of the same shape as the real
   rotation instead of the calibrated one - isolates whether the
   *calibrated* correspondence is what matters, versus any
   structurally-similar projection producing equally plausible output.
3. SELF ROUND-TRIP: the target model's OWN native hidden state for the
   same passage (no translation, no source model involved at all),
   captured and re-injected through the identical layer-24 hook
   mechanism used for conditions 1 and 2. Isolates whether the injection
   mechanism itself is lossy, independent of cross-model translation
   quality - if this also degrades, the problem is in the mechanism, not
   in what's being translated.
4. REAL TEXT (upper bound): the actual passage, tokenized and processed
   through the target model's ordinary unmodified forward pass - what a
   good continuation looks like, for the other three to be judged
   against.
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from piper_geometry.extractor import ResidualExtractor
from piper_geometry.congruence_optimizer import CROSS_MODEL_CONFIGS, DEFAULT_MODEL

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONCEPTS_PATH = REPO_ROOT / "data" / "checkpoints" / "concepts_dictionary.json"
EXPERIMENTS_DIR = REPO_ROOT / "obsidian" / "Experiments"

# Reuses the strongest, most thoroughly validated recipe found across
# ALIGNQ/XALIGNQ/XALIGNDS/XALIGNPHI: Phi-4-mini-instruct as target,
# center=False (settled across three independent cross-model pairings),
# calibration_size=280 (the top of the grid - cosine similarity hadn't
# plateaued there yet). Calibration itself stays concept-phrase-based
# (last-token summaries) - a fixed linear map doesn't care what kind of
# vector it's applied to afterward, so fitting W this way and applying it
# to every token of a passage are both valid uses of the same W.
TARGET_PREFIX = "XALIGNPHI"
CALIBRATION_SIZE = 280
RNG_SEED = 1234


def _load_concepts_by_domain() -> dict:
    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _primary_concepts(phrases: list) -> list:
    """Excludes the auto-generated padding entries also excluded elsewhere
    in this package (export_dual_graph.py) - not meant to be individually
    meaningful, just calibration bulk."""
    return [p for p in phrases if not p.startswith(("Advanced corollary", "Empirical boundary condition"))]


def _random_semi_orthogonal_like(W: torch.Tensor) -> torch.Tensor:
    """A random matrix of the same shape as W, with the same orthogonality
    property W actually has, for the negative control condition.

    W comes from an SVD (U @ Vh, U square orthogonal, Vh row-orthonormal),
    so W @ W.T = I on the source-dimension side - its ROWS are
    orthonormal, not its columns. That's the only orthogonality direction
    that's even possible here: source_dim (896) < target_dim (3072), and
    you cannot fit more mutually-orthogonal columns than there are
    dimensions to hold them in (QR on a "wide" matrix caps out at
    min(rows, cols) orthonormal columns - trying to get 3072 orthonormal
    columns out of 896-dimensional rows is a mathematical impossibility,
    not just an unlikely random draw). QR on the transpose (a "tall"
    matrix, cols > rows) gives orthonormal columns there instead;
    transposing back yields the row-orthonormal shape actually needed.
    """
    rows, cols = W.shape
    # QR runs on CPU regardless of W's own device, then the result moves
    # to W's device afterward - not just a style choice. This Jetson's
    # PyTorch build has a broken CUDA cusolver linkage for at least some
    # GPU linalg routines (torch.linalg.qr on CUDA fails here with
    # "undefined symbol: cusolverDnXsyevBatched_bufferSize"). CPU QR is
    # proven to work: build_rotation's SVD already runs entirely on CPU,
    # for the unrelated reason that ResidualExtractor.extract_activations()
    # always returns .cpu() tensors, and never hits this failure.
    Q, _ = torch.linalg.qr(torch.randn(cols, rows, dtype=W.dtype))
    return Q.t().to(device=W.device)


def build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts):
    """Same math as CongruenceOptimizer.run_trial's rotation step, extracted
    standalone since this script needs the rotation matrix itself, not
    just its congruence/accuracy/cosine-sim summary."""
    source_vecs, target_vecs = [], []
    for concept in calibration_concepts:
        src_acts = source_extractor.extract_activations(prompt=concept, target_layers=[source_layer])
        tgt_acts = target_extractor.extract_activations(prompt=concept, target_layers=[receiver_layer])
        source_vecs.append(src_acts[source_layer])
        target_vecs.append(tgt_acts[receiver_layer])

    A = torch.stack(source_vecs).to(torch.float32)
    B = torch.stack(target_vecs).to(torch.float32)
    M = A.t() @ B
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vh
    return W


def extract_full_sequence(model, tokenizer, device, text: str, layer_idx: int) -> torch.Tensor:
    """Every token's hidden state at layer_idx for `text`, not just a
    last-token summary - generation needs the whole sequence to avoid
    stripping away the context a real passage carries."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_idx][0]  # (seq_len, hidden_dim), batch dim dropped


def norm_stats(states: torch.Tensor) -> dict:
    """Per-token L2 norm summary, to check whether translated vectors
    carry a systematically different scale than what the target layer's
    own native activations look like. A Procrustes rotation is provably
    norm-preserving (W @ W.T = I by construction), so rotated/random
    states are mathematically guaranteed to carry Qwen's original
    magnitudes unchanged - if that turns out to differ from Phi-4-mini's
    own native scale at the same layer, translation can't close that gap
    on its own, no matter how good the rotation's direction is."""
    norms = states.norm(dim=-1)
    return {
        "mean": norms.mean().item(),
        "std": norms.std().item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
    }


def _print_norm_stats(label: str, stats: dict) -> None:
    print(f"[ConceptTransferTest]   {label}: mean={stats['mean']:.2f}  "
          f"std={stats['std']:.2f}  min={stats['min']:.2f}  max={stats['max']:.2f}")


class _LayerInjectionHook:
    """Overrides one transformer layer's output on its first invocation -
    the prefill pass over the full placeholder sequence - with externally
    supplied hidden states, then gets out of the way for every subsequent
    single-token decode step, which must run unmodified so newly
    generated tokens reflect the model's own real computation given
    whatever is now sitting in the stream at the injected positions."""

    def __init__(self, injected_states: torch.Tensor):
        self.injected_states = injected_states  # (1, seq_len, hidden_dim)
        self.applied = False

    def __call__(self, module, inputs, output):
        if self.applied:
            return output
        self.applied = True
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if hidden.shape[1] != self.injected_states.shape[1]:
            # Not the prefill pass we expected - leave it alone rather
            # than silently apply an injection with mismatched shape.
            return output
        new_hidden = self.injected_states.to(dtype=hidden.dtype, device=hidden.device)
        return (new_hidden,) + output[1:] if is_tuple else new_hidden


def generate_with_injection(model, tokenizer, device, layer_idx: int, injected_states: torch.Tensor,
                             max_new_tokens: int = 60) -> str:
    """Runs a neutral placeholder sequence through the model with
    layer_idx's output overridden (via _LayerInjectionHook) on the
    prefill pass only, then decodes just the newly generated continuation
    - not the placeholder prefix, which carries no information itself."""
    seq_len = injected_states.shape[0]
    placeholder_id = tokenizer.eos_token_id
    input_ids = torch.full((1, seq_len), placeholder_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    hook = _LayerInjectionHook(injected_states.unsqueeze(0))
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    if not hook.applied:
        print(f"[ConceptTransferTest] WARNING: injection hook never fired for layer {layer_idx}")
    return tokenizer.decode(output_ids[0, seq_len:], skip_special_tokens=True).strip()


def generate_normally(model, tokenizer, device, text: str, max_new_tokens: int = 60) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, seq_len:], skip_special_tokens=True).strip()


def run_test(domain: str = "physics", num_concepts: int = 4, max_new_tokens: int = 60) -> dict:
    config = CROSS_MODEL_CONFIGS[TARGET_PREFIX]
    target_model_name = config["target_model"]
    source_layer, receiver_layer = config["param_grid"]["layer_pair"][0]

    print(f"[ConceptTransferTest] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)

    domains = _load_concepts_by_domain()
    test_concepts = _primary_concepts(domains[domain])[:num_concepts]
    passage = ". ".join(test_concepts) + "."
    print(f"[ConceptTransferTest] Passage ({domain}): {passage}")

    # Calibration pool excludes only the chosen test concepts, not the
    # whole domain - excluding the whole domain only leaves 240 of the
    # dictionary's 300 concepts (60 removed per domain), short of the
    # calibration_size=280 the validated recipe uses.
    test_concepts_set = set(test_concepts)
    other_concepts = [p for phrases in domains.values() for p in phrases if p not in test_concepts_set]
    rng = random.Random(RNG_SEED)
    rng.shuffle(other_concepts)
    calibration_concepts = other_concepts[:CALIBRATION_SIZE]

    print(f"[ConceptTransferTest] Computing rotation from {len(calibration_concepts)} calibration concepts...")
    # build_rotation's inputs come from ResidualExtractor.extract_activations(),
    # which explicitly .cpu()s its output - so W ends up on CPU regardless of
    # which device the models themselves run on. extract_full_sequence below
    # does not move its output off the model's own device (CUDA here), so W
    # needs an explicit move before it's ever multiplied against those
    # tensors - matches source_hidden's device since that's the side W gets
    # matrix-multiplied against.
    W = build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)
    W = W.to(source_extractor.device)
    W_random = _random_semi_orthogonal_like(W)

    print("[ConceptTransferTest] Extracting full-sequence hidden states for the passage...")
    source_hidden = extract_full_sequence(
        source_extractor.model, source_extractor.tokenizer, source_extractor.device, passage, source_layer
    ).to(torch.float32)
    target_hidden_native = extract_full_sequence(
        target_extractor.model, target_extractor.tokenizer, target_extractor.device, passage, receiver_layer
    ).to(torch.float32)

    rotated_states = source_hidden @ W          # (T_source, target_dim)
    random_states = source_hidden @ W_random    # (T_source, target_dim)
    # target_hidden_native is already (T_target, target_dim) - no translation needed

    print("[ConceptTransferTest] Per-token vector norms (checking for a scale mismatch):")
    source_norms = norm_stats(source_hidden)
    rotated_norms = norm_stats(rotated_states)
    random_norms = norm_stats(random_states)
    target_native_norms = norm_stats(target_hidden_native)
    _print_norm_stats("source_hidden (Qwen L%d, native)" % source_layer, source_norms)
    _print_norm_stats("rotated_states (translated via W)", rotated_norms)
    _print_norm_stats("random_states (translated via random W)", random_norms)
    _print_norm_stats("target_hidden_native (Phi-4-mini L%d, native)" % receiver_layer, target_native_norms)

    # A rotation preserves direction, not magnitude - rotated/random_states
    # are mathematically guaranteed to carry Qwen's own scale, which the
    # norm check above confirms is substantially different from Phi-4-mini's
    # native scale at this layer. Rescaling by the ratio of mean norms is
    # the simplest fix a rotation can't provide on its own; not
    # position-aware (both distributions have large outlier positions, e.g.
    # a much larger first-token norm - a known "attention sink" pattern in
    # transformer residual streams), but a reasonable first attempt before
    # anything more elaborate.
    scale_factor = target_native_norms["mean"] / source_norms["mean"]
    print(f"[ConceptTransferTest] Rescaling translated vectors by {scale_factor:.3f}x to match target's native scale...")
    rotated_states = rotated_states * scale_factor
    random_states = random_states * scale_factor

    target_model = target_extractor.model
    target_tokenizer = target_extractor.tokenizer
    device = target_extractor.device

    print("[ConceptTransferTest] Generating: rotated condition (layer injection)...")
    rotated_output = generate_with_injection(target_model, target_tokenizer, device, receiver_layer, rotated_states, max_new_tokens)
    print("[ConceptTransferTest] Generating: random-rotation control (layer injection)...")
    random_output = generate_with_injection(target_model, target_tokenizer, device, receiver_layer, random_states, max_new_tokens)
    print("[ConceptTransferTest] Generating: self round-trip (layer injection, no translation)...")
    self_output = generate_with_injection(target_model, target_tokenizer, device, receiver_layer, target_hidden_native, max_new_tokens)
    print("[ConceptTransferTest] Generating: real text upper bound (unmodified)...")
    real_output = generate_normally(target_model, target_tokenizer, device, passage, max_new_tokens)

    result = {
        "domain": domain,
        "passage": passage,
        "source_model": DEFAULT_MODEL,
        "target_model": target_model_name,
        "source_layer": source_layer,
        "receiver_layer": receiver_layer,
        "calibration_size": len(calibration_concepts),
        "source_token_count": source_hidden.shape[0],
        "target_token_count": target_hidden_native.shape[0],
        "source_norms": source_norms,
        "rotated_norms": rotated_norms,
        "random_norms": random_norms,
        "target_native_norms": target_native_norms,
        "scale_factor": scale_factor,
        "rotated_output": rotated_output,
        "random_rotation_output": random_output,
        "self_roundtrip_output": self_output,
        "real_text_output": real_output,
    }

    print("\n=== 1. ROTATED (translated passage) ===")
    print(rotated_output)
    print("\n=== 2. RANDOM ROTATION (negative control) ===")
    print(random_output)
    print("\n=== 3. SELF ROUND-TRIP (mechanism-only control) ===")
    print(self_output)
    print("\n=== 4. REAL TEXT (upper bound) ===")
    print(real_output)

    return result


def log_markdown(result: dict) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    now_dt = datetime.now()
    exp_id = f"CTRANSFER-{now_dt.strftime('%Y%m%d-%H%M%S')}"
    note_file = EXPERIMENTS_DIR / f"{exp_id}.md"

    content = f"""---
id: {exp_id}
type: concept_transfer_interpretation_test
cycle_prefix: CTRANSFER
date: '{now_dt.isoformat()}'
domain: {result['domain']}
source_model: {result['source_model']}
target_model: {result['target_model']}
source_layer: {result['source_layer']}
receiver_layer: {result['receiver_layer']}
calibration_size: {result['calibration_size']}
source_token_count: {result['source_token_count']}
target_token_count: {result['target_token_count']}
tags:
- concept_transfer
- interpretation_test
- layer_injection
- {TARGET_PREFIX.lower()}
---

# Experiment: {exp_id}

**Passage** ({result['domain']}): {result['passage']}
**Pairing**: {result['source_model']} L{result['source_layer']} ({result['source_token_count']} tokens) -> {result['target_model']} L{result['receiver_layer']} ({result['target_token_count']} tokens)
**Calibration Size**: {result['calibration_size']}

## Per-token vector norms (scale-mismatch check)
A Procrustes rotation is provably norm-preserving, so rotated/random
should exactly match source's scale - any gap to target-native reflects
Qwen's and Phi-4-mini's own differing native scales, not something
translation could have fixed by picking a better rotation.

| | mean | std | min | max |
|---|---|---|---|---|
| source_hidden (Qwen L{result['source_layer']}, native) | {result['source_norms']['mean']:.2f} | {result['source_norms']['std']:.2f} | {result['source_norms']['min']:.2f} | {result['source_norms']['max']:.2f} |
| rotated_states (translated via W) | {result['rotated_norms']['mean']:.2f} | {result['rotated_norms']['std']:.2f} | {result['rotated_norms']['min']:.2f} | {result['rotated_norms']['max']:.2f} |
| random_states (translated via random W) | {result['random_norms']['mean']:.2f} | {result['random_norms']['std']:.2f} | {result['random_norms']['min']:.2f} | {result['random_norms']['max']:.2f} |
| target_hidden_native (Phi-4-mini L{result['receiver_layer']}, native) | {result['target_native_norms']['mean']:.2f} | {result['target_native_norms']['std']:.2f} | {result['target_native_norms']['min']:.2f} | {result['target_native_norms']['max']:.2f} |

**Scale correction applied**: rotated/random states rescaled by {result['scale_factor']:.3f}x (target-native mean norm / source-native mean norm) before injection, since a rotation preserves direction but not magnitude.

## 1. Rotated (translated passage, layer-{result['receiver_layer']} injection)
{result['rotated_output']}

## 2. Random Rotation (negative control, layer-{result['receiver_layer']} injection)
{result['random_rotation_output']}

## 3. Self Round-Trip (mechanism-only control, no translation, layer-{result['receiver_layer']} injection)
{result['self_roundtrip_output']}

## 4. Real Text (upper bound, unmodified generation)
{result['real_text_output']}

## Evaluation
Manual read for now, no automated scoring yet:
- Does condition 1 land closer to condition 4's territory (domain/theme)
  than condition 2 does? That's the actual translation-quality question.
- Does condition 3 come out coherent? If not, the injection mechanism
  itself is lossy independent of any cross-model translation, and that's
  the bottleneck to fix before cross-model quality is worth chasing
  further.
"""
    note_file.write_text(content, encoding="utf-8")
    print(f"\n[ConceptTransferTest] Logged to {note_file}")
    return note_file


if __name__ == "__main__":
    result = run_test(domain="physics", num_concepts=4)
    log_markdown(result)
