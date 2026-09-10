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
from piper_geometry.layer_injection import (
    build_rotation,
    random_semi_orthogonal_like,
    extract_full_sequence,
    norm_stats,
    print_norm_stats,
    generate_with_injection,
    generate_normally,
)
from piper_geometry.evaluate_adapter import load_trained_adapter

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


def run_test(domain: str = "physics", num_concepts: int = 4, max_new_tokens: int = 60,
             adapter_checkpoint: Path = None) -> dict:
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
    W_random = random_semi_orthogonal_like(W)

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
    print_norm_stats("source_hidden (Qwen L%d, native)" % source_layer, source_norms)
    print_norm_stats("rotated_states (translated via W)", rotated_norms)
    print_norm_stats("random_states (translated via random W)", random_norms)
    print_norm_stats("target_hidden_native (Phi-4-mini L%d, native)" % receiver_layer, target_native_norms)

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

    # Optional 5th condition: a trained adapter (train_adapter.py) instead
    # of the fixed rotation. Deliberately no manual rescale here, unlike
    # rotated/random above - the whole point of training past the fixed
    # rotation was for the adapter to learn its own scale from the task
    # loss directly; reapplying scale_factor on top would double-correct
    # a magnitude the adapter already accounts for in its own weights.
    adapter_states = None
    adapter_norms = None
    if adapter_checkpoint is not None:
        print(f"[ConceptTransferTest] Loading trained adapter from {adapter_checkpoint}...")
        trained_adapter, _ = load_trained_adapter(adapter_checkpoint, device=target_extractor.device)
        with torch.no_grad():
            adapter_states = trained_adapter(source_hidden.to(target_extractor.device))
        adapter_norms = norm_stats(adapter_states)
        print_norm_stats("adapter_states (translated via trained adapter, no manual rescale)", adapter_norms)

    target_model = target_extractor.model
    target_tokenizer = target_extractor.tokenizer
    device = target_extractor.device

    header_info = {
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
        "adapter_checkpoint": str(adapter_checkpoint) if adapter_checkpoint is not None else None,
        "adapter_norms": adapter_norms,
    }
    # Written before any generation is attempted, and each condition's
    # section appended immediately after it completes (or fails) - a
    # crash partway through (the random-rotation condition has triggered
    # a CUDA device-side assert on multiple runs now) previously lost
    # every earlier condition's output too, since the old version only
    # wrote the note once at the very end. Each generate_* call is now
    # wrapped so a crash still appends what happened before re-raising -
    # the batch runner still sees a nonzero exit code either way, but the
    # note now records exactly which condition failed and with what
    # error, instead of leaving nothing behind at all.
    note_file = create_note(header_info)

    def _run_condition(number: int, title: str, fn, *fn_args):
        print(f"[ConceptTransferTest] Generating: {title}...")
        try:
            output = fn(*fn_args)
        except Exception as e:
            append_condition(note_file, number, title, f"CRASHED: {type(e).__name__}: {e}")
            raise
        append_condition(note_file, number, title, output)
        print(f"\n=== {number}. {title.upper()} ===")
        print(output)
        return output

    rotated_output = _run_condition(
        1, "Rotated (translated passage, layer injection)",
        generate_with_injection, target_model, target_tokenizer, device, receiver_layer, rotated_states, max_new_tokens,
    )
    random_output = _run_condition(
        2, "Random Rotation (negative control, layer injection)",
        generate_with_injection, target_model, target_tokenizer, device, receiver_layer, random_states, max_new_tokens,
    )
    self_output = _run_condition(
        3, "Self Round-Trip (mechanism-only control, no translation, layer injection)",
        generate_with_injection, target_model, target_tokenizer, device, receiver_layer, target_hidden_native, max_new_tokens,
    )
    real_output = _run_condition(
        4, "Real Text (upper bound, unmodified generation)",
        generate_normally, target_model, target_tokenizer, device, passage, max_new_tokens,
    )
    adapter_output = None
    if adapter_states is not None:
        adapter_output = _run_condition(
            5, "Trained Adapter (learned map, layer injection)",
            generate_with_injection, target_model, target_tokenizer, device, receiver_layer, adapter_states, max_new_tokens,
        )

    append_evaluation_note(note_file, has_adapter_condition=adapter_output is not None)

    result = dict(header_info)
    result.update({
        "rotated_output": rotated_output,
        "random_rotation_output": random_output,
        "adapter_output": adapter_output,
        "self_roundtrip_output": self_output,
        "real_text_output": real_output,
        "note_file": note_file,
    })
    return result


def create_note(info: dict) -> Path:
    """Writes the note's frontmatter, passage, and norm table before any
    generation is attempted - each condition's section is appended to
    this same file afterward via append_condition(), so partial results
    survive a crash partway through."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    now_dt = datetime.now()
    exp_id = f"CTRANSFER-{now_dt.strftime('%Y%m%d-%H%M%S')}"
    note_file = EXPERIMENTS_DIR / f"{exp_id}.md"

    # Optional: only present when run_test() was given an
    # adapter_checkpoint. info.get(...) rather than info[...] so a caller
    # that never added this key (any pre-existing 4-condition-only info
    # dict) still works unchanged.
    adapter_norms = info.get("adapter_norms")
    adapter_row = ""
    if adapter_norms is not None:
        adapter_row = (
            f"| adapter_states (translated via trained adapter, checkpoint: {info.get('adapter_checkpoint')}) "
            f"| {adapter_norms['mean']:.2f} | {adapter_norms['std']:.2f} | {adapter_norms['min']:.2f} | {adapter_norms['max']:.2f} |\n"
        )

    content = f"""---
id: {exp_id}
type: concept_transfer_interpretation_test
cycle_prefix: CTRANSFER
date: '{now_dt.isoformat()}'
domain: {info['domain']}
source_model: {info['source_model']}
target_model: {info['target_model']}
source_layer: {info['source_layer']}
receiver_layer: {info['receiver_layer']}
calibration_size: {info['calibration_size']}
source_token_count: {info['source_token_count']}
target_token_count: {info['target_token_count']}
tags:
- concept_transfer
- interpretation_test
- layer_injection
- {TARGET_PREFIX.lower()}
---

# Experiment: {exp_id}

**Passage** ({info['domain']}): {info['passage']}
**Pairing**: {info['source_model']} L{info['source_layer']} ({info['source_token_count']} tokens) -> {info['target_model']} L{info['receiver_layer']} ({info['target_token_count']} tokens)
**Calibration Size**: {info['calibration_size']}

## Per-token vector norms (scale-mismatch check)
A Procrustes rotation is provably norm-preserving, so rotated/random
should exactly match source's scale - any gap to target-native reflects
Qwen's and Phi-4-mini's own differing native scales, not something
translation could have fixed by picking a better rotation.

| | mean | std | min | max |
|---|---|---|---|---|
| source_hidden (Qwen L{info['source_layer']}, native) | {info['source_norms']['mean']:.2f} | {info['source_norms']['std']:.2f} | {info['source_norms']['min']:.2f} | {info['source_norms']['max']:.2f} |
| rotated_states (translated via W) | {info['rotated_norms']['mean']:.2f} | {info['rotated_norms']['std']:.2f} | {info['rotated_norms']['min']:.2f} | {info['rotated_norms']['max']:.2f} |
| random_states (translated via random W) | {info['random_norms']['mean']:.2f} | {info['random_norms']['std']:.2f} | {info['random_norms']['min']:.2f} | {info['random_norms']['max']:.2f} |
| target_hidden_native (Phi-4-mini L{info['receiver_layer']}, native) | {info['target_native_norms']['mean']:.2f} | {info['target_native_norms']['std']:.2f} | {info['target_native_norms']['min']:.2f} | {info['target_native_norms']['max']:.2f} |
{adapter_row}
**Scale correction applied**: rotated/random states rescaled by {info['scale_factor']:.3f}x (target-native mean norm / source-native mean norm) before injection, since a rotation preserves direction but not magnitude. The trained adapter (if present above) is deliberately NOT rescaled - it learned its own effective scale from the training loss directly.
"""
    note_file.write_text(content, encoding="utf-8")
    return note_file


def append_condition(note_file: Path, number: int, title: str, output: str) -> None:
    with open(note_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {number}. {title}\n{output}\n")


def append_evaluation_note(note_file: Path, has_adapter_condition: bool = False) -> None:
    adapter_bullet = (
        "\n- Does condition 5 (trained adapter) read as more coherent than "
        "condition 1 (fixed rotation)? A classification accuracy number "
        "alone can't distinguish \"learned a real correction\" from "
        "\"memorized the training set\" - this is a human check on what "
        "the difference actually looks like in generated text.\n"
        if has_adapter_condition else ""
    )
    with open(note_file, "a", encoding="utf-8") as f:
        f.write(f"""
## Evaluation
Manual read for now, no automated scoring yet:
- Does condition 1 land closer to condition 4's territory (domain/theme)
  than condition 2 does? That's the actual translation-quality question.
- Does condition 3 come out coherent? If not, the injection mechanism
  itself is lossy independent of any cross-model translation, and that's
  the bottleneck to fix before cross-model quality is worth chasing
  further.{adapter_bullet}""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="physics",
                         choices=["physics", "computer_science", "philosophy", "mathematics", "cognitive_science", "biology"])
    parser.add_argument("--num-concepts", type=int, default=4)
    parser.add_argument("--adapter-checkpoint", type=str, default=None,
                         help="path to a train_adapter.py checkpoint; adds a 5th 'trained adapter' "
                              "generation condition alongside the fixed-rotation comparison")
    args = parser.parse_args()

    result = run_test(
        domain=args.domain, num_concepts=args.num_concepts,
        adapter_checkpoint=Path(args.adapter_checkpoint) if args.adapter_checkpoint else None,
    )
    print(f"\n[ConceptTransferTest] Logged to {result['note_file']}")
