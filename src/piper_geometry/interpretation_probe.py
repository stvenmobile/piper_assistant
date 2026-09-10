"""
Piper Geometry: Free-Interpretation Probe.

concept_transfer_test.py's conditions ask the receiving model to freely
continue from an injection (no guidance at all) or, for a trained
adapter, to hit one exact classification token or reconstruct exact
source tokens. This probe asks a third, softer question instead: given a
real natural-language framing prompt inviting interpretation ("try to
make sense of this and describe what it reminds you of"), what does the
receiving model actually say? Not exact-match, not classification -
whatever free association falls out.

This is explicitly NOT meant to be judged on a single output. The
discipline that makes it meaningful rather than cold-reading: always
generate from the REAL translated injection *and* a random-rotation
negative control, side by side, on the same passage - only trust a
difference that shows up as more genuinely related in the real condition
than the random one.

Four conditions, same passage:
1. REAL TRANSLATION: the calibrated rotation (or a trained adapter
   checkpoint, if --adapter-checkpoint is given), interpreted freely.
2. RANDOM ROTATION (negative control): same mechanism, meaningless input.
3. SELF ROUND-TRIP: the target model's own native hidden state,
   re-injected - isolates whether the injection mechanism itself
   supports free interpretation at all, independent of translation.
4. REAL TEXT (reference): the framing prompt applied to the actual
   passage text, no injection at all - what a good interpretation looks
   like when the model has the real content to work with.
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
    build_rotation, random_semi_orthogonal_like, extract_full_sequence,
    norm_stats, print_norm_stats, generate_interpretation, generate_normally,
)
from piper_geometry.train_adapter import TARGET_PREFIX, CALIBRATION_SIZE, RNG_SEED, CONCEPTS_PATH, _primary_concepts
from piper_geometry.evaluate_adapter import load_trained_adapter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "obsidian" / "Experiments"

# A first version of this prompt used "geometrical, spatial relationships"
# - an instruction-tuned model reads that literally as "solve a geometry
# problem," not as an invitation to freely associate. Confirmed by the
# REAL TEXT (no-injection) condition producing the same geometry-textbook
# pattern as the injected conditions - proof the wording, not the
# injection mechanism, was driving the output. Avoids any technical
# trigger word (geometry, spatial, dimensional, vector) in favor of
# experiential language, closer to asking a person "what does this
# feeling remind you of" than "describe this shape."
DEFAULT_FRAMING_PROMPT = " What does this remind you of? Describe your impression, in your own words:"


def run_probe(domain: str = "physics", num_concepts: int = 4, max_new_tokens: int = 60,
              framing_prompt: str = DEFAULT_FRAMING_PROMPT, adapter_checkpoint: Path = None) -> dict:
    config = CROSS_MODEL_CONFIGS[TARGET_PREFIX]
    target_model_name = config["target_model"]
    source_layer, receiver_layer = config["param_grid"]["layer_pair"][0]

    print(f"[InterpretationProbe] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)

    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        domains = json.load(f)
    test_concepts = _primary_concepts(domains[domain])[:num_concepts]
    passage = ". ".join(test_concepts) + "."
    print(f"[InterpretationProbe] Passage ({domain}): {passage}")
    print(f"[InterpretationProbe] Framing prompt:{framing_prompt}")

    test_concepts_set = set(test_concepts)
    other_concepts = [p for phrases in domains.values() for p in phrases if p not in test_concepts_set]
    rng = random.Random(RNG_SEED)
    rng.shuffle(other_concepts)
    calibration_concepts = other_concepts[:CALIBRATION_SIZE]

    print(f"[InterpretationProbe] Computing rotation from {len(calibration_concepts)} calibration concepts...")
    W = build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)
    W = W.to(source_extractor.device)
    W_random = random_semi_orthogonal_like(W)

    source_hidden = extract_full_sequence(
        source_extractor.model, source_extractor.tokenizer, source_extractor.device, passage, source_layer
    ).to(torch.float32)
    target_hidden_native = extract_full_sequence(
        target_extractor.model, target_extractor.tokenizer, target_extractor.device, passage, receiver_layer
    ).to(torch.float32)

    random_states = source_hidden @ W_random
    source_norms = norm_stats(source_hidden)
    target_native_norms = norm_stats(target_hidden_native)
    scale_factor = target_native_norms["mean"] / source_norms["mean"]
    random_states = random_states * scale_factor

    if adapter_checkpoint is not None:
        print(f"[InterpretationProbe] Loading trained adapter from {adapter_checkpoint}...")
        adapter, _ = load_trained_adapter(adapter_checkpoint, device=target_extractor.device)
        with torch.no_grad():
            real_translation_states = adapter(source_hidden.to(target_extractor.device))
        translation_source_desc = str(adapter_checkpoint)
    else:
        print("[InterpretationProbe] No adapter checkpoint given - using the fixed calibrated rotation.")
        real_translation_states = (source_hidden @ W) * scale_factor
        translation_source_desc = "fixed calibrated rotation (no trained adapter)"

    print_norm_stats("source_hidden (native)", source_norms)
    print_norm_stats("real_translation_states", norm_stats(real_translation_states))
    print_norm_stats("random_states", norm_stats(random_states))
    print_norm_stats("target_hidden_native", target_native_norms)

    target_model = target_extractor.model
    target_tokenizer = target_extractor.tokenizer
    device = target_extractor.device

    print("[InterpretationProbe] Generating: 1. Real Translation...")
    real_output = generate_interpretation(
        target_model, target_tokenizer, device, receiver_layer, real_translation_states, framing_prompt, max_new_tokens,
    )
    print("[InterpretationProbe] Generating: 2. Random Rotation (negative control)...")
    random_output = generate_interpretation(
        target_model, target_tokenizer, device, receiver_layer, random_states, framing_prompt, max_new_tokens,
    )
    print("[InterpretationProbe] Generating: 3. Self Round-Trip (mechanism-only control)...")
    self_output = generate_interpretation(
        target_model, target_tokenizer, device, receiver_layer, target_hidden_native, framing_prompt, max_new_tokens,
    )
    print("[InterpretationProbe] Generating: 4. Real Text (reference, no injection)...")
    reference_output = generate_normally(
        target_model, target_tokenizer, device, passage + framing_prompt, max_new_tokens,
    )

    for number, title, output in [
        (1, "Real Translation", real_output),
        (2, "Random Rotation (negative control)", random_output),
        (3, "Self Round-Trip (mechanism-only control)", self_output),
        (4, "Real Text (reference, no injection)", reference_output),
    ]:
        print(f"\n=== {number}. {title.upper()} ===")
        print(output)

    result = {
        "domain": domain, "passage": passage, "framing_prompt": framing_prompt,
        "translation_source": translation_source_desc,
        "real_output": real_output, "random_output": random_output,
        "self_output": self_output, "reference_output": reference_output,
        "source_layer": source_layer, "receiver_layer": receiver_layer,
        "target_model": target_model_name,
    }
    note_file = _log_note(result)
    result["note_file"] = note_file
    print(f"\n[InterpretationProbe] Logged to {note_file}")
    return result


def _log_note(info: dict) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = f"INTERP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    note_file = EXPERIMENTS_DIR / f"{exp_id}.md"
    content = f"""---
id: {exp_id}
type: free_interpretation_probe
date: '{datetime.now().isoformat()}'
domain: {info['domain']}
target_model: {info['target_model']}
source_layer: {info['source_layer']}
receiver_layer: {info['receiver_layer']}
translation_source: {info['translation_source']}
tags:
- interpretation_probe
- layer_injection
---

# Experiment: {exp_id}

**Passage** ({info['domain']}): {info['passage']}
**Framing prompt**:{info['framing_prompt']}
**Translation source**: {info['translation_source']}

Not meant to be judged from condition 1 alone - only trust a difference
that shows up as more genuinely related to the passage in condition 1
than in condition 2 (the negative control), across more than one run.

## 1. Real Translation
{info['real_output']}

## 2. Random Rotation (negative control)
{info['random_output']}

## 3. Self Round-Trip (mechanism-only control)
{info['self_output']}

## 4. Real Text (reference, no injection)
{info['reference_output']}
"""
    note_file.write_text(content, encoding="utf-8")
    return note_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="physics",
                         choices=["physics", "computer_science", "philosophy", "mathematics", "cognitive_science", "biology"])
    parser.add_argument("--num-concepts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--framing-prompt", type=str, default=DEFAULT_FRAMING_PROMPT)
    parser.add_argument("--adapter-checkpoint", type=str, default=None,
                         help="path to a train_adapter.py/train_reconstruction_adapter.py checkpoint; "
                              "omit to use the fixed calibrated rotation instead")
    args = parser.parse_args()

    run_probe(
        domain=args.domain, num_concepts=args.num_concepts, max_new_tokens=args.max_new_tokens,
        framing_prompt=args.framing_prompt,
        adapter_checkpoint=Path(args.adapter_checkpoint) if args.adapter_checkpoint else None,
    )
