"""
Piper Geometry: Concept Transfer Interpretation Test.

congruence_optimizer.py answers "does a translated vector land near the
right geometric target." This script answers a different question:
does the receiving model actually *behave* as if it understood a
translated concept, given nothing but that vector to work from? Those
are not the same thing - recognizing that a signal is meaningful and
correctly interpreting it are different capabilities (the same way
hearing an unfamiliar human language, you know it's language before you
know what it means, if ever).

Three conditions, same concept sequence, side by side:

1. ROTATED: concepts extracted from the source model, translated via the
   validated cross-model rotation (recomputed fresh from a calibration
   set, same math as CongruenceOptimizer.run_trial), fed to the target
   model as a sequence of soft-prompt embeddings - one embedding per
   concept, not one concept's vector repeated to fill several slots
   (the old soft_prompt_injection_dialogue.py did the latter, which
   also never actually crossed models - same model both ends).
2. RANDOM ROTATION (negative control): the same source vectors, mapped
   through a random semi-orthogonal matrix of the same shape as the real
   one instead of the calibrated W. A literally *unrotated* vector can't
   be fed to the target model at all - source and target hidden
   dimensions usually differ (896 vs 3072 here), so "no rotation" isn't
   representable as an input. A random rotation is the honest control:
   it isolates whether the *calibrated* correspondence is what matters,
   versus any structurally-similar projection producing equally
   plausible-sounding output from the target model.
3. REAL TEXT (upper bound): the actual concept words, tokenized and
   embedded through the target model's own ordinary path - what a good
   continuation looks like when the model receives the real semantic
   content normally, for the rotated/random conditions to be judged
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
# plateaued there yet).
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
    Q, _ = torch.linalg.qr(torch.randn(cols, rows))
    return Q.t()


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


def generate_from_embeds(model, tokenizer, device, embeds: torch.Tensor, max_new_tokens: int = 60) -> str:
    embeds = embeds.to(device=device, dtype=model.dtype)
    attention_mask = torch.ones(embeds.shape[:2], device=device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def run_test(domain: str = "physics", num_concepts: int = 4, max_new_tokens: int = 60) -> dict:
    config = CROSS_MODEL_CONFIGS[TARGET_PREFIX]
    target_model_name = config["target_model"]
    source_layer, receiver_layer = config["param_grid"]["layer_pair"][0]

    print(f"[ConceptTransferTest] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)

    domains = _load_concepts_by_domain()
    test_concepts = _primary_concepts(domains[domain])[:num_concepts]
    print(f"[ConceptTransferTest] Test sequence ({domain}): {test_concepts}")

    # Calibration pool excludes only the chosen test concepts, not the
    # whole domain - excluding the whole domain only leaves 240 of the
    # dictionary's 300 concepts (60 removed per domain), short of the
    # calibration_size=280 the validated recipe uses. Other same-domain
    # concepts (physics concepts not chosen as the test sequence) staying
    # in the calibration pool is standard practice - it mirrors how
    # HELDOUT_SIZE in congruence_optimizer.py excludes specific concepts,
    # not entire domains.
    test_concepts_set = set(test_concepts)
    other_concepts = [p for phrases in domains.values() for p in phrases if p not in test_concepts_set]
    rng = random.Random(RNG_SEED)
    rng.shuffle(other_concepts)
    calibration_concepts = other_concepts[:CALIBRATION_SIZE]

    print(f"[ConceptTransferTest] Computing rotation from {len(calibration_concepts)} calibration concepts...")
    W = build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)
    W_random = _random_semi_orthogonal_like(W)

    # Extract the test sequence's source-model vectors once, reuse for both rotated and random-rotation conditions.
    source_vecs = []
    for concept in test_concepts:
        acts = source_extractor.extract_activations(prompt=concept, target_layers=[source_layer])
        source_vecs.append(acts[source_layer])
    source_matrix = torch.stack(source_vecs).to(torch.float32)  # (num_concepts, source_dim)

    rotated = (source_matrix @ W).unsqueeze(0)          # (1, num_concepts, target_dim)
    random_rotated = (source_matrix @ W_random).unsqueeze(0)

    target_model = target_extractor.model
    target_tokenizer = target_extractor.tokenizer
    device = target_extractor.device

    real_text = ". ".join(test_concepts) + "."
    real_input_ids = target_tokenizer(real_text, return_tensors="pt").input_ids.to(device)
    real_embeds = target_model.get_input_embeddings()(real_input_ids)

    print("[ConceptTransferTest] Generating: rotated condition...")
    rotated_output = generate_from_embeds(target_model, target_tokenizer, device, rotated, max_new_tokens)
    print("[ConceptTransferTest] Generating: random-rotation control...")
    random_output = generate_from_embeds(target_model, target_tokenizer, device, random_rotated, max_new_tokens)
    print("[ConceptTransferTest] Generating: real-text upper bound...")
    real_output = generate_from_embeds(target_model, target_tokenizer, device, real_embeds, max_new_tokens)

    result = {
        "domain": domain,
        "test_concepts": test_concepts,
        "source_model": DEFAULT_MODEL,
        "target_model": target_model_name,
        "source_layer": source_layer,
        "receiver_layer": receiver_layer,
        "calibration_size": len(calibration_concepts),
        "rotated_output": rotated_output,
        "random_rotation_output": random_output,
        "real_text_output": real_output,
    }

    print("\n=== ROTATED (translated concepts) ===")
    print(rotated_output)
    print("\n=== RANDOM ROTATION (negative control) ===")
    print(random_output)
    print("\n=== REAL TEXT (upper bound) ===")
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
tags:
- concept_transfer
- interpretation_test
- {TARGET_PREFIX.lower()}
---

# Experiment: {exp_id}

**Test Concepts** ({result['domain']}): {result['test_concepts']}
**Pairing**: {result['source_model']} L{result['source_layer']} -> {result['target_model']} L{result['receiver_layer']}
**Calibration Size**: {result['calibration_size']}

## 1. Rotated (translated concepts)
{result['rotated_output']}

## 2. Random Rotation (negative control)
{result['random_rotation_output']}

## 3. Real Text (upper bound)
{result['real_text_output']}

## Evaluation
Manual read: does condition 1 land closer to condition 3's territory
(domain/theme) than condition 2 does? Qualitative for now - no automated
scoring yet.
"""
    note_file.write_text(content, encoding="utf-8")
    print(f"\n[ConceptTransferTest] Logged to {note_file}")
    return note_file


if __name__ == "__main__":
    result = run_test(domain="physics", num_concepts=4)
    log_markdown(result)
