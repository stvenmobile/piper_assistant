"""
Piper Geometry: Relational Congruence Analysis.

Every congruence number measured so far (ALIGNQ/XALIGNQ/XALIGNDS/XALIGNPHI,
~0.946 cosine similarity for XALIGNPHI) answers one question: does the SAME
concept, fed to both models, land in a geometrically similar place after
rotation? That's individual-POINT alignment. It says nothing about whether
the RELATIONSHIP between two different concepts survives the same rotation
- two points can each independently land close to their targets while the
vector connecting them (the thing that would carry an analogy like
"king - man + woman = queen") ends up scrambled.

This module measures that directly, with no generation, no injection into
a frozen model's residual stream, and none of the downstream confounds
that made train_adapter.py's results hard to interpret cleanly - just
extraction (already validated methodology, reusing build_rotation's own
ResidualExtractor.extract_activations) and vector arithmetic.

For every pair of test concepts (A, B):
  1. d_source = source_vec(A) - source_vec(B)      [Qwen's own relationship]
  2. d_target = target_vec(A) - target_vec(B)       [Phi-4-mini's own relationship]
  3. d_rotated = d_source @ W                       [Qwen's relationship, translated]
  4. score = cosine_similarity(d_rotated, d_target) [does it match?]

Compared against a shuffled-pairing baseline (the same computation with
target concepts randomly relabeled) - the chance floor this same math
would report if there were no real correspondence between the two models'
concepts at all.
"""

import sys
import json
import random
import itertools
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F

from piper_geometry.extractor import ResidualExtractor
from piper_geometry.congruence_optimizer import CROSS_MODEL_CONFIGS, DEFAULT_MODEL
from piper_geometry.layer_injection import build_rotation
from piper_geometry.train_adapter import CONCEPTS_PATH, TARGET_PREFIX, CALIBRATION_SIZE, RNG_SEED, _primary_concepts


def extract_concept_vectors(source_extractor, target_extractor, source_layer, receiver_layer,
                             concepts: list) -> tuple:
    """Same extraction method build_rotation itself uses
    (ResidualExtractor.extract_activations, last-token, L2-normalized) -
    deliberately kept consistent with how the ~0.946 point-congruence
    number was originally measured, so the point-congruence figure this
    module recomputes is a fair, apples-to-apples baseline for the new
    relational figure sitting next to it."""
    source_vectors, target_vectors = {}, {}
    for concept in concepts:
        src_acts = source_extractor.extract_activations(prompt=concept, target_layers=[source_layer])
        tgt_acts = target_extractor.extract_activations(prompt=concept, target_layers=[receiver_layer])
        source_vectors[concept] = src_acts[source_layer].to(torch.float32)
        target_vectors[concept] = tgt_acts[receiver_layer].to(torch.float32)
    return source_vectors, target_vectors


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_relational_congruence(source_vectors: dict, target_vectors: dict, W: torch.Tensor,
                                   num_shuffles: int = 10, seed: int = RNG_SEED) -> dict:
    """Pure vector arithmetic - no model calls - so this is where the real
    correctness risk in this module lives, and where the test suite
    concentrates. concepts is whatever keys source_vectors/target_vectors
    share; every unordered pair gets a relational-congruence score."""
    concepts = list(source_vectors.keys())
    n = len(concepts)
    if n < 2:
        raise ValueError("need at least 2 concepts to measure relational congruence")

    # Point congruence: the already-established metric (does the SAME
    # concept land near its target after rotation), recomputed on this
    # specific concept set for a fair side-by-side comparison against the
    # new relational figure below.
    point_cosines = [_cosine(source_vectors[c] @ W, target_vectors[c]) for c in concepts]

    pairs = list(itertools.combinations(concepts, 2))
    relational_cosines = []
    for a, b in pairs:
        d_source = source_vectors[a] - source_vectors[b]
        d_target = target_vectors[a] - target_vectors[b]
        d_rotated = d_source @ W
        relational_cosines.append(_cosine(d_rotated, d_target))

    # Chance baseline: the identical relational-congruence computation,
    # but with target concepts randomly relabeled first via a permutation
    # - what this same math would report if there were no real
    # correspondence between the two models' concepts at all. Averaged
    # over several independent shuffles for a stable estimate.
    rng = random.Random(seed)
    shuffled_means = []
    for _ in range(num_shuffles):
        shuffled_concepts = concepts[:]
        rng.shuffle(shuffled_concepts)
        relabel = dict(zip(concepts, shuffled_concepts))
        shuffled_relational_cosines = []
        for a, b in pairs:
            d_source = source_vectors[a] - source_vectors[b]
            d_target_shuffled = target_vectors[relabel[a]] - target_vectors[relabel[b]]
            d_rotated = d_source @ W
            shuffled_relational_cosines.append(_cosine(d_rotated, d_target_shuffled))
        shuffled_means.append(sum(shuffled_relational_cosines) / len(shuffled_relational_cosines))

    def _mean(xs):
        return sum(xs) / len(xs)

    def _std(xs):
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    return {
        "num_concepts": n,
        "num_pairs": len(pairs),
        "point_congruence_mean": _mean(point_cosines),
        "relational_congruence_mean": _mean(relational_cosines),
        "relational_congruence_std": _std(relational_cosines),
        "shuffled_baseline_mean": _mean(shuffled_means),
        "shuffled_baseline_std": _std(shuffled_means),
        "relational_cosines": relational_cosines,
    }


def run_relational_analysis(domain: str = None, max_concepts: int = 20,
                             calibration_size: int = CALIBRATION_SIZE, seed: int = RNG_SEED) -> dict:
    """Real-world entry point - needs the actual Qwen/Phi-4-mini models,
    so (like train_adapter.train()) not directly unit-testable;
    compute_relational_congruence above carries the tested logic.
    domain=None spreads the test set across every domain instead of just
    one, so cross-domain relationships get measured too, not only
    within-domain ones."""
    config = CROSS_MODEL_CONFIGS[TARGET_PREFIX]
    target_model_name = config["target_model"]
    source_layer, receiver_layer = config["param_grid"]["layer_pair"][0]

    print(f"[RelationalCongruence] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)

    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        by_domain = json.load(f)

    if domain is not None:
        test_concepts = _primary_concepts(by_domain[domain])[:max_concepts]
    else:
        test_concepts = []
        per_domain = max_concepts // len(by_domain) + 1
        for phrases in by_domain.values():
            test_concepts.extend(_primary_concepts(phrases)[:per_domain])
        test_concepts = test_concepts[:max_concepts]

    print(f"[RelationalCongruence] Testing {len(test_concepts)} concepts "
          f"({'domain=' + domain if domain else 'spread across all domains'})...")

    test_concepts_set = set(test_concepts)
    calibration_pool = [p for phrases in by_domain.values() for p in phrases if p not in test_concepts_set]
    rng = random.Random(seed)
    rng.shuffle(calibration_pool)
    calibration_concepts = calibration_pool[:calibration_size]

    print(f"[RelationalCongruence] Fitting rotation from {len(calibration_concepts)} calibration concepts...")
    W = build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)

    print("[RelationalCongruence] Extracting test-concept vectors from both models...")
    source_vectors, target_vectors = extract_concept_vectors(
        source_extractor, target_extractor, source_layer, receiver_layer, test_concepts,
    )

    print("[RelationalCongruence] Computing point and relational congruence...")
    result = compute_relational_congruence(source_vectors, target_vectors, W, seed=seed)

    print(f"[RelationalCongruence] Point congruence (same concept, rotated):  {result['point_congruence_mean']:.4f}")
    print(f"[RelationalCongruence] Relational congruence (pair differences):  "
          f"{result['relational_congruence_mean']:.4f}  (std {result['relational_congruence_std']:.4f}, "
          f"{result['num_pairs']} pairs)")
    print(f"[RelationalCongruence] Shuffled-pairing baseline (chance floor):  "
          f"{result['shuffled_baseline_mean']:.4f}  (std {result['shuffled_baseline_std']:.4f})")
    print(f"[RelationalCongruence] Signal above chance: "
          f"{result['relational_congruence_mean'] - result['shuffled_baseline_mean']:+.4f}")

    return result


if __name__ == "__main__":
    import argparse

    domains = ["physics", "computer_science", "philosophy", "mathematics", "cognitive_science", "biology"]
    parser = argparse.ArgumentParser(
        description="Measure whether relationships BETWEEN concepts survive the cross-model rotation, "
                    "not just individual points."
    )
    parser.add_argument("--domain", default="all", choices=["all"] + domains)
    parser.add_argument("--max-concepts", type=int, default=20)
    args = parser.parse_args()

    run_relational_analysis(domain=None if args.domain == "all" else args.domain, max_concepts=args.max_concepts)
