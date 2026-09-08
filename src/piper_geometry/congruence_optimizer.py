"""
Piper Geometry: Alignment Congruence Optimizer (ALIGNQ / XALIGNQ research tracks).

Random-search trial loop: each call to run_trial() samples a (layer pair,
calibration set size, centering choice) combination from a param grid,
computes an orthogonal Procrustes rotation between two residual-stream
layers, and measures two different things that are easy to conflate:

- In-sample congruence: how well an orthogonal rotation fits the
  calibration data it was built from, via the SVD singular values that
  compute_procrustes-style functions elsewhere in this package discard.
  1.0 means a rotation exists that maps the calibration set onto its
  targets essentially exactly; near 0 means no orthogonal map fits well
  at all, regardless of how much calibration data you throw at it.
- Held-out generalization: cosine similarity and top-1 nearest-neighbor
  accuracy on a fixed set of concepts never used for calibration.

Logging both per trial, across many (layer, calibration size, centering)
combinations, is the point - a rotation can fit its calibration set
closely and still fail to generalize, and only comparing the two reveals
that instead of a single cosine-similarity number taken in isolation.

CongruenceOptimizer also supports a target_model_name distinct from
model_name (the XALIGNQ track): source_layer and receiver_layer are then
indices into two different models rather than two layers of one. Same
math throughout - the SVD-based rotation and the congruence formula don't
assume square/equal-dimension inputs, since source and receiver vectors
only ever interact via matrix products (A.t() @ B, then (x - mean) @ W),
which are well-defined between differently-shaped spaces. The only real
change for the cross-model case is _extract_pair needing two separate
forward passes (one per model) instead of one pass capturing both layers
of a single model at once.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from piper_geometry.extractor import ResidualExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONCEPTS_PATH = REPO_ROOT / "data" / "checkpoints" / "concepts_dictionary.json"

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# calibration_size tops out at 280, not DEFAULT_MODEL's 896-dim hidden
# size, because that's the ceiling the concept dictionary's pool actually
# allows (300 total concepts - HELDOUT_SIZE reserved below = 284,
# rounded down for headroom) - the first 27 real trials at 12-48 showed
# congruence pinned near 1.0 regardless of layer pair, exactly the
# structurally-guaranteed near-perfect in-sample fit expected when
# calibration_size is far below the embedding dimension. Pushing size
# toward this pool's actual ceiling won't fully escape that regime, but
# should reveal whether congruence trends down as it's approached -
# itself informative - and remains capped below 896 until the concept
# dictionary grows larger than it is today.
PARAM_GRID = {
    "layer_pair": [(6, 10), (8, 14), (10, 16), (12, 18)],
    "calibration_size": [12, 24, 48, 96, 192, 280],
    "center": [True, False],
}

# Cross-model tracks (XALIGNQ, XALIGNDS, ...): one entry per (source
# model, target model) pairing being tested, keyed by the prefix the goal
# file/trial notes use. Each is a smallest-reasonable-step-up from the
# last: Qwen2.5-0.5B-Instruct -> TinyLlama-1.1B-Chat-v1.0 first (both
# small, confirmed comfortable on the Jetson's memory), then
# DeepSeek-R1-Distill-Qwen-1.5B (built on the Qwen2.5-1.5B architecture -
# 28 layers, ~3GB in fp16) as the next moderate step, before attempting
# anything as large as Phi-4-mini-instruct (~3.8B).
#
# Layer choice is a single fixed pair per config, not a swept list, in
# each case picked at ~75% depth in both models - the region ALIGNQ's
# same-model sweep found carried the most abstract, transferable
# structure - rather than swept, to keep each new pairing's first batch
# limited to calibration_size/center like ALIGNQ's own first batch was.
# ALIGNQ's "shallow source, deep receiver" finding was about one model's
# evolving residual stream, not a reason to expect two separate models to
# want different relative depths from each other, so both sides of every
# pairing here use the same ~75% mark rather than reusing that split.
CROSS_MODEL_CONFIGS = {
    "XALIGNQ": {
        "target_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "param_grid": {
            "layer_pair": [(18, 17)],  # Qwen L18/24 (75%), TinyLlama L17/22 (~77%)
            "calibration_size": [12, 24, 48, 96, 192, 280],
            "center": [True, False],
        },
    },
    "XALIGNDS": {
        "target_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "param_grid": {
            "layer_pair": [(18, 21)],  # Qwen L18/24 (75%), DeepSeek-R1-Distill L21/28 (75%)
            "calibration_size": [12, 24, 48, 96, 192, 280],
            "center": [True, False],
        },
    },
    # ~3.8B params, ~7.6GB in fp16 - a meaningfully bigger step than
    # XALIGNDS's ~1.5B. Jetson memory measured at 8.2GB (TinyLlama) then
    # 9.7GB (DeepSeek) resident alongside Qwen + Ollama's own model;
    # extrapolating the ~1.5GB-per-extra-billion-fp16-params trend puts
    # this around 14GB of the Jetson's 15.3GB - real risk of getting
    # close to the ceiling, unconfirmed until it's actually run.
    "XALIGNPHI": {
        "target_model": "microsoft/Phi-4-mini-instruct",
        "param_grid": {
            "layer_pair": [(18, 24)],  # Qwen L18/24 (75%), Phi-4-mini L24/32 (75%)
            "calibration_size": [12, 24, 48, 96, 192, 280],
            "center": [True, False],
        },
    },
}

# Fixed across every trial regardless of calibration_size, so accuracy and
# cosine-similarity numbers stay comparable trial to trial - only the
# calibration side of the sweep should vary between runs.
HELDOUT_SIZE = 16
HELDOUT_SEED = 1234


def _load_all_concepts() -> List[str]:
    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        domains = json.load(f)
    concepts = []
    for phrases in domains.values():
        concepts.extend(phrases)
    return concepts


class CongruenceOptimizer:
    """Lazily loads one extractor per distinct model name and reuses them
    across trials. When target_model_name equals model_name (the default -
    ALIGNQ's same-model case), source and target share one extractor and
    one forward pass per concept, same as before this class supported a
    second model at all."""

    def __init__(self, model_name: str = DEFAULT_MODEL, target_model_name: str = None):
        self.model_name = model_name
        self.target_model_name = target_model_name or model_name
        self._extractors: Dict[str, ResidualExtractor] = {}

        all_concepts = _load_all_concepts()
        rng = random.Random(HELDOUT_SEED)
        shuffled = all_concepts[:]
        rng.shuffle(shuffled)
        self.heldout_concepts = shuffled[:HELDOUT_SIZE]
        self.calibration_pool = shuffled[HELDOUT_SIZE:]

    def _get_extractor(self, model_name: str) -> ResidualExtractor:
        if model_name not in self._extractors:
            self._extractors[model_name] = ResidualExtractor(model_name_or_path=model_name)
        return self._extractors[model_name]

    def _extract_pair(self, concepts: List[str], source_layer: int, receiver_layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source_extractor = self._get_extractor(self.model_name)
        target_extractor = self._get_extractor(self.target_model_name)
        source_vecs, target_vecs = [], []

        if source_extractor is target_extractor:
            # Same model: one forward pass per concept captures both layers at once.
            for concept in concepts:
                acts = source_extractor.extract_activations(prompt=concept, target_layers=[source_layer, receiver_layer])
                source_vecs.append(acts[source_layer])
                target_vecs.append(acts[receiver_layer])
        else:
            # Different models: each needs its own forward pass, since a
            # single call to extract_activations only runs one model.
            for concept in concepts:
                src_acts = source_extractor.extract_activations(prompt=concept, target_layers=[source_layer])
                tgt_acts = target_extractor.extract_activations(prompt=concept, target_layers=[receiver_layer])
                source_vecs.append(src_acts[source_layer])
                target_vecs.append(tgt_acts[receiver_layer])

        return torch.stack(source_vecs), torch.stack(target_vecs)

    def run_trial(self, params: Dict) -> Dict:
        source_layer, receiver_layer = params["layer_pair"]
        calibration_size = params["calibration_size"]
        center = params["center"]

        calibration_concepts = random.sample(self.calibration_pool, calibration_size)

        source_calib, target_calib = self._extract_pair(calibration_concepts, source_layer, receiver_layer)
        source_heldout, target_heldout = self._extract_pair(self.heldout_concepts, source_layer, receiver_layer)

        # ResidualExtractor loads the model in float16 on CUDA for speed
        # and moves captured vectors to CPU without changing dtype, so
        # they arrive here as CPU tensors still in half precision.
        # torch.linalg.svd has no CPU kernel for Half - only float32+ - so
        # this cast is required, not just precision hygiene (matches the
        # same cast aligner.py's compute_procrustes already does).
        source_calib = source_calib.to(torch.float32)
        target_calib = target_calib.to(torch.float32)
        source_heldout = source_heldout.to(torch.float32)
        target_heldout = target_heldout.to(torch.float32)

        source_mean = source_calib.mean(dim=0, keepdim=True) if center else torch.zeros(1, source_calib.shape[1])
        target_mean = target_calib.mean(dim=0, keepdim=True) if center else torch.zeros(1, target_calib.shape[1])

        A = source_calib - source_mean
        B = target_calib - target_mean

        M = A.t() @ B
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        W = U @ Vh

        # Congruence coefficient: sum of singular values of the
        # cross-covariance, normalized by the calibration matrices'
        # Frobenius norms. 1.0 = a rotation exists that fits A onto B
        # exactly; near 0 = no orthogonal map fits well, independent of
        # whether it generalizes to anything held out.
        congruence = (S.sum() / (A.norm() * B.norm())).item()

        # Apply the calibration-fitted rotation (and its centering offset)
        # to held-out source vectors, landing them in the same uncentered
        # target space the calibration/held-out target vectors live in.
        rotated_heldout = (source_heldout - source_mean) @ W + target_mean

        # Candidate pool for top-1 classification: every concept this
        # trial has a target-space vector for. This legitimately gets
        # easier as calibration_size grows, which is the real question the
        # sweep is testing - not just "does congruence go up," but "does
        # more calibration data actually help pick the right concept."
        all_target_vectors = torch.cat([target_calib, target_heldout], dim=0)
        all_concepts = calibration_concepts + self.heldout_concepts

        correct = 0
        cos_sims = []
        for i, concept in enumerate(self.heldout_concepts):
            rvec = rotated_heldout[i]
            tvec = target_heldout[i]
            cos_sims.append(F.cosine_similarity(rvec.unsqueeze(0), tvec.unsqueeze(0)).item())

            scores = F.cosine_similarity(rvec.unsqueeze(0), all_target_vectors)
            pred_idx = torch.argmax(scores).item()
            if all_concepts[pred_idx] == concept:
                correct += 1

        accuracy = correct / len(self.heldout_concepts)
        avg_cos_sim = sum(cos_sims) / len(cos_sims)

        return {
            "source_layer": source_layer,
            "receiver_layer": receiver_layer,
            "calibration_size": calibration_size,
            "center": center,
            "congruence": congruence,
            "accuracy": accuracy,
            "cosine_sim": avg_cos_sim,
            "candidate_pool_size": len(all_concepts),
        }


if __name__ == "__main__":
    optimizer = CongruenceOptimizer()
    for _ in range(3):
        trial_params = {k: random.choice(v) for k, v in PARAM_GRID.items()}
        print(f"[CongruenceOptimizer] Running trial with {trial_params}...")
        print(optimizer.run_trial(trial_params))
