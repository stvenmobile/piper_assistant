"""
Piper Geometry: Blind Latent Transfer Validation Protocol.
Executes end-to-end concept transfer experiments and logs findings to Obsidian.
"""

import sys
from pathlib import Path

# Add src/ directory to sys.path for direct execution
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from typing import List, Dict, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F

from piper_geometry.extractor import ResidualExtractor
from piper_geometry.aligner import LatentAligner
from piper_tools.vault_compiler import VaultCompiler

CALIBRATION_CONCEPTS = [
    "Thermodynamic Entropy",
    "Shannon Information Density",
    "Riemannian Manifold Curvature",
    "Topological Knot Invariant",
    "Recursive Function Call",
    "Phase Transition in Condensed Matter",
    "Dynamic Equilibrium",
    "Symplectic Geometry",
    "Markov Decision Process",
    "Fourier Harmonic Transform",
    "Cellular Automata Dynamics",
    "Homology and Betti Numbers"
]

HELDOUT_TEST_CONCEPTS = [
    "Geodesic Flow on Riemannian Surface",
    "Attractor Basin in Dynamical Systems",
    "Superposition in Quantum States",
    "Isomorphism in Abstract Algebra"
]


class ConceptTransferExperiment:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.extractor = ResidualExtractor(model_name_or_path=model_name)
        self.compiler = VaultCompiler()

    def run_cross_layer_transfer(
        self,
        source_layer: int = 8,
        receiver_layer: int = 14
    ) -> Dict:
        """
        Tests whether an abstract concept extracted at an early semantic layer (e.g. L8)
        can be rotated via Procrustes to match its representation at a deeper reasoning layer (e.g. L14).
        """
        print(f"\n--- Running Cross-Layer Transfer Experiment: L{source_layer} -> L{receiver_layer} ---")

        # 1. Calibration Phase: Extract vectors for calibration concepts
        print(f"[Calibration] Extracting {len(CALIBRATION_CONCEPTS)} anchor concepts...")
        source_calib_list = []
        target_calib_list = []

        for concept in CALIBRATION_CONCEPTS:
            acts = self.extractor.extract_activations(prompt=concept, target_layers=[source_layer, receiver_layer])
            source_calib_list.append(acts[source_layer])
            target_calib_list.append(acts[receiver_layer])

        source_matrix = torch.stack(source_calib_list)
        target_matrix = torch.stack(target_calib_list)

        # 2. Compute Orthogonal Procrustes Rotation W
        print("[Alignment] Computing Procrustes transformation matrix...")
        W = LatentAligner.compute_procrustes(source_matrix, target_matrix)

        # 3. Blind Evaluation on Held-Out Concepts
        print(f"[Testing] Evaluating {len(HELDOUT_TEST_CONCEPTS)} held-out concepts (zero-shot transfer)...")
        
        test_source_list = []
        test_target_list = []
        for concept in HELDOUT_TEST_CONCEPTS:
            acts = self.extractor.extract_activations(prompt=concept, target_layers=[source_layer, receiver_layer])
            test_source_list.append(acts[source_layer])
            test_target_list.append(acts[receiver_layer])

        all_concepts = CALIBRATION_CONCEPTS + HELDOUT_TEST_CONCEPTS
        all_target_vectors = torch.cat([target_matrix, torch.stack(test_target_list)], dim=0)

        correct_top1 = 0
        similarities = []

        for idx, concept in enumerate(HELDOUT_TEST_CONCEPTS):
            source_vec = test_source_list[idx]
            true_target_vec = test_target_list[idx]

            rotated_vec, sim = LatentAligner.align_and_measure(source_vec, true_target_vec, W)
            similarities.append(sim)

            scores = F.cosine_similarity(rotated_vec.unsqueeze(0), all_target_vectors)
            pred_idx = torch.argmax(scores).item()
            predicted_concept = all_concepts[pred_idx]

            is_match = (predicted_concept == concept)
            if is_match:
                correct_top1 += 1

            print(f"  Concept: '{concept}' | CosSim: {sim:.4f} | Match: {is_match} (Decoded: '{predicted_concept}')")

        accuracy = correct_top1 / len(HELDOUT_TEST_CONCEPTS)
        avg_sim = sum(similarities) / len(similarities)
        print(f"\n[Results] Top-1 Accuracy: {accuracy * 100:.1f}% | Mean Cosine Sim: {avg_sim:.4f}")

        # 4. Log to Obsidian
        exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        summary = (
            f"Evaluated Procrustes orthogonal rotation across residual stream layers (L{source_layer} -> L{receiver_layer}). "
            f"Zero-shot blind identification achieved {accuracy * 100:.1f}% accuracy across {len(HELDOUT_TEST_CONCEPTS)} held-out domains "
            f"with an average alignment similarity of {avg_sim:.4f}."
        )

        metrics = {
            "cosine_sim": avg_sim,
            "success": accuracy >= 0.75,
            "accuracy": accuracy
        }

        note_path = self.compiler.record_experiment(
            experiment_id=exp_id,
            target_concept="Cross-Layer Semantic Invariance",
            source_layer=source_layer,
            receiver_layer=receiver_layer,
            metrics=metrics,
            findings_summary=summary
        )
        print(f"[Obsidian] Experiment recorded at: {note_path}")

        return {
            "experiment_id": exp_id,
            "accuracy": accuracy,
            "avg_cosine_sim": avg_sim,
            "note_path": str(note_path)
        }


if __name__ == "__main__":
    experiment = ConceptTransferExperiment()
    experiment.run_cross_layer_transfer(source_layer=8, receiver_layer=14)