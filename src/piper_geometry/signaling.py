"""
Piper Geometry: Continuous Soft-Prompt Signaling Game with 50+ Concept Bank.
Autonomous non-verbal referential game with persistent state and generalization testing.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from piper_tools.vault_compiler import VaultCompiler

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "src" / "piper_audio" / "models" / "comm_adapter_latest.pt"

# 40 Diverse Training Anchors across Physics, Topology, Math, and Information Theory
TRAIN_VOCABULARY = [
    # Physics & Thermodynamics
    "Thermodynamic Entropy", "Dynamic Equilibrium", "Dissipative System",
    "Lagrangian Mechanics", "Hamiltonian Phase Space", "Statistical Ensembles",
    "Spontaneous Symmetry Breaking", "Adiabatic Invariant", "Fluctuation-Dissipation Theorem",
    "Relativistic Spacetime Curvature",
    # Mathematics & Topology
    "Riemannian Curvature", "Topological Invariant", "Symplectic Geometry",
    "Homological Algebra", "Differential Form", "Lie Group Manifold",
    "Geodesic Flow", "Morse Theory Critical Point", "Poincare Conjecture Metric",
    "Chern Class Invariant", "Euler Characteristic",
    # Information & Nonlinear Dynamics
    "Information Bottleneck", "Shannon Information Density", "Kolmogorov Complexity",
    "Markov Decision Process", "Fourier Harmonic Transform", "Cellular Automata Dynamics",
    "Recursive Bifurcation", "Attractor Basin", "Lyapunov Exponent Chaos",
    "Percolation Threshold", "Self-Organized Criticality", "Spectral Graph Laplacian",
    # Computation & Philosophy of Mind
    "Turing Incompleteness", "Church-Turing Thesis", "Category Theory Functor",
    "Lambda Calculus Reduction", "Epistemic Closure Principle",
    "Supervenience in Physicalism", "Emergent Functionalism"
]

# 12 Held-Out Test Concepts (Never seen during adapter optimization)
HELDOUT_TEST_VOCABULARY = [
    "Quantum Superposition", "Phase Transition in Matter",
    "Holographic Principle", "Conformal Field Invariance",
    "Betti Numbers in Topology", "Symplectic Integrator Drift",
    "Renormalization Group Flow", "Wavelet Transform Multiresolution",
    "Strange Attractor Fractal Dimension", "Bayesian Surprise Metric",
    "Godelian Undecidability", "Integrated Information Theory"
]


class SoftPromptSignalingGame:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        packet_len: int = 4,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.dtype = torch.float32
        self.packet_len = packet_len
        self.compiler = VaultCompiler()
        self.total_epochs_trained = 0

        print(f"[SignalingGame] Loading agent model '{model_name}' onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype
        ).to(self.device)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size

        self.comm_adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.packet_len),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.packet_len, self.hidden_dim * self.packet_len)
        ).to(self.device).to(self.dtype)

        self.optimizer = torch.optim.AdamW(self.comm_adapter.parameters(), lr=1e-3, weight_decay=1e-4)

        self._load_checkpoint()
        print(f"[SignalingGame] Ready. Anchors: {len(TRAIN_VOCABULARY)} | Held-Out: {len(HELDOUT_TEST_VOCABULARY)}")

    def _save_checkpoint(self, loss: float):
        """Saves adapter state dict to disk."""
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "total_epochs": self.total_epochs_trained,
            "model_state_dict": self.comm_adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_loss": loss,
            "timestamp": datetime.now().isoformat()
        }, CHECKPOINT_PATH)
        print(f"[Persistence] Checkpoint saved ({CHECKPOINT_PATH.name}, Epochs: {self.total_epochs_trained})")

    def _load_checkpoint(self):
        """Restores adapter weights from disk if available."""
        if CHECKPOINT_PATH.exists():
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.comm_adapter.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.total_epochs_trained = checkpoint.get("total_epochs", 0)
            print(f"[Persistence] Loaded checkpoint ({self.total_epochs_trained} total epochs)")
        else:
            print("[Persistence] Initializing fresh adapter weights.")

    def _extract_concept_vector(self, text: str) -> torch.Tensor:
        """Extracts penultimate layer hidden state vector for a concept."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-2][0, -1, :].to(self.dtype)
            norm_vec = hidden / torch.norm(hidden, p=2)
        return norm_vec

    def sender_encode(self, concept_vec: torch.Tensor) -> torch.Tensor:
        """Encodes concept vector into a k-token soft prompt packet."""
        projected = self.comm_adapter(concept_vec)
        packet = projected.view(1, self.packet_len, self.hidden_dim)
        packet = packet / torch.norm(packet, dim=-1, keepdim=True)
        return packet

    def train_game(self, num_epochs: int = 20) -> float:
        """Trains the communication adapter on TRAIN_VOCABULARY."""
        self.comm_adapter.train()
        with torch.no_grad():
            train_vectors = torch.stack([self._extract_concept_vector(c) for c in TRAIN_VOCABULARY])

        last_loss = 0.0
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            self.optimizer.zero_grad()

            for target_idx, concept in enumerate(TRAIN_VOCABULARY):
                target_vec = train_vectors[target_idx]
                packet = self.sender_encode(target_vec)

                packet_summary = torch.mean(packet.squeeze(0), dim=0)
                packet_summary = packet_summary / torch.norm(packet_summary, p=2)

                logits = F.cosine_similarity(packet_summary.unsqueeze(0), train_vectors) / 0.07
                target_label = torch.tensor([target_idx], device=self.device)

                loss = F.cross_entropy(logits.unsqueeze(0), target_label)
                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
            last_loss = total_loss / len(TRAIN_VOCABULARY)

        self.total_epochs_trained += num_epochs
        self._save_checkpoint(last_loss)
        self.comm_adapter.eval()
        return last_loss

    def evaluate_generalization(self) -> Dict:
        """Evaluates zero-shot discrimination accuracy against the full concept pool."""
        self.comm_adapter.eval()
        full_pool = TRAIN_VOCABULARY + HELDOUT_TEST_VOCABULARY
        with torch.no_grad():
            candidate_vectors = torch.stack([self._extract_concept_vector(c) for c in full_pool])

        correct = 0
        confidences = []

        for idx, target in enumerate(HELDOUT_TEST_VOCABULARY):
            target_vec = self._extract_concept_vector(target)
            packet = self.sender_encode(target_vec)

            packet_summary = torch.mean(packet.squeeze(0), dim=0)
            packet_summary = packet_summary / torch.norm(packet_summary, p=2)

            similarities = F.cosine_similarity(packet_summary.unsqueeze(0), candidate_vectors)
            best_idx = torch.argmax(similarities).item()
            confidence = similarities[best_idx].item()
            predicted = full_pool[best_idx]

            is_correct = (predicted == target)
            if is_correct:
                correct += 1
            confidences.append(confidence)

        accuracy = correct / len(HELDOUT_TEST_VOCABULARY)
        avg_conf = sum(confidences) / len(confidences)

        exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        findings = (
            f"Evaluated soft-prompt generalization across {len(HELDOUT_TEST_VOCABULARY)} held-out concepts "
            f"against {len(full_pool)} total candidates (k={self.packet_len}, Total Epochs: {self.total_epochs_trained}). "
            f"Zero-shot discrimination achieved {accuracy * 100:.1f}% accuracy ({correct}/{len(HELDOUT_TEST_VOCABULARY)}) "
            f"with mean confidence {avg_conf:.4f}."
        )

        metrics = {
            "cosine_sim": avg_conf,
            "success": accuracy >= 0.75,
            "accuracy": accuracy
        }

        note_path = self.compiler.record_experiment(
            experiment_id=exp_id,
            target_concept="Zero-Shot Soft-Prompt Generalization",
            source_layer=self.model.config.num_hidden_layers - 2,
            receiver_layer=self.model.config.num_hidden_layers - 2,
            metrics=metrics,
            findings_summary=findings
        )

        return {
            "experiment_id": exp_id,
            "accuracy": accuracy,
            "avg_confidence": avg_conf,
            "note_path": str(note_path)
        }