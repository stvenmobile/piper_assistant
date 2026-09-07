"""
Phase 1 Transmitter Verification: Invertibility & Feature Variance Benchmark.
Evaluates whether an MLP encoder can compress Layer 18 hidden states into k=8
continuous vectors without representation collapse or loss of semantic integrity.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BENCHMARK_CONCEPTS = [
    "Gravity is the curvature of spacetime caused by mass and energy.",
    "Entropy dictates that the disorder of an isolated system always increases.",
    "Photosynthesis converts solar light into chemical energy inside chloroplasts.",
    "Natural selection drives biological evolution through differential reproduction.",
    "Special relativity posits that the speed of light is invariant across reference frames.",
    "Quantum superposition allows particles to exist across multiple states simultaneously.",
    "Plate tectonics describes large-scale motions of Earth's lithospheric plates.",
    "Cellular respiration converts biochemical energy from nutrients into ATP.",
    "Electromagnetic induction generates voltage across a conductor in a changing magnetic field.",
    "Thermodynamic equilibrium is reached when all net macroscopic flows cease.",
    "Action potentials propagate electrical signals along the axons of neurons.",
    "DNA replication uses DNA polymerase to produce two identical DNA strands.",
    "General relativity describes gravitation as metric curvature of spacetime.",
    "Quantum entanglement binds particle properties regardless of spatial separation.",
    "Dark matter exerts gravitational attraction without emitting electromagnetic radiation.",
    "CRISPR-Cas9 provides RNA-guided targeted cleavage of specific DNA sequences.",
    "Hubble's law demonstrates that galaxies recede at speeds proportional to their distance.",
    "Synaptic plasticity alters the strength of neural connections based on activity.",
    "Osmosis is the net movement of water molecules across a semipermeable membrane.",
    "Heisenberg uncertainty principle limits precision of conjugate physical variables.",
    "Nuclear fusion combines atomic nuclei to release binding energy.",
    "Endosymbiosis explains the evolutionary origin of eukaryotic mitochondria.",
    "Fermat's principle states that light traverses the path of least time.",
    "Epigenetics involves phenotypic modifications without alterations to underlying DNA.",
    "Superconductivity allows zero electrical resistance below a critical temperature.",
    "Apoptosis is programmed cell death required for biological tissue homeostasis.",
    "Bayesian inference updates probability estimates as new evidence is acquired.",
    "Information entropy quantifies the expected amount of information in a message.",
    "Brownian motion describes the random drift of particles suspended in a medium.",
    "Wave-particle duality shows physical entities exhibit both wave and particle aspects."
]


class LatentExtractor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        layer_idx: int = 18,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[Extractor] Loading {model_name} on {device} (Layer {layer_idx})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            device_map=self.device
        )
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size

    def extract_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer_idx]
            mask = inputs.attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        return pooled.float()


class TransmitterCompressor(nn.Module):
    """Encodes continuous thought vector (D) into k soft tokens (k x D)."""
    def __init__(self, hidden_dim: int = 896, k_tokens: int = 8):
        super().__init__()
        self.k_tokens = k_tokens
        self.hidden_dim = hidden_dim
        self.soft_dim = k_tokens * hidden_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, self.soft_dim),
            nn.GELU(),
            nn.LayerNorm(self.soft_dim),
            nn.Linear(self.soft_dim, self.soft_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = self.net(z)
        return proj.view(-1, self.k_tokens, self.hidden_dim)


class InverseReadoutHead(nn.Module):
    """Linear readout probe to test invertibility: R(k x D) -> D."""
    def __init__(self, hidden_dim: int = 896, k_tokens: int = 8):
        super().__init__()
        self.linear = nn.Linear(k_tokens * hidden_dim, hidden_dim)

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        flat = packet.view(packet.size(0), -1)
        return self.linear(flat)


def train_and_evaluate_transmitter(
    hidden_dim: int = 896,
    k_tokens: int = 8,
    epochs: int = 150,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    extractor = LatentExtractor(layer_idx=18, device=device)
    print(f"[Benchmark] Extracting embeddings for {len(BENCHMARK_CONCEPTS)} concepts...")
    
    all_latents = extractor.extract_batch(BENCHMARK_CONCEPTS)
    train_latents = all_latents[:24]
    val_latents = all_latents[24:]

    compressor = TransmitterCompressor(hidden_dim=extractor.hidden_dim, k_tokens=k_tokens).to(device)
    readout = InverseReadoutHead(hidden_dim=extractor.hidden_dim, k_tokens=k_tokens).to(device)

    optimizer = torch.optim.AdamW(list(compressor.parameters()) + list(readout.parameters()), lr=lr)

    print(f"\n[Training] Optimizing Transmitter Invertibility across {epochs} epochs (k={k_tokens})...")

    for epoch in range(1, epochs + 1):
        compressor.train()
        readout.train()
        optimizer.zero_grad()

        packet = compressor(train_latents)
        reconstructed = readout(packet)

        mse_loss = F.mse_loss(reconstructed, train_latents)
        cos_loss = 1.0 - F.cosine_similarity(reconstructed, train_latents, dim=-1).mean()
        
        # Log-barrier variance loss: strictly penalizes near-zero standard deviations
        packet_flat = packet.view(packet.size(0), -1)
        std_packet = torch.sqrt(packet_flat.var(dim=0) + 1e-6)
        var_loss = -torch.log(std_packet.clamp(min=1e-4)).mean()

        loss = mse_loss + cos_loss + 0.05 * var_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:03d} | Total: {loss.item():.4f} | MSE: {mse_loss.item():.4f} | CosLoss: {cos_loss.item():.4f} | VarLoss: {var_loss.item():.4f}")

    # Evaluation
    compressor.eval()
    readout.eval()
    with torch.no_grad():
        # 1. Evaluate Reconstruction Fidelity on Held-Out Concepts (N=6)
        val_packets = compressor(val_latents)
        val_reconstructed = readout(val_packets)
        cos_sims = F.cosine_similarity(val_reconstructed, val_latents, dim=-1)
        mean_cos_sim = cos_sims.mean().item()

        # 2. Evaluate Non-Collapse Variance Across the Full 30-Concept Manifold
        all_packets = compressor(all_latents)
        all_flat = all_packets.view(all_packets.size(0), -1)
        all_std = torch.sqrt(all_flat.var(dim=0) + 1e-6)
        min_feature_variance = all_std.min().item()
        zero_var_dims = int((all_std < 0.05).sum().item())

    print("\n" + "=" * 60)
    print("PHASE 1 TRANSMITTER VERIFICATION RESULTS")
    print("=" * 60)
    print(f"1. Mean Reconstruction Cosine Similarity: {mean_cos_sim:.4f} (Target: >= 0.9200)")
    print(f"2. Minimum Dimension Variance (Full Set): {min_feature_variance:.4f} (Target: > 0.0500)")
    print(f"3. Collapsed / Zero-Variance Dimensions:  {zero_var_dims}/{all_flat.size(1)} (Target: 0)")

    m1_1_passed = zero_var_dims == 0
    m1_2_passed = mean_cos_sim >= 0.92

    print("-" * 60)
    print(f"Milestone 1.1 (Non-Collapse):               {'[PASS]' if m1_1_passed else '[FAIL]'}")
    print(f"Milestone 1.2 (Reconstruction Fidelity):   {'[PASS]' if m1_2_passed else '[FAIL]'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_evaluate_transmitter()