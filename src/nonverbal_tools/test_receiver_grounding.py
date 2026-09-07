"""
Phase 2 Receiver Verification: Latent Ingestion, Discriminative Retrieval,
and Topological Neighborhood Preservation Benchmark (with Mean-Centering).
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

DATASET_CONCEPTS = [
    # Training Anchors (28)
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
    
    # Held-Out Evaluation Targets (12)
    "Brownian motion describes the random drift of particles suspended in a medium.",
    "Wave-particle duality shows physical entities exhibit both wave and particle aspects.",
    "Superfluidity allows liquid helium to flow without kinetic viscosity.",
    "Neurogenesis is the biological process by which new nervous tissue is generated.",
    "Plate boundary subduction drives deep mantle convection and volcanic arc formation.",
    "Casimir effect generates an attractive physical force between uncharged conducting plates.",
    "Keplerian orbits dictate planetary motion along elliptical trajectories.",
    "Ribosomes translate messenger RNA sequences into polypeptides during protein synthesis.",
    "Ferromagnetism produces spontaneous magnetic ordering below the Curie temperature.",
    "Stochastic resonance enhances weak signals through optimized ambient noise injection.",
    "Mitochondrial oxidative phosphorylation synthesizes ATP via an electrochemical proton gradient.",
    "Gravitational lensing bends electromagnetic radiation around massive galaxy clusters."
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


class ReceiverDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 896, k_tokens: int = 8):
        super().__init__()
        self.k_tokens = k_tokens
        self.hidden_dim = hidden_dim
        self.in_dim = k_tokens * hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden_dim)
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        flat = packet.view(packet.size(0), -1)
        return self.net(flat)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return float(cost.item())


def train_and_evaluate_receiver(
    hidden_dim: int = 896,
    k_tokens: int = 8,
    train_split: int = 28,
    epochs: int = 150,
    lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    extractor = LatentExtractor(layer_idx=18, device=device)
    print(f"[Dataset] Extracting embeddings for {len(DATASET_CONCEPTS)} total concepts...")

    all_latents_raw = extractor.extract_batch(DATASET_CONCEPTS)
    
    # 1. Compute Manifold Center Vector (Anisotropy Correction)
    train_mean = all_latents_raw[:train_split].mean(dim=0, keepdim=True)
    all_latents = all_latents_raw - train_mean
    
    train_latents = all_latents[:train_split]
    val_latents = all_latents[train_split:]

    transmitter = TransmitterCompressor(hidden_dim=extractor.hidden_dim, k_tokens=k_tokens).to(device)
    receiver = ReceiverDecoder(hidden_dim=extractor.hidden_dim, k_tokens=k_tokens).to(device)

    optimizer = torch.optim.AdamW(list(transmitter.parameters()) + list(receiver.parameters()), lr=lr)

    print(f"\n[Training] Optimizing Centered Transmitter-Receiver Pipeline ({epochs} epochs, k={k_tokens})...")

    for epoch in range(1, epochs + 1):
        transmitter.train()
        receiver.train()
        optimizer.zero_grad()

        packets = transmitter(train_latents)
        decoded = receiver(packets)

        mse_loss = F.mse_loss(decoded, train_latents)
        cos_loss = 1.0 - F.cosine_similarity(decoded, train_latents, dim=-1).mean()

        # Variance Regularization
        packets_flat = packets.view(packets.size(0), -1)
        std_packet = torch.sqrt(packets_flat.var(dim=0) + 1e-6)
        var_loss = -torch.log(std_packet.clamp(min=1e-4)).mean()

        # Sharpened InfoNCE on Mean-Centered representations
        d_norm = F.normalize(decoded, p=2, dim=-1)
        t_norm = F.normalize(train_latents, p=2, dim=-1)
        sim_matrix = torch.matmul(d_norm, t_norm.T) / 0.05
        targets = torch.arange(train_split, device=device)
        contrastive_loss = F.cross_entropy(sim_matrix, targets)

        loss = mse_loss + 2.0 * cos_loss + 1.0 * contrastive_loss + 0.05 * var_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(transmitter.parameters()) + list(receiver.parameters()), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:03d} | Total: {loss.item():.4f} | MSE: {mse_loss.item():.4f} | "
                f"CosLoss: {cos_loss.item():.4f} | InfoNCE: {contrastive_loss.item():.4f} | VarLoss: {var_loss.item():.4f}"
            )

    # Evaluation on Held-Out Validation Concepts (N=12 vs 40 candidate pool)
    transmitter.eval()
    receiver.eval()

    with torch.no_grad():
        val_packets = transmitter(val_latents)
        val_decoded = receiver(val_packets)

        val_norm = F.normalize(val_decoded, p=2, dim=-1)
        all_norm = F.normalize(all_latents, p=2, dim=-1)
        retrieval_sims = torch.matmul(val_norm, all_norm.T)

        correct_top1 = 0
        total_val = len(val_latents)

        for i in range(total_val):
            ground_truth_idx = train_split + i
            predicted_idx = torch.argmax(retrieval_sims[i]).item()
            if predicted_idx == ground_truth_idx:
                correct_top1 += 1

        top1_accuracy = (correct_top1 / total_val) * 100.0

        # Topological Neighborhood Preservation (Distance Correlation)
        orig_dist = 1.0 - torch.matmul(F.normalize(val_latents, p=2, dim=-1), F.normalize(val_latents, p=2, dim=-1).T)
        deco_dist = 1.0 - torch.matmul(val_norm, val_norm.T)

        mask = ~torch.eye(total_val, dtype=torch.bool, device=device)
        orig_flat = orig_dist[mask]
        deco_flat = deco_dist[mask]

        dist_correlation = pearson_correlation(orig_flat, deco_flat)
        mean_cos_alignment = F.cosine_similarity(val_decoded, val_latents, dim=-1).mean().item()

    print("\n" + "=" * 65)
    print("PHASE 2 RECEIVER VERIFICATION RESULTS")
    print("=" * 65)
    print(f"1. Top-1 Concept Retrieval Accuracy:  {top1_accuracy:5.1f}% ({correct_top1}/{total_val}) (Target: >= 85.0%)")
    print(f"2. Neighborhood Distance Correlation: {dist_correlation:.4f} (Target: >= 0.8000)")
    print(f"3. Mean Centered Cosine Alignment:    {mean_cos_alignment:.4f}")

    m2_1_passed = top1_accuracy >= 85.0
    m2_2_passed = dist_correlation >= 0.80

    print("-" * 65)
    print(f"Milestone 2.1 (Top-1 Retrieval):          {'[PASS]' if m2_1_passed else '[FAIL]'}")
    print(f"Milestone 2.2 (Neighborhood Correlation):  {'[PASS]' if m2_2_passed else '[FAIL]'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train_and_evaluate_receiver()