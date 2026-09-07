"""
Autonomous Phase 2 Parameter Optimizer for Piper Supervisor.
Features Latent Manifold Jittering, Multi-Head Attention Pooling, 
Cosine Annealing, and Margin-Enforced Grounding.
"""

import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nonverbal_tools.test_receiver_grounding import (
    LatentExtractor,
    TransmitterCompressor,
    DATASET_CONCEPTS,
    pearson_correlation
)

# Focused parameter grid centered on empirical sweet spots
PARAM_GRID = {
    "temperature": [0.07, 0.08, 0.10],
    "cos_weight": [4.0, 6.0, 8.0],
    "infonce_weight": [2.0, 2.5],
    "margin_weight": [0.5, 1.0],
    "jitter_std": [0.015, 0.02],
    "lr": [2e-4, 3e-4]
}


class AttentionReceiverDecoder(nn.Module):
    """Decodes k x D packet into D using learned cross-attention pooling."""
    def __init__(self, hidden_dim: int = 896, k_tokens: int = 8):
        super().__init__()
        self.k_tokens = k_tokens
        self.hidden_dim = hidden_dim

        # Learned query vector to pool k tokens into 1 vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        # packet: (B, k, D)
        B = packet.size(0)
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, D)
        
        # Cross-attention: query attends over the k packet tokens
        attn_out, _ = self.attn(query, packet, packet)  # (B, 1, D)
        pooled = attn_out.squeeze(1)  # (B, D)
        
        return self.mlp(pooled)


class AutonomousReceiverOptimizer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.extractor = LatentExtractor(layer_idx=18, device=device)
        self.raw_latents = self.extractor.extract_batch(DATASET_CONCEPTS)
        self.train_split = 28

    def standardize_manifold(self, latents: torch.Tensor) -> torch.Tensor:
        """L2-Normalized Mean Centering to preserve geometric rank."""
        train_latents = latents[:self.train_split]
        mean = train_latents.mean(dim=0, keepdim=True)
        centered = latents - mean
        return F.normalize(centered, p=2, dim=-1)

    def run_trial(self, params: dict, epochs: int = 350) -> dict:
        standardized = self.standardize_manifold(self.raw_latents)
        train_data = standardized[:self.train_split]
        val_data = standardized[self.train_split:]

        tx = TransmitterCompressor(hidden_dim=self.extractor.hidden_dim, k_tokens=8).to(self.device)
        rx = AttentionReceiverDecoder(hidden_dim=self.extractor.hidden_dim, k_tokens=8).to(self.device)

        optimizer = torch.optim.AdamW(
            list(tx.parameters()) + list(rx.parameters()),
            lr=params.get("lr", 5e-4),
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        jitter_std = params.get("jitter_std", 0.02)
        temp = params.get("temperature", 0.05)
        cos_w = params.get("cos_weight", 6.0)
        infonce_w = params.get("infonce_weight", 2.0)
        margin_w = params.get("margin_weight", 1.0)

        for _ in range(epochs):
            tx.train()
            rx.train()
            optimizer.zero_grad()

            # Dynamic continuous augmentation on anchors
            jitter = torch.randn_like(train_data) * jitter_std
            augmented_train = F.normalize(train_data + jitter, p=2, dim=-1)

            packets = tx(augmented_train)
            decoded = rx(packets)
            d_norm = F.normalize(decoded, p=2, dim=-1)

            # 1. Cosine Loss
            cos_loss = 1.0 - torch.sum(d_norm * train_data, dim=-1).mean()

            # 2. InfoNCE Contrastive Loss
            sim_matrix = torch.matmul(d_norm, train_data.T) / temp
            targets = torch.arange(self.train_split, device=self.device)
            infonce_loss = F.cross_entropy(sim_matrix, targets)

            # 3. Hard-Negative Margin Loss
            mask = ~torch.eye(self.train_split, dtype=torch.bool, device=self.device)
            distractor_sims = sim_matrix[mask].view(self.train_split, -1) * temp
            hardest_neg, _ = torch.max(distractor_sims, dim=-1)
            pos_sim = torch.sum(d_norm * train_data, dim=-1)
            margin_loss = F.relu(0.20 - (pos_sim - hardest_neg)).mean()

            loss = cos_w * cos_loss + infonce_w * infonce_loss + margin_w * margin_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tx.parameters()) + list(rx.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation Evaluation on held-out concepts (N=12)
        tx.eval()
        rx.eval()
        with torch.no_grad():
            val_packets = tx(val_data)
            val_decoded = rx(val_packets)
            v_norm = F.normalize(val_decoded, p=2, dim=-1)

            # Top-1 Retrieval across held-out candidates
            sims = torch.matmul(v_norm, val_data.T)
            correct = 0
            for i in range(len(val_data)):
                if torch.argmax(sims[i]).item() == i:
                    correct += 1

            acc = (correct / len(val_data)) * 100.0

            # Neighborhood Distance Correlation
            orig_dist = 1.0 - torch.matmul(val_data, val_data.T)
            deco_dist = 1.0 - torch.matmul(v_norm, v_norm.T)
            mask = ~torch.eye(len(val_data), dtype=torch.bool, device=self.device)
            corr = pearson_correlation(orig_dist[mask], deco_dist[mask])
            cossim = torch.sum(v_norm * val_data, dim=-1).mean().item()

        return {
            "params": params,
            "accuracy": acc,
            "correlation": corr,
            "mean_cossim": round(cossim, 4)
        }


if __name__ == "__main__":
    optimizer = AutonomousReceiverOptimizer()
    sample_params = {
        "temperature": 0.05,
        "cos_weight": 6.0,
        "infonce_weight": 2.0,
        "margin_weight": 1.0,
        "jitter_std": 0.02,
        "lr": 5e-4
    }
    print("[Optimizer] Running standalone test trial (350 epochs)...")
    res = optimizer.run_trial(sample_params, epochs=350)
    print(f"[Results] Accuracy: {res['accuracy']:.1f}%, Correlation: {res['correlation']:.4f}, CosSim: {res['mean_cossim']}")