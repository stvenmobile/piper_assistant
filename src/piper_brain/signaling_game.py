"""
Piper Brain: Autonomous Latent Signaling Game and Concept Alignment Engine.
Features FP32 loss stabilization, gradient clipping, k=8 soft tokens, and layer sweeps.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SignalingGame:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        k_tokens: int = 8,
        source_layer: int = 18,
        receiver_layer: int = 18,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.k_tokens = k_tokens
        self.source_layer = source_layer
        self.receiver_layer = receiver_layer
        self.learning_rate = learning_rate
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[SignalingGame] Loading agent model '{model_name}' onto {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype,
            device_map=self.device
        )
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.soft_prompt_dim = self.k_tokens * self.hidden_dim

        # Non-linear MLP Adapter
        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, self.soft_prompt_dim),
            nn.GELU(),
            nn.Linear(self.soft_prompt_dim, self.soft_prompt_dim)
        ).to(device=self.device, dtype=self.model_dtype)

        self._init_adapter_weights()
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.learning_rate)

    def _init_adapter_weights(self):
        """Initializes adapter linear layers with Xavier uniform weights."""
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_layer_representation(self, text: str, layer_idx: int) -> torch.Tensor:
        """Extracts the mean-pooled hidden state representation at a specific layer."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
            pooled = hidden_state.mean(dim=1).to(dtype=self.model_dtype)  # (1, hidden_dim)
        return pooled

    def generate_soft_prompt(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Projects concept representation into k soft tokens."""
        batch_size = latent_state.size(0)
        projected = self.adapter(latent_state)
        return projected.view(batch_size, self.k_tokens, self.hidden_dim)

    def train_burst(
        self,
        anchors: Dict[str, str],
        epochs: int = 10,
        temperature: float = 0.1,
        max_grad_norm: float = 1.0
    ) -> float:
        """Trains communication adapter using InfoNCE loss computed in FP32 with gradient clipping."""
        if not anchors:
            return 0.0

        self.adapter.train()
        labels = list(anchors.keys())
        total_loss = 0.0

        with torch.no_grad():
            latents = torch.cat([
                self.extract_layer_representation(anchors[lbl], self.source_layer)
                for lbl in labels
            ], dim=0)  # (N, hidden_dim)

        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            projected = self.adapter(latents)  # (N, k * hidden_dim)
            projected_anchors = projected.view(len(labels), self.k_tokens, self.hidden_dim).mean(dim=1)

            # Upcast to float32 to prevent half-precision exponent overflow
            p_norm = F.normalize(projected_anchors.float(), p=2, dim=-1)
            t_norm = F.normalize(latents.float(), p=2, dim=-1)
            sim_matrix = torch.matmul(p_norm, t_norm.T) / temperature

            targets = torch.arange(len(labels), device=self.device)
            loss = F.cross_entropy(sim_matrix, targets)

            # Guard against NaN before stepping
            if torch.isnan(loss) or torch.isinf(loss):
                print("[SignalingGame] NaN/Inf detected in loss! Performing self-healing weight reset.")
                self._init_adapter_weights()
                self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.learning_rate)
                return float("nan")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        self.adapter.eval()
        return total_loss / max(epochs, 1)

    def evaluate_layer_sweep(
        self,
        anchors: Dict[str, str],
        held_out: Dict[str, str],
        start_layer: int = 14,
        end_layer: int = 22
    ) -> List[Dict]:
        """Sweeps layers start_layer to end_layer, measuring zero-shot discrimination accuracy."""
        self.adapter.eval()
        results = []
        end_layer = min(end_layer, self.num_layers)

        print(f"\n[SignalingGame] Initiating Layer Sweep across Layers {start_layer} to {end_layer} (k={self.k_tokens})...")

        for layer in range(start_layer, end_layer + 1):
            correct = 0
            total = len(held_out)
            cosine_similarities = []

            candidate_latents = {}
            for label, text in {**anchors, **held_out}.items():
                candidate_latents[label] = self.extract_layer_representation(text, layer)

            for label, text in held_out.items():
                source_latent = self.extract_layer_representation(text, layer)
                
                with torch.no_grad():
                    projected = self.adapter(source_latent)
                    projected_concept = projected.view(1, self.k_tokens, self.hidden_dim).mean(dim=1)

                scores = {}
                p_vec = F.normalize(projected_concept.float(), p=2, dim=-1)
                for cand_label, cand_latent in candidate_latents.items():
                    c_vec = F.normalize(cand_latent.float(), p=2, dim=-1)
                    sim = torch.sum(p_vec * c_vec).item()
                    scores[cand_label] = sim

                top_prediction = max(scores, key=scores.get)
                target_sim = scores[label]
                cosine_similarities.append(target_sim)

                if top_prediction == label:
                    correct += 1

            accuracy = (correct / total) * 100.0 if total > 0 else 0.0
            mean_cossim = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0

            layer_result = {
                "layer": layer,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "mean_cossim": round(mean_cossim, 4)
            }
            results.append(layer_result)
            print(f"  * Layer {layer:02d} | Accuracy: {accuracy:5.1f}% ({correct}/{total}) | Mean CosSim: {mean_cossim:.4f}")

        best_layer = max(results, key=lambda x: (x["accuracy"], x["mean_cossim"]))
        print(f"[SignalingGame] Optimal layer identified: Layer {best_layer['layer']} ({best_layer['accuracy']:.1f}% Acc, {best_layer['mean_cossim']:.4f} CosSim)\n")
        return results

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Loads adapter weights with validation."""
        if not os.path.exists(checkpoint_path):
            print(f"[Persistence] No checkpoint found at {checkpoint_path}. Starting fresh adapter (k={self.k_tokens}).")
            return 0

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            saved_k = ckpt.get("k_tokens", 4)

            if saved_k != self.k_tokens:
                print(f"[Persistence] Token mismatch (saved: k={saved_k}, current: k={self.k_tokens}). Initializing new adapter.")
                return 0

            self.adapter.load_state_dict(ckpt["adapter_state"])
            total_epochs = ckpt.get("total_epochs", 0)
            self.source_layer = ckpt.get("source_layer", self.source_layer)
            self.receiver_layer = ckpt.get("receiver_layer", self.receiver_layer)
            print(f"[Persistence] Loaded checkpoint ({total_epochs} total epochs, Layer {self.source_layer}, k={self.k_tokens})")
            return total_epochs
        except Exception as e:
            print(f"[Persistence] Error loading checkpoint: {e}. Starting fresh adapter.")
            return 0

    def save_checkpoint(self, checkpoint_path: str, total_epochs: int):
        """Persists current weights and hyperparameters."""
        torch.save({
            "adapter_state": self.adapter.state_dict(),
            "total_epochs": total_epochs,
            "k_tokens": self.k_tokens,
            "source_layer": self.source_layer,
            "receiver_layer": self.receiver_layer
        }, checkpoint_path)
        print(f"[Persistence] Checkpoint saved ({Path(checkpoint_path).name}, Epochs: {total_epochs}, Layer: {self.source_layer})")