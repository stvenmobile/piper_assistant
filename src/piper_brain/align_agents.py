import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

CORPUS_PATH = Path("data/checkpoints/concepts_dictionary.json")

def load_anchor_corpus() -> list[str]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Concept dictionary not found at {CORPUS_PATH}")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        domains = json.load(f)
    corpus = []
    for domain, phrases in domains.items():
        corpus.extend(phrases)
    return corpus

class LatentSpaceAligner:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", target_model_name="Qwen/Qwen2.5-0.5B-Instruct", layer_idx=18):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer_idx = layer_idx
        
        print(f"[Aligner] Loading source model ({model_name})...")
        self.tokenizer_src = AutoTokenizer.from_pretrained(model_name)
        self.model_src = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        
        print(f"[Aligner] Loading target model ({target_model_name})...")
        self.tokenizer_tgt = AutoTokenizer.from_pretrained(target_model_name)
        self.model_tgt = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16).to(self.device)
        
        self.alignment_matrix = None

    def _extract_activations(self, model, tokenizer, texts):
        activations = []
        model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.layer_idx]
                mean_repr = hidden_states.mean(dim=1).squeeze(0).float()
                activations.append(mean_repr)
        return torch.stack(activations)

    def compute_procrustes_alignment(self, anchor_texts: list[str]):
        print(f"[Aligner] Extracting representations across {len(anchor_texts)} anchor concepts...")
        X = self._extract_activations(self.model_src, self.tokenizer_src, anchor_texts)
        Y = self._extract_activations(self.model_tgt, self.tokenizer_tgt, anchor_texts)

        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        H = (Y_centered.T @ X_centered).cpu()
        U, S, Vh = torch.linalg.svd(H)
        W = (U @ Vh).to(self.device)

        self.alignment_matrix = W
        print("[Aligner] Alignment matrix computed successfully on CPU fallback.")
        return W

    def translate_latent(self, z_source: torch.Tensor) -> torch.Tensor:
        if self.alignment_matrix is None:
            raise ValueError("Alignment matrix not computed. Run compute_procrustes_alignment first.")
        return z_source @ self.alignment_matrix

if __name__ == "__main__":
    anchors = load_anchor_corpus()
    print(f"[Aligner] Loaded {len(anchors)} anchor concepts from dictionary.")
    aligner = LatentSpaceAligner()
    W = aligner.compute_procrustes_alignment(anchors)
    Path("data/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(W, "data/checkpoints/agent_alignment_matrix.pt")
    print("[Aligner] Saved alignment matrix to data/checkpoints/agent_alignment_matrix.pt")