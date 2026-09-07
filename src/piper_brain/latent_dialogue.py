import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENTS_DIR = Path("obsidian/Experiments")
ALIGNMENT_PATH = Path("data/checkpoints/agent_alignment_matrix.pt")

class DirectLatentInjectionDialogue:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", layer_idx=18):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer_idx = layer_idx
        print(f"[Latent Injector] Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.alignment_matrix = torch.load(ALIGNMENT_PATH).to(self.device, dtype=self.model.dtype)
        self.model.eval()

    def text_to_latent(self, text: str, layer_idx=18) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Mean pool across sequence length to yield a stable concept vector: (hidden_dim,)
            hidden = outputs.hidden_states[layer_idx].mean(dim=1).squeeze(0)
        return hidden

    def _get_injection_hook(self, target_vector: torch.Tensor):
        """Creates a forward hook to substitute layer hidden states with the broadcasted concept vector."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Dynamically broadcast target vector (hidden_dim,) to match hidden_states shape (batch, seq_len, hidden_dim)
                injected = target_vector.view(1, 1, -1).expand_as(hidden_states)
                return (injected.to(hidden_states.dtype),) + output[1:]
            else:
                injected = target_vector.view(1, 1, -1).expand_as(output)
                return injected.to(output.dtype)
        return hook

    def latent_to_verbal_response(self, projected_vector: torch.Tensor, max_new_tokens=64) -> str:
        dummy_input = self.tokenizer(".", return_tensors="pt").to(self.device)
        
        layer_module = self.model.model.layers[self.layer_idx]
        hook_handle = layer_module.register_forward_hook(self._get_injection_hook(projected_vector))

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **dummy_input,
                    max_new_tokens=max_new_tokens,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        finally:
            hook_handle.remove()

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    def run_round_trip(self, initial_concept: str) -> dict:
        z1 = self.text_to_latent(initial_concept)
        z1_projected = z1 @ self.alignment_matrix
        r1_text = self.latent_to_verbal_response(z1_projected)

        q2_concept = f"Refinement trajectory based on residual state of: {initial_concept}"
        z2 = self.text_to_latent(q2_concept)
        z2_projected = z2 @ self.alignment_matrix
        r2_text = self.latent_to_verbal_response(z2_projected)

        return {
            "q1": initial_concept,
            "r1": r1_text,
            "q2": q2_concept,
            "r2": r2_text
        }

    def log_markdown(self, concept: str, exchange: dict):
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        now_dt = datetime.now()
        exp_id = f"P3LOOP-{now_dt.strftime('%Y%m%d-%H%M%S')}"
        note_file = EXPERIMENTS_DIR / f"{exp_id}.md"

        content = f"""---
id: {exp_id}
type: direct_latent_injection
cycle_prefix: P3LOOP
date: '{now_dt.isoformat()}'
target_concept: {concept}
turns: 2
status: completed
tags:
- p3loop
- latent_injection
- hook_decoding
---

# Experiment: {exp_id}

**Cycle Track**: `P3LOOP`
**Initial Concept**: {concept}
**Timestamp**: {now_dt.strftime("%Y-%m-%d %H:%M:%S")}

## Direct Residual Stream Injection Log
1. **First Query (Piper)**: {exchange['q1']}
2. **First Response (Hook Reconstruction)**: {exchange['r1']}
3. **Second Query (Piper Refinement)**: {exchange['q2']}
4. **Second Response (Hook Reconstruction)**: {exchange['r2']}

## Evaluation
- **Fidelity Check**: Evaluated direct hidden state substitution via PyTorch forward hooks with fixed-shape broadcasting.
"""
        note_file.write_text(content, encoding="utf-8")
        print(f"[Injection] Logged experiment note to {note_file}")

if __name__ == "__main__":
    engine = DirectLatentInjectionDialogue()
    concept = "All photons travel at the speed of light"
    exchange = engine.run_round_trip(concept)
    engine.log_markdown(concept, exchange)