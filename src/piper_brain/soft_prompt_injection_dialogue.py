import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENTS_DIR = Path("obsidian/Experiments")
ALIGNMENT_PATH = Path("data/checkpoints/agent_alignment_matrix.pt")

class SoftPromptInjectionDialogue:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", layer_idx=18, num_virtual_tokens=4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer_idx = layer_idx
        self.num_virtual_tokens = num_virtual_tokens
        print(f"[Soft Prompt Injector] Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.alignment_matrix = torch.load(ALIGNMENT_PATH).to(self.device, dtype=self.model.dtype)
        self.model.eval()

    def text_to_latent(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[self.layer_idx].mean(dim=1).squeeze(0)
        return hidden

    def latent_to_verbal_response(self, projected_vector: torch.Tensor, max_new_tokens=64) -> str:
        hidden_dim = projected_vector.shape[-1]
        # Replicate projected vector across virtual token slots: shape (1, num_virtual_tokens, hidden_dim)
        virtual_embeds = projected_vector.view(1, 1, hidden_dim).expand(1, self.num_virtual_tokens, hidden_dim)
        
        prompt_text = "After translation, this is what I heard you to be saying:"
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        
        embedding_layer = self.model.get_input_embeddings()
        text_embeds = embedding_layer(input_ids)
        
        # Concatenate continuous soft prompt embeddings as prefix to input token embeddings
        inputs_embeds = torch.cat([virtual_embeds.to(text_embeds.dtype), text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
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
type: soft_prompt_injection
cycle_prefix: P3LOOP
date: '{now_dt.isoformat()}'
target_concept: {concept}
turns: 2
status: completed
tags:
- p3loop
- soft_prompt
- embedding_injection
---

# Experiment: {exp_id}

**Cycle Track**: `P3LOOP`
**Initial Concept**: {concept}
**Timestamp**: {now_dt.strftime("%Y-%m-%d %H:%M:%S")}

## Soft Prompt Embedding Injection Log
1. **First Query (Piper)**: {exchange['q1']}
2. **First Response (Soft Prompt Reconstruction)**: {exchange['r1']}
3. **Second Query (Piper Refinement)**: {exchange['q2']}
4. **Second Response (Soft Prompt Reconstruction)**: {exchange['r2']}

## Evaluation
- **Fidelity Check**: Evaluated continuous embedding prefix injection via `inputs_embeds` concatenation.
"""
        note_file.write_text(content, encoding="utf-8")
        print(f"[SoftPrompt] Logged experiment note to {note_file}")

if __name__ == "__main__":
    engine = SoftPromptInjectionDialogue()
    concept = "All photons travel at the speed of light"
    exchange = engine.run_round_trip(concept)
    engine.log_markdown(concept, exchange)