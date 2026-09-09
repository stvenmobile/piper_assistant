"""
Piper Geometry: Trained Cross-Model Translation Adapter.

Everything built so far computed the source-to-target map once, in
closed form (build_rotation's SVD), and never touched it again. This
module replaces that fixed map with a small trainable module, optimized
via real gradient descent against a task-based loss - see
obsidian/Journals/2026-09-09.md for the reasoning: a rotation fit only to
match calibration vectors geometrically has no reason to produce
something the *receiving* model's frozen downstream layers know how to
use, and today's results (self round-trip losing topic even with a
perfect, untranslated vector) show that gap is real, not hypothetical.

Both base models (Qwen2.5-0.5B-Instruct, Phi-4-mini-instruct) stay
completely frozen throughout - nothing here fine-tunes either large
model. Only the small bridging module between them is trained, keeping
the actual question ("can two independently-trained, unmodified models
be bridged?") intact rather than quietly becoming "can one model be
retrained to accommodate the other."
"""

import sys
import json
import random
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from piper_geometry.extractor import ResidualExtractor
from piper_geometry.congruence_optimizer import CROSS_MODEL_CONFIGS, DEFAULT_MODEL
from piper_geometry.layer_injection import (
    build_rotation, extract_full_sequence, LayerInjectionHook, PartialLayerInjectionHook,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONCEPTS_PATH = REPO_ROOT / "data" / "checkpoints" / "concepts_dictionary.json"
CHECKPOINT_PATH = REPO_ROOT / "data" / "checkpoints" / "trained_adapter.pt"

TARGET_PREFIX = "XALIGNPHI"
CALIBRATION_SIZE = 280
RNG_SEED = 1234

# The closed-set classification labels the adapter is trained to make the
# receiving model predict correctly. Single common English words, not the
# concepts_dictionary.json domain keys themselves (e.g. "computer_science")
# - those aren't single clean tokens in Phi-4-mini's tokenizer, and a
# multi-token label complicates reading "the" next-token prediction out
# as one classification decision. Order matches DOMAINS in
# concept_transfer_test.py / ctransfer_batch.py for consistency.
DOMAIN_LABELS = {
    "physics": "physics",
    "computer_science": "computers",
    "philosophy": "philosophy",
    "mathematics": "math",
    "cognitive_science": "psychology",
    "biology": "biology",
}

PROMPT_SUFFIX = " This concept belongs to the field of"


class TranslationAdapter(nn.Module):
    """Trainable replacement for the fixed closed-form rotation W plus the
    hand-derived scale factor. Where W was computed once via SVD and never
    touched again, this is a real nn.Module - its weight and bias are
    nn.Parameters an optimizer updates via backprop against a real task
    loss, not a one-shot geometric fit. Structurally still just a linear
    map (same shape family as W) for a first version; a small MLP is a
    natural upgrade later if a linear map proves insufficient."""

    def __init__(self, source_dim: int, target_dim: int, warm_start: torch.Tensor = None):
        super().__init__()
        self.linear = nn.Linear(source_dim, target_dim, bias=True)
        if warm_start is not None:
            if warm_start.shape != (source_dim, target_dim):
                raise ValueError(f"warm_start shape {tuple(warm_start.shape)} != ({source_dim}, {target_dim})")
            with torch.no_grad():
                # nn.Linear stores weight as (out_features, in_features) and
                # computes x @ weight.T + bias - warm_start is (in, out) to
                # match how it's used elsewhere (source_hidden @ W), so the
                # transpose here is what lines the two conventions up.
                self.linear.weight.copy_(warm_start.t())
                self.linear.bias.zero_()

    def forward(self, source_states: torch.Tensor) -> torch.Tensor:
        return self.linear(source_states)


def build_warm_start(source_extractor, target_extractor, source_layer, receiver_layer,
                      calibration_concepts, scale_factor: float) -> torch.Tensor:
    """The adapter's starting point: the existing closed-form rotation,
    pre-multiplied by the scale factor already found (by hand, comparing
    mean norms) to help - giving training a reasonable place to start
    from rather than raw geometry alone, or nothing at all."""
    W = build_rotation(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)
    return W * scale_factor


def load_examples(concepts_path: Path = CONCEPTS_PATH, holdout_fraction: float = 0.2,
                   seed: int = RNG_SEED):
    """Everywhere else in the project, concepts_dictionary.json's phrases
    were consumed either as a flat pool (calibration) or one-at-a-time
    (a single test passage). Training needs two things neither of those
    did: (domain, passage) pairs so a loss can be computed against the
    right label, and a held-out split the training loop never samples
    from, so accuracy on it actually measures generalization rather than
    memorization. The split is per-domain (not global) so every domain -
    including the two with only 30 phrases instead of 60 - ends up
    represented in both sets, and it's seeded so re-running training
    starts from the same train/held-out boundary rather than a new
    random one each time."""
    with open(concepts_path, "r", encoding="utf-8") as f:
        by_domain = json.load(f)

    rng = random.Random(seed)
    train, holdout = [], []
    for domain, phrases in by_domain.items():
        if domain not in DOMAIN_LABELS:
            continue
        phrases = list(phrases)
        rng.shuffle(phrases)
        split = max(1, int(len(phrases) * (1 - holdout_fraction)))
        train.extend((domain, phrase) for phrase in phrases[:split])
        holdout.extend((domain, phrase) for phrase in phrases[split:])

    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def resolve_label_token_ids(tokenizer) -> dict:
    """Closed-form rotation testing never needed to name a specific
    vocabulary entry - it only ever compared vectors to each other.
    Cross-entropy loss needs an actual token id to compare logits
    against, so DOMAIN_LABELS' human-readable words have to be resolved
    through the target model's own tokenizer. A word can split into
    several sub-word pieces; only the first is kept, since PROMPT_SUFFIX
    ends right where a single next-token prediction is read, and that
    first piece is the one that prediction is judged against. encode()
    is used with add_special_tokens=False so no BOS/EOS token sneaks in
    ahead of the word piece we actually want."""
    label_token_ids = {}
    for domain, label_word in DOMAIN_LABELS.items():
        # Leading space matters: PROMPT_SUFFIX ends in "field of" with no
        # trailing space, so the model's next token is expected to START a
        # new word - most BPE tokenizers encode that as a distinct,
        # space-prefixed token from the same word appearing mid-string.
        token_ids = tokenizer.encode(" " + label_word, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"label word {label_word!r} for domain {domain!r} encoded to no tokens")
        label_token_ids[domain] = token_ids[0]
    return label_token_ids


def forward_with_injection(model, tokenizer, device, layer_idx: int,
                            injected_states: torch.Tensor, suffix_text: str = PROMPT_SUFFIX) -> torch.Tensor:
    """The training-time counterpart to layer_injection.generate_with_injection:
    one differentiable forward pass (model(...), not generate()) so a loss
    can backpropagate through the injected states into whatever produced
    them. No sampling, no autoregressive loop - the input is a placeholder
    span (standing in for the translated passage, actual content supplied
    by the injection hook) followed by suffix_text's real tokens, and the
    classification decision is read directly off logits at the final
    position: whatever token the model predicts would come right after
    "...field of"."""
    inject_len = injected_states.shape[0]
    placeholder_id = tokenizer.eos_token_id
    placeholder_ids = torch.full((1, inject_len), placeholder_id, dtype=torch.long, device=device)
    suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([placeholder_ids, suffix_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    hook = PartialLayerInjectionHook(injected_states.unsqueeze(0))
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        # Deliberately no torch.no_grad() here - this is the one call in
        # the whole pipeline that must stay differentiable end to end.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    return outputs.logits[0, -1, :]  # (vocab_size,) logits for the position right after the suffix
