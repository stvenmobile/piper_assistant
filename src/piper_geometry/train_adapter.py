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
    build_rotation, extract_full_sequence, PartialLayerInjectionHook,
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


def _load_calibration_concepts(concepts_path: Path = CONCEPTS_PATH, size: int = CALIBRATION_SIZE,
                                seed: int = RNG_SEED) -> list:
    """Same recipe concept_transfer_test.py's run_test already validated
    for the warm-start rotation: flatten every domain's phrases into one
    pool, shuffle deterministically, take the first `size`. Deliberately
    independent of load_examples's train/held-out split above - the
    warm-start rotation is a one-time geometric fit, not a training
    example, so there's no memorization concern about it seeing phrases
    that also appear in the training set."""
    with open(concepts_path, "r", encoding="utf-8") as f:
        by_domain = json.load(f)
    all_concepts = [phrase for phrases in by_domain.values() for phrase in phrases]
    rng = random.Random(seed)
    rng.shuffle(all_concepts)
    return all_concepts[:size]


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


def compute_scale_factor(source_extractor, target_extractor, source_layer, receiver_layer, concepts: list) -> float:
    """Same fix the qualitative test applied by hand (rotation preserves
    magnitude, so a rotated vector still carries the source model's own
    scale, not the target layer's) - computed here from the calibration
    concepts' last-token vectors rather than one passage's per-token
    vectors, since build_warm_start needs a single scalar before training
    has any passage to look at yet."""
    source_vecs, target_vecs = [], []
    for concept in concepts:
        source_vecs.append(source_extractor.extract_activations(prompt=concept, target_layers=[source_layer])[source_layer])
        target_vecs.append(target_extractor.extract_activations(prompt=concept, target_layers=[receiver_layer])[receiver_layer])
    source_mean_norm = torch.stack(source_vecs).to(torch.float32).norm(dim=-1).mean().item()
    target_mean_norm = torch.stack(target_vecs).to(torch.float32).norm(dim=-1).mean().item()
    return target_mean_norm / source_mean_norm


def compute_loss_and_prediction(adapter, source_extractor, target_extractor, source_layer, receiver_layer,
                                 domain: str, passage: str, label_token_ids: dict):
    """One example's worth of the full pipeline, used by both the training
    step and held-out evaluation so the two can never quietly diverge.
    Qwen's extraction is always no_grad - it stays frozen whether this is
    called during training or evaluation. Whether the adapter's own
    forward pass builds a graph is the caller's choice (wrap the whole
    call in torch.no_grad() for evaluation); this function itself makes
    no such decision."""
    with torch.no_grad():
        source_hidden = extract_full_sequence(
            source_extractor.model, source_extractor.tokenizer, source_extractor.device, passage, source_layer,
        ).to(torch.float32)

    injected_states = adapter(source_hidden.to(next(adapter.parameters()).device))

    logits = forward_with_injection(
        target_extractor.model, target_extractor.tokenizer, target_extractor.device, receiver_layer, injected_states,
    )
    label_id = label_token_ids[domain]
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label_id], device=logits.device))
    predicted_id = int(logits.argmax().item())
    return loss, predicted_id


def evaluate(adapter, source_extractor, target_extractor, source_layer, receiver_layer,
             examples: list, label_token_ids: dict) -> float:
    """Held-out accuracy: the fraction of examples.load_examples's held-out
    split where the injected translation actually makes Phi-4-mini predict
    the right domain - the concrete "did this produce the correct
    downstream behavior" measure the training design settled on, in place
    of judging generation fluency."""
    adapter.eval()
    correct = 0
    with torch.no_grad():
        for domain, passage in examples:
            _, predicted_id = compute_loss_and_prediction(
                adapter, source_extractor, target_extractor, source_layer, receiver_layer, domain, passage, label_token_ids,
            )
            if predicted_id == label_token_ids[domain]:
                correct += 1
    adapter.train()
    return correct / len(examples)


def save_checkpoint(adapter, path: Path, step: int, accuracy: float, source_layer: int, receiver_layer: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter.state_dict(),
        "step": step,
        "held_out_accuracy": accuracy,
        "source_layer": source_layer,
        "receiver_layer": receiver_layer,
        "target_prefix": TARGET_PREFIX,
    }, path)


def run_training(adapter, source_extractor, target_extractor, source_layer: int, receiver_layer: int,
                  train_examples: list, holdout_examples: list, label_token_ids: dict,
                  num_steps: int, learning_rate: float, eval_every: int, checkpoint_path: Path,
                  seed: int = RNG_SEED) -> dict:
    """The loop itself: sample an example, score it, backward, step -
    repeat. Takes already-constructed extractors and an already-built
    adapter rather than building them itself, so it can run against
    lightweight fakes in a test without ever touching a real model
    download; train() below is the only place that does real model
    loading, and it just hands off to this function once that's done.
    Checkpoints only on a held-out accuracy improvement, not every eval,
    so CHECKPOINT_PATH always holds the best model seen, not merely the
    most recent one."""
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    rng = random.Random(seed)

    best_accuracy = 0.0
    last_loss = None
    for step in range(1, num_steps + 1):
        domain, passage = rng.choice(train_examples)

        optimizer.zero_grad()
        loss, _ = compute_loss_and_prediction(
            adapter, source_extractor, target_extractor, source_layer, receiver_layer, domain, passage, label_token_ids,
        )
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

        if step % eval_every == 0 or step == num_steps:
            accuracy = evaluate(
                adapter, source_extractor, target_extractor, source_layer, receiver_layer, holdout_examples, label_token_ids,
            )
            print(f"[TrainAdapter] step {step}/{num_steps}  loss={last_loss:.4f}  held-out accuracy={accuracy:.3f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(adapter, checkpoint_path, step, accuracy, source_layer, receiver_layer)
        else:
            print(f"[TrainAdapter] step {step}/{num_steps}  loss={last_loss:.4f}")

    return {"best_accuracy": best_accuracy, "final_loss": last_loss, "checkpoint_path": str(checkpoint_path)}


def train(num_steps: int = 200, learning_rate: float = 1e-4, eval_every: int = 25,
          checkpoint_path: Path = CHECKPOINT_PATH) -> dict:
    """Real-world entry point: loads the actual Qwen/Phi-4-mini extractors
    (both frozen - only the adapter between them ever gets a gradient),
    builds the warm-start rotation the same way build_warm_start's
    docstring describes, constructs the adapter from it, then hands
    everything to run_training. Kept thin and separate from run_training
    on purpose - this is the one function in the module a test can't call
    directly, since ResidualExtractor(...) downloads/loads a real model."""
    config = CROSS_MODEL_CONFIGS[TARGET_PREFIX]
    target_model_name = config["target_model"]
    source_layer, receiver_layer = config["param_grid"]["layer_pair"][0]

    print(f"[TrainAdapter] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)

    # Both base models stay completely frozen - only the adapter's own
    # parameters are ever handed to the optimizer.
    for p in source_extractor.model.parameters():
        p.requires_grad_(False)
    for p in target_extractor.model.parameters():
        p.requires_grad_(False)

    train_examples, holdout_examples = load_examples()
    label_token_ids = resolve_label_token_ids(target_extractor.tokenizer)

    print(f"[TrainAdapter] Building warm-start rotation from {CALIBRATION_SIZE} calibration concepts...")
    calibration_concepts = _load_calibration_concepts()
    scale_factor = compute_scale_factor(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)
    print(f"[TrainAdapter] Scale factor: {scale_factor:.3f}x")
    warm_start = build_warm_start(
        source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts, scale_factor,
    ).to(target_extractor.device)

    source_dim, target_dim = warm_start.shape
    adapter = TranslationAdapter(source_dim, target_dim, warm_start=warm_start).to(target_extractor.device)

    print(f"[TrainAdapter] Training for {num_steps} steps "
          f"({len(train_examples)} train / {len(holdout_examples)} held-out examples)...")
    result = run_training(
        adapter, source_extractor, target_extractor, source_layer, receiver_layer,
        train_examples, holdout_examples, label_token_ids,
        num_steps=num_steps, learning_rate=learning_rate, eval_every=eval_every, checkpoint_path=checkpoint_path,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a cross-model translation adapter (Qwen -> Phi-4-mini).")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    result = train(num_steps=args.num_steps, learning_rate=args.learning_rate, eval_every=args.eval_every)
    print(f"[TrainAdapter] Done. Best held-out accuracy: {result['best_accuracy']:.3f}  "
          f"checkpoint: {result['checkpoint_path']}")
