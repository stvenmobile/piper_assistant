"""
Piper Geometry: Reconstruction-Loss Trained Adapter.

train_adapter.py's closed-set classification loss (predict the domain
label token after a fixed suffix) turned out to be gameable: the trained
adapter reached 100% held-out accuracy by collapsing into a single-token
trigger ("physics physics physics...") rather than preserving anything
about the passage's actual content - see obsidian/Journals/2026-09-10.md
for the full evidence trail (a random-init control nearly matched it, and
free generation showed the collapse directly). This module replaces that
objective with one a single-token shortcut can't satisfy: teacher-forced
reconstruction of the ENTIRE passage, token by token, using nothing but
the injected representation as context - the same way every
encoder-decoder translation model is trained (encoder produces a
representation, decoder is teacher-forced against the real target during
training, free-generated at evaluation time).

Because this objective needs no domain label, it also removes the
dataset ceiling that made memorization the leading explanation for the
classification track's results - training and evaluation now draw from
the FULL concept pool (300 phrases, padding included), not just the ~80
primary-only phrases the label-based track was restricted to.

Both base models stay completely frozen, exactly as in train_adapter.py -
only the small bridging adapter is ever trained.
"""

import sys
import json
import random
import time
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F

from piper_geometry.extractor import ResidualExtractor
from piper_geometry.congruence_optimizer import CROSS_MODEL_CONFIGS, DEFAULT_MODEL
from piper_geometry.layer_injection import extract_full_sequence, PartialLayerInjectionHook, generate_with_injection
from piper_geometry.train_adapter import (
    TranslationAdapter, build_warm_start, compute_scale_factor, _load_calibration_concepts,
    TARGET_PREFIX, CALIBRATION_SIZE, RNG_SEED, CONCEPTS_PATH, DOMAIN_LABELS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = REPO_ROOT / "data" / "checkpoints" / "reconstruction_adapter.pt"


def load_reconstruction_examples(concepts_path: Path = CONCEPTS_PATH, holdout_fraction: float = 0.2,
                                  seed: int = RNG_SEED):
    """Unlike train_adapter.load_examples, this keeps EVERY phrase in
    every domain - including the "Advanced corollary"/"Empirical
    boundary condition" padding train_adapter.py excludes. That filter
    existed because those phrases made poor CLASSIFICATION examples
    (longer, less individually meaningful) - but reconstruction doesn't
    care about individual meaningfulness, only "can the receiving model
    reproduce these exact tokens," so the padding is perfectly usable
    training signal here. That roughly triples the available pool (~300
    phrases vs. ~80), directly working against the memorization risk the
    classification track ran into."""
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


def reconstruction_loss_and_accuracy(adapter, source_extractor, target_extractor, source_layer, receiver_layer,
                                      passage: str):
    """One example's worth of the reconstruction objective: translate the
    passage's full-sequence Qwen encoding through the adapter, inject it
    into Phi-4-mini, then require Phi-4-mini to predict the passage's OWN
    tokens - in ITS OWN tokenizer, not Qwen's; the injected span's length
    and the reconstruction target's length are independent, since the
    hook only overrides the leading len(injected_states) positions
    regardless of what comes after - via standard teacher-forced
    next-token cross-entropy, the same loss every encoder-decoder
    translation model trains against.

    Position accounting: with the injected span occupying positions
    0..inject_len-1, logits at position inject_len-1 predict the FIRST
    real passage token using nothing but the injected representation -
    the single most information-dense position in the whole loss, and
    exactly the thing a one-token classification objective never
    required. Each later position's prediction is easier (it also has
    the real preceding passage tokens to work from, standard teacher
    forcing), but the gradient from every position still flows back
    through the injected representation via attention, the same way it
    does in any encoder-decoder translation model.

    Returns (loss, teacher_forced_token_accuracy) - the accuracy is a
    cheap by-product (argmax vs. target at each position, still under
    teacher forcing) useful as a fast per-eval diagnostic; it is NOT the
    same thing as the free-generation token-overlap check in
    evaluate_reconstruction_by_generation, which is closer to what a
    human would actually judge as "did this reconstruct the passage."""
    with torch.no_grad():
        source_hidden = extract_full_sequence(
            source_extractor.model, source_extractor.tokenizer, source_extractor.device, passage, source_layer,
        ).to(torch.float32)

    device = next(adapter.parameters()).device
    injected_states = adapter(source_hidden.to(device))
    inject_len = injected_states.shape[0]

    target_ids = target_extractor.tokenizer(
        passage, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(target_extractor.device)
    passage_len = target_ids.shape[1]
    placeholder_id = target_extractor.tokenizer.eos_token_id
    placeholder_ids = torch.full((1, inject_len), placeholder_id, dtype=torch.long, device=target_extractor.device)
    input_ids = torch.cat([placeholder_ids, target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    hook = PartialLayerInjectionHook(injected_states.unsqueeze(0))
    handle = target_extractor.model.model.layers[receiver_layer].register_forward_hook(hook)
    try:
        # Deliberately no torch.no_grad() here - same as
        # train_adapter.forward_with_injection, this is the one call that
        # must stay differentiable end to end.
        outputs = target_extractor.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    # logits[inject_len-1] predicts target_ids[0] (pure injection, no real
    # tokens revealed yet); logits[inject_len-1+k] predicts target_ids[k]
    # (injection + k real preceding tokens, standard teacher forcing) -
    # passage_len positions total, lining up exactly with passage_len
    # targets, no separate shift needed since the slice already starts
    # one position early.
    prediction_logits = outputs.logits[0, inject_len - 1: inject_len - 1 + passage_len, :]
    targets = target_ids[0, :]

    loss = F.cross_entropy(prediction_logits.float(), targets)
    predicted_ids = prediction_logits.argmax(dim=-1)
    token_accuracy = (predicted_ids == targets).float().mean().item()
    return loss, token_accuracy


def evaluate_reconstruction(adapter, source_extractor, target_extractor, source_layer, receiver_layer,
                             examples: list) -> tuple:
    """Held-out reconstruction loss/accuracy under teacher forcing - fast
    (one forward pass per example, same cost as a training step), meant
    to run every eval interval. See evaluate_reconstruction_by_generation
    for the free-generation token-overlap check, a slower, more
    human-facing diagnostic run less often."""
    adapter.eval()
    total_loss = 0.0
    total_token_accuracy = 0.0
    with torch.no_grad():
        for _, passage in examples:
            loss, token_accuracy = reconstruction_loss_and_accuracy(
                adapter, source_extractor, target_extractor, source_layer, receiver_layer, passage,
            )
            total_loss += loss.item()
            total_token_accuracy += token_accuracy
    adapter.train()
    n = len(examples)
    return total_loss / n, total_token_accuracy / n


def count_matching_tokens(generated_text: str, target_passage: str, tokenizer) -> int:
    """Rough, order-independent overlap: how many tokens the
    free-generated continuation shares with the target passage's own
    tokenization, using a multiset intersection so repeating a common
    token doesn't inflate the count past how often it actually appears in
    the target. A quick sanity gauge, not a rigorous metric - common
    short words (the, of, and) can count toward this without reflecting
    real content overlap; that's an accepted tradeoff for a rough "are we
    in the ballpark" check, not something worth tightening unless it
    turns out to matter in practice."""
    generated_ids = tokenizer(generated_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_passage, add_special_tokens=False).input_ids
    overlap = Counter(generated_ids) & Counter(target_ids)
    return sum(overlap.values())


def evaluate_reconstruction_by_generation(adapter, source_extractor, target_extractor, source_layer, receiver_layer,
                                           examples: list, max_new_tokens: int = 30) -> dict:
    """The free-generation version of the reconstruction check: inject
    the translated representation with NO teacher forcing at all (same
    mechanism concept_transfer_test.py uses for its qualitative
    comparison) and see how many of the passage's own tokens show up in
    what the model actually generates on its own. Slower than
    evaluate_reconstruction (a full autoregressive generate() loop per
    example, not one forward pass), so this is meant to run occasionally
    (once at the end of training here), not every eval interval."""
    adapter.eval()
    match_counts = []
    with torch.no_grad():
        for _, passage in examples:
            source_hidden = extract_full_sequence(
                source_extractor.model, source_extractor.tokenizer, source_extractor.device, passage, source_layer,
            ).to(torch.float32)
            injected_states = adapter(source_hidden.to(next(adapter.parameters()).device))
            generated_text = generate_with_injection(
                target_extractor.model, target_extractor.tokenizer, target_extractor.device,
                receiver_layer, injected_states, max_new_tokens=max_new_tokens,
            )
            match_counts.append(count_matching_tokens(generated_text, passage, target_extractor.tokenizer))
    adapter.train()
    return {
        "mean_matching_tokens": sum(match_counts) / len(match_counts),
        "examples_at_or_above_5": sum(1 for c in match_counts if c >= 5),
        "num_examples": len(match_counts),
    }


def save_reconstruction_checkpoint(adapter, path: Path, step: int, held_out_loss: float,
                                    held_out_token_accuracy: float, source_layer: int, receiver_layer: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter.state_dict(),
        "step": step,
        "held_out_loss": held_out_loss,
        "held_out_token_accuracy": held_out_token_accuracy,
        "source_layer": source_layer,
        "receiver_layer": receiver_layer,
        "target_prefix": TARGET_PREFIX,
        "objective": "reconstruction",
    }, path)


def run_reconstruction_training(adapter, source_extractor, target_extractor, source_layer: int, receiver_layer: int,
                                 train_examples: list, holdout_examples: list,
                                 num_steps: int, learning_rate: float, eval_every: int, checkpoint_path: Path,
                                 seed: int = RNG_SEED, max_seconds: float = None) -> dict:
    """Same safety structure as train_adapter.run_training (AdamW +
    weight_decay against magnitude drift, gradient clipping, skip
    non-finite steps rather than let them poison Adam's moments, optional
    wall-clock budget) - only the metric direction changed: this tracks
    held-out LOSS, and "improvement" means lower, not higher, since the
    label-based accuracy metric doesn't exist in this objective. Runs the
    slower free-generation token-overlap check once at the end, on the
    final adapter state, rather than at every eval interval."""
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    start_time = time.monotonic()

    best_loss = float("inf")
    has_saved = False
    last_loss = None
    skipped_steps = 0
    stopped_early = False
    step = 0
    for step in range(1, num_steps + 1):
        if max_seconds is not None and (time.monotonic() - start_time) > max_seconds:
            print(f"[ReconAdapter] step {step}/{num_steps}  time budget of {max_seconds/60:.1f}min reached - stopping")
            stopped_early = True
            step -= 1
            break

        _, passage = rng.choice(train_examples)

        optimizer.zero_grad()
        loss, _ = reconstruction_loss_and_accuracy(
            adapter, source_extractor, target_extractor, source_layer, receiver_layer, passage,
        )

        if not torch.isfinite(loss):
            skipped_steps += 1
            print(f"[ReconAdapter] step {step}/{num_steps}  loss=nan/inf - skipping this step's update "
                  f"({skipped_steps} skipped so far)")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()

        if step % eval_every == 0 or step == num_steps:
            held_out_loss, held_out_token_accuracy = evaluate_reconstruction(
                adapter, source_extractor, target_extractor, source_layer, receiver_layer, holdout_examples,
            )
            print(f"[ReconAdapter] step {step}/{num_steps}  loss={last_loss:.4f}  "
                  f"held-out loss={held_out_loss:.4f}  held-out token accuracy={held_out_token_accuracy:.3f}")
            if held_out_loss < best_loss or not has_saved:
                best_loss = min(held_out_loss, best_loss)
                has_saved = True
                save_reconstruction_checkpoint(
                    adapter, checkpoint_path, step, held_out_loss, held_out_token_accuracy, source_layer, receiver_layer,
                )
        else:
            print(f"[ReconAdapter] step {step}/{num_steps}  loss={last_loss:.4f}")

    print("[ReconAdapter] Running free-generation token-overlap check on held-out examples...")
    generation_check = evaluate_reconstruction_by_generation(
        adapter, source_extractor, target_extractor, source_layer, receiver_layer, holdout_examples,
    )
    print(f"[ReconAdapter] Free-generation check: mean matching tokens={generation_check['mean_matching_tokens']:.2f}  "
          f"({generation_check['examples_at_or_above_5']}/{generation_check['num_examples']} examples >= 5 matching tokens)")

    return {
        "best_held_out_loss": best_loss,
        "final_loss": last_loss,
        "checkpoint_path": str(checkpoint_path),
        "skipped_steps": skipped_steps,
        "steps_completed": step,
        "stopped_early_time_budget": stopped_early,
        "elapsed_seconds": time.monotonic() - start_time,
        "generation_check": generation_check,
    }


def train_reconstruction(num_steps: int = 200, learning_rate: float = 1e-4, eval_every: int = 25,
                          checkpoint_path: Path = CHECKPOINT_PATH, max_hours: float = None,
                          use_warm_start: bool = True) -> dict:
    """Real-world entry point - mirrors train_adapter.train()'s structure
    exactly (real model loading, both frozen, optional warm-start), just
    handing off to run_reconstruction_training instead of run_training."""
    config = CROSS_MODEL_CONFIGS[TARGET_PREFIX]
    target_model_name = config["target_model"]
    source_layer, receiver_layer = config["param_grid"]["layer_pair"][0]

    print(f"[ReconAdapter] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)

    for p in source_extractor.model.parameters():
        p.requires_grad_(False)
    for p in target_extractor.model.parameters():
        p.requires_grad_(False)

    train_examples, holdout_examples = load_reconstruction_examples()

    if use_warm_start:
        print(f"[ReconAdapter] Building warm-start rotation from {CALIBRATION_SIZE} calibration concepts...")
        calibration_concepts = _load_calibration_concepts()
        scale_factor = compute_scale_factor(source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts)
        print(f"[ReconAdapter] Scale factor: {scale_factor:.3f}x")
        warm_start = build_warm_start(
            source_extractor, target_extractor, source_layer, receiver_layer, calibration_concepts, scale_factor,
        ).to(target_extractor.device)
        source_dim, target_dim = warm_start.shape
        adapter = TranslationAdapter(source_dim, target_dim, warm_start=warm_start).to(target_extractor.device)
    else:
        print("[ReconAdapter] RANDOM-INIT CONTROL: skipping the warm-start rotation entirely.")
        source_dim = source_extractor.model.config.hidden_size
        target_dim = target_extractor.model.config.hidden_size
        adapter = TranslationAdapter(source_dim, target_dim, warm_start=None).to(target_extractor.device)

    max_seconds = max_hours * 3600 if max_hours is not None else None
    budget_desc = (
        f"up to {num_steps} steps" if max_seconds is None
        else f"up to {num_steps} steps or {max_hours:.1f}h, whichever comes first"
    )
    print(f"[ReconAdapter] Training for {budget_desc} "
          f"({len(train_examples)} train / {len(holdout_examples)} held-out examples)...")
    result = run_reconstruction_training(
        adapter, source_extractor, target_extractor, source_layer, receiver_layer,
        train_examples, holdout_examples,
        num_steps=num_steps, learning_rate=learning_rate, eval_every=eval_every, checkpoint_path=checkpoint_path,
        max_seconds=max_seconds,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a reconstruction-loss cross-model translation adapter.")
    parser.add_argument("--num-steps", type=int, default=200, help="upper bound on steps regardless of --max-hours")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--max-hours", type=float, default=None,
                         help="stop after this many wall-clock hours even if --num-steps hasn't been reached")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                         help="default: data/checkpoints/reconstruction_adapter.pt")
    parser.add_argument("--no-warm-start", action="store_true",
                         help="random-init control: skip the warm-start rotation entirely (use a different "
                              "--checkpoint-path so this doesn't overwrite a real trained checkpoint)")
    args = parser.parse_args()

    result = train_reconstruction(
        num_steps=args.num_steps, learning_rate=args.learning_rate, eval_every=args.eval_every,
        max_hours=args.max_hours, use_warm_start=not args.no_warm_start,
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else CHECKPOINT_PATH,
    )
    print(f"[ReconAdapter] Done. Best held-out loss: {result['best_held_out_loss']:.4f}  "
          f"steps completed: {result['steps_completed']}/{args.num_steps}  "
          f"elapsed: {result['elapsed_seconds']/60:.1f}min  "
          f"checkpoint: {result['checkpoint_path']}")
