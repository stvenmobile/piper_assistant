"""
Piper Geometry: Post-hoc validation for a trained cross-model adapter.

train_adapter.py's own held-out split only measures generalization within
the same ~100-phrase pool the adapter trained on - it can't rule out the
adapter having found a shortcut specific to that small, reused set of
phrases (a real concern: the adapter is a 2.75M-parameter linear map
trained on ~80 examples for 20,000 steps, and reached 100% held-out
accuracy - see obsidian/Journals/2026-09-09.md). This module runs a saved
checkpoint against text it has never seen in ANY form during training or
evaluation: the auto-generated "Advanced corollary N in {domain}:
Analysis of..." / "Empirical boundary condition regarding..." padding
phrases train_adapter._primary_concepts excludes entirely from both the
training and held-out pools.
"""

import sys
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from piper_geometry.extractor import ResidualExtractor
from piper_geometry.congruence_optimizer import CROSS_MODEL_CONFIGS, DEFAULT_MODEL
from piper_geometry.train_adapter import (
    CONCEPTS_PATH, CHECKPOINT_PATH, DOMAIN_LABELS, TranslationAdapter, evaluate, resolve_label_token_ids,
)


def padding_concepts(phrases: list) -> list:
    """The complement of train_adapter._primary_concepts - only the
    auto-generated padding entries, never seen in training or the
    regular held-out split."""
    return [p for p in phrases if p.startswith(("Advanced corollary", "Empirical boundary condition"))]


def load_padding_examples(concepts_path: Path = CONCEPTS_PATH) -> list:
    """Every domain's padding-only phrases, flattened into (domain,
    phrase) pairs - genuinely novel text from the adapter's perspective,
    since load_examples() (used for both training and the regular
    held-out split) excludes all of it via the same domain filter."""
    with open(concepts_path, "r", encoding="utf-8") as f:
        by_domain = json.load(f)
    examples = []
    for domain, phrases in by_domain.items():
        if domain not in DOMAIN_LABELS:
            continue
        examples.extend((domain, phrase) for phrase in padding_concepts(phrases))
    return examples


def load_trained_adapter(checkpoint_path: Path, device) -> tuple:
    """Reconstructs a TranslationAdapter from a saved checkpoint, reading
    source_dim/target_dim directly off the saved weight's shape rather
    than requiring the caller to already know them - one less place for
    the two to silently drift apart."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    target_dim, source_dim = checkpoint["state_dict"]["linear.weight"].shape
    adapter = TranslationAdapter(source_dim, target_dim).to(device)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    return adapter, checkpoint


def run_padding_eval(checkpoint_path: Path = None) -> dict:
    """Real-world entry point - needs the actual Qwen/Phi-4-mini models
    the checkpoint was trained against, so (like train_adapter.train())
    this isn't directly unit-testable; load_padding_examples,
    padding_concepts, and load_trained_adapter above carry the tested
    logic. Model choice and layer pair come from the checkpoint's own
    saved metadata, not restated here, so this can't silently evaluate
    against the wrong model pairing."""
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_layer = raw["source_layer"]
    receiver_layer = raw["receiver_layer"]
    target_model_name = CROSS_MODEL_CONFIGS[raw["target_prefix"]]["target_model"]

    print(f"[PaddingEval] Checkpoint: {checkpoint_path}")
    print(f"[PaddingEval] {DEFAULT_MODEL} L{source_layer} -> {target_model_name} L{receiver_layer}")
    print(f"[PaddingEval] Checkpoint's own held-out accuracy: {raw['held_out_accuracy']:.3f}  "
          f"{raw.get('held_out_per_domain_accuracy', {})}")

    source_extractor = ResidualExtractor(model_name_or_path=DEFAULT_MODEL)
    target_extractor = ResidualExtractor(model_name_or_path=target_model_name)
    adapter, _ = load_trained_adapter(checkpoint_path, device=target_extractor.device)

    label_token_ids = resolve_label_token_ids(target_extractor.tokenizer)
    padding_examples = load_padding_examples()
    print(f"[PaddingEval] Evaluating on {len(padding_examples)} padding-only examples "
          f"(never seen in training or the regular held-out split)...")

    accuracy, per_domain_accuracy = evaluate(
        adapter, source_extractor, target_extractor, source_layer, receiver_layer, padding_examples, label_token_ids,
    )
    domain_summary = "  ".join(f"{d}={a:.2f}" for d, a in sorted(per_domain_accuracy.items()))
    print(f"[PaddingEval] Padding-set accuracy: {accuracy:.3f}  ({domain_summary})")
    print(f"[PaddingEval] Compare: checkpoint's own held-out accuracy was {raw['held_out_accuracy']:.3f}")

    return {
        "padding_accuracy": accuracy,
        "padding_per_domain_accuracy": per_domain_accuracy,
        "original_held_out_accuracy": raw["held_out_accuracy"],
        "num_padding_examples": len(padding_examples),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a trained adapter checkpoint against the excluded padding phrases."
    )
    parser.add_argument("--checkpoint-path", type=str, default=None,
                         help="default: data/checkpoints/trained_adapter.pt")
    args = parser.parse_args()

    run_padding_eval(Path(args.checkpoint_path) if args.checkpoint_path else None)
