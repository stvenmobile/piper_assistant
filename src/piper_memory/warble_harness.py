"""
Piper Memory: End-to-end sanity-check harness for the warble/quaddle
experiment.

This is a PLUMBING sanity check, not a test of genuine learning. The
"study" step here is in-context only (study facts placed directly in the
prompt before each probe) - already identified in
obsidian/Journals/2026-09-10.md as a near-trivial signal on its own: if
the facts are visible in context at measurement time, loss improving
mostly reflects ordinary reading comprehension, not anything persisted
in the model. What this DOES meaningfully verify: MemoryStore
population, real loss measurement against a real model, persistence
round-tripping, and that the quaddle control stays completely isolated
throughout - its own facts are never touched, only warbles' context gets
prepended to its probes, to check whether irrelevant context generically
helps prediction (it shouldn't, for facts it doesn't answer).

Genuine learning requires the LoRA path (not yet built).

Scoped down for this first pass: only SmallModelEmbedder populates
MemoryStore here, to avoid loading two separate copies of the same base
model (one via QwenHiddenStateEmbedder's own ResidualExtractor, one as
the model actually being measured). Adding the second embedding method
back in is a natural follow-up once this core plumbing is proven.
"""

import sys
import tempfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformers import AutoTokenizer, AutoModelForCausalLM

from piper_memory.memory_store import MemoryStore
from piper_memory.fictional_entities import load_fictional_entities, get_study_facts, get_recall_probes
from piper_memory.embeddings import SmallModelEmbedder, populate_item
from piper_memory.loss_measurement import evaluate_recall_probes


def with_context_probes(probes: list, context_text: str) -> list:
    """Builds a modified probe list with study content prepended to each
    prompt - the in-context "study" step. Deliberately doesn't touch
    evaluate_recall_probes/compute_probe_loss at all; those stay exactly
    as already tested, this just constructs a different input for them."""
    return [{"prompt": f"{context_text} {p['prompt']}", "target": p["target"]} for p in probes]


def run_sanity_check(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu") -> dict:
    print(f"[Harness] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    entities = load_fictional_entities()
    warble_facts = get_study_facts(entities, "warbles")
    warble_context = " ".join(warble_facts)
    warble_probes = get_recall_probes(entities, "warbles")
    quaddle_probes = get_recall_probes(entities, "quaddles")  # control - facts NEVER touched

    print("[Harness] Populating MemoryStore with warbles (small_model embeddings only, this pass)...")
    store = MemoryStore()
    small_embedder = SmallModelEmbedder()
    populate_item(store, "warbles", warble_context, {small_embedder.METHOD_NAME: small_embedder})

    print("[Harness] Measuring BASELINE loss (before study) on warbles and quaddles...")
    warble_loss_before, warble_acc_before = evaluate_recall_probes(model, tokenizer, device, warble_probes)
    quaddle_loss_before, quaddle_acc_before = evaluate_recall_probes(model, tokenizer, device, quaddle_probes)
    store.record_loss("warbles", warble_loss_before)

    print("[Harness] 'Studying' warbles (in-context - see module docstring for why this is a "
          "sanity check, not a learning test)...")
    warble_probes_with_context = with_context_probes(warble_probes, warble_context)
    # Quaddle probes get WARBLE context prepended, never their own facts -
    # tests whether irrelevant context generically helps prediction.
    quaddle_probes_with_warble_context = with_context_probes(quaddle_probes, warble_context)

    print("[Harness] Measuring loss AFTER study...")
    warble_loss_after, warble_acc_after = evaluate_recall_probes(model, tokenizer, device, warble_probes_with_context)
    quaddle_loss_after, quaddle_acc_after = evaluate_recall_probes(model, tokenizer, device, quaddle_probes_with_warble_context)
    store.record_loss("warbles", warble_loss_after)

    progress = store.learning_progress("warbles")

    print(f"\n[Harness] warbles   loss: {warble_loss_before:.4f} -> {warble_loss_after:.4f}  "
          f"(delta {progress:+.4f})   accuracy: {warble_acc_before:.3f} -> {warble_acc_after:.3f}")
    print(f"[Harness] quaddles  loss: {quaddle_loss_before:.4f} -> {quaddle_loss_after:.4f}  "
          f"(delta {quaddle_loss_before - quaddle_loss_after:+.4f})   "
          f"accuracy: {quaddle_acc_before:.3f} -> {quaddle_acc_after:.3f}")

    print("\n[Harness] Checking persistence round-trip (save/reload)...")
    scratch = Path(tempfile.mkdtemp()) / "sanity_check_memory"
    store.save(scratch)
    reloaded = MemoryStore.load(scratch)
    assert reloaded.learning_progress("warbles") == progress, "learning_progress must survive save/load"
    print("[Harness] Persistence check OK - learning_progress survives a save/load round-trip.")

    return {
        "warble_loss_before": warble_loss_before, "warble_loss_after": warble_loss_after,
        "quaddle_loss_before": quaddle_loss_before, "quaddle_loss_after": quaddle_loss_after,
        "warble_learning_progress": progress,
    }


if __name__ == "__main__":
    run_sanity_check()
