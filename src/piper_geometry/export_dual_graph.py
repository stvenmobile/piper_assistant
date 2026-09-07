"""
Piper Geometry: Dual Concept-Graph Exporter for ai-graph.

Exports the same domain-organized concept set as seen through two
different (model, layer) activation extractions, plus within-each-set
cosine-similarity edges, as a single JSON file - ready for ai-graph's
ESP32 firmware to lay out and render side by side (one force-directed
graph per half of the screen), so differences in how two representations
organize the same concepts are visible directly rather than needing the
alignment rotation to interpret.

Defaults to the cross-layer framing already validated in tester.py
(Layer 8 vs Layer 14 of the same model), since that's the configuration
with a demonstrated non-trivial zero-shot signal. Pass a different
target_model_name to compare two genuinely separate models instead - the
graph-building step doesn't care which produced the embeddings.
"""

import sys
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F

from piper_geometry.extractor import ResidualExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONCEPTS_PATH = REPO_ROOT / "data" / "checkpoints" / "concepts_dictionary.json"
OUTPUT_PATH = REPO_ROOT / "data" / "dual_graph.json"

# Only the hand-written primary statements per domain are exported - the
# auto-generated "Advanced corollary"/"Empirical boundary condition"
# padding entries in the dictionary exist to bulk out Procrustes
# calibration data, not to be individually interesting graph nodes. Capped
# at 3/domain (18 nodes/side across the 6 domains) to match the node count
# ai-graph's single-graph mode already targets for legible rendering - here
# split across two half-width viewports instead of one full-width one.
MAX_CONCEPTS_PER_DOMAIN = 3
TOP_K_EDGES_PER_NODE = 4


def load_concepts(max_per_domain: int = MAX_CONCEPTS_PER_DOMAIN):
    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        domains = json.load(f)

    concepts = []  # list of (label, domain)
    for domain, phrases in domains.items():
        primary = [p for p in phrases if not p.startswith(("Advanced corollary", "Empirical boundary condition"))]
        for phrase in primary[:max_per_domain]:
            concepts.append((phrase, domain))
    return concepts


def extract_embeddings(extractor: ResidualExtractor, concepts, layer: int) -> torch.Tensor:
    vectors = []
    for label, _domain in concepts:
        acts = extractor.extract_activations(prompt=label, target_layers=[layer])
        vectors.append(acts[layer])
    return torch.stack(vectors)


def build_graph(concepts, embeddings: torch.Tensor, top_k: int = TOP_K_EDGES_PER_NODE) -> dict:
    n = len(concepts)
    sims = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

    nodes = [{"id": i, "label": label, "domain": domain} for i, (label, domain) in enumerate(concepts)]

    edges = []
    seen = set()
    for i in range(n):
        row = sims[i].clone()
        row[i] = -1.0  # exclude self-similarity
        top = torch.topk(row, min(top_k, n - 1)).indices.tolist()
        for j in top:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": key[0], "b": key[1], "weight": round(sims[i, j].item(), 4)})

    return {"nodes": nodes, "edges": edges}


def export(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    source_layer: int = 8,
    target_model_name: str = None,
    target_layer: int = 14,
) -> dict:
    target_model_name = target_model_name or model_name
    concepts = load_concepts()
    print(f"[export_dual_graph] Loaded {len(concepts)} concepts across domains.")

    print(f"[export_dual_graph] Extracting modelA ({model_name}, layer {source_layer})...")
    extractor_a = ResidualExtractor(model_name_or_path=model_name)
    embeddings_a = extract_embeddings(extractor_a, concepts, source_layer)

    if target_model_name == model_name:
        extractor_b = extractor_a
    else:
        print(f"[export_dual_graph] Extracting modelB ({target_model_name}, layer {target_layer})...")
        extractor_b = ResidualExtractor(model_name_or_path=target_model_name)
    embeddings_b = extract_embeddings(extractor_b, concepts, target_layer)

    result = {
        "modelA": {"label": f"{model_name} L{source_layer}", **build_graph(concepts, embeddings_a)},
        "modelB": {"label": f"{target_model_name} L{target_layer}", **build_graph(concepts, embeddings_b)},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[export_dual_graph] Wrote {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--source-layer", type=int, default=8)
    parser.add_argument("--target-model", dest="target_model_name", default=None,
                         help="Defaults to --model (same-model cross-layer comparison).")
    parser.add_argument("--target-layer", type=int, default=14)
    args = parser.parse_args()

    export(
        model_name=args.model_name,
        source_layer=args.source_layer,
        target_model_name=args.target_model_name,
        target_layer=args.target_layer,
    )
