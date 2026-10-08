"""
piper_memory - Piper's memory of her research: an SQLite store (the source of truth) with
embedding-based similarity, provenance for every finding and relation, and an activity log.
See schema.py for the structure and store.py for the interface.
"""
from piper_memory.embed import EmbedError, OllamaEmbedder
from piper_memory.schema import (CERTAINTY, FINDING_KINDS, FINDING_STATUS, RELATIONS, RELEVANCE, SOURCE_KINDS,
                                 STANCES, TOPIC_STATUS)
from piper_memory.store import MemoryStore


def open_memory(readonly: bool = False, path=None) -> MemoryStore:
    """The memory store as configured (config.yaml `memory` section), embedding via Ollama.
    path: another database file (e.g. a scratch copy for trying things out)."""
    from piper_brain.config import CONFIG, ROOT_DIR
    cfg = CONFIG["memory"]
    path = path or ROOT_DIR / cfg["path"]
    embedder = OllamaEmbedder(cfg["embed_url"] or CONFIG["llm"]["base_url"], cfg["embed_model"])
    return MemoryStore(path, embedder, readonly=readonly, concept_merge=cfg["concept_merge"],
                       finding_duplicate=cfg["finding_duplicate"], related=cfg["related"],
                       cross_topic=cfg["cross_topic"])


__all__ = ["MemoryStore", "OllamaEmbedder", "EmbedError", "open_memory", "RELATIONS", "FINDING_KINDS",
           "FINDING_STATUS", "SOURCE_KINDS", "TOPIC_STATUS"]
