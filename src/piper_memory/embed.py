"""
Embeddings for Piper's memory: text -> unit-length float32 vectors, so cosine similarity is a
plain dot product.

OllamaEmbedder uses Ollama's /api/embed with nomic-embed-text on the PC. nomic models expect a
task prefix; "clustering: " is used for everything (symmetric similarity between stored items),
which separated same / related / unrelated best when calibrated (2026-10-07):
    same concept (tide~tides, CO2~carbon dioxide)    0.90 - 0.97
    related concepts (Moon~Sun, gravity~magnetism)    0.71 - 0.83
    same finding reworded 0.94; same-field 0.85; cross-topic analogy 0.69; unrelated 0.60
"""
import json
import urllib.request

import numpy as np


class EmbedError(RuntimeError):
    pass


def normalise(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str = "nomic-embed-text", prefix: str = "clustering: ",
                 timeout: float = 60.0):
        self.url = base_url.rstrip("/") + "/api/embed"
        self.model = model
        self.prefix = prefix
        self.timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        """One unit vector per text (rows)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        body = json.dumps({"model": self.model, "input": [self.prefix + t for t in texts]}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except Exception as e:                      # network, HTTP or JSON
            raise EmbedError(f"embedding via {self.url} failed: {e}") from e
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise EmbedError(f"embedding via {self.url}: unexpected reply {str(data)[:200]}")
        return normalise(vectors)
