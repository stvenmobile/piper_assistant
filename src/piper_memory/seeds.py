"""
Seed topics: the starting points of Piper's research, from a YAML file (research_seeds.yaml at
the repo root - git-ignored, since it holds personal views; research_seeds.example.yaml shows
the format).

    topics:
      - name: Animal tool use
        thesis: >                          # optional: a position to examine, not to confirm
          Only primates truly make tools ...
        parent: Some other topic           # optional: by name, listed earlier
        origin: seed                       # seed (default) | calibration
        angles:                            # starter questions -> open 'question' findings
          - Which birds make tools, and how do they learn it?

    python3 -m piper_memory.seeds [path]   (from src/) - load into the configured memory

Re-loading is safe: topics are matched by name (their thesis/description/parent updated to
the file), and an angle already present on a topic isn't added twice.
"""
import sys
from pathlib import Path

import yaml


def load_seeds(mem, path) -> dict:
    """Load seed topics into MemoryStore `mem`. Returns {"topics": n, "questions_added": n}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    source = mem.source("user", "research seeds")
    added = 0
    topics = data.get("topics") or []
    for t in topics:
        name = t["name"].strip()
        parent = mem.topic(t["parent"]) if t.get("parent") else None
        thesis = " ".join((t.get("thesis") or "").split())
        description = " ".join((t.get("description") or "").split())
        tid = mem.topic(name, parent=parent, description=description, origin=t.get("origin", "seed"),
                        thesis=thesis)
        mem.update_topic(tid, thesis=thesis, description=description, parent_id=parent)
        existing = {f["text"] for f in mem.findings(topic_id=tid, limit=10_000) if f["kind"] == "question"}
        for angle in t.get("angles") or []:
            q = " ".join(angle.split())
            if q not in existing:
                mem.add_finding(q, tid, kind="question", confidence=1.0, sources=[source])
                existing.add(q)
                added += 1
    if topics:
        mem.log("seeds_loaded", topics=len(topics), questions_added=added, path=str(path))
    return {"topics": len(topics), "questions_added": added}


if __name__ == "__main__":
    from piper_brain.config import ROOT_DIR
    from piper_memory import open_memory
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / "research_seeds.yaml"
    m = open_memory()
    print(load_seeds(m, path), m.stats())
