"""
MemoryStore - Piper's memory: an SQLite file (the source of truth) plus in-RAM embedding
indexes for similarity search (brute-force cosine: a matrix multiply, instant at this scale).

One writer (the research loop) and any number of readers (the dashboard, the vault exporter,
conversation) - readers open it with readonly=True; SQLite's WAL mode lets them read while the
writer writes.

Without an embedder (or when the embedder is unreachable) everything still works except the
similarity features; items are stored unembedded and backfill_embeddings() catches up later.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from piper_memory.embed import EmbedError, normalise
from piper_memory.schema import (DDL, FINDING_KINDS, FINDING_STATUS, MIGRATIONS, RELATIONS,
                                 SCHEMA_VERSION, SOURCE_KINDS, STANCES, TOPIC_STATUS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def concept_key(name: str) -> str:
    """A concept name's identity for automatic merging: lower case, no punctuation, no leading
    article, each word made singular. 'The Tides' / 'tide' -> 'tide'. Deliberately literal -
    names that differ in a real word ('predator population' / 'prey population') stay apart."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in name.lower()).split()]
    if len(words) > 1 and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(_singular(w) for w in words)


def _blob(v: np.ndarray | None):
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def _vec(b) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


class VectorIndex:
    """Ids + a matrix of unit vectors (rows), searched by dot product."""

    def __init__(self):
        self.ids: list[int] = []
        self.matrix: np.ndarray | None = None

    def add(self, item_id: int, vector: np.ndarray):
        v = normalise(vector)
        self.matrix = v if self.matrix is None else np.vstack([self.matrix, v])
        self.ids.append(item_id)

    def search(self, vector: np.ndarray, k: int = 5, min_score: float = -1.0,
               allow=None) -> list[tuple[int, float]]:
        """Best (id, score) pairs, highest first; `allow(id)` filters candidates."""
        if self.matrix is None or not self.ids:
            return []
        scores = self.matrix @ normalise(vector)[0]
        out = []
        for i in np.argsort(-scores):
            s = float(scores[i])
            if s < min_score:
                break
            if allow is None or allow(self.ids[i]):
                out.append((self.ids[i], s))
                if len(out) >= k:
                    break
        return out

    def __len__(self):
        return len(self.ids)


class MemoryStore:
    def __init__(self, path: str | Path, embedder=None, readonly: bool = False,
                 concept_merge: float = 0.90, finding_duplicate: float = 0.92,
                 related: float = 0.80, cross_topic: float = 0.68):
        self.path = Path(path)
        self.embedder = embedder
        self.readonly = readonly
        self.concept_merge = concept_merge          # concepts this similar are put to the judge
        self.finding_duplicate = finding_duplicate  # findings this similar say the same thing
        self.related = related                      # findings this similar are closely related
        self.cross_topic = cross_topic              # cross-topic pairs above this are link candidates
        self.lock = threading.RLock()
        if readonly:
            self.db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            self._migrate()
        self.findings_index = VectorIndex()
        self.concepts_index = VectorIndex()
        self.aliases: dict[str, int] = {}           # concept_key(name or alias) -> concept id
        self._load_indexes()

    # ---- setup ------------------------------------------------------------------------------
    def _migrate(self):
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} is schema v{version}; this code knows v{SCHEMA_VERSION}")
        with self.db:
            if version > 0:                         # an older store: upgrade it step by step
                for v in range(version + 1, SCHEMA_VERSION + 1):
                    for sql in MIGRATIONS.get(v, []):
                        self.db.execute(sql)
            self.db.executescript(DDL)              # new store: everything; old: any new tables
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _load_indexes(self):
        for row in self.db.execute("SELECT id, embedding FROM findings WHERE embedding IS NOT NULL ORDER BY id"):
            self.findings_index.add(row["id"], _vec(row["embedding"]))
        self._load_concepts()

    def _load_concepts(self):
        self.concepts_index = VectorIndex()
        self.aliases = {}
        for row in self.db.execute("SELECT id, name, aliases, embedding FROM concepts ORDER BY id"):
            self.aliases[concept_key(row["name"])] = row["id"]
            for a in json.loads(row["aliases"]):
                self.aliases[concept_key(a)] = row["id"]
            if row["embedding"] is not None:
                self.concepts_index.add(row["id"], _vec(row["embedding"]))

    def close(self):
        self.db.close()

    def _write(self):
        if self.readonly:
            raise PermissionError("this MemoryStore was opened read-only")
        return self.db                              # use as `with self._write():` - a transaction

    def _embed(self, text: str) -> np.ndarray | None:
        """A unit vector for `text`, or None if there is no (working) embedder."""
        if self.embedder is None:
            return None
        try:
            return self.embedder.embed([text])[0]
        except EmbedError:
            return None

    # ---- events (the live activity log) ------------------------------------------------------
    def log(self, kind: str, **data) -> int:
        with self.lock, self._write():
            return self.db.execute("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
                                   (now(), kind, json.dumps(data))).lastrowid

    def events(self, since_id: int = 0, limit: int = 200) -> list[dict]:
        rows = self.db.execute("SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?", (since_id, limit))
        return [{"id": r["id"], "ts": r["ts"], "kind": r["kind"], **json.loads(r["data"])} for r in rows]

    # ---- topics -----------------------------------------------------------------------------
    def topic(self, name: str, parent: int | None = None, description: str = "", origin: str = "seed",
              thesis: str = "") -> int:
        """The id of topic `name` (case-insensitive), creating it if needed. thesis: a position
        to examine - findings then record their stance toward it."""
        row = self.db.execute("SELECT id FROM topics WHERE name = ?", (name.strip(),)).fetchone()
        if row:
            return row["id"]
        with self.lock, self._write():
            t = now()
            return self.db.execute(
                "INSERT INTO topics (name, parent_id, description, thesis, origin, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name.strip(), parent, description, thesis, origin, t, t)).lastrowid

    def get_topic(self, topic_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return dict(row) if row else None

    def topics(self, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM topics", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY id", args)]

    def update_topic(self, topic_id: int, **fields):
        """Set status / description / thesis / novelty / progress / saturation / parent_id."""
        allowed = {"status", "description", "thesis", "novelty", "progress", "saturation", "parent_id"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"can't update topic fields {bad}")
        if "status" in fields and fields["status"] not in TOPIC_STATUS:
            raise ValueError(f"topic status must be one of {TOPIC_STATUS}")
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.lock, self._write():
            self.db.execute(f"UPDATE topics SET {sets}, updated_at = ? WHERE id = ?",
                            (*fields.values(), now(), topic_id))

    # ---- episodes (the research diary) -------------------------------------------------------
    def start_episode(self, topic_id: int, angle: str = "", model: str = "") -> int:
        with self.lock, self._write():
            return self.db.execute("INSERT INTO episodes (topic_id, angle, model, started_at) VALUES (?, ?, ?, ?)",
                                   (topic_id, angle, model, now())).lastrowid

    def end_episode(self, episode_id: int, outcome: str, summary: str = "", **stats):
        with self.lock, self._write():
            self.db.execute("UPDATE episodes SET ended_at = ?, outcome = ?, summary = ?, stats = ? WHERE id = ?",
                            (now(), outcome, summary, json.dumps(stats), episode_id))

    def episodes(self, topic_id: int | None = None, limit: int = 50) -> list[dict]:
        sql, args = "SELECT * FROM episodes", []
        if topic_id is not None:
            sql, args = sql + " WHERE topic_id = ?", [topic_id]
        rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))
        return [{**dict(r), "stats": json.loads(r["stats"])} for r in rows]

    # ---- sources -----------------------------------------------------------------------------
    def source(self, kind: str, title: str = "", url: str | None = None, **meta) -> int:
        """The id of a source, de-duplicated by url (or kind + title when there's no url)."""
        if kind not in SOURCE_KINDS:
            raise ValueError(f"source kind must be one of {SOURCE_KINDS}")
        ref = url or f"{kind}:{title}"
        row = self.db.execute("SELECT id FROM sources WHERE ref = ?", (ref,)).fetchone()
        if row:
            return row["id"]
        with self.lock, self._write():
            return self.db.execute(
                "INSERT INTO sources (kind, title, url, ref, retrieved_at, meta) VALUES (?, ?, ?, ?, ?, ?)",
                (kind, title, url, ref, now(), json.dumps(meta))).lastrowid

    # ---- findings ---------------------------------------------------------------------------
    def add_finding(self, text: str, topic_id: int, episode_id: int | None = None, kind: str = "fact",
                    confidence: float = 0.5, status: str = "new", relates_to: int | None = None,
                    sources: list | None = None, tags: list[str] | None = None, stance: str = "neutral") -> int:
        """Store a finding. sources: [source_id, ...] or [(source_id, excerpt), ...].
        stance: toward the topic's thesis - supports | challenges | neutral."""
        if kind not in FINDING_KINDS:
            raise ValueError(f"finding kind must be one of {FINDING_KINDS}")
        if status not in FINDING_STATUS:
            raise ValueError(f"finding status must be one of {FINDING_STATUS}")
        if stance not in STANCES:
            raise ValueError(f"finding stance must be one of {STANCES}")
        if status != "new" and relates_to is None:
            raise ValueError(f"a '{status}' finding needs relates_to (the finding it {status})")
        vector = self._embed(text)
        with self.lock, self._write():
            fid = self.db.execute(
                "INSERT INTO findings (text, kind, topic_id, episode_id, confidence, status, stance, relates_to,"
                " created_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (text, kind, topic_id, episode_id, confidence, status, stance, relates_to, now(),
                 _blob(vector))).lastrowid
            for s in sources or []:
                sid, excerpt = (s, "") if isinstance(s, int) else s
                self.db.execute("INSERT OR IGNORE INTO finding_sources (finding_id, source_id, excerpt) VALUES (?, ?, ?)",
                                (fid, sid, excerpt))
            for tag in tags or []:
                self._tag(fid, tag)
            if vector is not None:
                self.findings_index.add(fid, vector)
        return fid

    def add_finding_source(self, finding_id: int, source_id: int, excerpt: str = "") -> bool:
        """Another source for an existing finding (a second page saying the same thing).
        Returns False if that source was already recorded for it."""
        with self.lock, self._write():
            return self.db.execute("INSERT OR IGNORE INTO finding_sources (finding_id, source_id, excerpt)"
                                   " VALUES (?, ?, ?)", (finding_id, source_id, excerpt)).rowcount > 0

    def finding_source_ids(self, finding_id: int) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT source_id FROM finding_sources WHERE finding_id = ?",
                                              (finding_id,))}

    def _tag(self, finding_id: int, tag: str):
        self.db.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag.strip(),))
        tid = self.db.execute("SELECT id FROM tags WHERE name = ?", (tag.strip(),)).fetchone()["id"]
        self.db.execute("INSERT OR IGNORE INTO finding_tags (finding_id, tag_id) VALUES (?, ?)", (finding_id, tid))

    def get_finding(self, finding_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        if not row:
            return None
        f = {k: row[k] for k in row.keys() if k != "embedding"}
        f["tags"] = [r["name"] for r in self.db.execute(
            "SELECT t.name FROM tags t JOIN finding_tags ft ON ft.tag_id = t.id WHERE ft.finding_id = ? ORDER BY t.name",
            (finding_id,))]
        return f

    def findings(self, topic_id: int | None = None, episode_id: int | None = None, limit: int = 100) -> list[dict]:
        sql, where, args = "SELECT id FROM findings", [], []
        if topic_id is not None:
            where.append("topic_id = ?"); args.append(topic_id)
        if episode_id is not None:
            where.append("episode_id = ?"); args.append(episode_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))
        return [self.get_finding(r["id"]) for r in rows]

    def stance_balance(self, topic_id: int) -> dict:
        """How the evidence on a topic stands toward its thesis: counts and confidence-weighted
        totals of supporting / challenging / neutral findings (questions excluded)."""
        out = {s: {"count": 0, "weight": 0.0} for s in STANCES}
        for r in self.db.execute("SELECT stance, COUNT(*) AS n, SUM(confidence) AS w FROM findings"
                                 " WHERE topic_id = ? AND kind != 'question' GROUP BY stance", (topic_id,)):
            out[r["stance"]] = {"count": r["n"], "weight": round(r["w"] or 0.0, 3)}
        return out

    def provenance(self, finding_id: int) -> dict:
        """Where a finding came from: its sources (+ excerpts), episode, topic, and what it relates to."""
        f = self.get_finding(finding_id)
        if f is None:
            raise KeyError(finding_id)
        sources = [dict(r) for r in self.db.execute(
            "SELECT s.id, s.kind, s.title, s.url, s.retrieved_at, fs.excerpt FROM sources s"
            " JOIN finding_sources fs ON fs.source_id = s.id WHERE fs.finding_id = ? ORDER BY s.id", (finding_id,))]
        episode = self.db.execute("SELECT * FROM episodes WHERE id = ?", (f["episode_id"],)).fetchone()
        return {
            "finding": f,
            "sources": sources,
            "episode": dict(episode) if episode else None,
            "topic": self.get_topic(f["topic_id"]),
            "relates_to": self.get_finding(f["relates_to"]) if f["relates_to"] else None,
        }

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over findings (SQLite FTS5 syntax: words, "phrases", OR, prefix*)."""
        rows = self.db.execute("SELECT rowid FROM findings_fts WHERE findings_fts MATCH ? ORDER BY rank LIMIT ?",
                               (query, limit))
        return [self.get_finding(r["rowid"]) for r in rows]

    # ---- similarity ---------------------------------------------------------------------------
    def similar(self, text_or_vector, k: int = 5, min_score: float = 0.0, topic_id: int | None = None,
                other_topics_than: int | None = None, exclude: tuple = (), questions: bool | None = None) -> list[dict]:
        """Findings most similar to a text (or a vector): [{...finding, "score"}], best first.
        questions: None = any finding, False = no open questions, True = only questions."""
        vector = self._embed(text_or_vector) if isinstance(text_or_vector, str) else text_or_vector
        if vector is None:
            return []
        topic_of, is_q = {}, {}
        if topic_id is not None or other_topics_than is not None or questions is not None:
            for r in self.db.execute("SELECT id, topic_id, kind FROM findings"):
                topic_of[r["id"]] = r["topic_id"]
                is_q[r["id"]] = r["kind"] == "question"

        def allow(fid):
            if fid in exclude:
                return False
            if questions is not None and is_q.get(fid) != questions:
                return False
            if topic_id is not None and topic_of.get(fid) != topic_id:
                return False
            if other_topics_than is not None and topic_of.get(fid) == other_topics_than:
                return False
            return True

        return [{**self.get_finding(fid), "score": s}
                for fid, s in self.findings_index.search(vector, k, min_score, allow)]

    def duplicate_of(self, text: str, topic_id: int | None = None, questions: bool | None = None) -> dict | None:
        """An existing finding that says the same thing (similarity >= finding_duplicate), if any.
        questions: as for similar() - a claim is compared with claims, a question with questions."""
        hits = self.similar(text, k=1, min_score=self.finding_duplicate, topic_id=topic_id, questions=questions)
        return hits[0] if hits else None

    def finding_vector(self, finding_id: int) -> np.ndarray | None:
        row = self.db.execute("SELECT embedding FROM findings WHERE id = ?", (finding_id,)).fetchone()
        return _vec(row["embedding"]) if row else None

    def cross_topic_candidates(self, finding_id: int, k: int = 5, min_score: float | None = None) -> list[dict]:
        """Findings in OTHER topics that are semantically close to this one but share no concept
        with it in the graph - candidate surprising links (for the LLM to judge, then flag).
        Open questions are left out on both sides: a link is between things learned."""
        vector = self.finding_vector(finding_id)
        f = self.get_finding(finding_id)
        if vector is None or f is None:
            return []
        mine = self.finding_concepts(finding_id)
        hits = self.similar(vector, k=k * 3, min_score=self.cross_topic if min_score is None else min_score,
                            other_topics_than=f["topic_id"], exclude=(finding_id,), questions=False)
        return [h for h in hits if not (mine & self.finding_concepts(h["id"]))][:k]

    # ---- concepts + relations (the knowledge graph) -------------------------------------------
    def concept(self, name: str, judge=None) -> int:
        """The id of a concept named `name`:
        1. an existing concept whose name or alias has the same concept_key (tides = the Tide);
        2. else, if `judge` is given, an existing concept at least concept_merge similar that
           judge(new_name, existing_name) confirms is the same thing (CO2 = carbon dioxide) -
           the new name becomes its alias;
        3. else a new concept.
        Without a judge nothing merges on similarity alone: names sharing most words embed as
        near-identical even when they're opposites ('predator population' ~ 'prey population'
        0.95), and a wrong merge corrupts the graph, while a duplicate can be merged later."""
        name = name.strip()
        cid = self.aliases.get(concept_key(name))
        if cid is not None:
            if name.lower() != self.get_concept(cid)["name"].lower():
                self._add_alias(cid, name)
            return cid
        vector = self._embed(name)
        if vector is not None and judge is not None:
            for cid, _ in self.concepts_index.search(vector, k=3, min_score=self.concept_merge):
                if judge(name, self.get_concept(cid)["name"]):
                    self._add_alias(cid, name)
                    return cid
        with self.lock, self._write():
            cid = self.db.execute("INSERT INTO concepts (name, created_at, embedding) VALUES (?, ?, ?)",
                                  (name, now(), _blob(vector))).lastrowid
        self.aliases[concept_key(name)] = cid
        if vector is not None:
            self.concepts_index.add(cid, vector)
        return cid

    def _add_alias(self, concept_id: int, name: str):
        with self.lock, self._write():
            row = self.db.execute("SELECT aliases FROM concepts WHERE id = ?", (concept_id,)).fetchone()
            aliases = json.loads(row["aliases"])
            if name not in aliases:
                self.db.execute("UPDATE concepts SET aliases = ? WHERE id = ?",
                                (json.dumps(aliases + [name]), concept_id))
        self.aliases[concept_key(name)] = concept_id

    def similar_concepts(self, name: str, k: int = 5, min_score: float | None = None) -> list[dict]:
        """OTHER concepts whose names embed close to `name` (not `name`'s own concept) - merge
        candidates for review or a judge."""
        vector = self._embed(name)
        if vector is None:
            return []
        own = self.aliases.get(concept_key(name))
        hits = self.concepts_index.search(vector, k, self.concept_merge if min_score is None else min_score,
                                          allow=lambda cid: cid != own)
        return [{**self.get_concept(cid), "score": s} for cid, s in hits]

    def merge_concepts(self, keep: int, drop: int):
        """Fold concept `drop` into `keep`: its relations are repointed, its name and aliases
        become aliases of `keep`, and it is deleted."""
        if keep == drop:
            return
        k, d = self.get_concept(keep), self.get_concept(drop)
        if not k or not d:
            raise KeyError(drop if k else keep)
        with self.lock, self._write():
            for col in ("subject_id", "object_id"):
                for r in self.db.execute(f"SELECT * FROM relations WHERE {col} = ?", (drop,)).fetchall():
                    s = keep if r["subject_id"] == drop else r["subject_id"]
                    o = keep if r["object_id"] == drop else r["object_id"]
                    self.db.execute("INSERT OR IGNORE INTO relations (subject_id, predicate, object_id, finding_id,"
                                    " confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                                    (s, r["predicate"], o, r["finding_id"], r["confidence"], r["created_at"]))
                    self.db.execute("DELETE FROM relations WHERE id = ?", (r["id"],))
            aliases = k["aliases"] + [a for a in [d["name"]] + d["aliases"] if a not in k["aliases"]]
            self.db.execute("UPDATE concepts SET aliases = ? WHERE id = ?", (json.dumps(aliases), keep))
            self.db.execute("DELETE FROM concepts WHERE id = ?", (drop,))
        self._load_concepts()

    def get_concept(self, concept_id: int) -> dict | None:
        row = self.db.execute("SELECT id, name, aliases, created_at FROM concepts WHERE id = ?", (concept_id,)).fetchone()
        return {**dict(row), "aliases": json.loads(row["aliases"])} if row else None

    def relate(self, subject, predicate: str, obj, finding_id: int | None = None, confidence: float = 0.5) -> int:
        """subject -predicate-> obj (concept names or ids), recorded with the finding it came from."""
        if predicate not in RELATIONS:
            raise ValueError(f"predicate {predicate!r} is not one of {RELATIONS}")
        s = subject if isinstance(subject, int) else self.concept(subject)
        o = obj if isinstance(obj, int) else self.concept(obj)
        with self.lock, self._write():
            cur = self.db.execute(
                "INSERT OR IGNORE INTO relations (subject_id, predicate, object_id, finding_id, confidence, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)", (s, predicate, o, finding_id, confidence, now()))
            if cur.lastrowid and cur.rowcount:
                return cur.lastrowid
            return self.db.execute("SELECT id FROM relations WHERE subject_id = ? AND predicate = ? AND object_id = ?"
                                   " AND finding_id IS ?", (s, predicate, o, finding_id)).fetchone()["id"]

    def relations(self, concept_id: int | None = None, finding_id: int | None = None) -> list[dict]:
        sql = ("SELECT r.*, cs.name AS subject, co.name AS object FROM relations r"
               " JOIN concepts cs ON cs.id = r.subject_id JOIN concepts co ON co.id = r.object_id")
        where, args = [], []
        if concept_id is not None:
            where.append("(r.subject_id = ? OR r.object_id = ?)"); args += [concept_id, concept_id]
        if finding_id is not None:
            where.append("r.finding_id = ?"); args.append(finding_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY r.id", args)]

    def finding_concepts(self, finding_id: int) -> set[int]:
        rows = self.db.execute("SELECT subject_id, object_id FROM relations WHERE finding_id = ?", (finding_id,))
        return {c for r in rows for c in (r["subject_id"], r["object_id"])}

    def neighbours(self, concept_id: int, depth: int = 1) -> dict[int, int]:
        """Concepts reachable within `depth` relation hops (either direction): {id: hops}."""
        seen = {concept_id: 0}
        frontier = [concept_id]
        for hop in range(1, depth + 1):
            nxt = []
            for c in frontier:
                for r in self.db.execute("SELECT subject_id, object_id FROM relations WHERE subject_id = ? OR object_id = ?",
                                         (c, c)):
                    for n in (r["subject_id"], r["object_id"]):
                        if n not in seen:
                            seen[n] = hop
                            nxt.append(n)
            frontier = nxt
        del seen[concept_id]
        return seen

    # ---- series (graphable numbers) -----------------------------------------------------------
    def add_series(self, label: str, points: list, finding_id: int | None = None,
                   x_label: str = "", y_label: str = "", units: str = "") -> int:
        with self.lock, self._write():
            return self.db.execute(
                "INSERT INTO series (finding_id, label, x_label, y_label, units, points, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (finding_id, label, x_label, y_label, units, json.dumps(points), now())).lastrowid

    def series(self, finding_id: int | None = None) -> list[dict]:
        sql, args = "SELECT * FROM series", ()
        if finding_id is not None:
            sql, args = sql + " WHERE finding_id = ?", (finding_id,)
        return [{**dict(r), "points": json.loads(r["points"])} for r in self.db.execute(sql + " ORDER BY id", args)]

    # ---- prepared remarks (spoken without an LLM) ---------------------------------------------
    def set_remark(self, kind: str, text: str, topic_id: int | None = None) -> int:
        """The current remark of this kind (per topic, or general when topic_id is None);
        the previous one is kept but marked superseded."""
        with self.lock, self._write():
            new = self.db.execute("INSERT INTO remarks (topic_id, kind, text, written_at) VALUES (?, ?, ?, ?)",
                                  (topic_id, kind, text, now())).lastrowid
            self.db.execute("UPDATE remarks SET superseded_by = ? WHERE kind = ? AND topic_id IS ? AND id != ?"
                            " AND superseded_by IS NULL", (new, kind, topic_id, new))
        return new

    def remarks(self, topic_id: int | None = None, current_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM remarks WHERE topic_id IS ?"
        if current_only:
            sql += " AND superseded_by IS NULL"
        return [dict(r) for r in self.db.execute(sql + " ORDER BY id", (topic_id,))]

    # ---- notable (flags for a human) ----------------------------------------------------------
    def flag(self, kind: str, title: str, **refs) -> int:
        with self.lock, self._write():
            return self.db.execute("INSERT INTO notable (kind, title, refs, created_at) VALUES (?, ?, ?, ?)",
                                   (kind, title, json.dumps(refs), now())).lastrowid

    def notable(self, unreviewed_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM notable" + (" WHERE reviewed_at IS NULL" if unreviewed_only else "")
        return [{**dict(r), "refs": json.loads(r["refs"])} for r in self.db.execute(sql + " ORDER BY id DESC")]

    def mark_reviewed(self, notable_id: int):
        with self.lock, self._write():
            self.db.execute("UPDATE notable SET reviewed_at = ? WHERE id = ?", (now(), notable_id))

    # ---- maintenance ----------------------------------------------------------------------------
    def backfill_embeddings(self, batch: int = 32) -> int:
        """Embed findings and concepts stored while the embedder was unavailable. Returns how many."""
        if self.embedder is None:
            return 0
        done = 0
        for table, index in (("findings", self.findings_index), ("concepts", self.concepts_index)):
            col = "text" if table == "findings" else "name"
            rows = self.db.execute(f"SELECT id, {col} AS t FROM {table} WHERE embedding IS NULL ORDER BY id").fetchall()
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                try:
                    vectors = self.embedder.embed([r["t"] for r in chunk])
                except EmbedError:
                    return done
                with self.lock, self._write():
                    for r, v in zip(chunk, vectors):
                        self.db.execute(f"UPDATE {table} SET embedding = ? WHERE id = ?", (_blob(v), r["id"]))
                        index.add(r["id"], v)
                done += len(chunk)
        return done

    def stats(self) -> dict:
        counts = {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("topics", "episodes", "sources", "findings", "concepts", "relations",
                            "series", "remarks", "notable", "events")}
        counts["unembedded_findings"] = self.db.execute("SELECT COUNT(*) FROM findings WHERE embedding IS NULL").fetchone()[0]
        counts["unreviewed_notable"] = self.db.execute("SELECT COUNT(*) FROM notable WHERE reviewed_at IS NULL").fetchone()[0]
        return counts
