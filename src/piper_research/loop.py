"""
The research loop - one cycle:

    CHOOSE       pick a topic: stale, thin, open questions and a split of evidence raise its
                 score; topics that stopped yielding anything new (saturation) fall back
    QUESTION     the model picks an open question or writes a better one, plus search queries
    READ         Wikipedia: search, fetch pages (by revision), cut into passages, keep the ones
                 closest to the question (embeddings)
    EXTRACT      the model lists findings - claim, exact quote, kind, confidence, stance,
                 relations - plus a short answer and follow-up questions
    VERIFY       a finding is kept only if its quote is really in the passages; duplicates add a
                 source to the existing finding instead; near matches are judged (confirms /
                 refines / contradicts); contradictions are flagged
    CONSOLIDATE  concepts and relations into the graph (similar names merged only if the model
                 judges them the same); a few new findings checked for links to other topics
    REFLECT      every few cycles on a topic: where the evidence stands, remarks to say aloud,
                 next questions

All of it is recorded in the memory store with provenance, and logged as events.
"""
import json
import random
import re
import time
from datetime import datetime, timezone

import numpy as np

from piper_memory.embed import EmbedError, normalise
from piper_research import prompts as P
from piper_research.llm import OllamaChat
from piper_research.verify import find_quote, norm
from piper_research.wiki import Wikipedia, passages


class Interrupted(Exception):
    pass


def _hours_since(iso: str | None, now: datetime) -> float:
    if not iso:
        return 1e6
    return (now - datetime.fromisoformat(iso)).total_seconds() / 3600


class Researcher:
    def __init__(self, mem, llm: OllamaChat, wiki: Wikipedia, cfg: dict, log=print, rng=None,
                 should_stop=lambda: False):
        self.mem = mem
        self.llm = llm
        self.wiki = wiki
        self.cfg = cfg
        self.say = log
        self.rng = rng or random.Random()
        self.should_stop = should_stop
        self.same_cache: dict[tuple, bool] = {}
        self.cross_judged: set[tuple] = set()
        self.last_topic: int | None = None

    def _check(self):
        if self.should_stop():
            raise Interrupted()

    # ---- CHOOSE -------------------------------------------------------------------------------
    def topic_scores(self) -> list[tuple[dict, float]]:
        now = datetime.now(timezone.utc)
        out = []
        for t in self.mem.topics():
            if t["status"] == "retired":
                continue
            eps = self.mem.episodes(t["id"], limit=3)
            n = self.mem.db.execute("SELECT COUNT(*) FROM findings WHERE topic_id = ? AND kind != 'question'",
                                    (t["id"],)).fetchone()[0]
            staleness = min(1.0, _hours_since(eps[0]["started_at"] if eps else None, now) / 24)
            thin = 1 / (1 + n / 15)
            bal = self.mem.stance_balance(t["id"])
            s, c = bal["supports"]["weight"], bal["challenges"]["weight"]
            tension = (2 * min(s, c) / (s + c)) if (t["thesis"] and s + c > 0) else 0.5
            has_open = 1.0 if self.open_questions(t["id"]) else 0.0
            done = [e for e in eps if e["outcome"] in ("progress", "stalled")]
            saturation = (sum(1 for e in done if not e["stats"].get("new")) / len(done)) if len(done) >= 3 else 0.0
            score = (0.35 * staleness + 0.30 * thin + 0.20 * tension + 0.15 * has_open) * (1 - 0.7 * saturation)
            if t["status"] == "resting":
                score *= 0.3
            self.mem.update_topic(t["id"], novelty=round(thin, 3), progress=round(min(1.0, n / 60), 3),
                                  saturation=round(saturation, 3))
            out.append((t, max(score, 0.01)))
        return out

    def choose(self, name: str | None = None) -> dict:
        if name:
            match = [t for t in self.mem.topics() if t["name"].lower() == name.lower()]
            if not match:
                raise KeyError(f"no topic named {name!r}")
            return match[0]
        scored = self.topic_scores()
        if not scored:
            raise LookupError("no topics to research (load research_seeds.yaml)")
        if len(scored) > 1:                          # not the same topic twice in a row
            scored = [(t, s) for t, s in scored if t["id"] != self.last_topic] or scored
        topics, weights = zip(*scored)
        return self.rng.choices(topics, weights=[w * w for w in weights])[0]

    # ---- questions ----------------------------------------------------------------------------
    def asked(self, topic_id: int) -> set[str]:
        return {norm(r[0]) for r in self.mem.db.execute("SELECT angle FROM episodes WHERE topic_id = ?", (topic_id,))}

    def open_questions(self, topic_id: int) -> list[str]:
        asked = self.asked(topic_id)
        rows = self.mem.db.execute("SELECT text FROM findings WHERE topic_id = ? AND kind = 'question' ORDER BY id",
                                   (topic_id,))
        return [r[0] for r in rows if r[0].rstrip().endswith("?") and norm(r[0]) not in asked]

    def next_question(self, topic: dict) -> dict:
        tid = topic["id"]
        open_q = self.open_questions(tid)[:10]
        asked = [e["angle"] for e in self.mem.episodes(tid, limit=10) if e["angle"]]
        known = [f["text"] for f in self.mem.findings(tid, limit=40) if f["kind"] != "question"][:12]
        bal = self.mem.stance_balance(tid)
        user = (f"Topic: {topic['name']}\n"
                + (f"Description: {topic['description']}\n" if topic["description"] else "")
                + (f"Thesis to examine: {topic['thesis']}\n"
                   f"Evidence so far: {bal['supports']['count']} supporting, {bal['challenges']['count']} challenging,"
                   f" {bal['neutral']['count']} neutral findings\n" if topic["thesis"] else "No thesis - open exploration.\n")
                + f"\nOpen questions:\n{P.numbered(open_q)}\n\nAlready asked:\n{P.numbered(asked)}\n"
                + f"\nSome of what is known:\n{P.numbered(known)}")
        r = self.llm.json(P.QUESTION, user, P.QUESTION_SCHEMA)
        k = r.get("open_question", 0)
        question = open_q[k - 1] if 1 <= k <= len(open_q) else " ".join(r["question"].split())
        queries = [q for q in r.get("search_queries", []) if q.strip()][:3] or [topic["name"]]
        return {"question": question, "why": r.get("why", ""), "queries": queries}

    # ---- READ ---------------------------------------------------------------------------------
    def read(self, question: str, queries: list[str]) -> tuple[list[dict], list[dict]]:
        """(pages read, the passages most relevant to the question)."""
        titles: list[str] = []
        results = [self.wiki.search(q, limit=4) for q in queries]
        for i in range(4):                          # interleave: each query's best first
            for r in results:
                if i < len(r) and r[i] not in titles:
                    titles.append(r[i])
        pages = []
        for title in titles:
            if len(pages) >= self.cfg["pages_per_question"]:
                break
            page = self.wiki.page(title)
            if page and all(p["title"] != page["title"] for p in pages):
                pages.append(page)
        chunks = [c for p in pages for c in passages(p, self.cfg["passage_chars"])]
        return pages, self.rank(question, chunks)[: self.cfg["passages"]]

    def rank(self, question: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        emb = self.mem.embedder
        if emb is not None:
            try:
                q = emb.embed([question])[0]
                vecs = []
                for i in range(0, len(chunks), 64):
                    vecs.append(emb.embed([f"{c['title']}: {c['text']}" for c in chunks[i:i + 64]]))
                scores = normalise(np.vstack(vecs)) @ q
                order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
                return [chunks[i] for i in order]
            except EmbedError:
                pass
        words = set(re.findall(r"[a-z]{4,}", question.lower()))   # fallback: shared words
        return sorted(chunks, key=lambda c: -len(words & set(re.findall(r"[a-z]{4,}", c["text"].lower()))))

    # ---- judges -------------------------------------------------------------------------------
    def same_concept(self, a: str, b: str) -> bool:
        key = tuple(sorted((a.lower(), b.lower())))
        if key not in self.same_cache:
            r = self.llm.json(P.SAME_CONCEPT, f'Name 1: "{a}"\nName 2: "{b}"', P.SAME_SCHEMA)
            self.same_cache[key] = bool(r.get("same"))
            self.mem.log("concept_judged", a=a, b=b, same=self.same_cache[key])
        return self.same_cache[key]

    def relate_to(self, claim: str, earlier: dict) -> str:
        r = self.llm.json(P.RELATE, f"EARLIER: {earlier['text']}\nNEW: {claim}", P.RELATE_SCHEMA)
        return r.get("relation", "neither")

    # ---- one cycle ----------------------------------------------------------------------------
    def cycle(self, topic_name: str | None = None) -> dict:
        t0 = time.monotonic()
        usage0 = self.llm.usage()
        topic = self.choose(topic_name)
        tid = topic["id"]
        self.last_topic = tid
        if topic["status"] == "queued":
            self.mem.update_topic(tid, status="active")
        self.mem.log("topic_chosen", topic=topic["name"], topic_id=tid)
        self.say(f"[Research] Topic: {topic['name']}")

        q = self.next_question(topic)
        ep = self.mem.start_episode(tid, angle=q["question"], model=self.llm.model)
        self.mem.log("question", topic_id=tid, episode=ep, question=q["question"], why=q["why"], queries=q["queries"])
        self.say(f"[Research]   Q: {q['question']}")
        stats = {"question": q["question"], "pages": [], "extracted": 0, "rejected": 0, "new": 0,
                 "duplicates": 0, "extra_sources": 0, "confirms": 0, "refines": 0, "contradicts": 0,
                 "relations": 0, "follow_ups": 0, "cross_links": 0}
        try:
            self._check()
            pages, chosen = self.read(q["question"], q["queries"])
            stats["pages"] = [p["title"] for p in pages]
            self.mem.log("pages_read", episode=ep, pages=stats["pages"], passages=len(chosen))
            self.say(f"[Research]   Read: {', '.join(stats['pages']) or '(nothing found)'}")
            if not chosen:
                self.mem.end_episode(ep, "stalled", "no pages found", **stats, **self._usage(usage0, t0))
                return {"topic": topic["name"], "episode": ep, "outcome": "stalled", **stats}
            self._check()
            answer, new_ids = self.extract(topic, ep, q["question"], pages, chosen, stats)
            self._check()
            self.cross_links(tid, new_ids, stats)
            outcome = "progress" if stats["new"] else "stalled"
            self.mem.end_episode(ep, outcome, answer, **stats, **self._usage(usage0, t0))
            self.say(f"[Research]   {stats['new']} new, {stats['duplicates'] + stats['extra_sources']} known,"
                     f" {stats['rejected']} rejected (quote not found)  [{time.monotonic() - t0:.0f} s]")
            n_eps = self.mem.db.execute("SELECT COUNT(*) FROM episodes WHERE topic_id = ? AND outcome IN"
                                        " ('progress', 'stalled')", (tid,)).fetchone()[0]
            if n_eps % self.cfg["reflect_every"] == 0:
                self._check()
                self.reflect(tid)
            self.mem.log("cycle_done", episode=ep, topic_id=tid, outcome=outcome, new=stats["new"])
            return {"topic": topic["name"], "episode": ep, "outcome": outcome, "answer": answer, **stats}
        except Interrupted:
            self.mem.end_episode(ep, "interrupted", "", **stats, **self._usage(usage0, t0))
            raise
        except Exception as e:
            self.mem.end_episode(ep, "error", f"{type(e).__name__}: {e}", **stats, **self._usage(usage0, t0))
            raise

    def _usage(self, before: dict, t0: float) -> dict:
        now = self.llm.usage()
        return {"llm_calls": now["llm_calls"] - before["llm_calls"],
                "llm_tokens": now["llm_tokens"] - before["llm_tokens"],
                "seconds": round(time.monotonic() - t0, 1)}

    # ---- EXTRACT + VERIFY ---------------------------------------------------------------------
    def extract(self, topic, ep, question, pages, chosen, stats) -> tuple[str, list[int]]:
        tid = topic["id"]
        text = "\n\n".join(f"[{i}] ({c['title']} - {c['section'] or 'introduction'})\n{c['text']}"
                           for i, c in enumerate(chosen, 1))
        user = (f"Topic: {topic['name']}\n"
                + (f"Thesis: {topic['thesis']}\n" if topic["thesis"] else "Thesis: (none)\n")
                + f"Question: {question}\nAt most {self.cfg['max_findings']} findings.\n\nPassages:\n{text}")
        r = self.llm.json(P.EXTRACT, user, P.EXTRACT_SCHEMA)
        by_title = {p["title"]: p for p in pages}
        new_ids = []
        for f in (r.get("findings") or [])[: self.cfg["max_findings"]]:
            stats["extracted"] += 1
            claim = " ".join(f.get("claim", "").split())
            if not claim:
                continue
            # VERIFY: the quote, in the passage it names first, else in any passage read
            texts = [c["text"] for c in chosen]
            n = f.get("passage", 0)
            hit = (n - 1) if 1 <= n <= len(chosen) and find_quote(f["quote"], [texts[n - 1]]) is not None \
                else find_quote(f["quote"], texts)
            if hit is None:
                stats["rejected"] += 1
                self.mem.log("rejected", episode=ep, claim=claim, quote=f["quote"][:300])
                continue
            c = chosen[hit]
            page = by_title[c["title"]]
            sid = self.mem.source("wikipedia", page["title"], url=page["url"], revid=page["revid"])
            quote = " ".join(f["quote"].split())
            is_q = f.get("kind") == "question"
            dup = self.mem.duplicate_of(claim, topic_id=tid, questions=is_q)
            if dup:
                if self.mem.add_finding_source(dup["id"], sid, quote):
                    stats["extra_sources"] += 1
                    self.mem.log("confirmed_by_source", finding=dup["id"], source=page["title"])
                else:
                    stats["duplicates"] += 1
                continue
            status, relates_to = "new", None
            near = [] if is_q else self.mem.similar(claim, k=1, min_score=self.mem.related, topic_id=tid, questions=False)
            if near:
                rel = self.relate_to(claim, near[0])
                if rel in ("confirms", "refines", "contradicts"):
                    status, relates_to = rel, near[0]["id"]
                    stats[rel] += 1
            kind = f.get("kind", "fact")
            fid = self.mem.add_finding(claim, tid, ep, kind=kind, confidence=min(1.0, max(0.0, float(f.get("confidence", 0.5)))),
                                       status=status, relates_to=relates_to, stance=f.get("stance", "neutral"),
                                       sources=[(sid, quote)])
            stats["new"] += 1
            new_ids.append(fid)
            self.mem.log("finding", episode=ep, finding=fid, text=claim, stance=f.get("stance"), status=status,
                         source=page["title"])
            if status == "contradicts":
                self.mem.flag("contradiction", f"{claim}  <->  {near[0]['text']}", findings=[fid, relates_to], topics=[tid])
            for rel in f.get("relations") or []:
                try:
                    s = self.mem.concept(rel["subject"], judge=self.same_concept)
                    o = self.mem.concept(rel["object"], judge=self.same_concept)
                    if s != o:
                        self.mem.relate(s, rel["predicate"], o, finding_id=fid, confidence=f.get("confidence", 0.5))
                        stats["relations"] += 1
                except (KeyError, ValueError):
                    continue
        # follow-up questions become open questions on the topic
        llm_src = self.mem.source("llm", self.llm.model)
        known = {norm(t) for t in self.open_questions(tid)} | self.asked(tid)
        for fq in (r.get("follow_up_questions") or [])[:3]:
            fq = " ".join(fq.split())
            if fq.endswith("?") and norm(fq) not in known and not self.mem.duplicate_of(fq, topic_id=tid, questions=True):
                self.mem.add_finding(fq, tid, ep, kind="question", confidence=1.0, sources=[llm_src])
                known.add(norm(fq))
                stats["follow_ups"] += 1
        return " ".join((r.get("answer") or "").split()), new_ids

    # ---- CONSOLIDATE: links between topics ----------------------------------------------------
    def cross_links(self, tid: int, new_ids: list[int], stats: dict):
        for fid in new_ids[: self.cfg["cross_topic_checks"]]:
            for cand in self.mem.cross_topic_candidates(fid, k=1):
                pair = tuple(sorted((fid, cand["id"])))
                if pair in self.cross_judged:
                    continue
                self.cross_judged.add(pair)
                mine = self.mem.get_finding(fid)
                other_topic = self.mem.get_topic(cand["topic_id"])["name"]
                r = self.llm.json(P.CROSS_TOPIC, f"Finding A ({self.mem.get_topic(tid)['name']}): {mine['text']}\n"
                                  f"Finding B ({other_topic}): {cand['text']}", P.CROSS_SCHEMA)
                if r.get("link"):
                    self.mem.flag("cross_topic", r.get("explanation", ""), findings=[fid, cand["id"]],
                                  topics=[tid, cand["topic_id"]])
                    self.mem.log("cross_link", findings=[fid, cand["id"]], explanation=r.get("explanation", ""))
                    stats["cross_links"] += 1

    # ---- REFLECT ------------------------------------------------------------------------------
    def reflect(self, tid: int) -> dict:
        topic = self.mem.get_topic(tid)
        rows = self.mem.db.execute("SELECT text, stance, confidence FROM findings WHERE topic_id = ? AND kind != 'question'"
                                   " ORDER BY confidence DESC, id DESC LIMIT 40", (tid,)).fetchall()
        if not rows:
            return {}
        bal = self.mem.stance_balance(tid)
        lines = [f"[{r['stance']}, conf {r['confidence']:.1f}] {r['text']}" for r in rows]
        user = (f"Topic: {topic['name']}\n"
                + (f"Thesis: {topic['thesis']}\nBalance: {bal['supports']['count']} supporting,"
                   f" {bal['challenges']['count']} challenging, {bal['neutral']['count']} neutral\n"
                   if topic["thesis"] else "Thesis: (none - open exploration)\n")
                + "\nFindings:\n" + P.numbered(lines))
        r = self.llm.json(P.REFLECT, user, P.REFLECT_SCHEMA)
        for kind in ("position", "learned", "unsure"):
            if r.get(kind, "").strip():
                self.mem.set_remark(kind, " ".join(r[kind].split()), topic_id=tid)
        src = self.mem.source("llm", self.llm.model)
        known = {norm(t) for t in self.open_questions(tid)} | self.asked(tid)
        for nq in (r.get("next_questions") or [])[:3]:
            nq = " ".join(nq.split())
            if nq.endswith("?") and norm(nq) not in known:
                self.mem.add_finding(nq, tid, kind="question", confidence=1.0, sources=[src])
        self.mem.log("reflected", topic_id=tid, position=r.get("position", ""), learned=r.get("learned", ""))
        self.say(f"[Research]   Reflected on {topic['name']}: {r.get('learned', '')}")
        return r

    # ---- the morning summary ------------------------------------------------------------------
    def overnight(self, since_iso: str) -> str | None:
        rows = self.mem.db.execute(
            "SELECT t.name, e.angle, e.summary, e.stats FROM episodes e JOIN topics t ON t.id = e.topic_id"
            " WHERE e.started_at >= ? AND e.outcome IN ('progress', 'stalled') ORDER BY e.id", (since_iso,)).fetchall()
        if not rows:
            return None
        lines = [f"{r['name']}: {r['angle']} -> {r['summary'] or '(no answer)'} ({json.loads(r['stats']).get('new', 0)} new findings)"
                 for r in rows][-30:]
        learned = [r["text"] for r in self.mem.db.execute(
            "SELECT text FROM remarks WHERE kind = 'learned' AND written_at >= ? AND superseded_by IS NULL", (since_iso,))]
        user = "Tonight's research cycles:\n" + P.numbered(lines) + ("\n\nThings you noted:\n" + P.numbered(learned) if learned else "")
        r = self.llm.json(P.OVERNIGHT, user, P.OVERNIGHT_SCHEMA)
        text = " ".join(r.get("summary", "").split())
        if text:
            self.mem.set_remark("overnight", text)
            self.mem.log("overnight", cycles=len(rows), summary=text)
        return text or None
