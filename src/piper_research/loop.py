"""
The research loop - one cycle:

    CHOOSE       pick a topic: stale, thin, open questions and a split of evidence raise its
                 score; topics that stopped yielding anything new (saturation) fall back
    QUESTION     the model picks an open question or writes a better one, plus search queries
    READ         Wikipedia (pages by revision) and the Stanford Encyclopedia of Philosophy (entries
                 by revision date): search, fetch, cut into passages, keep the ones closest to
                 the question (embeddings) - whichever source they come from
    EXTRACT      the model lists findings - claim, exact quote, kind, confidence, stance,
                 relations - plus a short answer and follow-up questions
    VERIFY       a finding is kept only if its quote is really in the passages; duplicates add a
                 source to the existing finding instead
    JUDGE        each new finding, one at a time (thinking on): relevance to the topic (off-topic
                 ones are dropped) and stance toward the thesis, with a reason; near matches are
                 judged (confirms / refines / contradicts); contradictions are flagged
    CONSOLIDATE  concepts and relations into the graph (similar names merged only if the model
                 judges them the same); a few new findings checked for links to other topics
                 (a few a night, and not between topics already known to be linked)
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
from piper_memory.schema import CERTAINTY
from piper_research import prompts as P
from piper_research.llm import LLMError, OllamaChat
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
                 should_stop=lambda: False, sep=None):
        self.mem = mem
        self.llm = llm
        self.wiki = wiki
        self.sep = sep                              # Stanford Encyclopedia of Philosophy (or None)
        self.cfg = cfg
        self.say = log
        self.rng = rng or random.Random()
        self.should_stop = should_stop
        self.same_cache: dict[tuple, bool] = {}
        self.cross_judged: set[tuple] = set()
        self.last_topic: int | None = None
        self.cross_links_left = cfg.get("cross_links_per_window", 5)

    def new_window(self):
        """A research window has opened: reset the per-window budgets."""
        self.cross_links_left = self.cfg.get("cross_links_per_window", 5)

    @property
    def judge_think(self) -> bool:
        return self.cfg.get("judge_think", True)

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

    def ranked_open_questions(self, topic: dict, k: int = 10) -> list[str]:
        """The open questions closest to what the topic is about (its name, description, thesis)."""
        open_q = self.open_questions(topic["id"])
        if len(open_q) <= k:
            return open_q
        about = " ".join(x for x in (topic["name"], topic["description"], topic["thesis"]) if x)
        hits = self.mem.similar(about, k=len(open_q) + 50, topic_id=topic["id"], questions=True)
        wanted = {norm(q) for q in open_q}
        ranked = [h["text"] for h in hits if norm(h["text"]) in wanted]
        return (ranked or open_q)[:k]

    def room_for_questions(self, topic_id: int) -> int:
        return max(0, self.cfg.get("max_open_questions", 30) - len(self.open_questions(topic_id)))

    def prune_questions(self, topic: dict, keep: int | None = None) -> int:
        """Delete the model's open questions beyond the `keep` most relevant (seed questions and
        questions already asked stay). Returns how many were deleted."""
        keep = self.cfg.get("max_open_questions", 30) if keep is None else keep
        ranked = self.ranked_open_questions(topic, k=10_000)
        drop = {norm(q) for q in ranked[keep:]}
        n = 0
        for row in self.mem.db.execute(
                "SELECT f.id, f.text FROM findings f WHERE f.topic_id = ? AND f.kind = 'question' AND NOT EXISTS"
                " (SELECT 1 FROM finding_sources fs JOIN sources s ON s.id = fs.source_id"
                "  WHERE fs.finding_id = f.id AND s.kind = 'user')", (topic["id"],)).fetchall():
            if norm(row["text"]) in drop:
                self.mem.delete_finding(row["id"])
                n += 1
        if n:
            self.mem.log("questions_pruned", topic_id=topic["id"], deleted=n)
        return n

    def next_question(self, topic: dict) -> dict:
        tid = topic["id"]
        open_q = self.ranked_open_questions(topic)
        asked = [e["angle"] for e in self.mem.episodes(tid, limit=10) if e["angle"]]
        known = [f["text"] for f in self.mem.findings(tid, limit=60)
                 if f["kind"] != "question" and f["relevance"] != "off_topic"][:12]
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
        pages += self.read_sep(queries)
        chunks = [c for p in pages for c in passages(p, self.cfg["passage_chars"])]
        return pages, self.rank(question, chunks)[: self.cfg["passages"]]

    def read_sep(self, queries: list[str]) -> list[dict]:
        """Up to sep_pages SEP entries for the queries (its 5 s crawl delay makes each request slow,
        so only the first two queries are searched). A failure here only costs this cycle's SEP."""
        n = self.cfg.get("sep_pages", 1)
        if self.sep is None or n <= 0:
            return []
        try:
            entries: list[str] = []
            for q in queries[:2]:
                for e in self.sep.search(q, limit=2):
                    if e not in entries:
                        entries.append(e)
            out = []
            for e in entries[:n]:
                page = self.sep.page(e)
                if page:
                    out.append(page)
            return out
        except Exception as e:                      # network, parsing: carry on with Wikipedia alone
            self.mem.log("sep_failed", error=str(e)[:200])
            return []

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

    def judge(self, topic: dict, claim: str, quote: str = "") -> dict:
        """Relevance to the topic and stance toward its thesis, with a reason (thinking on)."""
        user = (f"Topic: {topic['name']}\n"
                + (f"Description: {topic['description']}\n" if topic["description"] else "")
                + (f"Thesis: {topic['thesis']}\n" if topic["thesis"] else "Thesis: (none - stance is neutral)\n")
                + f"\nFinding: {claim}\n" + (f'Source quote: "{quote}"\n' if quote else ""))
        try:
            r = self.llm.json(P.JUDGE, user, P.JUDGE_SCHEMA, think=self.judge_think)
        except LLMError as e:                       # e.g. a runaway think: keep the finding, judge it later
            self.mem.log("judge_failed", claim=claim[:120], error=str(e)[:200])
            return {"relevance": "core", "stance": "neutral", "reason": "", "failed": True}
        relevance = r.get("relevance") if r.get("relevance") in ("core", "background", "off_topic") else "core"
        stance = r.get("stance") if (topic["thesis"] and r.get("stance") in ("supports", "challenges")) else "neutral"
        return {"relevance": relevance, "stance": stance, "reason": " ".join(r.get("reason", "").split())}

    def rejudge(self, limit: int = 50, topic_id: int | None = None) -> dict:
        """Judge findings stored before the judge existed (judged_by IS NULL): those with a stance
        first (the old stances were the unreliable part), then the rest, oldest first."""
        sql = ("SELECT f.id, f.text, f.topic_id, f.stance, (SELECT excerpt FROM finding_sources fs"
               " WHERE fs.finding_id = f.id LIMIT 1) AS quote FROM findings f"
               " WHERE f.judged_by IS NULL AND f.kind != 'question'")
        args = []
        if topic_id is not None:
            sql += " AND f.topic_id = ?"
            args.append(topic_id)
        rows = self.mem.db.execute(sql + " ORDER BY (f.stance = 'neutral'), f.id LIMIT ?", (*args, limit)).fetchall()
        counts = {"judged": 0, "off_topic": 0, "changed": 0}
        topics = {}
        for row in rows:
            self._check()
            topic = topics.setdefault(row["topic_id"], self.mem.get_topic(row["topic_id"]))
            j = self.judge(topic, row["text"], row["quote"] or "")
            if j.get("failed"):
                continue
            self.mem.judge_finding(row["id"], j["stance"], j["relevance"], j["reason"], self.llm.model)
            counts["judged"] += 1
            counts["off_topic"] += j["relevance"] == "off_topic"
            counts["changed"] += j["stance"] != row["stance"]
        if rows:
            self.mem.log("rejudged", **counts)
        return counts

    def unjudged(self) -> int:
        return self.mem.db.execute("SELECT COUNT(*) FROM findings WHERE judged_by IS NULL"
                                   " AND kind != 'question'").fetchone()[0]

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
                 "relations": 0, "follow_ups": 0, "cross_links": 0, "off_topic": 0,
                 "supports": 0, "challenges": 0, "neutral": 0}
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
        label = {"sep": "Stanford Encyclopedia of Philosophy: ", "wikipedia": "Wikipedia: "}
        text = "\n\n".join(f"[{i}] ({label.get(c.get('kind'), '')}{c['title']} - {c['section'] or 'introduction'})\n{c['text']}"
                           for i, c in enumerate(chosen, 1))
        user = (f"Topic: {topic['name']}\n"
                + (f"Thesis: {topic['thesis']}\n" if topic["thesis"] else "Thesis: (none)\n")
                + f"Question: {question}\nAt most {self.cfg['max_findings']} findings.\n\nPassages:\n{text}")
        r = self.llm.json(P.EXTRACT, user, P.EXTRACT_SCHEMA)
        by_key = {f"{p.get('kind', 'wikipedia')}:{p['title']}": p for p in pages}
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
            page = by_key[c.get("key", f"wikipedia:{c['title']}")]
            sid = self.mem.source(page.get("kind", "wikipedia"), page["title"], url=page["url"], revid=page["revid"])
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
            j = self.judge(topic, claim, quote)
            if j["relevance"] == "off_topic":
                stats["off_topic"] += 1
                self.mem.log("off_topic", episode=ep, claim=claim, reason=j["reason"])
                continue
            stats[j["stance"]] += 1
            status, relates_to = "new", None
            near = [] if is_q else self.mem.similar(claim, k=1, min_score=self.mem.related, topic_id=tid, questions=False)
            if near:
                rel = self.relate_to(claim, near[0])
                if rel in ("confirms", "refines", "contradicts"):
                    status, relates_to = rel, near[0]["id"]
                    stats[rel] += 1
            kind = f.get("kind", "fact")
            confidence = CERTAINTY.get(f.get("certainty"), 0.6)
            fid = self.mem.add_finding(claim, tid, ep, kind=kind, confidence=confidence, status=status,
                                       relates_to=relates_to, stance=j["stance"], relevance=j["relevance"],
                                       stance_reason=j["reason"], sources=[(sid, quote)],
                                       judged_by=None if j.get("failed") else self.llm.model)
            stats["new"] += 1
            new_ids.append(fid)
            self.mem.log("finding", episode=ep, finding=fid, text=claim, stance=j["stance"], relevance=j["relevance"],
                         status=status, source=page["title"])
            if status == "contradicts":
                self.mem.flag("contradiction", f"{claim}  <->  {near[0]['text']}", findings=[fid, relates_to], topics=[tid])
            for rel in f.get("relations") or []:
                if max(len(rel.get("subject", "").split()), len(rel.get("object", "").split())) > 4:
                    continue                        # a phrase, not a concept
                try:
                    s = self.mem.concept(rel["subject"], judge=self.same_concept)
                    o = self.mem.concept(rel["object"], judge=self.same_concept)
                    if s != o:
                        self.mem.relate(s, rel["predicate"], o, finding_id=fid, confidence=confidence)
                        stats["relations"] += 1
                except (KeyError, ValueError):
                    continue
        # follow-up questions become open questions on the topic
        llm_src = self.mem.source("llm", self.llm.model)
        known = {norm(t) for t in self.open_questions(tid)} | self.asked(tid)
        for fq in (r.get("follow_up_questions") or [])[:min(3, self.room_for_questions(tid))]:
            fq = " ".join(fq.split())
            if fq.endswith("?") and norm(fq) not in known and not self.mem.duplicate_of(fq, topic_id=tid, questions=True):
                self.mem.add_finding(fq, tid, ep, kind="question", confidence=1.0, sources=[llm_src])
                known.add(norm(fq))
                stats["follow_ups"] += 1
        return " ".join((r.get("answer") or "").split()), new_ids

    # ---- CONSOLIDATE: links between topics ----------------------------------------------------
    def links_between(self, a: int, b: int) -> int:
        """How many cross-topic links the judge has flagged between topics a and b (flags from
        before the judge - night 1's 755 - carry no "judged" mark and don't count)."""
        n = 0
        for (refs,) in self.mem.db.execute("SELECT refs FROM notable WHERE kind = 'cross_topic'"):
            r = json.loads(refs)
            if r.get("judged") and {a, b} <= set(r.get("topics", [])):
                n += 1
        return n

    def related_topics(self, a: int, b: int) -> bool:
        """Topics already known to be linked: parent and child, or links already flagged."""
        ta, tb = self.mem.get_topic(a), self.mem.get_topic(b)
        if ta["parent_id"] == b or tb["parent_id"] == a:
            return True
        return self.links_between(a, b) >= self.cfg.get("cross_links_per_pair", 2)

    def cross_links(self, tid: int, new_ids: list[int], stats: dict):
        for fid in new_ids[: self.cfg["cross_topic_checks"]]:
            if self.cross_links_left <= 0:
                return
            for cand in self.mem.cross_topic_candidates(fid, k=1, min_score=self.cfg.get("cross_topic_min", 0.75)):
                pair = tuple(sorted((fid, cand["id"])))
                if pair in self.cross_judged or self.related_topics(tid, cand["topic_id"]):
                    continue
                if self.mem.get_finding(cand["id"])["relevance"] == "off_topic":
                    continue
                self.cross_judged.add(pair)
                mine = self.mem.get_finding(fid)
                other_topic = self.mem.get_topic(cand["topic_id"])["name"]
                r = self.llm.json(P.CROSS_TOPIC, f"Finding A ({self.mem.get_topic(tid)['name']}): {mine['text']}\n"
                                  f"Finding B ({other_topic}): {cand['text']}", P.CROSS_SCHEMA, think=self.judge_think)
                if r.get("link"):
                    self.cross_links_left -= 1
                    self.mem.flag("cross_topic", r.get("explanation", ""), findings=[fid, cand["id"]],
                                  topics=[tid, cand["topic_id"]], judged=True)
                    self.mem.log("cross_link", findings=[fid, cand["id"]], explanation=r.get("explanation", ""))
                    stats["cross_links"] += 1

    # ---- REFLECT ------------------------------------------------------------------------------
    def reflect(self, tid: int) -> dict:
        topic = self.mem.get_topic(tid)
        rows = self.mem.db.execute("SELECT text, stance, confidence FROM findings WHERE topic_id = ? AND kind != 'question'"
                                   " AND relevance != 'off_topic' ORDER BY (relevance = 'core') DESC,"
                                   " (stance != 'neutral') DESC, confidence DESC, id DESC LIMIT 40", (tid,)).fetchall()
        if not rows:
            return {}
        bal = self.mem.stance_balance(tid)
        level = {0.9: "established", 0.6: "reported", 0.3: "speculative"}
        lines = [f"[{r['stance']}, {level.get(r['confidence'], 'reported')}] {r['text']}" for r in rows]
        user = (f"Topic: {topic['name']}\n"
                + (f"Thesis: {topic['thesis']}\nBalance: {bal['supports']['count']} supporting,"
                   f" {bal['challenges']['count']} challenging, {bal['neutral']['count']} neutral\n"
                   if topic["thesis"] else "Thesis: (none - open exploration)\n")
                + "\nFindings:\n" + P.numbered(lines))
        r = self.llm.json(P.REFLECT, user, P.REFLECT_SCHEMA, think=self.judge_think)
        for kind in ("position", "learned", "unsure"):
            if r.get(kind, "").strip():
                self.mem.set_remark(kind, " ".join(r[kind].split()), topic_id=tid)
        src = self.mem.source("llm", self.llm.model)
        known = {norm(t) for t in self.open_questions(tid)} | self.asked(tid)
        for nq in (r.get("next_questions") or [])[:min(3, self.room_for_questions(tid))]:
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
