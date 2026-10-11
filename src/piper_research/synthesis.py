"""
SYNTHESIS - the second kind of cycle. Research gathers findings; synthesis reads none: it looks at
findings Piper already has for RELATIONSHIPS between them, and keeps the trail of how it got there.

    MATERIAL     choose a handful of findings that belong together, and say why:
                   bridge        a concept that turns up in this topic and in others
                   tension       the strongest evidence for and against the topic's thesis
                   neighbourhood a finding and its closest relatives in other topics
    REASON       the model works in numbered steps, each citing the findings it uses ([F123]),
                 then proposes at most 2 insights (link | tension | hypothesis | analogy | pattern),
                 each with its basis and 1-2 new questions carrying its nuance
    CHECK        every citation must be one of the chosen findings; each insight gets a grounding
                 check (grounded | partly | unsupported - unsupported ones are kept, as REJECTED,
                 so the trail shows what didn't survive); a restated insight bumps the existing one
    STORE        the insight + its TRACE (selection, steps and their checks, grounding, the model's
                 raw thinking, model and prompt version); its questions become open questions
                 tagged "derived" (a cap per topic, separate from the other open questions)
    TEST         later, when research takes up a derived question, the new findings are weighed
                 against the insight (supports | weakens | neither) and its confidence moves

When: step() runs synthesis instead of research when the chosen topic is saturating (its recent
cycles find little new) or after synth_every_max research cycles without one - adaptive, with a
floor - and not twice in a row on the same topic.
"""
from piper_memory.schema import INSIGHT_CONFIDENCE
from piper_research import prompts as P
from piper_research.llm import LLMError
from piper_research.verify import norm

LEVEL = {0.9: "established", 0.6: "reported", 0.3: "speculative"}


class SynthesisMixin:
    cycles_since_synthesis = 0

    # ---- when ---------------------------------------------------------------------------------
    def wants_synthesis(self, topic: dict) -> bool:
        cfg = self.cfg
        if not cfg.get("synthesis", True):
            return False
        n = self.mem.db.execute("SELECT COUNT(*) FROM findings WHERE topic_id = ? AND kind != 'question'"
                                " AND relevance != 'off_topic'", (topic["id"],)).fetchone()[0]
        if n < cfg.get("synth_min_findings", 15):
            return False
        recent = self.mem.episodes(topic["id"], limit=cfg.get("synth_topic_gap", 3))
        if any(e.get("kind") == "synthesis" for e in recent):
            return False
        return ((topic.get("saturation") or 0) >= cfg.get("synth_saturation", 0.5)
                or self.cycles_since_synthesis >= cfg.get("synth_every_max", 6))

    def step(self, topic_name: str | None = None) -> dict:
        """One cycle: synthesis when it's wanted (and there is material), else research."""
        topic = self.choose(topic_name)
        if self.wants_synthesis(topic):
            out = self.synthesize(topic)
            if out is not None:
                self.cycles_since_synthesis = 0
                return out
        out = self.cycle(topic=topic)
        self.cycles_since_synthesis += 1
        return out

    # ---- MATERIAL -------------------------------------------------------------------------------
    def _recent_keys(self, n: int = 40) -> set[str]:
        import json
        return {json.loads(d).get("key", "") for (d,) in self.mem.db.execute(
            "SELECT data FROM events WHERE kind = 'synthesized' ORDER BY id DESC LIMIT ?", (n,))}

    def _rows(self, ids: list[int]) -> list[dict]:
        out = []
        for fid in ids:
            f = self.mem.get_finding(fid)
            if f and f["kind"] != "question" and f["relevance"] != "off_topic":
                out.append(f)
        return out

    def _bridge(self, topic: dict, used: set[str]) -> dict | None:
        tid = topic["id"]
        by_concept: dict[int, dict[int, list]] = {}
        for c, t, fid, conf in self.mem.db.execute(
                "SELECT r.subject_id, f.topic_id, f.id, f.confidence FROM relations r JOIN findings f ON f.id = r.finding_id"
                " WHERE f.kind != 'question' AND f.relevance != 'off_topic'"
                " UNION SELECT r.object_id, f.topic_id, f.id, f.confidence FROM relations r JOIN findings f ON f.id = r.finding_id"
                " WHERE f.kind != 'question' AND f.relevance != 'off_topic'"):
            by_concept.setdefault(c, {}).setdefault(t, []).append((conf, fid))
        # a SPECIFIC concept shared by a few topics - not a hub word ("experience", "brain") that
        # turns up everywhere and links nothing in particular
        most = max(2, self.cfg.get("bridge_max_topics", 3))
        cands = []
        for c, topics in by_concept.items():
            if tid in topics and 2 <= len(topics) <= most and f"bridge:{c}" not in used:
                total = sum(len(v) for v in topics.values())
                if total >= 4 and min(len(v) for v in topics.values()) >= 1:
                    cands.append((min(total, 12) + 3 * (len(topics) == 2), c, topics))
        if not cands:
            return None
        cands.sort(reverse=True)
        _, c, topics = self.rng.choice(cands[:5])
        mine = [fid for _, fid in sorted(topics[tid], reverse=True)][:4]
        others = []
        lists = [[fid for _, fid in sorted(v, reverse=True)] for t, v in topics.items() if t != tid]
        while len(others) < 4 and any(lists):                 # round robin over the other topics
            for lst in lists:
                if lst and len(others) < 4:
                    others.append(lst.pop(0))
        name = self.mem.get_concept(c)["name"]
        names = [self.mem.get_topic(t)["name"] for t in topics]
        return {"strategy": "bridge", "key": f"bridge:{c}", "concept": name,
                "why": f"The concept \"{name}\" turns up in {len(topics)} topics: {', '.join(names)}.",
                "findings": list(dict.fromkeys(mine + others))}

    def _tension(self, topic: dict, used: set[str]) -> dict | None:
        if not topic["thesis"]:
            return None
        side = {}
        for stance in ("supports", "challenges"):
            side[stance] = [r[0] for r in self.mem.db.execute(
                "SELECT id FROM findings WHERE topic_id = ? AND stance = ? AND kind != 'question'"
                " AND relevance != 'off_topic' ORDER BY (relevance = 'core') DESC, confidence DESC, id DESC LIMIT 4",
                (topic["id"], stance))]
        if min(len(side["supports"]), len(side["challenges"])) < 2:
            return None
        key = f"tension:{topic['id']}:{max(side['supports'] + side['challenges'])}"
        if key in used:
            return None
        return {"strategy": "tension", "key": key,
                "why": f"The thesis \"{topic['thesis']}\" has strong evidence on both sides: the top"
                       f" {len(side['supports'])} supporting and {len(side['challenges'])} challenging findings.",
                "findings": side["supports"] + side["challenges"]}

    def _neighbourhood(self, topic: dict, used: set[str]) -> dict | None:
        seeds = [r[0] for r in self.mem.db.execute(
            "SELECT id FROM findings WHERE topic_id = ? AND kind != 'question' AND relevance = 'core'"
            " AND embedding IS NOT NULL ORDER BY confidence DESC, id DESC LIMIT 30", (topic["id"],))]
        seeds = [s for s in seeds if f"near:{s}" not in used]
        if not seeds:
            return None
        seed = self.rng.choice(seeds)
        v = self.mem.finding_vector(seed)
        if v is None:
            return None
        far = [h["id"] for h in self.mem.similar(v, k=12, min_score=self.cfg.get("synth_near_min", 0.55),
                                                  other_topics_than=topic["id"], exclude=(seed,), questions=False)
               if h["relevance"] != "off_topic"][:4]
        if len(far) < 2:
            return None
        near = [h["id"] for h in self.mem.similar(v, k=6, min_score=0.6, topic_id=topic["id"], exclude=(seed,),
                                                   questions=False) if h["relevance"] != "off_topic"][:3]
        others = sorted({self.mem.get_finding(f)["topic_id"] for f in far})
        return {"strategy": "neighbourhood", "key": f"near:{seed}",
                "why": f"Finding F{seed} and its closest relatives in "
                       f"{', '.join(self.mem.get_topic(t)['name'] for t in others)}.",
                "findings": [seed] + near + far}

    def material(self, topic: dict) -> dict | None:
        used = self._recent_keys()
        options = []
        for fn, weight in ((self._bridge, 3), (self._tension, 2), (self._neighbourhood, 2)):
            m = fn(topic, used)
            if m and len(m["findings"]) >= 4:
                options.append((m, weight))
        if not options:
            return None
        ms, ws = zip(*options)
        return self.rng.choices(ms, weights=ws)[0]

    # ---- REASON + CHECK + STORE -----------------------------------------------------------------
    def room_for_derived(self, topic_id: int) -> int:
        asked = self.asked(topic_id)
        n = sum(1 for (fid, text) in self.mem.db.execute(
            "SELECT f.id, f.text FROM findings f JOIN finding_tags ft ON ft.finding_id = f.id JOIN tags t ON t.id = ft.tag_id"
            " WHERE f.topic_id = ? AND f.kind = 'question' AND t.name = 'derived'", (topic_id,)) if norm(text) not in asked)
        return max(0, self.cfg.get("max_derived_open", 10) - n)

    def _ground(self, statement: str, basis: list[dict]) -> dict:
        user = f"Insight: {statement}\n\nFindings it rests on:\n" + "\n".join(f"[F{f['id']}] {f['text']}" for f in basis)
        try:
            r = self.llm.json(P.GROUND, user, P.GROUND_SCHEMA, think=self.judge_think)
        except LLMError as e:
            return {"verdict": "unchecked", "reason": f"check failed: {str(e)[:160]}"}
        return {"verdict": r.get("verdict", "unchecked"), "reason": " ".join(r.get("reason", "").split()),
                "thinking": getattr(self.llm, "last_thinking", "") or ""}

    def synthesize(self, topic: dict) -> dict | None:
        """One synthesis cycle on `topic`; None if there is no material for one."""
        import time
        mat = self.material(topic)
        if mat is None:
            return None
        t0, usage0 = time.monotonic(), self.llm.usage()
        tid = topic["id"]
        self.last_topic = tid
        rows = self._rows(mat["findings"])
        ids = [f["id"] for f in rows]
        tname = {t["id"]: t["name"] for t in self.mem.topics()}
        ep = self.mem.start_episode(tid, angle=f"Synthesis ({mat['strategy']}): {mat['why']}", model=self.llm.model,
                                    kind="synthesis")
        self.mem.log("topic_chosen", topic=topic["name"], topic_id=tid)
        self.say(f"[Research] Synthesis on {topic['name']}: {mat['why']}")
        stats = {"strategy": mat["strategy"], "material": len(ids), "insights": 0, "rejected_insights": 0,
                 "restated": 0, "derived_questions": 0}
        lines = []
        for f in rows:
            th = self.mem.get_topic(f["topic_id"])
            tags = [tname.get(f["topic_id"], "?")]
            if th["thesis"]:
                tags.append(f["stance"])
            tags.append(LEVEL.get(f["confidence"], "reported"))
            lines.append(f"[F{f['id']}] ({'; '.join(tags)}) {f['text']}")
        user = (f"Why these findings: {mat['why']}\nTopic: {topic['name']}\n"
                + (f"Its thesis: {topic['thesis']}\n" if topic["thesis"] else "")
                + "\nFindings:\n" + "\n".join(lines))
        try:
            self._check()
            r = self.llm.json(P.SYNTHESIZE, user, P.SYNTH_SCHEMA, think=self.judge_think,
                              max_tokens=self.cfg.get("synth_max_tokens", 6000), timeout=self.cfg.get("synth_timeout_s", 600))
            thinking = getattr(self.llm, "last_thinking", "") or ""
            bundle = set(ids)
            steps = []
            for s in r.get("steps") or []:
                cites = [c for c in (s.get("cites") or []) if isinstance(c, int)]
                unknown = [c for c in cites if c not in bundle]
                check = "ok" if cites and not unknown else ("no citation" if not cites else f"unknown finding(s) {unknown}")
                steps.append({"text": " ".join(s.get("text", "").split()), "cites": cites, "check": check})
            made = []
            llm_src = None
            for ins in (r.get("insights") or [])[:2]:
                self._check()
                statement = " ".join(ins.get("statement", "").split())
                if not statement:
                    continue
                based = [c for c in dict.fromkeys(ins.get("based_on") or []) if c in bundle]
                stray = [c for c in (ins.get("based_on") or []) if c not in bundle]
                kind = ins.get("kind") if ins.get("kind") in ("link", "tension", "hypothesis", "analogy", "pattern") else "link"
                conf = INSIGHT_CONFIDENCE.get(ins.get("confidence"), 0.3)
                if len(based) < 2:
                    grounding = {"verdict": "unsupported", "reason": "cites fewer than 2 of the chosen findings"
                                 + (f" (unknown: {stray})" if stray else "")}
                else:
                    grounding = self._ground(statement, [f for f in rows if f["id"] in based])
                status = "rejected" if grounding["verdict"] == "unsupported" else "new"
                if grounding["verdict"] == "partly":
                    conf = round(max(0.2, conf - 0.2), 3)
                if status != "rejected":
                    dup = self.mem.similar_insight(statement)
                    if dup:
                        self.mem.insight_event(dup["id"], "restated", episode=ep, statement=statement, basis=based)
                        self.mem.update_insight(dup["id"], confidence=dup["confidence"] + 0.05)
                        stats["restated"] += 1
                        continue
                trace = {"selection": {k: mat[k] for k in ("strategy", "why", "key") if k in mat} | {"findings": ids},
                         "steps": steps, "grounding": grounding, "thinking": thinking, "model": self.llm.model,
                         "prompt_version": P.SYNTH_PROMPT_VERSION, "proposed": ins,
                         "stray_citations": stray}
                iid = self.mem.add_insight(statement, tid, based or ids, kind=kind, confidence=conf, trace=trace,
                                           episode_id=ep, status=status)
                if status == "rejected":
                    stats["rejected_insights"] += 1
                else:
                    stats["insights"] += 1
                    made.append(statement)
                    llm_src = llm_src or self.mem.source("llm", self.llm.model)
                    known = {norm(t) for t in self.open_questions(tid)} | self.asked(tid)
                    for q in (ins.get("questions") or [])[:min(2, self.room_for_derived(tid))]:
                        q = " ".join(q.split())
                        if q.endswith("?") and norm(q) not in known and not self.mem.duplicate_of(q, topic_id=tid, questions=True):
                            qid = self.mem.add_finding(q, tid, ep, kind="question", confidence=1.0,
                                                       sources=[llm_src], tags=["derived"])
                            self.mem.link_question(iid, qid)
                            known.add(norm(q))
                            stats["derived_questions"] += 1
                self.mem.log("insight", insight=iid, topic_id=tid, statement=statement, insight_kind=kind,
                             status=status, verdict=grounding["verdict"])
            outcome = "progress" if made else "stalled"
            self.mem.end_episode(ep, outcome, " | ".join(made) or "no new insight", **stats, **self._usage(usage0, t0))
            self.mem.log("synthesized", episode=ep, topic_id=tid, strategy=mat["strategy"], key=mat["key"],
                         insights=stats["insights"], rejected=stats["rejected_insights"])
            self.say(f"[Research]   {stats['insights']} insight(s), {stats['rejected_insights']} rejected,"
                     f" {stats['derived_questions']} derived question(s)")
            return {"topic": topic["name"], "episode": ep, "outcome": outcome, "kind": "synthesis", **stats}
        except Exception as e:
            from piper_research.loop import Interrupted
            self.mem.end_episode(ep, "interrupted" if isinstance(e, Interrupted) else "error",
                                 "" if isinstance(e, Interrupted) else f"{type(e).__name__}: {e}",
                                 **stats, **self._usage(usage0, t0))
            raise

    # ---- TEST: research on a derived question weighs on its insight -------------------------------
    def weigh_evidence(self, insight: dict, ep: int, new_ids: list[int], stats: dict):
        if not new_ids:
            self.mem.insight_event(insight["id"], "evidence", episode=ep, effect="none", findings=[],
                                   reason="research on its question found nothing new")
            stats["insight_evidence"] = "none"
            return
        texts = [self.mem.get_finding(f)["text"] for f in new_ids]
        user = (f"Insight: {insight['statement']}\n\nNew findings:\n"
                + "\n".join(f"[F{f}] {t}" for f, t in zip(new_ids, texts)))
        try:
            r = self.llm.json(P.EVIDENCE, user, P.EVIDENCE_SCHEMA, think=self.judge_think)
        except LLMError as e:
            self.mem.insight_event(insight["id"], "evidence", episode=ep, effect="unchecked", findings=new_ids,
                                   reason=str(e)[:200])
            return
        effect = r.get("effect", "neither")
        before = insight["confidence"]
        after, status = before, insight["status"]
        if effect == "supports":
            after, status = before + 0.1, "supported"
        elif effect == "weakens":
            after = before - 0.15
            status = "dropped" if after <= 0.15 else "weakened"
        self.mem.update_insight(insight["id"], confidence=after, status=status)
        self.mem.insight_event(insight["id"], "evidence", episode=ep, effect=effect, findings=new_ids,
                               reason=" ".join(r.get("reason", "").split()), confidence_before=before,
                               confidence_after=round(min(0.95, max(0.05, after)), 3), status=status)
        self.mem.log("insight_evidence", insight=insight["id"], effect=effect, episode=ep)
        stats["insight_evidence"] = effect
