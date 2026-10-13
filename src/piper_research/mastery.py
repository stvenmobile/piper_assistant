"""
MASTERY - the goal for every topic. Piper works towards it; her household confirms it.

    READINESS   a checklist per current topic, from the memory alone (the dashboard shows it too):
                  saturated         recent research cycles find little new
                  evidence base     enough findings (core + background)
                  synthesis         she has looked for relationships in it
                  insights tested   no derived questions left untested
                  guidance explored no open questions from your "not yet" guidance
                  a position        her reflection has said where the topic stands
    REQUEST     when every item is met, she REQUESTS mastery (topics.mastery = 'candidate')
    WITHDRAW    if the checklist slips before you answer (new research finds new things again, a
                derived question opens), she withdraws the request at her next checkpoint and asks
                again once every item is met - you're never asked to confirm a stale request
    CONFIRM     you confirm on the dashboard -> she writes the MASTERY BRIEF (bottom line, the
                established core with its findings, live conjectures, edge-case questions, what her
                insights added) and the topic moves to Deeply Explored (status 'explored'), freeing
                one of the current slots (research.max_current_topics)
    NOT YET     you say what else needs exploring -> she turns it into priority questions (tagged
                'mastery_gap', researched first), and the checklist waits for them

Topics close to mastery get a little more of her time (the completion drive in topic_scores):
finishing is worth more than starting.
"""
import json
from datetime import datetime, timezone

from piper_research import prompts as P
from piper_research.verify import norm


def _asked(mem, topic_id: int) -> set[str]:
    return {norm(r[0]) for r in mem.db.execute("SELECT angle FROM episodes WHERE topic_id = ?", (topic_id,))}


def open_tagged(mem, topic_id: int, tag: str) -> list[dict]:
    asked = _asked(mem, topic_id)
    return [dict(r) for r in mem.db.execute(
        "SELECT f.id, f.text FROM findings f JOIN finding_tags ft ON ft.finding_id = f.id JOIN tags t ON t.id = ft.tag_id"
        " WHERE f.topic_id = ? AND f.kind = 'question' AND t.name = ? ORDER BY f.id", (topic_id, tag))
        if norm(r["text"]) not in asked]


def readiness(mem, topic: dict, cfg: dict) -> dict:
    """The mastery checklist for a topic: {"score": 0..1, "ready": bool, "criteria": [...]}."""
    tid = topic["id"]
    sat = topic.get("saturation") or 0.0
    n = mem.db.execute("SELECT COUNT(*) FROM findings WHERE topic_id = ? AND kind != 'question'"
                       " AND relevance != 'off_topic'", (tid,)).fetchone()[0]
    need_sat, need_n = cfg.get("mastery_saturation", 0.8), cfg.get("mastery_min_findings", 60)
    synth = mem.db.execute("SELECT COUNT(*) FROM episodes WHERE topic_id = ? AND kind = 'synthesis'"
                           " AND outcome IN ('progress', 'stalled')", (tid,)).fetchone()[0]
    derived, gaps = len(open_tagged(mem, tid, "derived")), len(open_tagged(mem, tid, "mastery_gap"))
    position = mem.db.execute("SELECT 1 FROM remarks WHERE topic_id = ? AND kind = 'position' AND superseded_by IS NULL",
                              (tid,)).fetchone() is not None
    crit = [
        {"key": "saturated", "label": "Recent research finds little new", "met": sat >= need_sat,
         "detail": f"saturation {sat:.0%} (needs {need_sat:.0%})"},
        {"key": "evidence", "label": "A solid evidence base", "met": n >= need_n, "detail": f"{n} findings (needs {need_n})"},
        {"key": "synthesis", "label": "Relationships explored (synthesis)", "met": synth > 0,
         "detail": f"{synth} synthesis cycle(s)"},
        {"key": "insights", "label": "Insight questions tested", "met": derived == 0, "detail": f"{derived} derived question(s) open"},
        {"key": "guidance", "label": "Your guidance explored", "met": gaps == 0, "detail": f"{gaps} guidance question(s) open"},
        {"key": "position", "label": "A stated position", "met": position, "detail": "reflected" if position else "not yet reflected"},
    ]
    met = sum(c["met"] for c in crit)
    return {"score": round(met / len(crit), 3), "ready": met == len(crit), "criteria": crit}


def free_slots(mem, cfg: dict) -> int:
    return max(0, cfg.get("max_current_topics", 8) - len(mem.current_topics()))


class Mastery:
    """The mastery work that needs the model and writes: requests, briefs, guidance, proposals."""

    def __init__(self, r):
        self.r, self.mem, self.cfg = r, r.mem, r.cfg

    def check(self) -> list[str]:
        """Request mastery for every current topic whose checklist is complete, and withdraw an
        unanswered request whose checklist has slipped since. Returns the names newly requested."""
        out = []
        answered = {q["topic_id"] for q in pending_requests(self.mem) if q["kind"] in ("mastery_confirm", "mastery_decline")}
        for t in self.mem.current_topics():
            if t["status"] != "active":
                continue
            if t.get("mastery") == "candidate":
                if t["id"] in answered:                        # your answer is on its way: leave it
                    continue
                rd = readiness(self.mem, t, self.cfg)
                if not rd["ready"]:
                    self.mem.update_topic(t["id"], mastery="")
                    self.mem.log("mastery_withdrawn", topic_id=t["id"], name=t["name"],
                                 unmet={c["key"]: c["detail"] for c in rd["criteria"] if not c["met"]})
                continue
            if t.get("mastery"):
                continue
            rd = readiness(self.mem, t, self.cfg)
            if rd["ready"]:
                self.mem.update_topic(t["id"], mastery="candidate")
                self.mem.log("mastery_requested", topic_id=t["id"], name=t["name"],
                             criteria={c["key"]: c["detail"] for c in rd["criteria"]})
                out.append(t["name"])
        return out

    # ---- confirmed: the brief, and the move to Deeply Explored -----------------------------------------
    def brief(self, topic_id: int) -> int:
        mem, r = self.mem, self.r
        t = mem.get_topic(topic_id)
        rows = mem.db.execute("SELECT id, text, stance, confidence FROM findings WHERE topic_id = ? AND kind != 'question'"
                              " AND relevance != 'off_topic' ORDER BY (relevance = 'core') DESC, confidence DESC,"
                              " (stance != 'neutral') DESC, id DESC LIMIT 45", (topic_id,)).fetchall()
        level = {0.9: "established", 0.6: "reported", 0.3: "speculative"}
        bal = mem.stance_balance(topic_id)
        contra = [x["title"] for x in mem.notable(unreviewed_only=False) if x["kind"] == "contradiction"
                  and topic_id in (x["refs"].get("topics") or [])][:8]
        ins = mem.insights(topic_id=topic_id, limit=50)
        open_q = [q for q in r.open_questions(topic_id)][:15]
        rd = readiness(mem, t, self.cfg)
        user = (f"Topic: {t['name']}\n" + (f"Thesis: {t['thesis']}\nBalance: {bal['supports']['count']} supporting,"
                                          f" {bal['challenges']['count']} challenging, {bal['neutral']['count']} neutral\n"
                                          if t["thesis"] else "No thesis - open exploration.\n")
                + "\nFindings:\n" + "\n".join(f"[F{x['id']}] ({x['stance']}, {level.get(x['confidence'], 'reported')}) {x['text']}"
                                              for x in rows)
                + "\n\nContradictions found:\n" + P.numbered(contra)
                + "\n\nInsights:\n" + P.numbered([f"[{i['status']}, {i['confidence']:.2f}] {i['statement']}" for i in ins
                                                   if i["status"] != "rejected"])
                + "\n\nQuestions still open:\n" + P.numbered(open_q))
        b = r.llm.json(P.BRIEF, user, P.BRIEF_SCHEMA, think=r.judge_think,
                       max_tokens=self.cfg.get("synth_max_tokens", 6000), timeout=self.cfg.get("synth_timeout_s", 600))
        known = {x["id"] for x in rows}
        lines = [f"# Mastery brief - {t['name']}", "",
                 f"_Deeply explored {datetime.now().astimezone():%d %B %Y}"
                 + (f" · {bal['supports']['count']} supporting / {bal['challenges']['count']} challenging" if t["thesis"] else "")
                 + f" · {sum(1 for _ in rows)}+ findings_", ""]
        if t["thesis"]:
            lines += [f"**Thesis:** {t['thesis']}", ""]
        lines += ["## Bottom line", " ".join(b.get("bottom_line", "").split()), "", "## What is established"]
        for e in b.get("established") or []:
            cites = [f for f in e.get("findings") or [] if f in known]
            lines.append(f"- {' '.join(e.get('point', '').split())}" + (f" _({', '.join(f'F{c}' for c in cites)})_" if cites else ""))
        lines += ["", "## Live conjectures and disputes"]
        for c in b.get("conjectures") or []:
            lines.append(f"- **{' '.join(c.get('claim', '').split())}** _({c.get('status', '')})_ - for: {c.get('for', '')};"
                         f" against: {c.get('against', '')}")
        lines += ["", "## Edge-case questions worth exploring later"] + [f"- {q}" for q in b.get("edge_questions") or []]
        lines += ["", "## What the insights added", " ".join(b.get("insights_note", "").split())]
        held = [i for i in ins if i["status"] == "supported"]
        fell = [i for i in ins if i["status"] in ("weakened", "dropped")]
        lines += [f"- held up: {i['statement']} _(insight #{i['id']})_" for i in held]
        lines += [f"- fell: {i['statement']} _(insight #{i['id']})_" for i in fell]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rid = mem.add_report("mastery", t["created_at"], now, "\n".join(lines),
                             {"topic_id": topic_id, "name": t["name"], "brief": b, "readiness": rd,
                              "balance": {k: v["count"] for k, v in bal.items()}})
        mem.update_topic(topic_id, status="explored", stage="explored", mastery="", mastered_at=now)
        mem.log("topic_mastered", topic_id=topic_id, name=t["name"], report=rid)
        mem.set_remark("mastered", f"I have mastered {t['name']} - it is now one of my deeply explored topics.")
        return rid

    # ---- not yet: your guidance becomes priority questions ---------------------------------------------
    def guidance(self, topic_id: int, text: str) -> list[str]:
        mem, r = self.mem, self.r
        t = mem.get_topic(topic_id)
        text = " ".join(text.split())
        qs = []
        if text:
            try:
                qs = r.llm.json(P.GAPS, f"Topic: {t['name']}\nYour household's guidance: {text}", P.GAPS_SCHEMA).get("questions") or []
            except Exception as e:                   # the guidance itself is kept either way
                mem.log("research_error", error=f"guidance: {e}"[:200])
            if not qs:
                qs = [text if text.endswith("?") else f"{text.rstrip('.')}?"]
            src = mem.source("user", "your guidance")
            known = {norm(q["text"]) for q in open_tagged(mem, topic_id, "mastery_gap")}
            for q in qs[:3]:
                q = " ".join(q.split())
                if q and norm(q) not in known:
                    mem.add_finding(q, topic_id, kind="question", confidence=1.0, sources=[src], tags=["mastery_gap"])
            mem.set_remark("mastery_guidance", text, topic_id=topic_id)
        mem.update_topic(topic_id, mastery="")
        mem.log("mastery_not_yet", topic_id=topic_id, name=t["name"], guidance=text, questions=qs[:3])
        return qs[:3]

    # ---- proposals: new topics and sub-topics, added only by you ----------------------------------------
    def propose(self, name: str, description: str, why: str, thesis: str = "", parent: int | None = None,
                questions: list[int] | None = None, insights: list[int] | None = None) -> int | None:
        names = {x["name"].lower() for x in self.mem.topics()}
        open_props = {x["refs"].get("name", "").lower() for x in self.mem.notable() if x["kind"] == "proposal"}
        if not name or name.lower() in names or name.lower() in open_props:
            return None
        pid = self.mem.flag("proposal", why, name=name, description=description, thesis=thesis, parent=parent,
                            questions=questions or [], insights=insights or [])
        self.mem.log("topic_proposed", proposal=pid, name=name, why=why)
        return pid

    def add_topic(self, name: str, thesis: str = "", description: str = "", questions: list[str] | None = None,
                  origin: str = "user", parent: int | None = None, move_questions: list[int] | None = None) -> int | None:
        """A new current topic, if there's a free slot."""
        mem = self.mem
        name = " ".join(name.split())
        if not name or free_slots(mem, self.cfg) <= 0 or mem.db.execute("SELECT 1 FROM topics WHERE name = ?", (name,)).fetchone():
            return None
        tid = mem.topic(name, parent=parent, description=description, origin=origin, thesis=" ".join(thesis.split()))
        src = mem.source("user", "your topics")
        for q in questions or []:
            q = " ".join(q.split())
            if q:
                mem.add_finding(q if q.endswith("?") else q + "?", tid, kind="question", confidence=1.0, sources=[src])
        for fid in move_questions or []:
            if mem.get_finding(fid):
                mem.move_finding(fid, tid)
        mem.log("topic_added", topic_id=tid, name=name, origin=origin)
        return tid

    def reopen(self, topic_id: int) -> bool:
        t = self.mem.get_topic(topic_id)
        if not t or t["status"] not in ("explored", "retired") or free_slots(self.mem, self.cfg) <= 0:
            return False
        self.mem.update_topic(topic_id, status="active", stage="explore", mastery="")
        self.mem.log("topic_reopened", topic_id=topic_id, name=t["name"])
        return True


# ---- requests from the dashboard (events), carried out by the research service between steps ---------------
REQUESTS = ("mastery_confirm", "mastery_decline", "topic_add", "topic_reopen", "proposal_accept")


def pending_requests(mem) -> list[dict]:
    done = {json.loads(d).get("request") for (d,) in mem.db.execute("SELECT data FROM events WHERE kind = 'request_done'")}
    return [{"id": e["id"], "kind": e["kind"], **json.loads(e["data"])} for e in mem.db.execute(
        f"SELECT id, kind, data FROM events WHERE kind IN ({','.join('?' * len(REQUESTS))}) ORDER BY id", REQUESTS)
        if e["id"] not in done]


def apply_requests(r) -> int:
    """Carry out the dashboard's mastery / topic requests. Returns how many were handled."""
    mem, m, n = r.mem, Mastery(r), 0
    for req in pending_requests(mem):
        ok = False
        try:
            if req["kind"] == "mastery_confirm" and mem.get_topic(req["topic_id"]):
                ok = bool(m.brief(req["topic_id"]))
            elif req["kind"] == "mastery_decline" and mem.get_topic(req["topic_id"]):
                m.guidance(req["topic_id"], req.get("guidance", ""))
                ok = True
            elif req["kind"] == "topic_add":
                ok = m.add_topic(req.get("name", ""), req.get("thesis", ""), req.get("description", ""),
                                 req.get("questions") or []) is not None
            elif req["kind"] == "topic_reopen":
                ok = m.reopen(req["topic_id"])
            elif req["kind"] == "proposal_accept":
                p = next((x for x in mem.notable(unreviewed_only=False) if x["id"] == req["proposal"]), None)
                if p:
                    ref = p["refs"]
                    ok = m.add_topic(ref.get("name", ""), ref.get("thesis", ""), ref.get("description", ""),
                                     origin="spawned" if ref.get("parent") else "surprise", parent=ref.get("parent"),
                                     move_questions=ref.get("questions")) is not None
                    if ok:
                        mem.mark_reviewed(p["id"])
        except Exception as e:                       # e.g. the model unreachable: try again next step
            mem.log("research_error", error=f"request {req['kind']}: {type(e).__name__}: {e}"[:200])
            continue
        mem.log("request_done", request=req["id"], request_kind=req["kind"], ok=ok)
        n += 1
    return n
