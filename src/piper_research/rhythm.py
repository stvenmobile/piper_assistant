"""
Piper's research RHYTHM - research with no end, but in a structure: explore, pause, consolidate,
report. Local time; the defaults (config research.rhythm):

    05:00-07:00   CONSOLIDATE  no new reading: reflect on every topic researched since yesterday,
                               a synthesis pass over every topic with enough material, prune stale
                               questions, review insights; on the weekly day, the TOPIC DECISIONS
                               (retire, rest, wake, spawn a sub-topic, propose a new topic). Time
                               left over goes to the re-judging backlog
    07:00         REPORT       the daily report (and the weekly one on the weekly day)
    10:00-17:00   PAUSE        the PC's GPU is left alone - weekdays only (pause_days)
    otherwise     EXPLORE      research and synthesis cycles (synthesis.py decides which), with a
                               CHECKPOINT every hour: topic stages updated, logged

Rhythm says which phase a moment is in; Curator does the checkpoint / consolidation / report /
topic-decision work with a Researcher. service.py runs the loop.
"""
import json
import time
from datetime import datetime, timedelta, timezone

from piper_research import prompts as P
from piper_research.llm import LLMError
from piper_research.mastery import Mastery, free_slots
from piper_research.schedule import DAYS, parse_window
from piper_research.verify import norm, tidy_remark


def _in(window, now: datetime) -> bool:
    """window: (start, end) in minutes after midnight (parse_window); it may cross midnight."""
    a, b = window
    t = now.hour * 60 + now.minute
    return a <= t < b if a < b else (t >= a or t < b)


class Rhythm:
    def __init__(self, cfg: dict):
        self.pause = [parse_window(w) for w in cfg.get("pause", [])]
        days = cfg.get("pause_days", "weekdays")
        if isinstance(days, str):
            days = {"weekdays": DAYS[:5], "daily": DAYS}.get(days, [d.strip() for d in days.split(",")])
        bad = [d for d in days if d[:3].lower() not in DAYS]
        if bad:
            raise ValueError(f"unknown pause days {bad} (use mon..sun, 'weekdays' or 'daily')")
        self.pause_days = {DAYS.index(d[:3].lower()) for d in days}
        self.consolidate = parse_window(cfg.get("consolidate", "05:00-07:00"))
        h, m = map(int, cfg.get("report_at", "07:00").split(":"))
        self.report_at = (h, m)
        self.weekly_day = DAYS.index(cfg.get("weekly_day", "sun")[:3].lower())
        self.block = cfg.get("block_minutes", 60) * 60

    def phase(self, now: datetime) -> str:
        if _in(self.consolidate, now):
            return "consolidate"
        if any(_in(w, now) and self._pause_day(w, now) for w in self.pause):
            return "pause"
        return "explore"

    def _pause_day(self, window, now: datetime) -> bool:
        """A pause window belongs to the day it starts on (one crossing midnight runs into the next)."""
        a, _ = window
        started_yesterday = now.hour * 60 + now.minute < a
        return (now.weekday() - started_yesterday) % 7 in self.pause_days

    def report_time(self, now: datetime) -> datetime:
        """Today's report time."""
        return now.replace(hour=self.report_at[0], minute=self.report_at[1], second=0, microsecond=0)

    def is_weekly(self, now: datetime) -> bool:
        return now.weekday() == self.weekly_day

    def describe(self) -> str:
        f = lambda w: f"{w[0] // 60:02d}:{w[0] % 60:02d}-{w[1] // 60:02d}:{w[1] % 60:02d}"
        return (f"consolidate {f(self.consolidate)}, report {self.report_at[0]:02d}:{self.report_at[1]:02d},"
                f" pause {', '.join(f(w) for w in self.pause) or 'none'}"
                f" ({'weekdays' if self.pause_days == set(range(5)) else ', '.join(DAYS[d] for d in sorted(self.pause_days))}),"
                f" weekly on {DAYS[self.weekly_day]}")


def _local_date(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().date().isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Curator:
    """Checkpoints, consolidation, reports and topic decisions, for a Researcher `r`."""

    def __init__(self, r):
        self.r = r
        self.mem = r.mem
        self.cfg = r.cfg

    # ---- topic stages -----------------------------------------------------------------------------
    def open_derived(self, topic_id: int) -> list[dict]:
        asked = self.r.asked(topic_id)
        return [dict(x) for x in self.mem.db.execute(
            "SELECT f.id, f.text FROM findings f JOIN finding_tags ft ON ft.finding_id = f.id JOIN tags t ON t.id = ft.tag_id"
            " WHERE f.topic_id = ? AND f.kind = 'question' AND t.name = 'derived' ORDER BY f.id", (topic_id,))
            if norm(x["text"]) not in asked]

    def stage_of(self, t: dict) -> str:
        if t["status"] in ("retired", "explored"):
            return t["status"]
        if t["status"] == "resting":
            return "mature"
        if (t["saturation"] or 0) < self.cfg.get("synth_saturation", 0.5):
            return "explore"
        return "test" if len(self.open_derived(t["id"])) >= 3 else "synthesize"

    def update_stages(self) -> dict:
        out = {}
        for t in self.mem.topics():
            st = self.stage_of(t)
            if st != t.get("stage"):
                self.mem.update_topic(t["id"], stage=st)
            out[t["name"]] = st
        return out

    def checkpoint(self, cycles: int) -> dict:
        stages = self.update_stages()
        requested = Mastery(self.r).check()                 # any topic ready for mastery: ask for it
        self.mem.log("checkpoint", cycles=cycles, stages=stages, unjudged=self.r.unjudged(), mastery_requested=requested)
        return stages

    # ---- consolidation ----------------------------------------------------------------------------
    def prune_derived(self, topic: dict) -> int:
        """Derived questions beyond the cap: the oldest go."""
        cap = self.cfg.get("max_derived_open", 10)
        extra = self.open_derived(topic["id"])[:-cap] if cap else []
        for q in extra:
            self.mem.delete_finding(q["id"])
        return len(extra)

    def consolidate(self, deadline: float, weekly: bool = False, should_stop=lambda: False) -> dict:
        """The daily consolidation (no new reading). deadline: time.monotonic() to stop by."""
        r, mem = self.r, self.mem
        out = {"reflected": 0, "synthesized": 0, "insights": 0, "pruned": 0, "reviewed": 0, "decisions": [],
               "rejudged": 0}
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
        topics = mem.current_topics()                        # deeply explored topics are done

        def time_left():
            return not should_stop() and time.monotonic() < deadline

        recent = {row[0] for row in mem.db.execute("SELECT DISTINCT topic_id FROM episodes WHERE started_at >= ?", (since,))}
        for t in topics:                                     # 1. reflect on what was researched
            if not time_left():
                break
            if t["id"] in recent:
                try:
                    r.reflect(t["id"])
                    out["reflected"] += 1
                except LLMError as e:
                    mem.log("research_error", error=f"reflect: {e}"[:200])
        for t in topics:                                     # 2. a synthesis pass over every topic
            if not time_left():
                break
            n = mem.db.execute("SELECT COUNT(*) FROM findings WHERE topic_id = ? AND kind != 'question'"
                               " AND relevance != 'off_topic'", (t["id"],)).fetchone()[0]
            if n < self.cfg.get("synth_min_findings", 15):
                continue
            try:
                res = r.synthesize(mem.get_topic(t["id"]))
                if res:
                    out["synthesized"] += 1
                    out["insights"] += res.get("insights", 0)
            except LLMError as e:
                mem.log("research_error", error=f"synthesis: {e}"[:200])
        for t in topics:                                     # 3. prune stale and surplus questions
            out["pruned"] += r.prune_questions(t) + self.prune_derived(t)
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")
        for ins in mem.insights(status="new", limit=500):    # 4. insights untested for a week fade a little
            if ins["created_at"] < week_ago and not any(e["kind"] == "evidence" for e in ins["events"]) \
                    and not any(e["kind"] == "review" and e["ts"] >= week_ago for e in ins["events"]):
                mem.update_insight(ins["id"], confidence=ins["confidence"] - 0.05)
                mem.insight_event(ins["id"], "review", note="a week with no evidence either way",
                                  confidence_before=ins["confidence"])
                out["reviewed"] += 1
        if weekly and time_left():                           # 5. the weekly topic decisions
            out["decisions"] = self.topic_decisions()
        self.update_stages()
        out["mastery_requested"] = Mastery(r).check()
        while time_left() and r.unjudged():                  # 6. leftover time: the re-judging backlog
            try:
                out["rejudged"] += r.rejudge(limit=10)["judged"]
            except LLMError:
                break
        mem.log("consolidated", **{k: v for k, v in out.items() if k != "decisions"},
                decisions=len(out["decisions"]), weekly=weekly)
        return out

    # ---- weekly topic decisions -------------------------------------------------------------------
    def _decide(self, decisions: list, action: str, topic: str, reason: str, **extra):
        d = {"action": action, "topic": topic, "reason": reason, **extra}
        decisions.append(d)
        self.mem.log("topic_decision", **d)

    def _clusters(self, questions: list[dict], min_score: float = 0.75) -> list[list[dict]]:
        """Greedy clusters of similar questions (by embedding)."""
        vecs = {q["id"]: self.mem.finding_vector(q["id"]) for q in questions}
        left = [q for q in questions if vecs[q["id"]] is not None]
        out = []
        while left:
            seed = left.pop(0)
            group = [seed] + [q for q in left if float(vecs[q["id"]] @ vecs[seed["id"]]) >= min_score]
            left = [q for q in left if q not in group]
            out.append(group)
        return out

    def topic_decisions(self) -> list[dict]:
        r, mem, decisions = self.r, self.mem, []
        now = datetime.now(timezone.utc)
        m = Mastery(r)
        for t in mem.current_topics():
            eps = [e for e in mem.episodes(t["id"], limit=20) if e.get("kind", "research") == "research"][:10]
            n = mem.db.execute("SELECT COUNT(*) FROM findings WHERE topic_id = ? AND kind != 'question'"
                               " AND relevance != 'off_topic'", (t["id"],)).fetchone()[0]
            open_q = r.open_questions(t["id"])
            stalled = sum(1 for e in eps if e["outcome"] == "stalled")
            if t["status"] == "resting":
                if (now - datetime.fromisoformat(t["updated_at"])).days >= 7:
                    mem.update_topic(t["id"], status="active")
                    self._decide(decisions, "wake", t["name"], "rested for a week - worth another look")
                continue
            if len(eps) >= 6 and stalled / len(eps) >= 0.7:
                mem.update_topic(t["id"], status="resting")
                self._decide(decisions, "rest", t["name"], f"{stalled} of its last {len(eps)} research cycles stalled")
                continue
            for group in self._clusters(self.open_derived(t["id"]), self.cfg.get("spawn_similarity", 0.75)):   # spawn
                if len(group) < 3:
                    continue
                try:
                    s = r.llm.json(P.SPAWN, f"Parent topic: {t['name']}\nQuestions:\n" + P.numbered([q["text"] for q in group]),
                                   P.SPAWN_SCHEMA)
                except LLMError:
                    break
                name = " ".join(s.get("name", "").split())
                why = f"{len(group)} derived questions in {t['name']} kept pointing the same way"
                if m.propose(name, s.get("description", ""), why, thesis=s.get("thesis", "") or "", parent=t["id"],
                             questions=[q["id"] for q in group]):
                    self._decide(decisions, "propose sub-topic", name, why, parent=t["name"], questions=len(group))
                    break                                                      # one proposal per topic a week
        week_ago = (now - timedelta(days=7)).isoformat(timespec="seconds")
        strong = [i for i in mem.insights(limit=200) if i["created_at"] >= week_ago
                  and i["status"] in ("new", "supported") and i["confidence"] >= 0.5][:8]
        if len(strong) >= 3 and free_slots(mem, self.cfg) > 0:                # propose a new topic
            names = [t["name"] for t in mem.topics()]
            try:
                p = r.llm.json(P.PROPOSE, "Current topics:\n" + P.numbered(names) + "\n\nStrongest insights this week:\n"
                               + P.numbered([i["statement"] for i in strong]), P.PROPOSE_SCHEMA)
            except LLMError:
                p = {}
            name = " ".join(p.get("name", "").split())
            if p.get("propose") and m.propose(name, p.get("description", ""), p.get("why", ""),
                                              insights=[i["id"] for i in strong]):
                self._decide(decisions, "propose", name, p.get("why", ""), insights=[i["id"] for i in strong])
        return decisions

    # ---- reports ----------------------------------------------------------------------------------
    def _balances(self) -> dict:
        return {t["name"]: {k: v["count"] for k, v in self.mem.stance_balance(t["id"]).items()}
                for t in self.mem.topics() if t["thesis"]}

    def daily_report(self, since: str, until: str, decisions: list | None = None) -> int:
        mem, r = self.mem, self.r
        eps = mem.db.execute("SELECT e.*, t.name AS topic FROM episodes e JOIN topics t ON t.id = e.topic_id"
                             " WHERE e.started_at >= ? AND e.started_at < ?", (since, until)).fetchall()
        research = [e for e in eps if (e["kind"] or "research") == "research"]
        synth = [e for e in eps if e["kind"] == "synthesis"]
        new_findings = sum(json.loads(e["stats"]).get("new", 0) for e in research)
        errors = mem.db.execute("SELECT COUNT(*) FROM events WHERE kind = 'research_error' AND ts >= ? AND ts < ?",
                                (since, until)).fetchone()[0]
        made = [i for i in mem.insights(limit=500) if since <= i["created_at"] < until]
        good = sorted([i for i in made if i["status"] != "rejected"], key=lambda i: -i["confidence"])
        tname = {t["id"]: t["name"] for t in mem.topics()}
        changed = []
        for (iid, d, ts) in mem.db.execute("SELECT insight_id, data, ts FROM insight_events WHERE kind = 'evidence'"
                                           " AND ts >= ? AND ts < ?", (since, until)):
            d = json.loads(d)
            if d.get("effect") in ("supports", "weakens"):
                ins = mem.get_insight(iid)
                changed.append({"id": iid, "statement": ins["statement"], "effect": d["effect"],
                                "before": d.get("confidence_before"), "after": d.get("confidence_after")})
        bal = self._balances()
        prev = mem.reports("daily", limit=1)
        prev_bal = prev[0]["data"].get("balances", {}) if prev else {}
        stuck = []
        for t in mem.topics():
            mine = [e for e in research if e["topic_id"] == t["id"]]
            st = sum(1 for e in mine if e["outcome"] == "stalled")
            if t["status"] != "retired" and ((t["saturation"] or 0) >= 0.6 or (len(mine) >= 3 and st / len(mine) >= 0.5)):
                stuck.append({"topic": t["name"], "saturation": t["saturation"], "stalled": st, "cycles": len(mine)})
        nxt = {t["name"]: [q["text"] for q in self.open_derived(t["id"])][-3:] for t in mem.topics()
               if t["status"] != "retired" and self.open_derived(t["id"])}
        data = {"cycles": len(research), "synthesis": len(synth), "new_findings": new_findings, "errors": errors,
                "outcomes": {o: sum(1 for e in eps if e["outcome"] == o) for o in ("progress", "stalled", "error")},
                "insights": [{"id": i["id"], "statement": i["statement"], "kind": i["kind"], "confidence": i["confidence"],
                              "topic": tname.get(i["topic_id"], "")} for i in good],
                "rejected_insights": sum(1 for i in made if i["status"] == "rejected"),
                "changed": changed, "balances": bal, "stuck": stuck, "next": nxt,
                "unjudged": r.unjudged(), "decisions": decisions or []}
        narrative = {"thinking": "", "stuck": "", "next": "", "spoken": ""}
        top = [i for i in mem.insights(limit=300) if i["status"] in ("new", "supported")]
        top = sorted(top, key=lambda i: -i["confidence"])[:6]
        if top or eps:
            try:
                narrative = r.llm.json(P.DAILY, (
                    "Your strongest current insights:\n" + P.numbered([f"[{i['kind']}, {i['confidence']:.1f}] {i['statement']}" for i in top])
                    + "\n\nChanged today:\n" + P.numbered([f"{c['effect']}: {c['statement']}" for c in changed])
                    + "\n\nStuck:\n" + P.numbered([f"{s['topic']} ({s['stalled']} of {s['cycles']} cycles stalled)" for s in stuck])
                    + "\n\nDerived questions waiting:\n" + P.numbered([q for qs in nxt.values() for q in qs])
                    + f"\n\nToday: {len(research)} research cycles ({new_findings} new findings), {len(synth)} synthesis cycles."),
                    P.DAILY_SCHEMA)
            except LLMError as e:
                mem.log("research_error", error=f"daily report: {e}"[:200])
        day = datetime.fromisoformat(until).astimezone()
        lines = [f"# Daily report - {day:%A %d %B %Y}", "", "## What I'm starting to think",
                 tidy_remark(narrative.get("thinking", "")) or "_(no narrative this time)_", ""]
        for i in data["insights"][:5]:
            lines.append(f"- **[{i['kind']}]** {i['statement']} _(confidence {i['confidence']:.2f}, {i['topic']}; insight #{i['id']})_")
        lines += ["", "## What changed"]
        for name, b in bal.items():
            p = prev_bal.get(name)
            delta = "" if not p else (f" (was {p.get('supports', 0)} / {p.get('challenges', 0)})"
                                      if (p.get("supports"), p.get("challenges")) != (b["supports"], b["challenges"]) else " (unchanged)")
            lines.append(f"- {name}: {b['supports']} supporting, {b['challenges']} challenging{delta}")
        for c in changed:
            lines.append(f"- Insight #{c['id']} **{c['effect']}** by new research: {c['statement']}"
                         + (f" ({c['before']:.2f} -> {c['after']:.2f})" if c.get("before") is not None and c.get("after") is not None else ""))
        lines += ["", "## Where I'm stuck", tidy_remark(narrative.get("stuck", "")) or "_(nothing stuck)_"]
        for s in stuck:
            lines.append(f"- {s['topic']}: saturation {s['saturation'] or 0:.0%}"
                         + (f", {s['stalled']} of {s['cycles']} research cycles stalled" if s["cycles"] else ""))
        lines += ["", "## What's next", tidy_remark(narrative.get("next", "")) or ""]
        for name, qs in nxt.items():
            for q in qs:
                lines.append(f"- _{name}_: {q} `derived`")
        mastery = []
        for t in mem.topics():
            if t.get("mastery") == "candidate":
                mastery.append(f"- **{t['name']}** - I'm asking for mastery: every item on its checklist is met."
                               " Please confirm it, or tell me what else needs exploring.")
        for (d,) in mem.db.execute("SELECT data FROM events WHERE kind = 'topic_mastered' AND ts >= ? AND ts < ?", (since, until)):
            mastery.append(f"- **Mastered:** {json.loads(d)['name']} - now deeply explored, with its mastery brief.")
        if mastery:
            lines += ["", "## Mastery"] + mastery
        if decisions:
            lines += ["", "## Topic decisions"] + [f"- **{d['action']}** {d['topic']} - {d['reason']}" for d in decisions]
        lines += ["", "## Housekeeping",
                  f"- {len(research)} research cycles ({new_findings} new findings), {len(synth)} synthesis cycles,"
                  f" {len(made) - data['rejected_insights']} new insights ({data['rejected_insights']} rejected), {errors} errors",
                  f"- Re-judging backlog: {data['unjudged']} findings"]
        rid = mem.add_report("daily", since, until, "\n".join(lines), data)
        spoken = tidy_remark(narrative.get("spoken", ""))
        if spoken:
            mem.set_remark("overnight", spoken)
        mem.log("report", report=rid, report_kind="daily", spoken=spoken)
        return rid

    def weekly_report(self, since: str, until: str, decisions: list | None = None) -> int:
        mem, r = self.mem, self.r
        dailies = [d for d in mem.reports("daily", limit=10) if d["period_start"] >= since]
        first = dailies[-1]["data"].get("balances", {}) if dailies else {}
        bal = self._balances()
        held = [i for i in mem.insights(limit=500) if i["status"] == "supported" and i["updated_at"] >= since]
        fell = [i for i in mem.insights(limit=500) if i["status"] in ("weakened", "dropped") and i["updated_at"] >= since]
        made = [i for i in mem.insights(limit=500) if i["created_at"] >= since and i["status"] != "rejected"]
        try:
            ov = r.llm.json(P.WEEKLY, (
                "Insights this week:\n" + P.numbered([i["statement"] for i in made][:12])
                + "\n\nHeld up:\n" + P.numbered([i["statement"] for i in held]) + "\n\nFell:\n" + P.numbered([i["statement"] for i in fell])
                + "\n\nThesis balances (start -> now):\n" + P.numbered([f"{n}: {first.get(n, {})} -> {b}" for n, b in bal.items()])
                + "\n\nTopic decisions:\n" + P.numbered([f"{d['action']} {d['topic']}: {d['reason']}" for d in decisions or []])),
                P.WEEKLY_SCHEMA).get("overview", "")
        except LLMError:
            ov = ""
        day = datetime.fromisoformat(until).astimezone()
        lines = [f"# Weekly report - week to {day:%d %B %Y}", "", "## State of my mind", tidy_remark(ov) or "_(no overview)_", "",
                 "## Topics"]
        for t in mem.topics():
            b = bal.get(t["name"])
            f = first.get(t["name"])
            lines.append(f"- **{t['name']}** ({t['status']}, stage {t.get('stage', '')})"
                         + (f": {b['supports']} supporting / {b['challenges']} challenging" if b else "")
                         + (f", a week ago {f.get('supports', 0)} / {f.get('challenges', 0)}" if b and f else ""))
        lines += ["", f"## Insights - {len(made)} new this week"]
        lines += [f"- {i['statement']} _(#{i['id']}, {i['confidence']:.2f})_" for i in sorted(made, key=lambda i: -i["confidence"])[:10]]
        lines += ["", "## Held up"] + ([f"- {i['statement']} _(#{i['id']})_" for i in held] or ["- _(none tested yet)_"])
        lines += ["", "## Fell"] + ([f"- {i['statement']} _(#{i['id']}, {i['status']})_" for i in fell] or ["- _(none)_"])
        lines += ["", "## Topic decisions"] + ([f"- **{d['action']}** {d['topic']} - {d['reason']}" for d in decisions or []]
                                               or ["- _(no changes)_"])
        rid = mem.add_report("weekly", since, until, "\n".join(lines),
                             {"balances": bal, "first": first, "held": [i["id"] for i in held],
                              "fell": [i["id"] for i in fell], "decisions": decisions or []})
        mem.log("report", report=rid, report_kind="weekly")
        return rid
