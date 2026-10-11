"""
The dashboard's data: plain functions over a (read-only) MemoryStore, returning JSON-ready dicts.
server.py maps URLs onto them; tests call them directly.
"""
import json
from datetime import datetime, timezone

from piper_research.verify import norm, tidy_remark

CERTAINTY = {0.9: "established", 0.6: "reported", 0.3: "speculative"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_s(iso: str | None) -> float | None:
    return None if not iso else (_now() - datetime.fromisoformat(iso)).total_seconds()


def certainty(conf: float) -> str:
    return CERTAINTY.get(round(conf, 1), f"{conf:.1f}")


# ---- sessions (research_started ... research_ended in the event log) ---------------------------------
def sessions(mem, limit: int = 20) -> list[dict]:
    rows = mem.db.execute("SELECT id, ts, kind, data FROM events WHERE kind IN ('research_started', 'research_ended')"
                          " ORDER BY id").fetchall()
    out, cur = [], None
    for r in rows:
        data = json.loads(r["data"])
        if r["kind"] == "research_started":
            if cur:                                  # started again without an end (Piper stopped)
                out.append(cur)
            cur = {"start": r["ts"], "end": None, "model": data.get("model", ""),
                   "hours": data.get("session_hours"), "reason": None}
        elif cur:
            cur["end"], cur["reason"] = r["ts"], data.get("reason", "ended")
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    last_ev = mem.db.execute("SELECT ts FROM events ORDER BY id DESC LIMIT 1").fetchone()
    for i, s in enumerate(out):
        nxt = out[i + 1]["start"] if i + 1 < len(out) else None
        if s["end"] is None and (nxt or (_age_s(last_ev[0]) or 0) > 900):
            s["end"], s["reason"] = nxt or last_ev[0], "stopped"         # no end logged: Piper was stopped
        end = s["end"] or "9999"
        eps = mem.db.execute("SELECT outcome, stats FROM episodes WHERE started_at >= ? AND started_at <= ?",
                             (s["start"], end)).fetchall()
        tot = {}
        for e in eps:
            for k, v in json.loads(e["stats"]).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tot[k] = tot.get(k, 0) + v
        s["cycles"] = len(eps)
        s["outcomes"] = {o: sum(1 for e in eps if e["outcome"] == o) for o in ("progress", "stalled", "error", "interrupted")}
        s["totals"] = {k: round(v) for k, v in tot.items() if k in (
            "new", "extracted", "rejected", "off_topic", "duplicates", "extra_sources", "supports", "challenges",
            "neutral", "contradicts", "relations", "cross_links", "follow_ups", "seconds")}
        summ = mem.db.execute("SELECT text FROM remarks WHERE kind = 'overnight' AND topic_id IS NULL AND written_at >= ?"
                              " AND written_at <= ? ORDER BY id DESC LIMIT 1", (s["start"], end)).fetchone()
        s["summary"] = tidy_remark(summ["text"]) if summ else ""
        s["active"] = s["end"] is None
    return list(reversed(out))[:limit]


# ---- status (the header) --------------------------------------------------------------------------------
def status(mem) -> dict:
    last = mem.db.execute("SELECT ts, kind FROM events ORDER BY id DESC LIMIT 1").fetchone()
    cur = mem.db.execute("SELECT e.angle, e.started_at, t.name FROM episodes e JOIN topics t ON t.id = e.topic_id"
                         " ORDER BY e.id DESC LIMIT 1").fetchone()
    s = sessions(mem, limit=1)
    session = s[0] if s else None
    left = None
    if session and session["active"] and session.get("hours"):
        left = session["hours"] * 3600 - _age_s(session["start"])
    stats = mem.stats()
    stats["unjudged"] = mem.db.execute("SELECT COUNT(*) FROM findings WHERE judged_by IS NULL AND kind != 'question'").fetchone()[0]
    stats["live_insights"] = mem.db.execute("SELECT COUNT(*) FROM insights WHERE status IN ('new', 'supported', 'weakened')").fetchone()[0]
    ph = mem.db.execute("SELECT data FROM events WHERE kind = 'phase' ORDER BY id DESC LIMIT 1").fetchone()
    return {
        "researching": bool(session and session["active"] and _age_s(last["ts"] if last else None) is not None
                            and _age_s(last["ts"]) < 900),
        "session": session, "seconds_left": left,
        "last_event": {"ts": last["ts"], "kind": last["kind"], "age_s": _age_s(last["ts"])} if last else None,
        "current": {"topic": cur["name"], "question": cur["angle"], "started": cur["started_at"]} if cur else None,
        "stats": stats,
        "phase": json.loads(ph[0]).get("phase") if ph else None,
    }


# ---- live feed ----------------------------------------------------------------------------------------
def events(mem, since: int = 0, limit: int = 150) -> list[dict]:
    if since <= 0:                                   # first load: the latest `limit`, oldest first
        rows = mem.db.execute("SELECT * FROM (SELECT * FROM events ORDER BY id DESC LIMIT ?) ORDER BY id", (limit,))
    else:
        rows = mem.db.execute("SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?", (since, limit))
    topics = {r["id"]: r["name"] for r in mem.db.execute("SELECT id, name FROM topics")}
    out = []
    for r in rows:
        d = json.loads(r["data"])
        if "topic_id" in d:
            d["topic"] = topics.get(d["topic_id"], d.get("topic"))
        out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], **d})
    return out


# ---- topics -------------------------------------------------------------------------------------------
def derived_ids(mem) -> set[int]:
    return {r[0] for r in mem.db.execute("SELECT ft.finding_id FROM finding_tags ft JOIN tags t ON t.id = ft.tag_id"
                                         " WHERE t.name = 'derived'")}


def open_questions(mem, topic_id: int) -> list[str]:
    asked = {norm(r[0]) for r in mem.db.execute("SELECT angle FROM episodes WHERE topic_id = ?", (topic_id,))}
    rows = mem.db.execute("SELECT text FROM findings WHERE topic_id = ? AND kind = 'question' ORDER BY id DESC", (topic_id,))
    return [r[0] for r in rows if r[0].rstrip().endswith("?") and norm(r[0]) not in asked]


def topics(mem) -> list[dict]:
    out = []
    pending = pending_deletes(mem)
    for t in mem.topics():
        tid = t["id"]
        n = mem.db.execute("SELECT COUNT(*) FROM findings WHERE topic_id = ? AND kind != 'question' AND relevance != 'off_topic'",
                           (tid,)).fetchone()[0]
        by_src = {r[0]: r[1] for r in mem.db.execute(
            "SELECT s.kind, COUNT(DISTINCT f.id) FROM findings f JOIN finding_sources fs ON fs.finding_id = f.id"
            " JOIN sources s ON s.id = fs.source_id WHERE f.topic_id = ? AND f.kind != 'question' GROUP BY s.kind", (tid,))}
        last = mem.db.execute("SELECT started_at FROM episodes WHERE topic_id = ? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        eps = mem.db.execute("SELECT COUNT(*) FROM episodes WHERE topic_id = ?", (tid,)).fetchone()[0]
        remarks = {r["kind"]: {"text": tidy_remark(r["text"]), "written_at": r["written_at"]} for r in mem.remarks(tid)}
        oq = open_questions(mem, tid)
        derived = {r[0] for r in mem.db.execute(
            "SELECT f.text FROM findings f JOIN finding_tags ft ON ft.finding_id = f.id JOIN tags g ON g.id = ft.tag_id"
            " WHERE f.topic_id = ? AND g.name = 'derived'", (tid,))}
        out.append({
            "id": tid, "name": t["name"], "thesis": t["thesis"], "description": t["description"], "origin": t["origin"],
            "status": t["status"], "novelty": t["novelty"], "progress": t["progress"], "saturation": t["saturation"],
            "findings": n, "by_source": by_src, "episodes": eps, "last_researched": last[0] if last else None,
            "balance": mem.stance_balance(tid), "remarks": remarks,
            "open_questions": oq[:30], "open_questions_total": len(oq),
            "derived_questions": [q for q in oq if q in derived], "stage": t.get("stage", "explore"),
            "delete_pending": tid in pending,
            "insights": mem.db.execute("SELECT COUNT(*) FROM insights WHERE topic_id = ? AND status IN"
                                       " ('new', 'supported', 'weakened')", (tid,)).fetchone()[0],
        })
    return out


# ---- findings -----------------------------------------------------------------------------------------
def findings(mem, topic: int | None = None, stance: str = "", relevance: str = "", source: str = "",
             q: str = "", kind: str = "claims", judged: str = "", limit: int = 50, offset: int = 0) -> dict:
    where, args = [], []
    if topic:
        where.append("f.topic_id = ?"); args.append(topic)
    if stance:
        where.append("f.stance = ?"); args.append(stance)
    if relevance:
        where.append("f.relevance = ?"); args.append(relevance)
    if kind == "claims":
        where.append("f.kind != 'question'")
    elif kind == "questions":
        where.append("f.kind = 'question'")
    elif kind == "derived":
        where.append("f.kind = 'question' AND EXISTS (SELECT 1 FROM finding_tags ft JOIN tags g ON g.id = ft.tag_id"
                     " WHERE ft.finding_id = f.id AND g.name = 'derived')")
    if judged == "no":
        where.append("f.judged_by IS NULL")
    if source:
        where.append("EXISTS (SELECT 1 FROM finding_sources fs JOIN sources s ON s.id = fs.source_id"
                     " WHERE fs.finding_id = f.id AND s.kind = ?)"); args.append(source)
    if q:
        ids = [r[0] for r in mem.db.execute("SELECT rowid FROM findings_fts WHERE findings_fts MATCH ?", (q,))] \
            if _fts_ok(q) else []
        like = f"%{q}%"
        where.append("(f.text LIKE ? OR f.id IN (%s))" % (",".join(map(str, ids)) or "0")); args.append(like)
    sql_where = (" WHERE " + " AND ".join(where)) if where else ""
    total = mem.db.execute(f"SELECT COUNT(*) FROM findings f{sql_where}", args).fetchone()[0]
    rows = mem.db.execute(f"SELECT f.*, t.name AS topic_name FROM findings f JOIN topics t ON t.id = f.topic_id{sql_where}"
                          " ORDER BY f.id DESC LIMIT ? OFFSET ?", (*args, limit, offset)).fetchall()
    items = []
    derived = derived_ids(mem)
    for r in rows:
        srcs = [dict(s) for s in mem.db.execute(
            "SELECT s.kind, s.title, s.url, s.meta, fs.excerpt FROM finding_sources fs JOIN sources s ON s.id = fs.source_id"
            " WHERE fs.finding_id = ? ORDER BY s.id", (r["id"],))]
        for s in srcs:
            s["revid"] = json.loads(s.pop("meta") or "{}").get("revid", "")
        items.append({
            "id": r["id"], "text": r["text"], "kind": r["kind"], "topic": r["topic_name"], "topic_id": r["topic_id"],
            "stance": r["stance"], "relevance": r["relevance"], "reason": r["stance_reason"],
            "judged_by": r["judged_by"], "certainty": certainty(r["confidence"]), "status": r["status"],
            "relates_to": r["relates_to"], "created_at": r["created_at"], "sources": srcs,
            "tags": ["derived"] if r["id"] in derived else [],
        })
    return {"total": total, "items": items}


def _fts_ok(q: str) -> bool:
    return all(c.isalnum() or c in " -'\"*" for c in q)


# ---- notable: contradictions and cross-topic links ------------------------------------------------
def notable(mem, kind: str = "", show: str = "open", limit: int = 100) -> list[dict]:
    where, args = [], []
    if kind:
        where.append("kind = ?"); args.append(kind)
    if show == "open":
        where.append("reviewed_at IS NULL")
    # night 1's 755 cross-topic flags came before the judge (no "judged" mark): leave them out
    where.append("NOT (kind = 'cross_topic' AND refs NOT LIKE '%\"judged\": true%')")
    rows = mem.db.execute("SELECT * FROM notable WHERE " + " AND ".join(where) + " ORDER BY id DESC LIMIT ?",
                          (*args, limit)).fetchall()
    topics = {r["id"]: r["name"] for r in mem.db.execute("SELECT id, name FROM topics")}
    out = []
    for r in rows:
        refs = json.loads(r["refs"])
        fs = []
        for fid in refs.get("findings", []):
            f = mem.db.execute("SELECT id, text, topic_id, stance FROM findings WHERE id = ?", (fid,)).fetchone()
            if f:
                src = mem.db.execute("SELECT s.title, s.url, s.kind FROM finding_sources fs JOIN sources s ON s.id = fs.source_id"
                                     " WHERE fs.finding_id = ? ORDER BY s.id LIMIT 1", (fid,)).fetchone()
                fs.append({"id": f["id"], "text": f["text"], "topic": topics.get(f["topic_id"], ""), "stance": f["stance"],
                           "source": dict(src) if src else None})
        out.append({"id": r["id"], "kind": r["kind"], "title": r["title"], "created_at": r["created_at"],
                    "reviewed_at": r["reviewed_at"], "findings": fs,
                    "topics": [topics.get(t, "") for t in refs.get("topics", [])]})
    return out


def topic_impact(mem, topic_id: int) -> dict | None:
    t = mem.get_topic(topic_id)
    return {"id": topic_id, "name": t["name"], **mem.topic_impact(topic_id)} if t else None


def pending_deletes(mem) -> set[int]:
    state = {}
    for k, d in mem.db.execute("SELECT kind, data FROM events WHERE kind IN ('delete_requested', 'delete_cancelled',"
                               " 'topic_deleted') ORDER BY id"):
        state[json.loads(d).get("topic_id")] = k
    return {tid for tid, k in state.items() if k == "delete_requested"}


def request_delete(db_path, topic_id: int, cancel: bool = False) -> bool:
    """Ask the research service to delete a topic (or cancel the request). The service does the
    deleting between steps - it holds the similarity indexes in RAM."""
    import sqlite3
    con = sqlite3.connect(db_path, timeout=10)
    try:
        row = con.execute("SELECT name FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            return False
        with con:
            con.execute("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
                        (_now().isoformat(timespec="seconds"), "delete_cancelled" if cancel else "delete_requested",
                         json.dumps({"topic_id": topic_id, "name": row[0]})))
        return True
    finally:
        con.close()


def mark_reviewed(db_path, notable_id: int) -> bool:
    """The dashboard's one write: stamp a flag as reviewed. A short-lived connection of its own
    (the dashboard's store is read-only); SQLite's WAL lets it slip in beside a research session."""
    import sqlite3
    con = sqlite3.connect(db_path, timeout=10)
    try:
        with con:
            n = con.execute("UPDATE notable SET reviewed_at = ? WHERE id = ? AND reviewed_at IS NULL",
                            (_now().isoformat(timespec="seconds"), notable_id)).rowcount
        return n > 0
    finally:
        con.close()


# ---- the concept graph ------------------------------------------------------------------------------
def graph(mem, topic: int | None = None, limit: int = 120, vague: bool = False) -> dict:
    """The best-connected concepts and the relations between them. topic: only relations from that
    topic's findings. vague: include 'related_to' links (most of night 1's)."""
    where, args = ["f.relevance != 'off_topic'"], []
    if topic:
        where.append("f.topic_id = ?"); args.append(topic)
    if not vague:
        where.append("r.predicate != 'related_to'")
    rows = mem.db.execute(
        "SELECT r.subject_id, r.predicate, r.object_id, COUNT(*) AS n FROM relations r JOIN findings f ON f.id = r.finding_id"
        " WHERE " + " AND ".join(where) + " GROUP BY r.subject_id, r.predicate, r.object_id", args).fetchall()
    degree = {}
    for r in rows:
        for c in (r["subject_id"], r["object_id"]):
            degree[c] = degree.get(c, 0) + r["n"]
    keep = set(sorted(degree, key=lambda c: -degree[c])[:limit])
    names = {r["id"]: r["name"] for r in mem.db.execute("SELECT id, name FROM concepts")}
    edges = [{"s": r["subject_id"], "p": r["predicate"], "o": r["object_id"], "n": r["n"]} for r in rows
             if r["subject_id"] in keep and r["object_id"] in keep and r["subject_id"] != r["object_id"]]
    linked = {e["s"] for e in edges} | {e["o"] for e in edges}
    nodes = [{"id": c, "name": names.get(c, "?"), "degree": degree[c]} for c in keep if c in linked]
    return {"nodes": nodes, "edges": edges, "total_concepts": len(degree)}


def concept(mem, concept_id: int) -> dict | None:
    c = mem.get_concept(concept_id)
    if not c:
        return None
    rels = mem.db.execute(
        "SELECT r.predicate, cs.name AS s, co.name AS o, f.id AS fid, f.text, t.name AS topic"
        " FROM relations r JOIN concepts cs ON cs.id = r.subject_id JOIN concepts co ON co.id = r.object_id"
        " JOIN findings f ON f.id = r.finding_id JOIN topics t ON t.id = f.topic_id"
        " WHERE (r.subject_id = ? OR r.object_id = ?) AND f.relevance != 'off_topic' ORDER BY f.id DESC LIMIT 60",
        (concept_id, concept_id)).fetchall()
    return {"id": c["id"], "name": c["name"], "aliases": c["aliases"],
            "relations": [{"subject": r["s"], "predicate": r["predicate"], "object": r["o"], "finding": r["text"],
                           "finding_id": r["fid"], "topic": r["topic"]} for r in rels]}


# ---- insights: what synthesis concluded, and the trail that led there ------------------------------------
def _finding_card(mem, fid: int, topics: dict) -> dict | None:
    f = mem.db.execute("SELECT * FROM findings WHERE id = ?", (fid,)).fetchone()
    if not f:
        return {"id": fid, "missing": True}
    src = mem.db.execute("SELECT s.kind, s.title, s.url, s.meta, fs.excerpt FROM finding_sources fs JOIN sources s"
                         " ON s.id = fs.source_id WHERE fs.finding_id = ? ORDER BY s.id LIMIT 1", (fid,)).fetchone()
    source = None
    if src:
        source = {"kind": src["kind"], "title": src["title"], "url": src["url"], "excerpt": src["excerpt"],
                  "revid": json.loads(src["meta"] or "{}").get("revid", "")}
    return {"id": fid, "text": f["text"], "topic": topics.get(f["topic_id"], ""), "stance": f["stance"],
            "relevance": f["relevance"], "certainty": certainty(f["confidence"]), "source": source}


def insights(mem, status: str = "live", topic: int | None = None, limit: int = 200) -> list[dict]:
    where, args = [], []
    if status == "live":
        where.append("status IN ('new', 'supported', 'weakened')")
    elif status and status != "all":
        where.append("status = ?"); args.append(status)
    if topic:
        where.append("topic_id = ?"); args.append(topic)
    rows = mem.db.execute("SELECT id FROM insights" + (" WHERE " + " AND ".join(where) if where else "")
                          + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
    topics = {r["id"]: r["name"] for r in mem.db.execute("SELECT id, name FROM topics")}
    out = []
    for (iid,) in rows:
        i = mem.get_insight(iid)
        ev = [e for e in i["events"] if e["kind"] == "evidence"]
        out.append({"id": i["id"], "statement": i["statement"], "kind": i["kind"], "confidence": i["confidence"],
                    "status": i["status"], "topic": topics.get(i["topic_id"], ""), "created_at": i["created_at"],
                    "updated_at": i["updated_at"], "basis": len(i["basis"]), "questions": len(i["questions"]),
                    "evidence": len(ev), "verdict": i["trace"].get("grounding", {}).get("verdict", ""),
                    "strategy": i["trace"].get("selection", {}).get("strategy", "")})
    return out


def insight(mem, insight_id: int) -> dict | None:
    """One insight's full trail: why that material, the material (each finding back to its source),
    the reasoning steps and their checks, the grounding check, the questions it raised (and whether
    research has taken them up), its later history, and the model's raw thinking."""
    i = mem.get_insight(insight_id)
    if not i:
        return None
    topics = {r["id"]: r["name"] for r in mem.db.execute("SELECT id, name FROM topics")}
    tr = i["trace"]
    sel = tr.get("selection", {})
    material = [_finding_card(mem, fid, topics) for fid in sel.get("findings", i["basis"])]
    for m in material:
        m["basis"] = m["id"] in i["basis"]
    questions = []
    for qid in i["questions"]:
        q = mem.db.execute("SELECT id, text, topic_id FROM findings WHERE id = ?", (qid,)).fetchone()
        if not q:
            continue
        ep = mem.db.execute("SELECT id, started_at, outcome, stats FROM episodes WHERE angle = ? AND kind = 'research'"
                            " ORDER BY id LIMIT 1", (q["text"],)).fetchone()
        questions.append({"id": q["id"], "text": q["text"], "topic": topics.get(q["topic_id"], ""),
                          "researched": {"episode": ep["id"], "when": ep["started_at"], "outcome": ep["outcome"],
                                         "new": json.loads(ep["stats"]).get("new", 0)} if ep else None})
    events = []
    for e in i["events"]:
        e = dict(e)
        if e.get("findings"):
            e["finding_cards"] = [_finding_card(mem, f, topics) for f in e["findings"]]
        events.append(e)
    ep = mem.db.execute("SELECT id, started_at FROM episodes WHERE id = ?", (i["episode_id"],)).fetchone() if i["episode_id"] else None
    return {"id": i["id"], "statement": i["statement"], "kind": i["kind"], "confidence": i["confidence"],
            "status": i["status"], "topic": topics.get(i["topic_id"], ""), "created_at": i["created_at"],
            "episode": dict(ep) if ep else None,
            "selection": {"strategy": sel.get("strategy", ""), "why": sel.get("why", "")},
            "material": material, "steps": tr.get("steps", []), "grounding": tr.get("grounding", {}),
            "stray_citations": tr.get("stray_citations", []), "thinking": tr.get("thinking", ""),
            "model": tr.get("model", ""), "prompt_version": tr.get("prompt_version"),
            "questions": questions, "events": events}


# ---- reports -------------------------------------------------------------------------------------------
def reports(mem, kind: str = "", limit: int = 60) -> list[dict]:
    return [{"id": r["id"], "kind": r["kind"], "period_start": r["period_start"], "period_end": r["period_end"],
             "created_at": r["created_at"], "title": r["text"].splitlines()[0].lstrip("# ") if r["text"] else ""}
            for r in mem.reports(kind or None, limit=limit)]


def report(mem, report_id: int) -> dict | None:
    r = mem.db.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return {**dict(r), "data": json.loads(r["data"])} if r else None
