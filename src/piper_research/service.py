"""
The research service. Ways to run:
  * the RHYTHM (--session with research.session_hours 0, the default - start_quiet.sh starts it):
    research with no end, in a daily structure (rhythm.py): explore, pause during weekday working hours,
    consolidate before dawn, a daily report at 07:00, a weekly one and topic decisions on Sundays.
  * a SESSION of fixed length (--session HOURS): research for that long, then write the session
    summary (a remark Piper can say) and stop.
  * timed WINDOWS (research.windows, none by default): research inside them, a summary as each
    one closes. Without windows and without --session the service has nothing to do and exits.

    python3 src/piper_research/service.py --session         the rhythm (research.session_hours 0)
    python3 src/piper_research/service.py --session 2       a 2-hour session, then stop
    python3 src/piper_research/service.py                   timed windows (start_piper.sh starts it)
    python3 src/piper_research/service.py --now --once      one cycle now, whatever the time
        --topic "How migrating birds navigate"              ... on this topic
        --cycles 5                                          ... this many cycles, then stop
        --db /tmp/scratch.db --seeds research_seeds.yaml    ... on a scratch database
        --overnight                                         write the summary of the last 12 h
        --rejudge 100                                       judge 100 findings stored before the judge
        --tidy                                              prune open questions to the cap, and mark the
                                                            old cross-topic flags reviewed (one-off)
        --delete-topic "Name"                               remove a topic and everything that belongs to it

Deletion requests from the dashboard are carried out here, between steps: the service holds the
similarity indexes in RAM, so it is the one that deletes (and rebuilds them).

Ctrl+C (or SIGTERM) stops between steps; an unfinished cycle is recorded as 'interrupted'.
"""
import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):                       # run as a script: make src/ importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_brain.config import CONFIG, ROOT_DIR      # noqa: E402
from piper_memory import EmbedError, open_memory      # noqa: E402
from piper_memory.seeds import load_seeds             # noqa: E402
from piper_research import (Interrupted, LLMError, OllamaChat, Researcher, Schedule, Wikipedia,  # noqa: E402
                            WikiError)
from piper_research.mastery import apply_requests       # noqa: E402
from piper_research.rhythm import Curator, Rhythm       # noqa: E402
from piper_research.sep import SEP                      # noqa: E402

stop = False


def _stop(*_):
    global stop
    if stop:                                         # second Ctrl+C: leave now
        raise SystemExit(1)
    stop = True
    print("\n[Research] Stopping after the current step (Ctrl+C again to quit now)")


def nap(seconds: float):
    end = time.monotonic() + seconds
    while not stop and time.monotonic() < end:
        time.sleep(min(1.0, end - time.monotonic()))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pending_deletions(mem) -> list[dict]:
    """Deletion requests from the dashboard not yet carried out or cancelled (newest request per topic)."""
    state = {}
    for e in mem.db.execute("SELECT kind, data FROM events WHERE kind IN ('delete_requested', 'delete_cancelled',"
                            " 'topic_deleted') ORDER BY id"):
        d = json.loads(e["data"])
        state[d.get("topic_id")] = (e["kind"], d)
    return [d for tid, (k, d) in state.items() if k == "delete_requested"
            and mem.db.execute("SELECT 1 FROM topics WHERE id = ?", (tid,)).fetchone()]


def apply_deletions(r, mem) -> int:
    """Carry out pending deletion requests. Returns how many topics were removed."""
    n = 0
    for req in pending_deletions(mem):
        counts = mem.delete_topic(req["topic_id"])
        if r.last_topic == req["topic_id"]:
            r.last_topic = None
        print(f"[Research] Deleted topic {req.get('name')!r}: {counts}")
        n += 1
    return n


def _research_step(r, mem, cfg) -> str:
    """One step (research or synthesis) with the usual error handling: 'ok' | 'stop' | 'retry'."""
    try:
        r.step()
    except Interrupted:
        return "stop"
    except (LLMError, WikiError, EmbedError) as e:
        print(f"[Research] {e} - retrying in 5 min")
        mem.log("research_error", error=str(e))
        nap(300)
        return "retry"
    except LookupError as e:
        print(f"[Research] {e}")
        nap(600)
        return "retry"
    except Exception as e:
        traceback.print_exc()
        mem.log("research_error", error=f"{type(e).__name__}: {e}")
        nap(60)
        return "retry"
    if cfg["rejudge_per_cycle"] and r.unjudged():      # catch up on findings from before the judge
        try:
            r.rejudge(limit=cfg["rejudge_per_cycle"])
        except Interrupted:
            return "stop"
        except (LLMError, EmbedError) as e:
            print(f"[Research] Rejudging failed: {e}")
    return "ok"


def run_rhythm(r, mem, cfg):
    """Research with no end, in the daily rhythm (rhythm.py). Reports are written only for days the
    service was running at report time (no catching up on start)."""
    rh, cur = Rhythm(cfg["rhythm"]), Curator(r)
    started = datetime.now()
    print(f"[Research] Rhythm: {rh.describe()}. Ctrl+C stops")
    state = {"day_start": None, "cycles": 0, "block_t": time.monotonic(), "block_cycles": 0, "phase": None}

    def open_day():
        if state["day_start"] is None:
            state["day_start"] = utc_now()
            r.new_window()
            mem.log("research_started", model=cfg["model"], session_hours=None, rhythm=rh.describe())
            try:
                if n := mem.backfill_embeddings():
                    print(f"[Research] Embedded {n} items stored while Ollama was away")
            except EmbedError:
                pass

    def last_local_date(kind: str) -> str | None:
        row = mem.db.execute("SELECT ts FROM events WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        return datetime.fromisoformat(row[0]).astimezone().date().isoformat() if row else None

    def daily_reported_today(today: str) -> bool:
        return any(datetime.fromisoformat(x["created_at"]).astimezone().date().isoformat() == today
                   for x in mem.reports("daily", limit=1))

    while not stop:
        apply_deletions(r, mem)
        apply_requests(r)
        now = datetime.now()
        today = now.date().isoformat()
        rt = rh.report_time(now)
        if now >= rt and started < rt and not daily_reported_today(today):          # REPORT
            open_day()
            prev = mem.reports("daily", limit=1)
            since = prev[0]["period_end"] if prev else state["day_start"]
            until = utc_now()
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(timespec="seconds")
            decisions = [e for e in mem.events() if e["kind"] == "topic_decision" and e["ts"] >= cutoff]
            try:
                rid = cur.daily_report(since, until, decisions)
                print(f"[Research] Daily report #{rid} written")
                if rh.is_weekly(now):
                    week = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")
                    print(f"[Research] Weekly report #{cur.weekly_report(week, until, decisions)} written")
            except Exception as e:
                traceback.print_exc()
                mem.log("research_error", error=f"report: {type(e).__name__}: {e}"[:200])
            mem.log("research_ended", cycles=state["cycles"], reason="Daily report")
            state["day_start"], state["cycles"] = None, 0
            continue
        phase = rh.phase(now)
        if phase != state["phase"]:
            print(f"[Research] Phase: {phase}")
            mem.log("phase", phase=phase)
            state["phase"] = phase
        if phase == "pause":
            nap(60)
            continue
        if phase == "consolidate":
            if last_local_date("consolidated") != today:
                open_day()
                end = now.replace(hour=rh.consolidate[1] // 60, minute=rh.consolidate[1] % 60, second=0, microsecond=0)
                if end <= now:
                    end += timedelta(days=1)
                print(f"[Research] Consolidating until {end:%H:%M}" + (" (weekly topic decisions too)" if rh.is_weekly(now) else ""))
                try:
                    out = cur.consolidate(time.monotonic() + (end - now).total_seconds(), weekly=rh.is_weekly(now),
                                          should_stop=lambda: stop)
                    print(f"[Research] Consolidated: {out}")
                except Interrupted:
                    break
                except Exception as e:
                    traceback.print_exc()
                    mem.log("research_error", error=f"consolidate: {type(e).__name__}: {e}"[:200])
                    mem.log("consolidated", error=str(e)[:200])
            nap(30)
            continue
        open_day()                                                                  # EXPLORE
        result = _research_step(r, mem, cfg)
        if result == "stop":
            break
        if result == "ok":
            state["cycles"] += 1
            state["block_cycles"] += 1
        if time.monotonic() - state["block_t"] >= rh.block:                         # hourly CHECKPOINT
            cur.checkpoint(state["block_cycles"])
            state["block_t"], state["block_cycles"] = time.monotonic(), 0
        nap(cfg["pause_s"])
    if state["day_start"] is not None:
        mem.log("research_ended", cycles=state["cycles"], reason="Stopped")


def main():
    ap = argparse.ArgumentParser(description="Piper's research loop")
    ap.add_argument("--now", action="store_true", help="ignore the schedule")
    ap.add_argument("--session", type=float, nargs="?", const=-1, default=None, metavar="HOURS",
                    help="research from now for HOURS (default research.session_hours; 0 = no end), then summarise and stop")
    ap.add_argument("--once", action="store_true", help="one cycle, then stop")
    ap.add_argument("--cycles", type=int, default=0, help="stop after this many cycles")
    ap.add_argument("--topic", help="research this topic (by name)")
    ap.add_argument("--db", help="use this database instead of the configured one")
    ap.add_argument("--seeds", help="load these seed topics first")
    ap.add_argument("--overnight", action="store_true", help="write the summary of the last 12 h and stop")
    ap.add_argument("--rejudge", type=int, default=0, help="judge this many unjudged findings and stop")
    ap.add_argument("--tidy", action="store_true", help="prune open questions, mark old cross-topic flags reviewed")
    ap.add_argument("--delete-topic", metavar="NAME", help="remove this topic and everything that belongs to it")
    ap.add_argument("--recheck-stances", action="store_true",
                    help="queue every supports / challenges verdict for the judge again (after a prompt change)")
    args = ap.parse_args()
    if args.once:
        args.cycles = 1

    cfg = CONFIG["research"]
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    mem = open_memory(path=Path(args.db) if args.db else None)
    if args.seeds:
        print(f"[Research] Seeds: {load_seeds(mem, args.seeds)}")
    llm = OllamaChat(cfg["url"] or CONFIG["llm"]["base_url"], cfg["model"], temperature=cfg["temperature"],
                     num_ctx=cfg["num_ctx"], think=cfg["think"], keep_alive=cfg["keep_alive"],
                     timeout=cfg["timeout_s"], max_tokens=cfg["max_tokens"])
    wiki = Wikipedia(ROOT_DIR / "data" / "research" / "wiki")
    sep = SEP(ROOT_DIR / "data" / "research" / "sep") if cfg["sep_pages"] > 0 else None
    r = Researcher(mem, llm, wiki, cfg, should_stop=lambda: stop, sep=sep)
    schedule = Schedule(cfg["windows"], cfg["days"])

    if args.delete_topic:
        match = mem.db.execute("SELECT id FROM topics WHERE name = ?", (args.delete_topic.strip(),)).fetchone()
        if not match:
            print(f"[Research] No topic named {args.delete_topic!r}")
            return
        print(f"[Research] Deleted {args.delete_topic!r}: {mem.delete_topic(match[0])}")
        return
    apply_deletions(r, mem)
    if args.tidy:
        pruned = sum(r.prune_questions(t) for t in mem.topics())
        flags = [n["id"] for n in mem.notable() if n["kind"] == "cross_topic"]
        for nid in flags:
            mem.mark_reviewed(nid)
        print(f"[Research] Tidied: {pruned} open questions pruned, {len(flags)} cross-topic flags marked reviewed")
        return
    if args.recheck_stances:
        with mem.db:
            n = mem.db.execute("UPDATE findings SET judged_by = NULL WHERE kind != 'question'"
                               " AND stance != 'neutral'").rowcount
        print(f"[Research] {n} stance verdicts queued for the judge again")
        return
    if args.rejudge:
        t0 = time.monotonic()
        print(f"[Research] Rejudging up to {args.rejudge} of {r.unjudged()} unjudged findings ...")
        out = r.rejudge(limit=args.rejudge)
        print(f"[Research] {out} in {time.monotonic() - t0:.0f} s; {r.unjudged()} left")
        return
    if args.overnight:
        since = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(timespec="seconds")
        print(f"[Research] Overnight: {r.overnight(since)}")
        return

    session_end = None
    if args.session is not None:
        hours = cfg["session_hours"] if args.session < 0 else args.session
        args.now = True
        if hours <= 0 and not args.cycles:
            print(f"[Research] {cfg['model']} via {llm.url}; memory {mem.path}")
            run_rhythm(r, mem, cfg)
            mem.close()
            return
        session_end = time.monotonic() + hours * 3600 if hours > 0 else None
        if session_end:
            until = datetime.now() + timedelta(hours=hours)
            print(f"[Research] Session: {hours:g} h, until about {until:%H:%M} (the cycle in progress then finishes)")
    elif not args.now and not schedule.windows:
        print("[Research] No research windows set - nothing to do (start_quiet.sh starts a session)")
        return

    print(f"[Research] {cfg['model']} via {llm.url}; schedule {'ignored' if args.now else schedule};"
          f" memory {mem.path}")
    night_start, cycles, announced = None, 0, False

    def finish(why: str):
        try:
            text = r.overnight(night_start)
            print(f"[Research] {why} after {cycles} cycles. Summary: {text}")
        except (LLMError, EmbedError) as e:
            print(f"[Research] {why} after {cycles} cycles; summary failed: {e}")
        mem.log("research_ended", cycles=cycles, reason=why)

    while not stop:
        apply_deletions(r, mem)
        apply_requests(r)
        if session_end is not None and time.monotonic() >= session_end:
            break
        if args.now or schedule.active(datetime.now()):
            if night_start is None:
                night_start = utc_now()
                r.new_window()
                mem.log("research_started", model=cfg["model"],
                        session_hours=(cfg["session_hours"] if args.session < 0 else args.session) if args.session is not None else None)
                print("[Research] Window open - researching")
                try:
                    if n := mem.backfill_embeddings():
                        print(f"[Research] Embedded {n} items stored while Ollama was away")
                except EmbedError:
                    pass
            try:
                r.step(args.topic)
                cycles += 1
            except Interrupted:
                break
            except (LLMError, WikiError, EmbedError) as e:
                print(f"[Research] {e} - retrying in 5 min")
                mem.log("research_error", error=str(e))
                nap(300)
                continue
            except (LookupError, KeyError) as e:
                print(f"[Research] {e}")
                if args.cycles:
                    break
                nap(600)
                continue
            except Exception as e:
                traceback.print_exc()
                mem.log("research_error", error=f"{type(e).__name__}: {e}")
                nap(60)
                continue
            if args.cycles and cycles >= args.cycles:
                break
            if cfg["rejudge_per_cycle"] and r.unjudged():  # catch up on findings from before the judge
                try:
                    out = r.rejudge(limit=cfg["rejudge_per_cycle"])
                    print(f"[Research]   Rejudged {out['judged']} older findings ({out['changed']} changed stance,"
                          f" {out['off_topic']} off-topic); {r.unjudged()} left")
                except Interrupted:
                    break
                except (LLMError, EmbedError) as e:
                    print(f"[Research] Rejudging failed: {e}")
            nap(cfg["pause_s"])
            announced = False
        else:
            if night_start is not None:             # the window just closed
                finish("Window closed")
                night_start, cycles = None, 0
            if not announced:
                nxt = schedule.next_start(datetime.now())
                print(f"[Research] Waiting - next window {nxt:%a %H:%M}" if nxt else "[Research] No window scheduled")
                announced = True
            nap(30)
    if night_start is not None and cycles and not args.cycles:     # a session ended, or Piper stopped
        finish("Session ended" if session_end is not None and not stop else "Stopped")
    elif night_start is not None:                                   # --once / --cycles: just mark the end
        mem.log("research_ended", cycles=cycles, reason="manual run")
    mem.close()


if __name__ == "__main__":
    main()
