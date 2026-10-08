"""
The research service: runs research cycles inside the scheduled windows, and when a window
ends writes the overnight summary (a remark Piper can say in the morning).

    python3 src/piper_research/service.py                   the service (start_piper.sh starts it)
    python3 src/piper_research/service.py --now --once      one cycle now, whatever the time
        --topic "How migrating birds navigate"              ... on this topic
        --cycles 5                                          ... this many cycles, then stop
        --db /tmp/scratch.db --seeds research_seeds.yaml    ... on a scratch database
        --overnight                                         write the summary of the last 12 h
        --rejudge 100                                       judge 100 findings stored before the judge
        --tidy                                              prune open questions to the cap, and mark the
                                                            old cross-topic flags reviewed (one-off)

Ctrl+C (or SIGTERM) stops between steps; an unfinished cycle is recorded as 'interrupted'.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser(description="Piper's research loop")
    ap.add_argument("--now", action="store_true", help="ignore the schedule")
    ap.add_argument("--once", action="store_true", help="one cycle, then stop")
    ap.add_argument("--cycles", type=int, default=0, help="stop after this many cycles")
    ap.add_argument("--topic", help="research this topic (by name)")
    ap.add_argument("--db", help="use this database instead of the configured one")
    ap.add_argument("--seeds", help="load these seed topics first")
    ap.add_argument("--overnight", action="store_true", help="write the summary of the last 12 h and stop")
    ap.add_argument("--rejudge", type=int, default=0, help="judge this many unjudged findings and stop")
    ap.add_argument("--tidy", action="store_true", help="prune open questions, mark old cross-topic flags reviewed")
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
                     num_ctx=cfg["num_ctx"], think=cfg["think"], keep_alive=cfg["keep_alive"])
    wiki = Wikipedia(ROOT_DIR / "data" / "research" / "wiki")
    r = Researcher(mem, llm, wiki, cfg, should_stop=lambda: stop)
    schedule = Schedule(cfg["windows"], cfg["days"])

    if args.tidy:
        pruned = sum(r.prune_questions(t) for t in mem.topics())
        flags = [n["id"] for n in mem.notable() if n["kind"] == "cross_topic"]
        for nid in flags:
            mem.mark_reviewed(nid)
        print(f"[Research] Tidied: {pruned} open questions pruned, {len(flags)} cross-topic flags marked reviewed")
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

    print(f"[Research] {cfg['model']} via {llm.url}; schedule {'ignored' if args.now else schedule};"
          f" memory {mem.path}")
    night_start, cycles, announced = None, 0, False
    while not stop:
        if args.now or schedule.active(datetime.now()):
            if night_start is None:
                night_start = utc_now()
                r.new_window()
                mem.log("research_started", model=cfg["model"])
                print("[Research] Window open - researching")
                try:
                    if n := mem.backfill_embeddings():
                        print(f"[Research] Embedded {n} items stored while Ollama was away")
                except EmbedError:
                    pass
            try:
                r.cycle(args.topic)
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
                try:
                    text = r.overnight(night_start)
                    print(f"[Research] Window closed after {cycles} cycles. Overnight: {text}")
                except (LLMError, EmbedError) as e:
                    print(f"[Research] Overnight summary failed: {e}")
                mem.log("research_ended", cycles=cycles)
                night_start, cycles = None, 0
            if not announced:
                nxt = schedule.next_start(datetime.now())
                print(f"[Research] Waiting - next window {nxt:%a %H:%M}" if nxt else "[Research] No window scheduled")
                announced = True
            nap(30)
    mem.close()


if __name__ == "__main__":
    main()
