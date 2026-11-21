"""
Holding research while someone talks to Piper.

The Orin (main.py, piper_brain/research_hold.py) asks the dashboard on piper-mind to hold research
for a few minutes when Piper is woken, renews the hold while the conversation lasts, and releases
it when the conversation ends. The requests are events in the memory database (like the
dashboard's other requests), so nothing needs a new connection between the two programs:

    hold_requested  {"until": UTC ISO time}     written by the dashboard, renewed every minute
    hold_released   {}                          written by the dashboard
    hold_honoured   {"request": event id}       written by the research service once it has stopped
                                                using the model (an unfinished model call is dropped
                                                and asked again after the hold)
    hold_ended      {}                          written by the research service when it carries on

A hold that isn't renewed simply expires, so a crashed Orin can't leave research paused.
"""
import json
import time
from datetime import datetime, timedelta, timezone

KINDS = ("hold_requested", "hold_released", "hold_honoured", "hold_ended")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def state(db, now: datetime | None = None) -> dict:
    """{"held", "until", "request", "honoured"} from the hold events (db: a sqlite connection)."""
    now = now or utc_now()
    req = rel = hon = None
    for eid, kind, data in db.execute("SELECT id, kind, data FROM events WHERE kind IN (?, ?, ?, ?)"
                                      " ORDER BY id DESC LIMIT 50", KINDS):
        d = json.loads(data)
        if kind == "hold_requested" and req is None:
            req = (eid, d)
        elif kind == "hold_released" and rel is None:
            rel = eid
        elif kind == "hold_honoured" and hon is None:
            hon = eid
    held = bool(req and (rel is None or req[0] > rel) and _ts(req[1]["until"]) > now)
    # honoured: the service stopped after this hold began (after the last release)
    honoured = bool(held and hon is not None and (rel is None or hon > rel))
    return {"held": held, "until": req[1]["until"] if held else None,
            "request": req[0] if req else None, "honoured": honoured}


def request(con, seconds: float) -> int:
    """Hold research for `seconds` from now (a new request renews the hold). Returns the event id."""
    until = (utc_now() + timedelta(seconds=max(10.0, min(float(seconds), 1800.0)))).isoformat(timespec="seconds")
    with con:
        return con.execute("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
                           (utc_now().isoformat(timespec="seconds"), "hold_requested",
                            json.dumps({"until": until}))).lastrowid


def release(con) -> int:
    with con:
        return con.execute("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
                           (utc_now().isoformat(timespec="seconds"), "hold_released", "{}")).lastrowid


class HoldGate:
    """The research service's side: held() is cheap (checked at most once a second); wait() blocks
    while research is held, saying so in the log once per hold."""

    def __init__(self, mem, should_stop=lambda: False, check_s: float = 1.0, sleep=time.sleep):
        self.mem = mem
        self.should_stop = should_stop
        self.check_s = check_s
        self.sleep = sleep
        self._at = 0.0
        self._held = False

    def held(self) -> bool:
        if time.monotonic() - self._at >= self.check_s:
            self._at = time.monotonic()
            try:
                self._held = state(self.mem.db)["held"]
            except Exception:
                self._held = False
        return self._held

    def wait(self) -> bool:
        """Block while held. Returns True if it waited."""
        if not self.held():
            return False
        s = state(self.mem.db)
        print("[Research] Held for a conversation - waiting")
        self.mem.log("hold_honoured", request=s["request"])
        while not self.should_stop():
            self._at = 0.0
            if not self.held():
                break
            self.sleep(1.0)
        print("[Research] Hold over - carrying on")
        self.mem.log("hold_ended")
        return True
