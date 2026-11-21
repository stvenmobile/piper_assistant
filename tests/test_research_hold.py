"""Holding research while someone talks to Piper: the hold events, the research side's gate (and
its abortable model call), the Orin's client, "think carefully" and the attention check."""
import json
import sqlite3
import threading
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from piper_brain import deep_think
from piper_brain.research_hold import ResearchHold, just_attention
from piper_research import hold
from piper_research.llm import OllamaChat


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, data TEXT)")
    return c


class Mem:
    """Enough of MemoryStore for the gate: .db and .log()."""
    def __init__(self, con):
        self.db = con

    def log(self, kind, **data):
        with self.db:
            self.db.execute("INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
                            (hold.utc_now().isoformat(), kind, json.dumps(data)))


# ---- the hold events ---------------------------------------------------------------------------
def test_no_hold_by_default(con):
    assert hold.state(con) == {"held": False, "until": None, "request": None, "honoured": False}


def test_request_release_and_honoured(con):
    rid = hold.request(con, 300)
    s = hold.state(con)
    assert s["held"] and s["request"] == rid and not s["honoured"]
    Mem(con).log("hold_honoured", request=rid)
    assert hold.state(con)["honoured"]
    hold.release(con)
    assert not hold.state(con)["held"]
    hold.request(con, 300)                     # a new hold isn't honoured by the old honour
    assert hold.state(con)["held"] and not hold.state(con)["honoured"]


def test_a_hold_expires_without_renewal(con):
    hold.request(con, 60)
    assert hold.state(con, now=hold.utc_now() + timedelta(seconds=61))["held"] is False


def test_gate_waits_while_held_and_says_so(con):
    mem = Mem(con)
    gate = hold.HoldGate(mem, check_s=0.0, sleep=lambda s: hold.release(con))   # released while waiting
    assert gate.wait() is False                # nothing held: no wait
    hold.request(con, 300)
    assert gate.wait() is True
    kinds = [r["kind"] for r in con.execute("SELECT kind FROM events ORDER BY id")]
    assert kinds == ["hold_requested", "hold_honoured", "hold_released", "hold_ended"]


# ---- the research model call is dropped when a hold arrives, and asked again afterwards ----------
class SlowOllama(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        SlowOllama.calls.append(time.monotonic())
        if len(SlowOllama.calls) == 1:
            time.sleep(3)                      # the first call is a long research reply
        body = json.dumps({"message": {"content": json.dumps({"ok": True})}, "eval_count": 5}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass


def test_a_running_call_is_dropped_for_a_hold_and_retried(con):
    SlowOllama.calls = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SlowOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm = OllamaChat(f"http://127.0.0.1:{srv.server_address[1]}", "m", timeout=10)
    mem = Mem(con)
    llm.gate = hold.HoldGate(mem, check_s=0.0, sleep=lambda s: time.sleep(0.05))
    threading.Timer(0.4, lambda: hold.request(con, 300)).start()     # someone says "Piper" mid-call
    threading.Timer(1.5, lambda: hold.release(con)).start()          # ... and the conversation ends
    t = time.monotonic()
    assert llm.json("sys", "user", {"type": "object"}) == {"ok": True}
    assert len(SlowOllama.calls) == 2                                # asked again after the hold
    assert time.monotonic() - t < 2.9                                # didn't sit out the 3 s reply
    srv.shutdown()


def test_without_a_gate_calls_are_as_before():
    assert OllamaChat("http://127.0.0.1:9", "m").gate is None


# ---- the Orin's client ---------------------------------------------------------------------------
class FakeHTTP:
    def __init__(self, state=None):
        self.state = state or {"researching": True, "phase": "explore", "held": False, "honoured": False}
        self.posts = []

    def __call__(self, req, timeout=None):
        url, body = req.full_url, (json.loads(req.data) if req.data else None)
        if body is not None:
            self.posts.append((url, body))
        if url.endswith("/api/research/hold") and body is None:
            out = self.state
        elif url.endswith("/api/research/hold"):
            self.state = {**self.state, "held": True, "honoured": True}
            out = {"ok": True, "id": 7}
        else:
            out = {"ok": True}

        class R:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                pass

            def read(s):
                return json.dumps(out).encode()
        return R()


CFG = {"mind_url": "http://mind:8080", "hold_s": 300, "url": None, "model": "qwen3:30b-a3b"}
LLM = {"base_url": "http://quantum:11434", "model": "llama3.2:latest", "keep_alive": "2h"}


def test_client_holds_swaps_models_and_releases():
    http = FakeHTTP()
    h = ResearchHold(CFG, LLM, opener=http)
    assert h.active()
    assert h.start() and h.held
    assert h.wait_ready(timeout=2, poll_s=0.01)
    urls = [(u, b) for u, b in http.posts]
    assert ("http://quantum:11434/api/generate", {"model": "qwen3:30b-a3b", "keep_alive": 0}) in urls
    assert ("http://quantum:11434/api/generate", {"model": "llama3.2:latest", "keep_alive": "2h"}) in urls
    h.release()
    time.sleep(0.1)
    assert http.posts[-1] == ("http://mind:8080/api/research/release", {}) and not h.held


def test_client_does_nothing_without_piper_mind():
    h = ResearchHold({**CFG, "mind_url": ""}, LLM, opener=FakeHTTP())
    assert not h.active() and not h.start()


def test_research_in_its_pause_is_not_active():
    h = ResearchHold(CFG, LLM, opener=FakeHTTP({"researching": True, "phase": "pause", "held": False}))
    assert not h.active()


# ---- "think carefully" and "can I ask you a question?" -------------------------------------------
@pytest.mark.parametrize("text, q", [
    ("Think carefully: why is the sky blue?", "why is the sky blue?"),
    ("Piper, please think hard about this, what causes tides?", "what causes tides?"),
    ("take your time - is light a wave?", "is light a wave?"),
    ("why is the sky blue?", None),
    ("think carefully", None),
])
def test_think_carefully(text, q):
    assert deep_think.question(text) == q


@pytest.mark.parametrize("text, yes", [
    ("Piper", True), ("Hi Piper.", True), ("Piper, can I ask you a question?", True),
    ("can I ask a question", True), ("are you busy?", True),
    ("what is the capital of France?", False), ("Piper, what time is it?", False),
])
def test_just_attention(text, yes):
    assert just_attention(text) is yes
