"""
Pausing research while someone talks to Piper (the Orin's side of piper_research/hold.py).

    hold = ResearchHold()
    hold.active()        is research running right now? (asks piper-mind's dashboard, ~1 s at most)
    hold.start()         hold it (renewed every minute in the background while held)
    hold.wait_ready()    wait until research has let go of the model, then swap the models on
                         the Ollama server: unload the research model, load Piper's chat model
    hold.release()       the conversation is over: research carries on

Research and conversation share one Ollama (QUANTUM's RTX 5070) and its 12 GB can't hold both
models, so without this every question first waited for the research model to be swapped out.
Never raises; if piper-mind can't be reached, it does nothing and Piper answers as before.
"""
import json
import re
import threading
import time
import urllib.request

from piper_brain.config import CONFIG

RENEW_S = 60.0

# what someone says when they only want Piper's attention (no question yet)
_ATTENTION = re.compile(
    r"^\s*(?:(?:hey|hi|hello|ok|okay)\s*)?(?:piper|paper)?\s*[,.!?]*\s*"
    r"(?:(?:can|could|may)\s+i\s+ask\s+(?:you\s+)?(?:a\s+)?question"
    r"|are\s+you\s+(?:there|busy|free)|(?:have\s+you\s+)?got\s+a\s+(?:minute|second|moment)"
    r"|hey|hi|hello|good\s+(?:morning|afternoon|evening))?\s*(?:piper|paper)?\s*[,.!?]*\s*$",
    re.IGNORECASE)


def just_attention(text: str) -> bool:
    """'Piper', 'Hi Piper', 'Piper, can I ask you a question?' - not a question yet."""
    return bool(_ATTENTION.match(text or ""))


class ResearchHold:
    def __init__(self, cfg: dict | None = None, llm_cfg: dict | None = None, opener=None):
        cfg = cfg or CONFIG["research"]
        llm_cfg = llm_cfg or CONFIG["llm"]
        self.mind = (cfg.get("mind_url") or "").rstrip("/")      # piper-mind's dashboard; "" = off
        self.hold_s = float(cfg.get("hold_s", 300))
        self.chat_url = llm_cfg["base_url"].rstrip("/")
        self.chat_model = llm_cfg["model"]
        self.chat_keep_alive = llm_cfg.get("keep_alive", "2h")
        self.research_url = (cfg.get("url") or llm_cfg["base_url"]).rstrip("/")
        self.research_model = cfg["model"]
        self.opener = opener or urllib.request.urlopen
        self.held = False
        self.request_id = None
        self.lock = threading.Lock()

    # ---- HTTP --------------------------------------------------------------------------------
    def _call(self, url: str, body: dict | None = None, timeout: float = 2.0) -> dict | None:
        try:
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
            with self.opener(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except Exception:
            return None

    def state(self, timeout: float = 1.0) -> dict | None:
        return self._call(self.mind + "/api/research/hold", timeout=timeout) if self.mind else None

    # ---- what main.py calls --------------------------------------------------------------------
    def active(self) -> bool:
        """Research is running now (not in its scheduled pause, not already held)."""
        s = self.state()
        return bool(s and s.get("researching") and s.get("phase") in ("explore", "consolidate", None)
                    and not s.get("held"))

    def start(self) -> bool:
        """Hold research (and keep holding it until release()). False if piper-mind isn't there."""
        if not self.mind:
            return False
        r = self._call(self.mind + "/api/research/hold", {"seconds": self.hold_s})
        if not r or not r.get("ok"):
            return False
        with self.lock:
            first, self.held, self.request_id = not self.held, True, r.get("id")
        if first:
            threading.Thread(target=self._renew, daemon=True).start()
        return True

    def _renew(self):
        while True:
            time.sleep(RENEW_S)
            with self.lock:
                if not self.held:
                    return
            self._call(self.mind + "/api/research/hold", {"seconds": self.hold_s})

    def wait_ready(self, timeout: float = 20.0, poll_s: float = 0.5) -> bool:
        """Wait until research has stopped using the model (it says so; an unfinished call is
        dropped within a second), at most `timeout`, then swap the models. True when done."""
        end = time.monotonic() + timeout
        settled_at = time.monotonic() + 3.0      # if it never says so (between steps): go after 3 s
        while time.monotonic() < end:
            s = self.state()
            if s is None or s.get("honoured") or not s.get("researching") or time.monotonic() > settled_at:
                break
            time.sleep(poll_s)
        return self.swap_in_chat_model()

    def swap_in_chat_model(self) -> bool:
        """Unload the research model (if it shares this Ollama) and load Piper's chat model."""
        if self.research_url == self.chat_url and self.research_model != self.chat_model:
            self._call(self.research_url + "/api/generate", {"model": self.research_model, "keep_alive": 0},
                       timeout=30)
        r = self._call(self.chat_url + "/api/generate", {"model": self.chat_model, "keep_alive": self.chat_keep_alive},
                       timeout=120)
        return r is not None

    def release(self):
        """The conversation is over: let research carry on (in the background - never waits)."""
        with self.lock:
            if not self.held:
                return
            self.held = False
        if self.mind:
            threading.Thread(target=self._call, args=(self.mind + "/api/research/release", {}),
                             daemon=True).start()
