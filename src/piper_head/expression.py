"""
What the mouth OLED shows: its expression (mode) and the status symbol in its top-right corner,
decided from the assistant's state. main.py calls assistant() on every state change; the face
test skill and a few moments (a welcome, a misunderstanding) show something for a while.

    assistant state        mouth                                        symbol
    IDLE                   neutral - or sleep when research is paused   none (hidden while asleep)
                           (piper_research rhythm, e.g. weekdays
                           10:00-17:00) or she was told to sleep
    ENGAGED                neutral                                      listening (+ / -)
    PROCESSING             neutral                                      thinking (dots)
    SPEAKING               talk; happytalk when the mood is "warm"      none

    show(mode=, symbol=, seconds=)   override either for a while (ends early when she starts
                                     thinking or speaking)
    confused()                       ??? for a few seconds
    sleep()                          sleep until the next conversation ("Piper, go to sleep")

Speech moves the lips only in talk / happytalk (the mouth simulates speech by itself), so
frowning while talking can't happen. Never raises: the client drops messages if the head link
isn't running.
"""
import threading
import time
from datetime import datetime

from piper_brain.config import CONFIG

TICK_S = 1.0          # override expiry and the research-pause check
RESEND_S = 60.0       # re-send what should be showing, in case the link restarted


class Expression:
    def __init__(self, client, rhythm=None, clock=time.monotonic, now=datetime.now):
        self.client = client
        if rhythm is None:
            try:
                from piper_research.rhythm import Rhythm
                rhythm = Rhythm(CONFIG["research"]["rhythm"])
            except Exception:
                rhythm = None
        self.rhythm = rhythm
        self.clock, self.now = clock, now
        self.status, self.mood = "IDLE", "neutral"
        self.override = None              # (mode or None, symbol or None, until)
        self.sleep_hold = False           # told to sleep: until she is next engaged
        self.sent, self.sent_at = None, 0.0
        self.lock = threading.RLock()

    # ---- what main.py and the skills call ----------------------------------------------------
    def assistant(self, status: str, mood: str = "neutral"):
        with self.lock:
            if status != "IDLE":
                self.sleep_hold = False                 # any conversation wakes her
            if status in ("PROCESSING", "SPEAKING"):
                self.override = None                    # a shown face gives way to thinking / talking
            self.status, self.mood = status, mood or "neutral"
            self._apply()

    def show(self, mode: str | None = None, symbol: str | None = None, seconds: float = 5.0):
        with self.lock:
            self.override = (mode, symbol, self.clock() + seconds)
            self._apply()

    def clear(self):
        with self.lock:
            self.override = None
            self._apply()

    def confused(self, seconds: float = 4.0):
        self.show(symbol="confused", seconds=seconds)

    def sleep(self):
        with self.lock:
            self.sleep_hold = True
            self.override = None
            self._apply()

    def research_paused(self) -> bool:
        try:
            return self.rhythm is not None and self.rhythm.phase(self.now()) == "pause"
        except Exception:
            return False

    def wanted(self) -> tuple[str, str]:
        """(mode, symbol) that should be showing now."""
        with self.lock:
            s = self.status
            if s == "IDLE":
                mode = "sleep" if (self.sleep_hold or self.research_paused()) else "neutral"
                symbol = "idle"
            elif s == "ENGAGED":
                mode, symbol = "neutral", "listening"
            elif s == "PROCESSING":
                mode, symbol = "neutral", "thinking"
            elif s == "SPEAKING":
                mode, symbol = ("happytalk" if self.mood == "warm" else "talk"), "idle"
            else:
                mode, symbol = "neutral", "idle"
            if self.override:
                o_mode, o_symbol, until = self.override
                if self.clock() < until:
                    mode = o_mode or mode
                    symbol = o_symbol or symbol
            return mode, symbol

    # ---- keeping the mouth up to date ---------------------------------------------------------
    def tick(self):
        with self.lock:
            if self.override and self.clock() >= self.override[2]:
                self.override = None
            self._apply()

    def start(self):
        def run():
            while True:
                time.sleep(TICK_S)
                try:
                    self.tick()
                except Exception as e:                  # never let the mouth stop Piper
                    print(f"[Expression] {e}")
        threading.Thread(target=run, daemon=True).start()
        return self

    def _apply(self):
        w = self.wanted()
        if w != self.sent or self.clock() - self.sent_at > RESEND_S:
            self.client.mouth(mode=w[0], symbol=w[1])
            self.sent, self.sent_at = w, self.clock()
