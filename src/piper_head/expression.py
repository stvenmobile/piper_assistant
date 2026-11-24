"""
What the mouth OLED shows: its expression (mode) and the status symbol in its top-right corner,
decided from the assistant's state. main.py calls assistant() on every state change; the face
test skill and a few moments (a welcome, a misunderstanding) show something for a while.

    assistant state        mouth                                        symbol
    IDLE                   blank - the screen is dark (no burn-in;      -
                           research carries on)
    ENGAGED                neutral                                      listening (+ / -)
    PROCESSING             neutral                                      thinking (dots)
    SPEAKING               talk; happytalk when the mood is "warm",     none
                           excitedtalk when it is "excited"

    research_notice(True)  "Research in progress..." instead of the mouth while Piper pauses her
                           research after the wake word; research_notice(False) when she's ready
    show(mode=, symbol=, seconds=)   override either for a while (ends early when she starts
                                     thinking or speaking)
    confused()             ??? for a few seconds
    sleep()                the sleep mouth for SLEEP_S ("Piper, go to sleep"), then blank

Speech moves the lips only in talk / happytalk / excitedtalk (the mouth simulates speech by
itself), so frowning while talking can't happen. Never raises: the client drops messages if the
head link isn't running.

Version 1.1.0 - created before 2026-11-24 (see git history), last modified 2026-11-24
History:
    1.1.0  2026-11-24  "excited" mood -> excitedtalk while speaking.
    1.0.0  -           earlier work, unversioned (git history)
"""
import threading
import time

TICK_S = 1.0          # override expiry
RESEND_S = 60.0       # re-send what should be showing, in case the link restarted
SLEEP_S = 120.0       # "go to sleep": the sleep mouth this long, then the dark screen
TALK_BY_MOOD = {"warm": "happytalk", "excited": "excitedtalk"}   # the talking mouth per mood; else "talk"


class Expression:
    def __init__(self, client, clock=time.monotonic):
        self.client = client
        self.clock = clock
        self.status, self.mood = "IDLE", "neutral"
        self.override = None              # (mode or None, symbol or None, until)
        self.sleep_until = 0.0            # told to sleep: the sleep mouth until then (if still idle)
        self.notice = False               # "Research in progress..."
        self.sent, self.sent_at = None, 0.0
        self.lock = threading.RLock()

    # ---- what main.py and the skills call ----------------------------------------------------
    def assistant(self, status: str, mood: str = "neutral"):
        with self.lock:
            if status != "IDLE":
                self.sleep_until = 0.0                  # any conversation wakes her
            else:
                self.notice = False                     # nobody talking: no message either
            if status in ("PROCESSING", "SPEAKING"):
                self.override = None                    # a shown face gives way to thinking / talking
            self.status, self.mood = status, mood or "neutral"
            self._apply()

    def research_notice(self, on: bool):
        with self.lock:
            self.notice = bool(on)
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

    def sleep(self, seconds: float = SLEEP_S):
        with self.lock:
            self.sleep_until = self.clock() + seconds
            self.override = None
            self._apply()

    def wanted(self) -> tuple[str, str]:
        """(mode, symbol) that should be showing now."""
        with self.lock:
            if self.notice:
                return "research", "idle"
            s = self.status
            if s == "IDLE":
                return ("sleep" if self.clock() < self.sleep_until else "blank"), "idle"
            if s == "ENGAGED":
                mode, symbol = "neutral", "listening"
            elif s == "PROCESSING":
                mode, symbol = "neutral", "thinking"
            elif s == "SPEAKING":
                mode, symbol = TALK_BY_MOOD.get(self.mood, "talk"), "idle"
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
