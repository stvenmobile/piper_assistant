"""
Spoken test phrases for the mouth OLED - answered locally, no language model:

    "Piper, smile" / "frown" / "look surprised"   that face for 5 s, then back to normal
    "Piper, show me your faces"                   every mouth mode and status symbol once
    "Piper, go to sleep"                          the sleep mouth until the next conversation
                                                  (also "you can go to sleep now", "goodnight", ...)
    "Piper, wake up"                              (the wake word already woke her) - says so
    "Piper, test normal talk"                     a sentence in the plain talking mouth
    "Piper, test happy talk"                      a sentence in the happy talking mouth

handle(text) -> None (not one of these), "engaged" (stay in the conversation) or "sleep" (the
caller ends the conversation, so the sleep mouth shows).
"""
import re
import time

from piper_head import protocol

FACE_SECONDS = 5.0
STEP_SECONDS = 2.0

_CLEAN = re.compile(r"[^a-z\s]")
_WAKE = re.compile(r"^(hey |hi |ok |okay )?(piper|paper|hyper|viper)\s+|\s+(piper|paper|hyper|viper)$")
# "go to sleep", and the ways people actually say it: "you can go to sleep now", "time to go to bed",
# "goodnight", "OK, sleep now"
_SLEEP = re.compile(r"((ok|okay|alright|you can|you may|you should|why don t you|it s|its|it is)\s+)*"
                    r"(go to sleep|go to bed|time to sleep|time for bed|time to go to (sleep|bed)|sleep|good ?night)"
                    r"(\s+(now|then|for now|for a while))?")

FACES = {
    "smile": "smile",
    "frown": "frown",
    "look surprised": "surprise",
    "be surprised": "surprise",
    "surprise": "surprise",
    "surprised": "surprise",
}

NORMAL_TALK = "This is my normal talking face. I use it for most of what I say."
HAPPY_TALK = "And this is my happy talking face, for when I'm glad to see you!"


def clean(text: str) -> str:
    t = " ".join(_CLEAN.sub(" ", text.lower()).split())
    t = _WAKE.sub("", t).strip()
    for p in ("please ", "can you ", "could you "):
        if t.startswith(p):
            t = t[len(p):]
    return t.removesuffix(" please").strip()


class FaceTest:
    def __init__(self, expression, say, pause=time.sleep):
        self.expression = expression
        self.say = say                      # say(text, mood="neutral")
        self.pause = pause

    def handle(self, text: str) -> str | None:
        t = clean(text)
        if t in FACES:
            self.expression.show(mode=FACES[t], seconds=FACE_SECONDS)
            return "engaged"
        if re.fullmatch(r"show me your (faces?|expressions?|moods?)", t):
            self.show_all()
            return "engaged"
        if _SLEEP.fullmatch(t):
            self.say("Goodnight.")
            self.expression.sleep()
            return "sleep"
        if re.fullmatch(r"wake up|are you awake", t):
            self.say("I'm awake!", "warm")
            return "engaged"
        if re.fullmatch(r"test (your )?normal (talk|talking)", t):
            self.say(NORMAL_TALK, "neutral")
            return "engaged"
        if re.fullmatch(r"test (your )?happy (talk|talking)", t):
            self.say(HAPPY_TALK, "warm")
            return "engaged"
        return None

    def show_all(self):
        """Every mouth mode, then every status symbol, STEP_SECONDS each."""
        for mode in protocol.MOUTH_MODES:
            self.expression.show(mode=mode, symbol="idle", seconds=STEP_SECONDS + 1)
            self.pause(STEP_SECONDS)
        for symbol in protocol.MOUTH_SYMBOLS[1:]:
            self.expression.show(mode="neutral", symbol=symbol, seconds=STEP_SECONDS + 1)
            self.pause(STEP_SECONDS)
        self.expression.clear()
