"""
Meet a person: Piper only helps people she knows by name.

  * A stranger in view gets asked their name - straight away, no wake word needed. Until Piper
    has it she keeps asking, and answers nothing else ("first I need to know who you are").
  * She checks she heard it right ("Steve - did I get that right?"; a correction or a spelling
    works too), then asks them to look at her while vision stores a few looks at their face.
  * Someone who won't say is told they can say goodbye; after "goodbye" she leaves them be
    until they walk away and come back.
  * Someone she knows gets "Welcome back, Steve!" (not again if they were here a few minutes ago).
  * "Forget me" deletes their face prints.

Without vision (service down, or recognition unavailable) the gate stays open - Piper behaves
as she did before she could see.
"""
import random
import re
import time
from datetime import date
from pathlib import Path

from piper_skills import Context

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

MET, LEFT, GOODBYE, QUIET = "met", "left", "goodbye", "quiet"

# Words that are never a name (fillers, answers, and what "I'm ..." is often followed by)
NOT_NAMES = set("""
a an the and or but so to of in on at for with from it its is was are be been am i im me my
mine you your we they he she him her this that there here what whats who whos why how when
where which yes yeah yep yup no nope nah not none nothing never okay ok sure fine good great
well um uh er hmm huh oh ah hi hello hey piper paper goodbye bye thanks thank please sorry
just only really busy tired back looking trying going doing here name names called call
weather time today tomorrow anyone nobody somebody someone guess dunno know dont wont cant
rather telling tell business private pass again sir maam man guy friend human person stranger
""".split())

_NAME_WORD = r"[a-z][a-z'\-]*"
# "my name is Mary Ann" can be two words; "this is Bob speaking" / "I'm Bob here" only one
_INTRO = re.compile(
    r"\b(?:(my name is|my names|my name's|name is|call me|they call me)|"
    r"(?:i am|i'm|im|it's|it is|its|this is))\s+(" + _NAME_WORD + r")(?:\s+(" + _NAME_WORD + r"))?")
_SPELLED = re.compile(r"\b((?:[a-z][\s.,\-]+){1,}[a-z])\b")


def _clean(text: str) -> str:
    return " ".join(re.sub(r"[^a-z'\-\s]", " ", text.lower()).split())


def _title(words) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in words)


def extract_name(text: str) -> str | None:
    """A name from an answer to "what's your name?" - "I'm Steve", "my name is Mary Ann",
    "Steve.", "call me Al", "S-T-E-V-E" - or None if there isn't one."""
    raw = (text or "").lower()
    # spelled out: "S-T-E-V-E", "s t e v e"
    m = _SPELLED.search(raw)
    if m:
        letters = re.findall(r"[a-z]", m.group(1))
        if len(letters) >= 2:
            return _title(["".join(letters)])
    t = _clean(raw)
    if not t:
        return None
    m = _INTRO.search(t)
    if m:
        full_name, first, second = m.group(1), m.group(2), m.group(3)
        if first in NOT_NAMES:
            return None
        two = full_name and second and second not in NOT_NAMES and m.end() == len(t)
        words = [first] + ([second] if two else [])
        return _title(words)
    # a bare answer: one or two words, after dropping greetings / fillers
    words = [w for w in t.split() if w not in ("hi", "hello", "hey", "um", "uh", "er", "oh", "piper", "paper")]
    if 1 <= len(words) <= 2 and not any(w in NOT_NAMES for w in words):
        return _title(words)
    return None


def _has(t: str, phrases) -> bool:
    return any(re.search(r"\b" + re.escape(p) + r"\b", t) for p in phrases)


def is_no(text: str) -> bool:
    t = _clean(text)
    return bool(t) and _has(t, ("no", "nope", "nah", "wrong", "not quite", "not right", "that's not", "incorrect"))


def is_yes(text: str) -> bool:
    t = _clean(text)
    return bool(t) and not is_no(text) and _has(
        t, ("yes", "yeah", "yep", "yup", "correct", "right", "that's it", "sure", "exactly", "uh-huh", "you got it"))


def is_goodbye(text: str) -> bool:
    return _has(_clean(text), ("goodbye", "bye", "see you", "see ya", "good night", "farewell"))


def is_refusal(text: str) -> bool:
    t = _clean(text)
    return t in ("no", "nope", "nah", "no thanks", "no thank you", "pass") or _has(t, (
        "not telling", "won't tell", "wont tell", "rather not", "don't want to", "dont want to",
        "none of your business", "why do you need", "why do you want", "it's private",
        "not going to tell", "not gonna tell", "i'd rather not", "prefer not"))


def is_forget_request(text: str) -> bool:
    return _has(_clean(text), ("forget me", "forget my face", "forget who i am", "delete my face"))


def is_who_am_i(text: str) -> bool:
    return _has(_clean(text), ("who am i", "do you know me", "do you know who i am", "what's my name", "whats my name"))


def ensure_profile(name: str):
    """A starter profile for someone new (the supervisor reads profiles/<name>.md)."""
    path = PROFILES_DIR / f"{name.lower()}.md"
    if not path.exists():
        PROFILES_DIR.mkdir(exist_ok=True)
        path.write_text(f"# Profile: {name}\n\n- First met: {date.today().isoformat()}\n", encoding="utf-8")


class MeetPerson:
    MAX_SILENT = 3              # unanswered asks before Piper stops asking (until they speak)
    MAX_TRIES = 3               # misheard names before she asks for a spelling
    VERDICT_WAIT_S = 3.0

    def __init__(self, ctx: Context, welcome_back_s: float = 600, required: bool = True,
                 clock=time.monotonic):
        self.ctx = ctx
        self.welcome_back_s = welcome_back_s
        self.required = required
        self.clock = clock
        self.dismissed: set = set()      # tracks that said goodbye (or asked to be forgotten)
        self.quiet: set = set()          # tracks that never answered - wait until they speak
        self.last_present: dict = {}     # name -> when last seen
        self.greeted_track = None

    # --- when is the gate in force? ---------------------------------------------------------------
    def active(self) -> bool:
        v = self.ctx.vision
        return self.required and getattr(v, "connected", False) and v.who().get("recognition", False)

    def _seen(self, p):
        if p.get("who"):
            self.last_present[p["who"]] = self.clock()

    # --- called by the main loop while idle --------------------------------------------------------
    def on_idle(self) -> str | None:
        """Look at who is there: welcome back a known face, or meet a stranger. Returns the name
        of the person Piper is now talking to, or None."""
        if not self.active():
            return None
        p = self.ctx.vision.who()
        if not p["present"] or p["who"] is None:
            return None
        track, who = p["track"], p["who"]
        if who:                                             # someone she knows
            last = self.last_present.get(who)
            self._seen(p)
            if track == self.greeted_track:
                return None
            self.greeted_track = track
            if last is not None and self.clock() - last < self.welcome_back_s:
                return None                                 # stepped away briefly - no fuss
            self.ctx.log("PERSON", f"Welcome back {who}", f"track {track}")
            self.ctx.say(random.choice([
                f"Welcome back, {who}! How can I help you today?",
                f"Hi {who}, welcome back. What can I do for you?",
                f"Good to see you again, {who}. How can I help?",
            ]), "warm")
            return who
        if track in self.dismissed or track in self.quiet:
            return None
        return self._finish(track, self.meet(track))

    # --- called before Piper answers anything ------------------------------------------------------
    def gate(self, text: str) -> tuple[bool, str | None]:
        """(go ahead and answer?, who is speaking). When it says no, the skill has already
        handled the moment; a name means Piper is now talking to that person."""
        if not self.active():
            return True, None
        p = self.ctx.vision.wait_for_verdict(self.VERDICT_WAIT_S)
        if not p["present"]:
            self.ctx.say("I can hear you, but I can't see you. Come where I can see your face.", "curious")
            return False, None
        if p["who"] is None:
            self.ctx.say("Could you look at me for a moment, so I can see who you are?", "curious")
            return False, None
        if p["who"]:
            self._seen(p)
            self.greeted_track = p["track"]
            return True, p["who"]
        track = p["track"]
        if track in self.dismissed and not extract_name(text):
            self.ctx.say("Sorry, I can only help people I know.", "neutral")
            return False, None
        self.dismissed.discard(track)
        self.quiet.discard(track)
        return False, self._finish(track, self.meet(track, first=text))

    def handle_command(self, text: str, name: str | None) -> bool:
        """Face-related requests from someone she knows: "forget me", "who am I?"."""
        if not self.active() or not name:
            return False
        if is_forget_request(text):
            ok = self.ctx.vision.forget(name)
            self.ctx.log("PERSON", f"Forget {name}", "done" if ok else "failed")
            if ok:
                self.dismissed.add(self.ctx.vision.who().get("track"))
                self.last_present.pop(name, None)
                self.ctx.say(f"Okay, {name}. I've forgotten your face. Goodbye!", "neutral")
            else:
                self.ctx.say("Sorry, something went wrong and I couldn't do that.", "concerned")
            return True
        if is_who_am_i(text):
            self.ctx.say(f"You're {name}, of course!", "warm")
            return True
        return False

    # --- the conversation ---------------------------------------------------------------------
    def _finish(self, track, result) -> str | None:
        outcome, name = result
        self.ctx.log("PERSON", f"Meeting track {track}: {outcome}", name or "")
        if outcome == MET:
            self.greeted_track = track
            self.last_present[name] = self.clock()
            return name
        if outcome == GOODBYE:
            self.dismissed.add(track)
        elif outcome == QUIET:
            self.quiet.add(track)
        return None

    def meet(self, track, first: str | None = None) -> tuple[str, str | None]:
        """Get a stranger's name, then their face. Returns (outcome, name)."""
        say, hear = self.ctx.say, self.ctx.hear
        if first is None:
            say(random.choice([
                "Hi there! I don't think we've met. What's your name?",
                "Hello! I don't know you yet. What's your name?",
            ]), "curious")
            text = None
        else:
            text = first                    # they spoke to Piper first - treat it as the reply
        silent = tries = 0
        while True:
            if not self.ctx.vision.still_here(track):
                return LEFT, None
            if text is None:
                text = hear()
                if not self.ctx.vision.still_here(track):
                    return LEFT, None
            if not text:
                silent += 1
                if silent >= self.MAX_SILENT:
                    return QUIET, None
                if silent == 1:
                    say("Sorry, I didn't catch that. What's your name?", "curious")
            elif is_goodbye(text):
                say("Okay. Goodbye!", "neutral")
                return GOODBYE, None
            elif name := extract_name(text):
                outcome, name = self.confirm(track, name)
                if outcome in (MET, LEFT):
                    return outcome, name
                tries += 1
                say("Sorry about that. Could you spell your name for me?" if tries >= self.MAX_TRIES
                    else "Sorry about that. What's your name?", "uncertain")
            elif is_refusal(text):
                say("That's okay. But I only help people I know by name. "
                    "If you'd rather not tell me, just say goodbye.", "neutral")
            else:
                say(random.choice([
                    "I'd love to help, but first I need to know who you are. What's your name?",
                    "Before I can help, I need your name. What should I call you?",
                ]), "curious")
            text = None

    def confirm(self, track, name: str) -> tuple[str | None, str]:
        """Check the name, then enrol the face. (MET / LEFT, name), or (None, name) if the name
        was wrong."""
        say, hear = self.ctx.say, self.ctx.hear
        for _ in range(3):
            say(f"{name}? Did I get that right?", "curious")
            answer = hear()
            if not self.ctx.vision.still_here(track):
                return LEFT, name
            if is_yes(answer):
                break
            if not answer:
                continue                    # silence - ask again
            fixed = extract_name(answer)
            if fixed and fixed.lower() == name.lower():
                break                       # they just said it again
            if fixed:
                name = fixed                # "no, it's Stephen"
                continue
            return None, name
        else:
            return None, name
        return self.enrol(track, name), name

    def enrol(self, track, name: str) -> str:
        say = self.ctx.say
        say(f"Nice to meet you, {name}! Look at me for a moment while I memorize your face.", "warm")
        for attempt in range(2):
            result = self.ctx.vision.enroll(name, track)
            if result.get("t") == "ENROLLED":
                ensure_profile(name)
                self.ctx.log("PERSON", f"Met {name}", f"{result.get('looks')} looks stored")
                say(f"Got it. I'll remember you, {name}. How can I help you today?", "warm")
                return MET
            reason = result.get("reason")
            self.ctx.log("PERSON", f"Enrolling {name} failed", reason)
            if reason in ("lost", "not here"):
                return LEFT
            if reason == "no good look" and attempt == 0:
                say("I couldn't get a clear look. Could you face me and hold still?", "uncertain")
                continue
            break
        ensure_profile(name)
        say(f"I couldn't quite get your face, but that's okay, {name}. How can I help you?", "neutral")
        return MET
