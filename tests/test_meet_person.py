import pytest

from piper_skills import Context
from piper_skills import meet_person as mp
from piper_skills.meet_person import MeetPerson, extract_name, is_no, is_refusal, is_yes


@pytest.mark.parametrize("text, name", [
    ("Steve", "Steve"),
    ("Steve.", "Steve"),
    ("My name is Steve.", "Steve"),
    ("My name is Mary Ann", "Mary Ann"),
    ("I'm Bob here", "Bob"),
    ("Hi Piper, I'm Anna", "Anna"),
    ("Call me Al", "Al"),
    ("It's Bob.", "Bob"),
    ("This is Bob speaking", "Bob"),
    ("S-T-E-V-E", "Steve"),
    ("s t e v e", "Steve"),
    ("What's the weather like?", None),
    ("Hello Piper", None),
    ("No.", None),
    ("I'm not telling you", None),
    ("I'm fine", None),
    ("", None),
])
def test_extract_name(text, name):
    assert extract_name(text) == name


def test_yes_no_and_refusal():
    assert is_yes("Yes, that's right.") and not is_no("Yes, that's right.")
    assert is_no("No, that's not right.") and not is_yes("No, that's not right.")
    assert is_refusal("No.") and is_refusal("I'd rather not say") and is_refusal("None of your business")
    assert not is_refusal("Steve")


class FakeVision:
    def __init__(self, who="", track=1, enroll="ENROLLED"):
        self.connected = True
        self.p = {"present": True, "track": track, "who": who, "recognition": True}
        self.enroll_result = enroll
        self.enrolled, self.forgotten = [], []

    def who(self):
        return dict(self.p)

    def still_here(self, track):
        return self.p["present"] and self.p["track"] == track

    def wait_for_verdict(self, timeout):
        return self.who()

    def enroll(self, name, track, timeout=15):
        self.enrolled.append((name, track))
        if self.enroll_result == "ENROLLED":
            self.p["who"] = name
            return {"t": "ENROLLED", "name": name, "looks": 5}
        return {"t": "ENROLL_FAILED", "name": name, "reason": self.enroll_result}

    def forget(self, name):
        self.forgotten.append(name)
        return True


def run(vision, replies, **kw):
    said = []
    replies = list(replies)
    ctx = Context(say=lambda text, mood="neutral": said.append(text),
                  hear=lambda: replies.pop(0) if replies else "", vision=vision)
    clock = kw.pop("clock", None)
    skill = MeetPerson(ctx, clock=clock or (lambda: 0.0), **kw)
    return skill, said


@pytest.fixture(autouse=True)
def no_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "PROFILES_DIR", tmp_path)


def test_meets_a_stranger_and_learns_their_face():
    v = FakeVision()
    skill, said = run(v, ["I'm Steve", "Yes"])
    assert skill.on_idle() == "Steve"
    assert v.enrolled == [("Steve", 1)]
    assert "name" in said[0] and "Steve?" in said[1] and "remember you" in said[-1]


def test_keeps_asking_until_it_gets_a_name():
    v = FakeVision()
    skill, said = run(v, ["What's the weather?", "Tell me a joke", "Anna", "Yes"])
    assert skill.on_idle() == "Anna"
    assert sum("first I need" in s or "need your name" in s for s in said) == 2


def test_a_correction_replaces_the_misheard_name():
    v = FakeVision()
    skill, said = run(v, ["Steve", "No, it's Stephen", "Yes"])
    assert skill.on_idle() == "Stephen"
    assert v.enrolled == [("Stephen", 1)]


def test_refusal_suggests_goodbye_and_goodbye_ends_it():
    v = FakeVision()
    skill, said = run(v, ["No", "Goodbye"])
    assert skill.on_idle() is None
    assert any("say goodbye" in s for s in said)
    assert v.enrolled == []
    said.clear()
    assert skill.on_idle() is None and said == []          # leaves them be
    ok, name = skill.gate("What time is it?")
    assert not ok and name is None and "only help people I know" in said[-1]


def test_silence_stops_the_asking_until_they_speak():
    v = FakeVision()
    skill, said = run(v, [])
    assert skill.on_idle() is None
    n = len(said)
    assert skill.on_idle() is None and len(said) == n       # doesn't nag
    skill.ctx.hear = lambda: "Yes"
    ok, name = skill.gate("My name is Bob")                 # they speak up - straight to the name
    assert not ok and name == "Bob"


def test_stranger_asking_a_question_gets_asked_their_name():
    v = FakeVision()
    skill, said = run(v, ["Bob", "Yes"])
    ok, name = skill.gate("What's the weather?")
    assert not ok and name == "Bob"
    assert "first I need" in said[0] or "need your name" in said[0]


def test_welcomes_back_a_known_face_once():
    t = [0.0]
    v = FakeVision(who="Steve")
    skill, said = run(v, [], clock=lambda: t[0], welcome_back_s=600)
    assert skill.on_idle() == "Steve" and "Steve" in said[0]
    assert skill.on_idle() is None                          # same visit
    v.p["track"] = 2; t[0] = 60                             # looked away a minute
    assert skill.on_idle() is None
    v.p["track"] = 3; t[0] = 60 + 601                       # gone a while
    assert skill.on_idle() == "Steve"


def test_known_person_passes_the_gate():
    skill, said = run(FakeVision(who="Steve"), [])
    assert skill.gate("What's the weather?") == (True, "Steve")
    assert said == []


def test_cant_see_anyone():
    v = FakeVision()
    v.p["present"] = False
    skill, said = run(v, [])
    assert skill.gate("hello") == (False, None) and "can't see" in said[0]


def test_gate_open_without_vision():
    v = FakeVision()
    v.connected = False
    skill, said = run(v, [])
    assert skill.gate("hello") == (True, None) and skill.on_idle() is None


def test_person_leaving_ends_the_meeting_quietly():
    v = FakeVision()
    skill, said = run(v, [])
    skill.ctx.hear = lambda: (v.p.update(present=False), "")[1]
    assert skill.on_idle() is None and len(said) == 1


def test_enrol_retries_then_carries_on_with_just_the_name():
    v = FakeVision(enroll="no good look")
    skill, said = run(v, ["Steve", "Yes"])
    assert skill.on_idle() == "Steve"
    assert len(v.enrolled) == 2 and "that's okay" in said[-1]


def test_forget_me():
    v = FakeVision(who="Steve")
    skill, said = run(v, [])
    assert skill.handle_command("Please forget me", "Steve")
    assert v.forgotten == ["Steve"]
    v.p["who"] = ""
    said.clear()
    assert skill.on_idle() is None and said == []           # doesn't ask again straight away
