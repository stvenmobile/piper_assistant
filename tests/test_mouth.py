"""The mouth OLED: its messages, the link's translation, the expression rules, the test phrases."""
import pytest

from piper_brain.quick_responder import QuickResponder
from piper_head import protocol
from piper_head.expression import Expression
from piper_head.link import HeadLink, find_mouth_port, find_port
from piper_skills.face_test import FaceTest, clean


# ---- protocol --------------------------------------------------------------------------------
def test_mouth_message_fields():
    assert protocol.mouth(mode="smile") == {"t": "MOUTH", "mode": "smile"}
    assert protocol.mouth(symbol="thinking", demo=False) == {"t": "MOUTH", "symbol": "thinking", "demo": False}
    with pytest.raises(ValueError):
        protocol.mouth(mode="grin")
    with pytest.raises(ValueError):
        protocol.mouth(symbol="zzz")


def test_mouth_lines():
    assert protocol.mouth_lines({"t": "MOUTH", "mode": "happytalk", "symbol": "idle", "level": 0.5}) == \
        ["Mhappytalk", "Iidle", "L128"]
    assert protocol.mouth_lines({"t": "MOUTH", "demo": True}) == ["D1"]
    assert protocol.mouth_lines({"t": "MOUTH", "mode": "bogus", "level": "x"}) == []


# ---- the link ----------------------------------------------------------------------------------
def test_head_controller_only_gets_speech_levels():
    link = HeadLink()
    assert link.from_client(protocol.mouth(mode="smile")) == []
    assert link.from_client(protocol.mouth(level=0.4, mode="talk")) == [{"t": "MOUTH", "level": 0.4}]


def test_mouth_mode_and_symbol_are_remembered_for_reconnects():
    link = HeadLink()
    assert link.mouth_on_connect() == []
    assert link.to_mouth(protocol.mouth(mode="frown")) == ["Mfrown"]
    link.to_mouth(protocol.mouth(symbol="listening", level=0.2))
    assert link.mouth_on_connect() == ["Mfrown", "Ilistening"]
    link.to_mouth(protocol.mouth(demo=False))
    assert link.mouth_on_connect() == ["Mneutral", "Ilistening"]
    assert link.to_mouth(protocol.face("idle")) == []


def test_mouth_style_is_read_from_its_report():
    from piper_head.link import mouth_style_of
    assert mouth_style_of(b"mouth: LIVE, mode 0 neutral, status 0 idle, style spectrum, contrast 160") == "spectrum"
    assert mouth_style_of("mouth: LIVE, mode 0 neutral, status 0 idle, contrast 160") is None   # old firmware
    assert mouth_style_of("hello style lips") is None


def test_mouth_style_is_sent_first_on_connect():
    assert HeadLink("spectrum").mouth_on_connect() == ["Sspectrum"]
    link = HeadLink("lips")
    link.to_mouth(protocol.mouth(mode="smile"))
    assert link.mouth_on_connect() == ["Slips", "Msmile"]
    assert HeadLink("bogus").mouth_on_connect() == []


def test_listening_stops_talking():
    link = HeadLink()
    link.to_mouth(protocol.mouth(mode="talk"))
    link.to_mouth(protocol.mouth(symbol="listening"))
    assert link.mouth_last["mode"] == "neutral"
    link.to_mouth(protocol.mouth(mode="happytalk"))
    link.to_mouth(protocol.mouth(symbol="listening"))
    assert link.mouth_last["mode"] == "smile"
    link.to_mouth(protocol.mouth(mode="frown"))
    link.to_mouth(protocol.mouth(symbol="listening"))
    assert link.mouth_last["mode"] == "frown"


def _ports(monkeypatch, ports):
    from serial.tools import list_ports
    from types import SimpleNamespace as P
    monkeypatch.setattr(list_ports, "comports", lambda: [P(device=d, vid=v, pid=p) for d, v, p in ports])


def test_head_and_mouth_are_told_apart(monkeypatch):
    _ports(monkeypatch, [("/dev/ttyACM0", 0x303A, 0x1001), ("/dev/ttyACM1", 0x303A, 0x1001)])
    kinds = {"/dev/ttyACM0": "mouth", "/dev/ttyACM1": "other"}
    assert find_port("auto", kinds.get) == "/dev/ttyACM1"
    assert find_mouth_port("auto", kinds.get) == "/dev/ttyACM0"
    assert find_mouth_port("auto", kinds.get, exclude={"/dev/ttyACM0"}) is None
    assert find_mouth_port("off", kinds.get) is None
    assert find_mouth_port("/dev/ttyX", kinds.get) == "/dev/ttyX"


def test_the_face_is_still_preferred_and_the_mouth_never_taken_as_head(monkeypatch):
    _ports(monkeypatch, [("/dev/ttyACM0", 0x303A, 0x1001), ("/dev/ttyUSB0", 0x1A86, 0x7523)])
    assert find_port("auto", lambda d: "mouth") == "/dev/ttyUSB0"
    _ports(monkeypatch, [("/dev/ttyACM0", 0x303A, 0x1001)])
    assert find_port("auto", lambda d: "mouth") is None


# ---- expression --------------------------------------------------------------------------------
class FakeClient:
    def __init__(self):
        self.sent = []

    def mouth(self, **kw):
        self.sent.append((kw.get("mode"), kw.get("symbol")))
        return True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def ex():
    return Expression(FakeClient(), clock=Clock())


@pytest.mark.parametrize("status, mood, wanted", [
    ("IDLE", "neutral", ("blank", "idle")),
    ("ENGAGED", "neutral", ("neutral", "listening")),
    ("PROCESSING", "neutral", ("neutral", "thinking")),
    ("SPEAKING", "neutral", ("talk", "idle")),
    ("SPEAKING", "warm", ("happytalk", "idle")),
    ("SPEAKING", "concerned", ("talk", "idle")),
])
def test_states_map_to_the_mouth(ex, status, mood, wanted):
    ex.assistant(status, mood)
    assert ex.wanted() == wanted
    assert ex.client.sent[-1] == wanted


def test_told_to_sleep_then_the_screen_goes_dark(ex):
    ex.assistant("ENGAGED")
    ex.sleep(seconds=120)
    ex.assistant("IDLE")
    assert ex.wanted()[0] == "sleep"
    ex.clock.t = 121
    ex.tick()
    assert ex.wanted()[0] == "blank"
    assert ex.client.sent[-1] == ("blank", "idle")


def test_a_conversation_wakes_her(ex):
    ex.sleep(seconds=120)
    ex.assistant("ENGAGED")                  # "Piper, wake up"
    ex.assistant("IDLE")
    assert ex.wanted()[0] == "blank"


def test_research_notice_replaces_the_mouth_until_she_is_ready(ex):
    ex.assistant("ENGAGED")
    ex.research_notice(True)
    assert ex.wanted() == ("research", "idle")
    ex.assistant("SPEAKING")                 # "give me a second to pause my research" - still the message
    assert ex.wanted() == ("research", "idle")
    ex.research_notice(False)
    assert ex.wanted() == ("talk", "idle")
    ex.research_notice(True)
    ex.assistant("IDLE")                     # nobody talking any more: no message, dark screen
    assert ex.wanted() == ("blank", "idle")


def test_a_shown_face_lasts_its_time_then_goes_back(ex):
    ex.assistant("ENGAGED")
    ex.show(mode="smile", seconds=5)
    assert ex.wanted() == ("smile", "listening")
    ex.clock.t = 5.1
    ex.tick()
    assert ex.wanted() == ("neutral", "listening")
    assert ex.client.sent[-1] == ("neutral", "listening")


def test_speaking_ends_a_shown_face(ex):
    ex.assistant("ENGAGED")
    ex.show(mode="frown", seconds=5)
    ex.assistant("SPEAKING")
    assert ex.wanted() == ("talk", "idle")


def test_confused_shows_the_question_marks(ex):
    ex.assistant("ENGAGED")
    ex.confused(seconds=4)
    assert ex.wanted() == ("neutral", "confused")


def test_only_changes_are_sent(ex):
    ex.assistant("ENGAGED")
    n = len(ex.client.sent)
    ex.tick()
    ex.assistant("ENGAGED")
    assert len(ex.client.sent) == n


# ---- the spoken test phrases -------------------------------------------------------------------
@pytest.fixture
def ft(ex):
    said = []
    f = FaceTest(ex, lambda text, mood="neutral": said.append((text, mood)), pause=lambda s: None)
    f.said = said
    return f


@pytest.mark.parametrize("text, cleaned", [
    ("Piper, smile.", "smile"),
    ("Hey Piper, please look surprised!", "look surprised"),
    ("Test normal talk, Piper.", "test normal talk"),
])
def test_clean(text, cleaned):
    assert clean(text) == cleaned


@pytest.mark.parametrize("text, mode", [("smile", "smile"), ("Frown.", "frown"), ("Look surprised", "surprise")])
def test_faces_on_request(ft, ex, text, mode):
    ex.assistant("ENGAGED")
    assert ft.handle(text) == "engaged"
    assert ex.wanted()[0] == mode


def test_test_talk_phrases(ft):
    assert ft.handle("Piper, test normal talk") == "engaged"
    assert ft.handle("test happy talk") == "engaged"
    assert [m for _, m in ft.said] == ["neutral", "warm"]


def test_go_to_sleep(ft, ex):
    assert ft.handle("Piper, go to sleep.") == "sleep"
    ex.assistant("IDLE")
    assert ex.wanted()[0] == "sleep"


def test_show_me_your_faces_steps_through_everything(ft, ex):
    ex.assistant("ENGAGED")
    assert ft.handle("Show me your faces") == "engaged"
    shown = set(ex.client.sent)
    for mode in protocol.MOUTH_MODES:
        assert (mode, "idle") in shown
    for symbol in ("listening", "thinking", "confused"):
        assert ("neutral", symbol) in shown
    assert ex.wanted() == ("neutral", "listening")       # back to normal afterwards


def test_other_sentences_are_not_taken(ft):
    assert ft.handle("what time is it") is None
    assert ft.handle("can you smile at my brother") is None


# ---- quick responder: friendly replies use the happy talking mouth ----------------------------
def test_quick_responder_reports_the_kind(monkeypatch):
    qr = QuickResponder()
    assert qr.match("Hello Piper") and qr.last_kind == "greeting"
    assert qr.match("what time is it") and qr.last_kind == "time"
    assert qr.match("tell me about rockets") is None and qr.last_kind is None
    assert "greeting" in QuickResponder.WARM_KINDS and "time" not in QuickResponder.WARM_KINDS
