import pytest

from piper_brain import quick_responder
from piper_brain.quick_responder import QuickResponder


@pytest.fixture
def qr(monkeypatch):
    monkeypatch.setattr(quick_responder, "get_local_weather", lambda location=None: "Sunny, 70°F")
    return QuickResponder()


@pytest.mark.parametrize("text", ["What time is it?", "what's the time", "tell me the current time"])
def test_time(qr, text):
    assert qr.match(text).startswith("It is currently")


@pytest.mark.parametrize("text", ["What is the date?", "what's today", "today's date please"])
def test_date(qr, text):
    assert qr.match(text).startswith("Today is")


def test_weather(qr):
    assert qr.match("What's the weather like?") == "Right now it's Sunny, 70°F."


@pytest.mark.parametrize("text", ["Hey Piper.", "hello", "Good morning, Piper!", "Hi paper"])
def test_greetings(qr, text):
    assert qr.match(text)


@pytest.mark.parametrize("text", ["Goodbye, Piper.", "bye", "See you!", "Shut down."])
def test_farewells_are_answered_locally(qr, text):
    # main.py ends the conversation (or shuts down) only on a local reply
    assert qr.match(text) in ("Goodbye!", "See you later.", "Standing by.")


@pytest.mark.parametrize("text", [
    "Goodbye to that idea, tell me about black holes",
    "Hey, what is a manifold?",
    "Explain the status of the project",
])
def test_longer_requests_go_to_the_llm(qr, text):
    assert qr.match(text) is None
