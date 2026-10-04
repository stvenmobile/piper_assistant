import pytest

from piper_brain import tools


@pytest.fixture
def fake_weather(monkeypatch):
    """Replaces the network fetch with a counter; returns the list of fetched locations."""
    calls = []
    clock = {"now": 1000.0}

    def fetch(location):
        calls.append(location)
        if fetch.fail:
            raise OSError("offline")
        return f"Sunny in {location}"

    fetch.fail = False
    monkeypatch.setattr(tools, "_fetch_weather", fetch)
    monkeypatch.setattr(tools.time, "time", lambda: clock["now"])
    monkeypatch.setattr(tools, "_weather_cache", {})
    return calls, clock, fetch


def test_weather_is_cached(fake_weather):
    calls, clock, _ = fake_weather
    assert tools.get_local_weather("Here") == "Sunny in Here"
    clock["now"] += 60
    assert tools.get_local_weather("Here") == "Sunny in Here"
    assert calls == ["Here"]


def test_weather_refetched_after_cache_expires(fake_weather):
    calls, clock, _ = fake_weather
    tools.get_local_weather("Here")
    clock["now"] += tools.CONFIG["weather"]["cache_minutes"] * 60 + 1
    tools.get_local_weather("Here")
    assert calls == ["Here", "Here"]


def test_failed_fetch_retried_after_a_minute(fake_weather):
    calls, clock, fetch = fake_weather
    fetch.fail = True
    assert tools.get_local_weather("Here").startswith("Unavailable")
    clock["now"] += 30
    tools.get_local_weather("Here")
    assert len(calls) == 1                     # still within the retry delay
    fetch.fail = False
    clock["now"] += tools.FAILED_RETRY_SECONDS
    assert tools.get_local_weather("Here") == "Sunny in Here"
    assert len(calls) == 2


def test_default_location_comes_from_config(fake_weather):
    calls, _, _ = fake_weather
    tools.get_local_weather()
    assert calls == [tools.CONFIG["weather"]["location"]]
