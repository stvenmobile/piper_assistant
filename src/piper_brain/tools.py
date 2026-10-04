"""
Piper Brain: Dynamic context and local environment helpers.
"""

from datetime import datetime
import json
import time
import urllib.parse
import urllib.request

from piper_brain.config import CONFIG

FAILED_RETRY_SECONDS = 60          # after a failed fetch, try again this soon

# location -> (time fetched, summary, ok)
_weather_cache: dict[str, tuple[float, str, bool]] = {}


def get_current_datetime_str() -> str:
    """Returns formatted local date and time."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def _fetch_weather(location: str) -> str:
    """One request to wttr.in's JSON API; raises on any failure."""
    url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "PiperAssistant/1.0"})
    with urllib.request.urlopen(req, timeout=3.5) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    current = data["current_condition"][0]
    temp_f = current["temp_F"]
    feels_like_f = current["FeelsLikeF"]
    desc = current["weatherDesc"][0]["value"]
    humidity = current["humidity"]
    return f"{desc}, {temp_f}°F (feels like {feels_like_f}°F) with {humidity}% humidity in {location.replace(',', ', ')}"


def get_local_weather(location: str | None = None) -> str:
    """Concise current conditions, cached for weather.cache_minutes (it used to be fetched on
    every LLM turn). A failed fetch is retried after a minute rather than cached for long."""
    weather_cfg = CONFIG["weather"]
    location = location or weather_cfg["location"]
    now = time.time()
    cached = _weather_cache.get(location)
    if cached:
        fetched_at, summary, ok = cached
        max_age = weather_cfg["cache_minutes"] * 60 if ok else FAILED_RETRY_SECONDS
        if now - fetched_at < max_age:
            return summary

    try:
        summary, ok = _fetch_weather(location), True
    except Exception:
        summary, ok = "Unavailable (Network timeout or offline)", False
    _weather_cache[location] = (now, summary, ok)
    return summary
