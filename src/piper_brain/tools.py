"""
Piper Brain: Dynamic context and local environment helpers.
"""

from datetime import datetime
import urllib.request
import json

def get_current_datetime_str() -> str:
    """Returns formatted local date and time."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

def get_local_weather(location: str = "Matthews,NC") -> str:
    """Fetches concise real-time weather conditions via wttr.in JSON API."""
    url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "PiperAssistant/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        current = data["current_condition"][0]
        temp_f = current["temp_F"]
        feels_like_f = current["FeelsLikeF"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        
        return f"{desc}, {temp_f}°F (feels like {feels_like_f}°F) with {humidity}% humidity in {location.replace(',', ', ')}"
    except Exception as e:
        return f"Unavailable (Network timeout or offline)"