"""
Piper Brain: Dynamic context and local environment helpers.
"""

from datetime import datetime
import urllib.request
import json
from pathlib import Path
import yaml

VAULT_DIR = Path(__file__).resolve().parents[2] / "obsidian"
EXPERIMENTS_DIR = VAULT_DIR / "Experiments"

def get_latest_experiment_summary() -> str:
    """Reads the most recent experiment note from the Obsidian vault and returns a spoken summary."""
    if not EXPERIMENTS_DIR.exists():
        return "I haven't recorded any geometry experiments in the vault yet."

    notes = sorted(EXPERIMENTS_DIR.glob("EXP-*.md"), reverse=True)
    if not notes:
        return "No recent experiments found in the research vault."

    latest_file = notes[0]
    content = latest_file.read_text(encoding="utf-8")

    try:
        # Extract YAML frontmatter
        parts = content.split("---")
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            exp_id = metadata.get("id", latest_file.stem)
            concept = metadata.get("target_concept", "Concept transfer")
            accuracy = metadata.get("accuracy", None)
            sim = metadata.get("cosine_similarity", 0.0)
            status = "successful" if metadata.get("transfer_success") else "inconclusive"

            if accuracy is not None:
                return (
                    f"In my latest experiment, {exp_id}, testing {concept}, "
                    f"zero-shot transfer was {status} with {accuracy * 100:.1f} percent accuracy "
                    f"and an average cosine alignment of {sim:.3f}."
                )
            return (
                f"In my latest experiment, {exp_id}, testing {concept}, "
                f"the transfer was {status} with a cosine similarity of {sim:.3f}."
            )
    except Exception as e:
        return f"I completed an experiment recently, but encountered an error reading the log: {e}"

    return "Latest experiment log found, but could not parse the summary."


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