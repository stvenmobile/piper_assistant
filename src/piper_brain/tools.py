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

    # Experiment notes are named "<track-prefix>-<timestamp>.md" (WLCOMM,
    # P3LOOP, ACCIT, ...) rather than a fixed "EXP-" prefix, and prefixes
    # sort alphabetically ahead of/behind each other regardless of when
    # they were written - so pick the most recent by mtime, not filename,
    # or the "latest" experiment silently stays stuck on whichever track
    # happens to sort last as a string.
    notes = sorted(EXPERIMENTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
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
            sim = metadata.get("cosine_similarity", 0.0)
            status = "successful" if metadata.get("transfer_success") else "inconclusive"

            # tester.py's records store a 0-1 fraction under "accuracy";
            # supervisor.py's autonomous trial notes store an already-scaled
            # percentage under "top1_accuracy" - normalize both to a percent
            # so this doesn't silently drop the far more common case (every
            # autonomous trial) down to the cosine-only fallback below.
            accuracy_fraction = metadata.get("accuracy", None)
            accuracy_pct = accuracy_fraction * 100 if accuracy_fraction is not None else metadata.get("top1_accuracy", None)

            if accuracy_pct is not None:
                return (
                    f"In my latest experiment, {exp_id}, testing {concept}, "
                    f"zero-shot transfer was {status} with {accuracy_pct:.1f} percent accuracy "
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