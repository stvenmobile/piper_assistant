"""
Messages between the Jetson and the piper-watch ESP32: newline-delimited JSON, one object per
line, each with a "t" (type) field. Unknown types and fields are ignored on both sides.

    Jetson -> ESP32   HEARTBEAT {seq}, FACE {state, mood?, attention?}, CONFIG {...}, LOOK {...}
    ESP32 -> Jetson   STATUS {...}, EVENT {what}

The same lines are spoken between the head link service and its local clients (the assistant,
vision, test tools).
"""
import json

STATES = ("offline", "idle", "listening", "thinking", "speaking", "sleeping", "error")
MOODS = ("neutral", "warm", "curious", "uncertain", "concerned", "sleepy")

# the assistant's states (src/main.py) -> what the ring shows
ASSISTANT_TO_FACE = {
    "IDLE": "idle",
    "ENGAGED": "listening",
    "PROCESSING": "thinking",
    "SPEAKING": "speaking",
}

_UNSET = object()


def encode(msg: dict) -> bytes:
    """One message -> one line of compact JSON."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line) -> dict | None:
    """One line -> a message dict, or None if it isn't a JSON object with a "t"."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) and "t" in msg else None


def heartbeat(seq: int) -> dict:
    return {"t": "HEARTBEAT", "seq": seq}


def face(state: str, mood: str | None = None, attention=_UNSET) -> dict:
    """FACE message. attention: degrees (0 = up, clockwise seen from the front), None to clear
    it, or leave it out to keep whatever the ring has."""
    if state not in STATES:
        raise ValueError(f"unknown face state {state!r}")
    msg = {"t": "FACE", "state": state}
    if mood is not None:
        if mood not in MOODS:
            raise ValueError(f"unknown mood {mood!r}")
        msg["mood"] = mood
    if attention is not _UNSET:
        msg["attention"] = None if attention is None else float(attention)
    return msg


def config(**fields) -> dict:
    return {"t": "CONFIG", **fields}
