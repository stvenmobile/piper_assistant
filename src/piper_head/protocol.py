"""
Messages between the Jetson and the piper-watch ESP32: newline-delimited JSON, one object per
line, each with a "t" (type) field. Unknown types and fields are ignored on both sides.

    Jetson -> ESP32   HEARTBEAT {seq}, FACE {state, mood?, attention?}, MOUTH {level?, mode?, symbol?, demo?},
                      CONFIG {...}, LOOK {...}
    ESP32 -> Jetson   STATUS {...}, EVENT {what}

The same lines are spoken between the head link service and its local clients (the assistant,
vision, test tools).

The mouth OLED has its own XIAO ESP32-S3 (piper-watch firmware/mouth) on its own USB port. It
speaks plain text lines, not JSON; the link translates MOUTH messages for it (mouth_lines):

    M<mode>      neutral, talk, happytalk, smile, frown, surprise, sleep, happysleep
    I<symbol>    the status symbol, top right: idle (none), listening, thinking, confused
    L<0..255>    speech loudness (talk modes only; without it the mouth simulates speech)
    D1 / D0      its demo loop on / off
    ?            it answers "mouth: ..." - how the link tells it from the head controller
"""
import json

STATES = ("offline", "idle", "listening", "thinking", "speaking", "sleeping", "error")
MOODS = ("neutral", "warm", "curious", "uncertain", "concerned", "sleepy")
MOUTH_MODES = ("neutral", "talk", "happytalk", "smile", "frown", "surprise", "sleep", "happysleep")
MOUTH_SYMBOLS = ("idle", "listening", "thinking", "confused")

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


def attention(deg: float | None) -> dict:
    """Move (or, with None, clear) the ring's attention arc without touching its state/mood -
    what the vision service sends. The ESP32 and the link keep the state they have."""
    return {"t": "FACE", "attention": None if deg is None else float(deg)}


def mouth(level: float | None = None, mode: str | None = None, symbol: str | None = None,
          demo: bool | None = None) -> dict:
    """MOUTH message - any of:
    level   speech loudness 0..1 while speaking (~20 times a second; optional - without it the
            mouth simulates speech in its talk modes, and the ring pulses while "speaking")
    mode    the mouth's expression (MOUTH_MODES)
    symbol  the status symbol in its top-right corner (MOUTH_SYMBOLS)
    demo    True / False: the mouth's own demo loop"""
    msg = {"t": "MOUTH"}
    if level is not None:
        msg["level"] = round(max(0.0, min(1.0, float(level))), 3)
    if mode is not None:
        if mode not in MOUTH_MODES:
            raise ValueError(f"unknown mouth mode {mode!r}")
        msg["mode"] = mode
    if symbol is not None:
        if symbol not in MOUTH_SYMBOLS:
            raise ValueError(f"unknown mouth symbol {symbol!r}")
        msg["symbol"] = symbol
    if demo is not None:
        msg["demo"] = bool(demo)
    return msg


def mouth_lines(msg: dict) -> list[str]:
    """A MOUTH message -> the mouth XIAO's text commands (unknown values are dropped)."""
    lines = []
    if "demo" in msg:
        lines.append("D1" if msg["demo"] else "D0")
    if msg.get("mode") in MOUTH_MODES:
        lines.append("M" + msg["mode"])
    if msg.get("symbol") in MOUTH_SYMBOLS:
        lines.append("I" + msg["symbol"])
    if isinstance(msg.get("level"), (int, float)):
        lines.append(f"L{round(max(0.0, min(1.0, float(msg['level']))) * 255)}")
    return lines


def config(**fields) -> dict:
    return {"t": "CONFIG", **fields}
