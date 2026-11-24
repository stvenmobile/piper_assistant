"""
Messages from local clients (the assistant, tools) to the head link service: newline-delimited
JSON, one object per line, each with a "t" (type) field - MOUTH {level?, mode?, symbol?, demo?}.
Unknown types and fields are ignored.

The mouth OLED's XIAO ESP32-S3 (piper-watch firmware/mouth) speaks plain text lines, not JSON;
the link translates MOUTH messages for it (mouth_lines):

    M<mode>      neutral, talk, happytalk, excitedtalk, smile, frown, surprise, sleep, happysleep,
                 research ("Research in progress..."), blank (dark screen)
                 (excitedtalk: the face panel's full-height speech bars; talk / happytalk use half)
    I<symbol>    the status symbol, top right: idle (none), listening, thinking, confused
    L<0..255>    speech loudness (talk modes only; without it the mouth simulates speech)
    D1 / D0      its demo loop on / off
    S<style>     lips or spectrum (the link sends head.mouth_style when the mouth connects)
    ?            it answers "mouth: ..." - how the link recognises it

Version 1.1.0 - created before 2026-11-24 (see git history), last modified 2026-11-24
History:
    1.1.0  2026-11-24  the "excitedtalk" mouth mode (full-height speech bars on the face panel).
    1.0.0  -           earlier work, unversioned (git history)
"""
import json

MOUTH_MODES = ("neutral", "talk", "happytalk", "excitedtalk", "smile", "frown", "surprise", "sleep", "happysleep",
               "research", "blank")
MOUTH_SYMBOLS = ("idle", "listening", "thinking", "confused")

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


def mouth(level: float | None = None, mode: str | None = None, symbol: str | None = None,
          demo: bool | None = None) -> dict:
    """MOUTH message - any of:
    level   speech loudness 0..1 while speaking (~20 times a second; optional - without it the
            mouth simulates speech in its talk modes)
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

