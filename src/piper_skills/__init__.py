"""
Piper's skills: self-contained behaviours that hold a short conversation of their own (ask,
listen, act) instead of a single reply. Each skill gets a Context - the few things it may use -
so it can be unit-tested with a scripted fake conversation.

    meet_person   who are you? - the name gate, meeting new people, welcoming people back
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Context:
    say: Callable[..., None]                 # say(text, mood="neutral") - speak, ring shows mood
    hear: Callable[..., str]                 # hear() -> what was said ("" = nothing)
    vision: object                           # piper_vision.client.VisionClient (or a fake)
    log: Callable[..., None] = lambda *a, **k: None     # journal.log(category, message, details)
