"""
Piper Brain: Deterministic quick-response router for low-latency boilerplate interactions.
"""

import re
import random
from datetime import datetime
from piper_brain.tools import get_local_weather


class QuickResponder:
    def __init__(self):
        # Strips punctuation, so Whisper's "Goodbye, Piper." becomes "goodbye piper"
        self.clean_re = re.compile(r"[^a-z0-9\s]")

        # Whole-utterance routes: (pattern matched against the cleaned text, reply)
        self.routes = [
            # Greetings: "hey piper", "hello", "hi paper"
            (
                r"(hello|hi|hey|good morning|good afternoon|good evening)(\s+(piper|paper))?",
                lambda: random.choice([
                    "Hello! How can I help you?",
                    "Hey there! What are we working on?",
                    "Hi! I'm listening.",
                ])
            ),
            # Farewells & exit: "bye piper", "goodbye", "shut down" (main.py acts on these)
            (
                r"(goodbye|bye|see you|shut down|exit|stop listening)(\s+(piper|paper))?",
                lambda: random.choice([
                    "Goodbye!",
                    "See you later.",
                    "Standing by.",
                ])
            ),
            # Conversational checks
            (
                r"(whats up|what is up|how are you|hows it going)(\s+(piper|paper))?",
                lambda: random.choice([
                    "All systems operational. What's on your mind?",
                    "Doing well, ready to assist.",
                    "Everything is running smoothly.",
                ])
            ),
            # Status check
            (
                r"(status|system status|ping)",
                lambda: "All local subsystems online and ready."
            ),
        ]

    def match(self, text: str) -> str | None:
        t = text.lower().strip()
        clean = " ".join(self.clean_re.sub("", t).split())

        # Date queries
        if any(q in t for q in ["what is today", "what's today", "what date", "what is the date", "today's date"]):
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # Time queries
        if any(q in t for q in ["what time is it", "what's the time", "current time"]):
            now = datetime.now()
            return f"It is currently {now.strftime('%I:%M %p')}."

        # Weather queries
        if any(q in t for q in ["what is the weather", "what's the weather", "current weather", "weather outside"]):
            return f"Right now it's {get_local_weather()}."

        # Whole-utterance routes (greetings, farewells, status) - these were defined but never
        # checked before, so "goodbye" went to the LLM and never ended the conversation
        for pattern, reply in self.routes:
            if re.fullmatch(pattern, clean):
                return reply()

        return None
