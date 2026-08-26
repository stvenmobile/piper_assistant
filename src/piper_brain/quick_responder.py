"""
Piper Brain: Deterministic quick-response router for low-latency boilerplate interactions.
"""

import re
import random
from datetime import datetime
from typing import Optional
from datetime import datetime
from piper_brain.tools import get_current_datetime_str, get_local_weather

class QuickResponder:
    def __init__(self):
        # Clean text stripping regex
        self.clean_re = re.compile(r"[^a-zA-Z0-9\s]")
        
        self.routes = [
            # Greetings: "hey piper", "hello", "hi paper"
            (
                r"^(hello|hi|hey|good morning|good afternoon|good evening)(\s+(piper|paper))?$",
                lambda _: random.choice([
                    "Hello! How can I help you?",
                    "Hey there! What are we working on?",
                    "Hi! I'm listening.",
                ])
            ),
            # Farewells & Exit: "bye piper", "goodbye", "shut down"
            (
                r"^(goodbye|bye|see you|shut down|exit|stop listening)(\s+(piper|paper))?$",
                lambda _: random.choice([
                    "Goodbye!",
                    "See you later.",
                    "Standing by."
                ])
            ),
            # Conversational checks
            (
                r"^(whats up|what is up|how are you|hows it going)(\s+(piper|paper))?$",
                lambda _: random.choice([
                    "All systems operational. What's on your mind?",
                    "Doing well, ready to assist.",
                    "Everything is running smoothly."
                ])
            ),
            # Status check
            (
                r"^(status|system status|ping)$",
                lambda _: "All local subsystems online and ready."
            ),
            # Clock & Time
            (
                r"^what time is it$",
                lambda _: f"It is currently {datetime.now().strftime('%I:%M %p')}."
            ),
            # Date
            (
                r"^what is (todays date|the date)$",
                lambda _: f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
            ),
        ]

    def match(self, text: str) -> str | None:
        t = text.lower().strip()
        
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
            weather = get_local_weather("Matthews,NC")
            return f"In Matthews, it is currently {weather}."

        return None