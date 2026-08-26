"""
Piper Brain: Deterministic quick-response router for low-latency boilerplate interactions.
"""

import re
import random
from datetime import datetime
from typing import Optional

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

    def match(self, text: str) -> Optional[str]:
        if not text:
            return None

        # Normalize text: strip punctuation and lowercase
        cleaned = self.clean_re.sub("", text.lower()).strip()

        for pattern, handler in self.routes:
            if re.match(pattern, cleaned):
                return handler(cleaned)

        return None