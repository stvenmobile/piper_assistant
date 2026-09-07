"""
Piper Brain: Deterministic quick-response router for low-latency boilerplate interactions.
"""

import re
import random
from datetime import datetime
from typing import Optional
from piper_brain.tools import get_local_weather, get_latest_experiment_summary

IDLE_RESEARCH_CONFIRMATIONS = [
    "Entering idle state. Resuming latent research.",
    "Standing down. Back to geometric signaling experiments.",
    "Understood. Resuming background manifold optimization.",
    "Entering idle mode. Resuming soft-prompt research."
]


class QuickResponder:
    def __init__(self):
        self.clean_re = re.compile(r"[^a-zA-Z0-9\s]")
        
        self.routes = [
            # Greetings: "hey piper", "hello", "hi piper"
            (
                r"^(hello|hi|hey|good morning|good afternoon|good evening)(\s+(piper|paper))?$",
                lambda _: random.choice([
                    "Hello! How can I help you?",
                    "Hey there! What are we working on?",
                    "Hi! I'm listening.",
                ])
            ),
            # Idle & Autonomous Research Transitions
            (
                r"^(go\s+to|enter|switch\s+to)\s+idle(\s+state|\s+mode)?(\s+for\s+research)?(\s+(piper|paper))?$",
                lambda _: random.choice(IDLE_RESEARCH_CONFIRMATIONS)
            ),
            (
                r"^(resume|continue|back\s+to)(\s+your)?\s+(research|experiments?|work)(\s+(piper|paper))?$",
                lambda _: random.choice(IDLE_RESEARCH_CONFIRMATIONS)
            ),
            # Farewells & Exit: "bye piper", "goodbye", "shut down"
            (
                r"^(goodbye|bye|see you|shut down|exit|stop listening|stand\s*down)(\s+(piper|paper))?$",
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

        # Normalize text and strip punctuation for regex matching
        cleaned = self.clean_re.sub("", text).strip().lower()
        prompt_lower = text.lower().strip()

        # 1. Fast-path research experiment query
        if re.search(r"\b(latest|recent)\s+(experiment|research|test results?)\b", prompt_lower):
            return get_latest_experiment_summary()

        # 2. Date queries
        if any(q in prompt_lower for q in ["what is today", "what's today", "what date", "what is the date", "today's date"]):
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # 3. Time queries
        if any(q in prompt_lower for q in ["what time is it", "what's the time", "current time"]):
            now = datetime.now()
            return f"It is currently {now.strftime('%I:%M %p')}."

        # 4. Weather queries
        if any(q in prompt_lower for q in ["what is the weather", "what's the weather", "current weather", "weather outside"]):
            weather = get_local_weather("Matthews,NC")
            return f"In Matthews, it is currently {weather}."

        # 5. Regex route table evaluation (Greetings, Idle Transitions, Status)
        for pattern, handler in self.routes:
            if re.match(pattern, cleaned, re.IGNORECASE):
                return handler(cleaned)

        return None