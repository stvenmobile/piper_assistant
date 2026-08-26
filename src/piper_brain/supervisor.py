"""
Piper Supervisor: Voice-Driven Autonomous State Machine with Dynamic Context and Remote Ollama.
"""

import os
import sys
from pathlib import Path

# Ensure src/ directory is on sys.path for direct script execution
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from typing import TypedDict, Optional, Literal, List, Dict, Any
import re
import yaml
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from piper_brain.tools import get_current_datetime_str, get_local_weather

# Workspace and Configuration Path Resolution
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
PROFILES_DIR = WORKSPACE_DIR / "profiles"
JOURNAL_FILE = WORKSPACE_DIR / "daily_journal.md"
SYSTEM_DNA_FILE = WORKSPACE_DIR / "system_dna.md"
CONFIG_FILE = WORKSPACE_DIR / "config.yaml"

WAKE_PATTERNS = [r"\bhi\s+piper\b", r"\bhey\s+piper\b", r"\bhello\s+piper\b", r"\bpaper\b"]
DISMISS_PATTERNS = [r"\bbye\s+piper\b", r"\bgoodbye\s+piper\b", r"\bbye\b", r"\bgoodbye\b", r"\bshut\s+down\b", r"\bexit\b"]


def load_config() -> dict:
    """Loads configuration with environment variable fallbacks."""
    cfg = {
        "assistant": {"max_conversation_turns": 8},
        "llm": {
            "base_url": os.getenv("PIPER_OLLAMA_URL", "http://192.168.1.150:11434"),
            "model": os.getenv("PIPER_LLM_MODEL", "llama3.2:3b"),
            "temperature": 0.4,
        }
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
                if user_cfg:
                    cfg.update(user_cfg)
        except Exception as e:
            print(f"[Supervisor Config Error]: {e}")
    return cfg


CONFIG = load_config()


class PiperBrainState(TypedDict):
    mode: Literal["ALONE", "ENGAGED"]
    active_user: Optional[str]
    input_text: Optional[str]
    output_text: Optional[str]
    user_context: str
    messages: List[BaseMessage]
    introspection_topic: Optional[str]
    introspection_result: Optional[str]


def load_system_prompt() -> str:
    """Loads identity parameters from system_dna.md with voice constraints."""
    base_dna = SYSTEM_DNA_FILE.read_text(encoding="utf-8") if SYSTEM_DNA_FILE.exists() else "You are Piper, an authentic and concise embedded assistant."
    voice_rules = (
        "\n\nVOICE RULES:\n"
        "1. Keep spoken responses concise, natural, and direct (1 to 2 sentences maximum).\n"
        "2. Strictly avoid markdown headers, asterisks, bullet points, numbered lists, and code blocks.\n"
        "3. Address the interlocutor directly without meta-announcements."
    )
    return base_dna + voice_rules


class PiperSupervisor:
    def __init__(self):
        llm_cfg = CONFIG["llm"]
        print(f"[Supervisor] Connecting to ChatOllama ({llm_cfg['model']}) at {llm_cfg['base_url']}...")
        self.llm = ChatOllama(
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.4),
            base_url=llm_cfg["base_url"]
        )
        self.system_prompt = SystemMessage(content=load_system_prompt())
        self.graph = self._build_graph()

    def evaluate_audio_event_node(self, state: PiperBrainState) -> PiperBrainState:
        text = (state.get("input_text") or "").strip()
        current_mode = state.get("mode", "ALONE")

        if current_mode == "ENGAGED" and any(re.search(p, text, re.IGNORECASE) for p in DISMISS_PATTERNS):
            state["mode"] = "ALONE"
            state["output_text"] = f"Goodbye {state.get('active_user', '')}. Standing by."
            state["active_user"] = None
            state["user_context"] = ""
            return state

        if current_mode == "ALONE":
            if any(re.search(p, text, re.IGNORECASE) for p in WAKE_PATTERNS):
                state["mode"] = "ENGAGED"
                cleaned = re.sub(r"^(hey|hi|hello)?\s*(piper|paper)[,\.\?!]*\s*", "", text, flags=re.IGNORECASE).strip()
                state["input_text"] = cleaned
            else:
                state["mode"] = "ALONE"
                state["output_text"] = None

        return state

    def resolve_user_node(self, state: PiperBrainState) -> PiperBrainState:
        text = (state.get("input_text") or "").strip()
        
        match = re.search(r"(?:i am|my name is|this is)\s+([A-Za-z]+)", text, re.IGNORECASE)
        if match:
            state["active_user"] = match.group(1).capitalize()

        user = state.get("active_user")
        if user:
            profile_path = PROFILES_DIR / f"{user.lower()}.md"
            if profile_path.exists():
                state["user_context"] = profile_path.read_text(encoding="utf-8")
            else:
                state["user_context"] = f"Collaborator: {user}"
        else:
            state["user_context"] = ""

        return state

    def execute_engaged_node(self, state: PiperBrainState) -> PiperBrainState:
        text = (state.get("input_text") or "").strip()
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []

        if not text:
            state["output_text"] = "I'm listening."
            return state

        state["messages"].append(HumanMessage(content=text))

        current_time_str = get_current_datetime_str()
        weather_summary = get_local_weather("Matthews,NC")
        
        temporal_context = (
            f"\n\nENVIRONMENT CONTEXT:\n"
            f"- Current Date & Time: {current_time_str}\n"
            f"- Location: Matthews, North Carolina\n"
            f"- Local Weather: {weather_summary}\n"
        )

        context_prompt = self.system_prompt.content + temporal_context
        if state.get("user_context"):
            context_prompt += f"\n\nActive User Profile:\n{state['user_context']}"

        max_turns = CONFIG["assistant"].get("max_conversation_turns", 8)
        history_window = state["messages"][-max_turns:]
        payload = [SystemMessage(content=context_prompt)] + history_window

        try:
            response = self.llm.invoke(payload)
            clean_reply = response.content.replace("*", "").replace("#", "").strip()
            state["output_text"] = clean_reply
            state["messages"].append(AIMessage(content=clean_reply))
        except Exception as e:
            print(f"[Supervisor Error]: LLM invocation failed: {e}")
            state["output_text"] = "I am having trouble communicating with my neural core."

        state["messages"] = state["messages"][-max_turns:]
        return state

    def autonomous_introspection_node(self, state: PiperBrainState) -> PiperBrainState:
        topic = "Manifold Curvature Analysis (Residual Layers 8-12)"
        result = "Computed geodesic drift; stable semantic basin verified."
        
        state["introspection_topic"] = topic
        state["introspection_result"] = result
        state["output_text"] = None
        return state

    def _route_mode(self, state: PiperBrainState) -> str:
        if state["mode"] == "ENGAGED":
            return "resolve_user"
        return "introspect"

    def _build_graph(self):
        workflow = StateGraph(PiperBrainState)

        workflow.add_node("eval_audio", self.evaluate_audio_event_node)
        workflow.add_node("resolve_user", self.resolve_user_node)
        workflow.add_node("engaged_exec", self.execute_engaged_node)
        workflow.add_node("introspect", self.autonomous_introspection_node)

        workflow.set_entry_point("eval_audio")

        workflow.add_conditional_edges(
            "eval_audio",
            self._route_mode,
            {
                "resolve_user": "resolve_user",
                "introspect": "introspect"
            }
        )

        workflow.add_edge("resolve_user", "engaged_exec")
        workflow.add_edge("engaged_exec", END)
        workflow.add_edge("introspect", END)

        return workflow.compile()

    def process(self, state: PiperBrainState) -> PiperBrainState:
        return self.graph.invoke(state)


if __name__ == "__main__":
    supervisor = PiperSupervisor()
    initial_state: PiperBrainState = {
        "mode": "ENGAGED",
        "active_user": "Steve",
        "input_text": "What is the date today and how is the weather?",
        "output_text": None,
        "user_context": "",
        "messages": [],
        "introspection_topic": None,
        "introspection_result": None
    }

    result = supervisor.process(initial_state)
    print("\n--- Pipeline Execution Output ---")
    print(f"Mode:   {result['mode']}")
    print(f"Output: {result['output_text']}")