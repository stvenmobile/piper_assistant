"""
Piper Supervisor: Voice-Driven Autonomous State Machine.
"""

from typing import TypedDict, List, Dict, Optional, Literal
from pathlib import Path
import re
from langgraph.graph import StateGraph, END

WORKSPACE_DIR = Path(__file__).resolve().parents[3]
PROFILES_DIR = WORKSPACE_DIR / "profiles"
TASK_REQUESTS_FILE = WORKSPACE_DIR / "task_requests.md"
JOURNAL_FILE = WORKSPACE_DIR / "daily_journal.md"

WAKE_PATTERNS = [r"\bhi\s+piper\b", r"\bhey\s+piper\b", r"\bhello\s+piper\b"]
DISMISS_PATTERNS = [r"\bbye\s+piper\b", r"\bgoodbye\s+piper\b", r"\bbye\b", r"\bgoodbye\b"]

class PiperBrainState(TypedDict):
    mode: Literal["ALONE", "ENGAGED"]
    active_user: Optional[str]
    input_text: Optional[str]
    output_text: Optional[str]
    user_context: str
    introspection_topic: Optional[str]
    introspection_result: Optional[str]

# ----------------- Nodes ----------------- #

def evaluate_audio_event_node(state: PiperBrainState) -> PiperBrainState:
    """Evaluates STT text against wake and dismiss triggers."""
    text = (state.get("input_text") or "").strip().lower()
    current_mode = state.get("mode", "ALONE")

    # Check for dismiss command first
    if current_mode == "ENGAGED" and any(re.search(p, text) for p in DISMISS_PATTERNS):
        state["mode"] = "ALONE"
        state["output_text"] = f"Goodbye {state.get('active_user', '')}. Resuming background introspection."
        state["active_user"] = None
        state["user_context"] = ""
        return state

    # Check for wake command when ALONE
    if current_mode == "ALONE":
        if any(re.search(p, text) for p in WAKE_PATTERNS):
            state["mode"] = "ENGAGED"
            # Strip the wake phrase to check if identity/command was passed in the same burst
            cleaned_text = re.sub(r"\b(hi|hey|hello)\s+piper\b", "", text).strip()
            state["input_text"] = cleaned_text
        else:
            # Ambient noise or irrelevant speech: stay ALONE
            state["mode"] = "ALONE"
            state["output_text"] = None
            return state

    return state

def resolve_user_node(state: PiperBrainState) -> PiperBrainState:
    """Extracts speaker identity or prompts for introduction."""
    text = (state.get("input_text") or "").strip()
    
    # Check if speaker introduces themselves
    match = re.search(r"(?:i am|my name is|this is)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if match:
        state["active_user"] = match.group(1).capitalize()
        
    user = state.get("active_user")
    if user:
        profile_path = PROFILES_DIR / f"{user.lower()}.md"
        if profile_path.exists():
            state["user_context"] = profile_path.read_text()
        else:
            state["user_context"] = f"Collaborator: {user}"
    else:
        state["user_context"] = ""

    return state

def execute_engaged_node(state: PiperBrainState) -> PiperBrainState:
    """Processes interlocutor requests while engaged."""
    user = state.get("active_user")
    text = (state.get("input_text") or "").strip()

    if not user:
        state["output_text"] = "Hello! I am Piper. Who am I speaking with?"
        return state

    if not text:
        state["output_text"] = f"I'm listening, {user}."
    else:
        # LLM inference / directive execution hook
        state["output_text"] = f"Acknowledged {user}. Processing: '{text}'"

    return state

def autonomous_introspection_node(state: PiperBrainState) -> PiperBrainState:
    """Executes background latent geometric exploration."""
    topic = "Manifold Curvature Analysis (Residual Layers 8-12)"
    result = "Computed geodesic drift; detected low-entropy semantic cluster."
    
    state["introspection_topic"] = topic
    state["introspection_result"] = result
    
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, "a") as f:
            f.write(f"\n- **Introspection [{topic}]**: {result}")
            
    state["output_text"] = None
    return state

# ----------------- Router & Graph ----------------- #

def route_mode(state: PiperBrainState) -> str:
    if state["mode"] == "ENGAGED":
        return "resolve_user"
    return "introspect"

def build_piper_supervisor_graph():
    workflow = StateGraph(PiperBrainState)

    workflow.add_node("eval_audio", evaluate_audio_event_node)
    workflow.add_node("resolve_user", resolve_user_node)
    workflow.add_node("engaged_exec", execute_engaged_node)
    workflow.add_node("introspect", autonomous_introspection_node)

    workflow.set_entry_point("eval_audio")

    workflow.add_conditional_edges(
        "eval_audio",
        route_mode,
        {
            "resolve_user": "resolve_user",
            "introspect": "introspect"
        }
    )

    workflow.add_edge("resolve_user", "engaged_exec")
    workflow.add_edge("engaged_exec", END)
    workflow.add_edge("introspect", END)

    return workflow.compile()