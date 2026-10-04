"""
Piper Brain: Global Agent State Definition with Conversation Windowing.
"""

from typing import TypedDict, Literal, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from piper_brain.config import CONFIG

StatusType = Literal["IDLE", "ENGAGED", "PROCESSING", "SPEAKING"]

# Messages kept (8 = 4 user turns + 4 replies), from assistant.max_conversation_turns
MAX_CONVERSATION_TURNS = int(CONFIG["assistant"]["max_conversation_turns"])

class AgentState(TypedDict):
    messages: List[BaseMessage]
    status: StatusType
    last_interaction_time: float
    user_intent: str | None

def create_initial_state() -> AgentState:
    return {
        "messages": [],
        "status": "IDLE",
        "last_interaction_time": 0.0,
        "user_intent": None
    }

def append_and_truncate_message(state: AgentState, message: BaseMessage, max_turns: int = MAX_CONVERSATION_TURNS):
    """Appends a dialogue turn and ensures message history does not exceed context limits."""
    state["messages"].append(message)
    if len(state["messages"]) > max_turns:
        # Slice off oldest turns while keeping the latest context
        state["messages"] = state["messages"][-max_turns:]