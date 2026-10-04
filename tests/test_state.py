from langchain_core.messages import AIMessage, HumanMessage

from piper_brain.state import (MAX_CONVERSATION_TURNS, append_and_truncate_message,
                               create_initial_state)


def test_initial_state_is_idle():
    state = create_initial_state()
    assert state["status"] == "IDLE"
    assert state["messages"] == []


def test_history_is_truncated_to_the_newest_messages():
    state = create_initial_state()
    for i in range(MAX_CONVERSATION_TURNS + 3):
        msg = HumanMessage(content=f"q{i}") if i % 2 == 0 else AIMessage(content=f"a{i}")
        append_and_truncate_message(state, msg)
    assert len(state["messages"]) == MAX_CONVERSATION_TURNS
    assert state["messages"][-1].content == f"q{MAX_CONVERSATION_TURNS + 2}"


def test_window_comes_from_config():
    from piper_brain.config import CONFIG
    assert MAX_CONVERSATION_TURNS == CONFIG["assistant"]["max_conversation_turns"]
