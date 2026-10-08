"""
Piper Assistant: State-driven interactive runtime loop with LangGraph Supervisor.

    python3 src/main.py            normal: listens for the wake word, greets people she sees
    python3 src/main.py --quiet    do not disturb: the microphone is never opened and Piper
                                   never speaks (also /quiet); research, vision and the head
                                   still run. 'q' + Enter exits either way.
"""

import sys
import os
import time
import select
import threading

os.environ["ORT_LOGGING_LEVEL"] = "3"

from langchain_core.messages import HumanMessage, AIMessage

from piper_audio.listener import PiperListener
from piper_brain.config import CONFIG
from piper_brain.quick_responder import QuickResponder
from piper_brain.supervisor import PiperSupervisor, PiperBrainState
from piper_brain.state import AgentState, create_initial_state, append_and_truncate_message
from piper_brain.journal import ActivityJournal
from piper_head.client import HeadClient
from piper_skills import Context
from piper_skills.meet_person import MeetPerson, is_forget_request
from piper_vision.client import VisionClient

ENGAGED_TIMEOUT_SECONDS = float(CONFIG["assistant"]["engaged_timeout_seconds"])
running = True


head = HeadClient()        # the light ring on piper-watch (silently does nothing if it isn't there)


def set_status(state: AgentState, status: str, mood: str = "neutral"):
    """Change the assistant's state and show it on the head's light ring."""
    state["status"] = status
    head.assistant_state(status, mood)


def make_speaker():
    """The TTS engine named by audio.tts_engine: "kokoro" (GPU) or "piper" (CPU).
    (main() used to build a PiperSpeaker regardless, so Kokoro was never used.)"""
    engine = CONFIG["audio"]["tts_engine"].lower()
    if engine == "kokoro":
        from piper_audio.kokoro_speaker import KokoroSpeaker
        return KokoroSpeaker()
    from piper_audio.speaker import PiperSpeaker
    return PiperSpeaker()


def keyboard_monitor(journal: ActivityJournal):
    """Background thread watching for 'q' or 'exit' on stdin."""
    global running
    print("\n[Controls] Type 'q' and press [Enter] to exit cleanly.\n")
    while running:
        if select.select([sys.stdin], [], [], 0.5)[0]:
            line = sys.stdin.readline().strip().lower()
            if line in ("q", "quit", "exit"):
                print("\n[System] Exit signal received from keyboard.")
                journal.log("SHUTDOWN", "Operator requested exit via keyboard CLI.")
                running = False
                break

def quiet_requested(argv: list[str]) -> bool:
    return any(a.lower() in ("--quiet", "/quiet", "-q") for a in argv)


def run_quiet():
    """Do not disturb: no microphone, no speech - wait for 'q' (research runs separately)."""
    global running
    journal = ActivityJournal()
    head.assistant_state("IDLE", "neutral")
    print("--- Piper Assistant: QUIET mode (not listening, not speaking) ---")
    journal.log("SYSTEM", "Piper started in quiet mode.", "Microphone closed; no speech.")
    threading.Thread(target=keyboard_monitor, args=(journal,), daemon=True).start()
    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    journal.log("SYSTEM", "Piper assistant (quiet mode) terminated.")
    print("--- Shutting Down Piper Assistant ---")


def main():
    global running
    if quiet_requested(sys.argv[1:]):
        run_quiet()
        return
    print("--- Starting Piper Assistant ---")
    speaker = make_speaker()
    listener = PiperListener()
    quick_responder = QuickResponder()
    supervisor = PiperSupervisor()
    journal = ActivityJournal()
    
    state: AgentState = create_initial_state()
    set_status(state, "IDLE")

    # Who is Piper talking to? With vision, whoever she recognises (and she won't help anyone
    # she doesn't know by name - piper_skills/meet_person.py). Without it, the old default.
    vision = VisionClient()
    active_user = None

    def say(text: str, mood: str = "neutral"):
        print(f"[Piper]: {text}")
        set_status(state, "SPEAKING", mood)
        speaker.speak(text)
        set_status(state, "ENGAGED", mood)

    def hear() -> str:
        text = listener.listen_command_window(max_duration=8.0, silence_timeout=1.0)
        text = listener._strip_wake_word(text) if text else ""
        if text:
            print(f"[User]: {text}")
        return text

    meet = MeetPerson(Context(say=say, hear=hear, vision=vision, log=journal.log),
                      welcome_back_s=60 * float(CONFIG["assistant"]["welcome_back_minutes"]),
                      required=bool(CONFIG["assistant"]["require_known_person"]))

    def engage(name: str):
        """Piper has just greeted / met `name`: listen for them without the wake word."""
        nonlocal active_user
        active_user = name
        set_status(state, "ENGAGED")
        state["last_interaction_time"] = time.time()

    journal.log("SYSTEM", "Piper assistant runtime initialized.", f"Inactivity timeout: {ENGAGED_TIMEOUT_SECONDS:g}s | Memory window: {CONFIG['assistant']['max_conversation_turns']} turns | TTS: {CONFIG['audio']['tts_engine']}")

    kb_thread = threading.Thread(target=keyboard_monitor, args=(journal,), daemon=True)
    kb_thread.start()

    speaker.speak("Piper assistant is online and ready.")

    while running:
        try:
            # 1. Inactivity timeout check
            now = time.time()
            if state["status"] == "ENGAGED" and (now - state["last_interaction_time"] > ENGAGED_TIMEOUT_SECONDS):
                print(f"\n[State Transition] ENGAGED -> IDLE ({ENGAGED_TIMEOUT_SECONDS}s inactivity reached)")
                journal.log("STATE", "ENGAGED -> IDLE", f"Inactivity window exceeded ({ENGAGED_TIMEOUT_SECONDS}s).")
                set_status(state, "IDLE")

            # 2. State-dependent listening strategy
            if state["status"] == "IDLE":
                # someone new in view, or someone back? (meets / welcomes them - no wake word)
                name = meet.on_idle()
                if name:
                    engage(name)
                    continue

                raw_text = listener.listen_for_wake_word_and_command(max_command_duration=6.0)
                if not raw_text:
                    continue

                print("\n[State Transition] IDLE -> ENGAGED (Wake word detected)")
                journal.log("STATE", "IDLE -> ENGAGED", f"Wake-word triggered with: '{raw_text}'")
                speaker.play_chime()
                set_status(state, "ENGAGED")
                state["last_interaction_time"] = time.time()

                # Strip wake phrase to isolate actual command if spoken together
                user_text = listener._strip_wake_word(raw_text)
                if not user_text:
                    # Only the wake-word was spoken ("Hi Piper")
                    user_text = raw_text
            else:
                print("[Listener] In conversation (listening for follow-up)...")
                user_text = listener.listen_command_window(max_duration=6.0, silence_timeout=1.0)
                if not user_text:
                    continue
                state["last_interaction_time"] = time.time()

            if not running or not user_text:
                continue

            print(f"\n[User]: {user_text}")

            # 2b. Only people Piper knows by name get an answer
            go_ahead, name = meet.gate(user_text)
            if name:
                active_user = name
            if not go_ahead:
                if name:
                    engage(name)
                else:
                    set_status(state, "IDLE")
                continue
            if meet.handle_command(user_text, active_user):
                if is_forget_request(user_text):
                    active_user = None
                    set_status(state, "IDLE")
                else:
                    set_status(state, "ENGAGED")
                state["last_interaction_time"] = time.time()
                continue

            append_and_truncate_message(state, HumanMessage(content=user_text))

            # 3. Fast-Path Local Intent Check
            quick_reply = quick_responder.match(user_text)
            if quick_reply:
                print(f"[Piper (Local)]: {quick_reply}")
                journal.log("INTENT_LOCAL", f"Matched '{user_text}'", f"Replied: '{quick_reply}'")
                
                set_status(state, "SPEAKING")
                speaker.speak(quick_reply)
                append_and_truncate_message(state, AIMessage(content=quick_reply))
                
                if any(k in user_text.lower() for k in ["goodbye", "bye", "see you"]):
                    set_status(state, "IDLE")
                    journal.log("STATE", "ENGAGED -> IDLE", "User issued dismissal.")
                    print("[State Transition] ENGAGED -> IDLE (Dismissed)")
                elif any(k in user_text.lower() for k in ["shut down", "exit"]):
                    journal.log("SHUTDOWN", "Voice shutdown issued.")
                    running = False
                    break
                else:
                    set_status(state, "ENGAGED")
                    state["last_interaction_time"] = time.time()
                continue

            # 4. LangGraph Supervisor (Remote Ollama)
            set_status(state, "PROCESSING")
            print(f"[Piper (Supervisor Processing)]: Escalating '{user_text}'...")
            journal.log("INTENT_LLM", f"Escalated prompt to Ollama: '{user_text}'")

            brain_state: PiperBrainState = {
                "mode": "ENGAGED",
                "active_user": active_user or ("Steve" if not meet.active() else None),
                "input_text": user_text,
                "output_text": None,
                "user_context": "",
                "messages": state["messages"],
                "introspection_topic": None,
                "introspection_result": None
            }

            result = supervisor.process(brain_state)
            reply_text = result.get("output_text") or "I processed your request."
            state["messages"] = result.get("messages", state["messages"])
            llm_failed = bool(result.get("error"))

            print(f"[Piper (LLM Reply)]: {reply_text}")
            journal.log("REPLY_LLM", f"Generated: '{reply_text}'")

            if llm_failed:                                  # e.g. Ollama unreachable
                head.face("error", "concerned")
                time.sleep(1.0)
            set_status(state, "SPEAKING", "concerned" if llm_failed else "neutral")
            speaker.speak(reply_text)

            set_status(state, "ENGAGED")
            state["last_interaction_time"] = time.time()

        except KeyboardInterrupt:
            journal.log("SHUTDOWN", "SIGINT caught.")
            running = False
            break

    print("--- Shutting Down Piper Assistant ---")
    journal.log("SYSTEM", "Piper assistant terminated.")
    speaker.speak("Goodbye.")
    sys.exit(0)

if __name__ == "__main__":
    main()