"""
piper_research - Piper's research loop: in a research session (start_quiet.sh starts one,
research.session_hours long) or in timed windows (research.windows) she picks a topic from her memory, asks a question, reads Wikipedia, and stores
verified findings with their sources (piper_memory). See loop.py for one cycle, service.py for
the scheduler.
"""
from piper_research.llm import LLMError, OllamaChat
from piper_research.loop import Interrupted, Researcher
from piper_research.schedule import Schedule
from piper_research.wiki import WikiError, Wikipedia

__all__ = ["Researcher", "Schedule", "OllamaChat", "Wikipedia", "LLMError", "WikiError", "Interrupted"]
