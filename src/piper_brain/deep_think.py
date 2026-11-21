"""
"Piper, think carefully: <question>" - the question goes to the research model (qwen3:30b-a3b on
QUANTUM) with its thinking mode on, instead of the quick chat model. Slower (~10-15 s, plus loading
the model if the chat model was in its place) but better reasoned. Only the answer is spoken; the
model's thinking is kept apart by Ollama.

    question(text) -> the question after the "think carefully" words, or None
    answer(question, history, system) -> the spoken answer, or None if the model can't be reached
"""
import json
import re
import urllib.request

from piper_brain.config import CONFIG

_LEAD = re.compile(
    r"^\s*(?:(?:hey|ok|okay)\s+)?(?:piper|paper)?[\s,]*(?:please\s+)?"
    r"(?:think\s+(?:about\s+(?:this|that|it)\s+)?(?:carefully|hard|properly|deeply)|take\s+your\s+time)"
    r"(?:\s+about\s+(?:this|that|it))?\s*[,:.!-]*\s*",
    re.IGNORECASE)

SPOKEN = " Answer for speaking aloud: plain sentences, no lists, no markdown, at most four sentences."


def question(text: str) -> str | None:
    m = _LEAD.match(text or "")
    if not m or m.end() == 0:
        return None
    rest = text[m.end():].strip()
    return rest or None


def answer(q: str, history=(), system: str = "", cfg: dict | None = None, timeout: float = 240.0) -> str | None:
    cfg = cfg or CONFIG["research"]
    url = (cfg.get("url") or CONFIG["llm"]["base_url"]).rstrip("/") + "/api/chat"
    msgs = [{"role": "system", "content": (system or "You are Piper, a friendly home robot.") + SPOKEN}]
    for m in list(history)[-8:]:                       # langchain messages or {"role","content"} dicts
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if role in ("human", "user") and content:
            msgs.append({"role": "user", "content": content})
        elif role in ("ai", "assistant") and content:
            msgs.append({"role": "assistant", "content": content})
    msgs.append({"role": "user", "content": q})
    body = {"model": cfg["model"], "messages": msgs, "stream": False, "think": True,
            "keep_alive": cfg.get("keep_alive", "30m"), "options": {"num_predict": 2500, "temperature": 0.4}}
    try:
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = (json.load(r).get("message") or {}).get("content", "").strip()
        return text or None
    except Exception as e:
        print(f"[DeepThink] {cfg['model']} unavailable: {e}")
        return None
