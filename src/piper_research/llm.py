"""
The research model: Ollama's /api/chat with structured output - every call passes a JSON schema
(Ollama's `format`), so replies are parsed, never scraped from prose.
"""
import json
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class OllamaChat:
    def __init__(self, base_url: str, model: str, temperature: float = 0.3, num_ctx: int = 8192,
                 think: bool = False, keep_alive: str = "30m", timeout: float = 900.0):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.calls = 0
        self.tokens = 0
        self.seconds = 0.0

    def json(self, system: str, user: str, schema: dict, retries: int = 1) -> dict:
        """The model's reply to `user` (with instructions `system`), as a dict matching `schema`."""
        body = {
            "model": self.model, "stream": False, "think": self.think, "keep_alive": self.keep_alive,
            "format": schema,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        last = None
        for _ in range(retries + 1):
            t = time.monotonic()
            req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
            except (urllib.error.URLError, OSError, ValueError) as e:
                raise LLMError(f"{self.model} via {self.url}: {e}") from e
            self.calls += 1
            self.tokens += data.get("eval_count", 0)
            self.seconds += time.monotonic() - t
            try:
                return json.loads(data["message"]["content"])
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                last = e                            # truncated / malformed: ask again
        raise LLMError(f"{self.model} gave no valid JSON: {last}")

    def usage(self) -> dict:
        return {"llm_calls": self.calls, "llm_tokens": self.tokens, "llm_seconds": round(self.seconds, 1)}
