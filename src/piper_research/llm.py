"""
The research model: Ollama's /api/chat with structured output - every call passes a JSON schema
(Ollama's `format`), so replies are parsed, never scraped from prose.

With a `gate` (piper_research.hold.HoldGate - the research service sets one), every call first
waits while research is held for a conversation, and a call already running is dropped the moment
a hold arrives (closing the connection stops Ollama generating) and asked again afterwards - so
Piper gets the GPU within a second or two, not after a minutes-long research reply.
"""
import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class LLMError(RuntimeError):
    pass


class _Held(Exception):
    """A running call was dropped because research is now held."""


class OllamaChat:
    def __init__(self, base_url: str, model: str, temperature: float = 0.3, num_ctx: int = 8192,
                 think: bool = False, keep_alive: str = "30m", timeout: float = 300.0,
                 max_tokens: int = 3000):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.max_tokens = max_tokens          # thinking included: stops a runaway think early
        self.calls = 0
        self.tokens = 0
        self.seconds = 0.0
        self.last_thinking = ""               # the model's raw thinking on the last call (thinking mode)
        self.gate = None                      # a HoldGate: wait while research is held (service.py sets it)

    def _post(self, body: dict, timeout: float) -> dict:
        payload = json.dumps(body).encode()
        if self.gate is None:
            req = urllib.request.Request(self.url, payload, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        # abortable: the request runs in a thread; a hold closes its connection
        u = urllib.parse.urlparse(self.url)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        out = {}

        def run():
            try:
                conn.request("POST", u.path, payload, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise OSError(f"HTTP {resp.status}: {data[:200]!r}")
                out["data"] = json.loads(data)
            except Exception as e:
                out["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        while t.is_alive():
            t.join(0.5)
            if t.is_alive() and self.gate.held():
                try:
                    if conn.sock is not None:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
                t.join(2.0)
                raise _Held()
        conn.close()
        if "error" in out:
            raise out["error"]
        return out["data"]

    def json(self, system: str, user: str, schema: dict, retries: int = 1, think: bool | None = None,
             max_tokens: int | None = None, timeout: float | None = None) -> dict:
        """The model's reply to `user` (with instructions `system`), as a dict matching `schema`.
        think: qwen3's thinking mode for this call (None = the default set at construction);
        max_tokens / timeout: this call's own budget (synthesis thinks for longer)."""
        body = {
            "model": self.model, "stream": False, "think": self.think if think is None else think,
            "keep_alive": self.keep_alive,
            "format": schema,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens or self.max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        last = None
        attempts = 0
        while attempts <= retries:
            if self.gate is not None:
                self.gate.wait()                    # held for a conversation: wait it out first
            t = time.monotonic()
            try:
                data = self._post(body, timeout or self.timeout)
            except _Held:
                continue                            # dropped for a hold: ask again afterwards
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
                raise LLMError(f"{self.model} via {self.url}: {e}") from e
            attempts += 1
            self.calls += 1
            self.tokens += data.get("eval_count", 0)
            self.seconds += time.monotonic() - t
            try:
                self.last_thinking = (data.get("message") or {}).get("thinking") or ""
                return json.loads(data["message"]["content"])
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                last = e                            # truncated / malformed: ask again
        raise LLMError(f"{self.model} gave no valid JSON: {last}")

    def usage(self) -> dict:
        return {"llm_calls": self.calls, "llm_tokens": self.tokens, "llm_seconds": round(self.seconds, 1)}
