"""
How fast is each candidate model at a normal Piper conversation turn? Numbers for choosing one
model for both conversation and research (so the Ollama server never has to swap models).

    python3 tools/ollama_bench.py                       # chat model, research model, qwen3:14b
    python3 tools/ollama_bench.py --models qwen3:30b-a3b qwen3:14b --runs 5
    python3 tools/ollama_bench.py --think               # also one "think carefully" answer per model
    python3 tools/ollama_bench.py --no-hold             # don't pause research on piper-mind

Each turn is what the supervisor sends: Piper's system prompt (system_dna.md + the voice rules),
Steve's profile, three earlier exchanges and a new question, thinking off, the same num_ctx for
every model (Ollama reloads a model whenever num_ctx changes, so a shared model needs one value).

For every model it unloads whatever is loaded, measures the cold load, then runs each question
--runs times warm. Reported per model:
    load        seconds to load the model (cold, once)
    first word  seconds until the first piece of the reply arrives (what streaming speech would wait for)
    reply       seconds for the whole reply
    out tok     reply length in tokens;  gen tok/s  generation speed
    prompt tok  prompt size;  prompt tok/s  how fast the prompt is read
    on GPU      share of the loaded model in GPU memory (the rest runs on the CPU)
Results are printed and saved to ollama_bench_<date>_<time>.txt in the current folder.

While it runs, research is held on piper-mind (as when someone talks to Piper), so the two don't
compete for the GPU; it is released at the end, also after Ctrl-C.

Version 1.0.0 - created 2026-11-24, last modified 2026-11-24
History:
    1.0.0  2026-11-24  First version: cold load, first-word and full-reply times per model.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from piper_brain.config import CONFIG
except Exception as e:                       # still usable without the project's settings
    print(f"(config not loaded: {e} - using built-in defaults)")
    CONFIG = {"llm": {"base_url": "http://192.168.1.150:11434", "model": "llama3.2:latest"},
              "research": {"model": "qwen3:30b-a3b", "url": None, "mind_url": ""}}

VOICE_RULES = (          # as supervisor.load_system_prompt() adds them
    "\n\nVOICE RULES:\n"
    "1. Keep spoken responses concise, natural, and direct (1 to 2 sentences maximum).\n"
    "2. Strictly avoid markdown headers, asterisks, bullet points, numbered lists, and code blocks.\n"
    "3. Address the interlocutor directly without meta-announcements.")

HISTORY = [
    ("Good morning Piper.", "Good morning, Steve! What can I do for you today?"),
    ("What's the weather going to be like?", "It's 48 degrees and clear in Matthews right now, warming up to the low sixties this afternoon."),
    ("Nice. I'm working on your head again today.", "Exciting! I'm curious how the new face is coming along."),
]
QUESTIONS = [
    "Why do cats purr?",
    "Can you give me a tip for keeping a 3D printed part from warping?",
    "What's something interesting you learned in your research this week?",
]
THINK_QUESTION = "If I double the gear ratio on a pan motor, what happens to its speed and its holding torque, and why?"


def system_prompt() -> str:
    dna = ROOT / "system_dna.md"
    text = dna.read_text(encoding="utf-8") if dna.exists() else "You are Piper, a friendly home robot."
    text += VOICE_RULES
    profile = ROOT / "profiles" / "steve.md"
    if profile.exists():
        text += "\n\nABOUT THE PERSON YOU ARE TALKING TO:\n" + profile.read_text(encoding="utf-8")
    text += f"\n\nIt is {datetime.now():%A, %B %d, %Y, %I:%M %p}."
    return text


def call(url: str, body: dict | None = None, timeout: float = 300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str, body: dict | None = None, timeout: float = 300) -> dict:
    with call(url, body, timeout) as r:
        return json.loads(r.read() or b"{}")


def unload_all(base: str):
    for m in get_json(base + "/api/ps").get("models", []):
        get_json(base + "/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=60)
    for _ in range(30):                                     # wait until it is really gone
        if not get_json(base + "/api/ps").get("models"):
            return
        time.sleep(1)


def gpu_share(base: str, model: str) -> str:
    for m in get_json(base + "/api/ps").get("models", []):
        if m["name"] == model or m.get("model") == model:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            return f"{100 * vram / size:.0f}%" if size else "?"
    return "?"


def chat(base: str, model: str, messages: list, num_ctx: int, think: bool, use_think_flag: bool = True) -> dict:
    """One streamed /api/chat. Returns timings (s) and Ollama's own counters."""
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": "30m",
            "options": {"num_ctx": num_ctx, "temperature": 0.4}}
    if use_think_flag:
        body["think"] = think
    t0 = time.monotonic()
    first, text, final = None, "", {}
    try:
        with call(base + "/api/chat", body) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = (chunk.get("message") or {}).get("content", "")
                if piece and first is None:
                    first = time.monotonic() - t0
                text += piece
                if chunk.get("done"):
                    final = chunk
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        if use_think_flag and "think" in detail.lower() and not think:
            return chat(base, model, messages, num_ctx, think, use_think_flag=False)   # model without a thinking mode
        raise RuntimeError(f"{e.code}: {detail[:200]}")
    ns = 1e9
    return {
        "wall": time.monotonic() - t0,
        "first": first if first is not None else float("nan"),
        "load": final.get("load_duration", 0) / ns,
        "prompt_tok": final.get("prompt_eval_count", 0),
        "prompt_s": final.get("prompt_eval_duration", 0) / ns,
        "out_tok": final.get("eval_count", 0),
        "gen_s": final.get("eval_duration", 0) / ns,
        "text": text.strip(),
    }


def messages_for(system: str, question: str) -> list:
    msgs = [{"role": "system", "content": system}]
    for q, a in HISTORY:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": question}]


def mean(xs):
    xs = [x for x in xs if x == x]                          # drop NaN
    return statistics.mean(xs) if xs else float("nan")


def bench(base: str, model: str, runs: int, num_ctx: int, think: bool, log) -> dict:
    system = system_prompt()
    log(f"\n=== {model} ===")
    log("  unloading everything ...")
    unload_all(base)
    cold = chat(base, model, messages_for(system, QUESTIONS[0]), num_ctx, think=False)
    log(f"  cold: load {cold['load']:.1f} s, whole first reply {cold['wall']:.1f} s")
    warm = []
    for i in range(runs):
        for q in QUESTIONS:
            r = chat(base, model, messages_for(system, q), num_ctx, think=False)
            warm.append(r)
            log(f"  run {i + 1}: first word {r['first']:.2f} s, reply {r['wall']:.2f} s, "
                f"{r['out_tok']} tok  | {r['text'][:90]!r}")
            if r["load"] > 1.0:
                log(f"    !! the model was reloaded ({r['load']:.1f} s) - something else is using Ollama?")
    res = {
        "model": model, "load": cold["load"], "gpu": gpu_share(base, model),
        "first": mean([r["first"] for r in warm]), "reply": mean([r["wall"] for r in warm]),
        "reply_max": max(r["wall"] for r in warm),
        "out_tok": mean([r["out_tok"] for r in warm]),
        "gen_tps": mean([r["out_tok"] / r["gen_s"] for r in warm if r["gen_s"]]),
        "prompt_tok": mean([r["prompt_tok"] for r in warm]),
        "prompt_tps": mean([r["prompt_tok"] / r["prompt_s"] for r in warm if r["prompt_s"] > 0.001]),
        "think": None,
    }
    if think:
        try:
            r = chat(base, model, messages_for(system, THINK_QUESTION), num_ctx, think=True)
            res["think"] = r["wall"]
            log(f"  think carefully: {r['wall']:.1f} s | {r['text'][:120]!r}")
        except RuntimeError as e:
            log(f"  think carefully: skipped ({str(e)[:80]})")
    return res


def table(results: list, num_ctx: int) -> str:
    head = (f"{'model':<22}{'load s':>8}{'on GPU':>8}{'first word':>12}{'reply':>8}{'worst':>8}"
            f"{'out tok':>9}{'gen tok/s':>11}{'prompt tok':>12}{'prompt tok/s':>14}{'think s':>9}")
    lines = [f"num_ctx {num_ctx}, thinking off (times in seconds, averages of the warm runs)", head, "-" * len(head)]
    for r in results:
        think = f"{r['think']:.1f}" if r["think"] is not None else "-"
        lines.append(f"{r['model']:<22}{r['load']:>8.1f}{r['gpu']:>8}{r['first']:>12.2f}{r['reply']:>8.2f}"
                     f"{r['reply_max']:>8.2f}{r['out_tok']:>9.0f}{r['gen_tps']:>11.1f}{r['prompt_tok']:>12.0f}"
                     f"{r['prompt_tps']:>14.0f}{think:>9}")
    return "\n".join(lines)


def main():
    llm, research = CONFIG["llm"], CONFIG["research"]
    default_models = list(dict.fromkeys([llm["model"], research["model"], "qwen3:14b"]))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default=(research.get("url") or llm["base_url"]).rstrip("/"), help="Ollama server")
    p.add_argument("--models", nargs="+", default=default_models)
    p.add_argument("--runs", type=int, default=3, help="times each question is asked warm")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--think", action="store_true", help="also time one 'think carefully' answer per model")
    p.add_argument("--no-hold", action="store_true", help="don't pause research on piper-mind")
    a = p.parse_args()

    out = Path.cwd() / f"ollama_bench_{datetime.now():%Y%m%d_%H%M}.txt"
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"Ollama {a.url}  models: {', '.join(a.models)}  runs: {a.runs} x {len(QUESTIONS)} questions")
    try:
        ver = get_json(a.url + "/api/version", timeout=5).get("version")
        log(f"Ollama version {ver}")
    except Exception as e:
        sys.exit(f"Can't reach Ollama at {a.url}: {e}")

    hold = None
    if not a.no_hold and research.get("mind_url"):
        from piper_brain.research_hold import ResearchHold
        hold = ResearchHold()
        if hold.start():
            log("research held on piper-mind; waiting for it to let go of the model ...")
            for _ in range(40):
                s = hold.state() or {}
                if s.get("honoured") or not s.get("researching"):
                    break
                time.sleep(0.5)
        else:
            log("(piper-mind not reachable - research is not held; stop it by hand if it's running)")
            hold = None

    results = []
    try:
        installed = {m["name"] for m in get_json(a.url + "/api/tags").get("models", [])}
        for model in a.models:
            if model not in installed and f"{model}:latest" not in installed:
                log(f"\n=== {model} === not installed on the server (ollama pull {model}) - skipped")
                continue
            try:
                results.append(bench(a.url, model, a.runs, a.num_ctx, a.think, log))
            except Exception as e:
                log(f"  failed: {e}")
    except KeyboardInterrupt:
        log("\n(stopped)")
    finally:
        if hold:
            hold.release()
            log("research released")
    if results:
        log("\n" + table(results, a.num_ctx))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
