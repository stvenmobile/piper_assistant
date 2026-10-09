"""
Wikipedia - the research loop's source. Search, fetch a page's plain text (with its revision id,
so every finding can point at the exact version it came from), and cut it into passages.

Pages are cached on disk for a week (data/research/wiki/), so re-reading a page costs nothing
and Wikipedia sees few requests.
"""
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "PiperAssistant/0.2 (https://github.com/stvenmobile/piper_assistant; research robot)"
SKIP_SECTIONS = {"see also", "references", "external links", "further reading", "notes", "bibliography",
                 "sources", "citations", "footnotes", "works cited", "gallery"}
CACHE_DAYS = 7


class WikiError(RuntimeError):
    pass


def page_url(title: str, revid: int | None = None) -> str:
    """The permanent link to this revision of the page (or the page itself)."""
    t = urllib.parse.quote(title.replace(" ", "_"))
    return f"https://en.wikipedia.org/w/index.php?title={t}&oldid={revid}" if revid else f"https://en.wikipedia.org/wiki/{t}"


def passages(page: dict, max_chars: int = 1200) -> list[dict]:
    """A page's text in passages of up to ~max_chars, split at section and paragraph breaks,
    each tagged with its section. Reference / link sections are left out."""
    out, section, skip, buf = [], "", False, []
    kind = page.get("kind", "wikipedia")

    def flush():
        if buf:
            out.append({"title": page["title"], "section": section, "text": " ".join(buf),
                        "kind": kind, "key": f"{kind}:{page['title']}"})
            buf.clear()

    for line in page["text"].split("\n"):
        line = line.strip()
        m = re.fullmatch(r"(=+)\s*(.*?)\s*\1", line)
        if m:
            flush()
            section = m.group(2)
            if len(m.group(1)) == 2:                # a top-level section decides skipping
                skip = section.lower() in SKIP_SECTIONS
            continue
        if not line or skip:
            continue
        for para in _split_long(line, max_chars):
            if buf and sum(len(b) + 1 for b in buf) + len(para) > max_chars:
                flush()
            buf.append(para)
    flush()
    return out


def _split_long(text: str, max_chars: int) -> list[str]:
    """A paragraph longer than max_chars, cut at sentence ends."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(s) + 1 > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


class Wikipedia:
    kind = "wikipedia"

    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.requests = 0

    def _get(self, **params) -> dict:
        params = {"format": "json", "formatversion": "2", **params}
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.requests += 1
                return json.load(r)
        except Exception as e:
            raise WikiError(f"Wikipedia request failed: {e}") from e

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Page titles for a search, best first."""
        data = self._get(action="query", list="search", srsearch=query, srlimit=limit, srnamespace=0)
        return [s["title"] for s in data.get("query", {}).get("search", [])]

    def page(self, title: str) -> dict | None:
        """{"title", "revid", "url", "text"} (redirects followed), or None if there's no such page."""
        cached = self._cache_path(title)
        if cached and cached.exists() and time.time() - cached.stat().st_mtime < CACHE_DAYS * 86400:
            return json.loads(cached.read_text(encoding="utf-8"))
        data = self._get(action="query", prop="extracts|revisions", rvprop="ids", explaintext=1, redirects=1,
                         titles=title)
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("extract"):
            return None
        p = pages[0]
        revid = (p.get("revisions") or [{}])[0].get("revid")
        page = {"title": p["title"], "revid": revid, "url": page_url(p["title"], revid), "text": p["extract"],
                "kind": "wikipedia"}
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(page), encoding="utf-8")
        return page

    def _cache_path(self, title: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16] + ".json")
