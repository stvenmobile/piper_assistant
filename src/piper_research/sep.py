"""
The Stanford Encyclopedia of Philosophy - the research loop's second source: peer-reviewed,
long-form entries, strong on exactly the questions Wikipedia covers thinly (consciousness, art,
death, mind). Same interface as wiki.Wikipedia: search() -> entry ids, page() -> {"title", "revid",
"url", "text", "kind"}, with the text in "== Section ==" form so wiki.passages() can cut it.

Polite by design: SEP's robots.txt asks for a 5 s crawl delay, so requests are spaced at least that
far apart, and entries are cached on disk for 30 days (they change rarely - each carries its last
"substantive revision" date, used as the revision for provenance).
"""
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from piper_research.wiki import USER_AGENT, WikiError

BASE = "https://plato.stanford.edu"
CRAWL_DELAY = 5.0
CACHE_DAYS = 30
SKIP_SECTIONS = {"bibliography", "academic tools", "other internet resources", "related entries",
                 "acknowledgments", "acknowledgements"}


class SEPError(WikiError):
    pass


class _Text(HTMLParser):
    """The entry's preamble + main text as plain text, headings as == ... == lines."""
    BLOCKS = {"p", "li", "blockquote", "dd", "dt", "tr", "div", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.depth, self.heading, self.skip = [], 0, None, 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "sup"):           # footnote markers too
            self.skip += 1
        elif tag in ("h2", "h3", "h4"):
            self.heading = {"h2": "==", "h3": "===", "h4": "===="}[tag]
            self.out.append("\n" + self.heading + " ")
        elif tag in self.BLOCKS:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "sup"):
            self.skip = max(0, self.skip - 1)
        elif tag in ("h2", "h3", "h4") and self.heading:
            self.out.append(" " + self.heading + "\n")
            self.heading = None
        elif tag in self.BLOCKS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data if self.heading else data.replace("\n", " "))

    def text(self) -> str:
        t = "".join(self.out)
        t = re.sub(r"[ \t]+", " ", t)
        lines = [l.strip() for l in t.split("\n")]
        return "\n".join(l for l in lines if l)


def entry_text(page_html: str) -> str:
    """Preamble + main text of an SEP entry page."""
    parts = []
    for div in ("preamble", "main-text"):
        m = re.search(r'<div id="%s">(.*?)(?=<div id="(?:main-text|bibliography|academic-tools|'
                      r'other-internet-resources|related-entries|acknowledgments)")' % div, page_html, re.S)
        if m:
            p = _Text()
            p.feed(m.group(1))
            parts.append(p.text())
    return "\n".join(parts)


class SEP:
    kind = "sep"

    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.requests = 0
        self._last = 0.0

    def _get(self, url: str) -> str:
        wait = self._last + CRAWL_DELAY - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.requests += 1
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            raise SEPError(f"SEP request failed: {e}") from e
        finally:
            self._last = time.monotonic()

    def search(self, query: str, limit: int = 3) -> list[str]:
        """Entry ids (e.g. 'consciousness-animal') for a search, best first."""
        page = self._get(f"{BASE}/search/searcher.py?" + urllib.parse.urlencode({"query": query}))
        out = []
        for slug in re.findall(r"entry=/entries/([a-z0-9-]+)/", page):
            if slug not in out:
                out.append(slug)
            if len(out) >= limit:
                break
        return out

    def page(self, entry: str) -> dict | None:
        """{"title", "revid", "url", "text", "kind"} for an entry id, or None."""
        cached = self._cache_path(entry)
        if cached and cached.exists() and time.time() - cached.stat().st_mtime < CACHE_DAYS * 86400:
            return json.loads(cached.read_text(encoding="utf-8"))
        url = f"{BASE}/entries/{entry}/"
        try:
            raw = self._get(url)
        except SEPError:
            return None
        title = re.search(r"<h1>(.*?)</h1>", raw, re.S)
        pub = re.search(r'<div id="pubinfo">\s*<em>(.*?)</em>', raw, re.S)
        text = entry_text(raw)
        if not title or not text:
            return None
        revised = html.unescape(pub.group(1)) if pub else ""
        m = re.search(r"substantive revision ([^;<]+)", revised) or re.search(r"First published ([^;<]+)", revised)
        page = {"title": html.unescape(re.sub("<[^>]+>", "", title.group(1))).strip(), "kind": self.kind,
                "revid": m.group(1).strip() if m else "", "url": url, "text": text}
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(page), encoding="utf-8")
        return page

    def _cache_path(self, entry: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(entry.encode()).hexdigest()[:16] + ".json")
