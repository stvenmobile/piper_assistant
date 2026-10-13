"""
Research papers - the research loop's scholarly sources, beside Wikipedia and the SEP:

    OpenAlex   ~250 million works from every field, with abstracts (api.openalex.org). Metered:
               without a key the free allowance is small (about 100 searches a day), so set
               research.openalex_api_key for more. When the allowance runs out, OpenAlex says when
               it resets and Piper leaves it alone until then.
    arXiv      preprints in physics, biology, computer science (export.arxiv.org). Asks for a few
               seconds between requests; on "too many requests" Piper backs off for an hour.
               Preprints are NOT peer-reviewed, and their passages are labelled that way.

Same interface as wiki.Wikipedia and sep.SEP: search() -> ids, page() -> {"title", "revid", "url",
"text", "kind"} with the text in "== Section ==" form, so wiki.passages() can cut it. A paper is its
abstract (plus authors, year and venue), which is what both services give for free. Entries are
cached on disk for 30 days.
"""
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from piper_research.wiki import USER_AGENT, WikiError

CACHE_DAYS = 30


class PaperError(WikiError):
    pass


class _Source:
    kind = ""
    min_gap = 0.0                                   # seconds between requests

    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.requests = 0
        self._last = 0.0
        self.paused_until = 0.0                     # time.time() before which we don't ask
        self.pause_reason = ""

    def available(self) -> bool:
        return time.time() >= self.paused_until

    def pause(self, seconds: float, reason: str):
        self.paused_until = time.time() + max(60.0, seconds)
        self.pause_reason = reason

    def _open(self, url: str):
        wait = self._last + self.min_gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            r = urllib.request.urlopen(req, timeout=self.timeout)
            self.requests += 1
            return r
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self.on_limited(e.headers)
            raise PaperError(f"{self.kind} request failed: HTTP {e.code}") from e
        except Exception as e:
            raise PaperError(f"{self.kind} request failed: {e}") from e
        finally:
            self._last = time.monotonic()

    def on_limited(self, headers):
        self.pause(3600, "too many requests")

    # ---- the cache, keyed by paper id ----------------------------------------------------------------
    def _cache_path(self, pid: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(pid.encode()).hexdigest()[:16] + ".json")

    def _cached(self, pid: str) -> dict | None:
        p = self._cache_path(pid)
        if p and p.exists() and time.time() - p.stat().st_mtime < CACHE_DAYS * 86400:
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _store(self, pid: str, page: dict):
        p = self._cache_path(pid)
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(page), encoding="utf-8")
        self._recent[pid] = page

    _recent: dict

    def page(self, pid: str) -> dict | None:
        """The paper found by search() (searches return the whole record, so this costs no request)."""
        return self._recent.get(pid) or self._cached(pid)


def _text(abstract: str, byline: str) -> str:
    return f"== Abstract ==\n{abstract}\n== About ==\n{byline}"


def abstract_from_index(inv: dict | None) -> str:
    """OpenAlex keeps abstracts as {word: [positions]}; put the words back in order."""
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


class OpenAlex(_Source):
    kind = "openalex"
    API = "https://api.openalex.org/works"
    FIELDS = "id,doi,display_name,publication_year,type,cited_by_count,abstract_inverted_index,primary_location,authorships"
    # published papers and reviews in journals and conferences (preprints come from arXiv, labelled)
    FILTER = "has_abstract:true,type:article|review,primary_location.source.type:journal|conference"
    MIN_CITED = 5                                   # papers others have cited go first; new ones still count

    def __init__(self, cache_dir=None, timeout: float = 30.0, api_key: str = ""):
        super().__init__(cache_dir, timeout)
        self.api_key = api_key or ""
        self._recent = {}
        self.remaining = None                       # searches' credits left today, as OpenAlex reports

    def on_limited(self, headers):
        reset = headers.get("X-RateLimit-Reset") if headers else None
        self.pause(float(reset) if reset and str(reset).isdigit() else 3600, "daily allowance used up")

    def search(self, query: str, limit: int = 2) -> list[str]:
        """OpenAlex work ids for a search: published papers with an abstract, the most relevant of
        those cited at least MIN_CITED times first."""
        params = {"search": query, "filter": self.FILTER, "per_page": min(10, max(5, 3 * limit)), "select": self.FIELDS}
        if self.api_key:
            params["api_key"] = self.api_key
        with self._open(self.API + "?" + urllib.parse.urlencode(params)) as r:
            data = json.load(r)
            left = r.headers.get("X-RateLimit-Remaining")
        if left is not None and str(left).isdigit():
            self.remaining = int(left)
            if self.remaining <= 0:
                reset = r.headers.get("X-RateLimit-Reset")
                self.pause(float(reset) if reset and str(reset).isdigit() else 3600, "daily allowance used up")
        out = []
        works = data.get("results", [])                 # relevance order; the cited ones first
        works = ([w for w in works if (w.get("cited_by_count") or 0) >= self.MIN_CITED]
                 + [w for w in works if (w.get("cited_by_count") or 0) < self.MIN_CITED])
        for w in works:
            if len(out) >= limit:
                break
            page = self._page_from(w)
            if page:
                pid = w["id"].rsplit("/", 1)[-1]
                self._store(pid, page)
                out.append(pid)
        return out

    def _page_from(self, w: dict) -> dict | None:
        abstract = abstract_from_index(w.get("abstract_inverted_index"))
        title = " ".join((w.get("display_name") or "").split())
        if len(abstract) < 200 or not title:        # no real abstract: nothing to learn from
            return None
        names = [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])][:4]
        authors = ", ".join(n for n in names if n) + (" et al." if len(w.get("authorships") or []) > 4 else "")
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        year = w.get("publication_year") or ""
        byline = (f"{authors} ({year}). {title}." + (f" {venue}." if venue else "")
                  + f" Type: {w.get('type', 'work')}. Cited by {w.get('cited_by_count', 0)}.")
        return {"title": f"{title} ({year})" if year else title, "kind": self.kind,
                "revid": str(year), "url": w.get("doi") or w["id"], "text": _text(abstract, byline)}


class ArXiv(_Source):
    kind = "arxiv"
    API = "https://export.arxiv.org/api/query"
    min_gap = 3.0                                   # arXiv's API guidance: one request every 3 s
    NS = {"a": "http://www.w3.org/2005/Atom"}

    def __init__(self, cache_dir=None, timeout: float = 30.0):
        super().__init__(cache_dir, timeout)
        self._recent = {}

    def search(self, query: str, limit: int = 1) -> list[str]:
        """arXiv ids (with version) for a search: every word first, then any of them."""
        words = [w for w in re.findall(r"[A-Za-z0-9-]{3,}", query)][:6]
        if not words:
            return []
        out = self._search(" AND ".join(f"all:{w}" for w in words), limit)
        if not out and len(words) > 1:
            out = self._search(" OR ".join(f"all:{w}" for w in words), limit)
        return out

    def _search(self, q: str, limit: int) -> list[str]:
        url = self.API + "?" + urllib.parse.urlencode({"search_query": q, "max_results": max(1, min(limit, 10))})
        with self._open(url) as r:
            root = ET.fromstring(r.read())
        out = []
        for e in root.findall("a:entry", self.NS):
            page = self._page_from(e)
            if page:
                pid = page["url"].rsplit("/abs/", 1)[-1]
                self._store(pid, page)
                out.append(pid)
        return out

    def _page_from(self, e) -> dict | None:
        get = lambda tag: " ".join((e.findtext(f"a:{tag}", "", self.NS) or "").split())
        title, abstract, url = get("title"), get("summary"), get("id")
        if not title or len(abstract) < 200 or "/abs/" not in url:
            return None
        names = [" ".join((a.findtext("a:name", "", self.NS) or "").split()) for a in e.findall("a:author", self.NS)]
        authors = ", ".join(names[:4]) + (" et al." if len(names) > 4 else "")
        year = get("published")[:4]
        version = url.rsplit("v", 1)[-1] if re.search(r"v\d+$", url) else ""
        byline = f"{authors} ({year}). {title}. arXiv preprint - not peer-reviewed."
        return {"title": f"{title} ({year})" if year else title, "kind": self.kind,
                "revid": f"v{version}" if version else year, "url": url, "text": _text(abstract, byline)}
