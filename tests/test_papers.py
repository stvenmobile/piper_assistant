"""Research papers: OpenAlex and arXiv records -> pages, and how the loop reads (and backs off) them."""
import time

from piper_research.papers import ArXiv, OpenAlex, abstract_from_index
from piper_research.wiki import passages
from test_research import CFG, PAGE, finding, world  # noqa: F401  (fixture)

ABSTRACT = ("Night-migratory songbirds orient using the Earth's magnetic field. Cryptochrome 4a in the retina "
            "is a candidate magnetoreceptor: its levels rise during the migratory season, and its radical pairs "
            "are magnetically sensitive in vitro. Here we compare expression in migratory and resident birds.")


def inverted(text: str) -> dict:
    inv = {}
    for i, w in enumerate(text.split()):
        inv.setdefault(w, []).append(i)
    return inv


WORK = {"id": "https://openalex.org/W123", "doi": "https://doi.org/10.1/abc", "display_name": "Cry4a in the robin retina",
        "publication_year": 2021, "type": "article", "cited_by_count": 40,
        "abstract_inverted_index": inverted(ABSTRACT),
        "primary_location": {"source": {"display_name": "Journal of Bird Science"}},
        "authorships": [{"author": {"display_name": n}} for n in ("A. One", "B. Two", "C. Three", "D. Four", "E. Five")]}


def test_openalex_abstract_is_put_back_in_word_order():
    assert abstract_from_index(inverted("the cat saw the dog")) == "the cat saw the dog"
    assert abstract_from_index(None) == ""


def test_openalex_record_becomes_a_page_with_abstract_and_byline():
    p = OpenAlex()._page_from(WORK)
    assert p["kind"] == "openalex" and p["url"] == "https://doi.org/10.1/abc" and p["revid"] == "2021"
    assert p["title"] == "Cry4a in the robin retina (2021)"
    ps = passages(p, 1200)
    assert [x["section"] for x in ps] == ["Abstract", "About"]
    assert "radical pairs" in ps[0]["text"] and "et al." in ps[1]["text"] and "Cited by 40" in ps[1]["text"]
    assert OpenAlex()._page_from({**WORK, "abstract_inverted_index": inverted("Too short.")}) is None


ATOM = f"""<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2308.08708v3</id><published>2023-08-17T00:00:00Z</published>
<title>Consciousness in Artificial
  Intelligence</title><summary>{ABSTRACT}</summary>
<author><name>Patrick Butlin</name></author><author><name>Robert Long</name></author></entry></feed>"""


def test_arxiv_entry_becomes_a_page_marked_as_a_preprint():
    import xml.etree.ElementTree as ET
    ax = ArXiv()
    e = ET.fromstring(ATOM).find("a:entry", ax.NS)
    p = ax._page_from(e)
    assert p["title"] == "Consciousness in Artificial Intelligence (2023)" and p["revid"] == "v3"
    assert p["url"].endswith("/abs/2308.08708v3") and "not peer-reviewed" in p["text"]


class FakePapers:
    def __init__(self, kind, page, fail=False):
        self.kind, self.p, self.fail, self.searched = kind, page, fail, []
        self.paused_until, self.pause_reason = 0.0, ""
    def available(self): return time.time() >= self.paused_until
    def search(self, q, limit=1):
        self.searched.append(q)
        if self.fail:
            self.paused_until, self.pause_reason = time.time() + 3600, "too many requests"
            raise RuntimeError("HTTP 429")
        return ["W1"]
    def page(self, pid): return self.p


def test_a_paper_finding_is_credited_to_the_paper(world):  # noqa: F811
    mem, llm, r, tid = world
    paper = {"title": "Cry4a in the robin retina (2021)", "kind": "openalex", "revid": "2021",
             "url": "https://doi.org/10.1/abc", "text": "== Abstract ==\nCry4a levels in the robin retina rise during migration."}
    r.papers = [FakePapers("openalex", paper)]
    r.cfg = {**CFG, "openalex_papers": 1}
    llm.extract = [finding("Cry4a rises during migration.", "Cry4a levels in the robin retina rise during migration", passage=1)]
    out = r.cycle()
    assert out["new"] == 1 and "Cry4a in the robin retina (2021)" in out["pages"]
    f = [x for x in mem.findings(tid) if x["kind"] != "question"][0]
    src = mem.provenance(f["id"])["sources"][0]
    assert (src["kind"], src["url"]) == ("openalex", "https://doi.org/10.1/abc")


def test_a_limited_source_is_logged_once_and_then_left_alone(world):  # noqa: F811
    mem, llm, r, tid = world
    src = FakePapers("arxiv", None, fail=True)
    r.papers = [src]
    r.cfg = {**CFG, "arxiv_papers": 1}
    assert r.read_papers(["bird compass"]) == []
    assert r.read_papers(["bird compass"]) == []                       # paused: not asked again
    assert len(src.searched) == 1
    kinds = [e["kind"] for e in mem.events()]
    assert kinds.count("source_failed") == 1 and kinds.count("source_paused") == 1
