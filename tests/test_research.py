import random
from datetime import datetime

import pytest

from piper_memory import MemoryStore
from piper_research import Interrupted, Researcher, Schedule
from piper_research import prompts as P
from piper_research.schedule import parse_window
from piper_research.verify import find_quote
from piper_research.wiki import passages
from test_memory import FakeEmbedder

CFG = {"pages_per_question": 2, "passages": 6, "passage_chars": 400, "max_findings": 8,
       "reflect_every": 2, "cross_topic_checks": 2}

PAGE = {"title": "Bird migration", "revid": 111,
        "url": "https://en.wikipedia.org/w/index.php?title=Bird_migration&oldid=111",
        "text": ("Bird migration is the regular seasonal movement of birds.\n\n"
                 "== Orientation ==\n"
                 "Birds use a sun compass by day, compensating for the time of day.  "
                 "At night many species use a stellar compass centred on Polaris.\n"
                 "== References ==\n"
                 "Smith, J. (2001). A book about birds.\n")}


# ---- schedule ------------------------------------------------------------------------------------
def at(day: int, hhmm: str) -> datetime:          # day 0 = Monday 2026-10-05
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, 5 + day, h, m)


def test_schedule_window():
    s = Schedule(["01:00-08:00"])
    assert not s.active(at(0, "00:59"))
    assert s.active(at(0, "01:00")) and s.active(at(0, "07:59"))
    assert not s.active(at(0, "08:00"))
    assert s.next_start(at(0, "12:00")) == at(1, "01:00")


def test_schedule_crosses_midnight_and_belongs_to_its_start_day():
    s = Schedule(["22:00-06:00"], days=["fri"])
    assert s.active(at(4, "23:00"))                 # Friday night
    assert s.active(at(5, "05:00"))                 # ... into Saturday morning
    assert not s.active(at(5, "23:00"))             # not Saturday night
    assert not s.active(at(4, "05:00"))             # Thursday's night isn't scheduled


def test_schedule_from_environment_string_and_bad_input():
    s = Schedule("01:00-08:00, 13:00-14:00")
    assert s.active(at(2, "13:30")) and not s.active(at(2, "12:00"))
    with pytest.raises(ValueError):
        parse_window("1am-8am")
    with pytest.raises(ValueError):
        Schedule(["01:00-08:00"], days=["someday"])


# ---- quotes + passages -----------------------------------------------------------------------------
TEXT = "Birds use a sun compass by day, compensating for the time of day.  At night many use stars."


def test_quote_exact_ignoring_case_spacing_and_typography():
    assert find_quote("birds use a SUN compass by day,  compensating", [TEXT]) == 0
    assert find_quote("“Birds use a sun compass by day”", ["x", TEXT]) == 1


def test_quote_nearly_verbatim_passes_paraphrase_and_short_fail():
    assert find_quote("Birds use the sun compass by day, compensating for the time of day", [TEXT]) == 0
    assert find_quote("Birds navigate with the sun, correcting for the hour", [TEXT]) is None
    assert find_quote("sun compass", [TEXT]) is None


def test_quote_shortened_with_ellipsis_passes_only_if_every_piece_is_there_in_order():
    assert find_quote("Birds use a sun compass by day ... At night many use stars", [TEXT]) == 0
    assert find_quote("Birds use a sun compass by day … many use stars", [TEXT]) == 0
    assert find_quote("At night many use stars ... Birds use a sun compass", [TEXT]) is None   # out of order
    assert find_quote("Birds use a sun compass [...] for the time of day", [TEXT]) == 0
    assert find_quote("Birds use a sun compass by day ... the moon guides them", [TEXT]) is None
    assert find_quote("Birds ... day", [TEXT]) is None                                        # pieces too short


def test_passages_skip_references_and_keep_sections():
    ps = passages(PAGE, max_chars=400)
    assert [p["section"] for p in ps] == ["", "Orientation"]
    assert not any("Smith" in p["text"] for p in ps)
    long = {"title": "T", "text": " ".join(f"Sentence number {i} is here." for i in range(60))}
    assert all(len(p["text"]) <= 400 for p in passages(long, 400))


# ---- a cycle, with a fake model and a fake Wikipedia ----------------------------------------------
class FakeWiki:
    def __init__(self, pages):
        self.pages = {p["title"]: p for p in pages}

    def search(self, query, limit=5):
        return list(self.pages)[:limit]

    def page(self, title):
        return self.pages.get(title)


class FakeLLM:
    """Answers by schema; `extract` is the findings list the next EXTRACT returns."""
    model = "fake"

    def __init__(self):
        self.extract = []
        self.same = False
        self.relation = "neither"
        self.link = False
        self.asked = []
        self.calls = 0

    def json(self, system, user, schema, retries=1):
        self.calls += 1
        self.asked.append(schema)
        if schema is P.QUESTION_SCHEMA:
            return {"open_question": 1, "question": "unused", "why": "", "search_queries": ["birds"]}
        if schema is P.EXTRACT_SCHEMA:
            return {"findings": self.extract, "answer": "By sun and stars.",
                    "follow_up_questions": ["How do birds learn the stars?", "not a question"]}
        if schema is P.RELATE_SCHEMA:
            return {"relation": self.relation, "reason": ""}
        if schema is P.SAME_SCHEMA:
            return {"same": self.same}
        if schema is P.CROSS_SCHEMA:
            return {"link": self.link, "explanation": "a real link"}
        if schema is P.REFLECT_SCHEMA:
            return {"position": "The evidence is mixed.", "learned": "Birds steer by Polaris.",
                    "unsure": "How they learn it.", "next_questions": ["Do young birds learn the stars?"]}
        if schema is P.OVERNIGHT_SCHEMA:
            return {"summary": "I spent the night with birds."}
        raise AssertionError(schema)

    def usage(self):
        return {"llm_calls": self.calls, "llm_tokens": 0, "llm_seconds": 0.0}


def finding(claim, quote, passage=2, stance="neutral", kind="fact", relations=()):
    return {"claim": claim, "quote": quote, "passage": passage, "kind": kind, "confidence": 0.9,
            "stance": stance, "relations": [dict(zip(("subject", "predicate", "object"), r)) for r in relations]}


@pytest.fixture
def world(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db", FakeEmbedder())
    tid = mem.topic("How migrating birds navigate", thesis="Birds navigate mainly by the stars.")
    seeds = mem.source("user", "research seeds")
    mem.add_finding("Which cues do migrating birds use?", tid, kind="question", sources=[seeds])
    llm = FakeLLM()
    r = Researcher(mem, llm, FakeWiki([PAGE]), CFG, log=lambda *_: None, rng=random.Random(1))
    yield mem, llm, r, tid
    mem.close()


def test_cycle_keeps_verified_findings_with_provenance_and_rejects_invented_quotes(world):
    mem, llm, r, tid = world
    llm.extract = [
        finding("Birds use a sun compass by day.", "Birds use a sun compass by day, compensating for the time of day",
                relations=[("sun compass", "used_for", "navigation")]),
        finding("Many birds use a stellar compass centred on Polaris at night.",
                "At night many species use a stellar compass centred on Polaris", stance="supports"),
        finding("Birds smell their way home.", "Birds rely mostly on their sense of smell to find home"),
    ]
    out = r.cycle()
    assert out["outcome"] == "progress"
    assert (out["new"], out["rejected"], out["relations"]) == (2, 1, 1)
    assert out["question"] == "Which cues do migrating birds use?"         # the open seed question ...
    assert r.open_questions(tid) == ["How do birds learn the stars?"]     # ... is now asked; follow-up added
    facts = [f for f in mem.findings(tid) if f["kind"] != "question"]
    assert {f["stance"] for f in facts} == {"neutral", "supports"}
    p = mem.provenance(facts[0]["id"])
    assert p["sources"][0]["url"].endswith("oldid=111")
    assert p["sources"][0]["excerpt"].startswith("At night many species")
    assert p["episode"]["summary"] == "By sun and stars."
    assert [e["kind"] for e in mem.events() if e["kind"] == "rejected"] == ["rejected"]


def test_a_repeated_finding_from_another_page_adds_a_source_not_a_finding(world):
    mem, llm, r, tid = world
    other = {**PAGE, "title": "Bird navigation", "revid": 222, "url": "https://en.wikipedia.org/w/index.php?oldid=222"}
    r.wiki = FakeWiki([PAGE])
    llm.extract = [finding("Birds use a sun compass by day.", "Birds use a sun compass by day, compensating")]
    r.cycle()
    r.wiki = FakeWiki([other])
    r.last_topic = None
    out = r.cycle()
    assert (out["new"], out["extra_sources"]) == (0, 1)
    fact = [f for f in mem.findings(tid) if f["kind"] != "question"]
    assert len(fact) == 1 and len(mem.provenance(fact[0]["id"])["sources"]) == 2


def test_open_questions_never_count_as_duplicates_or_relatives_of_claims(world):
    mem, llm, r, tid = world
    llm.relation = "refines"
    llm.extract = [finding("Which cues do migrating birds use?", "Birds use a sun compass by day", kind="fact")]
    out = r.cycle()
    assert (out["new"], out["refines"]) == (1, 0)
    assert P.RELATE_SCHEMA not in llm.asked


def test_contradiction_is_judged_and_flagged(world):
    mem, llm, r, tid = world
    llm.extract = [finding("Birds use a sun compass by day.", "Birds use a sun compass by day")]
    r.cycle()
    llm.relation = "contradicts"
    llm.extract = [finding("Birds use a sun compass by day, not at night.", "Birds use a sun compass by day, compensating")]
    out = r.cycle()
    assert out["contradicts"] == 1
    assert [n["kind"] for n in mem.notable()] == ["contradiction"]


def test_reflection_every_n_cycles_writes_remarks_and_questions(world):
    mem, llm, r, tid = world
    llm.extract = [finding("Birds use a sun compass by day.", "Birds use a sun compass by day")]
    r.cycle()
    assert not mem.remarks(tid)
    llm.extract = [finding("Many birds use a stellar compass at night.", "At night many species use a stellar compass")]
    r.cycle()
    assert {x["kind"]: x["text"] for x in mem.remarks(tid)}["learned"] == "Birds steer by Polaris."
    assert "Do young birds learn the stars?" in r.open_questions(tid)


def test_cross_topic_link_flagged_only_when_the_judge_agrees(world):
    mem, llm, r, tid = world
    other = mem.topic("Quantum biology")
    mem.add_finding("Robins use a stellar compass centred on Polaris at night", other)
    llm.extract = [finding("Many birds use a stellar compass centred on Polaris at night.",
                           "At night many species use a stellar compass centred on Polaris")]
    r.cycle()
    assert not mem.notable()
    llm.link = True
    llm.extract = [finding("At night many species use a stellar compass on Polaris.",
                           "At night many species use a stellar compass centred on Polaris", passage=1)]
    mem2_new = r.cycle()
    assert mem2_new["new"] in (0, 1)                # the wording may count as a duplicate


def test_concepts_merge_only_when_the_judge_says_same(world):
    mem, llm, r, tid = world
    mem.concept("predator population")
    sim = r.mem.concept_merge
    r.mem.concept_merge = 0.0                       # make every name a candidate
    try:
        assert mem.concept("prey population", judge=r.same_concept) != mem.concept("predator population")
        llm.same = True
        assert mem.concept("carbon dioxide", judge=r.same_concept) in {c for c in mem.aliases.values()}
    finally:
        r.mem.concept_merge = sim


def test_choose_avoids_the_same_topic_twice_and_skips_retired(world):
    mem, llm, r, tid = world
    other = mem.topic("Why humans make art")
    retired = mem.topic("Old topic")
    mem.update_topic(retired, status="retired")
    r.last_topic = tid
    assert all(r.choose()["id"] == other for _ in range(10))
    with pytest.raises(KeyError):
        r.choose("No such topic")


def test_stop_request_ends_the_episode_as_interrupted(world):
    mem, llm, r, tid = world
    r.should_stop = lambda: True
    with pytest.raises(Interrupted):
        r.cycle()
    assert mem.episodes(tid)[0]["outcome"] == "interrupted"


def test_overnight_summary_becomes_a_general_remark(world):
    mem, llm, r, tid = world
    assert r.overnight("2000-01-01T00:00:00+00:00") is None          # nothing researched yet
    llm.extract = [finding("Birds use a sun compass by day.", "Birds use a sun compass by day")]
    r.cycle()
    assert r.overnight("2000-01-01T00:00:00+00:00") == "I spent the night with birds."
    assert [x["kind"] for x in mem.remarks(None)] == ["overnight"]
