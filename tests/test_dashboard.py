import json
import sqlite3

import pytest

from piper_dashboard import api
from piper_memory import MemoryStore
from test_memory import FakeEmbedder


@pytest.fixture
def mem(tmp_path):
    m = MemoryStore(tmp_path / "mem.db", FakeEmbedder())
    tid = m.topic("Ocean tides", thesis="The Moon's gravity is the main cause of the tides.")
    src = m.source("wikipedia", "Tide", url="https://en.wikipedia.org/w/index.php?title=Tide&oldid=5", revid=5)
    sep = m.source("sep", "Tides", url="https://plato.stanford.edu/entries/tides/", revid="Tue Feb 2, 2021")
    m.log("research_started", model="qwen3:30b-a3b", session_hours=8)
    ep = m.start_episode(tid, angle="What role does the Sun play?")
    a = m.add_finding("The Moon raises two tidal bulges.", tid, ep, confidence=0.9, stance="supports",
                      stance_reason="the mechanism the thesis names", judged_by="m", sources=[(src, "two bulges")])
    b = m.add_finding("Wind drives the tides.", tid, ep, confidence=0.3, stance="challenges", judged_by="m",
                      sources=[(sep, "wind drives")])
    m.add_finding("Tidal gauges sit in harbours.", tid, ep, relevance="off_topic", judged_by="m", sources=[src])
    m.add_finding("Why are there two tides a day?", tid, kind="question")
    m.relate("moon", "causes", "tide", finding_id=a)
    m.relate("wind", "causes", "tide", finding_id=b)
    m.end_episode(ep, "progress", "The Moon, mostly.", new=3, off_topic=1, supports=1, challenges=1)
    m.set_remark("position", 'The Moon dominates.\\n\\n"learned": "x"', topic_id=tid)   # a model slip, tidied for display
    m.set_remark("overnight", "I studied the tides.")
    m.flag("contradiction", "moon vs wind", findings=[a, b], topics=[tid])
    m.flag("cross_topic", "old pre-judge flag", topics=[tid, tid])
    m.log("research_ended", cycles=1, reason="Session ended")
    yield m
    m.close()


def test_status_and_sessions(mem):
    s = api.status(mem)
    assert s["current"]["question"] == "What role does the Sun play?"
    assert not s["researching"] and s["session"]["cycles"] == 1
    sess = api.sessions(mem)[0]
    assert (sess["hours"], sess["totals"]["new"], sess["summary"]) == (8, 3, "I studied the tides.")


def test_topics_balance_ignores_off_topic_and_remarks_are_tidied(mem):
    t = api.topics(mem)[0]
    assert t["findings"] == 2 and t["by_source"] == {"wikipedia": 2, "sep": 1}
    assert t["balance"]["supports"]["count"] == 1 and t["balance"]["neutral"]["count"] == 0
    assert t["remarks"]["position"]["text"] == "The Moon dominates."
    assert t["open_questions"] == ["Why are there two tides a day?"]


def test_findings_filters_and_provenance(mem):
    assert api.findings(mem)["total"] == 3
    r = api.findings(mem, source="sep")
    assert r["total"] == 1 and r["items"][0]["sources"][0]["revid"] == "Tue Feb 2, 2021"
    assert r["items"][0]["certainty"] == "speculative"
    assert api.findings(mem, relevance="off_topic")["total"] == 1
    assert api.findings(mem, q="bulges")["items"][0]["sources"][0]["excerpt"] == "two bulges"
    assert api.findings(mem, kind="questions")["total"] == 1


def test_notable_skips_pre_judge_links_and_can_be_marked_reviewed(mem, tmp_path):
    n = api.notable(mem)
    assert [x["kind"] for x in n] == ["contradiction"] and len(n[0]["findings"]) == 2
    assert api.mark_reviewed(tmp_path / "mem.db", n[0]["id"])
    assert api.notable(mem) == [] and len(api.notable(mem, show="all")) == 1


def test_graph_and_concept(mem):
    g = api.graph(mem)
    assert {n["name"] for n in g["nodes"]} == {"moon", "wind", "tide"} and len(g["edges"]) == 2
    tide = next(n for n in g["nodes"] if n["name"] == "tide")
    c = api.concept(mem, tide["id"])
    assert {(r["subject"], r["predicate"]) for r in c["relations"]} == {("moon", "causes"), ("wind", "causes")}


def test_events_feed(mem):
    ev = api.events(mem)
    assert ev[0]["kind"] == "research_started" and ev[-1]["kind"] == "research_ended"
    assert api.events(mem, since=ev[-1]["id"]) == []
