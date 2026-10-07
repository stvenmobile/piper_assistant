import os
import re
import sqlite3
import zlib

import numpy as np
import pytest

from piper_memory import EmbedError, MemoryStore, OllamaEmbedder
from piper_memory.embed import normalise

DIM = 64


class FakeEmbedder:
    """Deterministic stand-in for Ollama: hashed bag of words (shared words -> similar), with
    optional fixed vectors for exact control. `down = True` simulates Ollama being unreachable."""

    def __init__(self, fixed: dict | None = None):
        self.fixed = fixed or {}
        self.down = False
        self.calls = 0

    def embed(self, texts):
        if self.down:
            raise EmbedError("down")
        self.calls += 1
        out = []
        for t in texts:
            if t in self.fixed:
                out.append(self.fixed[t])
                continue
            v = np.zeros(DIM, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[zlib.crc32(w.encode()) % DIM] += 1.0
            out.append(v if v.any() else np.ones(DIM, dtype=np.float32))
        return normalise(out)


def axis(i, j=None, mix=0.0):
    """A unit vector along axis i (optionally tilted toward axis j): exact similarities for tests."""
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    if j is not None:
        v[j] = mix
    return normalise(v)[0]


@pytest.fixture
def mem(tmp_path):
    m = MemoryStore(tmp_path / "mem.db", FakeEmbedder())
    yield m
    m.close()


def test_topics_are_unique_case_insensitively_and_nest(mem):
    tides = mem.topic("Tides")
    assert mem.topic("tides") == tides
    moon = mem.topic("Lunar gravity", parent=tides, origin="spawned")
    assert mem.get_topic(moon)["parent_id"] == tides
    mem.update_topic(tides, status="active", novelty=0.8)
    assert mem.get_topic(tides)["status"] == "active"
    assert [t["name"] for t in mem.topics(status="active")] == ["Tides"]
    with pytest.raises(ValueError):
        mem.update_topic(tides, status="sleeping")


def test_episodes_record_the_research_diary(mem):
    t = mem.topic("Tides")
    e = mem.start_episode(t, angle="why two tides a day?", model="llama3.2")
    mem.end_episode(e, "progress", summary="two bulges", findings=3)
    ep = mem.episodes(t)[0]
    assert ep["outcome"] == "progress" and ep["stats"] == {"findings": 3} and ep["ended_at"]


def test_sources_are_deduplicated(mem):
    a = mem.source("wikipedia", "Tide", "https://en.wikipedia.org/wiki/Tide")
    assert mem.source("wikipedia", "Tide (again)", "https://en.wikipedia.org/wiki/Tide") == a
    assert mem.source("llm", "llama3.2 answer") == mem.source("llm", "llama3.2 answer")
    with pytest.raises(ValueError):
        mem.source("rumour", "x")


def test_a_finding_keeps_its_provenance(mem):
    t = mem.topic("Tides")
    e = mem.start_episode(t, "causes")
    src = mem.source("wikipedia", "Tide", "https://en.wikipedia.org/wiki/Tide")
    f = mem.add_finding("Tides are caused by the Moon's gravity.", t, e, confidence=0.9,
                        sources=[(src, "Tides are the rise and fall of sea levels caused by ...")], tags=["physics"])
    p = mem.provenance(f)
    assert p["topic"]["name"] == "Tides" and p["episode"]["angle"] == "causes"
    assert p["sources"][0]["url"].endswith("/Tide") and p["sources"][0]["excerpt"].startswith("Tides are")
    assert p["finding"]["tags"] == ["physics"] and p["finding"]["confidence"] == 0.9


def test_status_other_than_new_needs_what_it_relates_to(mem):
    t = mem.topic("Tides")
    first = mem.add_finding("Two high tides a day.", t)
    with pytest.raises(ValueError):
        mem.add_finding("Actually it varies.", t, status="refines")
    second = mem.add_finding("Some places get one tide a day.", t, status="refines", relates_to=first)
    assert mem.provenance(second)["relates_to"]["id"] == first


def test_full_text_search(mem):
    t = mem.topic("Tides")
    mem.add_finding("Spring tides happen at new and full moon.", t)
    mem.add_finding("Neap tides are weaker.", t)
    assert [f["text"] for f in mem.search("spring")] == ["Spring tides happen at new and full moon."]
    assert len(mem.search("tides")) == 2


def test_similar_and_duplicates(mem):
    t = mem.topic("Tides")
    a = mem.add_finding("the moon pulls the ocean", t)
    mem.add_finding("bread rises with yeast", mem.topic("Baking"))
    hits = mem.similar("the moon pulls the ocean water", k=1)
    assert hits[0]["id"] == a and hits[0]["score"] > 0.8
    assert mem.duplicate_of("the moon pulls the ocean")["id"] == a
    assert mem.duplicate_of("yeast makes gas") is None


def test_concept_key_normalises_case_articles_and_plurals():
    from piper_memory.store import concept_key
    assert concept_key("The Tides") == concept_key("tide") == "tide"
    assert concept_key("predator populations") == "predator population"
    assert concept_key("species") == "specy" or concept_key("species") == concept_key("specie")  # crude, but consistent
    assert concept_key("gas") == "gas" and concept_key("process") == "process"
    assert concept_key("predator population") != concept_key("prey population")


def test_concepts_merge_on_name_identity_without_any_embedding(tmp_path):
    mem = MemoryStore(tmp_path / "m.db", None)
    tide = mem.concept("tide")
    assert mem.concept("TIDE") == tide and mem.concept("The Tides") == tide
    assert "The Tides" in mem.get_concept(tide)["aliases"]


def test_similar_names_never_merge_without_a_judge(tmp_path):
    # the real failure: 'predator population' ~ 'prey population' embeds at 0.95
    emb = FakeEmbedder({"predator population": axis(0), "prey population": axis(0, 1, 0.33)})
    mem = MemoryStore(tmp_path / "m.db", emb)
    pred = mem.concept("predator population")
    assert mem.concept("prey population") != pred
    assert mem.similar_concepts("prey population")[0]["id"] == pred      # offered as a candidate only


def test_a_judge_decides_merges_on_similarity(tmp_path):
    emb = FakeEmbedder({"carbon dioxide": axis(0), "CO2": axis(0, 1, 0.3),
                        "predator population": axis(3), "prey population": axis(3, 4, 0.33)})
    mem = MemoryStore(tmp_path / "m.db", emb)
    same = {("CO2", "carbon dioxide")}
    judge = lambda new, old: (new, old) in same
    co2 = mem.concept("carbon dioxide")
    assert mem.concept("CO2", judge=judge) == co2 and "CO2" in mem.get_concept(co2)["aliases"]
    assert mem.concept("CO2") == co2                                     # now a known alias
    pred = mem.concept("predator population", judge=judge)
    assert mem.concept("prey population", judge=judge) != pred           # the judge said no


def test_merge_concepts_repoints_relations(tmp_path):
    mem = MemoryStore(tmp_path / "m.db", None)
    t = mem.topic("Air")
    f = mem.add_finding("Plants absorb CO2.", t)
    mem.relate("plants", "used_for", "carbon dioxide", finding_id=f)
    mem.relate("CO2", "causes", "warming", finding_id=f)
    co2, cd = mem.concept("CO2"), mem.concept("carbon dioxide")
    mem.merge_concepts(keep=cd, drop=co2)
    assert mem.get_concept(co2) is None and "CO2" in mem.get_concept(cd)["aliases"]
    assert {(r["subject"], r["predicate"], r["object"]) for r in mem.relations(cd)} == {
        ("plants", "used_for", "carbon dioxide"), ("carbon dioxide", "causes", "warming")}
    assert mem.concept("co2") == cd


def test_relations_build_a_graph_with_provenance(mem):
    t = mem.topic("Tides")
    f = mem.add_finding("The Moon's gravity causes tides.", t)
    r = mem.relate("Moon", "causes", "tides", finding_id=f, confidence=0.9)
    assert mem.relate("moon", "causes", "TIDES", finding_id=f) == r      # same edge, not a duplicate
    mem.relate("tides", "located_in", "ocean", finding_id=f)
    moon = mem.concept("Moon")
    assert mem.neighbours(moon, depth=1) == {mem.concept("tides"): 1}
    assert mem.neighbours(moon, depth=2)[mem.concept("ocean")] == 2
    rel = mem.relations(finding_id=f)
    assert {(x["subject"], x["predicate"], x["object"]) for x in rel} == {("Moon", "causes", "tides"),
                                                                         ("tides", "located_in", "ocean")}
    with pytest.raises(ValueError):
        mem.relate("Moon", "loves", "tides")


def test_cross_topic_candidates_skip_findings_that_already_share_a_concept(tmp_path):
    emb = FakeEmbedder({
        "predator and prey numbers cycle": axis(0),
        "inventory and orders oscillate": axis(0, 1, 0.5),        # close (0.89), other topic
        "stock levels swing with demand": axis(0, 2, 0.5),         # close, other topic, but linked below
        "wolves hunt deer": axis(0, 3, 0.4),                       # close, SAME topic
    })
    mem = MemoryStore(tmp_path / "m.db", emb)
    eco, biz = mem.topic("Ecology"), mem.topic("Supply chains")
    a = mem.add_finding("predator and prey numbers cycle", eco)
    b = mem.add_finding("inventory and orders oscillate", biz)
    c = mem.add_finding("stock levels swing with demand", biz)
    mem.add_finding("wolves hunt deer", eco)
    mem.relate("population", "related_to", "cycles", finding_id=a)
    mem.relate("cycles", "example_of", "oscillation", finding_id=c)          # c shares "cycles" with a
    ids = [h["id"] for h in mem.cross_topic_candidates(a)]
    assert ids == [b]


def test_series_remarks_notable_and_events(mem):
    t = mem.topic("Tides")
    f = mem.add_finding("Tidal range grows with bay length.", t, kind="quantity")
    sid = mem.add_series("tidal range vs bay length", [[10, 1.2], [50, 3.1]], finding_id=f,
                         x_label="bay length", y_label="range", units="km, m")
    assert mem.series(f)[0]["points"] == [[10, 1.2], [50, 3.1]] and mem.series()[0]["id"] == sid
    first = mem.set_remark("learned", "Tides come from the Moon.", topic_id=t)
    mem.set_remark("learned", "Tides come from the Moon and the Sun.", topic_id=t)
    assert [r["text"] for r in mem.remarks(t)] == ["Tides come from the Moon and the Sun."]
    assert len(mem.remarks(t, current_only=False)) == 2 and first
    n = mem.flag("contradiction", "Tide counts disagree", findings=[f])
    assert mem.notable()[0]["refs"] == {"findings": [f]}
    mem.mark_reviewed(n)
    assert mem.notable() == []
    e1 = mem.log("topic_chosen", topic="Tides")
    mem.log("query_sent", question="why?")
    assert [e["kind"] for e in mem.events(since_id=e1)] == ["query_sent"]
    assert mem.stats()["findings"] == 1


def test_memory_survives_a_restart(tmp_path):
    emb = FakeEmbedder()
    m1 = MemoryStore(tmp_path / "m.db", emb)
    t = m1.topic("Tides")
    f = m1.add_finding("the moon pulls the ocean", t)
    m1.concept("tide")
    m1.close()
    m2 = MemoryStore(tmp_path / "m.db", emb)
    assert m2.similar("the moon pulls the ocean", k=1)[0]["id"] == f
    assert m2.concept("Tide") == 1


def test_works_without_an_embedder_then_catches_up(tmp_path):
    emb = FakeEmbedder()
    emb.down = True
    mem = MemoryStore(tmp_path / "m.db", emb)
    t = mem.topic("Tides")
    f = mem.add_finding("the moon pulls the ocean", t)
    assert mem.similar("the moon pulls the ocean") == [] and mem.stats()["unembedded_findings"] == 1
    emb.down = False
    assert mem.backfill_embeddings() == 1
    assert mem.similar("the moon pulls the ocean", k=1)[0]["id"] == f


def test_readonly_reader_sees_writes_but_cannot_write(tmp_path):
    writer = MemoryStore(tmp_path / "m.db", FakeEmbedder())
    t = writer.topic("Tides")
    writer.add_finding("the moon pulls the ocean", t)
    reader = MemoryStore(tmp_path / "m.db", FakeEmbedder(), readonly=True)
    assert reader.stats()["findings"] == 1
    with pytest.raises((PermissionError, sqlite3.OperationalError)):
        reader.topic("Baking")


def test_schema_version_is_recorded(mem):
    assert mem.db.execute("PRAGMA user_version").fetchone()[0] == 1


@pytest.mark.skipif(not os.environ.get("PIPER_TEST_OLLAMA"), reason="set PIPER_TEST_OLLAMA=<url> to test the real embedder")
def test_real_ollama_embedder_separates_same_from_related(tmp_path):
    mem = MemoryStore(tmp_path / "m.db", OllamaEmbedder(os.environ["PIPER_TEST_OLLAMA"]))
    tide = mem.concept("tide")
    assert mem.concept("tides") == tide
    assert mem.concept("orbit") != tide
    pred = mem.concept("predator population")
    assert mem.concept("prey population") != pred                       # 0.95 similar, still apart
    t = mem.topic("Tides")
    f = mem.add_finding("Tides are caused mainly by the Moon's gravitational pull on Earth's oceans.", t)
    assert mem.duplicate_of("Ocean tides result from lunar gravity acting on seawater.")["id"] == f
