from piper_dashboard import api
from piper_memory.seeds import load_seeds
from piper_research.service import apply_deletions, pending_deletions
from test_synthesis import insight, two_topics  # noqa: F401  (fixture)


def _count(mem, sql, *a):
    return mem.db.execute(sql, a).fetchone()[0]


def test_deleting_a_topic_removes_everything_that_belongs_to_it(two_topics):  # noqa: F811
    mem, llm, r, birds, quantum, ids = two_topics
    llm.synth = insight
    r.synthesize(mem.get_topic(birds))                         # an insight + derived questions on birds
    mem.set_remark("position", "Mostly the stars.", topic_id=birds)
    mem.flag("contradiction", "x", findings=ids[birds][:2], topics=[birds])
    child = mem.topic("Robin compasses", parent=birds)
    before = mem.topic_impact(birds)
    assert before["findings"] == 3 and before["insights"] == 1 and before["subtopics"] == 1
    counts = mem.delete_topic(birds)
    assert counts["findings"] == 3 and counts["flags"] == 1
    assert mem.get_topic(birds) is None
    for table in ("findings", "episodes", "insights", "remarks"):
        assert _count(mem, f"SELECT COUNT(*) FROM {table} WHERE topic_id = ?", birds) == 0
    assert _count(mem, "SELECT COUNT(*) FROM finding_sources WHERE finding_id IN (%s)" % ",".join(map(str, ids[birds]))) == 0
    assert _count(mem, "SELECT COUNT(*) FROM insight_events") == 0 and not mem.notable()
    assert mem.get_topic(child)["parent_id"] is None                    # sub-topic kept, now top-level
    assert len(mem.findings(topic_id=quantum)) == 2                      # the other topic untouched
    assert mem.get_concept(mem.concept("radical pair"))                  # still used by quantum's findings
    assert all(fid in mem.findings_index.ids for fid in ids[quantum])
    assert not any(fid in mem.findings_index.ids for fid in ids[birds])  # indexes rebuilt
    assert mem.similar("radical pairs under blue light", k=5)           # similarity still works
    assert [e for e in mem.events() if e["kind"] == "topic_deleted"][0]["name"] == "How migrating birds navigate"


def test_concepts_and_pages_nothing_uses_any_more_go_too(two_topics):  # noqa: F811
    mem, llm, r, birds, quantum, ids = two_topics
    page = mem.source("wikipedia", "Robin", url="https://en.wikipedia.org/w/index.php?title=Robin&oldid=1")
    fid = mem.add_finding("Robins migrate at night.", birds, sources=[page])
    mem.relate(mem.concept("robin"), "example_of", mem.concept("night migrant"), finding_id=fid)
    counts = mem.delete_topic(birds)
    assert counts["pages"] >= 1 and counts["concepts"] >= 2
    assert not mem.db.execute("SELECT 1 FROM concepts WHERE name = 'robin'").fetchone()
    assert not mem.db.execute("SELECT 1 FROM sources WHERE title = 'Robin'").fetchone()


def test_an_insight_in_another_topic_keeps_its_trail_but_loses_the_deleted_basis(two_topics):  # noqa: F811
    mem, llm, r, birds, quantum, ids = two_topics
    iid = mem.add_insight("Shared light chemistry.", quantum, [ids[birds][0], ids[quantum][0]], trace={"selection": {"findings": []}})
    assert mem.topic_impact(birds)["other_insights_touched"] == 1
    mem.delete_topic(birds)
    assert mem.get_insight(iid)["basis"] == [ids[quantum][0]]


def test_dashboard_requests_and_the_service_carries_them_out(two_topics, tmp_path):  # noqa: F811
    mem, llm, r, birds, quantum, ids = two_topics
    db = mem.path
    assert api.topic_impact(mem, birds)["findings"] == 3
    assert api.request_delete(db, birds)
    assert [t["delete_pending"] for t in api.topics(mem) if t["id"] == birds] == [True]
    assert api.request_delete(db, birds, cancel=True)                    # changed my mind
    assert pending_deletions(mem) == []
    api.request_delete(db, birds)
    assert [d["topic_id"] for d in pending_deletions(mem)] == [birds]
    assert apply_deletions(r, mem) == 1
    assert mem.get_topic(birds) is None and pending_deletions(mem) == []
    assert apply_deletions(r, mem) == 0
    assert not api.request_delete(db, birds)                             # gone: nothing to request


def test_reloading_seeds_does_not_bring_a_deleted_topic_back(two_topics, tmp_path):  # noqa: F811
    mem, llm, r, birds, quantum, ids = two_topics
    mem.delete_topic(birds)
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text("topics:\n  - name: How migrating birds navigate\n  - name: Tides\n", encoding="utf-8")
    out = load_seeds(mem, seeds)
    assert out["skipped_deleted"] == ["How migrating birds navigate"] and out["topics"] == 1
    assert {t["name"] for t in mem.topics()} == {"Quantum biology", "Tides"}
