import pytest

from piper_research.mastery import Mastery, apply_requests, free_slots, open_tagged, pending_requests, readiness
from test_rhythm import CurLLM, cur  # noqa: F401  (fixture)
from test_synthesis import SCFG, insight, two_topics  # noqa: F401  (fixture)

MCFG = {**SCFG, "mastery_saturation": 0.8, "mastery_min_findings": 3, "max_current_topics": 3, "mastery_drive": 0.4}


def _ready(mem, r, tid):
    """Make a topic meet every item on the mastery checklist."""
    mem.update_topic(tid, saturation=0.9, status="active")
    r.llm.inner.synth = insight
    out = r.synthesize(mem.get_topic(tid))
    for q in open_tagged(mem, tid, "derived"):                        # its insight questions get researched
        ep = mem.start_episode(tid, angle=q["text"])
        mem.end_episode(ep, "stalled")
    mem.set_remark("position", "Mostly the stars.", topic_id=tid)
    return out


@pytest.fixture
def m(cur):  # noqa: F811
    mem, llm, r, birds, quantum, c = cur
    r.cfg = MCFG
    return mem, r, birds, quantum, Mastery(r)


def test_the_checklist_shows_what_stands_between_a_topic_and_mastery(m):
    mem, r, birds, quantum, mm = m
    rd = readiness(mem, mem.get_topic(birds), MCFG)
    assert not rd["ready"] and 0 < rd["score"] < 1
    unmet = {c["key"] for c in rd["criteria"] if not c["met"]}
    assert unmet == {"saturated", "synthesis", "position"}
    _ready(mem, r, birds)
    assert readiness(mem, mem.get_topic(birds), MCFG)["ready"]


def test_piper_requests_mastery_once_the_checklist_is_complete(m):
    mem, r, birds, quantum, mm = m
    assert mm.check() == []
    _ready(mem, r, birds)
    assert mm.check() == ["How migrating birds navigate"]
    assert mem.get_topic(birds)["mastery"] == "candidate"
    assert mm.check() == []                                          # asked once
    assert [e for e in mem.events() if e["kind"] == "mastery_requested"][0]["criteria"]["evidence"].startswith("3 findings")


def test_a_request_is_withdrawn_if_the_checklist_slips_before_you_answer(m):
    mem, r, birds, quantum, mm = m
    _ready(mem, r, birds)
    assert mm.check() == ["How migrating birds navigate"]
    mem.update_topic(birds, saturation=0.3)                          # new research finds new things again
    assert mm.check() == []
    assert mem.get_topic(birds)["mastery"] == ""
    w = [e for e in mem.events() if e["kind"] == "mastery_withdrawn"]
    assert len(w) == 1 and set(w[0]["unmet"]) == {"saturated"}
    mem.update_topic(birds, saturation=0.9)                          # ... and asks again once it's met
    assert mm.check() == ["How migrating birds navigate"]


def test_an_answered_request_is_not_withdrawn(m):
    mem, r, birds, quantum, mm = m
    _ready(mem, r, birds)
    mm.check()
    mem.log("mastery_confirm", topic_id=birds)                       # you confirmed on the dashboard
    mem.update_topic(birds, saturation=0.3)
    mm.check()
    assert mem.get_topic(birds)["mastery"] == "candidate"


def test_confirmed_mastery_writes_the_brief_and_frees_the_slot(m):
    mem, r, birds, quantum, mm = m
    _ready(mem, r, birds)
    slots = free_slots(mem, MCFG)
    rid = mm.brief(birds)
    t = mem.get_topic(birds)
    assert (t["status"], t["stage"], t["mastery"]) == ("explored", "explored", "") and t["mastered_at"]
    rep = mem.reports("mastery")[0]
    assert rep["id"] == rid and rep["data"]["topic_id"] == birds
    for section in ("Bottom line", "What is established", "Live conjectures and disputes", "Edge-case questions"):
        assert f"## {section}" in rep["text"]
    assert "F99999" not in rep["text"]                                # citations checked against the findings
    assert free_slots(mem, MCFG) == slots + 1
    assert all(t["id"] != birds for t, _ in r.topic_scores())        # no longer researched
    assert [e for e in mem.events() if e["kind"] == "topic_mastered"]


def test_not_yet_turns_your_guidance_into_priority_questions(m):
    mem, r, birds, quantum, mm = m
    _ready(mem, r, birds)
    mm.check()
    qs = mm.guidance(birds, "You haven't looked at olfactory navigation in seabirds.")
    assert mem.get_topic(birds)["mastery"] == ""
    gaps = open_tagged(mem, birds, "mastery_gap")
    assert [g["text"] for g in gaps] == qs
    assert r.ranked_open_questions(mem.get_topic(birds))[:2] == qs     # researched first
    rd = readiness(mem, mem.get_topic(birds), MCFG)
    assert not rd["ready"] and [c for c in rd["criteria"] if c["key"] == "guidance"][0]["detail"].startswith("2 guidance")
    assert mm.check() == []                                           # not asked again until they're explored
    assert mem.remarks(birds)[-1]["kind"] == "mastery_guidance"


def test_completion_drive_favours_topics_close_to_mastery(m):
    mem, r, birds, quantum, mm = m
    before = dict((t["id"], s) for t, s in r.topic_scores())[birds]
    _ready(mem, r, birds)
    mem.update_topic(birds, saturation=0.0)                           # same saturation term either way...
    after = dict((t["id"], s) for t, s in r.topic_scores())[birds]
    r.cfg = {**MCFG, "mastery_drive": 0.0}
    plain = dict((t["id"], s) for t, s in r.topic_scores())[birds]
    assert after > plain                                              # ... the drive adds to it


def test_slots_limit_new_topics_and_reopening(m):
    mem, r, birds, quantum, mm = m
    assert free_slots(mem, MCFG) == 1
    assert mm.add_topic("Bee navigation", thesis="Bees use polarised light.", questions=["How do bees find home"])
    assert free_slots(mem, MCFG) == 0
    assert mm.add_topic("Whale song") is None                         # no free slot
    _ready(mem, r, birds)
    mm.brief(birds)
    assert free_slots(mem, MCFG) == 1
    bee = [t for t in mem.topics() if t["name"] == "Bee navigation"][0]
    assert mem.findings(topic_id=bee["id"])[0]["text"] == "How do bees find home?"
    assert mm.add_topic("Whale song")
    assert not mm.reopen(birds)                                       # full again
    mem.update_topic([t for t in mem.topics() if t["name"] == "Whale song"][0]["id"], status="explored")
    assert mm.reopen(birds) and mem.get_topic(birds)["status"] == "active"


def test_dashboard_requests_are_carried_out_once(m):
    mem, r, birds, quantum, mm = m
    from piper_dashboard import api
    _ready(mem, r, birds)
    mm.check()
    api.request(mem.path, "mastery_decline", topic_id=birds, guidance="Look at magnetite in beaks.")
    api.request(mem.path, "topic_add", name="Bee navigation", thesis="", questions=["How do bees find home?"])
    assert [p["kind"] for p in pending_requests(mem)] == ["mastery_decline", "topic_add"]
    assert apply_requests(r) == 2 and pending_requests(mem) == []
    assert open_tagged(mem, birds, "mastery_gap") and any(t["name"] == "Bee navigation" for t in mem.topics())
    for g in open_tagged(mem, birds, "mastery_gap"):                   # guidance explored ...
        mem.end_episode(mem.start_episode(birds, angle=g["text"]), "stalled")
    mm.check()
    api.request(mem.path, "mastery_confirm", topic_id=birds)           # ... confirmed
    assert apply_requests(r) == 1 and mem.get_topic(birds)["status"] == "explored"


def test_accepting_a_sub_topic_proposal_moves_its_questions(m):
    mem, r, birds, quantum, mm = m
    from piper_dashboard import api
    src = mem.source("llm", "fake")
    qids = [mem.add_finding(q, quantum, kind="question", sources=[src], tags=["derived"])
            for q in ("Do plants sense magnetism?", "Do plants sense magnetic fields at night?")]
    pid = mm.propose("Plant magnetoreception", "d", "they kept pointing there", parent=quantum, questions=qids)
    assert mm.propose("Plant magnetoreception", "d", "again") is None  # not twice
    api.request(mem.path, "proposal_accept", proposal=pid)
    apply_requests(r)
    new = [t for t in mem.topics() if t["name"] == "Plant magnetoreception"][0]
    assert new["parent_id"] == quantum and new["origin"] == "spawned"
    assert {f["id"] for f in mem.findings(topic_id=new["id"])} == set(qids)
    assert not [x for x in mem.notable() if x["kind"] == "proposal"]  # handled
