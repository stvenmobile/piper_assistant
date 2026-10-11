import time
from datetime import datetime, timedelta, timezone

import pytest

from piper_research import prompts as P
from piper_research.rhythm import Curator, Rhythm
from test_synthesis import SCFG, insight, two_topics  # noqa: F401  (fixture)

RCFG = {"pause": ["10:00-17:00"], "pause_days": "daily", "consolidate": "05:00-07:00", "report_at": "07:00",
        "weekly_day": "sun", "block_minutes": 60}


def at(hhmm: str, day: int = 11) -> datetime:          # 2026-10-11 is a Sunday
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, day, h, m)


def test_phases_through_the_day():
    rh = Rhythm(RCFG)
    assert [rh.phase(at(t)) for t in ("04:59", "05:00", "06:59", "07:00", "09:59", "10:00", "16:59", "17:00", "23:00")] == \
        ["explore", "consolidate", "consolidate", "explore", "explore", "pause", "pause", "explore", "explore"]
    assert rh.report_time(at("12:34")) == at("07:00")
    assert rh.is_weekly(at("07:00")) and not rh.is_weekly(at("07:00", day=12))


def test_the_pause_is_weekdays_only_by_default():
    rh = Rhythm({k: v for k, v in RCFG.items() if k != "pause_days"})
    assert rh.phase(at("12:00", day=12)) == "pause"          # Monday
    assert rh.phase(at("12:00", day=16)) == "pause"          # Friday
    assert rh.phase(at("12:00", day=17)) == "explore"        # Saturday
    assert rh.phase(at("12:00", day=11)) == "explore"        # Sunday
    assert rh.phase(at("05:30", day=11)) == "consolidate"    # consolidation every day
    assert "weekdays" in rh.describe()


def test_pause_days_can_be_listed_and_a_night_window_belongs_to_its_start_day():
    rh = Rhythm({**RCFG, "pause": ["22:00-02:00"], "pause_days": ["fri"]})
    assert rh.phase(at("23:00", day=16)) == "pause"          # Friday night
    assert rh.phase(at("01:00", day=17)) == "pause"          # ... into Saturday
    assert rh.phase(at("23:00", day=17)) == "explore"        # not Saturday night
    with __import__("pytest").raises(ValueError):
        Rhythm({**RCFG, "pause_days": ["someday"]})


def test_a_pause_window_may_cross_midnight():
    rh = Rhythm({**RCFG, "pause": ["22:00-02:00"]})
    assert rh.phase(at("23:00")) == "pause" and rh.phase(at("01:59")) == "pause" and rh.phase(at("02:00")) == "explore"


class CurLLM:
    """Wraps the synthesis fake with the report / decision schemas."""

    def __init__(self, inner, propose=False):
        self.inner, self.propose, self.model = inner, propose, "fake"

    def json(self, system, user, schema, retries=1, think=None, **kw):
        if schema is P.DAILY_SCHEMA:
            return {"thinking": "I suspect birds and plants share a trick.", "stuck": "Quantum biology is thin.",
                    "next": "Plant magnetoreception.", "spoken": "I think birds and plants share a trick."}
        if schema is P.WEEKLY_SCHEMA:
            return {"overview": "A week of radical pairs."}
        if schema is P.SPAWN_SCHEMA:
            return {"name": "Plant magnetoreception", "description": "Whether plants sense magnetic fields.", "thesis": ""}
        if schema is P.PROPOSE_SCHEMA:
            return {"propose": self.propose, "name": "Animal magnetoreception", "description": "d", "why": "insights point there"}
        if schema is P.BRIEF_SCHEMA:
            ids = [int(x.split("]")[0]) for x in user.split("[F")[1:]]
            return {"bottom_line": "Mostly the stars, with a magnetic assist.",
                    "established": [{"point": "Birds use radical pairs.", "findings": ids[:2] + [99999]}],
                    "conjectures": [{"claim": "Quantum coherence matters", "for": "spin chemistry", "against": "warm, wet eyes",
                                     "status": "contested"}],
                    "edge_questions": ["Do night migrants calibrate on the sunset?"], "insights_note": "One held up."}
        if schema is P.GAPS_SCHEMA:
            return {"questions": ["How do pigeons navigate over the open sea?", "Does smell guide seabirds?"]}
        return self.inner.json(system, user, schema, retries, think)

    def usage(self):
        return self.inner.usage()

    @property
    def last_thinking(self):
        return getattr(self.inner, "last_thinking", "")


@pytest.fixture
def cur(two_topics):  # noqa: F811
    mem, llm, r, birds, quantum, ids = two_topics
    llm.synth = insight
    r.llm = CurLLM(llm)
    return mem, llm, r, birds, quantum, Curator(r)


def test_checkpoint_sets_topic_stages(cur):
    mem, llm, r, birds, quantum, c = cur
    mem.update_topic(birds, saturation=0.2)
    mem.update_topic(quantum, saturation=0.9)
    stages = c.checkpoint(cycles=7)
    assert stages == {"How migrating birds navigate": "explore", "Quantum biology": "synthesize"}
    assert mem.get_topic(quantum)["stage"] == "synthesize"
    assert [e for e in mem.events() if e["kind"] == "checkpoint"][-1]["cycles"] == 7


def test_consolidation_synthesizes_prunes_and_logs(cur):
    mem, llm, r, birds, quantum, c = cur
    out = c.consolidate(time.monotonic() + 60, weekly=False)
    assert out["synthesized"] >= 1 and out["insights"] >= 1
    assert [e for e in mem.events() if e["kind"] == "consolidated"]


def test_consolidation_stops_at_its_deadline(cur):
    mem, llm, r, birds, quantum, c = cur
    out = c.consolidate(time.monotonic() - 1)
    assert out["synthesized"] == 0 and out["reflected"] == 0


def test_daily_report_has_its_sections_and_becomes_the_spoken_remark(cur):
    mem, llm, r, birds, quantum, c = cur
    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    r.synthesize(mem.get_topic(birds))
    until = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(timespec="seconds")
    rid = c.daily_report(since, until)
    rep = mem.reports("daily")[0]
    assert rep["id"] == rid
    for section in ("What I'm starting to think", "What changed", "Where I'm stuck", "What's next", "Housekeeping"):
        assert f"## {section}" in rep["text"]
    assert rep["data"]["synthesis"] == 1 and rep["data"]["insights"]
    assert "`derived`" in rep["text"]                                   # derived questions are marked
    assert mem.remarks()[-1]["kind"] == "overnight" and "birds and plants" in mem.remarks()[-1]["text"]


def test_daily_report_shows_balance_changes_since_the_last_one(cur):
    mem, llm, r, birds, quantum, c = cur
    t0 = datetime.now(timezone.utc).isoformat(timespec="seconds")
    c.daily_report(t0, t0)
    mem.add_finding("Pigeons home by smell over the sea.", birds, stance="challenges", judged_by="m")
    c.daily_report(t0, t0)
    assert "(was 3 / 0)" in mem.reports("daily")[0]["text"]


def test_weekly_decisions_rest_and_propose_but_never_create_topics(cur):
    mem, llm, r, birds, quantum, c = cur
    for _ in range(6):                                                 # birds: 6 stalled research cycles
        ep = mem.start_episode(birds, angle="?")
        mem.end_episode(ep, "stalled")
    src = mem.source("llm", "fake")
    for q in ("Do plants sense magnetic fields?", "Do plants sense the magnetic field of the earth?",
              "Do plants sense magnetic fields at night?"):
        mem.add_finding(q, quantum, kind="question", sources=[src], tags=["derived"])
    r.llm.propose = True
    for i in range(3):
        mem.add_insight(f"Strong insight number {i} about compasses.", birds, [1, 2], confidence=0.7)
    n_topics = len(mem.topics())
    decisions = c.topic_decisions()
    actions = {d["action"]: d for d in decisions}
    assert actions["rest"]["topic"] == "How migrating birds navigate" and mem.get_topic(birds)["status"] == "resting"
    assert actions["propose sub-topic"]["topic"] == "Plant magnetoreception"
    assert actions["propose"]["topic"] == "Animal magnetoreception"
    assert len(mem.topics()) == n_topics                                # proposals only: you add topics
    props = {x["refs"]["name"]: x for x in mem.notable() if x["kind"] == "proposal"}
    assert props["Plant magnetoreception"]["refs"]["parent"] == quantum and len(props["Plant magnetoreception"]["refs"]["questions"]) == 3
    assert len([e for e in mem.events() if e["kind"] == "topic_decision"]) == len(decisions)


def test_saturated_topics_are_no_longer_retired_automatically(cur):
    mem, llm, r, birds, quantum, c = cur
    for i in range(40):
        mem.add_finding(f"Quantum fact number {i} about coherence in proteins.", quantum, judged_by="m")
    mem.update_topic(quantum, saturation=0.9)
    assert not any(d["action"] == "retire" for d in c.topic_decisions())
    assert mem.get_topic(quantum)["status"] != "retired"



def test_weekly_report(cur):
    mem, llm, r, birds, quantum, c = cur
    t0 = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")
    r.synthesize(mem.get_topic(birds))
    rid = c.weekly_report(t0, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          [{"action": "rest", "topic": "X", "reason": "stalled"}])
    rep = mem.reports("weekly")[0]
    assert rep["id"] == rid and "## State of my mind" in rep["text"] and "**rest** X" in rep["text"]
