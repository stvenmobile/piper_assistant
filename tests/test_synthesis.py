import random

import pytest

from piper_memory import MemoryStore
from piper_research import Researcher
from piper_research import prompts as P
from test_memory import FakeEmbedder
from test_research import CFG, PAGE, FakeLLM, FakeWiki, finding

SCFG = {**CFG, "synth_min_findings": 3, "synth_saturation": 0.5, "synth_every_max": 6, "synth_topic_gap": 3,
        "max_derived_open": 10, "synth_near_min": 0.0}


class SynthLLM(FakeLLM):
    """FakeLLM that also answers the synthesis schemas. `synth` is the next SYNTHESIZE reply
    (with {A}, {B}... standing for finding ids), `ground` the grounding verdict."""

    def __init__(self):
        super().__init__()
        self.synth = None
        self.ground = "grounded"
        self.effect = "supports"
        self.last_thinking = ""

    def json(self, system, user, schema, retries=1, think=None):
        if schema is P.SYNTH_SCHEMA:
            self.calls += 1
            self.last_thinking = "Hmm - both mention radical pairs..."
            ids = [int(x.split("]")[0]) for x in user.split("[F")[1:]]
            return self.synth(ids) if callable(self.synth) else self.synth
        if schema is P.GROUND_SCHEMA:
            self.calls += 1
            return {"verdict": self.ground, "reason": "they say so"}
        if schema is P.EVIDENCE_SCHEMA:
            self.calls += 1
            return {"effect": self.effect, "reason": "the new finding bears on it"}
        self.last_thinking = ""
        return super().json(system, user, schema, retries, think)


def insight(ids, statement="Birds and plants may both use radical pairs to sense light-driven fields.",
            questions=("Do plant cryptochromes respond to magnetic fields?",), confidence="plausible"):
    return {"steps": [{"text": "Both topics name radical pairs.", "cites": ids[:2]},
                      {"text": "A step citing something not shown.", "cites": [99999]}],
            "insights": [{"statement": statement, "kind": "analogy", "confidence": confidence,
                          "based_on": ids[:3], "questions": list(questions)}]}


@pytest.fixture
def two_topics(tmp_path):
    mem = MemoryStore(tmp_path / "mem.db", FakeEmbedder())
    birds = mem.topic("How migrating birds navigate", thesis="Birds navigate mainly by magnetism.")
    quantum = mem.topic("Quantum biology")
    rp = mem.concept("radical pair")
    texts = {birds: ["Cryptochrome in bird eyes forms radical pairs when hit by blue light.",
                     "Robins lose their magnetic compass under weak radio-frequency noise.",
                     "Radical pair spin states in birds are sensitive to weak magnetic fields."],
             quantum: ["Plant cryptochromes form radical pairs under blue light.",
                       "Radical pair reactions are a candidate quantum effect in biology."]}
    ids = {}
    for tid, ts in texts.items():
        for t in ts:
            fid = mem.add_finding(t, tid, confidence=0.6, stance="supports" if tid == birds else "neutral",
                                  judged_by="m")
            mem.relate(rp, "part_of", mem.concept("light sensing"), finding_id=fid)
            ids.setdefault(tid, []).append(fid)
    llm = SynthLLM()
    r = Researcher(mem, llm, FakeWiki([PAGE]), SCFG, log=lambda *_: None, rng=random.Random(2))
    yield mem, llm, r, birds, quantum, ids
    mem.close()


def test_synthesis_stores_an_insight_with_its_trail_and_derived_questions(two_topics):
    mem, llm, r, birds, quantum, ids = two_topics
    llm.synth = insight
    out = r.synthesize(mem.get_topic(birds))
    assert out["kind"] == "synthesis" and out["insights"] == 1 and out["derived_questions"] == 1
    ins = mem.insights()[0]
    assert ins["kind"] == "analogy" and ins["status"] == "new" and ins["confidence"] == 0.5
    t = ins["trace"]
    assert t["selection"]["strategy"] in ("bridge", "neighbourhood", "tension") and t["selection"]["why"]
    assert t["thinking"].startswith("Hmm") and t["grounding"]["verdict"] == "grounded"
    assert [s["check"] for s in t["steps"]] == ["ok", "unknown finding(s) [99999]"]     # each step checked
    assert set(ins["basis"]) <= set(t["selection"]["findings"])
    q = mem.get_finding(ins["questions"][0])
    assert q["kind"] == "question" and "derived" in q["tags"]
    ep = mem.episodes(birds, limit=1)[0]
    assert ep["kind"] == "synthesis" and ep["angle"].startswith("Synthesis (")
    assert any(e["kind"] == "synthesized" for e in mem.events())


def test_unsupported_insight_is_kept_as_rejected_without_questions(two_topics):
    mem, llm, r, birds, *_ = two_topics
    llm.synth, llm.ground = insight, "unsupported"
    out = r.synthesize(mem.get_topic(birds))
    assert out["insights"] == 0 and out["rejected_insights"] == 1 and out["outcome"] == "stalled"
    ins = mem.insights()[0]
    assert ins["status"] == "rejected" and ins["questions"] == []


def test_an_insight_must_rest_on_two_of_the_chosen_findings(two_topics):
    mem, llm, r, birds, *_ = two_topics
    llm.synth = lambda ids: {**insight(ids), "insights": [{**insight(ids)["insights"][0], "based_on": [ids[0], 4242]}]}
    r.synthesize(mem.get_topic(birds))
    ins = mem.insights()[0]
    assert ins["status"] == "rejected" and "fewer than 2" in ins["trace"]["grounding"]["reason"]
    assert ins["trace"]["stray_citations"] == [4242]


def test_partly_grounded_lowers_confidence(two_topics):
    mem, llm, r, birds, *_ = two_topics
    llm.synth, llm.ground = (lambda ids: insight(ids, confidence="strong")), "partly"
    r.synthesize(mem.get_topic(birds))
    assert mem.insights()[0]["confidence"] == 0.5               # strong 0.7 - 0.2


def test_a_restated_insight_strengthens_the_existing_one(two_topics):
    mem, llm, r, birds, *_ = two_topics
    llm.synth = insight
    r.synthesize(mem.get_topic(birds))
    first = mem.insights()[0]
    mem.db.execute("DELETE FROM events WHERE kind = 'synthesized'")     # let the same material be chosen again
    out = r.synthesize(mem.get_topic(birds))
    assert out["restated"] == 1 and len(mem.insights()) == 1
    again = mem.get_insight(first["id"])
    assert again["confidence"] == pytest.approx(0.55) and again["events"][-1]["kind"] == "restated"


def test_derived_questions_have_their_own_cap(two_topics):
    mem, llm, r, birds, *_ = two_topics
    r.cfg = {**SCFG, "max_derived_open": 1}
    llm.synth = lambda ids: insight(ids, questions=("Do plants sense magnetism?", "Do insects use radical pairs?"))
    out = r.synthesize(mem.get_topic(birds))
    assert out["derived_questions"] == 1


def test_step_synthesizes_when_saturated_and_not_twice_running(two_topics):
    mem, llm, r, birds, *_ = two_topics
    llm.synth = insight
    topic = {**mem.get_topic(birds), "saturation": 0.8}
    assert r.wants_synthesis(topic)
    r.synthesize(mem.get_topic(birds))
    assert not r.wants_synthesis(topic)                         # its last cycle was a synthesis
    assert not r.wants_synthesis({**mem.get_topic(birds), "saturation": 0.0})


def test_step_synthesizes_after_too_long_without_one_but_needs_enough_findings(two_topics):
    mem, llm, r, birds, quantum, _ = two_topics
    topic = {**mem.get_topic(birds), "saturation": 0.0}
    r.cycles_since_synthesis = 6
    assert r.wants_synthesis(topic)
    r.cfg = {**SCFG, "synth_min_findings": 50}
    assert not r.wants_synthesis(topic)


def test_research_on_a_derived_question_moves_the_insights_confidence(two_topics):
    mem, llm, r, birds, *_ = two_topics
    llm.synth = insight
    r.synthesize(mem.get_topic(birds))
    ins = mem.insights()[0]
    q = mem.get_finding(ins["questions"][0])["text"]
    llm.extract = [finding("Birds use a stellar compass at night.", "At night many species use a stellar compass")]
    r.next_question = lambda topic: {"question": q, "why": "", "queries": ["birds"]}
    out = r.cycle(topic=mem.get_topic(birds))
    assert out["insight_evidence"] == "supports"
    after = mem.get_insight(ins["id"])
    assert after["status"] == "supported" and after["confidence"] == pytest.approx(0.6)
    ev = after["events"][-1]
    assert ev["kind"] == "evidence" and ev["effect"] == "supports" and ev["findings"]
    llm.effect = "weakens"
    out = r.cycle(topic=mem.get_topic(birds))                      # nothing new this time
    assert mem.get_insight(ins["id"])["events"][-1]["effect"] == "none"
