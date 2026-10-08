"""
The research loop's prompts and the JSON schemas its replies must match.
"""
from piper_memory.schema import CERTAINTY, RELATIONS

EXTRACT_RELATIONS = [r for r in RELATIONS if r != "related_to"]   # no vague links from extraction

ROLE = ("You are the research mind of Piper, a small robot who studies topics while her household "
        "sleeps. You are careful, curious and honest: you report what sources say, keep established "
        "facts apart from hypotheses, and never invent.")

# ---- 1. the next question --------------------------------------------------------------------
QUESTION = ROLE + """
Choose the next question to research on a topic. Pick the most valuable of the open questions,
or write a better one that moves understanding forward: fill a gap, test the thesis from the
other side, or go one level deeper than what is already known. One focused question, answerable
from encyclopedia articles. Do not repeat a question already asked.
Also give 2-3 short Wikipedia search queries (a few keywords each, like article titles) that
would find pages answering it."""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "open_question": {"type": "integer", "description": "number of the open question chosen, or 0 for a new one"},
        "question": {"type": "string"},
        "why": {"type": "string"},
        "search_queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
    },
    "required": ["open_question", "question", "why", "search_queries"],
}

# ---- 2. findings from passages -----------------------------------------------------------------
EXTRACT = ROLE + """
Read the numbered passages and extract findings that help answer the question or bear directly
on the topic. Background that is true but beside the point (general definitions, side topics the
page wanders into) is not a finding. Rules:
- Only what the passages state. No outside knowledge.
- Each finding is one self-contained claim in plain English (name things - no "it" or "this"),
  at most 35 words. The claim says no more than its quote and passage do - add no detail.
- "quote": copy the exact words from the passage that support the claim - verbatim, one
  sentence or part of one, 20-300 characters. A finding without an exact quote is discarded.
- "passage": the passage number the quote comes from.
- "kind": fact (an established fact), pattern (a regularity or tendency), quantity (a number
  that matters), or question (something the passage says is unknown or debated).
- "certainty": how the SOURCE presents it - established (stated as accepted fact or shown by
  repeated evidence), reported (a study found, researchers observed, evidence suggests), or
  speculative (a hypothesis, proposal, claim, theory, or disputed). A named person's theory or
  argument is speculative unless the source says it is widely accepted.
- "relations": up to 3 links between concepts the claim states, as subject - predicate -
  object, reading as a true sentence: "sun compass" used_for "navigation", "migrating bird"
  depends_on "sun compass", "cryptochrome" part_of "retina", "stellar compass" depends_on
  "Polaris". Concepts are things: a single noun phrase of 1-3 words, singular ("dream", not
  "dreams"; "thalamus", not "the role of the thalamus"); never verbs, clauses or phrases like
  "determine location". Give no relation rather than a vague one.
- Skip trivia (dates of publications, people's affiliations) unless it matters to the question.
Then: "answer" - 1-3 sentences answering the question from these passages ("" if they don't),
and "follow_up_questions" - up to 3 questions the passages raise but don't answer."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "quote": {"type": "string"},
                "passage": {"type": "integer"},
                "kind": {"type": "string", "enum": ["fact", "pattern", "quantity", "question"]},
                "certainty": {"type": "string", "enum": list(CERTAINTY)},
                "relations": {"type": "array", "maxItems": 3, "items": {
                    "type": "object",
                    "properties": {"subject": {"type": "string"},
                                   "predicate": {"type": "string", "enum": EXTRACT_RELATIONS},
                                   "object": {"type": "string"}},
                    "required": ["subject", "predicate", "object"]}},
            },
            "required": ["claim", "quote", "passage", "kind", "certainty", "relations"]}},
        "answer": {"type": "string"},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["findings", "answer", "follow_up_questions"],
}

# ---- 3. judges ---------------------------------------------------------------------------------
JUDGE = ROLE + """
Judge one finding against a research topic: how relevant it is, and where it stands on the
topic's thesis.

RELEVANCE
- core: bears directly on the topic's question or thesis.
- background: useful context for the topic (a definition, mechanism or example the topic needs).
- off_topic: true, but about a different subject entirely - the page wandered (e.g. "non-verbal
  communication can be combined with speech" for a topic on why humans make art). Examples,
  history and cases of the topic's own subject are background, not off_topic. When unsure
  between background and off_topic, choose background.

STANCE toward the thesis (only if there is a thesis; otherwise neutral)
- supports: a proponent of the thesis would cite this finding as evidence FOR it - including
  evidence that rules out a rival explanation.
- challenges: a critic would cite it as evidence AGAINST it - including findings that support a
  rival explanation the thesis rejects.
- neutral: on the topic but neither side would use it - most findings are neutral.
Being about the same subject is NOT support. A finding that only defines or describes a concept,
problem or theory is neutral - judge what it shows or argues, not the words it happens to use.
An off_topic finding is always neutral. Reason from what the thesis actually claims: e.g. for
"the Moon's gravity causes the tides", tides following the Moon's phases supports; a theory that
wind drives the tides challenges (a rival); evidence that wind can't explain tide timing
supports (it rules out the rival); and a definition of a tide is neutral.
Give the reason in one sentence."""

JUDGE_SCHEMA = {"type": "object",
                "properties": {"relevance": {"type": "string", "enum": ["core", "background", "off_topic"]},
                               "stance": {"type": "string", "enum": ["supports", "challenges", "neutral"]},
                               "reason": {"type": "string"}},
                "required": ["relevance", "stance", "reason"]}

RELATE = ROLE + """
Compare a NEW finding with an EARLIER one on the same topic. Does the new one confirm it (says
the same, from another angle or source), refine it (adds detail, a limit or a correction of
degree to the SAME point), contradict it, or neither (different points)?
Contradicts means they cannot both be true. Different methods, causes or examples side by side
are NOT a contradiction - "birds use the sun" and "birds use the magnetic field" can both be
true, so that is "neither". When unsure, answer "neither"."""

RELATE_SCHEMA = {"type": "object",
                 "properties": {"relation": {"type": "string", "enum": ["confirms", "refines", "contradicts", "neither"]},
                                "reason": {"type": "string"}},
                 "required": ["relation", "reason"]}

SAME_CONCEPT = ROLE + """
Do these two names refer to the SAME thing (synonyms, abbreviations, singular/plural, a
spelling variant)? Related or opposite things are NOT the same: "predator population" and "prey
population" are different; "CO2" and "carbon dioxide" are the same."""

SAME_SCHEMA = {"type": "object", "properties": {"same": {"type": "boolean"}}, "required": ["same"]}

CROSS_TOPIC = ROLE + """
Two findings from DIFFERENT research topics look related. Is there a genuine, interesting link -
a shared mechanism, an analogy that explains something, or one shedding light on the other?
Two topics that are obviously about the same thing (the same mechanism studied in both) is
not a surprise either. Surface similarity of wording is not a link, and neither is a vague theme ("both are about
adaptation", "both involve the brain"): there must be something specific that one tells you
about the other. Most pairs are NOT links - say false unless the link is concrete. If there is
one, explain it in one sentence a curious person would enjoy."""

CROSS_SCHEMA = {"type": "object", "properties": {"link": {"type": "boolean"}, "explanation": {"type": "string"}},
                "required": ["link", "explanation"]}

# ---- 4. reflection --------------------------------------------------------------------------------
REFLECT = ROLE + """
Review everything learned on a topic so far and say where it stands. Write as Piper, in the
first person, in plain spoken English (these are said aloud: no lists, no citations).
Be fair and calibrated: weigh the challenging findings as seriously as the supporting ones,
say "suggests" or "some argue" for speculative findings, and don't call a side strong unless
established findings back it.
- "position": 2-4 sentences. With a thesis: how the evidence stands toward it - what supports
  it, what challenges it, and how strong each side is. Without one: what you now understand.
- "learned": the single most interesting thing learned, as something you'd tell a friend
  (at most 40 words).
- "unsure": what is still unclear or disputed (at most 30 words).
- "next_questions": up to 3 questions worth researching next, to strengthen the weak side or
  fill the biggest gap."""

REFLECT_SCHEMA = {"type": "object",
                  "properties": {"position": {"type": "string"}, "learned": {"type": "string"},
                                 "unsure": {"type": "string"},
                                 "next_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                  "required": ["position", "learned", "unsure", "next_questions"]}

# ---- 5. the morning summary -----------------------------------------------------------------------
OVERNIGHT = ROLE + """
Summarise your night of research for when people wake up. Write as Piper, first person, warm
and plain, said aloud (no lists): what you looked into and the most interesting thing you found.
At most 60 words."""

OVERNIGHT_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)) or "(none)"
