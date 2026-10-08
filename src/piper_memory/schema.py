"""
Piper's memory - the SQLite schema (the source of truth; the Obsidian vault and the dashboard
are views of it).

    topics      the hierarchy + agenda: what Piper studies, with her curiosity scores, and an
                optional THESIS - a position to examine (map who argues what, not to confirm)
    episodes    the research diary: one row per research cycle on a topic
    sources     where things were learned (wikipedia | web | llm | user), de-duplicated by `ref`
    findings    single claims (fact | pattern | quantity | question), each with its topic, the
                episode that produced it, a confidence, its STANCE toward the topic's thesis
                (supports | challenges | neutral), and how it relates to what was already
                known (new | confirms | refines | contradicts -> relates_to); embedded.
                A judge sets RELEVANCE (core | background | off_topic - off-topic findings are
                kept but left out of the balance and reflection) and the stance's reason
    finding_sources   finding <-> source, with the supporting excerpt        (PROVENANCE)
    concepts    the knowledge graph's nodes, with aliases; embedded
    relations   subject concept -predicate-> object concept, each from a finding  (PROVENANCE)
    series      numbers worth graphing, attached to a finding
    remarks     prepared remarks (spoken without an LLM), superseded as understanding grows
    notable     things flagged for a human to look at
    tags        light labels for browsing
    events      the activity log the dashboard's live view follows

Times are ISO-8601 UTC strings. JSON columns hold flexible extras.
"""

SCHEMA_VERSION = 3              # v2: topics.thesis, findings.stance; v3: relevance, stance_reason, judged_by

# upgrades from each older version (applied in order by MemoryStore)
MIGRATIONS = {
    2: ["ALTER TABLE topics ADD COLUMN thesis TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE findings ADD COLUMN stance TEXT NOT NULL DEFAULT 'neutral'"],
    3: ["ALTER TABLE findings ADD COLUMN relevance TEXT NOT NULL DEFAULT 'core'",
        "ALTER TABLE findings ADD COLUMN stance_reason TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE findings ADD COLUMN judged_by TEXT"],
}

# relation types - a small fixed vocabulary, so the graph stays queryable
RELATIONS = (
    "causes",          # A causes / drives B
    "enables",         # A makes B possible
    "prevents",        # A stops / limits B
    "part_of",         # A is a part / component of B
    "example_of",      # A is an instance / example of B
    "similar_to",      # A resembles B (analogy)
    "contrasts_with",  # A differs from / opposes B
    "precedes",        # A comes before B (time / sequence / stage)
    "depends_on",      # A needs B
    "produces",        # A makes / yields B
    "measured_by",     # A is quantified by B
    "located_in",      # A is found in / at B
    "used_for",        # A is used to do B
    "related_to",      # fallback: linked, nature unclear
)

FINDING_KINDS = ("fact", "pattern", "quantity", "question")
FINDING_STATUS = ("new", "confirms", "refines", "contradicts")
STANCES = ("supports", "challenges", "neutral")          # toward the topic's thesis
RELEVANCE = ("core", "background", "off_topic")          # to the topic
CERTAINTY = {"established": 0.9, "reported": 0.6, "speculative": 0.3}   # -> findings.confidence
SOURCE_KINDS = ("wikipedia", "web", "llm", "user")
TOPIC_STATUS = ("queued", "active", "resting", "retired")

DDL = """
CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    parent_id   INTEGER REFERENCES topics(id),
    description TEXT NOT NULL DEFAULT '',
    thesis      TEXT NOT NULL DEFAULT '',          -- a position to examine ('' = open exploration)
    origin      TEXT NOT NULL DEFAULT 'seed',      -- seed | user | spawned | surprise | calibration
    status      TEXT NOT NULL DEFAULT 'queued',    -- queued | active | resting | retired
    novelty     REAL,                              -- curiosity scores, 0..1 (set by the loop)
    progress    REAL,
    saturation  REAL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    topic_id    INTEGER NOT NULL REFERENCES topics(id),
    angle       TEXT NOT NULL DEFAULT '',          -- the question / direction of this cycle
    model       TEXT NOT NULL DEFAULT '',
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    outcome     TEXT,                              -- e.g. progress | stalled | interrupted | error
    summary     TEXT NOT NULL DEFAULT '',
    stats       TEXT NOT NULL DEFAULT '{}'         -- JSON
);

CREATE TABLE IF NOT EXISTS sources (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,                    -- wikipedia | web | llm | user
    title        TEXT NOT NULL DEFAULT '',
    url          TEXT,
    ref          TEXT NOT NULL UNIQUE,             -- de-duplication key: the url, or kind:title
    retrieved_at TEXT NOT NULL,
    meta         TEXT NOT NULL DEFAULT '{}'        -- JSON
);

CREATE TABLE IF NOT EXISTS findings (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'fact',      -- fact | pattern | quantity | question
    topic_id    INTEGER NOT NULL REFERENCES topics(id),
    episode_id  INTEGER REFERENCES episodes(id),
    confidence  REAL NOT NULL DEFAULT 0.5,
    status      TEXT NOT NULL DEFAULT 'new',       -- new | confirms | refines | contradicts
    stance      TEXT NOT NULL DEFAULT 'neutral',   -- supports | challenges | neutral (toward the thesis)
    relevance   TEXT NOT NULL DEFAULT 'core',      -- core | background | off_topic (to the topic)
    stance_reason TEXT NOT NULL DEFAULT '',        -- the judge's one-line reason
    judged_by   TEXT,                              -- model that judged stance + relevance (NULL = not yet)
    relates_to  INTEGER REFERENCES findings(id),   -- the finding it confirms / refines / contradicts
    created_at  TEXT NOT NULL,
    embedding   BLOB                               -- float32, unit length (NULL if not embedded yet)
);

CREATE TABLE IF NOT EXISTS finding_sources (
    finding_id  INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    excerpt     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (finding_id, source_id)
);

CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    aliases     TEXT NOT NULL DEFAULT '[]',        -- JSON list of other names merged into it
    created_at  TEXT NOT NULL,
    embedding   BLOB
);

CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY,
    subject_id  INTEGER NOT NULL REFERENCES concepts(id),
    predicate   TEXT NOT NULL,
    object_id   INTEGER NOT NULL REFERENCES concepts(id),
    finding_id  INTEGER REFERENCES findings(id) ON DELETE CASCADE,
    confidence  REAL NOT NULL DEFAULT 0.5,
    created_at  TEXT NOT NULL,
    UNIQUE (subject_id, predicate, object_id, finding_id)
);

CREATE TABLE IF NOT EXISTS series (
    id          INTEGER PRIMARY KEY,
    finding_id  INTEGER REFERENCES findings(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    x_label     TEXT NOT NULL DEFAULT '',
    y_label     TEXT NOT NULL DEFAULT '',
    units       TEXT NOT NULL DEFAULT '',
    points      TEXT NOT NULL DEFAULT '[]',        -- JSON [[x, y], ...]
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS remarks (
    id            INTEGER PRIMARY KEY,
    topic_id      INTEGER REFERENCES topics(id),   -- NULL = about Piper's research in general
    kind          TEXT NOT NULL,                   -- e.g. learned | interesting | unsure | next
    text          TEXT NOT NULL,
    written_at    TEXT NOT NULL,
    superseded_by INTEGER REFERENCES remarks(id)   -- NULL = current
);

CREATE TABLE IF NOT EXISTS notable (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,                     -- contradiction | surprise | loop | jump | garbled ...
    title       TEXT NOT NULL,
    refs        TEXT NOT NULL DEFAULT '{}',        -- JSON: {"findings": [...], "topics": [...], ...}
    created_at  TEXT NOT NULL,
    reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE TABLE IF NOT EXISTS finding_tags (
    finding_id  INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id),
    PRIMARY KEY (finding_id, tag_id)
);

CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY,
    ts    TEXT NOT NULL,
    kind  TEXT NOT NULL,                           -- e.g. topic_chosen | query_sent | finding ...
    data  TEXT NOT NULL DEFAULT '{}'               -- JSON
);

CREATE INDEX IF NOT EXISTS ix_findings_topic   ON findings(topic_id);
CREATE INDEX IF NOT EXISTS ix_findings_episode ON findings(episode_id);
CREATE INDEX IF NOT EXISTS ix_relations_subj   ON relations(subject_id);
CREATE INDEX IF NOT EXISTS ix_relations_obj    ON relations(object_id);
CREATE INDEX IF NOT EXISTS ix_relations_find   ON relations(finding_id);
CREATE INDEX IF NOT EXISTS ix_episodes_topic   ON episodes(topic_id);
CREATE INDEX IF NOT EXISTS ix_remarks_topic    ON remarks(topic_id);

-- full-text search over findings, kept in step by triggers
CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(text, content='findings', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
    INSERT INTO findings_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE OF text ON findings BEGIN
    INSERT INTO findings_fts(findings_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO findings_fts(rowid, text) VALUES (new.id, new.text);
END;
"""
