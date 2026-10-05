"""
Who is that? The pure-logic half of face recognition - no camera, no OpenCV, so it can be
unit-tested. (recognizer.py turns a face into an embedding; this file decides what it means.)

  * FaceLibrary - the people Piper has met: a few face embeddings (SFace, 128 numbers) per
    name, stored in faces/ (git-ignored - they're biometric data and never leave the Jetson).
  * Identity    - per tracked person, a vote over the last few looks, so one bad frame
    can't turn Steve into a stranger (or a stranger into Steve).
  * usable_face - is this detection good enough to recognise / enrol from?
"""
import json
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np

FACES_DIR = Path(__file__).resolve().parents[2] / "faces"
UNKNOWN = ""                    # Identity's verdict for "a stranger" (None = not decided yet)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def normalise(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceLibrary:
    """faces/people.json maps slug -> display name; faces/<slug>.npy holds that person's
    embeddings (n x 128, unit length)."""

    MAX_PER_PERSON = 20

    def __init__(self, folder: Path = FACES_DIR, threshold: float = 0.40):
        self.dir = Path(folder)
        self.threshold = threshold          # cosine similarity; SFace's published cut is 0.363
        self.people: dict[str, np.ndarray] = {}     # display name -> embeddings
        self.load()

    def load(self):
        self.people = {}
        index = self.dir / "people.json"
        if not index.exists():
            return
        for key, name in json.loads(index.read_text(encoding="utf-8")).items():
            f = self.dir / f"{key}.npy"
            if f.exists():
                self.people[name] = np.load(f)

    def _save_index(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        index = {slug(n): n for n in self.people}
        (self.dir / "people.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    def names(self) -> list[str]:
        return sorted(self.people)

    def find(self, name: str) -> str | None:
        """The stored spelling of a name, matched case-insensitively."""
        return next((n for n in self.people if slug(n) == slug(name)), None)

    def add(self, name: str, embeddings) -> int:
        """Add looks to a person (new or existing); keeps the newest MAX_PER_PERSON.
        Returns how many they now have."""
        name = self.find(name) or name.strip()
        new = np.stack([normalise(e) for e in embeddings])
        old = self.people.get(name)
        allv = new if old is None else np.vstack([old, new])
        allv = allv[-self.MAX_PER_PERSON:]
        self.people[name] = allv
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / f"{slug(name)}.npy", allv)
        self._save_index()
        return len(allv)

    def forget(self, name: str) -> bool:
        stored = self.find(name)
        if stored is None:
            return False
        del self.people[stored]
        (self.dir / f"{slug(stored)}.npy").unlink(missing_ok=True)
        self._save_index()
        return True

    def match(self, embedding) -> tuple[str | None, float]:
        """(name, similarity) of the best match, or (None, best similarity) if nobody is
        close enough."""
        e = normalise(embedding)
        best, best_sim = None, -1.0
        for name, embs in self.people.items():
            sim = float(np.max(embs @ e))
            if sim > best_sim:
                best, best_sim = name, sim
        return (best, best_sim) if best_sim >= self.threshold else (None, best_sim)


class Identity:
    """Votes over the last `window` looks at one tracked person.
    verdict: None = not sure yet, UNKNOWN ("") = a stranger, otherwise their name."""

    def __init__(self, window: int = 5, known_votes: int = 3, unknown_votes: int = 4):
        self.votes = deque(maxlen=window)
        self.known_votes = known_votes
        self.unknown_votes = unknown_votes
        self.verdict: str | None = None

    def add(self, name: str | None) -> bool:
        """One recognition result (None = no match). Returns True if the verdict changed."""
        self.votes.append(name or UNKNOWN)
        counts = Counter(self.votes)
        verdict = self.verdict
        for who, n in counts.most_common():
            if who != UNKNOWN and n >= self.known_votes:
                verdict = who
                break
        else:
            if counts[UNKNOWN] >= self.unknown_votes:
                verdict = UNKNOWN
        changed = verdict != self.verdict
        self.verdict = verdict
        return changed

    def settle(self, name: str):
        """Just enrolled: we know who this is."""
        self.votes.clear()
        self.votes.extend([name] * self.known_votes)
        self.verdict = name


def usable_face(face, min_width: float = 90, max_turn: float = 0.25, min_score: float = 0.85) -> bool:
    """Good enough to recognise from? face = (x, y, w, h, score, landmarks) where landmarks is
    YuNet's row (x, y, w, h, right eye x/y, left eye x/y, nose x/y, mouth corners..., score).
    Rejects small (far), low-confidence and turned-away faces - YuNet happily tracks a profile,
    but a profile makes a poor face print."""
    x, y, w, h, score = face[:5]
    if w < min_width or score < min_score or len(face) < 6:
        return False
    lm = face[5]
    ex1, ex2, nx = lm[4], lm[6], lm[8]
    eye_dist = abs(ex2 - ex1)
    if eye_dist < 1:
        return False
    turn = abs(nx - (ex1 + ex2) / 2) / eye_dist        # 0 = facing the camera, ~0.5 = profile
    return turn <= max_turn
