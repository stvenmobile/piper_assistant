import numpy as np

from piper_vision.faces import UNKNOWN, FaceLibrary, Identity, usable_face

rng = np.random.default_rng(0)


def person():
    """A random 'face print' and a way to make noisy looks at the same face."""
    base = rng.normal(size=128)
    return lambda: base + rng.normal(scale=0.3, size=128)


def test_library_matches_the_right_person_and_rejects_strangers(tmp_path):
    steve, anna, stranger = person(), person(), person()
    lib = FaceLibrary(tmp_path)
    lib.add("Steve", [steve() for _ in range(5)])
    lib.add("Anna", [anna() for _ in range(5)])
    assert lib.match(steve())[0] == "Steve"
    assert lib.match(anna())[0] == "Anna"
    assert lib.match(stranger())[0] is None


def test_library_persists_adds_and_forgets(tmp_path):
    steve = person()
    FaceLibrary(tmp_path).add("Steve", [steve() for _ in range(3)])
    lib = FaceLibrary(tmp_path)                    # reload from disk
    assert lib.names() == ["Steve"]
    assert lib.add("steve", [steve()]) == 4        # same person, any case
    assert lib.names() == ["Steve"]
    assert lib.forget("STEVE")
    assert FaceLibrary(tmp_path).names() == []
    assert not lib.forget("Steve")


def test_library_keeps_only_the_newest_looks(tmp_path):
    steve = person()
    lib = FaceLibrary(tmp_path)
    assert lib.add("Steve", [steve() for _ in range(30)]) == FaceLibrary.MAX_PER_PERSON


def test_identity_needs_a_majority():
    ident = Identity()
    assert ident.verdict is None
    ident.add("Steve")
    ident.add(None)
    assert ident.verdict is None                   # not sure yet
    ident.add("Steve")
    assert ident.add("Steve") and ident.verdict == "Steve"
    ident.add(None)                                # one bad look doesn't change it
    assert ident.verdict == "Steve"


def test_identity_calls_a_stranger_after_several_misses():
    ident = Identity()
    for _ in range(3):
        ident.add(None)
    assert ident.verdict is None
    ident.add(None)
    assert ident.verdict == UNKNOWN
    ident.settle("Bob")                            # just enrolled
    assert ident.verdict == "Bob"


def face(w=120, nose_offset=0.0, score=0.95):
    lm = np.zeros(15, dtype=np.float32)
    lm[4], lm[6] = 100, 140                        # eyes 40 px apart
    lm[8] = 120 + nose_offset * 40
    return (80, 50, w, w, score, lm)


def test_usable_face_rejects_far_turned_and_doubtful_faces():
    assert usable_face(face())
    assert not usable_face(face(w=60))             # too far away
    assert not usable_face(face(nose_offset=0.4))  # turned away
    assert not usable_face(face(score=0.7))
    assert not usable_face(face()[:5])             # no landmarks
