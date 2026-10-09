"""
Quote checking - the research loop's defence against the model inventing things. A finding is
only kept if the quote it gives as support really appears in what Piper read.

Matching ignores case, spacing, quote/dash styles and punctuation at the ends; a quote that is
nearly verbatim (one word slipped) still passes, a paraphrase doesn't. A quote shortened with
"..." passes when every piece is found, in order, in the same text.
"""
import difflib
import re

_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "−": "-", " ": " "})
MIN_QUOTE = 20          # characters - shorter "quotes" prove nothing
FUZZY = 0.92            # share of the quote that must match in one stretch


def norm(text: str) -> str:
    return " ".join(text.translate(_TRANS).lower().split()).strip(" .,;:\"'")


def find_quote(quote: str, texts: list[str]) -> int | None:
    """The index of the text (passage) containing `quote`, or None. Exact (normalised) first,
    then nearly verbatim; an elided quote ("a ... b") piece by piece."""
    q = norm(quote)
    if len(q) < MIN_QUOTE:
        return None
    normed = [norm(t) for t in texts]
    pieces = [norm(p) for p in re.split(r"\[\.\.\.\]|\.\.\.|…", quote.translate(_TRANS))]
    pieces = [p for p in pieces if p]
    if len(pieces) > 1:
        if sum(len(p) for p in pieces) < MIN_QUOTE or min(len(p) for p in pieces) < 8:
            return None
        for i, t in enumerate(normed):
            pos = 0
            for p in pieces:
                pos = t.find(p, pos)
                if pos < 0:
                    break
                pos += len(p)
            else:
                return i
        return None
    for i, t in enumerate(normed):
        if q in t:
            return i
    for i, t in enumerate(normed):
        m = difflib.SequenceMatcher(None, q, t, autojunk=False)
        blocks = m.get_matching_blocks()
        # the quote's characters matched in order, within a stretch of the text not much longer
        matched = sum(b.size for b in blocks)
        if matched >= FUZZY * len(q):
            used = [b for b in blocks if b.size]
            span = used[-1].b + used[-1].size - used[0].b
            if span <= len(q) * 1.15:
                return i
    return None


def tidy_remark(text: str) -> str:
    """A remark as the model wrote it, minus its slips: other JSON fields run into it
    ('... \\n\\n"learned": ...'), literal \\n sequences, and word counts tacked on ('(39 words)')."""
    t = text or ""
    t = re.split(r'(?:\\n|\n)\s*"?[a-z_]+"?\s*:', t, maxsplit=1)[0]     # another field begins
    t = t.replace("\\n", " ")
    t = re.sub(r"\s*\(\s*\d+\s*words?\s*\)\s*$", "", t, flags=re.I)
    return " ".join(t.split()).strip(' "')


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
