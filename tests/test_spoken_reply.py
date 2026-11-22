"""Spoken replies are trimmed to a few complete sentences, without markdown or list numbering."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("langchain_ollama")
from piper_brain.supervisor import spoken_reply  # noqa: E402

LONG = """That's a fascinating idea. The concept of free will is debated.

In quantum mechanics, superposition refers to multiple states. Some challenges include:

1. The measurement problem: how does measurement cause collapse?
2. The problem of decoherence.
"""


def test_keeps_the_first_sentences():
    assert spoken_reply(LONG, 3) == ("That's a fascinating idea. The concept of free will is debated. "
                                     "In quantum mechanics, superposition refers to multiple states.")


def test_strips_list_numbers_and_markdown():
    out = spoken_reply("**Sure.**\n- First, rest.\n2) Then eat.", 5)
    assert out == "Sure. First, rest. Then eat."


def test_drops_a_sentence_cut_off_by_the_token_cap():
    assert spoken_reply("It is tomorrow. That said, the opposite of yester", 3) == "It is tomorrow."


def test_short_reply_unchanged():
    assert spoken_reply("The opposite of yesterday would be tomorrow.") == "The opposite of yesterday would be tomorrow."
