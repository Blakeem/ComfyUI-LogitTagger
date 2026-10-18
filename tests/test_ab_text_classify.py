"""tests-AB/ab_text_classify.py accuracy and per-wording lines over small synthetic records."""

import subprocess
import sys
from pathlib import Path

import pytest

TESTS_AB = Path(__file__).resolve().parent.parent / "tests-AB"
SCRIPT = TESTS_AB / "ab_text_classify.py"

# The harness imports its loaders from the harnesses beside it, as a script run finds them.
if str(TESTS_AB) not in sys.path:
    sys.path.append(str(TESTS_AB))

# The release-gates loaders it imports read the library's tag helpers at module scope.
pytest.importorskip("logit_classifier.toolkit.tags")

import ab_text_classify  # noqa: E402

UNITS = ("g01", "g02", "g03", "g04")


def item(unit, wording, medium, choice):
    return {"unit": unit, "wording": wording, "medium": medium, "choice": choice, "confidence": 0.5}


def test_accuracy_is_the_share_of_choices_that_are_the_medium():
    items = [item("g01", "q", "photograph", "photograph"), item("g02", "q", "pixel art", "comic book"),
             item("g03", "q", "watercolor", "watercolor"), item("g04", "q", "3D render", "3D render")]

    assert ab_text_classify.accuracy(items) == 0.75


def test_wording_lines_give_each_wording_its_accuracy_over_its_own_prompts():
    items = ([item(unit, "every", "oil painting", "oil painting") for unit in UNITS]
             + [item(unit, "half", "oil painting", choice)
                for unit, choice in zip(UNITS, ("oil painting", "watercolor", "oil painting", "watercolor"),
                                        strict=True)])

    lines = ab_text_classify.wording_lines(items, ("every", "half"), seed=0)

    assert lines[:3] == ["every", "  accuracy 1.0000 [1.0000, 1.0000] (4/4)", "half"]
    assert lines[3].startswith("  accuracy 0.5000 [")
    assert lines[3].endswith("(2/4)")
    assert lines[4].startswith("  minus 'every' -0.5000 [")


def test_cue_text_drops_the_leading_medium_phrase_and_keeps_the_cues():
    assert ab_text_classify.cue_text("Comic book panel of a hero, halftone shading", "comic book") == (
        "a hero, halftone shading")
    assert ab_text_classify.cue_text("Pixel art farm scene, a red barn, 16-bit", "pixel art") == "a red barn, 16-bit"
    assert ab_text_classify.cue_text("Classical oil portrait of a noblewoman", "oil painting") == "a noblewoman"


def test_cue_text_keeps_a_prompt_whose_head_names_no_medium_word():
    prompt = "A golden retriever running along a beach, clear sky"

    assert ab_text_classify.cue_text(prompt, "photograph") == prompt
    assert ab_text_classify.cue_text("no separator here", "watercolor") == "no separator here"


def test_help_runs_without_a_model():
    run = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False)

    assert run.returncode == 0, run.stderr
    assert "TEXT_CLASSIFY_QUESTION" in run.stdout
