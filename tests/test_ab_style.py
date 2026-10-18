"""tests-AB/ab_style.py medium word match, report lines and run folder over small synthetic records."""

import subprocess
import sys
from pathlib import Path

import pytest

TESTS_AB = Path(__file__).resolve().parent.parent / "tests-AB"
SCRIPT = TESTS_AB / "ab_style.py"

# The harness imports its loaders from the harnesses beside it, as a script run finds them.
if str(TESTS_AB) not in sys.path:
    sys.path.append(str(TESTS_AB))

# The release-gates loaders it imports read the library's tag helpers at module scope.
pytest.importorskip("logit_classifier.toolkit.tags")

import ab_style  # noqa: E402


def test_a_text_names_a_medium_when_it_holds_every_word_of_the_medium_name():
    assert ab_style.names_medium("glossy 3d render, soft studio light", "3D render")
    assert ab_style.names_medium("loose watercolors on rough paper", "watercolor")
    assert not ab_style.names_medium("oil on canvas, impasto", "oil painting")
    assert not ab_style.names_medium("", "photograph")


def test_prompt_lines_score_the_render_styles_and_count_the_textset_styles():
    records = [
        {"set": "render", "id": "g01", "medium": "pixel art", "style": ["pixel art", "16-bit"]},
        {"set": "render", "id": "g02", "medium": "oil painting", "style": ["impasto"]},
        {"set": "textset", "id": "t01", "medium": None, "style": ["anime"]},
        {"set": "textset", "id": "t02", "medium": None, "style": []},
        {"set": "textset", "id": "t03", "medium": None, "style": []},
    ]

    lines = ab_style.prompt_lines(records, seed=0)

    assert lines[0].startswith("render prompts whose kept style names their medium 0.5000 [")
    assert lines[0].endswith("(1/2)")
    assert lines[1] == "textset prompts with a style 1/3"


def test_image_lines_score_the_render_lines_and_time_each_set():
    records = [
        {"set": "render", "id": "g01", "medium": "pencil sketch", "line": "A loose pencil sketch.", "ms": 100.0},
        {"set": "render", "id": "g02", "medium": "comic book", "line": "Comic book inks.", "ms": 300.0},
        {"set": "coco", "id": "coco/1.jpg", "medium": None, "line": "A candid photograph.", "ms": 50.0},
    ]

    lines = ab_style.image_lines(records, seed=0)

    assert lines[0] == "renders whose style line names their medium 1.0000 [1.0000, 1.0000] (2/2), mean 200 ms"
    assert lines[1] == "COCO photos 1, mean 50 ms"


def test_variants_lead_with_default_and_refuse_a_field_outside_their_stage(tmp_path):
    good = tmp_path / "good.json"
    good.write_text('{"listings": {"short": {"prompt_max_tokens": 64}}, '
                    '"scorings": {"q": {"style_question": "Is \\"{TAG}\\" a style"}}}', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text('{"listings": {"x": {"style_threshold": 0.1}}}', encoding="utf-8")

    listings, scorings = ab_style.load_variants(good)

    assert list(listings) == ["default", "short"] and listings["default"] == {}
    assert list(scorings) == ["default", "q"]
    assert ab_style.load_variants(None) == ({"default": {}}, {"default": {}})
    with pytest.raises(SystemExit, match="listing 'x' sets \\['style_threshold'\\]"):
        ab_style.load_variants(bad)


def test_arm_settings_turn_json_criteria_into_pairs():
    from logit_classifier.toolkit.comfyui import StyleSettings

    settings = ab_style.arm_settings(StyleSettings(), {"style_criteria": [["style", "a medium"], ["other", "a cat"]]})

    assert settings.style_criteria == (("style", "a medium"), ("other", "a cat"))


def record(unit, set_name, listed, p, medium=None, listing="default"):
    return {"id": unit, "set": set_name, "medium": medium, "prompt": f"prompt {unit}", "listing": listing,
            "listed": listed, "p_style": {"default": p}}


def test_kept_style_keeps_phrases_at_or_above_the_threshold():
    kept = ab_style.kept_style(record("g01", "render", ["oil painting", "harbor"], [0.5, 0.49]), "default", 0.5)

    assert kept == ["oil painting"]


def test_judge_entries_merge_listings_and_hide_the_arm():
    results = {"path": "prompt", "records": [
        record("t02", "textset", ["cat"], [0.1]),
        record("g01", "render", ["watercolor", "canal"], [0.9, 0.1]),
        record("g01", "render", ["canal", "gondola"], [0.1, 0.1], listing="other"),
        record("t03", "textset", [], []),
    ]}

    entries = ab_style.judge_entries(results)

    assert entries == [{"key": "g01", "prompt": "prompt g01", "phrases": ["canal", "gondola", "watercolor"]},
                       {"key": "t02", "prompt": "prompt t02", "phrases": ["cat"]}]
    images = {"path": "image", "records": [{"id": "g01", "line": "Watercolor.", "medium": "watercolor"},
                                           {"id": "coco/1.jpg", "line": "Photographic.", "medium": None}]}
    assert ab_style.judge_entries(images) == [
        {"key": "g01", "line": "Watercolor.", "medium": "watercolor"},
        {"key": "coco/1.jpg", "line": "Photographic.", "medium": "photograph"}]


def test_merge_labels_keeps_agreed_grades_and_marks_the_rest_unsure():
    entries = [{"key": "g01", "phrases": ["watercolor", "canal"]},
               {"key": "g02", "line": "Pixel art.", "medium": "pixel art"}]
    judges = {"judge-1.json": {"g01": {"watercolor": "yes", "canal": "no"}, "g02": {"line": "yes", "medium": "yes"}},
              "judge-2.json": {"g01": {"watercolor": "yes", "canal": "yes"}, "g02": {"line": "yes", "medium": "no"}}}

    labels, agreed, total = ab_style.merge_labels(entries, judges)

    assert labels == {"g01": {"watercolor": "yes", "canal": "unsure"}, "g02": {"line": "yes", "medium": "unsure"}}
    assert (agreed, total) == (2, 4)
    with pytest.raises(SystemExit, match="needs two or more"):
        ab_style.merge_labels(entries, {"judge-1.json": judges["judge-1.json"]})
    with pytest.raises(SystemExit, match="gave no valid grade for g01 'canal'"):
        ab_style.merge_labels(entries, {**judges, "judge-3.json": {"g01": {"watercolor": "yes"}}})


def test_replay_lines_count_yes_and_no_phrases_per_prompt_at_each_threshold():
    records = [record("g01", "render", ["watercolor", "canal"], [0.9, 0.3], medium="watercolor"),
               record("t01", "textset", ["neon"], [0.6])]
    labels = {"g01": {"watercolor": "yes", "canal": "no"}, "t01": {"neon": "no"}}

    lines = ab_style.replay_lines(records, labels, ["default"], (0.2, 0.5), seed=0)

    assert lines[1].startswith("default / default / 0.2: 1.0000 [")
    assert ", 1.00/1.00, 0.00/1.00, 0.333, " in lines[1]
    assert ", 1.00/0.00, 0.00/1.00, 0.500, " in lines[2]


def test_image_replay_lines_share_the_medium_and_content_grades_per_set():
    records = [{"id": "g01", "set": "render"}, {"id": "g02", "set": "render"}, {"id": "coco/1.jpg", "set": "coco"}]
    labels = {"g01": {"line": "yes", "medium": "yes"}, "g02": {"line": "no", "medium": "yes"},
              "coco/1.jpg": {"line": "yes", "medium": "yes"}}

    lines = ab_style.image_replay_lines(records, labels, seed=0)

    assert lines[0].startswith("render: lines naming the medium 1.0000 [1.0000, 1.0000], naming content 0.5000 [")
    assert lines[0].endswith("(2 lines)")
    assert lines[1].startswith("coco: lines naming the medium 1.0000")


def test_a_run_folder_that_holds_results_is_refused(tmp_path):
    taken = tmp_path / "2026-10-17-prompt" / "results.json"
    taken.parent.mkdir()
    taken.write_text("{}", encoding="utf-8")

    assert ab_style.results_path(tmp_path, "fresh") == tmp_path / "fresh" / "results.json"
    with pytest.raises(SystemExit, match="already holds a run"):
        ab_style.results_path(tmp_path, "2026-10-17-prompt")


def test_help_runs_without_a_model():
    run = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False)

    assert run.returncode == 0, run.stderr
    assert "--path {prompt,image}" in run.stdout
