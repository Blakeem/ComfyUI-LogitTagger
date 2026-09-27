"""tests-AB/ab_tagger_score.py per-image counts, paired deltas and the offline replay over small synthetic records."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_AB = Path(__file__).resolve().parent.parent / "tests-AB"
SCRIPT = TESTS_AB / "ab_tagger_score.py"

# The scorer imports its settings helpers from ab_tagger_compare.py beside it, as a script run finds them.
if str(TESTS_AB) not in sys.path:
    sys.path.append(str(TESTS_AB))

import ab_tagger_compare as compare  # noqa: E402
import ab_tagger_score as score  # noqa: E402

TagSettings = pytest.importorskip("logit_classifier.toolkit.comfyui").TagSettings

SUPERSET = TagSettings(propose_cap=4, merge_cap=6, thing_check="prompt", subsets_rule="head-noun", order="score")

# p("other") marks "sunset" as no thing. "curtain" is prompt only below the 0.9999 prompt_only_threshold.
CANDIDATES = [
    {"tag": "cat", "origin": "model", "score": 0.99, "p_other": 0.01},
    {"tag": "cat ear", "origin": "model", "score": 0.95, "p_other": 0.02},
    {"tag": "sunset", "origin": "model", "score": 0.97, "p_other": 0.95},
    {"tag": "lamp", "origin": "both", "score": 0.93, "p_other": 0.03},
    {"tag": "window", "origin": "prompt", "score": 0.99999, "p_other": 0.1},
    {"tag": "curtain", "origin": "prompt", "score": 0.9995, "p_other": 0.2},
]
SUPERSET_KEPT = ["window", "cat", "sunset", "cat ear", "lamp"]


def superset_record(**changes):
    """One image as a superset run records it: its reply, prompt tags, scored candidates and kept list."""
    record = {
        "tags": list(SUPERSET_KEPT),
        "reply": "cat, cat ear, sunset, lamp",
        "echo_reply": None,
        "proposed": ["cat", "cat ear", "sunset", "lamp"],
        "prompt_tags": ["lamp", "window", "curtain"],
        "candidates": [dict(candidate) for candidate in CANDIDATES],
    }
    record.update(changes)
    return record


def replayed(**overrides):
    return list(score.replay_image(superset_record(), score.covered_settings(SUPERSET, "arm", overrides)))


def test_the_superset_settings_reproduce_the_recorded_kept_list():
    assert list(score.replay_image(superset_record(), SUPERSET)) == SUPERSET_KEPT


def test_a_lower_propose_cap_cuts_the_reply_and_turns_a_both_tag_prompt_only():
    # "lamp" falls out of the model's first two tags, so it stays a prompt tag and fails 0.9999.
    assert replayed(propose_cap=2) == ["window", "cat", "cat ear"]


def test_a_lower_merge_cap_leaves_out_the_prompt_tags_past_it():
    assert replayed(merge_cap=4) == ["cat", "sunset", "cat ear", "lamp"]


def test_a_higher_verify_threshold_drops_the_model_tags_below_it():
    assert replayed(verify_threshold=0.96) == ["window", "cat", "sunset"]


def test_the_thing_check_on_every_origin_drops_a_model_tag_with_a_high_p_other():
    assert replayed(thing_check="all") == ["window", "cat", "cat ear", "lamp"]


@pytest.mark.parametrize(("rule", "kept"), [
    ("words", ["window", "sunset", "cat ear", "lamp"]),
    ("head-noun", SUPERSET_KEPT),
])
def test_each_subsets_rule_keeps_its_own_list(rule, kept):
    assert replayed(subsets_rule=rule) == kept


def test_the_proposed_order_keeps_candidate_order():
    assert replayed(order="proposed") == ["cat", "cat ear", "sunset", "lamp", "window"]


def test_the_cut_counts_repeated_and_overlong_items_as_the_decode_stop_does():
    reply = "cat, cat, dog, a tag with far too many words to be one, sofa"

    assert score.proposed_tags(reply, 3) == ["cat", "dog"]
    assert score.proposed_tags(reply, 4) == ["cat", "dog"]
    assert score.proposed_tags(reply, 5) == ["cat", "dog", "sofa"]


def test_a_candidate_without_a_recorded_score_leaves_the_image_out():
    record = superset_record(prompt_tags=["lamp", "window", "curtain", "rug"])
    settings = score.covered_settings(SUPERSET, "arm", {"propose_cap": 2})

    with pytest.raises(score.NotRecorded, match="rug"):
        score.replay_image(record, settings)
    replayed_records, notes = score.replay_arms({"render-prompt": {"g01": record}}, {"arm": settings})
    assert replayed_records == {"render-prompt": {"arm": {}}}
    assert notes == ["render-prompt arm leaves out g01: no recorded score for ['rug']"]


def test_an_arm_that_would_run_the_echo_fallback_the_superset_never_ran_leaves_the_image_out():
    settings = score.covered_settings(SUPERSET, "arm", {"category_nouns": ["cat", "cat ear"], "propose_cap": 2})

    with pytest.raises(score.NotRecorded, match="echo fallback"):
        score.replay_image(superset_record(), settings)


def test_the_thing_check_without_recorded_p_other_leaves_the_image_out():
    candidates = [{key: value for key, value in candidate.items() if key != "p_other"} for candidate in CANDIDATES]

    with pytest.raises(score.NotRecorded, match="--record-thing-scores"):
        score.replay_image(superset_record(candidates=candidates),
                           score.covered_settings(SUPERSET, "arm", {"thing_check": "all"}))


@pytest.mark.parametrize("field", ["propose_cap", "merge_cap"])
def test_a_cap_above_the_superset_is_rejected_naming_the_field(field):
    with pytest.raises(SystemExit, match=f"{field}=7 above the superset's"):
        score.covered_settings(SUPERSET, "wide", {field: 7, "merge_cap": 7})


@pytest.mark.parametrize(("field", "value"), [
    ("propose_instruction", "Tag it."),
    ("ungrounded_streak", 3),
    ("thing_threshold", 0.5),
    ("language_question", None),
    ("propose_max_tokens", 64),
])
def test_a_field_the_records_cannot_re_derive_is_rejected_naming_it(field, value):
    with pytest.raises(SystemExit, match=f"changes {field} from the superset's value"):
        score.covered_settings(SUPERSET, "other", {field: value})


def test_the_self_check_stops_at_the_first_image_whose_kept_list_differs():
    images = {"coco": {"a.jpg": superset_record(), "b.jpg": superset_record(tags=["cat"])}}

    with pytest.raises(SystemExit, match=r"self-check: coco b\.jpg replays to"):
        score.self_check(images, SUPERSET)


def test_the_self_check_passes_a_reproduced_run():
    score.self_check({"coco": {"a.jpg": superset_record()}}, SUPERSET)


def test_settings_round_trip_through_the_settings_file_json():
    written = json.loads(json.dumps(compare.settings_json(SUPERSET)))

    assert compare.arm_settings(TagSettings(), "S", written) == SUPERSET


VERDICTS = {"a": {"cat": "yes", "dog": "no", "sky": "vague"}, "b": {"cat": "yes", "tree": "unsure"}}


def rows_of(images):
    return {key: score.count_image(tags, VERDICTS.get(key, {})) for key, tags in images.items()}


def test_per_image_counts_are_means_over_the_arm_images():
    rows = list(rows_of({"a": ["cat", "dog", "sky", "sun"], "b": ["cat", "tree"]}).values())

    assert score.metric_values(rows, score.ARM_METRICS) == {
        "precision": 0.5, "correct": 1.0, "wrong": 0.5, "vague": 0.5, "unsure": 0.5, "unjudged": 0.5}


def test_paired_deltas_pair_the_images_both_arms_hold():
    arm = rows_of({"a": ["cat", "dog", "sky"], "b": ["cat", "tree"], "c": ["dog"]})
    base = rows_of({"a": ["cat"], "b": ["cat", "tree"]})

    rows, base_rows = score.paired_rows(arm, base)

    assert len(rows) == 2
    assert score.paired_deltas(rows, base_rows, score.DELTA_METRICS) == {
        "precision": -0.5, "correct": 0.0, "wrong": 0.5, "wrong+vague": 1.0}


def test_the_paired_interval_of_identical_images_is_the_delta_itself():
    rows = [score.count_image(["cat", "dog"], VERDICTS["a"])] * 3
    base_rows = [score.count_image(["cat"], VERDICTS["a"])] * 3

    ci = score.bootstrap([rows, base_rows], score.DELTA_METRICS, 20, random.Random(0))

    assert ci["wrong"] == (1.0, 1.0)
    assert ci["correct"] == (0.0, 0.0)


def test_record_thing_scores_writes_each_candidate_p_other_from_the_cache():
    records = [superset_record(), superset_record()]
    known = {candidate["tag"]: candidate["p_other"] + 0.001 for candidate in CANDIDATES}

    # Every tag is known, so thing_scores asks nothing and needs no classifier.
    compare.record_thing_scores(None, records, SUPERSET, known)

    assert [candidate["p_other"] for candidate in records[1]["candidates"]] == [
        candidate["p_other"] + 0.001 for candidate in CANDIDATES]


def test_the_thing_cache_of_another_wording_is_refused(tmp_path):
    cache = tmp_path / "thing_scores_clip.json"
    cache.write_text(json.dumps({"thing_question": "Is {TAG} a thing", "thing_criteria": [], "p_other": {}}))

    with pytest.raises(SystemExit, match="another thing check wording"):
        compare.load_thing_cache(cache, [SUPERSET])


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_replay_prints_each_replayed_arm_paired_with_the_superset(tmp_path):
    results = write(tmp_path / "results_s.json", {"coco": {"S": {"a": superset_record(), "b": superset_record()}}})
    write(tmp_path / "settings_s.json", {"S": compare.settings_json(SUPERSET)})
    arms = write(tmp_path / "arms.json", {"T96": {"verify_threshold": 0.96}})
    verdicts = write(tmp_path / "verdicts.json", {key: {"cat": "yes", "cat ear": "yes", "lamp": "no",
                                                        "sunset": "vague", "window": "yes"} for key in "ab"})

    run = subprocess.run([sys.executable, str(SCRIPT), str(verdicts), "--replay", str(arms), "--results",
                          str(results), "--samples", "20"], capture_output=True, text=True, check=False)

    assert run.returncode == 0, run.stderr
    assert "T96 minus S, 2 paired images: precision +0.067" in run.stdout
    assert "wrong/img -1.00 [-1.00, -1.00]" in run.stdout


def test_help_shows_replay():
    run = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False)

    assert run.returncode == 0, run.stderr
    assert "--replay ARMS.json" in run.stdout
