"""tests-AB/ab_judge.py split and merge over small verdict stores and run records in tmp_path."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tests-AB" / "ab_judge.py"


def load_script():
    spec = importlib.util.spec_from_file_location("ab_judge_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclass resolves the module's string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


judge = load_script()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(tmp_path, name, records):
    """A folder holding one results.json, {condition: {arm: {key: record}}}."""
    return write(tmp_path / name / "results.json", records).parent


@pytest.fixture
def images(tmp_path):
    paths = {key: tmp_path / "images" / f"{key}.png" for key in ("g01", "g02", "g03")}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    return paths


@pytest.fixture
def store(tmp_path):
    return write(tmp_path / "verdicts.json", {
        "g01": {"cat": "yes", "dog": "no", "sky": "vague"},
        "g02": {"tree": "yes"},
    })


@pytest.fixture
def runs(tmp_path):
    first = run_dir(tmp_path, "first", {
        "render-noprompt": {"new": {"g01": {"tags": ["cat", "dog", "hat"]}, "g02": {"tags": ["tree"]}}},
    })
    second = run_dir(tmp_path, "second", {
        "render-prompt": {"N1": {"g01": {"tags": ["cat", "sky", "boat"]}, "g03": {"tags": ["a", "b", "c", "d"]}}},
    })
    return [first, second]


def test_split_batches_the_ungraded_kept_tags_of_every_run_balanced_by_tag_count(tmp_path, images, store, runs):
    out = tmp_path / "round"

    loads = judge.split(runs, store, out, batches=2, calibration=0, images=images)

    assert [read(out / f"judge_in_{number}.json") for number in (1, 2)] == loads
    assert not (out / "judge_in_3.json").exists()
    assert sorted((entry["key"], entry["tags"]) for batch in loads for entry in batch) == [
        ("g01", ["boat", "hat"]),
        ("g03", ["a", "b", "c", "d"]),
    ]
    assert sorted(judge.tag_count(batch) for batch in loads) == [2, 4]
    assert all(entry["image"] == str(images[entry["key"]]) for batch in loads for entry in batch)
    assert read(out / "calibration.json") == {}
    assert (out / "RUBRIC.md").read_text(encoding="utf-8") == judge.RUBRIC.read_text(encoding="utf-8")


def test_split_mixes_k_graded_kept_tags_into_each_image_and_records_their_verdicts(tmp_path, images, store, runs):
    out = tmp_path / "round"

    loads = judge.split(runs, store, out, batches=1, calibration=2, images=images)
    entries = {entry["key"]: entry["tags"] for entry in loads[0]}
    calibration = read(out / "calibration.json")

    # g01 keeps cat, dog and sky graded, so two of them join its new tags. g02 has no new tag, so no entry.
    assert set(entries) == {"g01", "g03"}
    assert set(calibration) == {"g01"}
    assert len(calibration["g01"]) == 2
    assert set(calibration["g01"]) <= {"cat", "dog", "sky"}
    assert calibration["g01"] == {tag: read(store)["g01"][tag] for tag in calibration["g01"]}
    assert entries["g01"] == sorted({"boat", "hat", *calibration["g01"]})
    assert judge.split(runs, store, tmp_path / "again", batches=1, calibration=2, images=images) == loads


def test_split_never_writes_more_batches_than_images(tmp_path, images, store, runs):
    loads = judge.split(runs, store, tmp_path / "round", batches=5, calibration=0, images=images)

    assert len(loads) == 2
    assert all(loads)


def test_split_writes_nothing_when_every_kept_tag_is_graded(tmp_path, images, store):
    graded = run_dir(tmp_path, "graded", {"coco": {"new": {"g02": {"tags": ["tree"]}}}})
    out = tmp_path / "round"

    assert judge.split([graded], store, out, batches=2, calibration=1, images=images) == []
    assert not out.exists()


def test_split_refuses_a_folder_that_already_holds_a_round(tmp_path, images, store, runs):
    out = tmp_path / "round"
    write(out / "verdicts_1.json", {})

    with pytest.raises(SystemExit, match=r"already holds verdicts_1\.json"):
        judge.split(runs, store, out, batches=1, calibration=0, images=images)


def test_split_names_an_image_no_manifest_lists(tmp_path, images, store, runs):
    del images["g03"]

    with pytest.raises(SystemExit, match="g03"):
        judge.split(runs, store, tmp_path / "round", batches=1, calibration=0, images=images)


def test_split_names_a_folder_without_run_records(tmp_path, images, store):
    (tmp_path / "empty").mkdir()

    with pytest.raises(SystemExit, match="holds no results"):
        judge.split([tmp_path / "empty"], store, tmp_path / "round", batches=1, calibration=0, images=images)


def judged_round(tmp_path, batches, calibration, grades):
    """A split --out folder holding each batch's input and its judge's verdicts."""
    out = tmp_path / "round"
    for number, (entries, verdicts) in enumerate(zip(batches, grades, strict=True), 1):
        write(out / f"judge_in_{number}.json", entries)
        write(out / f"verdicts_{number}.json", verdicts)
    write(out / "calibration.json", calibration)
    return out


def test_merge_adds_new_pairs_keeps_the_first_verdict_and_reports_calibration(tmp_path, store):
    batches = [
        [{"image": "g01.png", "key": "g01", "tags": ["boat", "cat", "dog"]}],
        [{"image": "g03.png", "key": "g03", "tags": ["a", "tree"]}, {"image": "g02.png", "key": "g02", "tags": ["x"]}],
    ]
    calibration = {"g01": {"cat": "yes", "dog": "no"}}
    grades = [
        {"g01": {"boat": "yes", "cat": "yes", "dog": "yes"}},
        {"g03": {"a": "vague", "tree": "no"}, "g02": {"x": "unsure"}},
    ]
    out = judged_round(tmp_path, batches, calibration, grades)
    write(store, {**read(store), "g03": {"tree": "yes"}})

    report = judge.merge(out, store)

    assert report.added == 3
    assert report.total == 8
    assert report.conflicts == [("g03", "tree", "yes", "no")]
    assert sorted(report.calibration) == [("no", "yes"), ("yes", "yes")]
    assert read(store) == {
        "g01": {"boat": "yes", "cat": "yes", "dog": "no", "sky": "vague"},
        "g02": {"tree": "yes", "x": "unsure"},
        "g03": {"a": "vague", "tree": "yes"},
    }
    assert list(read(store)) == ["g01", "g02", "g03"]


def test_calibration_line_prints_count_agreement_and_both_precisions():
    line = judge.calibration_line([("yes", "yes"), ("no", "yes"), ("vague", "vague"), ("yes", "unsure")])

    assert line == ("calibration: 4 pairs, agreement 0.500, precision 0.500 under the stored grades and 0.667 "
                    "under the new grades")
    assert judge.calibration_line([]) == "calibration: no pairs"


@pytest.mark.parametrize(("verdicts", "message"), [
    ({"g01": {"boat": "yes"}}, "1 missing and 0 unasked"),
    ({"g01": {"boat": "yes", "hat": "yes", "cap": "no"}}, "0 missing and 1 unasked"),
    ({"g01": {"boat": "yes", "hat": "maybe"}}, "grades outside"),
])
def test_merge_rejects_a_batch_that_differs_from_its_input_and_leaves_the_store_unchanged(tmp_path, store, verdicts,
                                                                                          message):
    before = store.read_text(encoding="utf-8")
    out = judged_round(tmp_path, [[{"image": "g01.png", "key": "g01", "tags": ["boat", "hat"]}]], {}, [verdicts])

    with pytest.raises(SystemExit, match=rf"verdicts_1\.json.*{message}"):
        judge.merge(out, store)
    assert store.read_text(encoding="utf-8") == before


def test_merge_names_a_missing_verdicts_file(tmp_path, store):
    out = judged_round(tmp_path, [[{"image": "g01.png", "key": "g01", "tags": ["boat"]}]], {}, [{"g01": {"boat": "no"}}])
    write(out / "judge_in_2.json", [{"image": "g02.png", "key": "g02", "tags": ["x"]}])

    with pytest.raises(SystemExit, match=r"verdicts_2\.json is missing"):
        judge.merge(out, store)


def test_merge_rejects_a_calibration_pair_no_batch_holds(tmp_path, store):
    out = judged_round(tmp_path, [[{"image": "g01.png", "key": "g01", "tags": ["boat"]}]], {"g01": {"cat": "yes"}},
                       [{"g01": {"boat": "no"}}])

    with pytest.raises(SystemExit, match=r"calibration\.json lists pairs no batch holds"):
        judge.merge(out, store)


def test_split_then_merge_round_trips_every_ungraded_pair_into_the_store(tmp_path, images, store, runs):
    out = tmp_path / "round"
    loads = judge.split(runs, store, out, batches=2, calibration=1, images=images)
    for number, batch in enumerate(loads, 1):
        write(out / f"verdicts_{number}.json", {entry["key"]: dict.fromkeys(entry["tags"], "yes") for entry in batch})

    report = judge.merge(out, store)

    assert report.added == 6
    assert report.total == 10
    assert read(store)["g03"] == dict.fromkeys(["a", "b", "c", "d"], "yes")
    assert len(report.calibration) == 1


def compare_output(folder, suffix=""):
    """The three files ab_tagger_compare.py writes for one run of arms S and T."""
    record = {"tags": ["a"], "reply": "a, b", "echo_reply": None, "prompt_tags": [], "proposed": ["a", "b"],
              "candidates": [{"tag": "a", "origin": "model", "score": 0.99, "p_thing": 0.9}], "dropped": [],
              "prompt_ms": 0.0, "propose_ms": 1.0, "verify_ms": 2.0, "total_ms": 3.0,
              "propose_decode_ms_per_token": 30.0, "size_in": [8, 8], "propose_tokens": 4}
    write(folder / f"settings{suffix}.json", {"S": {"propose_cap": 40}, "T": {"propose_cap": 25}})
    write(folder / f"run_meta{suffix}.json", {"runs": [{"arms": ["S", "T"], "variants": {"S": {"propose_cap": 40},
                                                                                          "T": {}}}]})
    return write(folder / f"results{suffix}.json", {"coco": {"S": {"g01": record}, "T": {"g01": record}}})


def test_store_keeps_the_scored_fields_the_settings_and_the_arm_overrides(tmp_path):
    results = compare_output(tmp_path / "out", "_r2")

    folder = judge.store(results, "exp", tmp_path / "runs")

    stored = read(folder / "results.json")["coco"]["S"]["g01"]
    assert set(stored) == set(judge.STORED_FIELDS)
    assert read(folder / "settings.json") == read(tmp_path / "out" / "settings_r2.json")
    assert read(folder / "arms.json") == {"S": {"propose_cap": 40}, "T": {}}
    assert read(folder / "run_meta.json")["runs"][0]["arms"] == ["S", "T"]


def test_store_never_overwrites_a_stored_run(tmp_path):
    results = compare_output(tmp_path / "out")
    judge.store(results, "exp", tmp_path / "runs")

    with pytest.raises(SystemExit, match="already exists"):
        judge.store(results, "exp", tmp_path / "runs")


def test_store_names_the_files_a_results_file_lacks(tmp_path):
    results = write(tmp_path / "out" / "results.json", {})

    with pytest.raises(SystemExit, match=r"settings\.json, run_meta\.json"):
        judge.store(results, "exp", tmp_path / "runs")


def test_stored_records_split_like_the_run_they_came_from(tmp_path):
    folder = judge.store(compare_output(tmp_path / "out"), "exp", tmp_path / "runs")

    assert judge.kept_tags([folder]) == {"g01": {"a"}}


def test_help_lists_every_command():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)

    assert "store" in result.stdout
    assert "split" in result.stdout
    assert "merge" in result.stdout
