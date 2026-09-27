"""tests-AB/ab_tagger_compare.py stops on an image whose bytes differ from the ones its verdicts were judged on."""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tests-AB" / "ab_tagger_compare.py"


def load_script():
    spec = importlib.util.spec_from_file_location("ab_tagger_compare_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclass resolves the module's string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = load_script()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    """A render manifest, a coco60.json and a coco30.json whose hashes match the files on disk."""
    renders = tmp_path / "renders"
    coco = tmp_path / "coco60"
    renders.mkdir()
    (coco / "coco").mkdir(parents=True)
    (renders / "g01.png").write_bytes(b"render")
    (coco / "coco" / "1.jpg").write_bytes(b"photo")
    manifest = renders / "manifest.json"
    manifest.write_text(json.dumps([{"id": "g01", "file": "g01.png", "sha256": sha256(b"render"), "prompt": "p"}]))
    (coco / "coco60.json").write_text(json.dumps({"images": [{"file": "coco/1.jpg"}]}))
    hashes = tmp_path / "coco30.json"
    hashes.write_text(json.dumps([{"file": "coco/1.jpg", "sha256": sha256(b"photo")}]))
    return manifest, coco / "coco60.json", hashes


def test_matching_images_load_every_condition(corpus):
    conditions = compare.load_conditions(*corpus, coco_count=30, limit=None)

    assert [item.key for item in conditions["render-prompt"]] == ["g01"]
    assert [item.key for item in conditions["coco"]] == ["coco/1.jpg"]


@pytest.mark.parametrize("changed", ["renders/g01.png", "coco60/coco/1.jpg"])
def test_an_image_whose_bytes_changed_stops_the_run_naming_it(corpus, tmp_path, changed):
    (tmp_path / changed).write_bytes(b"re-rendered")

    with pytest.raises(SystemExit, match=re.escape(f"{Path(changed).name} differs from its sha256")):
        compare.load_conditions(*corpus, coco_count=30, limit=None)


def test_an_image_without_a_hash_entry_stops_the_run_naming_it(corpus):
    manifest, coco, hashes = corpus
    hashes.write_text("[]")

    with pytest.raises(SystemExit, match=r"1\.jpg has no sha256"):
        compare.load_conditions(manifest, coco, hashes, coco_count=30, limit=None)
