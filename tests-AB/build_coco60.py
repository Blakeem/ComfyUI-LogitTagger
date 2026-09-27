"""Build the 60 image COCO val2017 corpus that tests-AB/ab_question_shapes.py reads.

Downloads the 2017 annotations once, samples 60 images holding two or more categories (seed 7),
and labels each category present when it covers at least 1% of the image and ambiguous below that.
Usage: python tests-AB/build_coco60.py [--out DIR]
"""

import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_FILES = ("annotations/instances_val2017.json", "annotations/captions_val2017.json")
DEFAULT_OUT = Path(__file__).resolve().parent / "cache" / "coco60"
IMAGE_COUNT = 60
SEED = 7
# Below this share of the image a labeled object is too small for a yes or no to be fair.
PRESENT_AREA = 0.01


def fetch_annotations(out: Path) -> None:
    archive = out / "annotations_trainval2017.zip"

    if all((out / name).is_file() for name in ANNOTATION_FILES):
        return
    if not archive.is_file():
        urllib.request.urlretrieve(ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        for name in ANNOTATION_FILES:
            bundle.extract(name, out)


def build(out: Path) -> dict:
    instances = json.loads((out / ANNOTATION_FILES[0]).read_text(encoding="utf-8"))
    captions = json.loads((out / ANNOTATION_FILES[1]).read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in instances["categories"]}
    images = {image["id"]: image for image in instances["images"]}
    by_image: dict[int, list] = {}
    texts: dict[int, list] = {}
    records = []

    for annotation in instances["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    for caption in captions["annotations"]:
        texts.setdefault(caption["image_id"], []).append(caption["caption"].strip())
    candidates = [key for key, found in by_image.items() if len({a["category_id"] for a in found}) >= 2]
    random.Random(SEED).shuffle(candidates)
    (out / "coco").mkdir(exist_ok=True)
    for image_id in candidates[:IMAGE_COUNT]:
        image = images[image_id]
        area = image["width"] * image["height"]
        shares: dict[str, float] = {}
        for annotation in by_image[image_id]:
            name = names[annotation["category_id"]]
            shares[name] = max(shares.get(name, 0.0), annotation["area"] / area)
        path = out / "coco" / image["file_name"]
        if not path.is_file():
            urllib.request.urlretrieve(image["coco_url"], path)
        records.append({
            "file": f"coco/{image['file_name']}",
            "present": sorted(name for name, share in shares.items() if share >= PRESENT_AREA),
            "ambiguous": sorted(name for name, share in shares.items() if share < PRESENT_AREA),
            "captions": texts.get(image_id, []),
        })
    return {"categories": sorted(names.values()), "images": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"corpus folder, default {DEFAULT_OUT}")
    out = parser.parse_args().out

    out.mkdir(parents=True, exist_ok=True)
    fetch_annotations(out)
    corpus = build(out)
    (out / "coco60.json").write_text(json.dumps(corpus, indent=1), encoding="utf-8")
    print(f"{len(corpus['images'])} images written to {out / 'coco60.json'}")


if __name__ == "__main__":
    main()
