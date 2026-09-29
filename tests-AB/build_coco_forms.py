"""Build the COCO val2017 corpus that tests-AB/ab_presence_wording.py reads: one tag per grammatical form.

Each image holds two or more thing categories and one of them at least twice. It is asked about:
  singular   a thing category by its COCO name ("dog"), present or absent
  plural     its plural ("dogs"), present when 2 or more instances are counted, or absent
  irregular  the plural of person, mouse, knife and sheep ("people", "mice"), present or absent
  count      "two dogs" when exactly 2 or 3 are counted, or a count of an absent category
  wrong count  the count one above the true count ("three dogs" for 2), scored apart from the pooled rows
  mass       a panoptic stuff class read as a mass noun ("sand", "grass", "sky"), present or absent
  stuff plural  a panoptic stuff class read as a plural ("trees", "mountains"), present or absent
  related    an absent category of a present category's supercategory ("cat" beside a dog), singular
A thing is present when one instance covers at least 1% of the image and ambiguous below that. An instance
counts toward the number when it covers at least 0.1%, and a crowd annotation makes the number unknown.
A stuff class is present at 2% or more, absent with no segment, and ambiguous between. Ambiguous tags are
never asked. The panoptic JSON is read out of COCO's zip with HTTP range requests, so the 860 MB zip is
never downloaded.
Needs cache/coco60/coco60.json, which tests-AB/build_coco60.py writes.
Usage: python tests-AB/build_coco_forms.py [--out DIR] [--images N]
"""

import argparse
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from build_coco60 import ANNOTATION_FILES, fetch_annotations

HERE = Path(__file__).resolve().parent
COCO60 = HERE / "cache" / "coco60"
DEFAULT_OUT = HERE / "cache" / "coco-forms"
PANOPTIC_ZIP = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
PANOPTIC_JSON = "annotations/panoptic_val2017.json"
IMAGE_COUNT = 160
SEED = 11

PRESENT_AREA = 0.01
COUNTED_AREA = 0.001
STUFF_PRESENT_AREA = 0.02
ABSENT_THINGS = 3
ABSENT_MASS = 3
ABSENT_STUFF_PLURAL = 2
RELATED_ABSENT = 3

NUMBER_WORDS = {2: "two", 3: "three", 4: "four"}
# COCO names whose plural a trailing "s" does not spell.
PLURALS = {
    "person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "bus": "buses",
    "bench": "benches", "couch": "couches", "sandwich": "sandwiches", "toothbrush": "toothbrushes",
    "wine glass": "wine glasses", "broccoli": "broccoli", "skis": "skis", "scissors": "scissors",
    "hair drier": "hair driers",
}
IRREGULAR = frozenset({"person", "mouse", "knife", "sheep"})
# Plural-only names have no singular tag.
PLURAL_ONLY = frozenset({"skis", "scissors"})
# broccoli is a mass noun, so it has no count form.
UNCOUNTED = frozenset({"broccoli"})
MASS_STUFF = {
    "sand": "sand", "snow": "snow", "gravel": "gravel", "grass-merged": "grass", "dirt-merged": "dirt",
    "pavement-merged": "pavement", "water-other": "water", "sea": "water", "river": "water",
    "sky-other-merged": "sky",
}
PLURAL_STUFF = {
    "tree-merged": "trees", "mountain-merged": "mountains", "flower": "flowers", "rock-merged": "rocks",
    "curtain": "curtains", "pillow": "pillows",
}


class RangeFile(io.RawIOBase):
    """A read-only seekable view of a remote file, one HTTP range request per read."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.position = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def readinto(self, buffer) -> int:
        end = min(self.size, self.position + len(buffer)) - 1
        data = b""

        if end < self.position:
            return 0
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


def fetch_panoptic(out: Path) -> Path:
    target = out / PANOPTIC_JSON

    if not target.is_file():
        with zipfile.ZipFile(RangeFile(PANOPTIC_ZIP)) as bundle:
            bundle.extract(PANOPTIC_JSON, out)
    return target


def plural(name: str) -> str:
    return PLURALS.get(name, f"{name}s")


def thing_counts(annotations: list[dict], names: dict[int, str], area: float) -> dict[str, dict]:
    """Per category: the largest instance's share of the image, the counted instances, and whether a crowd hides the number."""
    things: dict[str, dict] = {}

    for annotation in annotations:
        name = names[annotation["category_id"]]
        share = annotation["area"] / area
        thing = things.setdefault(name, {"share": 0.0, "count": 0, "crowd": False})
        thing["share"] = max(thing["share"], share)
        if annotation.get("iscrowd"):
            thing["crowd"] = True
        elif share >= COUNTED_AREA:
            thing["count"] += 1
    return things


def stuff_shares(segments: list[dict], names: dict[int, str], area: float) -> dict[str, float]:
    """Per tag, the summed share of the image its panoptic stuff segments cover."""
    shares: dict[str, float] = {}

    for segment in segments:
        name = names[segment["category_id"]]
        tag = MASS_STUFF.get(name) or PLURAL_STUFF.get(name)
        if tag:
            shares[tag] = shares.get(tag, 0.0) + segment["area"] / area
    return shares


def item(text: str, form: str, truth: bool | None, source: str, **extra) -> dict:
    return {"text": text, "form": form, "truth": truth, "source": source, **extra}


def thing_items(things: dict[str, dict], categories: list[str], rng: random.Random) -> list[dict]:
    """Present categories in every form their number allows, then absent categories in every form."""
    items: list[dict] = []
    present = sorted(name for name, thing in things.items() if thing["share"] >= PRESENT_AREA)
    absent_pool = [name for name in categories if name not in things]
    absent_irregular = sorted(name for name in IRREGULAR if name in absent_pool and name != "person")
    absent = rng.sample([name for name in absent_pool if name not in IRREGULAR], ABSENT_THINGS)
    absent += ["person"] if "person" in absent_pool else []
    absent += rng.sample(absent_irregular, min(1, len(absent_irregular)))

    for name in present:
        thing = things[name]
        many = thing["crowd"] or thing["count"] >= 2
        form = "irregular" if name in IRREGULAR else "plural"
        if name not in PLURAL_ONLY:
            items.append(item(name, "singular", True, "thing", category=name, count=thing["count"]))
        if many:
            items.append(item(plural(name), form, True, "thing", category=name, count=thing["count"]))
        if not thing["crowd"] and thing["count"] in (2, 3) and name not in UNCOUNTED:
            items.append(item(f"{NUMBER_WORDS[thing['count']]} {plural(name)}", "count", True, "thing",
                              category=name, count=thing["count"]))
            items.append(item(f"{NUMBER_WORDS[thing['count'] + 1]} {plural(name)}", "wrong count", False, "thing",
                              category=name, count=thing["count"]))
    for name in absent:
        form = "irregular" if name in IRREGULAR else "plural"
        if name not in PLURAL_ONLY:
            items.append(item(name, "singular", False, "thing", category=name, count=0))
        items.append(item(plural(name), form, False, "thing", category=name, count=0))
    counted = [name for name in absent if name not in UNCOUNTED]
    items.append(item(f"two {plural(counted[0])}", "count", False, "thing", category=counted[0], count=0))
    return items


def stuff_items(shares: dict[str, float], rng: random.Random) -> list[dict]:
    items: list[dict] = []

    for tags, form, absent_count in ((sorted(set(MASS_STUFF.values())), "mass", ABSENT_MASS),
                                     (sorted(PLURAL_STUFF.values()), "stuff plural", ABSENT_STUFF_PLURAL)):
        present = [tag for tag in tags if shares.get(tag, 0.0) >= STUFF_PRESENT_AREA]
        absent = [tag for tag in tags if tag not in shares]
        items += [item(tag, form, True, "stuff", share=round(shares[tag], 4)) for tag in present]
        items += [item(tag, form, False, "stuff", share=0.0)
                  for tag in rng.sample(absent, min(absent_count, len(absent)))]
    return items


def related_items(things: dict[str, dict], supercategories: dict[str, str], asked: set[str],
                  rng: random.Random) -> list[dict]:
    """Absent categories of a present category's supercategory ("cat" beside a dog), the hardest absent tags."""
    present_groups = {supercategories[name] for name, thing in things.items() if thing["share"] >= PRESENT_AREA}
    pool = sorted(name for name, group in supercategories.items()
                  if group in present_groups and name not in things and name not in asked
                  and name not in PLURAL_ONLY)

    return [item(name, "related", False, "thing", category=name, count=0)
            for name in rng.sample(pool, min(RELATED_ABSENT, len(pool)))]


def build(out: Path, image_count: int) -> dict:
    instances = json.loads((COCO60 / ANNOTATION_FILES[0]).read_text(encoding="utf-8"))
    panoptic = json.loads(fetch_panoptic(out).read_text(encoding="utf-8"))
    coco60 = {Path(record["file"]).name for record in
              json.loads((COCO60 / "coco60.json").read_text(encoding="utf-8"))["images"]}
    names = {category["id"]: category["name"] for category in instances["categories"]}
    supercategories = {category["name"]: category["supercategory"] for category in instances["categories"]}
    stuff_names = {category["id"]: category["name"] for category in panoptic["categories"]}
    categories = sorted(names.values())
    images = {image["id"]: image for image in instances["images"]}
    segments = {entry["image_id"]: entry["segments_info"] for entry in panoptic["annotations"]}
    by_image: dict[int, list] = {}
    records = []
    rng = random.Random(SEED)

    for annotation in instances["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    candidates = sorted(by_image)
    rng.shuffle(candidates)
    (out / "coco").mkdir(parents=True, exist_ok=True)
    for image_id in candidates:
        image = images[image_id]
        area = image["width"] * image["height"]
        things = thing_counts(by_image[image_id], names, area)
        present = [thing for thing in things.values() if thing["share"] >= PRESENT_AREA]
        if (image["file_name"] in coco60 or image_id not in segments or len(present) < 2
                or not any(thing["crowd"] or thing["count"] >= 2 for thing in present)):
            continue
        path = out / "coco" / image["file_name"]
        if not path.is_file():
            urllib.request.urlretrieve(image["coco_url"], path)
        items = (thing_items(things, categories, rng)
                 + stuff_items(stuff_shares(segments[image_id], stuff_names, area), rng))
        # Their own seed per image, so adding them left every earlier tag of the corpus as it was.
        items += related_items(things, supercategories, {entry["text"] for entry in items},
                               random.Random(f"{SEED}-related-{image_id}"))
        records.append({"file": f"coco/{image['file_name']}", "items": items})
        if len(records) == image_count:
            break
    return {"source": "COCO val2017 instances and panoptic", "seed": SEED, "images": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"corpus folder, default {DEFAULT_OUT}")
    parser.add_argument("--images", type=int, default=IMAGE_COUNT, help=f"images to sample, default {IMAGE_COUNT}")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fetch_annotations(COCO60)
    corpus = build(args.out, args.images)
    counts: dict[str, list[int]] = {}
    for record in corpus["images"]:
        for entry in record["items"]:
            counts.setdefault(entry["form"], [0, 0])[0 if entry["truth"] else 1] += 1
    (args.out / "coco_forms.json").write_text(json.dumps(corpus, indent=1), encoding="utf-8")
    print(f"{len(corpus['images'])} images written to {args.out / 'coco_forms.json'}")
    for form, (present, absent) in counts.items():
        print(f"  {form:<13} {present:>4} present {absent:>4} absent")


if __name__ == "__main__":
    main()
