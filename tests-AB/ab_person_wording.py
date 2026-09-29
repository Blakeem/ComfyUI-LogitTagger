"""The person question's wording for Logit Text Check and Logit Image Check, singular against plural.

The shipped questions name "a person", and a prompt of a crowd read 0.003 on Text Check. Each wording is asked
alone, one request per text or image, as the nodes ask it.
  text   the release-gates text set (80 prompts, person labeled asks, negated or absent) and GROUP_PROMPTS below,
         whose kind was set when each was written
  image  the 72 judged renders (person, and group for two or more people) and the COCO60 photos (person present,
         counted by its COCO instances), read at the check nodes' 0.5 MP cap
Every interval is a 95% bootstrap over prompts or images.
Usage: python tests-AB/ab_person_wording.py [--clip NAME] [--out FILE]
"""

import argparse
import json
from pathlib import Path

import ab_env
from ab_question_shapes import Asker, load_corpus
from ab_release_gates import (
    DEFAULT_CORPUS,
    DEFAULT_DATA,
    accuracy,
    estimate,
    load_labels,
    load_renders,
    load_textset,
    separation,
    yes_rate,
)
from ab_tagger import DEFAULT_CLIP, load_image

TEXT_WORDINGS = (
    "Does this prompt ask for a person in the image?",
    "Does this prompt ask for a person or people in the image?",
    "Does this prompt ask for people in the image?",
    "Does this prompt ask for one or more people in the image?",
    "Does this prompt ask for at least one person in the image?",
)
IMAGE_WORDINGS = (
    "Is there a person in this image?",
    "Is there a person or people in this image?",
    "Are there people in this image?",
    "Are there any people in this image?",
    "Is there at least one person in this image?",
)
# Written for this probe, so each kind is known: a group of people, one person, people negated, or no people.
GROUP_PROMPTS = {
    "group": (
        "A crowd of fans cheers under the lights of a packed stadium",
        "Two children build a sandcastle on the beach",
        "A group of hikers rests on a mountain ridge",
        "A family shares a picnic on a blanket in the park",
        "A team of firefighters sprays water on a burning warehouse",
        "A couple dances the tango in a candlelit ballroom",
        "Three monks in orange robes walk along a temple path",
        "A rock band plays on a smoky stage",
        "Soldiers march in formation across a parade ground",
        "Students listen to a lecture in a crowded hall",
        "A busy market street full of shoppers and street vendors",
        "Wedding guests raise their glasses in a garden at dusk",
    ),
    "single": (
        "A woman reads a book by the window",
        "An old fisherman mends his nets on the dock",
        "An astronaut floats outside the space station",
        "A chef slices vegetables in a restaurant kitchen",
        "A girl swings on a tire swing under an oak tree",
        "A knight rides his horse across a misty field",
        "A street musician plays the violin on a corner",
        "A businessman checks his phone at a crosswalk",
    ),
    "negated": (
        "An empty stadium at night with no fans or players",
        "A deserted beach at sunrise with no people in sight",
        "An abandoned classroom, with no students or teacher",
        "A quiet forest trail with no hikers",
        "An empty subway car with nobody aboard",
        "A city square at dawn, completely empty of people",
    ),
    "absent": (
        "A bowl of ramen on a wooden table",
        "A mountain lake at sunrise",
        "A red sports car parked in a garage",
        "A cat asleep on a windowsill",
        "A lighthouse in a storm",
        "A bouquet of tulips in a glass vase",
    ),
}
GROUP_MIN_PEOPLE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cache" / "person_wording.json",
                        help="raw per-item JSON")
    return parser.parse_args()


def text_units(prompts: list[dict]) -> list[dict]:
    """One unit per prompt: its text, whether it asks for a person, and its kind."""
    units = [{"unit": prompt["id"], "text": prompt["text"], "truth": prompt["labels"]["person"] == "asks",
              "kind": f"set {prompt['labels']['person']}"} for prompt in prompts]
    units += [{"unit": f"{kind}{index}", "text": text, "truth": kind in ("group", "single"), "kind": kind}
              for kind, texts in GROUP_PROMPTS.items() for index, text in enumerate(texts)]
    return units


def image_units(renders, labels: dict, samples) -> list[dict]:
    """One unit per image with a clear person label: whether a person is present, and single or group."""
    units = []
    for render in renders:
        judged = labels[render.id]["checklist"]
        if judged.get("person") not in ("yes", "no"):
            continue
        kind = "none" if judged["person"] == "no" else ("group" if judged.get("group") == "yes" else "single")
        units.append({"unit": render.id, "file": render.file, "truth": judged["person"] == "yes", "kind": f"render {kind}"})
    for sample in samples:
        if "person" in sample.ambiguous:
            continue
        people = sum(name == "person" for name, _area in sample.instances)
        present = "person" in sample.present
        kind = "none" if not present else ("group" if people >= GROUP_MIN_PEOPLE else "single")
        units.append({"unit": sample.file.stem, "file": sample.file, "truth": present, "kind": f"coco {kind}"})
    return units


def ask(asker: Asker, units: list[dict], wordings: tuple[str, ...], images: dict | None) -> list[dict]:
    from logit_classifier import NoulQuestion

    items = []
    for unit in units:
        image = None if images is None else images[unit["unit"]]
        state = unit.get("text", "")
        for wording in wordings:
            answers, _ms = asker.ask(wording, state, {"q": NoulQuestion(instructions=wording)}, image)
            items.append({"unit": unit["unit"], "kind": unit["kind"], "truth": unit["truth"], "wording": wording,
                          "p": answers["q"].noul})
    return items


def report(title: str, items: list[dict], wordings: tuple[str, ...], kinds: list[str], seed: int) -> list[str]:
    lines = [f"== {title}", "p >= 0.5 rate per kind: " + ", ".join(kinds)]
    for wording in wordings:
        rows = [item for item in items if item["wording"] == wording]
        rates = [yes_rate([item for item in rows if item["kind"] == kind]) for kind in kinds]
        lines += [f"{wording}",
                  f"  AUC {estimate(separation, rows, seed)}  accuracy {estimate(accuracy, rows, seed)}",
                  "  " + "  ".join(f"{kind} {rate:.2f}" for kind, rate in zip(kinds, rates, strict=True))]
    return lines


def main() -> None:
    ab_env.bootstrap()
    from logit_classifier import NoulQuestion
    from logit_classifier.toolkit.comfyui import fit_image

    from logit_tagger.engine import CHECK_MAX_PIXELS, build_classifier

    args = parse_args()
    renders = load_renders(DEFAULT_DATA / "renders" / "manifest.json")
    labels = load_labels(DEFAULT_DATA / "renders" / "labels.json")
    _categories, _supercategories, samples = load_corpus(DEFAULT_CORPUS, None)
    texts = text_units(load_textset(DEFAULT_DATA / "textset" / "prompts.json"))
    pictures = image_units([render for render in renders if render.id in labels], labels, samples)
    images = {unit["unit"]: fit_image(load_image(unit["file"]), CHECK_MAX_PIXELS) for unit in pictures}
    text_kinds = ["set asks", "set negated", "set absent", "group", "single", "negated", "absent"]
    image_kinds = ["render group", "render single", "render none", "coco group", "coco single", "coco none"]

    asker = Asker(build_classifier(ab_env.load_clip(args.clip), "ab_person_wording"))
    # The first calls pay CUDA and kernel warmup.
    asker.ask("warmup", "warmup", {"q": NoulQuestion(instructions="warmup")})
    asker.ask("warmup", "", {"q": NoulQuestion(instructions="warmup")}, next(iter(images.values())))
    text_items = ask(asker, texts, TEXT_WORDINGS, None)
    image_items = ask(asker, pictures, IMAGE_WORDINGS, images)

    print(*report(f"Logit Text Check, {len(texts)} prompts", text_items, TEXT_WORDINGS, text_kinds, args.seed),
          "", *report(f"Logit Image Check, {len(pictures)} images", image_items, IMAGE_WORDINGS, image_kinds,
                      args.seed), sep="\n")
    counts = {kind: sum(unit["kind"] == kind for unit in units) for units, kinds in
              ((texts, text_kinds), (pictures, image_kinds)) for kind in kinds}
    print("units per kind:", counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"clip": args.clip, "text": text_items, "image": image_items}, indent=1),
                        encoding="utf-8")
    print(f"raw results: {args.out}")


if __name__ == "__main__":
    main()
