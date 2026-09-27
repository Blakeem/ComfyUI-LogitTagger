"""Which thing check wording best separates things from non-things, over every judged tag of both packs.

A tag judged yes or no names something to point at, present or not, so it counts as a thing. A tag judged vague
counts as a non-thing. A tag with both kinds of verdict, or only unsure, is left out. Prints each wording's AUC and,
at each share of things kept, the share of non-things removed.
Usage: python ab_thing_wordings.py [--catr-judgments FILE] [--clip NAME]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import ab_env

HERE = Path(__file__).resolve().parent
LT_VERDICTS = HERE / "data" / "tagger-ab" / "verdicts.json"
CATR_JUDGMENTS = (HERE.parent.parent / "ComfyUI-ContextAnchoredTileRefine" / "tests-AB" / "cache" / "tags_bench"
                  / "judgments.json")

THING = ("something that could be pointed at in a picture: an object, a person, a group of people "
         "or animals, an animal, a plant, a body part, a garment, a material, a building, a "
         "landscape feature such as hills, cliffs, sky, the moon or stars, a light source such as "
         "lamps, signs or lit windows, or a substance such as water, smoke, fire, clouds or rain")
OTHER = ("nothing to point at: a kind of place or a whole scene, a time of day, a color alone, a "
         "shape or form, lighting in general, a camera or render effect, a style, a mood, a "
         "quality or an idea")
SHORT_THING = "a physical thing or substance you could point at in a picture"
SHORT_OTHER = ("not a thing you could point at: a type of place or scene, a time of day, a color, a shape, "
               "lighting, an effect, a style, a mood, a quality or an idea")
QUESTION = 'What does the image tag "{TAG}" name'

# A choice wording scores P("other"). A noul wording scores 1 - P(true), so both read as p(non-thing).
WORDINGS: dict[str, tuple[Any, ...]] = {
    "current": ("choice", QUESTION, (("thing", THING), ("other", OTHER))),
    "current-swapped": ("choice", QUESTION, (("other", OTHER), ("thing", THING))),
    "short": ("choice", QUESTION, (("thing", SHORT_THING), ("other", SHORT_OTHER))),
    "noul-point": ("noul", 'The image tag "{TAG}" names something you could point at in a picture'),
    "noul-thing": ("noul", 'The image tag "{TAG}" names ' + THING),
    "noul-physical": ("noul", 'The image tag "{TAG}" names a physical object, being, material or substance'),
}


def labels(lt_verdicts: Path, catr_judgments: Path | None) -> dict[str, int]:
    """Each tag with one kind of verdict: 0 for a thing, 1 for a non-thing."""
    votes: dict[str, Counter[str]] = {}
    out: dict[str, int] = {}

    for grades in json.loads(lt_verdicts.read_text(encoding="utf-8")).values():
        for tag, verdict in grades.items():
            votes.setdefault(tag, Counter())[verdict] += 1
    if catr_judgments is not None:
        for entry in json.loads(catr_judgments.read_text(encoding="utf-8")).values():
            for tag, verdict in entry["items"].items():
                votes.setdefault(tag, Counter())[verdict] += 1
    for tag, counts in votes.items():
        things = counts["yes"] + counts["no"]
        if things and not counts["vague"]:
            out[tag] = 0
        elif counts["vague"] and not things:
            out[tag] = 1
    return out


def non_thing_scores(classifier: Any, wording: tuple[Any, ...], tags: list[str]) -> list[float]:
    """p(non-thing) per tag under one wording, 256 questions per request."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    scores: list[float] = []

    if wording[0] == "noul":
        return [1.0 - p for p in classifier.nouls([wording[1].replace("{TAG}", tag) for tag in tags])]
    for start in range(0, len(tags), 256):
        window = tags[start:start + 256]
        questions = {f"t{index}": ChoiceQuestion(instructions=wording[1].replace("{TAG}", tag),
                                                 criteria=dict(wording[2]))
                     for index, tag in enumerate(window)}
        response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions))
        scores += [response.answers[qid].probabilities["other"] for qid in questions]
    return scores


def auc(positives: list[float], negatives: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def report_line(name: str, scores: dict[str, float], label: dict[str, int]) -> str:
    things = sorted(scores[tag] for tag in label if label[tag] == 0)
    others = [scores[tag] for tag in label if label[tag] == 1]
    line = f"{name:<16} AUC {auc(others, things):.3f}"

    for keep in (0.99, 0.98, 0.95, 0.90):
        cutoff = things[min(len(things) - 1, int(keep * len(things)))]
        line += f"  keep {keep:.2f} of things: removes {sum(o > cutoff for o in others) / len(others):.2f}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catr-judgments", type=Path, default=CATR_JUDGMENTS,
                        help="ContextAnchoredTileRefine's tile judgments, skipped when the file is missing")
    parser.add_argument("--clip", default="qwen3vl_4b_fp8_scaled.safetensors")
    args = parser.parse_args()
    catr = args.catr_judgments if args.catr_judgments.is_file() else None
    label = labels(LT_VERDICTS, catr)
    tags = sorted(label)

    ab_env.bootstrap()
    import torch
    from logit_classifier.toolkit.comfyui import comfy_classifier

    print(f"{len(tags)} tags: {sum(v == 0 for v in label.values())} things, {sum(label.values())} non-things"
          f"{'' if catr else ', no tile judgments'}")
    classifier = comfy_classifier(ab_env.load_clip(args.clip), needs_images=False)
    with torch.inference_mode():
        for name, wording in WORDINGS.items():
            scores = dict(zip(tags, non_thing_scores(classifier, wording, tags), strict=True))
            print(report_line(name, scores, label), flush=True)


if __name__ == "__main__":
    main()
