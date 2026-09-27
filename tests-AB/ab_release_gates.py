"""Measure each candidate node on the inputs its users feed it, over Qwen3-VL-4B, to decide which nodes ship.

The data folder holds renders/ (manifest.json, labels.json, JUDGE.md and the Krea 2 renders), textset/prompts.json
(prompts with 12 things each labeled asks, negated or absent). The COCO corpus comes from build_coco60.py.
Each probe stands for one candidate node:
  render-check   image noul, every judge checklist question on the renders (image check node)
  render-style   image choice over the 8 render styles (style router node)
  adherence      whole-prompt noul and per-thing nouls against judged and foreign things (prompt adherence node)
  text-check     text noul, does a prompt ask for a thing, with no image (prompt gate node)
  declared-tags  image noul over the 80 COCO categories as a declared tag list (declared tagger node)
The render probes use the renders that have a judge labels entry. Every interval is a 95% bootstrap over images,
or over prompts for text-check. Results merge by probe into the --out JSON, so separate runs accumulate.
Usage: python tests-AB/ab_release_gates.py [<data dir>] [--corpus FILE] [--probes NAME ...] [--labels FILE] [--limit N] [--max-pixels N] [--out FILE]
"""

import argparse
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import ab_env
from ab_question_shapes import (
    MATCH_STATEMENT,
    QUESTION,
    SHOWN_FAILURES,
    TEXT_STATEMENT,
    Asker,
    auc,
    bootstrap,
    derangement,
    interval,
    load_corpus,
    mean,
    median,
    noun,
    quantiles,
    spearman,
)
from ab_tagger import DEFAULT_CLIP, load_image
from logit_classifier.toolkit.tags import head_noun, presence_question, reads_plural

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "release-gates"
DEFAULT_CORPUS = Path(__file__).resolve().parent / "cache" / "coco60" / "coco60.json"
PROBES = ("render-check", "render-style", "adherence", "text-check", "declared-tags")
RENDER_PROBES = frozenset({"render-check", "render-style", "adherence"})
THRESHOLD = 0.5
WORST_ERRORS = 10

# The JUDGE.md checklist question ids, grouped by what a user would ask the check node.
CHECK_GROUPS = {
    "presence": ("person", "face", "hand", "dog", "cat", "horse", "bird", "vehicle", "sword", "food", "plant",
                 "building", "water", "sky", "text"),
    "scene": ("night", "indoors", "monochrome", "smiling", "group"),
    "style": ("photo", "anime", "painting", "render3d", "pixel", "sketch", "comic"),
}
# The check node asks one question per image, so one question is also timed alone.
SINGLE_QID = "person"

STYLE_QUESTION = "The art style of this image"
STYLE_CRITERIA = {
    "photograph": "a real camera photo",
    "anime illustration": "Japanese anime or manga drawing",
    "oil painting": "visible brushwork in oil paint on canvas",
    "3D render": "computer generated 3D graphics",
    "watercolor": "transparent watercolor washes on paper",
    "pencil sketch": "graphite or ink lines on paper",
    "pixel art": "low resolution art built from visible square pixels",
    "comic book": "western comic art with bold ink outlines",
}
# The judge's style question per manifest style. The judge's painting question covers oil and watercolor.
JUDGE_STYLE = {
    "photograph": "photo",
    "anime illustration": "anime",
    "oil painting": "painting",
    "3D render": "render3d",
    "watercolor": "painting",
    "pencil sketch": "sketch",
    "pixel art": "pixel",
    "comic book": "comic",
}

# The judges found almost every intended thing present, so absent examples come from other renders.
FOREIGN_PER_IMAGE = 3
MATCH_KINDS = ("same-style", "other-style", "near-miss")
TEXT_WORDINGS = {
    "W1": TEXT_STATEMENT,
    "W2": "Does this text mention {noun}?",
    "W3": "Does this prompt ask for {noun} in the image?",
}
THING_PHRASES = {
    "cat": "a cat",
    "dog": "a dog",
    "person": "a person",
    "woman": "a woman",
    "car": "a car",
    "sword": "a sword",
    "water": "water",
    "night": "a night scene",
    "snow": "snow",
    "text": "text or lettering",
    "nsfw": "nudity",
    "violence": "violence",
}
# A decoy is an absent thing whose name sits inside a longer word of the prompt, such as cattle for cat.
TEXT_CLASSES = ("asks", "negated", "absent", "decoy")

DECLARED_WORDINGS = {"question": QUESTION, "statement": "This image visibly contains {noun}"}
DECLARED_THRESHOLDS = (0.5, 0.9, 0.99, 0.9999)


@dataclass(frozen=True)
class Render:
    """One manifest render and the prompt it was generated from."""

    id: str
    file: Path
    prompt: str
    style: str
    intended: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("data", type=Path, nargs="?", default=DEFAULT_DATA, help=f"folder holding renders/ and textset/, default {DEFAULT_DATA}")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help=f"coco60.json from build_coco60.py, default {DEFAULT_CORPUS}")
    parser.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES), help="probes to run")
    parser.add_argument("--labels", type=Path, default=None, help="judge labels, default renders/labels.json")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items of each set")
    parser.add_argument("--seed", type=int, default=7, help="seed for swaps, foreign things and resamples")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="pixel cap for every render and COCO image, default engine.CHECK_MAX_PIXELS")
    parser.add_argument("--out", type=Path, default=None,
                        help="raw per-item JSON, default release_gates.json in the data folder")
    args = parser.parse_args()
    if args.max_pixels is not None and args.max_pixels < 1:
        parser.error(f"--max-pixels must be a positive pixel count, got {args.max_pixels}")
    return args


# ---------------------------------------------------------------- data


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]

    if missing:
        raise SystemExit(f"input files not found: {', '.join(missing)}")


def load_renders(manifest_path: Path) -> list[Render]:
    require_files([manifest_path])
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The manifest holds absolute paths from the rendering machine, so only the file name is kept.
    return [Render(record["id"], manifest_path.parent / PureWindowsPath(record["file"]).name, record["prompt"],
                   record["style"], tuple(record["intended"])) for record in records]


def load_labels(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"judge labels not found: {path}. The render probes score against them.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_checklist(path: Path) -> dict[str, str]:
    """Question id to question text, from the Checklist section of JUDGE.md."""
    require_files([path])
    section = path.read_text(encoding="utf-8").split("## Checklist", 1)[-1].split("\n## ", 1)[0]
    checklist = dict(re.findall(r"^(\w+): (.+)$", section, flags=re.MULTILINE))
    expected = {qid for qids in CHECK_GROUPS.values() for qid in qids}

    if set(checklist) != expected:
        raise SystemExit(f"{path} checklist ids differ from CHECK_GROUPS on {sorted(set(checklist) ^ expected)}")
    return checklist


def load_textset(path: Path) -> list[dict]:
    require_files([path])
    textset = json.loads(path.read_text(encoding="utf-8"))
    differing = set(textset["things"]) ^ set(THING_PHRASES)

    if differing:
        raise SystemExit(f"{path} things differ from THING_PHRASES on {sorted(differing)}")
    return textset["prompts"]


# ---------------------------------------------------------------- metrics over items
# Every item carries "unit" (the image or prompt one resample draws), "p", and "truth": True, False, or None
# for an item left out of scoring.


def estimate(statistic: Callable[[list[dict]], float], items: list[dict], seed: int) -> str:
    """The statistic over every item, then its 95% interval over resampled units."""
    import numpy as np

    by_unit: dict[str, list[dict]] = {}
    units: list[list[dict]] = []
    spread = None

    if not items:
        return "n/a"
    for item in items:
        by_unit.setdefault(item["unit"], []).append(item)
    units = list(by_unit.values())
    spread = bootstrap(lambda indices: statistic([item for index in indices for item in units[index]]),
                       len(units), seed)
    if np.isnan(spread).all():
        return f"{statistic(items):.4f} [n/a]"
    return f"{statistic(items):.4f} {interval(spread)}"


def separation(items: list[dict]) -> float:
    """AUC of p, items labeled True against items labeled False."""
    return auc([item["p"] for item in items if item["truth"] is True],
               [item["p"] for item in items if item["truth"] is False])


def accuracy(items: list[dict]) -> float:
    rows = [item for item in items if item["truth"] is not None]
    return sum((item["p"] >= THRESHOLD) == item["truth"] for item in rows) / len(rows) if rows else math.nan


def yes_rate(items: list[dict]) -> float:
    return sum(item["p"] >= THRESHOLD for item in items) / len(items) if items else math.nan


def precision(items: list[dict], flagged: Callable[[dict], bool]) -> float:
    """Of the labeled items flagged, the fraction labeled True."""
    hits = [item["truth"] for item in items if item["truth"] is not None and flagged(item)]
    return sum(hits) / len(hits) if hits else math.nan


def recall(items: list[dict], flagged: Callable[[dict], bool]) -> float:
    """Of the items labeled True, the fraction flagged."""
    found = [flagged(item) for item in items if item["truth"] is True]
    return sum(found) / len(found) if found else math.nan


def extremes(title: str, items: list[dict], describe: Callable[[dict], str], highest: bool) -> str:
    rows = sorted(items, key=lambda item: item["p"], reverse=highest)[:SHOWN_FAILURES]
    return f"  {title}: " + ", ".join(f"{describe(item)} {item['p']:.4f}" for item in rows)


# ---------------------------------------------------------------- render-check


def probe_render_check(asker: Asker, renders: list[Render], images: dict, labels: dict, checklist: dict[str, str],
                       seed: int) -> tuple[list[str], dict]:
    """Every checklist question, verbatim, packed per image. Then one question alone per image, for its cost."""
    from logit_classifier import NoulQuestion

    items: list[dict] = []
    drift: list[float] = []
    questions = {qid: NoulQuestion(instructions=text) for qid, text in checklist.items()}
    alone = {SINGLE_QID: questions[SINGLE_QID]}

    for render in renders:
        judged = labels[render.id]["checklist"]
        answers, _ms = asker.ask("render-check packed", "", questions, images[render.id])
        items += [{"unit": render.id, "qid": qid, "truth": judged[qid] == "yes", "p": answers[qid].noul}
                  for qid in checklist if qid in judged]
        single, _ms = asker.ask("render-check single", "", alone, images[render.id])
        drift.append(abs(single[SINGLE_QID].noul - answers[SINGLE_QID].noul))

    lines = ["per question: positives, negatives, AUC, accuracy at 0.5"]
    for qids in CHECK_GROUPS.values():
        for qid in qids:
            rows = [item for item in items if item["qid"] == qid]
            positives = sum(item["truth"] for item in rows)
            lines.append(f"  {qid:<10} {positives:>3} {len(rows) - positives:>3}  AUC {estimate(separation, rows, seed)}"
                         f"  acc {estimate(accuracy, rows, seed)}")
    lines.append("pooled by group")
    for group, qids in (*CHECK_GROUPS.items(), ("all", tuple(checklist))):
        rows = [item for item in items if item["qid"] in qids]
        lines.append(f"  {group:<9} {len(rows):>4} items  AUC {estimate(separation, rows, seed)}  "
                     f"acc {estimate(accuracy, rows, seed)}")
    worst = sorted(items, key=lambda item: abs(item["p"] - item["truth"]), reverse=True)[:WORST_ERRORS]
    lines += [
        "worst errors: " + ", ".join(f"{item['unit']} {item['qid']} {item['p']:.4f} {'yes' if item['truth'] else 'no'}"
                                     for item in worst),
        f"mean {asker.mean_ms('render-check packed'):.0f} ms per image for {len(questions)} questions packed, "
        f"{asker.mean_ms('render-check single'):.0f} ms for one question alone",
        f"|p alone - p packed| max {max(drift, default=0.0):.2e}",
    ]
    return lines, {"items": items}


# ---------------------------------------------------------------- render-style


def probe_render_style(asker: Asker, renders: list[Render], images: dict, labels: dict,
                       seed: int) -> tuple[list[str], dict]:
    """One choice over the manifest styles, scored against the manifest and against the judges' style answers."""
    from logit_classifier import ChoiceQuestion

    items: list[dict] = []
    question = {"style": ChoiceQuestion(instructions=STYLE_QUESTION, criteria=STYLE_CRITERIA)}
    correct = lambda rows: mean([row["choice"] == row["style"] for row in rows])

    for render in renders:
        judged = labels[render.id]["checklist"]
        answers, _ms = asker.ask("render-style", "", question, images[render.id])
        answer = answers["style"]
        items.append({"unit": render.id, "style": render.style, "choice": answer.choice,
                      "confidence": answer.confidence, "abstain": answer.abstain,
                      "probabilities": answer.probabilities,
                      "judge": sorted(qid for qid in CHECK_GROUPS["style"] if judged.get(qid) == "yes")})

    right = [item for item in items if item["choice"] == item["style"]]
    wrong = [item for item in items if item["choice"] != item["style"]]
    confusions = Counter((item["style"], item["choice"]) for item in wrong)
    judged_rows = [item for item in items if item["judge"]]
    agreed = [item for item in judged_rows if JUDGE_STYLE[item["style"]] in item["judge"]]
    disagreed = [item for item in judged_rows if JUDGE_STYLE[item["style"]] not in item["judge"]]
    lines = [
        f"accuracy against the manifest style {estimate(correct, items, seed)} ({len(right)}/{len(items)}), "
        f"mean {asker.mean_ms('render-style'):.0f} ms",
        "  per style: " + ", ".join(f"{style} {correct([i for i in items if i['style'] == style]):.2f}"
                                    for style in STYLE_CRITERIA),
        "  confusions, manifest -> choice: " + (", ".join(f"{truth} -> {choice} {count}"
                                                         for (truth, choice), count in confusions.most_common())
                                                or "none"),
        f"  confidence right {mean([i['confidence'] for i in right]):.3f}  wrong {mean([i['confidence'] for i in wrong]):.3f}"
        f"  AUC right vs wrong {auc([i['confidence'] for i in right], [i['confidence'] for i in wrong]):.3f}",
        f"  abstain    right {mean([i['abstain'] or 0.0 for i in right]):.3f}  "
        f"wrong {mean([i['abstain'] or 0.0 for i in wrong]):.3f}",
        f"judge check on {len(judged_rows)} renders with an agreed style yes:",
        f"  manifest style matches a judge style {len(agreed)}/{len(judged_rows)}",
        f"  choice matches a judge style {sum(JUDGE_STYLE[i['choice']] in i['judge'] for i in judged_rows)}"
        f"/{len(judged_rows)}",
        f"  accuracy against the manifest where the judges agree with it {correct(agreed):.3f} "
        f"({sum(i['choice'] == i['style'] for i in agreed)}/{len(agreed)})",
        "  judges disagree with the manifest: " + (", ".join(
            f"{i['unit']} {i['style']} judge {'+'.join(i['judge'])} choice {i['choice']}" for i in disagreed)
            or "none"),
    ]
    return lines, {"items": items}


# ---------------------------------------------------------------- adherence


def word_forms(word: str) -> set[str]:
    forms = {word, f"{word}s", f"{word}es"}

    if word.endswith("es"):
        forms.add(word[:-2])
    if word.endswith("s"):
        forms.add(word[:-1])
    return forms


def phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)


def foreign_things(render: Render, renders: list[Render], seed: int) -> list[tuple[str, str]]:
    """(source id, thing) pairs drawn from other renders' intended things, each with a head noun this render's
    prompt and intended list never name, and no two sharing a head noun."""
    named = set(re.findall(r"[a-z]+", " ".join((render.prompt, *render.intended)).lower()))
    pool = [(other.id, thing) for other in renders if other.id != render.id for thing in other.intended
            if not word_forms(head_noun(thing)) & named]
    chosen: list[tuple[str, str]] = []
    heads: set[str] = set()

    random.Random(f"{seed}-foreign-{render.id}").shuffle(pool)
    for source, thing in pool:
        if len(chosen) == FOREIGN_PER_IMAGE:
            break
        if head_noun(thing) not in heads:
            heads.add(head_noun(thing))
            chosen.append((source, thing))
    return chosen


def near_miss(render: Render, foreign: list[tuple[str, str]], judged: dict, seed: int) -> dict | None:
    """The render's prompt with one intended phrase, found verbatim and not judged absent, replaced by a foreign
    thing, preferring a pair of the same grammatical number so the edit leaves the prompt grammatical."""
    found = [phrase for phrase in render.intended
             if judged.get(phrase) != "absent" and phrase_pattern(phrase).search(render.prompt)]
    pairs: list[tuple[str, str]] = []
    same_number: list[tuple[str, str]] = []

    if not found or not foreign:
        return None
    random.Random(f"{seed}-near-{render.id}").shuffle(found)
    pairs = [(phrase, thing) for phrase in found for _source, thing in foreign]
    same_number = [(phrase, thing) for phrase, thing in pairs if reads_plural(phrase) == reads_plural(thing)]
    phrase, thing = (same_number or pairs)[0]
    return {"replaced": phrase, "thing": thing,
            "prompt": phrase_pattern(phrase).sub(lambda _match: thing, render.prompt, count=1)}


def swap_sources(renders: list[Render], seed: int) -> dict[str, dict[str, Render]]:
    """Per render id, the other render whose prompt stands as its same-style swap and its other-style swap."""
    by_style: dict[str, list[Render]] = {}
    sources: dict[str, dict[str, Render]] = {}

    for render in renders:
        by_style.setdefault(render.style, []).append(render)
    for group in by_style.values():
        order = derangement(len(group), seed)
        for index, render in enumerate(group):
            others = [other for other in renders if other.style != render.style]
            sources[render.id] = {}
            if order[index] != index:
                sources[render.id]["same-style"] = group[order[index]]
            if others:
                sources[render.id]["other-style"] = random.Random(f"{seed}-other-{render.id}").choice(others)
    return sources


def probe_adherence(asker: Asker, renders: list[Render], all_renders: list[Render], images: dict, labels: dict,
                    seed: int) -> tuple[list[str], dict]:
    """Per render, one packed request: the whole-prompt noul on its own prompt, two swaps and a near miss, and
    one presence noul per intended thing and per foreign thing."""
    from logit_classifier import NoulQuestion

    sources = swap_sources(all_renders, seed)
    match_items: list[dict] = []
    thing_items: list[dict] = []
    near_misses: list[dict] = []

    for render in renders:
        judged = labels[render.id]["intended"]
        foreign = foreign_things(render, all_renders, seed)
        near = near_miss(render, foreign, judged, seed)
        prompts = {"matched": (render.id, render.prompt)}
        prompts |= {kind: (source.id, source.prompt) for kind, source in sources[render.id].items()}
        if near is not None:
            prompts["near-miss"] = (render.id, near["prompt"])
            near_misses.append({"unit": render.id, **near})
        things = [("intended", render.id, thing, judged.get(thing)) for thing in render.intended]
        things += [("foreign", source, thing, "absent") for source, thing in foreign]
        questions = {kind: NoulQuestion(instructions=MATCH_STATEMENT.format(caption=text))
                     for kind, (_source, text) in prompts.items()}
        questions |= {f"t{index}": NoulQuestion(instructions=presence_question(thing))
                      for index, (_origin, _source, thing, _label) in enumerate(things)}
        answers, _ms = asker.ask("adherence", "", questions, images[render.id])
        match_items += [{"unit": render.id, "kind": kind, "source": source, "p": answers[kind].noul}
                        for kind, (source, _text) in prompts.items()]
        thing_items += [{"unit": render.id, "origin": origin, "source": source, "thing": thing,
                         "asked": questions[f"t{index}"].instructions,
                         "truth": None if label is None else label == "present", "p": answers[f"t{index}"].noul}
                        for index, (origin, source, thing, label) in enumerate(things)]

    flagged = sorted(({"id": item["unit"], "thing": item["thing"], "p": item["p"], "source": item["source"]}
                      for item in thing_items if item["origin"] == "foreign" and item["p"] >= THRESHOLD),
                     key=lambda row: row["p"], reverse=True)
    lines = [
        *match_lines(match_items, seed),
        *thing_lines(thing_items, len(flagged), seed),
        *fraction_lines(match_items, thing_items, seed),
        *near_miss_lines(match_items, thing_items, near_misses, seed),
        *image_level_lines(thing_items, near_misses, seed),
        f"mean {asker.mean_ms('adherence'):.0f} ms per packed request",
    ]
    return lines, {"match_items": match_items, "thing_items": thing_items, "near_misses": near_misses,
                   "foreign_flagged": flagged}


def match_lines(match_items: list[dict], seed: int) -> list[str]:
    """(a) The whole-prompt noul, own prompt against each swapped or edited prompt."""
    matched = {item["unit"]: item for item in match_items if item["kind"] == "matched"}
    lines = [f"(a) whole-prompt noul {MATCH_STATEMENT!r}: matched on {len(matched)} renders, "
             f"median {median([item['p'] for item in matched.values()]):.4f}, "
             f"p >= 0.5 rate {yes_rate(list(matched.values())):.3f}"]

    for kind in MATCH_KINDS:
        swapped = [item for item in match_items if item["kind"] == kind]
        pair = [{**matched[item["unit"]], "truth": True} for item in swapped]
        pair += [{**item, "truth": False} for item in swapped]
        wins = sum(matched[item["unit"]]["p"] > item["p"] for item in swapped)
        lines.append(f"  matched vs {kind:<11} AUC {estimate(separation, pair, seed)}  matched higher on "
                     f"{wins}/{len(swapped)}  median {median([item['p'] for item in swapped]):.4f}  "
                     f"p >= 0.5 rate {yes_rate(swapped):.3f}")
    return lines


def thing_lines(thing_items: list[dict], flagged_count: int, seed: int) -> list[str]:
    """(b) Presence nouls: judged present against foreign, and the missing list (p < 0.5)."""
    scored = [item for item in thing_items if item["truth"] is not None]
    present = [item for item in scored if item["truth"]]
    foreign = [item for item in scored if item["origin"] == "foreign"]
    judged_absent = [item for item in scored if item["origin"] == "intended" and not item["truth"]]
    # The missing list flags the absent side, so its truth is the flip of presence.
    missing = [{**item, "truth": not item["truth"]} for item in scored]
    flagged_missing = lambda item: item["p"] < THRESHOLD
    missing_precision = estimate(lambda rows: precision(rows, flagged_missing), missing, seed)
    missing_recall = estimate(lambda rows: recall(rows, flagged_missing), missing, seed)
    describe = lambda item: f"{item['unit']} {item['thing']!r}"

    return [
        f"(b) presence noul {QUESTION!r}: {len(present)} present, {len(foreign)} foreign, "
        f"{len(judged_absent)} judged absent, {len(thing_items) - len(scored)} unlabeled skipped",
        f"  AUC present vs foreign {estimate(separation, present + foreign, seed)}",
        f"  p quantiles 5/25/50/75/95  present {quantiles([i['p'] for i in present])}",
        f"                             foreign {quantiles([i['p'] for i in foreign])}",
        "  judged absent: " + (", ".join(f"{describe(i)} {i['p']:.4f}" for i in judged_absent) or "none"),
        f"  missing list (p < 0.5) against foreign and judged absent: precision {missing_precision}  "
        f"recall {missing_recall}",
        f"  present flagged missing: {sum(i['p'] < THRESHOLD for i in present)}/{len(present)}",
        extremes("lowest present", present, describe, False),
        f"  foreign at p >= 0.5: {flagged_count}/{len(foreign)}, written for a judge to check",
        extremes("highest foreign", foreign, describe, True),
    ]


def fraction_lines(match_items: list[dict], thing_items: list[dict], seed: int) -> list[str]:
    """(c) Does the whole-prompt noul follow the judged fraction of intended things present?"""
    rows: list[dict] = []
    rho = lambda chosen: spearman([row["fraction"] for row in chosen], [row["p"] for row in chosen])

    for item in match_items:
        labeled = [thing["truth"] for thing in thing_items if thing["unit"] == item["unit"]
                   and thing["origin"] == "intended" and thing["truth"] is not None]
        if item["kind"] == "matched" and labeled:
            rows.append({"unit": item["unit"], "fraction": mean(labeled), "p": item["p"]})
    return [f"(c) Spearman(judged fraction present, matched noul) {estimate(rho, rows, seed)} over {len(rows)} "
            f"renders, {sum(row['fraction'] < 1.0 for row in rows)} with a fraction below 1"]


def near_miss_lines(match_items: list[dict], thing_items: list[dict], near_misses: list[dict],
                    seed: int) -> list[str]:
    """(d) On each near-miss prompt, the substituted thing's presence noul beside the whole-prompt noul."""
    foreign_p = {(item["unit"], item["thing"]): item["p"] for item in thing_items if item["origin"] == "foreign"}
    whole_p = {item["unit"]: item["p"] for item in match_items if item["kind"] == "near-miss"}
    rows = [{"unit": near["unit"], "thing": near["thing"], "replaced": near["replaced"],
             "p": foreign_p[(near["unit"], near["thing"])], "whole": whole_p[near["unit"]]} for near in near_misses]
    thing_missed = lambda chosen: mean([row["p"] < THRESHOLD for row in chosen])
    whole_missed = lambda chosen: mean([row["whole"] < THRESHOLD for row in chosen])

    return [
        f"(d) near miss on {len(rows)} renders: substituted thing flagged missing (p < 0.5) "
        f"{estimate(thing_missed, rows, seed)}, whole-prompt noul below 0.5 {estimate(whole_missed, rows, seed)}",
        "  substituted thing not flagged: " + (", ".join(
            f"{row['unit']} {row['replaced']!r} -> {row['thing']!r} {row['p']:.4f}"
            for row in rows if row["p"] >= THRESHOLD) or "none"),
    ]


def image_level_lines(thing_items: list[dict], near_misses: list[dict], seed: int) -> list[str]:
    """(e) Whole-image completeness from the presence nouls: an image reads complete when every labeled intended
    thing, and on a near miss the substituted thing in place of the replaced one, scores p >= 0.5."""
    foreign_p = {(item["unit"], item["thing"]): item["p"] for item in thing_items if item["origin"] == "foreign"}
    intended: dict[str, list[dict]] = {}
    own: list[dict] = []
    near: list[dict] = []
    complete: list[dict] = []

    for item in thing_items:
        if item["origin"] == "intended" and item["truth"] is not None:
            intended.setdefault(item["unit"], []).append(item)
    # An image reads complete only when its least likely thing does, so the lowest p stands for the image.
    own = [{"unit": unit, "truth": all(item["truth"] for item in things), "p": min(item["p"] for item in things)}
           for unit, things in intended.items()]
    near = [{"unit": miss["unit"], "truth": False,
             "p": min([foreign_p[(miss["unit"], miss["thing"])],
                       *(item["p"] for item in intended.get(miss["unit"], []) if item["thing"] != miss["replaced"])])}
            for miss in near_misses]
    complete = [row for row in own if row["truth"]]

    return [
        f"(e) image complete when every labeled thing reads p >= 0.5: own prompt {estimate(accuracy, own, seed)} "
        f"on {len(own)} renders, near miss {estimate(accuracy, near, seed)} on {len(near)}, "
        f"both {estimate(accuracy, own + near, seed)} on {len(own) + len(near)}",
        f"  truly complete predicted incomplete: {sum(row['p'] < THRESHOLD for row in complete)}/{len(complete)}",
    ]


# ---------------------------------------------------------------- text-check


def decoy_word(text: str, thing: str) -> str | None:
    """A longer word of the text holding the thing's name, which a substring match would count as a mention."""
    for word in re.findall(r"[a-z]+", text.lower()):
        if thing in word and word not in (thing, f"{thing}s", f"{thing}es"):
            return word
    return None


def probe_text_check(asker: Asker, prompts: list[dict], seed: int) -> tuple[list[str], dict]:
    """Three wordings of 'does this prompt ask for the thing', packed per prompt, with the prompt as the state."""
    from logit_classifier import NoulQuestion

    items: list[dict] = []
    lines: list[str] = []

    for prompt in prompts:
        decoys = {thing: decoy_word(prompt["text"], thing) for thing, label in prompt["labels"].items()
                  if label == "absent"}
        for wording, template in TEXT_WORDINGS.items():
            questions = {thing: NoulQuestion(instructions=template.format(noun=THING_PHRASES[thing]))
                         for thing in prompt["labels"]}
            answers, _ms = asker.ask(f"text-check {wording}", prompt["text"], questions)
            items += [{"unit": prompt["id"], "wording": wording, "thing": thing,
                       "class": "decoy" if decoys.get(thing) else label, "decoy": decoys.get(thing),
                       "truth": label == "asks", "p": answers[thing].noul}
                      for thing, label in prompt["labels"].items()]

    for wording, template in TEXT_WORDINGS.items():
        lines += text_wording_lines(wording, template, [item for item in items if item["wording"] == wording],
                                    asker, seed)
    lines += decoy_lines(items)
    return lines, {"items": items}


def text_wording_lines(wording: str, template: str, items: list[dict], asker: Asker, seed: int) -> list[str]:
    classes = {name: [item for item in items if item["class"] == name] for name in TEXT_CLASSES}
    asks = classes["asks"]
    describe = lambda item: f"{item['unit']} {item['thing']}"
    lines = [f"{wording} {template!r}: mean {asker.mean_ms(f'text-check {wording}'):.0f} ms per request of "
             f"{len(THING_PHRASES)} questions"]

    for other in TEXT_CLASSES[1:]:
        lines.append(f"  AUC asks vs {other:<8} {estimate(separation, asks + classes[other], seed)}  "
                     f"({len(asks)} asks, {len(classes[other])} {other})")
    lines += [f"  p >= 0.5 rate {name:<8} {estimate(yes_rate, rows, seed)}" for name, rows in classes.items()]
    lines.append(f"  accuracy, negated, absent and decoy count as no: {estimate(accuracy, items, seed)}")
    lines.append("  per thing, p >= 0.5 rate on asks / negated / absent (n asks, n negated)")
    for thing in THING_PHRASES:
        rates = [yes_rate([item for item in classes[name] if item["thing"] == thing])
                 for name in ("asks", "negated", "absent")]
        counts = [sum(item["thing"] == thing for item in classes[name]) for name in ("asks", "negated")]
        lines.append(f"    {thing:<9} {' / '.join(f'{rate:.2f}' for rate in rates)}  ({counts[0]}, {counts[1]})")
    lines += [
        extremes("lowest asks", asks, describe, False),
        extremes("highest negated", classes["negated"], describe, True),
        extremes("highest absent", classes["absent"], describe, True),
    ]
    return lines


def decoy_lines(items: list[dict]) -> list[str]:
    decoys: dict[tuple[str, str], dict[str, float]] = {}
    word: dict[tuple[str, str], str] = {}

    for item in items:
        if item["class"] == "decoy":
            decoys.setdefault((item["unit"], item["thing"]), {})[item["wording"]] = item["p"]
            word[(item["unit"], item["thing"])] = item["decoy"]
    return [
        f"decoys, p under {' / '.join(TEXT_WORDINGS)}",
        *(f"  {unit} {thing:<6} {word[(unit, thing)]!r:<14} "
          + " / ".join(f"{scores[wording]:.4f}" for wording in TEXT_WORDINGS)
          for (unit, thing), scores in decoys.items()),
    ]


# ---------------------------------------------------------------- declared-tags


def probe_declared_tags(asker: Asker, samples: list, images: dict, categories: list[str],
                        seed: int) -> tuple[list[str], dict]:
    """All 80 COCO categories as one declared tag list, packed per image, in two wordings."""
    from logit_classifier import NoulQuestion

    items: list[dict] = []
    lines: list[str] = []
    describe = lambda item: f"{item['unit']} {item['category']}"

    for sample in samples:
        for wording, template in DECLARED_WORDINGS.items():
            questions = {f"q{index}": NoulQuestion(instructions=template.format(noun=noun(category)))
                         for index, category in enumerate(categories)}
            answers, _ms = asker.ask(f"declared-tags {wording}", "", questions, images[sample.file])
            items += [{"unit": sample.file.name, "wording": wording, "category": category,
                       "truth": None if category in sample.ambiguous else category in sample.present,
                       "p": answers[f"q{index}"].noul} for index, category in enumerate(categories)]

    for wording, template in DECLARED_WORDINGS.items():
        rows = [item for item in items if item["wording"] == wording]
        scored = [item for item in rows if item["truth"] is not None]
        positives = sum(item["truth"] for item in scored)
        lines += [
            f"{wording} {template!r}: mean {asker.mean_ms(f'declared-tags {wording}'):.0f} ms per image of "
            f"{len(categories)} questions",
            f"  AUC present vs absent {estimate(separation, scored, seed)}  ({positives} present, "
            f"{len(scored) - positives} absent, {len(rows) - len(scored)} ambiguous left out)",
        ]
        for threshold in DECLARED_THRESHOLDS:
            kept = lambda item, threshold=threshold: item["p"] >= threshold
            tags = sum(kept(item) for item in rows) / len(samples)
            kept_precision = estimate(lambda chosen, kept=kept: precision(chosen, kept), scored, seed)
            kept_recall = estimate(lambda chosen, kept=kept: recall(chosen, kept), scored, seed)
            lines.append(f"  {threshold:<7} precision {kept_precision}  recall {kept_recall}  "
                         f"{tags:.1f} tags per image")
        lines += [
            extremes("highest absent", [item for item in scored if not item["truth"]], describe, True),
            extremes("lowest present", [item for item in scored if item["truth"]], describe, False),
        ]
    return lines, {"items": items}


# ---------------------------------------------------------------- main


def read_record(path: Path) -> dict:
    """The earlier runs' results, so a run of some probes keeps the others."""
    if not path.is_file():
        return {"probes": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ab_env.bootstrap()
    from logit_classifier import NoulQuestion
    from logit_classifier.toolkit.comfyui import fit_picture

    from logit_tagger.engine import CHECK_MAX_PIXELS, build_classifier

    args = parse_args()
    max_pixels = CHECK_MAX_PIXELS if args.max_pixels is None else args.max_pixels
    out_path = args.out or args.data / "release_gates.json"
    flagged_path = out_path.parent / "foreign_flagged.json"
    wanted = set(args.probes)
    all_renders: list[Render] = []
    renders: list[Render] = []
    labels: dict = {}
    checklist: dict[str, str] = {}
    prompts: list[dict] = []
    categories: list[str] = []
    samples: list = []
    render_images: dict = {}
    coco_images: dict = {}
    record = read_record(out_path)

    # Created before the model loads, so a new --out folder never fails the run after its first probe.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Every input is read and checked before the CLIP loads, so a missing file fails in seconds.
    if wanted & RENDER_PROBES:
        all_renders = load_renders(args.data / "renders" / "manifest.json")
        labels = load_labels(args.labels or args.data / "renders" / "labels.json")
        checklist = load_checklist(args.data / "renders" / "JUDGE.md")
        renders = [render for render in all_renders if render.id in labels][: args.limit]
        require_files([render.file for render in renders])
        render_images = {render.id: fit_picture(load_image(render.file), max_pixels)
                         for render in renders}
    if "text-check" in wanted:
        prompts = load_textset(args.data / "textset" / "prompts.json")[: args.limit]
    if "declared-tags" in wanted:
        categories, _supercategories, samples = load_corpus(args.corpus, args.limit)
        require_files([sample.file for sample in samples])
        coco_images = {sample.file: fit_picture(load_image(sample.file), max_pixels)
                       for sample in samples}

    asker = Asker(build_classifier(ab_env.load_clip(args.clip), "ab_release_gates"))
    # The first calls pay CUDA and kernel warmup, which no probe should carry.
    asker.ask("warmup", "warmup", {"q": NoulQuestion(instructions="warmup")})
    for image in list(render_images.values())[:1] + list(coco_images.values())[:1]:
        asker.ask("warmup", "", {"q": NoulQuestion(instructions="warmup")}, image)

    runs = {
        "render-check": lambda: probe_render_check(asker, renders, render_images, labels, checklist, args.seed),
        "render-style": lambda: probe_render_style(asker, renders, render_images, labels, args.seed),
        "adherence": lambda: probe_adherence(asker, renders, all_renders, render_images, labels, args.seed),
        "text-check": lambda: probe_text_check(asker, prompts, args.seed),
        "declared-tags": lambda: probe_declared_tags(asker, samples, coco_images, categories, args.seed),
    }
    for name in args.probes:
        lines, result = runs[name]()
        result["timings_ms"] = {label: ms for label, ms in asker.timings.items() if label.startswith(name)}
        record["probes"][name] = {"clip": args.clip, "seed": args.seed, "limit": args.limit,
                                "max_pixels": max_pixels, **result}
        print(f"== {name}", *lines, "", sep="\n", flush=True)
        # Written after every probe, so a later probe's failure keeps the earlier results.
        out_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        if name == "adherence":
            flagged_path.write_text(json.dumps(result["foreign_flagged"], indent=1), encoding="utf-8")
            print(f"foreign things at p >= 0.5: {flagged_path}", flush=True)
    print(f"raw results: {out_path}")


if __name__ == "__main__":
    main()
