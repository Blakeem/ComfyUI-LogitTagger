"""Measure how well each question shape answers over Qwen3-VL-4B, to decide which nodes are worth building.

The corpus JSON holds "categories" (the 80 COCO names) and "images", each with "file" (relative to the
JSON), "present" (categories covering >= 1% of the image), "ambiguous" (present below 1%) and "captions".
tests-AB/build_coco60.py builds it, with COCO's annotations/instances_val2017.json beside it. Each probe stands for one candidate node:
  presence   image noul, object presence (gate node), two wordings, and the cost of one question alone
  router     image choice over the 12 COCO supercategories (router node)
  degrade    image score and noul under controlled blur and darkness (score node)
  count      image score of the object count against COCO instance counts
  text       text noul, category mentions in a caption, no image (text gate node)
  adherence  caption against its own image and a mismatched one (prompt check node)
Usage: python tests-AB/ab_question_shapes.py <coco60.json> [--probes NAME ...] [--limit N] [--out FILE]
"""

import argparse
import json
import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import ab_env
from ab_tagger import DEFAULT_CLIP, load_image

PROBES = ("presence", "router", "degrade", "count", "text", "adherence")
ABSENT_PER_IMAGE = 8
PRESENCE_THRESHOLDS = (0.5, 0.9, 0.99)
TEXT_THRESHOLDS = (0.5, 0.9, 0.99, 0.9999)
QUANTILES = (5, 25, 50, 75, 95)
# NoulAnswer rounds to 6 decimals, so a p at or above this reads as 1 and ties every other one.
SATURATED = 0.999999
# Image level resamples behind every interval, so one image's correlated items move together.
BOOTSTRAP_ROUNDS = 2000
SHOWN_FAILURES = 6

STATEMENT = "This image contains {noun}"
QUESTION = "Is there {noun} in this image?"
TEXT_STATEMENT = "This text mentions {noun}"
MATCH_STATEMENT = "This image matches the description: {caption}"

ROUTER_INSTRUCTIONS = "The main subject of this image"

DEGRADE_IMAGES = 20
# Sigmas are in pixels of a 1 MP image and scale with the image's side, so a 640x480 COCO
# image gets the same blur relative to its size as a 1 MP render would.
BLUR_SIGMAS = (0.0, 1.0, 2.5, 5.0, 10.0)
DARK_FACTORS = (1.0, 0.7, 0.45, 0.25, 0.1)
SHARP_SCORE = ("How sharp is this image", ["very blurry", "blurry", "slightly soft", "sharp", "very sharp"])
BRIGHT_SCORE = ("How bright is this image", ["very dark", "dark", "normal", "bright", "very bright"])
SHARP_NOUL = "This image is sharp"
BRIGHT_NOUL = "This image is bright"

COUNT_SCORE = ("How many distinct objects does this image show",
               ["one", "two or three", "four to six", "seven to ten", "more than ten"])
# The lowest count each COUNT_SCORE level covers.
COUNT_BIN_FLOORS = (1, 2, 4, 7, 11)

# COCO names that captions rarely spell out. Without these, "person" matched almost no caption.
SYNONYMS = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children",
               "kid", "guy", "lady", "ladies", "player", "skier", "surfer", "skateboarder", "snowboarder"],
    "airplane": ["airplane", "plane", "jet", "airliner"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle", "motorbike"],
    "couch": ["couch", "sofa"],
    "tv": ["tv", "television"],
    "cell phone": ["cell phone", "cellphone", "phone"],
    "dining table": ["dining table", "table"],
    "sports ball": ["sports ball", "ball"],
    "donut": ["donut", "doughnut"],
    "hair drier": ["hair drier", "hair dryer"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skis": ["skis", "ski"],
}
PLURAL_CATEGORIES = frozenset({"skis"})


@dataclass
class Sample:
    """One corpus image with its labels and COCO annotations."""

    file: Path
    present: list[str]
    ambiguous: list[str]
    captions: list[str]
    instances: list[tuple[str, float]] = field(default_factory=list)


class Asker:
    """Times each classify call on a synchronized GPU and keeps the milliseconds per probe."""

    def __init__(self, classifier) -> None:
        self.classifier = classifier
        self.timings: dict[str, list[float]] = {}

    def ask(self, probe: str, state: str, questions: dict, image=None) -> tuple[dict, float]:
        import torch
        from logit_classifier import SystemOneRequest

        torch.cuda.synchronize()
        started = time.perf_counter()
        response, _diagnostics = self.classifier.classify(SystemOneRequest(state=state, questions=questions),
                                                          image=image)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        self.timings.setdefault(probe, []).append(elapsed)
        return response.answers, elapsed

    def mean_ms(self, probe: str) -> float:
        values = self.timings.get(probe, [])
        return sum(values) / len(values) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("corpus", type=Path, help="coco60.json, with annotations/instances_val2017.json beside it")
    parser.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES), help="probes to run")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument("--seed", type=int, default=7, help="seed for absent categories and caption swaps")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--out", type=Path, default=None,
                        help="raw per-item JSON, default question_shapes.json beside the corpus")
    return parser.parse_args()


# ---------------------------------------------------------------- corpus


def load_corpus(path: Path, limit: int | None) -> tuple[list[str], dict[str, str], list[Sample]]:
    """Categories, category to supercategory, and the samples with every COCO instance attached."""
    corpus = json.loads(path.read_text(encoding="utf-8"))
    instances_path = path.parent / "annotations" / "instances_val2017.json"
    samples: list[Sample] = []
    by_image: dict[int, list[tuple[str, float]]] = {}

    if not instances_path.is_file():
        raise SystemExit(f"COCO instances file not found: {instances_path}")
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in instances["categories"]}
    supercategories = {category["name"]: category["supercategory"] for category in instances["categories"]}

    for record in corpus["images"][:limit]:
        file = path.parent / record["file"].replace("\\", "/")
        samples.append(Sample(file, record["present"], record["ambiguous"], record["captions"]))
    wanted = {int(sample.file.stem) for sample in samples}
    for annotation in instances["annotations"]:
        if annotation["image_id"] in wanted:
            by_image.setdefault(annotation["image_id"], []).append(
                (names[annotation["category_id"]], annotation["area"]))
    for sample in samples:
        sample.instances = by_image.get(int(sample.file.stem), [])
    return corpus["categories"], supercategories, samples


def noun(category: str) -> str:
    if category in PLURAL_CATEGORIES:
        return category
    return f"{'an' if category[0] in 'aeiou' else 'a'} {category}"


def mentioned(caption: str, categories: list[str]) -> list[str]:
    """Categories the caption names, by whole word match on the name, its synonyms, and a plural s or es."""
    text = caption.lower()
    found: list[str] = []

    for category in categories:
        words = "|".join(re.escape(word) for word in SYNONYMS.get(category, [category]))
        if re.search(rf"\b(?:{words})(?:s|es)?\b", text):
            found.append(category)
    return found


def sample_absent(sample: Sample, categories: list[str], excluded: set[str], seed: int) -> list[str]:
    pool = [category for category in categories if category not in excluded]
    return random.Random(f"{seed}-{sample.file.name}").sample(pool, ABSENT_PER_IMAGE)


def derangement(count: int, seed: int) -> list[int]:
    order = list(range(count))
    rng = random.Random(seed)

    while count > 1 and any(index == target for index, target in enumerate(order)):
        rng.shuffle(order)
    return order


# ---------------------------------------------------------------- image operations


def gaussian_blur(image, sigma: float):
    """Separable Gaussian blur of a [1, H, W, 3] image with reflected edges."""
    import torch
    from torch.nn.functional import conv2d, pad

    radius = 0
    kernel = None
    channels = image.movedim(-1, 1)

    if sigma <= 0.0:
        return image
    radius = math.ceil(3.0 * sigma)
    kernel = torch.exp(-(torch.arange(-radius, radius + 1, dtype=image.dtype) ** 2) / (2.0 * sigma ** 2))
    kernel = (kernel / kernel.sum()).repeat(3, 1, 1, 1)
    channels = conv2d(pad(channels, (radius, radius, 0, 0), mode="reflect"), kernel, groups=3)
    channels = conv2d(pad(channels, (0, 0, radius, radius), mode="reflect"), kernel.transpose(2, 3), groups=3)
    return channels.movedim(1, -1)


# ---------------------------------------------------------------- metrics


def auc(positives: list[float], negatives: list[float]) -> float:
    """ROC AUC as the Mann-Whitney statistic, ties counting half."""
    from scipy.stats import rankdata

    ranks = None
    positive_rank_sum = 0.0

    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(positives + negatives)
    positive_rank_sum = float(ranks[: len(positives)].sum())
    return (positive_rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))


def image_groups(items: list[dict], label: str, value: str) -> list[tuple[list[float], list[float]]]:
    """Per image, in first seen order, the values of its positive items and of its negative items."""
    groups: dict[str, tuple[list[float], list[float]]] = {}

    for item in items:
        positives, negatives = groups.setdefault(item["file"], ([], []))
        (positives if item[label] else negatives).append(item[value])
    return list(groups.values())


def pooled_auc(groups: list[tuple[list[float], list[float]]], indices: list[int]) -> float:
    return auc([p for i in indices for p in groups[i][0]], [p for i in indices for p in groups[i][1]])


def bootstrap(statistic: Callable[[list[int]], float], count: int, seed: int):
    """The statistic over image resamples. One seed draws the same resamples for every statistic,
    so two arrays from one seed subtract into a paired difference."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.array([statistic(rng.integers(0, count, count).tolist()) for _ in range(BOOTSTRAP_ROUNDS)])


def interval(values) -> str:
    import numpy as np

    low, high = np.nanpercentile(values, (2.5, 97.5))
    return f"[{low:.4f}, {high:.4f}]"


def quantiles(values: list[float]) -> str:
    import numpy as np

    if not values:
        return "n/a"
    return " ".join(f"{p:.4f}" for p in np.percentile(values, QUANTILES))


def failure_line(title: str, items: list[dict], label: str, positive: bool) -> str:
    """The positives scored lowest, or the negatives scored highest."""
    rows = sorted((item for item in items if item[label] == positive), key=lambda item: item["p"],
                  reverse=not positive)[:SHOWN_FAILURES]
    return f"  {title}: " + ", ".join(f"{item['file']} {item['category']} {item['p']:.4f}" for item in rows)


def threshold_rows(positives: list[float], negatives: list[float], thresholds: tuple[float, ...]) -> list[str]:
    rows = []

    for threshold in thresholds:
        true_positive = sum(p >= threshold for p in positives)
        true_negative = sum(p < threshold for p in negatives)
        accuracy = (true_positive + true_negative) / (len(positives) + len(negatives))
        rows.append(f"  {threshold:<7} acc {accuracy:.3f}  TPR {true_positive / len(positives):.3f}  "
                    f"TNR {true_negative / len(negatives):.3f}")
    return rows


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman's rho, 0 when either side is constant, since a flat readout orders nothing."""
    from scipy.stats import spearmanr

    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    return float(spearmanr(x, y).statistic)


def pairwise_order(scores: list[float]) -> tuple[int, float]:
    """Pairs, and the fraction where the less degraded image scores higher. Scores run least degraded first."""
    pairs = [(scores[i], scores[j]) for i in range(len(scores)) for j in range(i + 1, len(scores))]
    right = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs)
    return len(pairs), right


def median(values: list[float]) -> float:
    import numpy as np

    return float(np.median(values)) if values else float("nan")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def separation_lines(title: str, groups: list[tuple[list[float], list[float]]], thresholds: tuple[float, ...],
                     seed: int) -> list[str]:
    positives = [p for group in groups for p in group[0]]
    negatives = [p for group in groups for p in group[1]]
    spread = interval(bootstrap(lambda indices: pooled_auc(groups, indices), len(groups), seed))
    return [
        f"{title}: AUC {auc(positives, negatives):.4f} {spread}  "
        f"({len(positives)} positives, {len(negatives)} negatives)",
        f"  p quantiles {'/'.join(map(str, QUANTILES))}",
        f"    positives  {quantiles(positives)}",
        f"    negatives  {quantiles(negatives)}",
        f"  saturated at p >= {SATURATED}: positives {sum(p >= SATURATED for p in positives)}, "
        f"negatives {sum(p >= SATURATED for p in negatives)}",
        *threshold_rows(positives, negatives, thresholds),
    ]


# ---------------------------------------------------------------- probes


def probe_presence(asker: Asker, samples: list[Sample], images: dict, categories: list[str],
                   seed: int) -> tuple[list[str], dict]:
    """Probes 1 and 2: presence nouls in two wordings, packed, and one noul alone per image."""
    from logit_classifier import NoulQuestion

    items: list[dict] = []
    packed_ms: list[float] = []
    single_ms: list[float] = []
    single_drift: list[float] = []

    for sample in samples:
        absent = sample_absent(sample, categories, set(sample.present) | set(sample.ambiguous), seed)
        asked = [(category, True) for category in sample.present] + [(category, False) for category in absent]
        for wording, template in (("statement", STATEMENT), ("question", QUESTION)):
            questions = {f"q{i}": NoulQuestion(instructions=template.format(noun=noun(category)))
                         for i, (category, _present) in enumerate(asked)}
            answers, ms = asker.ask(f"presence {wording}", "", questions, images[sample.file])
            if wording == "statement":
                packed_ms.append(ms)
                statement_p = answers["q0"].noul
            items += [{"file": sample.file.name, "wording": wording, "category": category, "present": present,
                       "p": answers[f"q{i}"].noul} for i, (category, present) in enumerate(asked)]
        alone = {"q0": NoulQuestion(instructions=STATEMENT.format(noun=noun(asked[0][0])))}
        answers, ms = asker.ask("presence single", "", alone, images[sample.file])
        single_ms.append(ms)
        single_drift.append(abs(answers["q0"].noul - statement_p))

    lines = []
    groups = {}
    for wording, template in (("statement", STATEMENT), ("question", QUESTION)):
        rows = [item for item in items if item["wording"] == wording]
        groups[wording] = image_groups(rows, "present", "p")
        lines += separation_lines(f"{wording} {template!r}", groups[wording], PRESENCE_THRESHOLDS, seed)
        lines.append(failure_line("lowest positives", rows, "present", True))
        lines.append(failure_line("highest negatives", rows, "present", False))
        lines.append(f"  mean {asker.mean_ms(f'presence {wording}'):.0f} ms per packed request")
    difference = (bootstrap(lambda indices: pooled_auc(groups["question"], indices), len(samples), seed)
                  - bootstrap(lambda indices: pooled_auc(groups["statement"], indices), len(samples), seed))
    questions_per_request = len(items) / 2 / len(samples)
    lines += [
        f"AUC question minus statement, paired: {mean(difference.tolist()):+.4f} {interval(difference)}",
        "cost of one question alone (probe 2):",
        f"  packed, {questions_per_request:.1f} questions: {mean(packed_ms):.0f} ms per request",
        f"  one question alone:        {mean(single_ms):.0f} ms per request",
        f"  each added question:       {(mean(packed_ms) - mean(single_ms)) / (questions_per_request - 1):.1f} ms",
        f"  |p alone - p packed| max {max(single_drift):.2e}",
    ]
    return lines, {"items": items, "single_ms": single_ms, "packed_ms": packed_ms}


def router_criteria(supercategories: dict[str, str]) -> dict[str, dict]:
    """Two criteria sets: bare supercategory names, and each name described by its member categories."""
    members: dict[str, list[str]] = {}

    for category, supercategory in supercategories.items():
        members.setdefault(supercategory, []).append(category)
    return {
        "names": dict.fromkeys(sorted(members)),
        "members": {name: ", ".join(members[name]) for name in sorted(members)},
    }


def probe_router(asker: Asker, samples: list[Sample], images: dict,
                 supercategories: dict[str, str]) -> tuple[list[str], dict]:
    """Probe 3: choose the supercategory of the largest annotated object."""
    from logit_classifier import ChoiceQuestion

    items: list[dict] = []
    lines: list[str] = []
    variants = router_criteria(supercategories)

    for sample in samples:
        largest = max(sample.instances, key=lambda instance: instance[1])[0]
        truth = supercategories[largest]
        present_supercategories = sorted({supercategories[category] for category in sample.present})
        for variant, criteria in variants.items():
            question = {"q": ChoiceQuestion(instructions=ROUTER_INSTRUCTIONS, criteria=criteria)}
            answers, _ms = asker.ask(f"router {variant}", "", question, images[sample.file])
            answer = answers["q"]
            items.append({"file": sample.file.name, "variant": variant, "largest": largest, "truth": truth,
                          "present_supercategories": present_supercategories,
                          "choice": answer.choice, "confidence": answer.confidence, "abstain": answer.abstain,
                          "probabilities": answer.probabilities})

    truths = [item["truth"] for item in items if item["variant"] == "names"]
    majority = max(set(truths), key=truths.count)
    lines.append(f"majority class {majority!r}: {truths.count(majority) / len(truths):.3f} of images")
    for variant in variants:
        rows = [item for item in items if item["variant"] == variant]
        right = [item for item in rows if item["choice"] == item["truth"]]
        wrong = [item for item in rows if item["choice"] != item["truth"]]
        confident = [item for item in rows if item["confidence"] >= 0.5]
        confident_right = sum(item["choice"] == item["truth"] for item in confident)
        lenient = sum(item["choice"] in item["present_supercategories"] for item in rows)
        lines += [
            f"criteria {variant}: top-1 {len(right) / len(rows):.3f} ({len(right)}/{len(rows)}), "
            f"mean {asker.mean_ms(f'router {variant}'):.0f} ms",
            f"  choice is the supercategory of any object covering >= 1%: {lenient / len(rows):.3f}",
            f"  AUC of confidence, right against wrong: "
            f"{auc([i['confidence'] for i in right], [i['confidence'] for i in wrong]):.3f}",
            f"  confidence right {mean([i['confidence'] for i in right]):.3f}  "
            f"wrong {mean([i['confidence'] for i in wrong]):.3f}",
            f"  abstain    right {mean([i['abstain'] for i in right]):.3f}  "
            f"wrong {mean([i['abstain'] for i in wrong]):.3f}",
            f"  confidence >= 0.5: accuracy {confident_right / max(1, len(confident)):.3f} "
            f"on {len(confident)}/{len(rows)} images",
            "  wrong: " + ", ".join(f"{i['file']} {i['truth']}->{i['choice']}" for i in wrong),
        ]
    return lines, {"items": items}


def degrade_questions() -> dict:
    from logit_classifier import NoulQuestion, ScoreQuestion

    return {
        "sharp_score": ScoreQuestion(instructions=SHARP_SCORE[0], criteria=SHARP_SCORE[1]),
        "sharp_noul": NoulQuestion(instructions=SHARP_NOUL),
        "bright_score": ScoreQuestion(instructions=BRIGHT_SCORE[0], criteria=BRIGHT_SCORE[1]),
        "bright_noul": NoulQuestion(instructions=BRIGHT_NOUL),
    }


def degrade_series(image, series: str) -> list:
    """Five images ordered least degraded first."""
    from logit_tagger.tagging import VL_MAX_PIXELS

    scale = math.sqrt(image.shape[1] * image.shape[2] / VL_MAX_PIXELS)

    if series == "blur":
        return [gaussian_blur(image, sigma * scale) for sigma in BLUR_SIGMAS]
    return [image * factor for factor in DARK_FACTORS]


def probe_degrade(asker: Asker, samples: list[Sample], images: dict) -> tuple[list[str], dict]:
    """Probe 4: does the score float, or a noul, order one image's degradations?"""
    items: list[dict] = []
    lines: list[str] = []
    questions = degrade_questions()
    readouts = {"sharp_score": "score", "sharp_noul": "noul", "bright_score": "score", "bright_noul": "noul"}
    on_axis = {"blur": ("sharp_score", "sharp_noul"), "dark": ("bright_score", "bright_noul")}
    levels = list(range(len(BLUR_SIGMAS)))

    for sample in samples[:DEGRADE_IMAGES]:
        for series in on_axis:
            for level, degraded in enumerate(degrade_series(images[sample.file], series)):
                answers, _ms = asker.ask("degrade", "", questions, degraded)
                items.append({"file": sample.file.name, "series": series, "level": level,
                              **{qid: getattr(answers[qid], readout) for qid, readout in readouts.items()}})

    for series, measures in on_axis.items():
        files = sorted({item["file"] for item in items})
        # One list per image and question, least degraded first, so a perfect readout falls along it.
        blocks = [{qid: [item[qid] for item in items if item["file"] == file and item["series"] == series]
                   for qid in questions} for file in files]
        lines.append(f"{series} series, {len(blocks)} images, {len(levels)} levels each:")
        for qid in questions:
            correlations = [-spearman(levels, block[qid]) for block in blocks]
            if qid not in measures:
                lines.append(f"  off-axis {qid:<12} Spearman mean {mean(correlations):+.3f}")
                continue
            pair_counts = [pairwise_order(block[qid]) for block in blocks]
            pair_accuracy = sum(right for _n, right in pair_counts) / sum(n for n, _right in pair_counts)
            perfect = sum(value > 0.999 for value in correlations)
            lines.append(f"  {qid:<12} Spearman mean {mean(correlations):+.3f}  perfect order {perfect}/{len(blocks)}"
                         f"  pairwise {pair_accuracy:.3f}")
            lines.append("    per level mean " + " ".join(f"{mean([b[qid][lv] for b in blocks]):.3f}"
                                                         for lv in levels))
    lines.append(f"mean {asker.mean_ms('degrade'):.0f} ms per request of 2 scores and 2 nouls")
    return lines, {"items": items}


def count_bin(count: int) -> int:
    return max(index for index, floor in enumerate(COUNT_BIN_FLOORS) if count >= floor)


def probe_count(asker: Asker, samples: list[Sample], images: dict, seed: int) -> tuple[list[str], dict]:
    """Probe 5: the object count score against the COCO instance count."""
    from logit_classifier import ScoreQuestion

    items: list[dict] = []
    question = {"q": ScoreQuestion(instructions=COUNT_SCORE[0], criteria=COUNT_SCORE[1])}

    for sample in samples:
        answers, _ms = asker.ask("count", "", question, images[sample.file])
        count = len(sample.instances)
        items.append({"file": sample.file.name, "instances": count, "bin": count_bin(count),
                      "score": answers["q"].score, "probabilities": answers["q"].probabilities})

    counts = [item["instances"] for item in items]
    scores = [item["score"] for item in items]
    spread = bootstrap(lambda indices: spearman([counts[i] for i in indices], [scores[i] for i in indices]),
                       len(items), seed)
    levels = [round(score) for score in scores]
    exact = sum(level == item["bin"] for level, item in zip(levels, items, strict=True))
    near = sum(abs(level - item["bin"]) <= 1 for level, item in zip(levels, items, strict=True))
    lines = [
        f"Spearman(instances, score) {spearman(counts, scores):+.3f} {interval(spread)} over {len(items)} images",
        f"rounded score in the true bin {exact / len(items):.3f}, within one bin {near / len(items):.3f}",
        "true bin histogram " + str([sum(item["bin"] == b for item in items) for b in range(5)]),
        "rounded score histogram " + str([levels.count(b) for b in range(5)]),
        f"mean {asker.mean_ms('count'):.0f} ms",
    ]
    return lines, {"items": items}


def probe_text(asker: Asker, samples: list[Sample], categories: list[str], seed: int) -> tuple[list[str], dict]:
    """Probe 6: text nouls about the first caption, with no image."""
    from logit_classifier import NoulQuestion

    items: list[dict] = []

    for sample in samples:
        caption = sample.captions[0]
        named = mentioned(caption, categories)
        positives = [category for category in sample.present if category in named]
        excluded = set(sample.present) | set(sample.ambiguous) | set(named)
        asked = [(category, True) for category in positives] + \
                [(category, False) for category in sample_absent(sample, categories, excluded, seed)]
        questions = {f"q{i}": NoulQuestion(instructions=TEXT_STATEMENT.format(noun=noun(category)))
                     for i, (category, _mentioned) in enumerate(asked)}
        answers, _ms = asker.ask("text", caption, questions)
        items += [{"file": sample.file.name, "caption": caption, "category": category, "mentioned": flag,
                   "p": answers[f"q{i}"].noul} for i, (category, flag) in enumerate(asked)]

    lines = separation_lines(f"text {TEXT_STATEMENT!r}", image_groups(items, "mentioned", "p"), TEXT_THRESHOLDS,
                             seed)
    lines.append(failure_line("lowest positives", items, "mentioned", True))
    lines.append(failure_line("highest negatives", items, "mentioned", False))
    lines.append(f"  mean {asker.mean_ms('text'):.0f} ms per packed request")
    lines.append("  missed at 0.5: " + ", ".join(f"{i['category']} in {i['caption']!r}"
                                                  for i in items if i["mentioned"] and i["p"] < 0.5))
    return lines, {"items": items}


def probe_adherence(asker: Asker, samples: list[Sample], images: dict, categories: list[str],
                    seed: int) -> tuple[list[str], dict]:
    """Probe 7: one caption noul, and the mean and min of its category nouls, on matched and swapped captions."""
    from logit_classifier import NoulQuestion

    items: list[dict] = []
    swapped = derangement(len(samples), seed)

    for index, sample in enumerate(samples):
        for matched, source in ((True, sample), (False, samples[swapped[index]])):
            caption = source.captions[0]
            named = mentioned(caption, categories)
            questions = {"match": NoulQuestion(instructions=MATCH_STATEMENT.format(caption=caption))}
            questions |= {f"c{i}": NoulQuestion(instructions=STATEMENT.format(noun=noun(category)))
                          for i, category in enumerate(named)}
            answers, _ms = asker.ask("adherence", "", questions, images[sample.file])
            contains = [answers[f"c{i}"].noul for i in range(len(named))]
            items.append({"file": sample.file.name, "caption": caption, "matched": matched,
                          "match_p": answers["match"].noul, "categories": dict(zip(named, contains, strict=True)),
                          "mean_p": mean(contains) if contains else None, "min_p": min(contains, default=None)})

    # An image is covered only when both its captions name a category, so each covered image holds a pair.
    uncovered = {item["file"] for item in items if item["mean_p"] is None}
    covered = [item for item in items if item["file"] not in uncovered]
    lines = []
    resampled = {}
    for label, signal, rows in (("match_p all", "match_p", items), ("match_p", "match_p", covered),
                                ("mean_p", "mean_p", covered), ("min_p", "min_p", covered)):
        groups = image_groups(rows, "matched", signal)
        positives = [item[signal] for item in rows if item["matched"]]
        negatives = [item[signal] for item in rows if not item["matched"]]
        right = sum(p >= 0.5 for p in positives) + sum(p < 0.5 for p in negatives)
        resampled[label] = bootstrap(lambda indices, groups=groups: pooled_auc(groups, indices), len(groups), seed)
        lines.append(f"{label:<11} AUC {auc(positives, negatives):.4f} {interval(resampled[label])} "
                     f"on {len(groups)} images  median matched {median(positives):.4f} "
                     f"swapped {median(negatives):.4f}  acc at 0.5 {right / len(rows):.3f}")
    difference = resampled["match_p"] - resampled["mean_p"]
    lines.append(f"match_p minus mean_p on the covered images, paired: {mean(difference.tolist()):+.4f} "
                 f"{interval(difference)}")
    lines.append(f"mean {asker.mean_ms('adherence'):.0f} ms per request")
    worst = sorted((item for item in items if item["matched"]), key=lambda item: item["match_p"])[:3]
    lines.append("lowest matched match_p: " + "; ".join(f"{i['file']} {i['match_p']:.4f} {i['caption']!r}"
                                                        for i in worst))
    return lines, {"items": items}


# ---------------------------------------------------------------- main


def main() -> None:
    ab_env.bootstrap()
    from logit_classifier import Classifier, Config, NoulQuestion
    from logit_classifier.backends.comfy_clip import ComfyClipBackend

    from logit_tagger.tagging import resize_for_vl

    args = parse_args()
    out_path = args.out or args.corpus.parent / "question_shapes.json"
    categories, supercategories, samples = load_corpus(args.corpus, args.limit)
    missing = [str(sample.file) for sample in samples if not sample.file.is_file()]
    record: dict = {"clip": args.clip, "seed": args.seed, "images": len(samples), "probes": {}}

    if missing:
        raise SystemExit(f"corpus images not found: {', '.join(missing)}")
    images = {sample.file: resize_for_vl(load_image(sample.file)) for sample in samples}
    backend = ComfyClipBackend(ab_env.load_clip(args.clip))
    asker = Asker(Classifier(Config(use_prior_debias=False, calibration_path=None), backend))
    # The first call pays CUDA and kernel warmup, which no probe should carry.
    asker.ask("warmup", "", {"q": NoulQuestion(instructions="warmup")}, images[samples[0].file])

    runs = {
        "presence": lambda: probe_presence(asker, samples, images, categories, args.seed),
        "router": lambda: probe_router(asker, samples, images, supercategories),
        "degrade": lambda: probe_degrade(asker, samples, images),
        "count": lambda: probe_count(asker, samples, images, args.seed),
        "text": lambda: probe_text(asker, samples, categories, args.seed),
        "adherence": lambda: probe_adherence(asker, samples, images, categories, args.seed),
    }
    for name in args.probes:
        lines, result = runs[name]()
        record["probes"][name] = result
        print(f"== {name}", *lines, "", sep="\n", flush=True)
    record["timings_ms"] = asker.timings
    out_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"raw results: {out_path}")


if __name__ == "__main__":
    main()
