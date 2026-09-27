"""Score ab_tagger_compare.py results against blind judges' verdicts, arms side by side.

Each VERDICTS file maps an image key to {tag: "yes" | "no" | "vague" | "unsure"}. Later files add pairs.
precision = yes / (yes + no + vague). wrong = no / judged, where judged counts every verdict.
The 95% intervals come from a bootstrap over images. Each delta row pairs an arm with the baseline image by image.
Usage: python ab_tagger_score.py [VERDICTS.json ...] [--results results.json ...] [--rename new=N0]
       [--arms N0 N1 N2 old] [--baseline N0] [--samples 10000]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

VERDICTS = ("yes", "no", "vague", "unsure")
HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "out" / "tagger-ab" / "results.json"
DEFAULT_VERDICTS = HERE / "data" / "tagger-ab" / "verdicts.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("verdicts", type=Path, nargs="*", default=[DEFAULT_VERDICTS],
                        help="judges' {key: {tag: verdict}} files, the first verdict of a pair counting")
    parser.add_argument("--results", type=Path, nargs="+", default=[DEFAULT_RESULTS],
                        help="ab_tagger_compare.py results files, their arm names unique across files")
    parser.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW", help="rename an arm, such as new=N0")
    parser.add_argument("--arms", nargs="*", default=None, help="the arms to print, in order, all by default")
    parser.add_argument("--baseline", default=None, help="the arm every delta row pairs with, the first by default")
    parser.add_argument("--samples", type=int, default=10000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def merge_results(paths: list[Path], renames: dict[str, str]) -> dict[str, dict[str, dict]]:
    """{condition: {arm: {key: record}}} over every file. Two files naming one arm in one condition is an error."""
    merged: dict[str, dict[str, dict]] = {}

    for path in paths:
        for condition, arms in json.loads(path.read_text(encoding="utf-8")).items():
            for arm, images in arms.items():
                name = renames.get(arm, arm)
                if name in merged.setdefault(condition, {}):
                    raise SystemExit(f"arm {name!r} of {condition} appears twice, the second time in {path}. "
                                     "Rename one with --rename.")
                merged[condition][name] = images
    return merged


def merge_verdicts(paths: list[Path]) -> tuple[dict[str, dict[str, str]], int]:
    """Every file's verdicts, the first verdict of a pair kept, and how many later verdicts disagreed with it."""
    merged: dict[str, dict[str, str]] = {}
    conflicts = 0

    for path in paths:
        for key, verdicts in json.loads(path.read_text(encoding="utf-8")).items():
            known = merged.setdefault(key, {})
            for tag, verdict in verdicts.items():
                conflicts += tag in known and known[tag] != verdict
                known.setdefault(tag, verdict)
    return merged, conflicts


def count_image(tags: list[str], verdicts: dict[str, str]) -> dict[str, int]:
    """One image's verdict counts. A tag the judge never graded counts as unjudged."""
    counts = dict.fromkeys((*VERDICTS, "unjudged"), 0)

    for tag in tags:
        verdict = verdicts.get(tag, "unjudged")
        if verdict not in counts:
            raise SystemExit(f"verdict {verdict!r} for tag {tag!r} is none of {VERDICTS}")
        counts[verdict] += 1
    return counts


def precision(rows: list[dict[str, int]]) -> float | None:
    graded = sum(row["yes"] + row["no"] + row["vague"] for row in rows)

    return sum(row["yes"] for row in rows) / graded if graded else None


def correct_per_image(rows: list[dict[str, int]]) -> float:
    return sum(row["yes"] for row in rows) / len(rows)


def interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)

    if not ordered:
        return float("nan"), float("nan")
    return ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]


def bootstrap(arms: list[list[dict[str, int]]], samples: int, rng: random.Random) -> dict[str, tuple[float, float]]:
    """95% intervals of precision and correct per image, or of the first arm minus the second when two are given."""
    size = len(arms[0])
    signs = (1.0,) if len(arms) == 1 else (1.0, -1.0)
    stats: dict[str, list[float]] = {"precision": [], "correct": []}

    for _ in range(samples):
        picks = [rng.randrange(size) for _ in range(size)]
        drawn = [[rows[index] for index in picks] for rows in arms]
        precisions = [precision(rows) for rows in drawn]
        if None in precisions:
            continue
        stats["precision"].append(sum(sign * value for sign, value in zip(signs, precisions, strict=True)))
        stats["correct"].append(sum(sign * correct_per_image(rows) for sign, rows in zip(signs, drawn, strict=True)))
    return {name: interval(values) for name, values in stats.items()}


def mean_seconds(records: list[dict], field: str) -> float:
    return sum(record.get(field, 0.0) for record in records) / len(records) / 1000.0


def arm_line(condition: str, arm: str, records: list[dict], rows: list[dict[str, int]],
             ci: dict[str, tuple[float, float]]) -> str:
    judged = sum(row[verdict] for row in rows for verdict in VERDICTS)
    value = precision(rows)
    shown = "n/a" if value is None else f"{value:.3f}"
    low, high = ci["precision"]
    correct_low, correct_high = ci["correct"]
    times = " ".join(f"{mean_seconds(records, field):6.2f}" for field in ("prompt_ms", "propose_ms", "verify_ms", "total_ms"))

    return (f"{condition:16} {arm:6} {len(rows):4d} {sum(len(r['tags']) for r in records) / len(records):8.2f} "
            f"{shown:>5} [{low:.3f}, {high:.3f}] {sum(row['no'] for row in rows) / judged if judged else 0.0:6.3f} "
            f"{correct_per_image(rows):5.2f} [{correct_low:5.2f}, {correct_high:5.2f}] "
            f"{sum(row['unsure'] for row in rows):6d} {sum(row['unjudged'] for row in rows):8d}   {times}")


def delta_line(condition: str, arm: str, baseline: str, rows: list[dict[str, int]], base_rows: list[dict[str, int]],
               ci: dict[str, tuple[float, float]]) -> str:
    arm_precision = precision(rows)
    base_precision = precision(base_rows)
    shown = "n/a" if arm_precision is None or base_precision is None else f"{arm_precision - base_precision:+.3f}"
    correct = correct_per_image(rows) - correct_per_image(base_rows)

    return (f"{condition:16} {arm} minus {baseline}, paired: precision {shown} "
            f"[{ci['precision'][0]:+.3f}, {ci['precision'][1]:+.3f}], "
            f"correct/img {correct:+.2f} [{ci['correct'][0]:+.2f}, {ci['correct'][1]:+.2f}]")


def main() -> None:
    args = parse_args()
    renames = dict(pair.split("=", 1) for pair in args.rename)
    results = merge_results(args.results, renames)
    verdicts, conflicts = merge_verdicts(args.verdicts)
    rng = random.Random(args.seed)
    order = args.arms or list(dict.fromkeys(arm for arms in results.values() for arm in arms))
    baseline = args.baseline or order[0]

    if conflicts:
        print(f"{conflicts} pairs got a different verdict in a later file. The first verdict counts.")
    print(f"{'condition':16} {'arm':6} {'imgs':>4} {'tags/img':>8} {'precision [95% CI]':>21} {'wrong':>6} "
          f"{'correct/img [95% CI]':>20} {'unsure':>6} {'unjudged':>8}   s/img: prompt propose verify total")
    for condition, arms in results.items():
        present = [arm for arm in order if arm in arms]
        keys = sorted(arms[baseline] if baseline in arms else arms[present[0]])
        rows_of: dict[str, list[dict[str, int]]] = {}
        for arm in present:
            records = [arms[arm][key] for key in keys]
            rows_of[arm] = [count_image(arms[arm][key]["tags"], verdicts.get(key, {})) for key in keys]
            print(arm_line(condition, arm, records, rows_of[arm], bootstrap([rows_of[arm]], args.samples, rng)))
        for arm in present:
            if arm == baseline or baseline not in rows_of:
                continue
            ci = bootstrap([rows_of[arm], rows_of[baseline]], args.samples, rng)
            print(delta_line(condition, arm, baseline, rows_of[arm], rows_of[baseline], ci))


if __name__ == "__main__":
    main()
