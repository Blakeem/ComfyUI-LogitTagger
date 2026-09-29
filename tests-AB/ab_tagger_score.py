"""Score ab_tagger_compare.py results against blind judges' verdicts, arms side by side, or replay stricter arms.

Each VERDICTS file maps an image key to {tag: "yes" | "no" | "vague" | "unsure"}. Later files add pairs.
Each kept tag of an image counts once: correct when judged yes, wrong when judged no, vague when judged vague,
unsure when judged unsure, and unjudged when no file grades it. A count prints as its mean per image.
precision = correct / (correct + wrong + vague), pooled over an arm's images.
The 95% intervals come from a bootstrap over images. Each delta row pairs an arm with the baseline image by image,
over the images both hold.
--replay re-derives each ARMS.json arm's kept lists from one superset run that ab_tagger_compare.py recorded with
--record-thing-scores, reading its settings<suffix>.json beside its results<suffix>.json. ARMS.json maps an arm to
TagSettings field overrides, the shape --variants takes, with verify_thresholds as a [proposed, declared] list.
Settings and arms recorded with verify_threshold and prompt_only_threshold replay as recorded, with their
thing_threshold read as 1 - thing_threshold.
The replay first reproduces the superset's own kept lists.
Usage: python ab_tagger_score.py [VERDICTS.json ...] [--results results.json ...] [--rename new=N0]
       [--arms N0 N1 N2 old] [--baseline N0] [--samples 10000]
       python ab_tagger_score.py [VERDICTS.json ...] --replay ARMS.json --results SUPERSET.json
       [--superset-arm NAME] [--baseline NAME]
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import itertools
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ab_tagger_compare import arm_settings, recorded_p_thing, settings_path_of

VERDICTS = ("yes", "no", "vague", "unsure")
HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "out" / "tagger-ab" / "results.json"
DEFAULT_VERDICTS = HERE / "data" / "tagger-ab" / "verdicts.json"
COUNTS = {"correct": ("yes",), "wrong": ("no",), "vague": ("vague",), "unsure": ("unsure",),
          "unjudged": ("unjudged",)}
DELTAS = {"correct": ("yes",), "wrong": ("no",), "wrong+vague": ("no", "vague")}
TIMES = ("prompt_ms", "propose_ms", "verify_ms", "total_ms")
CAPS = ("propose_cap", "merge_cap")
# The fields the records re-derive. thing_threshold is not one, since the recorded prompt tags already passed the
# prompt stage's thing check at the superset's value.
REPLAYED_FIELDS = frozenset({*CAPS, "category_nouns", "thing_check", "verify_thresholds", "subsets_rule", "order"})

Rows = list[dict[str, int]]


class NotRecorded(Exception):
    """The superset's records lack what an arm needs to re-derive one image."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("verdicts", type=Path, nargs="*", default=[DEFAULT_VERDICTS],
                        help="judges' {key: {tag: verdict}} files, the first verdict of a pair counting")
    parser.add_argument("--results", type=Path, nargs="+", default=[DEFAULT_RESULTS],
                        help="ab_tagger_compare.py results files, their arm names unique across files")
    parser.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW", help="rename an arm, such as new=N0")
    parser.add_argument("--arms", nargs="*", default=None, help="the arms to print, in order, all by default")
    parser.add_argument("--baseline", default=None,
                        help="the arm every delta row pairs with, the superset with --replay, else the first")
    parser.add_argument("--samples", type=int, default=10000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path, default=None, metavar="ARMS.json",
                        help="{arm: {TagSettings field: value}} to re-derive from the one --results superset run")
    parser.add_argument("--superset-arm", default=None,
                        help="the superset arm of --results, needed when it holds more than one arm")
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


def precision(rows: Rows) -> float | None:
    graded = sum(row["yes"] + row["no"] + row["vague"] for row in rows)

    return sum(row["yes"] for row in rows) / graded if graded else None


def mean_count(verdicts: tuple[str, ...]) -> Callable[[Rows], float]:
    """The metric of how many tags per image got one of verdicts."""
    return lambda rows: sum(row[verdict] for row in rows for verdict in verdicts) / len(rows)


ARM_METRICS: dict[str, Callable[[Rows], float | None]] = {
    "precision": precision, **{name: mean_count(verdicts) for name, verdicts in COUNTS.items()}}
DELTA_METRICS: dict[str, Callable[[Rows], float | None]] = {
    "precision": precision, **{name: mean_count(verdicts) for name, verdicts in DELTAS.items()}}


def metric_values(rows: Rows, metrics: dict[str, Callable[[Rows], float | None]]) -> dict[str, float | None]:
    return {name: metric(rows) for name, metric in metrics.items()}


def paired_deltas(rows: Rows, base_rows: Rows,
                  metrics: dict[str, Callable[[Rows], float | None]]) -> dict[str, float | None]:
    """Each metric of rows minus the same metric of base_rows, None when either has nothing graded."""
    values = metric_values(rows, metrics)
    base_values = metric_values(base_rows, metrics)

    return {name: None if values[name] is None or base_values[name] is None else values[name] - base_values[name]
            for name in metrics}


def paired_rows(rows_of: dict[str, dict[str, int]], base_rows_of: dict[str, dict[str, int]]) -> tuple[Rows, Rows]:
    """Both arms' rows over the image keys both hold, in key order, so index i is one image in each."""
    shared = sorted(rows_of.keys() & base_rows_of.keys())

    return [rows_of[key] for key in shared], [base_rows_of[key] for key in shared]


def interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)

    if not ordered:
        return float("nan"), float("nan")
    return ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]


def bootstrap(arms: list[Rows], metrics: dict[str, Callable[[Rows], float | None]], samples: int,
              rng: random.Random) -> dict[str, tuple[float, float]]:
    """95% intervals of each metric, or of the first arm minus the second when two are given.

    A resample where an arm has nothing graded gives no precision, so precision skips it.
    """
    size = len(arms[0])
    signs = (1.0,) if len(arms) == 1 else (1.0, -1.0)
    stats: dict[str, list[float]] = {name: [] for name in metrics}
    picks: list[int] = []
    drawn: list[Rows] = []
    values: list[float | None] = []

    for _ in range(samples):
        picks = [rng.randrange(size) for _ in range(size)]
        drawn = [[rows[index] for index in picks] for rows in arms]
        for name, metric in metrics.items():
            values = [metric(rows) for rows in drawn]
            if None not in values:
                stats[name].append(sum(sign * value for sign, value in zip(signs, values, strict=True)))
    return {name: interval(found) for name, found in stats.items()}


def shown(value: float | None, bounds: tuple[float, float], spec: str) -> str:
    """value and its interval in one format spec, n/a when nothing was graded."""
    low, high = bounds

    if value is None:
        return "n/a"
    return f"{value:{spec}} [{low:{spec}}, {high:{spec}}]"


def mean_seconds(records: list[dict], field: str) -> str:
    """The mean of a timing field in seconds, n/a for records without it, such as replayed ones."""
    timed = [record[field] for record in records if field in record]

    return f"{sum(timed) / len(timed) / 1000.0:6.2f}" if timed else f"{'n/a':>6}"


def header_line() -> str:
    counts = " ".join(f"{name + '/img [95% CI]':>21}" for name in COUNTS)

    return (f"{'condition':16} {'arm':6} {'imgs':>4} {'tags/img':>8} {'precision [95% CI]':>21} {counts}   "
            "s/img: prompt propose verify total")


def arm_line(condition: str, arm: str, records: list[dict], rows: Rows, ci: dict[str, tuple[float, float]]) -> str:
    values = metric_values(rows, ARM_METRICS)
    tags_per_image = sum(len(record["tags"]) for record in records) / len(records)
    counts = " ".join(f"{shown(values[name], ci[name], '5.2f'):>21}" for name in COUNTS)
    times = " ".join(mean_seconds(records, field) for field in TIMES)

    return (f"{condition:16} {arm:6} {len(rows):4d} {tags_per_image:8.2f} "
            f"{shown(values['precision'], ci['precision'], '.3f'):>21} {counts}   {times}")


def delta_line(condition: str, arm: str, baseline: str, pairs: int, deltas: dict[str, float | None],
               ci: dict[str, tuple[float, float]]) -> str:
    counts = ", ".join(f"{name}/img {shown(deltas[name], ci[name], '+.2f')}" for name in DELTAS)

    return (f"{condition:16} {arm} minus {baseline}, {pairs} paired images: precision "
            f"{shown(deltas['precision'], ci['precision'], '+.3f')}, {counts}")


def condition_lines(condition: str, arms: dict[str, dict], order: list[str], baseline: str,
                    verdicts: dict[str, dict[str, str]], samples: int, rng: random.Random) -> list[str]:
    """One line per arm over its own images, then one delta line per arm paired with the baseline."""
    present = [arm for arm in order if arm in arms]
    rows_of: dict[str, dict[str, dict[str, int]]] = {}
    lines: list[str] = []
    keys: list[str] = []
    rows: Rows = []
    base_rows: Rows = []

    rows_of = {arm: {key: count_image(record["tags"], verdicts.get(key, {})) for key, record in arms[arm].items()}
               for arm in present}
    for arm in present:
        keys = sorted(rows_of[arm])
        if not keys:
            lines.append(f"{condition:16} {arm:6} holds no image")
            continue
        rows = [rows_of[arm][key] for key in keys]
        lines.append(arm_line(condition, arm, [arms[arm][key] for key in keys], rows,
                              bootstrap([rows], ARM_METRICS, samples, rng)))
    for arm in present:
        if arm == baseline or baseline not in rows_of:
            continue
        rows, base_rows = paired_rows(rows_of[arm], rows_of[baseline])
        if not rows:
            lines.append(f"{condition:16} {arm} minus {baseline}: no image in both")
            continue
        lines.append(delta_line(condition, arm, baseline, len(rows), paired_deltas(rows, base_rows, DELTA_METRICS),
                                bootstrap([rows, base_rows], DELTA_METRICS, samples, rng)))
    return lines


def superset_settings(results_path: Path, arm: str) -> Any:
    """The superset arm's TagSettings, from the settings file ab_tagger_compare.py wrote beside its results."""
    from logit_classifier.toolkit.comfyui import TagSettings

    path = settings_path_of(results_path)
    recorded: dict[str, dict[str, Any]] = {}

    if not path.is_file():
        raise SystemExit(f"{path} not found. --replay reads the superset's settings from the settings file "
                         "ab_tagger_compare.py writes beside its results.")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if arm not in recorded:
        raise SystemExit(f"{path} holds no settings for arm {arm!r}, only for {sorted(recorded)}")
    return arm_settings(TagSettings(), arm, recorded[arm], source=str(path))


def covered_settings(superset: Any, arm: str, overrides: dict[str, Any]) -> Any:
    """superset with the arm's overrides. Stops on a field whose value the superset's records cannot re-derive."""
    settings = arm_settings(superset, arm, overrides, source="--replay")
    above = [f"{name}={getattr(settings, name)} above the superset's {getattr(superset, name)}" for name in CAPS
             if getattr(settings, name) > getattr(superset, name)]
    fixed = [field.name for field in dataclasses.fields(settings)
             if field.name not in REPLAYED_FIELDS and getattr(settings, field.name) != getattr(superset, field.name)]

    if above:
        raise SystemExit(f"--replay arm {arm!r} sets {', '.join(above)}. The superset recorded no tag past its "
                         "caps, so rerun it with the higher cap.")
    if fixed:
        raise SystemExit(f"--replay arm {arm!r} changes {', '.join(fixed)} from the superset's value, which its "
                         "records cannot re-derive. Run that arm with ab_tagger_compare.py.")
    return settings


def proposed_tags(reply: str, cap: int) -> list[str]:
    """The tags a propose decode that stops at cap complete tags keeps, from a reply decoded under a higher cap.

    The decode's stop counts every complete item, repeats and overlong ones included, so the cut counts them too.
    """
    from logit_classifier.toolkit.tags import complete_tags, parse_candidates

    # A longer prefix never holds fewer complete tags, so the shortest one holding cap is where the decode stopped.
    end = bisect.bisect_left(range(len(reply) + 1), cap, key=lambda size: len(complete_tags(reply[:size])))

    return parse_candidates(reply[:end], max_candidates=cap)


def replay_image(record: dict[str, Any], settings: Any) -> tuple[str, ...]:
    """The image's kept list under settings, re-derived as the toolkit's tag_image derives it.

    Raises NotRecorded when settings need a reply, a candidate or a score the superset run did not record.
    """
    from logit_classifier.toolkit.tags import drop_subsets, merge_candidates, passed_by_origin

    thresholds = {"model": settings.verify_thresholds.proposed, "both": settings.verify_thresholds.proposed,
                  "prompt": settings.verify_thresholds.declared}
    recorded = {candidate["tag"]: candidate for candidate in record["candidates"]}
    echo_ran = record.get("echo_reply") is not None
    proposed: list[str] = []
    candidates: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    missing: list[str] = []
    p_thing: list[float | None] = []
    is_thing: list[bool] = []
    scores: list[float | None] = []
    verified: tuple[str, ...] = ()
    kept: list[str] = []

    if "prompt_tags" not in record:
        raise NotRecorded("the record holds no prompt_tags. Rerun the superset with ab_tagger_compare.py.")
    proposed = proposed_tags(record["echo_reply"] if echo_ran else record["reply"], settings.propose_cap)
    if (settings.echo_fallback_instruction is not None
            and not merge_candidates(proposed, (), cap=settings.merge_cap, drop=settings.category_nouns).candidates):
        if not echo_ran:
            raise NotRecorded("the arm runs the echo fallback, which the superset never ran on this image")
        proposed = proposed_tags(record["reply"], settings.propose_cap)
    candidates, origins, _dropped = merge_candidates(proposed, record["prompt_tags"], cap=settings.merge_cap,
                                                     drop=settings.category_nouns)
    missing = [tag for tag in candidates if tag not in recorded]
    if missing:
        raise NotRecorded(f"no recorded score for {missing}")
    if settings.thing_check == "all":
        p_thing = [recorded_p_thing(recorded[tag]) for tag in candidates]
        if None in p_thing:
            raise NotRecorded("no recorded thing check score. Record the superset with --record-thing-scores.")
        is_thing = [p >= settings.thing_threshold for p in p_thing]
        candidates = tuple(itertools.compress(candidates, is_thing))
        origins = tuple(itertools.compress(origins, is_thing))
    verified = candidates
    if settings.verify_statement:
        scores = [recorded[tag]["score"] for tag in candidates]
        if None in scores:
            raise NotRecorded("the superset ran without verify, so no candidate carries a verify score")
        verified = passed_by_origin(candidates, origins, scores, thresholds=thresholds)

    kept = drop_subsets(list(verified), rule=settings.subsets_rule)
    if settings.order == "score" and settings.verify_statement:
        # sorted is stable, so tied tags keep candidate order, as in tag_image.
        kept = sorted(kept, key=lambda tag: recorded[tag]["score"], reverse=True)
    return tuple(kept)


def self_check(images_of: dict[str, dict[str, dict]], settings: Any) -> None:
    """Stops at the first image whose replay under the superset's own settings differs from its recorded kept list."""
    kept: tuple[str, ...] = ()

    for condition, images in images_of.items():
        for key in sorted(images):
            try:
                kept = replay_image(images[key], settings)
            except NotRecorded as error:
                raise SystemExit(f"self-check: {condition} {key} does not replay under the superset's own "
                                 f"settings: {error}") from error
            if list(kept) != images[key]["tags"]:
                raise SystemExit(f"self-check: {condition} {key} replays to {list(kept)}, but the superset kept "
                                 f"{images[key]['tags']}. The replay does not reproduce this run, so no arm "
                                 "replayed from it is trusted.")


def replay_arms(images_of: dict[str, dict[str, dict]],
                arms: dict[str, Any]) -> tuple[dict[str, dict[str, dict[str, dict]]], list[str]]:
    """Each arm's replayed records per condition, and one note per image an arm leaves out."""
    replayed: dict[str, dict[str, dict[str, dict]]] = {}
    notes: list[str] = []
    records: dict[str, dict] = {}

    for condition, images in images_of.items():
        for arm, settings in arms.items():
            records = replayed.setdefault(condition, {}).setdefault(arm, {})
            for key in sorted(images):
                try:
                    records[key] = {"tags": list(replay_image(images[key], settings))}
                except NotRecorded as error:
                    notes.append(f"{condition} {arm} leaves out {key}: {error}")
    return replayed, notes


def pick_superset(results: dict[str, dict[str, dict]], name: str | None) -> str:
    held = sorted({arm for arms in results.values() for arm in arms})

    if name is None and len(held) != 1:
        raise SystemExit(f"the superset results hold arms {held}. Name the superset with --superset-arm.")
    if name is not None and name not in held:
        raise SystemExit(f"--superset-arm {name!r} is none of the results' arms {held}")
    return name if name is not None else held[0]


def replay(args: argparse.Namespace, results: dict[str, dict[str, dict]],
           renames: dict[str, str]) -> tuple[str, list[str]]:
    """Adds each --replay arm's records to results. Returns the superset arm and one note per left-out image."""
    originals = {new: old for old, new in renames.items()}
    held = {arm for arms in results.values() for arm in arms}
    superset = ""
    settings = None
    overrides: dict[str, dict[str, Any]] = {}
    arms: dict[str, Any] = {}
    images_of: dict[str, dict[str, dict]] = {}
    replayed: dict[str, dict[str, dict[str, dict]]] = {}
    notes: list[str] = []

    if len(args.results) != 1:
        raise SystemExit(f"--replay takes one --results file, the superset run's, and got {len(args.results)}")
    superset = pick_superset(results, args.superset_arm)
    settings = superset_settings(args.results[0], originals.get(superset, superset))
    overrides = json.loads(args.replay.read_text(encoding="utf-8"))
    if held & overrides.keys():
        raise SystemExit(f"{args.replay} names arms {sorted(held & overrides.keys())} that {args.results[0]} "
                         "already holds. Name each replayed arm apart.")
    arms = {arm: covered_settings(settings, arm, fields) for arm, fields in overrides.items()}
    images_of = {condition: arms_of[superset] for condition, arms_of in results.items() if superset in arms_of}

    self_check(images_of, settings)
    replayed, notes = replay_arms(images_of, arms)
    for condition, records in replayed.items():
        results[condition].update(records)
    return superset, notes


def main() -> None:
    args = parse_args()
    renames = dict(pair.split("=", 1) for pair in args.rename)
    results = merge_results(args.results, renames)
    verdicts, conflicts = merge_verdicts(args.verdicts)
    rng = random.Random(args.seed)
    superset = None
    notes: list[str] = []
    order: list[str] = []
    baseline = ""

    if args.replay is not None:
        superset, notes = replay(args, results, renames)
    order = args.arms or list(dict.fromkeys(arm for arms in results.values() for arm in arms))
    baseline = args.baseline or superset or order[0]

    for note in notes:
        print(note)
    if conflicts:
        print(f"{conflicts} pairs got a different verdict in a later file. The first verdict counts.")
    print(header_line())
    for condition, arms in results.items():
        for line in condition_lines(condition, arms, order, baseline, verdicts, args.samples, rng):
            print(line)


if __name__ == "__main__":
    main()
