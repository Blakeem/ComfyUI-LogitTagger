"""Logit Style Extractor's two paths over a real Qwen3-VL-4B, to decide whether each names the medium and its style.

Each path calls the toolkit function the node calls, with the node's default StyleSettings:
  prompt  prompt_style on the 72 render prompts and the 80 textset prompts, which name things and mostly no medium.
          Each --variants listing lists every prompt once, and each scoring reads p("style") of every listed phrase,
          so --replay derives every listing, scoring and threshold arm offline from one run.
          Its output sets the StyleSettings prompt fields: prompt_instruction, prompt_max_tokens, ungrounded_streak,
          style_question, style_criteria and style_threshold.
  image   style_line on the 72 renders and the 30 COCO photos the Tagger A/B judged, each line timed.
          Its output sets the StyleSettings image fields: image_instruction, image_max_tokens and image_pixels.
A render's medium is its manifest style, and a text names it when every content word of the medium name shares a
form with a word of the text, the match in_prompt grounds a tag with. Every interval is a 95% bootstrap over items.
A run writes data/style/<run>/results.json with every record, the CLIP file name and the library version.
--judge-file writes the run's to_judge.json with no arm or score, for blind judges by data/style/RUBRIC.md.
--merge reads the run's judge-*.json into labels.json, and --replay scores every arm against those labels.
Usage: python tests-AB/ab_style.py --path prompt|image [--run NAME] [--variants V.json] [--clip FILE] [--limit N]
       python tests-AB/ab_style.py --judge-file RUN | --merge RUN | --replay RUN [--seed N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import ab_env
from ab_question_shapes import mean
from ab_release_gates import DEFAULT_DATA, Render, estimate, load_renders, load_textset
from ab_tagger import DEFAULT_CLIP, load_image
from ab_tagger_compare import DEFAULT_COCO, DEFAULT_COCO_HASHES, DEFAULT_MANIFEST, Item, load_conditions

HERE = Path(__file__).resolve().parent
STYLE_ROOT = HERE / "data" / "style"
RESULTS_NAME = "results.json"
JUDGE_NAME = "to_judge.json"
LABELS_NAME = "labels.json"
JUDGE_GLOB = "judge-*.json"
HARNESS_NAME = "ab_style"
PATH_PROMPT = "prompt"
PATH_IMAGE = "image"
SET_RENDER = "render"
SET_TEXTSET = "textset"
SET_COCO = "coco"
# The Tagger A/B's --coco-count, the photos its judges graded.
COCO_COUNT = 30
DEFAULT_ARM = "default"
LISTING_FIELDS = ("prompt_instruction", "prompt_max_tokens", "ungrounded_streak")
SCORING_FIELDS = ("style_question", "style_criteria")
REPLAY_THRESHOLDS = (0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9)
GRADE_YES = "yes"
GRADE_NO = "no"
GRADE_UNSURE = "unsure"
LINE_KEY = "line"
MEDIUM_KEY = "medium"
# Every COCO photo is a photograph, so a judge grades its line against that medium.
COCO_MEDIUM = "photograph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument("--path", choices=(PATH_PROMPT, PATH_IMAGE), help="the style path to run")
    mode.add_argument("--judge-file", metavar="RUN", help="write data/style/RUN/to_judge.json for blind judges")
    mode.add_argument("--merge", metavar="RUN", help="merge data/style/RUN/judge-*.json into labels.json")
    mode.add_argument("--replay", metavar="RUN", help="score every arm of data/style/RUN against its labels.json")
    parser.add_argument("--run", default=None, help="the run folder's name under data/style, default <date>-<path>")
    parser.add_argument("--variants", type=Path, default=None,
                        help='prompt path JSON {"listings": {NAME: {field: value}}, "scorings": {...}}')
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items of each set")
    parser.add_argument("--seed", type=int, default=7, help="seed for the bootstrap resamples")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be 1 or more, got {args.limit}")
    if args.variants is not None and args.path != PATH_PROMPT:
        parser.error("--variants applies to --path prompt only")
    return args


def results_path(root: Path, run: str) -> Path:
    """root/run/results.json, refused when it exists, since judges grade the records a run wrote."""
    path = root / run / RESULTS_NAME

    if path.is_file():
        raise SystemExit(f"{path} already holds a run. Pass a new --run name.")
    return path


def write_results(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path, needed_by: str) -> Any:
    if not path.is_file():
        raise SystemExit(f"{path} not found. {needed_by}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- arms


def load_variants(path: Path | None) -> tuple[dict[str, dict], dict[str, dict]]:
    """The listings and scorings of a --variants file, each led by "default", which overrides nothing."""
    value: dict[str, Any] = read_json(path, "Pass an existing --variants file.") if path is not None else {}
    unknown = sorted(set(value) - {"listings", "scorings"})
    listings = {DEFAULT_ARM: {}, **value.get("listings", {})}
    scorings = {DEFAULT_ARM: {}, **value.get("scorings", {})}
    extra: list[str] = []

    if unknown:
        raise SystemExit(f"{path}: unknown keys {unknown}. Use listings and scorings.")
    for kind, arms, fields in (("listing", listings, LISTING_FIELDS), ("scoring", scorings, SCORING_FIELDS)):
        for name, overrides in arms.items():
            extra = sorted(set(overrides) - set(fields))
            if extra:
                raise SystemExit(f"{path}: {kind} {name!r} sets {extra}. A {kind} sets only {list(fields)}.")
    return listings, scorings


def arm_settings(base: Any, overrides: dict[str, Any]) -> Any:
    """base with the overrides, style_criteria turned from JSON lists into the pairs StyleSettings holds."""
    values = dict(overrides)

    if "style_criteria" in values:
        values["style_criteria"] = tuple(tuple(pair) for pair in values["style_criteria"])
    return dataclasses.replace(base, **values)


def kept_style(record: dict, scoring: str, threshold: float) -> list[str]:
    """The record's listed phrases whose p("style") under the scoring is at or above the threshold."""
    return [phrase for phrase, p in zip(record["listed"], record["p_style"][scoring], strict=True) if p >= threshold]


# ---------------------------------------------------------------- scoring


def names_medium(text: str, medium: str) -> bool:
    """Whether every content word of the medium name shares a form with a word of text."""
    from logit_classifier.toolkit.tags import in_prompt, prompt_word_forms

    return in_prompt(medium, prompt_word_forms(text))


def medium_items(records: list[dict], text_of: Callable[[dict], str]) -> list[dict]:
    """One scoring item per record with a medium: its unit and whether its text names the medium."""
    return [{"unit": record["id"], "named": names_medium(text_of(record), record["medium"])}
            for record in records if record["medium"] is not None]


def named_share(items: list[dict]) -> float:
    return mean([item["named"] for item in items])


def prompt_lines(records: list[dict], seed: int) -> list[str]:
    """The share of render prompts whose kept style names their medium, and the textset prompts with a style."""
    renders = [record for record in records if record["set"] == SET_RENDER]
    textset = [record for record in records if record["set"] == SET_TEXTSET]
    items = medium_items(renders, lambda record: ", ".join(record["style"]))
    named = sum(item["named"] for item in items)
    styled = sum(bool(record["style"]) for record in textset)

    return [f"render prompts whose kept style names their medium {estimate(named_share, items, seed)} "
            f"({named}/{len(items)})",
            f"textset prompts with a style {styled}/{len(textset)}"]


def run_lines(records: list[dict], scorings: list[str], threshold: float, seed: int) -> list[str]:
    """prompt_lines per listing and scoring at the default threshold, the run's own summary before any judge."""
    listings = list(dict.fromkeys(record["listing"] for record in records))
    arm_records: list[dict] = []
    lines: list[str] = []

    for listing in listings:
        for scoring in scorings:
            arm_records = [{**record, "style": kept_style(record, scoring, threshold)}
                           for record in records if record["listing"] == listing]
            lines += [f"listing {listing}, scoring {scoring}, threshold {threshold}",
                      *(f"  {line}" for line in prompt_lines(arm_records, seed))]
    return lines


def image_lines(records: list[dict], seed: int) -> list[str]:
    """The share of renders whose style line names their medium, and each set's mean time per line."""
    renders = [record for record in records if record["set"] == SET_RENDER]
    photos = [record for record in records if record["set"] == SET_COCO]
    items = medium_items(renders, lambda record: record["line"])
    named = sum(item["named"] for item in items)

    return [f"renders whose style line names their medium {estimate(named_share, items, seed)} "
            f"({named}/{len(items)}), mean {mean([record['ms'] for record in renders]):.0f} ms",
            f"COCO photos {len(photos)}, mean {mean([record['ms'] for record in photos]):.0f} ms"]


# ---------------------------------------------------------------- judges


def judge_entries(results: dict) -> list[dict]:
    """Every distinct phrase each prompt listed under any listing, or each image's line and medium, with no arm or
    score."""
    phrases_by_key: dict[str, set[str]] = {}
    prompt_by_key: dict[str, str] = {}

    if results["path"] == PATH_IMAGE:
        return [{"key": record["id"], LINE_KEY: record["line"], MEDIUM_KEY: record["medium"] or COCO_MEDIUM}
                for record in results["records"]]
    for record in results["records"]:
        phrases_by_key.setdefault(record["id"], set()).update(record["listed"])
        prompt_by_key[record["id"]] = record["prompt"]
    # Sorted, so the order a judge reads tells nothing about the listing that wrote a phrase.
    return [{"key": key, "prompt": prompt_by_key[key], "phrases": sorted(phrases)}
            for key, phrases in sorted(phrases_by_key.items()) if phrases]


def graded_items(entry: dict) -> list[str]:
    """The names a judge grades in one entry: its phrases, or a line's content and medium grades."""
    return entry.get("phrases", [LINE_KEY, MEDIUM_KEY])


def merge_labels(entries: list[dict], judges: dict[str, dict]) -> tuple[dict[str, dict[str, str]], int, int]:
    """One grade per key and phrase: the judges' grade where they agree, else "unsure". Also the agreed and total counts.

    Each judge must grade every phrase or line of every entry, so a skipped one fails here and not in the replay.
    """
    labels: dict[str, dict[str, str]] = {}
    grades: list[str] = []
    agreed = 0
    total = 0

    if len(judges) < 2:
        raise SystemExit(f"got {len(judges)} judge files. Merge needs two or more.")
    for entry in entries:
        labels[entry["key"]] = {}
        for item in graded_items(entry):
            grades = [judge.get(entry["key"], {}).get(item) for judge in judges.values()]
            missing = [name for name, grade in zip(judges, grades, strict=True)
                       if grade not in (GRADE_YES, GRADE_NO, GRADE_UNSURE)]
            if missing:
                raise SystemExit(f"{missing} gave no valid grade for {entry['key']} {item!r}.")
            total += 1
            if len(set(grades)) == 1:
                agreed += 1
                labels[entry["key"]][item] = grades[0]
            else:
                labels[entry["key"]][item] = GRADE_UNSURE
    return labels, agreed, total


def phrase_counts(record: dict, phrases: list[str], labels: dict[str, dict[str, str]]) -> dict:
    """One replay item: the record's unit and set, and how many phrases its judges graded yes, no and unsure."""
    grades = [labels[record["id"]][phrase] for phrase in phrases]

    return {"unit": record["id"], "set": record["set"], "yes": grades.count(GRADE_YES),
            "no": grades.count(GRADE_NO), "unsure": grades.count(GRADE_UNSURE)}


def per_prompt(field: str) -> Callable[[list[dict]], float]:
    return lambda items: mean([item[field] for item in items])


def precision(items: list[dict]) -> float:
    yes = sum(item["yes"] for item in items)
    no = sum(item["no"] for item in items)

    return yes / (yes + no) if yes + no else float("nan")


def replay_lines(records: list[dict], labels: dict[str, dict[str, str]], scorings: list[str],
                 thresholds: tuple[float, ...], seed: int) -> list[str]:
    """One row per listing, scoring and threshold: medium named on renders, then yes and no phrases per prompt."""
    listings = list(dict.fromkeys(record["listing"] for record in records))
    arm_records: list[dict] = []
    items: list[dict] = []
    renders: list[dict] = []
    textset: list[dict] = []
    named: list[dict] = []
    lines = ["listing / scoring / threshold: render medium named, render yes/no per prompt, textset yes/no per "
             "prompt, precision, no per prompt over both sets"]

    for listing in listings:
        for scoring in scorings:
            for threshold in thresholds:
                arm_records = [record for record in records if record["listing"] == listing]
                items = [phrase_counts(record, kept_style(record, scoring, threshold), labels)
                         for record in arm_records]
                renders = [item for item in items if item["set"] == SET_RENDER]
                textset = [item for item in items if item["set"] == SET_TEXTSET]
                named = medium_items([{**record, "style": kept_style(record, scoring, threshold)}
                                      for record in arm_records if record["set"] == SET_RENDER],
                                     lambda record: ", ".join(record["style"]))
                lines.append(
                    f"{listing} / {scoring} / {threshold}: {estimate(named_share, named, seed)}, "
                    f"{per_prompt('yes')(renders):.2f}/{per_prompt('no')(renders):.2f}, "
                    f"{per_prompt('yes')(textset):.2f}/{per_prompt('no')(textset):.2f}, "
                    f"{precision(items):.3f}, {estimate(per_prompt('no'), items, seed)}")
    return lines


def image_replay_lines(records: list[dict], labels: dict[str, dict[str, str]], seed: int) -> list[str]:
    """Per set, the share of style lines its judges graded as naming the image's medium, and as naming content."""
    items: list[dict] = []
    lines: list[str] = []

    for name in (SET_RENDER, SET_COCO):
        items = [{"unit": record["id"], "medium": labels[record["id"]][MEDIUM_KEY] == GRADE_YES,
                  "content": labels[record["id"]][LINE_KEY] == GRADE_NO}
                 for record in records if record["set"] == name]
        lines.append(f"{name}: lines naming the medium {estimate(per_prompt('medium'), items, seed)}, "
                     f"naming content {estimate(per_prompt('content'), items, seed)} ({len(items)} lines)")
    return lines


# ---------------------------------------------------------------- runs


def prompt_units(renders: list[Render], prompts: list[dict]) -> list[dict]:
    """The render prompts with their medium, then the textset prompts, which carry no medium label."""
    return ([{"set": SET_RENDER, "id": render.id, "medium": render.style, "prompt": render.prompt}
             for render in renders]
            + [{"set": SET_TEXTSET, "id": prompt["id"], "medium": None, "prompt": prompt["text"]}
               for prompt in prompts])


def image_units(renders: list[Render], photos: list[Item]) -> list[dict]:
    """The renders with their medium, then the COCO photos, each file relative to tests-AB so a judge can open it."""
    return ([{"set": SET_RENDER, "id": render.id, "medium": render.style,
              "file": render.file.relative_to(HERE).as_posix()} for render in renders]
            + [{"set": SET_COCO, "id": photo.key, "medium": None, "file": photo.path.relative_to(HERE).as_posix()}
               for photo in photos])


def read_prompts(clip: Any, toolkit: Any, classifier: Any, units: list[dict], listings: dict[str, Any],
                 scorings: dict[str, Any], tag_settings: Any) -> list[dict]:
    """prompt_style per listing and prompt, then p("style") of its listed phrases under every scoring.

    Each listing and each scoring keeps one style cache for the run, as one node execution keeps.
    """
    listing_caches: dict[str, dict[str, float]] = {name: {} for name in listings}
    scoring_caches: dict[str, dict[str, float]] = {name: {} for name in scorings}
    trace: Any = None
    p_style: dict[str, list[float]] = {}
    records: list[dict] = []

    for listing, listing_settings in listings.items():
        for unit in units:
            trace = toolkit.prompt_style(clip, classifier, unit["prompt"], settings=listing_settings,
                                         tag_settings=tag_settings, style_cache=listing_caches[listing])
            p_style = {name: list(toolkit.style_scores(classifier, trace.listed, settings, scoring_caches[name]))
                       for name, settings in scorings.items()}
            records.append({**unit, "listing": listing, "text": trace.text, "reply": trace.reply,
                            "listed": list(trace.listed), "p_style": p_style})
            print(f"{listing} {unit['id']}: {', '.join(trace.listed) or '-'}", flush=True)
    return records


def read_images(clip: Any, toolkit: Any, units: list[dict], settings: Any) -> list[dict]:
    """style_line per image at full size, since style_line resamples to settings.image_pixels itself."""
    image: Any = None
    started = 0.0
    line = ""
    records: list[dict] = []

    # The first generate pays CUDA graph capture and kernel warmup, which no record should carry.
    toolkit.style_line(clip, load_image(HERE / units[0]["file"]), settings=settings, node=HARNESS_NAME)
    for unit in units:
        image = load_image(HERE / unit["file"])
        started = time.perf_counter()
        line = toolkit.style_line(clip, image, settings=settings, node=HARNESS_NAME)
        records.append({**unit, "line": line, "ms": (time.perf_counter() - started) * 1000.0})
        print(f"{unit['id']}: {line}", flush=True)
    return records


def write_judge_file(run: str) -> None:
    results = read_json(STYLE_ROOT / run / RESULTS_NAME, "Pass the name of a finished run.")
    entries = judge_entries(results)
    path = STYLE_ROOT / run / JUDGE_NAME

    write_results(path, {"path": results["path"], "entries": entries})
    print(f"{len(entries)} entries, {sum(len(graded_items(entry)) for entry in entries)} grades: {path}")


def merge_judges(run: str) -> None:
    folder = STYLE_ROOT / run
    entries = read_json(folder / JUDGE_NAME, "Run --judge-file first.")["entries"]
    judges = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob(JUDGE_GLOB))}
    labels, agreed, total = merge_labels(entries, judges)

    write_results(folder / LABELS_NAME, labels)
    print(f"{len(judges)} judges agreed on {agreed}/{total} grades, the rest are unsure: {folder / LABELS_NAME}")


def replay(run: str, seed: int) -> None:
    folder = STYLE_ROOT / run
    results = read_json(folder / RESULTS_NAME, "Pass the name of a finished run.")
    labels = read_json(folder / LABELS_NAME, "Run --merge first.")
    lines: list[str] = []

    if results["path"] == PATH_IMAGE:
        lines = image_replay_lines(results["records"], labels, seed)
    else:
        lines = replay_lines(results["records"], labels, list(results["scorings"]), REPLAY_THRESHOLDS, seed)
    print(*lines, sep="\n")


def run(args: argparse.Namespace) -> None:
    out_path = results_path(STYLE_ROOT, args.run or f"{date.today().isoformat()}-{args.path}")
    renders = load_renders(DEFAULT_MANIFEST)[: args.limit]
    listings, scorings = load_variants(args.variants)
    units: list[dict] = []
    toolkit: Any = None
    settings: Any = None
    listing_settings: dict[str, Any] = {}
    scoring_settings: dict[str, Any] = {}
    clip: Any = None
    records: list[dict] = []
    lines: list[str] = []

    # Every input is read and checked before the CLIP loads, so a missing file fails in seconds.
    if args.path == PATH_PROMPT:
        units = prompt_units(renders, load_textset(DEFAULT_DATA / "textset" / "prompts.json")[: args.limit])
    else:
        # The Tagger A/B's loader, which checks every render and photo against the sha256 its judges graded.
        units = image_units(renders, load_conditions(DEFAULT_MANIFEST, DEFAULT_COCO, DEFAULT_COCO_HASHES, COCO_COUNT,
                                                     args.limit)["coco"])
    ab_env.bootstrap()
    import logit_classifier

    from logit_tagger import engine, style, tagging

    toolkit = engine.toolkit(HARNESS_NAME)
    settings = toolkit.StyleSettings()
    # Built before the CLIP loads, so StyleSettings rejects a bad variant in seconds.
    listing_settings = {name: arm_settings(settings, overrides) for name, overrides in listings.items()}
    scoring_settings = {name: arm_settings(settings, overrides) for name, overrides in scorings.items()}
    clip = ab_env.load_clip(args.clip)
    with toolkit.skip_resident_loads(clip):
        if args.path == PATH_PROMPT:
            records = read_prompts(clip, toolkit, engine.build_classifier(clip, HARNESS_NAME, needs_images=False),
                                   units, listing_settings, scoring_settings,
                                   tagging.tag_settings(node=style.STYLE_EXTRACTOR_NAME))
        else:
            records = read_images(clip, toolkit, units, settings)

    if args.path == PATH_PROMPT:
        lines = run_lines(records, list(scorings), settings.style_threshold, args.seed)
    else:
        lines = image_lines(records, args.seed)
    print("", *lines, sep="\n")
    write_results(out_path, {"path": args.path, "clip": args.clip, "library_version": logit_classifier.__version__,
                             "settings": dataclasses.asdict(settings), "listings": listings, "scorings": scorings,
                             "records": records})
    print(f"raw results: {out_path}")


def main() -> None:
    args = parse_args()

    if args.judge_file:
        write_judge_file(args.judge_file)
    elif args.merge:
        merge_judges(args.merge)
    elif args.replay:
        replay(args.replay, args.seed)
    else:
        run(args)


if __name__ == "__main__":
    main()
