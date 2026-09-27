"""Store a run's records, split their kept tags into blind judge batches, and merge the verdicts into the store.

The store, data/tagger-ab/verdicts.json, maps an image key to {tag: "yes" | "no" | "vague" | "unsure"}.
split reads every results*.json in each RUN_DIR, such as a data/tagger-ab/runs/<experiment>/ folder. It writes
judge_in_<n>.json (each image's kept tags the store does not grade yet, plus K of its graded kept tags mixed in blind),
calibration.json (those mixed-in pairs and their stored verdicts) and a copy of RUBRIC.md. Judges write
verdicts_<n>.json beside them.
merge checks each verdicts_<n>.json against its judge_in_<n>.json, prints how the judges agree with the store on the
calibration pairs, and adds every new pair to the store. A pair keeps its first verdict, and a later different verdict
is listed as a conflict.
store copies one ab_tagger_compare.py run into data/tagger-ab/runs/<name>/: its records cut to the fields scoring,
judging and --replay read, its settings, its run meta, and arms.json of each arm's overrides.
Usage: python ab_judge.py store RESULTS.json --name EXPERIMENT
       python ab_judge.py split RUN_DIR... --batches N --calibration K --out DIR [--verdicts FILE]
       python ab_judge.py merge DIR [--verdicts FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STORE_DIR = HERE / "data" / "tagger-ab"
RUNS_DIR = STORE_DIR / "runs"
DEFAULT_VERDICTS = STORE_DIR / "verdicts.json"
RUBRIC = STORE_DIR / "RUBRIC.md"
RENDER_MANIFEST = HERE / "data" / "release-gates" / "renders" / "manifest.json"
COCO_HASHES = STORE_DIR / "coco30.json"
COCO_DIR = HERE / "cache" / "coco60"
VERDICTS = ("yes", "no", "vague", "unsure")
BATCH_FILES = ("judge_in_*.json", "verdicts_*.json")
# The fields ab_tagger_score.py and --replay read, plus the timings. The rest of a record is run diagnostics.
STORED_FIELDS = ("tags", "reply", "echo_reply", "prompt_tags", "proposed", "candidates", "dropped", "prompt_ms",
                 "propose_ms", "verify_ms", "total_ms", "propose_decode_ms_per_token")


@dataclass(frozen=True)
class MergeReport:
    """One merge's new pairs, the store's size after it, its conflicts and each calibration pair's two grades."""

    added: int
    total: int
    conflicts: list[tuple[str, str, str, str]]
    calibration: list[tuple[str, str]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def image_paths(manifest_path: Path, coco_hashes_path: Path, coco_dir: Path) -> dict[str, Path]:
    """Each image's file by its key: a render's id, or a COCO file relative to coco_dir."""
    renders = load_json(manifest_path)
    coco = load_json(coco_hashes_path)

    return ({entry["id"]: (manifest_path.parent / entry["file"]).resolve() for entry in renders}
            | {entry["file"]: (coco_dir / entry["file"]).resolve() for entry in coco})


def kept_tags(run_dirs: list[Path]) -> dict[str, set[str]]:
    """Each image's kept tags over every condition and arm of each folder's results*.json."""
    files: list[Path] = []
    kept: dict[str, set[str]] = {}

    for run_dir in run_dirs:
        found = sorted(run_dir.glob("results*.json"))
        if not found:
            raise SystemExit(f"{run_dir} holds no results*.json. Name a folder ab_tagger_compare.py wrote.")
        files += found
    for path in files:
        records = [(key, record) for arms in load_json(path).values() for images in arms.values()
                   for key, record in images.items()]
        for key, record in records:
            kept.setdefault(key, set()).update(record["tags"])
    return kept


def pair_hash(key: str, tag: str) -> str:
    return hashlib.sha256(f"{key}|{tag}".encode()).hexdigest()


def calibration_tags(key: str, tags: set[str], graded: dict[str, str], count: int) -> list[str]:
    """Up to count of the image's graded kept tags, in hash order so a rerun picks the same ones without a seed."""
    pool = sorted((tag for tag in tags if tag in graded), key=lambda tag: pair_hash(key, tag))

    return pool[:count]


def tag_count(batch: list[dict[str, Any]]) -> int:
    return sum(len(entry["tags"]) for entry in batch)


def plan_batches(kept: dict[str, set[str]], verdicts: dict[str, dict[str, str]], images: dict[str, Path],
                 batches: int, calibration: int) -> tuple[list[list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """The judge batches, balanced by tag count, and the graded pairs mixed into them.

    Only an image with an ungraded kept tag gets an entry, so no batch is empty.
    """
    pending = [key for key in sorted(kept) if kept[key] - verdicts.get(key, {}).keys()]
    unknown = [key for key in pending if key not in images]
    entries: list[dict[str, Any]] = []
    mixed: dict[str, dict[str, str]] = {}
    loads: list[list[dict[str, Any]]] = []

    if batches < 1 or calibration < 0:
        raise SystemExit(f"--batches must be 1 or more and --calibration 0 or more, got {batches} and {calibration}")
    if unknown:
        raise SystemExit(f"images {unknown} are in neither the renders' manifest.json nor coco30.json")
    for key in pending:
        graded = verdicts.get(key, {})
        new = kept[key] - graded.keys()
        picked = calibration_tags(key, kept[key], graded, calibration)
        if picked:
            mixed[key] = {tag: graded[tag] for tag in picked}
        entries.append({"image": str(images[key]), "key": key, "tags": sorted(new | set(picked))})
    loads = [[] for _ in range(min(batches, len(entries)))]
    for entry in sorted(entries, key=lambda entry: -len(entry["tags"])):
        min(loads, key=tag_count).append(entry)
    return [sorted(batch, key=lambda entry: entry["key"]) for batch in loads], mixed


def split(run_dirs: list[Path], verdicts_path: Path, out_dir: Path, batches: int, calibration: int,
          images: dict[str, Path]) -> list[list[dict[str, Any]]]:
    """Writes the judge batches, calibration.json and RUBRIC.md into out_dir, and returns the batches.

    Writes nothing when every kept tag is already graded.
    """
    kept = kept_tags(run_dirs)
    verdicts = load_json(verdicts_path)
    taken = sorted(path.name for pattern in BATCH_FILES for path in out_dir.glob(pattern))
    loads: list[list[dict[str, Any]]] = []
    mixed: dict[str, dict[str, str]] = {}
    absent: list[str] = []

    # A second split into one folder would overwrite batches a judge may still be grading.
    if taken:
        raise SystemExit(f"{out_dir} already holds {', '.join(taken)}. Merge that round or pick another --out.")
    loads, mixed = plan_batches(kept, verdicts, images, batches, calibration)
    absent = sorted({entry["image"] for batch in loads for entry in batch if not Path(entry["image"]).is_file()})
    if absent:
        raise SystemExit(f"image files not found, so no judge could grade them: {', '.join(absent)}")
    if not loads:
        return loads
    out_dir.mkdir(parents=True, exist_ok=True)
    for number, batch in enumerate(loads, 1):
        write_json(out_dir / f"judge_in_{number}.json", batch)
    write_json(out_dir / "calibration.json", mixed)
    shutil.copyfile(RUBRIC, out_dir / "RUBRIC.md")
    return loads


def store(results_path: Path, name: str, runs_dir: Path = RUNS_DIR) -> Path:
    """Write runs_dir/name/ from results_path and the settings and run meta beside it, and return the folder."""
    suffix = results_path.stem.removeprefix("results")
    settings_path = results_path.with_name(f"settings{suffix}.json")
    meta_path = results_path.with_name(f"run_meta{suffix}.json")
    out_dir = runs_dir / name
    results: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    missing: list[str] = []

    missing = [path.name for path in (results_path, settings_path, meta_path) if not path.is_file()]
    if missing:
        raise SystemExit(f"{results_path.parent} lacks {', '.join(missing)}. Name a results file ab_tagger_compare.py "
                         f"wrote.")
    # A stored run is the record its verdicts and scores cite, so it is never overwritten.
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists. Pick another --name.")
    results = {condition: {arm: {key: {field: record[field] for field in STORED_FIELDS if field in record}
                                 for key, record in images.items()}
                           for arm, images in by_arm.items()}
               for condition, by_arm in load_json(results_path).items()}
    meta = load_json(meta_path)
    for run in meta.get("runs", []):
        arms |= run.get("variants", {})
    out_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(json.dumps(results, separators=(",", ":"), ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    shutil.copyfile(settings_path, out_dir / "settings.json")
    write_json(out_dir / "run_meta.json", meta)
    write_json(out_dir / "arms.json", arms)
    return out_dir


def batch_number(path: Path) -> int:
    return int(path.stem.removeprefix("judge_in_"))


def check_batch(number: int, entries: list[dict[str, Any]], grades: dict[str, dict[str, str]]) -> None:
    """Stops on a batch whose judge skipped, added or renamed a pair, or wrote a grade outside the rubric."""
    asked = {(entry["key"], tag) for entry in entries for tag in entry["tags"]}
    judged = {(key, tag) for key, tags in grades.items() for tag in tags}
    missing = sorted(asked - judged)
    extra = sorted(judged - asked)
    wrong = sorted((key, tag, grade) for key, tags in grades.items() for tag, grade in tags.items()
                   if grade not in VERDICTS)

    if missing or extra:
        raise SystemExit(f"verdicts_{number}.json differs from judge_in_{number}.json: {len(missing)} missing and "
                         f"{len(extra)} unasked pairs, such as {(missing + extra)[:3]}. Have that batch judged again.")
    if wrong:
        raise SystemExit(f"verdicts_{number}.json has grades outside {VERDICTS}: {wrong[:3]}. "
                         "Have that batch judged again.")


def merge(batch_dir: Path, verdicts_path: Path) -> MergeReport:
    """Adds every new pair of the judged batches in batch_dir to the store at verdicts_path.

    Every batch is checked before the store is written, so a bad batch leaves the store unchanged.
    """
    inputs = sorted((path for path in batch_dir.glob("judge_in_*.json")
                     if path.stem.removeprefix("judge_in_").isdigit()), key=batch_number)
    calibration_path = batch_dir / "calibration.json"
    judged: list[tuple[str, str, str]] = []
    first: dict[tuple[str, str], str] = {}
    mixed: dict[tuple[str, str], str] = {}
    store: dict[str, dict[str, str]] = {}
    conflicts: list[tuple[str, str, str, str]] = []
    added = 0

    if not inputs:
        raise SystemExit(f"{batch_dir} holds no judge_in_<n>.json. Run split first.")
    if not calibration_path.is_file():
        raise SystemExit(f"{calibration_path} is missing. split writes it beside the batches.")
    for path in inputs:
        number = batch_number(path)
        verdict_path = batch_dir / f"verdicts_{number}.json"
        if not verdict_path.is_file():
            raise SystemExit(f"{verdict_path} is missing. Every judge_in_<n>.json needs its judge's verdicts_<n>.json.")
        grades = load_json(verdict_path)
        check_batch(number, load_json(path), grades)
        judged += [(key, tag, grade) for key, tags in grades.items() for tag, grade in tags.items()]
    for key, tag, grade in judged:
        first.setdefault((key, tag), grade)
    mixed = {(key, tag): grade for key, tags in load_json(calibration_path).items() for tag, grade in tags.items()}
    unasked = sorted(set(mixed) - set(first))
    if unasked:
        raise SystemExit(f"calibration.json lists pairs no batch holds: {unasked[:3]}")

    store = load_json(verdicts_path)
    for key, tag, grade in judged:
        if (key, tag) in mixed:
            continue
        known = store.setdefault(key, {})
        if tag not in known:
            known[tag] = grade
            added += 1
        elif known[tag] != grade:
            conflicts.append((key, tag, known[tag], grade))
    write_json(verdicts_path, {key: dict(sorted(store[key].items())) for key in sorted(store)})
    return MergeReport(
        added=added,
        total=sum(len(tags) for tags in store.values()),
        conflicts=conflicts,
        calibration=[(mixed[pair], first[pair]) for pair in sorted(mixed)],
    )


def precision(grades: list[str]) -> float | None:
    """yes over yes, no and vague, as ab_tagger_score.py defines it."""
    graded = sum(grade in ("yes", "no", "vague") for grade in grades)

    return sum(grade == "yes" for grade in grades) / graded if graded else None


def calibration_line(pairs: list[tuple[str, str]]) -> str:
    """How the judges agree with the store on the calibration pairs, as (stored, new) grades."""
    count = len(pairs)
    agreed = sum(stored == new for stored, new in pairs)
    shown = [precision([pair[side] for pair in pairs]) for side in (0, 1)]
    stored_text, new_text = ("n/a" if value is None else f"{value:.3f}" for value in shown)

    if not count:
        return "calibration: no pairs"
    return (f"calibration: {count} pairs, agreement {agreed / count:.3f}, precision {stored_text} under the stored "
            f"grades and {new_text} under the new grades")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    store_parser = commands.add_parser("store", help="copy one run's compact records into data/tagger-ab/runs/<name>/")
    store_parser.add_argument("results", type=Path, metavar="RESULTS.json", help="an ab_tagger_compare.py results file")
    store_parser.add_argument("--name", required=True, help="the experiment folder, such as 2026-09-27-caps")

    split_parser = commands.add_parser("split", help="write blind judge batches of the run records' ungraded tags")
    split_parser.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR", help="folders holding results*.json")
    split_parser.add_argument("--batches", type=int, required=True, help="judge batches, balanced by tag count")
    split_parser.add_argument("--calibration", type=int, required=True,
                              help="already-graded kept tags mixed into each image's entry")
    split_parser.add_argument("--out", type=Path, required=True,
                              help="folder for judge_in_<n>.json, calibration.json and RUBRIC.md")
    split_parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS, help="the verdict store")
    merge_parser = commands.add_parser("merge", help="check the judges' verdicts_<n>.json and add new pairs to the store")
    merge_parser.add_argument("dir", type=Path, metavar="DIR", help="the split --out folder, with verdicts_<n>.json")
    merge_parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS, help="the verdict store")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loads: list[list[dict[str, Any]]] = []
    report: MergeReport | None = None

    if args.command == "store":
        print(f"stored in {store(args.results, args.name)}")
        return
    if args.command == "split":
        loads = split(args.run_dirs, args.verdicts, args.out, args.batches, args.calibration,
                      image_paths(RENDER_MANIFEST, COCO_HASHES, COCO_DIR))
        for number, batch in enumerate(loads, 1):
            print(f"judge_in_{number}.json: {len(batch)} images, {tag_count(batch)} tags")
        print(f"{sum(tag_count(batch) for batch in loads)} tags to judge" if loads else "nothing to judge")
        return
    report = merge(args.dir, args.verdicts)
    print(calibration_line(report.calibration))
    for key, tag, kept, later in report.conflicts:
        print(f"conflict {key} | {tag}: kept {kept}, a batch said {later}")
    print(f"added {report.added} pairs, {len(report.conflicts)} conflicts, the store holds {report.total} pairs")


if __name__ == "__main__":
    main()
