"""A/B the Logit Tagger pipeline over a real Qwen3-VL-4B: the package at --new-root, its variants and an older tagging.py.

Arm "new" runs logit_tagger.tagging.tag_images once per condition, as one node execution does. A variant arm
does the same with the TagSettings fields its --variants entry overrides. Arm "old" loads --old-tagging, a file
from `git show <rev>:logit_tagger/tagging.py`, and runs its tag_image once per image, one node execution per
distinct prompt, since that node took one prompt per batch.
Writes results<suffix>.json (every image's stages per condition and arm), settings<suffix>.json (each package
arm's full TagSettings), to_judge<suffix>.json (each image's tags from every arm and condition that no --graded
file grades yet, unlabeled and sorted, for blind judges) and run_meta<suffix>.json (decode speed, walls, cap and
fallback counts). Judges grade to_judge by data/tagger-ab/RUBRIC.md, and ab_tagger_score.py scores the arms
against their verdicts.
--record-thing-scores adds each candidate's thing check p("other") to its record, cached per tag in
cache/thing_scores_<clip stem>.json, so ab_tagger_score.py --replay can re-derive stricter arms offline.
A superset run for --replay sets the loosest value of each field under test, such as propose_cap 40, merge_cap 120
and thing_check "prompt", so every merged candidate carries a verify score.
Each render must match the sha256 in its manifest entry and each COCO image its entry in data/tagger-ab/coco30.json.
Usage: python ab_tagger_compare.py [--out DIR] [--arms new old N1 ...] [--old-tagging OLD.py] [--variants V.json]
       [--graded data/tagger-ab/verdicts.json] [--record-thing-scores]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import itertools
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
DEFAULT_MANIFEST = HERE / "data" / "release-gates" / "renders" / "manifest.json"
DEFAULT_COCO = HERE / "cache" / "coco60" / "coco60.json"
DEFAULT_COCO_HASHES = HERE / "data" / "tagger-ab" / "coco30.json"
DEFAULT_OUT = HERE / "out" / "tagger-ab"
THING_SCORES_DIR = HERE / "cache"
CONDITIONS = ("render-noprompt", "render-prompt", "coco")
OLD_ARM = "old"
NEW_ARM = "new"
SCRIPT_NAME = "ab_tagger_compare.py"


@dataclass(frozen=True)
class Item:
    """One image to tag. key names the image the same way in every condition."""

    key: str
    path: Path
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the three JSON files")
    parser.add_argument("--suffix", default="", help="appended to each output file's stem, such as _r2")
    parser.add_argument("--append", action="store_true",
                        help="add the arms to an existing results file instead of starting a new one")
    parser.add_argument("--arms", nargs="+", default=[NEW_ARM, OLD_ARM],
                        help=f"'{NEW_ARM}', '{OLD_ARM}' or a --variants name, run in this order")
    parser.add_argument("--old-tagging", type=Path, default=None, help="the old arm's tagging.py")
    parser.add_argument("--variants", type=Path, default=None,
                        help="JSON of {arm: {TagSettings field: value}} for arms that vary the new package")
    parser.add_argument("--graded", type=Path, nargs="*", default=[],
                        help="verdict files whose (image, tag) pairs to_judge leaves out")
    parser.add_argument("--new-root", type=Path, default=HERE.parent,
                        help="directory holding the new arm's logit_tagger and tests-AB/ab_env.py")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="the release-gate renders' manifest.json")
    parser.add_argument("--coco", type=Path, default=DEFAULT_COCO, help="coco60.json, its file paths relative to it")
    parser.add_argument("--coco-hashes", type=Path, default=DEFAULT_COCO_HASHES,
                        help="the judged COCO images' file and sha256, the file relative to --coco's folder")
    parser.add_argument("--coco-count", type=int, default=30, help="how many COCO images, from the first")
    parser.add_argument("--limit", type=int, default=None, help="tag only the first N images of each condition")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--record-thing-scores", action="store_true",
                        help="after tagging, add each package arm candidate's thing check p(other) to its record")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(expected: dict[Path, str | None], source: Path) -> None:
    """Stops on an image whose bytes differ from its entry in source, since verdicts belong to the pixels judged."""
    for path, sha256 in expected.items():
        if sha256 is None:
            raise SystemExit(f"{path} has no sha256 in {source}, so no verdict can be tied to its pixels. "
                             "Add its sha256 there once its tags are judged.")
        if file_sha256(path) != sha256:
            raise SystemExit(f"{path} differs from its sha256 in {source}, so its verdicts were judged on other "
                             "pixels. Restore the judged file.")


def load_conditions(manifest_path: Path, coco_path: Path, coco_hashes_path: Path, coco_count: int,
                    limit: int | None) -> dict[str, list[Item]]:
    """Every condition's images in run order, each checked against the sha256 its verdicts were judged on."""
    renders = json.loads(manifest_path.read_text(encoding="utf-8"))
    coco = json.loads(coco_path.read_text(encoding="utf-8"))["images"][:coco_count]
    coco_sha256 = {entry["file"]: entry["sha256"] for entry in json.loads(coco_hashes_path.read_text(encoding="utf-8"))}
    render_dir = manifest_path.parent
    conditions: dict[str, list[Item]] = {}
    missing: list[str] = []

    conditions = {
        "render-noprompt": [Item(entry["id"], render_dir / entry["file"], "") for entry in renders],
        "render-prompt": [Item(entry["id"], render_dir / entry["file"], entry["prompt"]) for entry in renders],
        "coco": [Item(entry["file"], coco_path.parent / entry["file"], "") for entry in coco],
    }
    missing = sorted({str(item.path) for items in conditions.values() for item in items if not item.path.is_file()})
    if missing:
        raise SystemExit(f"image files not found: {', '.join(missing)}")
    check_hashes({render_dir / entry["file"]: entry.get("sha256") for entry in renders}, manifest_path)
    check_hashes({coco_path.parent / entry["file"]: coco_sha256.get(entry["file"]) for entry in coco}, coco_hashes_path)
    return {name: items[:limit] for name, items in conditions.items()}


def load_variants(path: Path | None, arms: list[str]) -> dict[str, dict[str, Any]]:
    """Each arm's tagging overrides, empty for 'new'. Rejects an arm that is neither built in nor a variant."""
    variants = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
    unknown = [arm for arm in arms if arm not in (NEW_ARM, OLD_ARM) and arm not in variants]

    if unknown:
        raise SystemExit(f"arms {unknown} are neither '{NEW_ARM}', '{OLD_ARM}' nor a name in {path}")
    return {arm: variants.get(arm, {}) for arm in arms if arm != OLD_ARM}


def field_value(default: Any, value: Any) -> Any:
    """A JSON value as the field's own type, since JSON has no frozenset or tuple."""
    if isinstance(value, list) and isinstance(default, frozenset):
        return frozenset(value)
    if isinstance(value, list) and isinstance(default, tuple):
        return tuple(tuple(item) if isinstance(item, list) else item for item in value)
    return value


def arm_settings(base: Any, arm: str, overrides: dict[str, Any], *, source: str = "--variants") -> Any:
    """base with the arm's field overrides. A misspelled field or a value TagSettings rejects exits naming the arm."""
    fields = {name: field_value(getattr(base, name, None), value) for name, value in overrides.items()}

    try:
        return dataclasses.replace(base, **fields)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{source} arm {arm!r}: {error}") from error


def settings_json(settings: Any) -> dict[str, Any]:
    """Every TagSettings field as JSON, which arm_settings reads back. JSON has no frozenset, so a set is sorted."""
    return {name: sorted(value) if isinstance(value, frozenset) else value
            for name, value in dataclasses.asdict(settings).items()}


def settings_path_of(results_path: Path) -> Path:
    """The settings<suffix>.json written beside results<suffix>.json."""
    if not results_path.name.startswith("results"):
        raise SystemExit(f"{results_path} is not named results<suffix>.json, so its settings<suffix>.json "
                         "cannot be found. Keep the name ab_tagger_compare.py wrote.")
    return results_path.with_name("settings" + results_path.name.removeprefix("results"))


def thing_wording(settings: Any) -> dict[str, Any]:
    """The thing check's wording as JSON, the question every cached p("other") answers."""
    return {"thing_question": settings.thing_question,
            "thing_criteria": [list(criterion) for criterion in settings.thing_criteria]}


def load_thing_cache(path: Path, settings_list: list[Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """The one thing check wording every arm asks and the cached p("other") per tag.

    Stops when the arms word the check two ways or the cache answers another wording, since one tag's answer
    depends on the wording.
    """
    wordings: list[dict[str, Any]] = []
    wording: dict[str, Any] = {}
    cache: dict[str, Any] = {}

    for settings in settings_list:
        wording = thing_wording(settings)
        if wording not in wordings:
            wordings.append(wording)
    if not wordings:
        raise SystemExit(f"--record-thing-scores records the arms that run the package, and --arms holds only "
                         f"'{OLD_ARM}'.")
    if len(wordings) > 1:
        raise SystemExit(f"--record-thing-scores got arms that word the thing check {len(wordings)} ways, and "
                         f"{path} holds one wording's answers. Record each wording in its own run.")
    if not path.is_file():
        return wordings[0], {}
    cache = json.loads(path.read_text(encoding="utf-8"))
    if {name: cache.get(name) for name in wordings[0]} != wordings[0]:
        raise SystemExit(f"{path} holds p(other) for another thing check wording than the arms ask. Move it "
                         "aside to record this wording.")
    return wordings[0], cache["p_other"]


def record_thing_scores(classifier: Any, records: list[dict[str, Any]], settings: Any,
                        known: dict[str, float]) -> None:
    """Adds each candidate's p("other") under settings to its record, asking text only the tags known lacks."""
    from logit_classifier.toolkit.comfyui import thing_scores

    candidates = [candidate for record in records for candidate in record["candidates"]]
    p_other = thing_scores(classifier, [candidate["tag"] for candidate in candidates], settings, known)

    for candidate, p in zip(candidates, p_other, strict=True):
        candidate["p_other"] = p


@contextlib.contextmanager
def tag_settings_of_arm(tagging: Any, settings: Any) -> Iterator[None]:
    """tagging.tag_settings returns settings inside the block, so every stage the node runs reads the arm's values."""
    original = tagging.tag_settings

    tagging.tag_settings = lambda: settings
    try:
        yield
    finally:
        tagging.tag_settings = original


class GenerateMeter:
    """Times each measured generate one decode step at a time, through core's sample_token.

    It sits on the transformer instance beneath each arm's stop hook, which saves and restores it.
    """

    def __init__(self, transformer: Any) -> None:
        self.records: list[dict[str, Any]] = []
        self._stamps: list[float] | None = None
        original = transformer.sample_token

        def sample_token(*args: Any, **kwargs: Any) -> Any:
            token = original(*args, **kwargs)
            if self._stamps is not None:
                # A sync, so the stamp marks the finished step. Both stop hooks sync on this token anyway.
                token.reshape(-1)[0].item()
                self._stamps.append(time.perf_counter())
            return token

        transformer.sample_token = sample_token

    def measure(self, function: Callable[..., Any], stage_of: Callable[[dict[str, Any]], str]) -> Callable[..., Any]:
        """function, adding one record per call: its stage, tokens, prefill and decode ms per token."""
        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            self._stamps = []
            try:
                return function(*args, **kwargs)
            finally:
                self.records.append(summarize_steps(stage_of(kwargs), started, self._stamps))
                self._stamps = None
        return measured

    def take(self) -> list[dict[str, Any]]:
        records, self.records = self.records, []
        return records


def summarize_steps(stage: str, started: float, stamps: list[float]) -> dict[str, Any]:
    """decode_ms_per_token spans every decode step. median_step_ms leaves out the graph capture steps."""
    steps = len(stamps)
    step_ms = [(later - earlier) * 1000.0 for earlier, later in itertools.pairwise(stamps)]

    return {
        "stage": stage,
        "tokens": steps,
        "prefill_ms": (stamps[0] - started) * 1000.0 if stamps else None,
        "decode_ms_per_token": sum(step_ms) / len(step_ms) if step_ms else None,
        "median_step_ms": statistics.median(step_ms) if step_ms else None,
    }


def timed(function: Callable[..., Any], calls: list[float]) -> Callable[..., Any]:
    """function, appending each call's milliseconds to calls."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            calls.append((time.perf_counter() - started) * 1000.0)
    return wrapper


def returning(function: Callable[..., Any], values: list[Any]) -> Callable[..., Any]:
    """function, appending each call's return value to values."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        value = function(*args, **kwargs)
        values.append(value)
        return value
    return wrapper


def load_old_tagging(path: Path) -> Any:
    """The old arm's tagging.py as a module. It imports nothing from its own package."""
    spec = importlib.util.spec_from_file_location("old_tagging", path)
    module = None

    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path} as a python module")
    module = importlib.util.module_from_spec(spec)
    # dataclass resolves the module's string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def transformer_of(clip: Any) -> Any:
    """Core's generating transformer, whose sample_token both arms' stop hooks wrap."""
    model = clip.cond_stage_model

    return getattr(model, model.clip).transformer


def new_record(trace: Any, prompt_ms: float, proposes: list[dict[str, Any]], settings: Any,
               prompt_tags: list[str]) -> dict[str, Any]:
    """The record of one new-package trace under its arm's settings.

    proposes holds its propose generates, the used reply's last. prompt_tags are the prompt stage's tags the merge
    received, in listing order.
    """
    kept = set(trace.kept)
    dropped = [{"tag": name, "reason": reason} for name, reason in trace.dropped]
    used = proposes[-1]
    threshold = 0.0
    below = False

    for tag, origin, score in zip(trace.candidates, trace.origins, trace.scores, strict=True):
        threshold = settings.prompt_only_threshold if origin == "prompt" else settings.verify_threshold
        if tag in kept:
            continue
        # A None score means the settings turned verify off, so only drop_subsets removed the tag.
        below = score is not None and score < threshold
        reason = f"below the {threshold} verify threshold" if below else "its words sit inside a kept tag"
        dropped.append({"tag": tag, "reason": reason})
    return {
        "tags": list(trace.kept),
        "reply": trace.reply,
        "echo_reply": getattr(trace, "echo_reply", None),
        "prompt_ms": prompt_ms,
        "propose_ms": trace.propose_ms,
        "verify_ms": trace.verify_ms,
        "total_ms": prompt_ms + trace.propose_ms + trace.verify_ms,
        "propose_tokens": used["tokens"],
        "propose_decode_ms_per_token": used["decode_ms_per_token"],
        "propose_cap_hit": used["tokens"] >= settings.propose_max_tokens,
        "proposed_cap_hit": len(trace.proposed) >= settings.propose_cap,
        "size_in": list(trace.size_in),
        "size_out": list(trace.size_out),
        "proposed": list(trace.proposed),
        "prompt_tags": prompt_tags,
        "candidates": [{"tag": tag, "origin": origin, "score": score}
                       for tag, origin, score in zip(trace.candidates, trace.origins, trace.scores, strict=True)],
        "dropped": dropped,
    }


def old_record(detail: Any, propose: dict[str, Any], old: Any) -> dict[str, Any]:
    return {
        "tags": [tag for tag in detail.tags.split(", ") if tag],
        "reply": detail.proposal,
        "propose_ms": detail.propose_ms,
        "verify_ms": detail.verify_ms,
        "total_ms": detail.propose_ms + detail.verify_ms,
        "propose_tokens": propose["tokens"],
        "propose_decode_ms_per_token": propose["decode_ms_per_token"],
        "propose_cap_hit": propose["tokens"] >= old.PROPOSE_MAX_TOKENS,
        "size_in": list(detail.size_in),
        "size_out": list(detail.size_out),
        "candidates": [{"tag": tag, "score": score} for tag, score in detail.candidates],
        "dropped": [{"tag": tag, "reason": f"below the {old.NOUL_THRESHOLD} verify threshold"}
                    for tag, score in detail.candidates if score < old.NOUL_THRESHOLD],
    }


def run_new_arm(tagging: Any, clip: Any, items: list[Item], images: list[Any], meter: GenerateMeter,
                prompt_calls: list[float], prompt_listings: list[Any],
                end_node: Callable[[], None]) -> dict[str, dict[str, Any]]:
    """One tag_images call over the condition, as one node execution, then one record per image.

    prompt_calls and prompt_listings receive each prompt stage's milliseconds and PromptTags.
    """
    settings = tagging.tag_settings()
    prompts = [item.prompt for item in items]
    distinct = list(dict.fromkeys(prompts))
    traces: list[Any] = []
    proposes: list[dict[str, Any]] = []
    prompt_ms_of: dict[str, float] = {}
    prompt_tags_of: dict[str, list[str]] = {}
    records: dict[str, dict[str, Any]] = {}
    cursor = 0

    prompt_calls.clear()
    prompt_listings.clear()
    traces = tagging.tag_images(clip, images, prompts)
    end_node()
    proposes = [record for record in meter.records if record["stage"] == "propose"]
    # The prompt stage runs once per distinct prompt, so that prompt's images share its time.
    prompt_ms_of = {prompt: ms / prompts.count(prompt) for prompt, ms in zip(distinct, prompt_calls, strict=True)}
    prompt_tags_of = {prompt: list(listing.tags) for prompt, listing in zip(distinct, prompt_listings, strict=True)}
    for item, trace in zip(items, traces, strict=True):
        # The echo fallback proposes a second time.
        used = 2 if getattr(trace, "echo_reply", None) is not None else 1
        records[item.key] = new_record(trace, prompt_ms_of[item.prompt], proposes[cursor:cursor + used], settings,
                                       prompt_tags_of[item.prompt])
        cursor += used
    if cursor != len(proposes):
        raise SystemExit(f"{len(proposes)} propose generates ran for {cursor} expected by the traces")
    return records


def run_old_arm(old: Any, clip: Any, classifier: Any, items: list[Item], images: list[Any], meter: GenerateMeter,
                end_node: Callable[[], None]) -> dict[str, dict[str, Any]]:
    """tag_image per image, one node execution per distinct prompt, then one record per image."""
    prompts = [item.prompt for item in items]
    order: list[int] = []
    details: list[Any] = []
    proposes: list[dict[str, Any]] = []

    for prompt in dict.fromkeys(prompts):
        group = [index for index, own in enumerate(prompts) if own == prompt]
        details += [old.tag_image(clip, images[index], prompt, classifier=classifier) for index in group]
        order += group
        end_node()
    proposes = [record for record in meter.records if record["stage"] == "propose"]
    return {items[index].key: old_record(detail, propose, old)
            for index, detail, propose in zip(order, details, proposes, strict=True)}


def load_graded(paths: list[Path]) -> dict[str, set[str]]:
    graded: dict[str, set[str]] = {}

    for path in paths:
        for key, verdicts in json.loads(path.read_text(encoding="utf-8")).items():
            graded.setdefault(key, set()).update(verdicts)
    return graded


def judge_list(conditions: dict[str, list[Item]], results: dict[str, dict[str, dict[str, Any]]],
               graded: dict[str, set[str]]) -> list[dict[str, Any]]:
    """One entry per image with every ungraded tag any arm kept in any condition, sorted, so no tag names its arm."""
    paths = {item.key: item.path.resolve() for items in conditions.values() for item in items}
    tags_of: dict[str, set[str]] = {}

    for arms in results.values():
        for images in arms.values():
            for key, record in images.items():
                tags_of.setdefault(key, set()).update(set(record["tags"]) - graded.get(key, set()))
    return [{"image": str(paths[key]), "key": key, "tags": sorted(tags_of[key])} for key in sorted(tags_of)
            if tags_of[key]]


def arm_counts(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    """How often an arm hit each cap and ran the echo fallback over one condition."""
    values = list(records.values())

    return {
        "images": len(values),
        "echo_fallbacks": sum(record.get("echo_reply") is not None for record in values),
        "token_cap_hits": sum(record["propose_cap_hit"] for record in values),
        "tag_cap_hits": sum(record.get("proposed_cap_hit", False) for record in values),
        "merge_cap_drops": sum(drop["reason"] == "over the cap" for record in values for drop in record["dropped"]),
    }


def decode_speed(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled and median decode ms per token over the records with at least two steps."""
    rated = [record for record in records if record["decode_ms_per_token"] is not None]
    steps = sum(record["tokens"] - 1 for record in rated)
    decode_ms = sum(record["decode_ms_per_token"] * (record["tokens"] - 1) for record in rated)

    if not rated:
        return {"generates": len(records), "pooled_ms_per_token": None, "median_step_ms": None}
    return {
        "generates": len(records),
        "tokens": sum(record["tokens"] for record in records),
        "pooled_ms_per_token": decode_ms / steps,
        "median_ms_per_token": statistics.median(record["decode_ms_per_token"] for record in rated),
        "median_step_ms": statistics.median(record["median_step_ms"] for record in rated),
        "median_prefill_ms": statistics.median(record["prefill_ms"] for record in rated),
    }


def rounded(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    new_root = args.new_root.resolve()
    conditions = load_conditions(args.manifest, args.coco, args.coco_hashes, args.coco_count, args.limit)
    variants = load_variants(args.variants, args.arms)
    graded = load_graded(args.graded)
    results_path = args.out / f"results{args.suffix}.json"
    settings_path = settings_path_of(results_path)
    meta_path = args.out / f"run_meta{args.suffix}.json"
    thing_cache_path = THING_SCORES_DIR / f"thing_scores_{Path(args.clip).stem}.json"
    results: dict[str, dict[str, dict[str, Any]]] = {}
    recorded_settings: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {"runs": []}
    generates: dict[str, list[dict[str, Any]]] = {arm: [] for arm in args.arms}
    walls: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    prompt_calls: list[float] = []
    prompt_listings: list[Any] = []
    settings_of: dict[str, Any] = {}
    thing_wording_asked: dict[str, Any] = {}
    known_p_other: dict[str, float] = {}
    old = None
    old_classifier = None
    classifier = None

    if OLD_ARM in args.arms and args.old_tagging is None:
        raise SystemExit(f"arm '{OLD_ARM}' needs --old-tagging")
    if args.append:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        taken = sorted({arm for arms in results.values() for arm in arms} & set(args.arms))
        if taken:
            raise SystemExit(f"{results_path} already holds arms {taken}")
        # A results file from before the settings file existed appends without one.
        if settings_path.is_file():
            recorded_settings = json.loads(settings_path.read_text(encoding="utf-8"))

    # The harness modules live beside the new arm, so its ab_env puts that logit_tagger on sys.path.
    sys.path.insert(0, str(new_root / "tests-AB"))
    ab_env = importlib.import_module("ab_env")
    if new_root != ab_env.PACK_ROOT:
        raise SystemExit(f"ab_env.py came from {ab_env.PACK_ROOT}, expected {new_root}")
    ab_env.bootstrap()
    load_image = importlib.import_module("ab_tagger").load_image
    from logit_classifier.toolkit import comfyui
    from logit_classifier.toolkit.comfyui import tagger as toolkit_tagger

    from logit_tagger import engine, tagging

    if Path(tagging.__file__).resolve().parent != new_root / "logit_tagger":
        raise SystemExit(f"logit_tagger resolved to {tagging.__file__}, expected {new_root / 'logit_tagger'}")
    # Built before the model loads, so a misspelled field or a rejected value fails in seconds.
    settings_of = {arm: arm_settings(tagging.tag_settings(), arm, overrides) for arm, overrides in variants.items()}
    if args.record_thing_scores:
        thing_wording_asked, known_p_other = load_thing_cache(thing_cache_path, list(settings_of.values()))
    if OLD_ARM in args.arms:
        old = load_old_tagging(args.old_tagging.resolve())
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_prefetch

    args.out.mkdir(parents=True, exist_ok=True)
    recorded_settings.update({arm: settings_json(settings) for arm, settings in settings_of.items()})
    write_json(settings_path, recorded_settings)
    clip = ab_env.load_clip(args.clip)
    meter = GenerateMeter(transformer_of(clip))
    # Measurement wrappers only. The toolkit's tagger and the pack's tag_images look these names up at call time.
    toolkit_tagger.generate_text = meter.measure(
        toolkit_tagger.generate_text, lambda kwargs: "propose" if kwargs.get("image") is not None else "prompt")
    comfyui.prompt_tags = timed(returning(comfyui.prompt_tags, prompt_listings), prompt_calls)
    if old is not None:
        old.propose = meter.measure(old.propose, lambda _kwargs: "propose")
        old_classifier = old.build_classifier(clip)

    # The first generate loads the weights and captures graphs, so each package tags one image untimed first.
    warm_image = load_image(next(iter(conditions.values()))[0].path)
    if variants:
        tagging.tag_images(clip, [warm_image], [""])
        ab_env.end_node_execution()
    if old is not None:
        old.tag_image(clip, warm_image, "", classifier=old_classifier)
        ab_env.end_node_execution()
    meter.take()

    for condition in args.conditions:
        items = conditions[condition]
        images = [load_image(item.path) for item in items]
        results.setdefault(condition, {})
        walls[condition] = {}
        counts[condition] = {}
        for arm in args.arms:
            started = time.perf_counter()
            if arm == OLD_ARM:
                results[condition][arm] = run_old_arm(old, clip, old_classifier, items, images, meter,
                                                      ab_env.end_node_execution)
            else:
                with tag_settings_of_arm(tagging, settings_of[arm]):
                    results[condition][arm] = run_new_arm(tagging, clip, items, images, meter, prompt_calls,
                                                          prompt_listings, ab_env.end_node_execution)
            walls[condition][arm] = (time.perf_counter() - started) * 1000.0
            counts[condition][arm] = arm_counts(results[condition][arm])
            records = meter.take()
            generates[arm] += records
            write_json(results_path, results)
            speed = decode_speed([record for record in records if record["stage"] == "propose"])
            print(f"{condition} {arm}: {len(items)} images in {walls[condition][arm] / 1000.0:.1f} s, propose decode "
                  f"{rounded(speed['pooled_ms_per_token'])} ms per token, steady step "
                  f"{rounded(speed['median_step_ms'])} ms, {counts[condition][arm]}", flush=True)

    if args.record_thing_scores:
        classifier = engine.build_classifier(clip, SCRIPT_NAME, needs_images=False)
        for arm, settings in settings_of.items():
            record_thing_scores(classifier, [record for condition in args.conditions
                                             for record in results[condition][arm].values()],
                                settings, known_p_other)
        write_json(results_path, results)
        # The cache folder is gitignored, so a fresh checkout lacks it.
        thing_cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(thing_cache_path, {**thing_wording_asked, "p_other": dict(sorted(known_p_other.items()))})
        print(f"thing check p(other) recorded for arms {list(settings_of)}, {len(known_p_other)} tags cached in "
              f"{thing_cache_path}", flush=True)
    write_json(args.out / f"to_judge{args.suffix}.json", judge_list(conditions, results, graded))
    meta["runs"].append({
        "arms": args.arms,
        "variants": variants,
        "clip": args.clip,
        "new_root": str(new_root),
        "old_tagging": None if args.old_tagging is None else str(args.old_tagging.resolve()),
        "dynamic_vram": bool(comfy.memory_management.aimdo_enabled),
        "disable_cuda_graphs": bool(comfy.model_management.args.disable_cuda_graphs),
        "graph_warmed_modules": len(comfy.model_prefetch.GRAPH_WARMED_MODULES),
        "wall_ms": walls,
        "counts": counts,
        "decode": {arm: {stage: decode_speed([record for record in records if record["stage"] == stage])
                         for stage in ("propose", "prompt")}
                   for arm, records in generates.items()},
    })
    write_json(meta_path, meta)


if __name__ == "__main__":
    main()
