"""A/B the Logit Tagger pipeline over a real Qwen3-VL-4B: the package at --new-root, its variants and an older tagging.py.

Arm "new" runs logit_tagger.tagging.tag_images once per condition, as one node execution does. A variant arm
does the same with the tagging constants its --variants entry overrides. Arm "old" loads --old-tagging, a file
from `git show <rev>:logit_tagger/tagging.py`, and runs its tag_image once per image, one node execution per
distinct prompt, since that node took one prompt per batch.
Writes results<suffix>.json (every image's stages per condition and arm), to_judge<suffix>.json (each image's
tags from every arm and condition that no --graded file grades yet, unlabeled and sorted, for blind judges) and
run_meta<suffix>.json (decode speed, walls, cap and fallback counts). Judges grade to_judge by
data/tagger-ab/RUBRIC.md, and ab_tagger_score.py scores the arms against their verdicts.
Usage: python ab_tagger_compare.py [--out DIR] [--arms new old N1 ...] [--old-tagging OLD.py] [--variants V.json]
       [--graded data/tagger-ab/verdicts.json]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
DEFAULT_MANIFEST = HERE / "data" / "release-gates" / "renders" / "manifest.json"
DEFAULT_COCO = HERE / "cache" / "coco60" / "coco60.json"
DEFAULT_OUT = HERE / "out" / "tagger-ab"
CONDITIONS = ("render-noprompt", "render-prompt", "coco")
OLD_ARM = "old"
NEW_ARM = "new"


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
                        help="JSON of {arm: {tagging constant: value}} for arms that vary the new package")
    parser.add_argument("--graded", type=Path, nargs="*", default=[],
                        help="verdict files whose (image, tag) pairs to_judge leaves out")
    parser.add_argument("--new-root", type=Path, default=HERE.parent,
                        help="directory holding the new arm's logit_tagger and tests-AB/ab_env.py")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="the release-gate renders' manifest.json")
    parser.add_argument("--coco", type=Path, default=DEFAULT_COCO, help="coco60.json, its file paths relative to it")
    parser.add_argument("--coco-count", type=int, default=30, help="how many COCO images, from the first")
    parser.add_argument("--limit", type=int, default=None, help="tag only the first N images of each condition")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    return parser.parse_args()


def load_conditions(manifest_path: Path, coco_path: Path, coco_count: int, limit: int | None) -> dict[str, list[Item]]:
    """Every condition's images in run order."""
    renders = json.loads(manifest_path.read_text(encoding="utf-8"))
    coco = json.loads(coco_path.read_text(encoding="utf-8"))["images"][:coco_count]
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
    return {name: items[:limit] for name, items in conditions.items()}


def load_variants(path: Path | None, arms: list[str]) -> dict[str, dict[str, Any]]:
    """Each arm's tagging overrides, empty for 'new'. Rejects an arm that is neither built in nor a variant."""
    variants = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
    unknown = [arm for arm in arms if arm not in (NEW_ARM, OLD_ARM) and arm not in variants]

    if unknown:
        raise SystemExit(f"arms {unknown} are neither '{NEW_ARM}', '{OLD_ARM}' nor a name in {path}")
    return {arm: variants.get(arm, {}) for arm in arms if arm != OLD_ARM}


def apply_overrides(module: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Set each named module constant and return the values it replaced. A list becomes a frozenset where one was."""
    previous: dict[str, Any] = {}

    for name, value in overrides.items():
        if not hasattr(module, name):
            raise SystemExit(f"{module.__name__} has no constant {name} to override")
        previous[name] = getattr(module, name)
        setattr(module, name, frozenset(value) if isinstance(previous[name], frozenset) else value)
    return previous


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


def new_record(trace: Any, prompt_ms: float, proposes: list[dict[str, Any]], tagging: Any) -> dict[str, Any]:
    """The record of one new-package trace. proposes holds its propose generates, the used reply's last."""
    kept = set(trace.kept)
    dropped = [{"tag": name, "reason": reason} for name, reason in trace.dropped]
    used = proposes[-1]
    threshold = 0.0

    for tag, origin, score in zip(trace.candidates, trace.origins, trace.scores, strict=True):
        threshold = tagging.PROMPT_ONLY_THRESHOLD if origin == "prompt" else tagging.VERIFY_THRESHOLD
        if tag in kept:
            continue
        reason = f"below the {threshold} verify threshold" if score < threshold else "its words sit inside a kept tag"
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
        "propose_cap_hit": used["tokens"] >= tagging.PROPOSE_MAX_TOKENS,
        "proposed_cap_hit": len(trace.proposed) >= tagging.MAX_PROPOSED_TAGS,
        "size_in": list(trace.size_in),
        "size_out": list(trace.size_out),
        "proposed": list(trace.proposed),
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
                prompt_calls: list[float], end_node: Callable[[], None]) -> dict[str, dict[str, Any]]:
    """One tag_images call over the condition, as one node execution, then one record per image."""
    prompts = [item.prompt for item in items]
    distinct = list(dict.fromkeys(prompts))
    traces: list[Any] = []
    proposes: list[dict[str, Any]] = []
    prompt_ms_of: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    cursor = 0

    prompt_calls.clear()
    traces = tagging.tag_images(clip, images, prompts)
    end_node()
    proposes = [record for record in meter.records if record["stage"] == "propose"]
    # The prompt stage runs once per distinct prompt, so that prompt's images share its time.
    prompt_ms_of = {prompt: ms / prompts.count(prompt) for prompt, ms in zip(distinct, prompt_calls, strict=True)}
    for item, trace in zip(items, traces, strict=True):
        # The echo fallback proposes a second time.
        used = 2 if getattr(trace, "echo_reply", None) is not None else 1
        records[item.key] = new_record(trace, prompt_ms_of[item.prompt], proposes[cursor:cursor + used], tagging)
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
    conditions = load_conditions(args.manifest, args.coco, args.coco_count, args.limit)
    variants = load_variants(args.variants, args.arms)
    graded = load_graded(args.graded)
    results_path = args.out / f"results{args.suffix}.json"
    meta_path = args.out / f"run_meta{args.suffix}.json"
    results: dict[str, dict[str, dict[str, Any]]] = {}
    meta: dict[str, Any] = {"runs": []}
    generates: dict[str, list[dict[str, Any]]] = {arm: [] for arm in args.arms}
    walls: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    prompt_calls: list[float] = []
    old = None
    old_classifier = None

    if OLD_ARM in args.arms and args.old_tagging is None:
        raise SystemExit(f"arm '{OLD_ARM}' needs --old-tagging")
    if args.append:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        taken = sorted({arm for arms in results.values() for arm in arms} & set(args.arms))
        if taken:
            raise SystemExit(f"{results_path} already holds arms {taken}")

    # The harness modules live beside the new arm, so its ab_env puts that logit_tagger on sys.path.
    sys.path.insert(0, str(new_root / "tests-AB"))
    ab_env = importlib.import_module("ab_env")
    if new_root != ab_env.PACK_ROOT:
        raise SystemExit(f"ab_env.py came from {ab_env.PACK_ROOT}, expected {new_root}")
    ab_env.bootstrap()
    load_image = importlib.import_module("ab_tagger").load_image
    from logit_tagger import engine, tagging

    if Path(tagging.__file__).resolve().parent != new_root / "logit_tagger":
        raise SystemExit(f"logit_tagger resolved to {tagging.__file__}, expected {new_root / 'logit_tagger'}")
    # Fails on a misspelled constant before the model loads.
    for overrides in variants.values():
        apply_overrides(tagging, apply_overrides(tagging, overrides))
    if OLD_ARM in args.arms:
        old = load_old_tagging(args.old_tagging.resolve())
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_prefetch

    args.out.mkdir(parents=True, exist_ok=True)
    clip = ab_env.load_clip(args.clip)
    meter = GenerateMeter(transformer_of(clip))
    # Measurement wrappers only. Each arm's module looks these names up at call time.
    engine.generate_text = meter.measure(engine.generate_text,
                                         lambda kwargs: "propose" if kwargs.get("image") is not None else "prompt")
    tagging.prompt_things = timed(tagging.prompt_things, prompt_calls)
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
                previous = apply_overrides(tagging, variants[arm])
                try:
                    results[condition][arm] = run_new_arm(tagging, clip, items, images, meter, prompt_calls,
                                                          ab_env.end_node_execution)
                finally:
                    apply_overrides(tagging, previous)
            walls[condition][arm] = (time.perf_counter() - started) * 1000.0
            counts[condition][arm] = arm_counts(results[condition][arm])
            records = meter.take()
            generates[arm] += records
            write_json(results_path, results)
            speed = decode_speed([record for record in records if record["stage"] == "propose"])
            print(f"{condition} {arm}: {len(items)} images in {walls[condition][arm] / 1000.0:.1f} s, propose decode "
                  f"{rounded(speed['pooled_ms_per_token'])} ms per token, steady step "
                  f"{rounded(speed['median_step_ms'])} ms, {counts[condition][arm]}", flush=True)

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
