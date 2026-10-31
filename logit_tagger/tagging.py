"""The Logit Tagger's settings and its runs over images through logit_classifier.toolkit.

The Tagger tags each image, and Logit Custom Tagger tags it by the user's own instruction. Logit Tag Check checks
each image against a connected tag list. The prompt check asks each image about the things its prompt names.

No comfy, torch or logit_classifier module is imported at module scope, so the unit tests
run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

import functools
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from . import engine

if TYPE_CHECKING:
    from logit_classifier.toolkit.comfyui import TagSettings

NODE_NAME = "Logit Tagger"
CUSTOM_TAGGER_NAME = "Logit Custom Tagger"
TAG_CHECK_NAME = "Logit Tag Check"
PROMPT_CHECK_NAME = "Logit Prompt Check"

STAGE_WRITING = "writing tags"
STAGE_CHECKING = "checking tags"
STAGE_READING_PROMPT = "reading prompt"
STAGE_CHECKING_THINGS = "checking"
STAGE_PLACING = "placing tags"

# The toolkit names its stages, and the status line shows the pack's words for them.
_STAGE_WORDS = {"propose": STAGE_WRITING, "verify": STAGE_CHECKING}

# On 72 Krea 2 renders under "This image visibly contains {TAG}", 0.01 flagged a missing thing at precision 0.976 and
# recall 0.931. It was chosen on those renders (tests-AB/ab_release_gates.py, probe adherence).
MISSING_THRESHOLD = 0.01

# The toolkit's locate_tags splits each side into thirds and rejects a side below 3 pixels, in an error that names no
# node.
PLACE_MIN_SIDE = 3


@dataclass(frozen=True)
class TagTrace:
    """One image's pass through every stage, for the node and the harness to read.

    `origins` is "model", "prompt" or "both" per candidate the Tagger merged, and "list" per tag Logit Tag Check
    was given. `dropped` holds (name, reason) pairs from the merge and the thing check, `scores` holds None per
    candidate when the settings turn verify off, and `kept` is in output order. `echo_reply` holds the replaced reply when the echo fallback ran.
    With positions on, `terms` holds one position term per kept tag, "" for none, and `unplaced` the kept tags no
    strip holds, which `tags` leaves out.
    """

    size_in: tuple[int, int]
    size_out: tuple[int, int]
    reply: str
    proposed: tuple[str, ...]
    candidates: tuple[str, ...]
    origins: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]
    scores: tuple[float | None, ...]
    kept: tuple[str, ...]
    tags: str
    propose_ms: float
    verify_ms: float
    echo_reply: str | None = None
    terms: tuple[str, ...] = ()
    unplaced: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptCheck:
    """One image checked against its prompt: the prompt's things, each one's presence probability, the missing
    things in prompt order, and the share of things present, 1.0 when the prompt names none."""

    things: tuple[str, ...]
    probabilities: tuple[float, ...]
    missing: tuple[str, ...]
    score: float


def tag_settings(strict: bool = False, *, node: str = NODE_NAME) -> TagSettings:
    """The Tagger's settings: the toolkit's defaults, which hold its broad thresholds, or its strict thresholds when
    strict is on. A missing or outdated library raises the pip command under node's name.

    On the fp8 encoder the defaults kept 12.8 correct, 0.12 wrong and 0.61 vague tags per render, and the strict
    thresholds 8.4, 0.01 and 0.33 (tests-AB/ab_tagger_score.py --replay, runs/2026-09-27-settings). A function, since
    module scope imports no library.
    """
    engine.toolkit(node)
    return _cached_settings(bool(strict))


@functools.cache
def _cached_settings(strict: bool) -> TagSettings:
    # Keyed on the bool alone, so every call form of tag_settings returns the one object per setting.
    from logit_classifier.toolkit.comfyui import TagSettings
    from logit_classifier.toolkit.tags import STRICT_THRESHOLDS

    if not strict:
        return TagSettings()
    return TagSettings(verify_thresholds=STRICT_THRESHOLDS)


def custom_settings(instruction: str, *, strict: bool, things_only: bool) -> TagSettings:
    """Logit Custom Tagger's settings, which are the Tagger's strict or broad settings proposing with instruction, or
    with the default wording when instruction is blank.

    things_only off judges only a prompt's tags, and the node passes no prompt. The echo fallback and the category
    noun drop both answer the default wording's own nouns coming back as tags, so another instruction runs without
    either.
    """
    base = tag_settings(strict, node=CUSTOM_TAGGER_NAME)
    wording = instruction.strip() or base.propose_instruction
    is_default = wording == base.propose_instruction

    return replace(
        base,
        propose_instruction=wording,
        thing_check="all" if things_only else "prompt",
        echo_fallback_instruction=base.echo_fallback_instruction if is_default else None,
        category_nouns=base.category_nouns if is_default else frozenset(),
    )


def _check_one_image(image: Any, node: str = NODE_NAME) -> None:
    engine.check_images(image, node)
    if image.shape[0] != 1:
        raise ValueError(
            f"{node}: each image must be one image of shape [1, H, W, C], got a batch of "
            f"{image.shape[0]}. Split the batch into single images first."
        )


def _check_placeable(image: Any, node: str) -> None:
    height, width = int(image.shape[1]), int(image.shape[2])

    if height < PLACE_MIN_SIDE or width < PLACE_MIN_SIDE:
        raise ValueError(
            f"{node}: positions needs an image at least {PLACE_MIN_SIDE} pixels on each side, got height {height} and "
            f"width {width}. Turn positions off or connect a larger image."
        )


def _place(classifier: Any, image: Any, trace: TagTrace, settings: TagSettings) -> TagTrace:
    """The trace with each kept tag's position term, its unplaced tags and its tags rendered with their terms.

    image is the node's input at full resolution, since the toolkit resamples each strip to its own pixel count. A
    trace with no kept tags returns unchanged and asks nothing.
    """
    places = None

    if not trace.kept:
        return trace
    from logit_classifier.toolkit.comfyui import locate_tags, render_placed

    places = locate_tags(classifier, image, trace.kept, statement=settings.verify_statement)
    return replace(trace, terms=places.terms, unplaced=places.unplaced, tags=render_placed(places.placed))


def tag_image(clip: Any, classifier: Any, image: Any, *, prompt_tags: tuple[str, ...] = (),
              on_stage: Callable[[str], None] | None = None, thing_cache: dict[str, float],
              settings: TagSettings | None = None) -> TagTrace:
    """Run every stage on one [1, H, W, C] image and report what each stage produced.

    thing_cache carries the thing check answers across a run. settings None means tag_settings().
    """
    image_settings = tag_settings() if settings is None else settings
    hear_stage: Callable[[str], None] | None = None
    fitted_image = None
    trace = None

    _check_one_image(image)
    # Through the module, since this function's own name is tag_image.
    from logit_classifier.toolkit import comfyui as toolkit

    if on_stage is not None:
        hear_stage = lambda stage: on_stage(_STAGE_WORDS[stage])
    fitted_image = toolkit.fit_image(image, engine.VL_MAX_PIXELS)
    trace = toolkit.tag_image(clip, classifier, fitted_image, prompt_tags=prompt_tags, settings=image_settings,
                              thing_cache=thing_cache, on_stage=hear_stage)
    return TagTrace(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(fitted_image.shape[1]), int(fitted_image.shape[2])),
        reply=trace.reply,
        proposed=trace.proposed,
        candidates=trace.candidates,
        origins=trace.origins,
        dropped=trace.dropped,
        scores=trace.scores,
        kept=trace.kept,
        tags=", ".join(trace.kept),
        propose_ms=trace.propose_ms,
        verify_ms=trace.verify_ms,
        echo_reply=trace.echo_reply,
    )


def parse_tag_list(text: str, *, node: str = NODE_NAME) -> tuple[str, ...]:
    """The cleaned tags of a comma or newline separated list, in order, each once."""
    parts: list[str] = []

    if not text.strip():
        return ()
    # The node parses the list before build_classifier checks the library.
    engine.toolkit(node)
    from logit_classifier.toolkit.tags import clean_item

    parts = [clean_item(part) for part in re.split(r"[,\n]", text)]
    return tuple(dict.fromkeys(part for part in parts if part))


def check_tag_list(clip: Any, classifier: Any, image: Any, tags: tuple[str, ...], settings: TagSettings) -> TagTrace:
    """Score every listed tag about one [1, H, W, C] image in the settings' verify statement, and keep the tags that
    reach the settings' declared threshold, since a listed tag is one the encoder did not propose.

    On the 80 COCO categories 0.9999 kept precision 0.903 at recall 0.756, and 0.99998 kept 0.950 at 0.422. 8 of the 11
    wrong tags at 0.9999 were "orange" (tests-AB/ab_release_gates.py, probe declared-tags).
    """
    fitted_image = None
    questions: list[str] = []
    scores: tuple[float, ...] = ()
    passed: list[tuple[str, float]] = []
    kept: tuple[str, ...] = ()
    started = 0.0
    verify_ms = 0.0

    _check_one_image(image, TAG_CHECK_NAME)
    from logit_classifier.toolkit.comfyui import fit_image, shared_vision_encode
    from logit_classifier.toolkit.tags import presence_statement

    fitted_image = fit_image(image, engine.CHECK_MAX_PIXELS)
    questions = [presence_statement(tag, wording=settings.verify_statement) for tag in tags]
    started = time.perf_counter()
    # nouls splits a long list into several requests, and one scope over them encodes the image once.
    with shared_vision_encode(clip):
        scores = tuple(classifier.nouls(questions, image=fitted_image))
    verify_ms = (time.perf_counter() - started) * 1000.0

    passed = [(tag, p) for tag, p in zip(tags, scores, strict=True) if p >= settings.verify_thresholds.declared]
    # sorted is stable, so tied tags keep their list order.
    kept = tuple(tag for tag, _p in sorted(passed, key=lambda pair: pair[1], reverse=True))
    return TagTrace(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(fitted_image.shape[1]), int(fitted_image.shape[2])),
        reply="",
        proposed=(),
        candidates=tuple(tags),
        origins=("list",) * len(tags),
        dropped=(),
        scores=scores,
        kept=kept,
        tags=", ".join(kept),
        propose_ms=0.0,
        verify_ms=verify_ms,
    )


def check_tag_lists(clip: Any, images: list[Any], tag_lists: list[tuple[str, ...]], *, settings: TagSettings,
                    on_progress: Callable[[int, str], None] | None = None,
                    positions: bool = False) -> list[TagTrace]:
    """One TagTrace per [1, H, W, C] image, in order, each checked against the tag list at its index and placed on
    its strips when positions is on."""
    classifier = None
    trace: TagTrace | None = None
    traces: list[TagTrace] = []

    if len(images) != len(tag_lists):
        raise ValueError(
            f"{TAG_CHECK_NAME}: got {len(images)} images and {len(tag_lists)} tag lists. Give one tag list per image."
        )
    for image in images:
        _check_one_image(image, TAG_CHECK_NAME)
        if positions:
            _check_placeable(image, TAG_CHECK_NAME)
    classifier = engine.build_classifier(clip, TAG_CHECK_NAME)
    from logit_classifier.toolkit.comfyui import skip_resident_loads

    with skip_resident_loads(clip):
        for index, (image, tags) in enumerate(zip(images, tag_lists, strict=True)):
            if on_progress is not None:
                on_progress(index, STAGE_CHECKING)
            trace = check_tag_list(clip, classifier, image, tags, settings)
            if positions:
                if on_progress is not None:
                    on_progress(index, STAGE_PLACING)
                trace = _place(classifier, image, trace, settings)
            traces.append(trace)
    return traces


def tag_images(clip: Any, images: list[Any], prompts: list[str], *,
               on_progress: Callable[[int, str], None] | None = None,
               settings: TagSettings | None = None, node: str = NODE_NAME,
               positions: bool = False) -> list[TagTrace]:
    """One TagTrace per [1, H, W, C] image, in order, each tagged with the prompt at its index and placed on its
    strips when positions is on.

    settings None means tag_settings(). Errors name node.
    """
    run_settings = None
    classifier = None
    trace: TagTrace | None = None
    thing_cache: dict[str, float] = {}
    things_by_prompt: dict[str, tuple[str, ...]] = {}
    on_stage: Callable[[str], None] | None = None
    traces: list[TagTrace] = []

    if len(images) != len(prompts):
        raise ValueError(
            f"{node}: got {len(images)} images and {len(prompts)} prompts. Give one prompt per "
            "image, an empty one for an image without a prompt."
        )
    for image in images:
        _check_one_image(image, node)
        if positions:
            _check_placeable(image, node)
    # Built before any generate, so a wrong CLIP fails before the model runs.
    classifier = engine.build_classifier(clip, node)
    run_settings = tag_settings(node=node) if settings is None else settings
    from logit_classifier.toolkit.comfyui import skip_resident_loads, tag_prompt

    # Core's load_models_gpu costs about 0.1 s per call on a resident model (ContextAnchoredTileRefine
    # tests-AB/tags-bench-log.md, section 4), and core's CLIP.generate calls load_model on every generate (comfy/sd.py).
    with skip_resident_loads(clip):
        for index, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
            # Read at its first image, so the prompt's generate runs under that image's progress share and status.
            if prompt not in things_by_prompt:
                if on_progress is not None and prompt.strip():
                    on_progress(index, STAGE_READING_PROMPT)
                things_by_prompt[prompt] = tag_prompt(clip, classifier, prompt, run_settings, thing_cache).tags
            on_stage = None if on_progress is None else (lambda stage, index=index: on_progress(index, stage))
            trace = tag_image(clip, classifier, image, prompt_tags=things_by_prompt[prompt], on_stage=on_stage,
                              thing_cache=thing_cache, settings=run_settings)
            if positions:
                if on_progress is not None:
                    on_progress(index, STAGE_PLACING)
                trace = _place(classifier, image, trace, run_settings)
            traces.append(trace)
    return traces


def _presence_probabilities(classifier: Any, image: Any, things: tuple[str, ...]) -> tuple[float, ...]:
    """The presence probability of every thing about image, or () and no request without things."""
    fitted_image = None
    questions: list[str] = []

    if not things:
        return ()
    from logit_classifier.toolkit.comfyui import fit_image
    from logit_classifier.toolkit.tags import presence_statement

    fitted_image = fit_image(image, engine.CHECK_MAX_PIXELS)
    questions = [presence_statement(thing) for thing in things]
    return tuple(classifier.nouls(questions, image=fitted_image))


def check_prompt(clip: Any, images: list[Any], prompts: list[str], *,
                 on_progress: Callable[[int, str], None] | None = None) -> list[PromptCheck]:
    """One PromptCheck per [1, H, W, C] image, in order, asking it about the things its paired prompt names."""
    announce = on_progress if on_progress is not None else (lambda _index, _stage: None)
    run_settings = None
    classifier = None
    thing_cache: dict[str, float] = {}
    things_by_prompt: dict[str, tuple[str, ...]] = {}
    things: tuple[str, ...] = ()
    probabilities: tuple[float, ...] = ()
    missing: tuple[str, ...] = ()
    score = 0.0
    checks: list[PromptCheck] = []

    if len(images) != len(prompts):
        raise ValueError(
            f"{PROMPT_CHECK_NAME}: got {len(images)} images and {len(prompts)} prompts. Give one prompt per image."
        )
    for image in images:
        _check_one_image(image, PROMPT_CHECK_NAME)
    # Built before any generate, so a wrong CLIP fails before the model runs.
    classifier = engine.build_classifier(clip, PROMPT_CHECK_NAME)
    from logit_classifier.toolkit.comfyui import skip_resident_loads, tag_prompt

    run_settings = tag_settings(node=PROMPT_CHECK_NAME)
    with skip_resident_loads(clip):
        for index, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
            if prompt not in things_by_prompt:
                announce(index, STAGE_READING_PROMPT)
                things_by_prompt[prompt] = tag_prompt(clip, classifier, prompt, run_settings, thing_cache).tags
            things = things_by_prompt[prompt]
            announce(index, STAGE_CHECKING_THINGS)
            probabilities = _presence_probabilities(classifier, image, things)
            missing = tuple(thing for thing, p in zip(things, probabilities, strict=True) if p < MISSING_THRESHOLD)
            score = (len(things) - len(missing)) / len(things) if things else 1.0
            checks.append(PromptCheck(things=things, probabilities=probabilities, missing=missing, score=score))
    return checks
