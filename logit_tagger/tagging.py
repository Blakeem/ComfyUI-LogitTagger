"""The Logit Tagger's settings and its runs over images through logit_classifier.toolkit.

The tagger tags each image or checks a connected tag list, and the prompt check asks each image about the things
its prompt names.

No comfy, torch or logit_classifier module is imported at module scope, so the unit tests
run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import engine

if TYPE_CHECKING:
    from logit_classifier.toolkit.comfyui import TagSettings

NODE_NAME = "Logit Tagger"
PROMPT_CHECK_NAME = "Logit Prompt Check"

STAGE_WRITING = "writing tags"
STAGE_CHECKING = "checking tags"
STAGE_READING_PROMPT = "reading prompt"
STAGE_CHECKING_THINGS = "checking"

# The toolkit names its stages, and the status line shows the pack's words for them.
_STAGE_WORDS = {"propose": STAGE_WRITING, "verify": STAGE_CHECKING}

# The 80 COCO categories listed on 60 images kept precision 0.971 at recall 0.748, and 0.99 fell to precision
# 0.786 (tests-AB/ab_release_gates.py, probe declared-tags).
TAG_LIST_THRESHOLD = 0.9999

# On 72 Krea 2 renders a thing the image lacked scored below 0.5 at precision 0.976 and recall 0.927
# (tests-AB/ab_release_gates.py, probe adherence).
MISSING_THRESHOLD = 0.5


@dataclass(frozen=True)
class TagTrace:
    """One image's pass through every stage, for the node and the harness to read.

    `origins` is "model", "prompt", "both" or "list" per candidate, `dropped` holds (name, reason) pairs
    from the merge and the thing check, `scores` holds None per candidate when the settings turn verify off,
    and `kept` is in output order. `echo_reply` holds the replaced reply when the echo fallback ran.
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


@dataclass(frozen=True)
class PromptCheck:
    """One image checked against its prompt: the prompt's things, each one's presence probability, the missing
    things in prompt order, and the share of things present, 1.0 when the prompt names none."""

    things: tuple[str, ...]
    probabilities: tuple[float, ...]
    missing: tuple[str, ...]
    score: float


@functools.cache
def tag_settings() -> TagSettings:
    """The Tagger's settings: the toolkit's defaults, which hold this pack's wordings and values, and four overrides.

    A function and not a constant, since TagSettings lives in the library and module scope imports no library.
    """
    try:
        from logit_classifier.toolkit.comfyui import TagSettings
    except ImportError as error:
        raise engine.library_error(NODE_NAME) from error

    return TagSettings(
        # 40 over 25 added 1.2 correct tags per image on 72 Krea 2 renders, at precision 0.918 against 0.926
        # (tests-AB/ab_tagger_compare.py).
        propose_cap=40,
        # The model's 40 and a long prompt's 30 to 40 tags in one packed verify pass. Model tags come first.
        merge_cap=80,
        # On the model's tags the thing check dropped "watermark" and "pixel art", which the judges graded correct
        # (tests-AB/ab_tagger_compare.py), so it judges the prompt's tags only.
        thing_check="prompt",
        # The "words" rule dropped "cat" beside "cat ear", 20% of the correct tags the port lost
        # (tests-AB/ab_tagger_compare.py). The "head-noun" rule keeps it.
        subsets_rule="head-noun",
        order="score",
    )


def _check_one_image(image: Any, node: str = NODE_NAME) -> None:
    engine.check_images(image, node)
    if image.shape[0] != 1:
        raise ValueError(
            f"{node}: each image must be one image of shape [1, H, W, C], got a batch of "
            f"{image.shape[0]}. Split the batch into single images first."
        )


def tag_image(clip: Any, classifier: Any, image: Any, *, prompt_tags: tuple[str, ...] = (),
              instruction: str | None = None, on_stage: Callable[[str], None] | None = None,
              known: dict[str, float], settings: TagSettings | None = None) -> TagTrace:
    """Run every stage on one [1, H, W, C] image and report what each stage produced.

    known carries the thing check answers across a run. settings None means tag_settings().
    """
    base_settings = tag_settings() if settings is None else settings
    image_settings = base_settings
    hear_stage: Callable[[str], None] | None = None
    picture = None
    trace = None

    _check_one_image(image)
    from logit_classifier.toolkit.comfyui import fit_picture, tag_picture

    if instruction is not None:
        # The echo fallback's own wording would replace the connected instruction.
        image_settings = dataclasses.replace(base_settings, propose_instruction=instruction,
                                             echo_fallback_instruction=None)
    if on_stage is not None:
        hear_stage = lambda stage: on_stage(_STAGE_WORDS[stage])
    picture = fit_picture(image, engine.VL_MAX_PIXELS)
    trace = tag_picture(clip, classifier, picture, prompt_tags=prompt_tags, settings=image_settings, known=known,
                        on_stage=hear_stage)
    return TagTrace(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(picture.shape[1]), int(picture.shape[2])),
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


def parse_tag_list(text: str) -> tuple[str, ...]:
    """The cleaned tags of a comma or newline separated list, in order, each once."""
    parts: list[str] = []

    if not text.strip():
        return ()
    # The node parses the list before build_classifier checks the library.
    try:
        from logit_classifier.toolkit.tags import clean_item
    except ImportError as error:
        raise engine.library_error(NODE_NAME) from error

    parts = [clean_item(part) for part in re.split(r"[,\n]", text)]
    return tuple(dict.fromkeys(part for part in parts if part))


def check_tag_list(clip: Any, classifier: Any, image: Any, tags: tuple[str, ...]) -> TagTrace:
    """Ask the presence question of every listed tag about one [1, H, W, C] image and keep the tags it shows."""
    picture = None
    questions: list[str] = []
    scores: tuple[float, ...] = ()
    passed: list[tuple[str, float]] = []
    kept: tuple[str, ...] = ()
    started = 0.0
    verify_ms = 0.0

    _check_one_image(image)
    from logit_classifier.toolkit.comfyui import fit_picture, shared_vision_encode
    from logit_classifier.toolkit.tags import presence_question

    picture = fit_picture(image, engine.CHECK_MAX_PIXELS)
    questions = [presence_question(tag) for tag in tags]
    started = time.perf_counter()
    # nouls splits a long list into several requests, and one scope over them encodes the picture once.
    with shared_vision_encode(clip):
        scores = tuple(classifier.nouls(questions, image=picture))
    verify_ms = (time.perf_counter() - started) * 1000.0

    passed = [(tag, p) for tag, p in zip(tags, scores, strict=True) if p >= TAG_LIST_THRESHOLD]
    # sorted is stable, so tied tags keep their list order.
    kept = tuple(tag for tag, _p in sorted(passed, key=lambda pair: pair[1], reverse=True))
    return TagTrace(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(picture.shape[1]), int(picture.shape[2])),
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


def tag_images(clip: Any, images: list[Any], prompts: list[str], *, instruction: str | None = None,
               tag_list: tuple[str, ...] = (), on_progress: Callable[[int, str], None] | None = None,
               settings: TagSettings | None = None) -> list[TagTrace]:
    """One TagTrace per [1, H, W, C] image, in order, checked against tag_list when it holds tags.

    Otherwise each image is tagged with the prompt at its index. settings None means tag_settings().
    """
    run_settings = None
    classifier = None
    known: dict[str, float] = {}
    things_by_prompt: dict[str, tuple[str, ...]] = {}
    on_stage: Callable[[str], None] | None = None
    traces: list[TagTrace] = []

    if len(images) != len(prompts):
        raise ValueError(
            f"{NODE_NAME}: got {len(images)} images and {len(prompts)} prompts. Give one prompt per "
            "image, an empty one for an image without a prompt."
        )
    for image in images:
        _check_one_image(image)
    # Built before any generate, so a wrong CLIP fails before the model runs.
    classifier = engine.build_classifier(clip, NODE_NAME)
    if tag_list:
        for index, image in enumerate(images):
            if on_progress is not None:
                on_progress(index, STAGE_CHECKING)
            traces.append(check_tag_list(clip, classifier, image, tag_list))
        return traces
    from logit_classifier.toolkit.comfyui import prompt_tags

    run_settings = tag_settings() if settings is None else settings
    for index, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
        # Read at its first image, so the prompt's generate runs under that image's progress share and status.
        if prompt not in things_by_prompt:
            if on_progress is not None and prompt.strip():
                on_progress(index, STAGE_READING_PROMPT)
            things_by_prompt[prompt] = prompt_tags(clip, classifier, prompt, run_settings, known).tags
        on_stage = None if on_progress is None else (lambda stage, index=index: on_progress(index, stage))
        traces.append(tag_image(clip, classifier, image, prompt_tags=things_by_prompt[prompt],
                                instruction=instruction, on_stage=on_stage, known=known, settings=run_settings))
    return traces


def _presence_probabilities(classifier: Any, image: Any, things: tuple[str, ...]) -> tuple[float, ...]:
    """The presence probability of every thing about image, or () and no request without things."""
    picture = None
    questions: list[str] = []

    if not things:
        return ()
    from logit_classifier.toolkit.comfyui import fit_picture
    from logit_classifier.toolkit.tags import presence_question

    picture = fit_picture(image, engine.CHECK_MAX_PIXELS)
    questions = [presence_question(thing) for thing in things]
    return tuple(classifier.nouls(questions, image=picture))


def check_prompt(clip: Any, images: list[Any], prompts: list[str], *,
                 on_progress: Callable[[int, str], None] | None = None) -> list[PromptCheck]:
    """One PromptCheck per [1, H, W, C] image, in order, asking it about the things its paired prompt names."""
    announce = on_progress if on_progress is not None else (lambda _index, _stage: None)
    run_settings = None
    classifier = None
    known: dict[str, float] = {}
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
    from logit_classifier.toolkit.comfyui import prompt_tags

    run_settings = tag_settings()
    for index, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
        if prompt not in things_by_prompt:
            announce(index, STAGE_READING_PROMPT)
            things_by_prompt[prompt] = prompt_tags(clip, classifier, prompt, run_settings, known).tags
        things = things_by_prompt[prompt]
        announce(index, STAGE_CHECKING_THINGS)
        probabilities = _presence_probabilities(classifier, image, things)
        missing = tuple(thing for thing, p in zip(things, probabilities, strict=True) if p < MISSING_THRESHOLD)
        score = (len(things) - len(missing)) / len(things) if things else 1.0
        checks.append(PromptCheck(things=things, probabilities=probabilities, missing=missing, score=score))
    return checks
