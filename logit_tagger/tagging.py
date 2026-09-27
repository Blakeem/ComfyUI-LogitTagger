"""The Logit Tagger pipeline for one whole image: propose, merge, verify and clean, or a tag list check.

The prompt check lists a prompt's things with the pipeline's prompt stage and asks each image about them.

No comfy, torch or logit_classifier module is imported at module scope, so the unit tests
run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import engine

NODE_NAME = "Logit Tagger"
PROMPT_CHECK_NAME = "Logit Prompt Check"

STAGE_WRITING = "writing tags"
STAGE_CHECKING = "checking tags"
STAGE_READING_PROMPT = "reading prompt"
STAGE_CHECKING_THINGS = "checking"

PROMPT_PLACEHOLDER = "{PROMPT}"
TAG_PLACEHOLDER = "{TAG}"

# The fastest of eight wordings on 28 images in tests-AB/ab_propose_prompts.py, with the fewest rejected
# tags. No wording cut the loops, which followed the image.
PROPOSE_INSTRUCTION = (
    "Tag this image. List every distinct visible thing in it: objects, people, animals, "
    "clothing, materials, and the setting. Use short lowercase noun phrases separated by "
    "commas. List each thing once. Do not list moods, styles, or ideas. Output only the tags."
)

# A reply that echoed only the propose instruction's category nouns left 4 of 174 images without a tag
# (tests-AB/ab_tagger_compare.py). This wording names no category.
ECHO_FALLBACK_INSTRUCTION = (
    "List the things you can see in this image as short lowercase noun phrases separated by commas. "
    "Output only the list."
)

# Asked text only, so a long prompt never lengthens the image's own reply. It must hold {PROMPT}.
PROMPT_TAGS_INSTRUCTION = (
    "This is a prompt for an image:\n{PROMPT}\n\nList the physical things the prompt names that "
    "could be pointed at in the image: objects, people, animals, plants, clothing, materials, "
    "buildings and parts of the scene. Use short lowercase noun phrases of one to four words, "
    "separated by commas. Keep each thing's own descriptive words, such as its color. Leave out "
    "places and settings, actions, sizes, moods, the image's style, medium, quality, camera and "
    "lighting, artist names and position words. List each thing once. Output only the list."
)

# It read 0.000 to 0.027 on 13 prompts in 10 other languages and 1.000 on 10 English ones, style-only prompts
# included (tests-AB/ab_prompt_language.py).
LANGUAGE_QUESTION = "Is this text written in English?"
ENGLISH_THRESHOLD = 0.5

# Grounding and its padding stop match the prompt's own words, so a non-English prompt is listed from its English
# translation. It must hold {PROMPT}.
TRANSLATE_INSTRUCTION = "Translate this image prompt into English. Output only the translation.\n\n{PROMPT}"

# About 380 English words, so only a very long prompt loses its tail.
TRANSLATE_MAX_TOKENS = 512

# In tests-AB/ab_propose_budget.py both images that reached 128 were repeating tags, and 192 added no tag.
PROPOSE_MAX_TOKENS = 128

# 40 over 25 added 1.2 correct tags per image on 72 Krea 2 renders, at precision 0.918 against 0.926
# (tests-AB/ab_tagger_compare.py).
MAX_PROPOSED_TAGS = 40

# The model's 40 and a long prompt's 30 to 40 tags in one packed verify pass. Model tags come first.
MAX_MERGED_TAGS = 80

# The propose instruction's own nouns came back as tags and passed verify.
CATEGORY_NOUNS = frozenset({"objects", "people", "animals", "clothing", "materials", "setting"})

# A 500 word prompt listed 28 to 45 tags in 70 to 256 tokens (ContextAnchoredTileRefine tests-AB/tags-bench-log.md).
PROMPT_TAGS_MAX_TOKENS = 256

# Past the prompt's own things the model pads the list with words the prompt lacks, so two in a row end it.
UNGROUNDED_STREAK = 2
_STOP_WORDS = frozenset({"a", "an", "the", "of", "with", "and", "in", "on", "at", "to", "for", "by", "from"})

# Adapted from the library's measured crop wording "This crop visibly contains {f}", AUC 0.963
# in its tests-AB/ab_multi_label.py.
VERIFY_STATEMENT = "This image visibly contains {TAG}"

# Tags scoring 0.5 to 0.9 on the reference pack's storm sky tile named a bay and buildings the tile lacks.
VERIFY_THRESHOLD = 0.9
# A near name passes 0.9. 0.9999 cut wrong tags per tile 0.92 to 0.29 (ContextAnchoredTileRefine tags-bench-log.md).
PROMPT_ONLY_THRESHOLD = 0.9999

# No verify threshold separates a prompt's scene, lighting, color and quality words from its things. On model tags
# it dropped "watermark" and "pixel art", graded correct (tests-AB/ab_tagger_compare.py), so it judges prompts only.
THING_QUESTION = 'What does the image tag "{TAG}" name'
THING_CRITERIA = {
    "thing": "something that could be pointed at in a picture: an object, a person, a group of people "
             "or animals, an animal, a plant, a body part, a garment, a material, a building, a "
             "landscape feature such as hills, cliffs, sky, the moon or stars, a light source such as "
             "lamps, signs or lit windows, or a substance such as water, smoke, fire, clouds or rain",
    "other": "nothing to point at: a kind of place or a whole scene, a time of day, a color alone, a "
             "shape or form, lighting in general, a camera or render effect, a style, a mood, a "
             "quality or an idea",
}
# Mid scores mix real things with non-things ("chinese characters" 0.45), so only a confident "other" drops a tag.
THING_THRESHOLD = 0.9

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
    from the merge, and `kept` is in output order. `echo_reply` holds the replaced reply when the echo
    fallback ran.
    """

    size_in: tuple[int, int]
    size_out: tuple[int, int]
    reply: str
    proposed: tuple[str, ...]
    candidates: tuple[str, ...]
    origins: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]
    scores: tuple[float, ...]
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


def _word_forms(text: str) -> dict[str, set[str]]:
    # One cut per word would stem "trees" to "tre" and leave "tree" whole, so each word keeps every cut.
    forms: dict[str, set[str]] = {}

    for word in re.findall(r"[a-z0-9]+", text.lower()):
        candidates = {word}
        for suffix, ending in (("s", ""), ("es", ""), ("ies", "y")):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                candidates.add(word[:-len(suffix)] + ending)
        forms[word] = candidates
    return forms


def prompt_forms(prompt: str) -> set[str]:
    """Every form of every prompt word, the set `grounded` matches a tag against."""
    return set().union(*_word_forms(prompt).values())


def grounded(tag: str, prompt_words: set[str]) -> bool:
    """Whether every content word of tag shares a form with a word of the prompt."""
    content = [forms for word, forms in _word_forms(tag).items() if word not in _STOP_WORDS]

    return bool(content) and all(forms & prompt_words for forms in content)


def merge_trace(proposed: tuple[str, ...], prompt_tags: tuple[str, ...]) -> tuple[
        tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """(candidates, origins, dropped): the proposed tags, then the prompt tags, normalized.

    A category noun, a repeat, or a name past MAX_MERGED_TAGS is dropped with its reason. A prompt
    tag the model also proposed turns that candidate's origin to "both".
    """
    from logit_classifier.tags import normalize_item

    index_of: dict[str, int] = {}
    origins: list[str] = []
    dropped: list[tuple[str, str]] = []

    for origin, items in (("model", proposed), ("prompt", prompt_tags)):
        for item in items:
            name = normalize_item(item)
            index = index_of.get(name)
            if not name:
                continue
            if name in CATEGORY_NOUNS:
                dropped.append((name, "category noun"))
            elif index is not None and origin == "prompt" and origins[index] == "model":
                origins[index] = "both"
            elif index is not None:
                dropped.append((name, "repeat"))
            elif len(index_of) == MAX_MERGED_TAGS:
                dropped.append((name, "over the cap"))
            else:
                index_of[name] = len(origins)
                origins.append(origin)
    return tuple(index_of), tuple(origins), tuple(dropped)


def thing_scores(classifier: Any, tags: tuple[str, ...], known: dict[str, float]) -> tuple[float, ...]:
    """p("other") of the thing check per tag, text only, each tag missing from known asked once."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    unknown = tuple(dict.fromkeys(tag for tag in tags if tag not in known))
    questions: dict[str, Any] = {}

    if unknown:
        questions = {f"t{index}": ChoiceQuestion(instructions=THING_QUESTION.replace(TAG_PLACEHOLDER, tag),
                                                 criteria=THING_CRITERIA)
                     for index, tag in enumerate(unknown)}
        response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions))
        known.update((tag, response.answers[qid].probabilities["other"])
                     for tag, qid in zip(unknown, questions, strict=True))
    return tuple(known[tag] for tag in tags)


def is_thing(p_other: float) -> bool:
    return p_other < THING_THRESHOLD


def _passed(candidates: tuple[str, ...], origins: tuple[str, ...], scores: tuple[float, ...]) -> tuple[str, ...]:
    """The candidates whose score reaches their origin's threshold."""
    passed: list[str] = []

    for tag, origin, p in zip(candidates, origins, scores, strict=True):
        threshold = PROMPT_ONLY_THRESHOLD if origin == "prompt" else VERIFY_THRESHOLD
        if p >= threshold:
            passed.append(tag)
    return tuple(passed)


def _full_or_repeating(tags: list[str]) -> bool:
    from logit_classifier.tags import repeated_block

    return len(tags) >= MAX_PROPOSED_TAGS or bool(repeated_block(tags))


def _off_prompt_or_repeating(tags: list[str], prompt_words: set[str]) -> bool:
    from logit_classifier.tags import repeated_block

    trailing = tags[-UNGROUNDED_STREAK:]
    off_prompt = len(trailing) == UNGROUNDED_STREAK and not any(grounded(tag, prompt_words) for tag in trailing)
    return off_prompt or bool(repeated_block(tags))


def propose(clip: Any, image: Any, instruction: str) -> str:
    """The CLIP's greedy list of what image holds, as comma separated text."""
    return engine.generate_text(clip, instruction, image=image, max_tokens=PROPOSE_MAX_TOKENS,
                                should_stop=_full_or_repeating)


def english_prompt(clip: Any, classifier: Any, prompt: str) -> str:
    """prompt when it reads as English, otherwise its English translation, or prompt when that comes back blank."""
    translation = ""

    if engine.nouls(classifier, [LANGUAGE_QUESTION], state=prompt)[0] >= ENGLISH_THRESHOLD:
        return prompt
    # A literal replace, so braces in the prompt are never read as format fields.
    translation = engine.generate_text(clip, TRANSLATE_INSTRUCTION.replace(PROMPT_PLACEHOLDER, prompt),
                                       max_tokens=TRANSLATE_MAX_TOKENS, whole_tags=False).strip()
    return translation or prompt


def prompt_things(clip: Any, classifier: Any, prompt: str, known_things: dict[str, float]) -> tuple[str, ...]:
    """The physical things prompt names that its own words ground and the thing check keeps.

    The things come from the prompt's English translation when the prompt is not English.
    """
    text = ""
    prompt_words: set[str] = set()
    reply = ""
    listed: tuple[str, ...] = ()
    p_other: tuple[float, ...] = ()

    if not prompt.strip():
        return ()
    from logit_classifier.tags import parse_candidates

    text = english_prompt(clip, classifier, prompt)
    prompt_words = prompt_forms(text)
    # A literal replace, so braces in the prompt are never read as format fields.
    reply = engine.generate_text(clip, PROMPT_TAGS_INSTRUCTION.replace(PROMPT_PLACEHOLDER, text),
                                 max_tokens=PROMPT_TAGS_MAX_TOKENS,
                                 should_stop=lambda tags: _off_prompt_or_repeating(tags, prompt_words))
    listed = tuple(tag for tag in parse_candidates(reply) if grounded(tag, prompt_words))
    p_other = thing_scores(classifier, listed, known_things)
    return tuple(tag for tag, p in zip(listed, p_other, strict=True) if is_thing(p))


def _check_one_image(image: Any, node: str = NODE_NAME) -> None:
    engine.check_images(image, node)
    if image.shape[0] != 1:
        raise ValueError(
            f"{node}: each image must be one image of shape [1, H, W, C], got a batch of "
            f"{image.shape[0]}. Split the batch into single images first."
        )


def tag_image(clip: Any, classifier: Any, image: Any, *, prompt_tags: tuple[str, ...] = (),
              instruction: str | None = None, on_stage: Callable[[str], None] | None = None) -> TagTrace:
    """Run every stage on one [1, H, W, C] image and report what each stage produced."""
    wording = PROPOSE_INSTRUCTION if instruction is None else instruction
    announce = on_stage if on_stage is not None else (lambda _stage: None)
    resized = None
    reply = ""
    echo_reply: str | None = None
    proposed: tuple[str, ...] = ()
    dropped: tuple[tuple[str, str], ...] = ()
    candidates: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    statements: list[str] = []
    scores: tuple[float, ...] = ()
    score_of: dict[str, float] = {}
    kept: tuple[str, ...] = ()
    started = 0.0
    propose_ms = 0.0
    verify_ms = 0.0

    _check_one_image(image)
    from logit_classifier.backends.comfy_clip import shared_vision_encode
    from logit_classifier.tags import drop_subsets, parse_candidates

    resized = engine.resize_for_vl(image)
    # One scope over propose and verify, so the vision tower encodes the picture once.
    with shared_vision_encode(clip):
        announce(STAGE_WRITING)
        started = time.perf_counter()
        reply = propose(clip, resized, wording)
        proposed = tuple(parse_candidates(reply, max_candidates=MAX_PROPOSED_TAGS))
        if instruction is None and not merge_trace(proposed, ())[0]:
            echo_reply = reply
            reply = propose(clip, resized, ECHO_FALLBACK_INSTRUCTION)
            proposed = tuple(parse_candidates(reply, max_candidates=MAX_PROPOSED_TAGS))
        propose_ms = (time.perf_counter() - started) * 1000.0
        candidates, origins, dropped = merge_trace(proposed, prompt_tags)
        announce(STAGE_CHECKING)
        started = time.perf_counter()
        statements = [VERIFY_STATEMENT.replace(TAG_PLACEHOLDER, tag) for tag in candidates]
        scores = tuple(engine.nouls(classifier, statements, image=resized))
        verify_ms = (time.perf_counter() - started) * 1000.0

    score_of = dict(zip(candidates, scores, strict=True))
    # sorted is stable, so tied tags keep the order they were proposed in.
    kept = tuple(sorted(drop_subsets(list(_passed(candidates, origins, scores))),
                        key=lambda tag: score_of[tag], reverse=True))
    return TagTrace(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(resized.shape[1]), int(resized.shape[2])),
        reply=reply,
        proposed=proposed,
        candidates=candidates,
        origins=origins,
        dropped=dropped,
        scores=scores,
        kept=kept,
        tags=", ".join(kept),
        propose_ms=propose_ms,
        verify_ms=verify_ms,
        echo_reply=echo_reply,
    )


def parse_tag_list(text: str) -> tuple[str, ...]:
    """The cleaned tags of a comma or newline separated list, in order, each once."""
    parts: list[str] = []

    if not text.strip():
        return ()
    # The node parses the list before build_classifier checks the library.
    try:
        from logit_classifier.tags import clean_item
    except ImportError as error:
        raise engine.library_error(NODE_NAME) from error

    parts = [clean_item(part) for part in re.split(r"[,\n]", text)]
    return tuple(dict.fromkeys(part for part in parts if part))


def check_tag_list(clip: Any, classifier: Any, image: Any, tags: tuple[str, ...]) -> TagTrace:
    """Ask the presence question of every listed tag about one [1, H, W, C] image and keep the tags it shows."""
    resized = None
    questions: list[str] = []
    scores: list[float] = []
    passed: list[tuple[str, float]] = []
    kept: tuple[str, ...] = ()
    started = 0.0
    verify_ms = 0.0

    _check_one_image(image)
    from logit_classifier.backends.comfy_clip import shared_vision_encode
    from logit_classifier.config import JEV_MAX_QUESTIONS

    resized = engine.resize_for_vl(image, max_pixels=engine.CHECK_MAX_PIXELS)
    questions = [engine.presence_question(tag) for tag in tags]
    started = time.perf_counter()
    # One scope over every request, so the vision tower encodes the picture once.
    with shared_vision_encode(clip):
        # The library rejects a request of more than JEV_MAX_QUESTIONS questions.
        for start in range(0, len(questions), JEV_MAX_QUESTIONS):
            scores += engine.nouls(classifier, questions[start:start + JEV_MAX_QUESTIONS], image=resized)
    verify_ms = (time.perf_counter() - started) * 1000.0

    passed = [(tag, p) for tag, p in zip(tags, scores, strict=True) if p >= TAG_LIST_THRESHOLD]
    # sorted is stable, so tied tags keep their list order.
    kept = tuple(tag for tag, _p in sorted(passed, key=lambda pair: pair[1], reverse=True))
    return TagTrace(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(resized.shape[1]), int(resized.shape[2])),
        reply="",
        proposed=(),
        candidates=tuple(tags),
        origins=("list",) * len(tags),
        dropped=(),
        scores=tuple(scores),
        kept=kept,
        tags=", ".join(kept),
        propose_ms=0.0,
        verify_ms=verify_ms,
    )


def tag_images(clip: Any, images: list[Any], prompts: list[str], *, instruction: str | None = None,
               tag_list: tuple[str, ...] = (),
               on_progress: Callable[[int, str], None] | None = None) -> list[TagTrace]:
    """One TagTrace per [1, H, W, C] image, in order, checked against tag_list when it holds tags.

    Otherwise each image is tagged with the prompt at its index.
    """
    classifier = None
    known_things: dict[str, float] = {}
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
    for index, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
        # Read at its first image, so the prompt's generate runs under that image's progress share and status.
        if prompt not in things_by_prompt:
            if on_progress is not None and prompt.strip():
                on_progress(index, STAGE_READING_PROMPT)
            things_by_prompt[prompt] = prompt_things(clip, classifier, prompt, known_things)
        on_stage = None if on_progress is None else (lambda stage, index=index: on_progress(index, stage))
        traces.append(tag_image(clip, classifier, image, prompt_tags=things_by_prompt[prompt],
                                instruction=instruction, on_stage=on_stage))
    return traces


def _presence_probabilities(classifier: Any, image: Any, things: tuple[str, ...]) -> tuple[float, ...]:
    """The presence probability of every thing from one request about image, or () and no request without things."""
    if not things:
        return ()
    # The prompt list holds at most one thing per PROMPT_TAGS_MAX_TOKENS token, within the library's 256 question cap.
    return tuple(engine.nouls(classifier, [engine.presence_question(thing) for thing in things],
                              image=engine.resize_for_vl(image, max_pixels=engine.CHECK_MAX_PIXELS)))


def check_prompt(clip: Any, images: list[Any], prompts: list[str], *,
                 on_progress: Callable[[int, str], None] | None = None) -> list[PromptCheck]:
    """One PromptCheck per [1, H, W, C] image, in order, asking it about the things its paired prompt names."""
    announce = on_progress if on_progress is not None else (lambda _index, _stage: None)
    classifier = None
    known_things: dict[str, float] = {}
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
    for index, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
        if prompt not in things_by_prompt:
            announce(index, STAGE_READING_PROMPT)
            things_by_prompt[prompt] = prompt_things(clip, classifier, prompt, known_things)
        things = things_by_prompt[prompt]
        announce(index, STAGE_CHECKING_THINGS)
        probabilities = _presence_probabilities(classifier, image, things)
        missing = tuple(thing for thing, p in zip(things, probabilities, strict=True) if p < MISSING_THRESHOLD)
        score = (len(things) - len(missing)) / len(things) if things else 1.0
        checks.append(PromptCheck(things=things, probabilities=probabilities, missing=missing, score=score))
    return checks
