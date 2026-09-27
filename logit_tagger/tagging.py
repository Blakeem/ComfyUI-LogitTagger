"""The Logit Tagger pipeline: resize, propose, parse, verify and filter, per image.

No comfy module and no logit_classifier module is imported at module scope, so the unit
tests run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

NODE_NAME = "Logit Tagger"

# Core only resizes a VL image above 12.8 MP, so a 4k input would cost about 12k vision tokens.
VL_MAX_PIXELS = 1024 * 1024

# The fastest of eight wordings on 28 images in tests-AB/ab_propose_prompts.py, with the fewest rejected
# tags. No wording cut the loops, which followed the image.
PROPOSE_INSTRUCTION = (
    "Tag this image. List every distinct visible thing in it: objects, people, animals, "
    "clothing, materials, and the setting. Use short lowercase noun phrases separated by "
    "commas. List each thing once. Do not list moods, styles, or ideas. Output only the tags."
)

# Anchors the proposal vocabulary to the prompt's words. In tests-AB/ab_tagger.py the proposal copied
# three objects the image lacked, and the verify stage scored them 0.0001 to 0.0002.
PROMPT_ANCHOR = (
    "This image was made from the prompt: {prompt}\n"
    "Use the prompt's words for the things the image shows.\n\n"
)

# Core's empty <think> block (thinking=False) made Qwen3-VL-4B-Instruct reply empty, refuse, or open with
# "assistant" or "Wait, I need to..." in the operator's probe. krea2.py:28 and qwen3vl.py:162 default it oppositely.
PROPOSE_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}"
    "<|im_end|>\n<|im_start|>assistant\n"
)

# Bounds a greedy decode that loops. In tests-AB/ab_propose_budget.py both images that reached 128
# were repeating tags, and 192 added no tag on nine images.
PROPOSE_MAX_TOKENS = 128

# In tests-AB/ab_propose_loops.py greedy decode looped on 3 of 17 images, one on a 3 tag cycle. A stop on a
# block of up to 4 tags repeated back to back ended all three and changed no kept tag.
MAX_REPEAT_BLOCK = 4

# Bounds the packed verify pass. In tests-AB/ab_tagger.py a 42 candidate image fit one pass
# beside a 1 MP image and took 0.79 s.
MAX_CANDIDATES = 48

# Adapted from the library's measured crop wording "This crop visibly contains {f}", AUC 0.963
# in its tests-AB/ab_multi_label.py.
VERIFY_STATEMENT = "This image visibly contains {tag}"

# On six runs of tests-AB/ab_tagger.py every tag scored either 0.80 and up or 0.23 and below, and
# the three absent prompt objects scored 0.0002 at most, so 0.5 sits inside that gap.
NOUL_THRESHOLD = 0.5

# A tag longer than this is a sentence or a run-on, not a noun phrase.
MAX_TAG_WORDS = 6
MAX_TAG_CHARS = 60

# The `|$` also removes an unclosed block, the shape a decode cut off mid thought leaves.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_SEPARATORS = re.compile(r"[,;\n]")
# A numbered bullet needs whitespace after it, so a tag like "2.5 liter bottle" survives.
_BULLET = re.compile(r"^(?:[-*•]\s*|\d+[.)](?:\s+|$))")
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"

_LIBRARY_MISSING = (
    f"{NODE_NAME}: this node needs logit-classifier>=0.2.0, which provides the ComfyUI CLIP "
    "backend. Install or upgrade it in the ComfyUI Python environment with: "
    "pip install -U logit-classifier"
)


@dataclass(frozen=True)
class TagDetail:
    """One image's pass through every stage, for the harness to measure."""

    size_in: tuple[int, int]
    size_out: tuple[int, int]
    proposal: str
    candidates: tuple[tuple[str, float], ...]
    tags: str
    propose_ms: float
    verify_ms: float


def _check_images(images: Any) -> None:
    shape = tuple(getattr(images, "shape", ()))

    if len(shape) != 4:
        raise ValueError(
            f"{NODE_NAME}: the image input must be a ComfyUI IMAGE batch of shape [B, H, W, C], "
            f"got shape {shape}"
        )
    if shape[-1] < 3:
        raise ValueError(
            f"{NODE_NAME}: the image input must be an RGB or RGBA IMAGE of shape [B, H, W, 3] "
            f"or [B, H, W, 4], got shape {shape}"
        )


def resize_for_vl(image: Any) -> Any:
    """Downscale one [1, H, W, C] image to at most VL_MAX_PIXELS, keeping the first three channels."""
    import torch

    height, width = image.shape[1], image.shape[2]
    rgb = image[..., :3]
    scale = 0.0
    samples = None

    if height * width <= VL_MAX_PIXELS:
        return rgb
    scale = math.sqrt(VL_MAX_PIXELS / (height * width))
    # Floor, so the product can never land above the cap.
    size = (max(1, math.floor(height * scale)), max(1, math.floor(width * scale)))
    samples = torch.nn.functional.interpolate(rgb.movedim(-1, 1), size=size, mode="area")
    return samples.movedim(1, -1)


def build_instruction(prompt: str, instruction: str) -> str:
    anchor = prompt.strip()

    if not anchor:
        return instruction
    return PROMPT_ANCHOR.format(prompt=anchor) + instruction


@contextmanager
def cuda_graphs_disabled() -> Iterator[None]:
    """Turn off core's CUDA graph decode for the duration, then restore the previous flag.

    Core's Qwen3 CUDA graph decode (PR #15623) raises a device side assert ("scatter gather
    kernel index out of bounds") on the second multi step image generate in one process, still
    at core b5cc8830. Reported as Comfy-Org/ComfyUI issue #16441. The flag avoids it.
    """
    try:
        import comfy.model_management as model_management
    except ImportError:
        yield
        return

    previous = model_management.args.disable_cuda_graphs
    model_management.args.disable_cuda_graphs = True
    try:
        yield
    finally:
        model_management.args.disable_cuda_graphs = previous


def repeated_block(tags: list[str]) -> int:
    """The size of the block of tags the list just wrote twice back to back, or 0 for none."""
    for size in range(1, MAX_REPEAT_BLOCK + 1):
        if len(tags) >= 2 * size and tags[-size:] == tags[-2 * size:-size]:
            return size
    return 0


def complete_tags(text: str) -> list[str]:
    """The tags followed by a separator, since the text after the last one may still be mid tag."""
    parts = _SEPARATORS.split(text)[:-1]

    return [tag for tag in (_clean_item(part) for part in parts) if tag]


def _sampler(clip: Any) -> tuple[Any, int] | None:
    """Core's generating transformer and its first stop token, or None for a CLIP without them."""
    model = getattr(clip, "cond_stage_model", None)
    encoder_name = getattr(model, "clip", None)
    encoder = getattr(model, encoder_name, None) if isinstance(encoder_name, str) else None
    transformer = getattr(encoder, "transformer", None)
    config = getattr(getattr(transformer, "model", None), "config", None)
    stop_tokens = getattr(config, "stop_tokens", None)

    if not callable(getattr(transformer, "sample_token", None)) or not stop_tokens:
        return None
    return transformer, stop_tokens[0]


@contextmanager
def stop_on_repeat(clip: Any) -> Iterator[None]:
    """End core's decode on a stop token once the reply repeats a block of tags back to back.

    Core's generate loop takes no stop condition, so the token sample_token returns is replaced.
    Core copies that token into the sequence and breaks on it, at comfy/text_encoders/llama.py:1170-1187.
    """
    sampler = _sampler(clip)
    history: list[int] = []

    if sampler is None:
        yield
        return
    transformer, stop_id = sampler
    original = transformer.sample_token
    shadowed = "sample_token" in vars(transformer)
    previous = vars(transformer).get("sample_token")

    def watch(*args: Any, **kwargs: Any) -> Any:
        token = original(*args, **kwargs)
        history.append(int(token.reshape(-1)[0]))
        if repeated_block(complete_tags(clip.decode(history))):
            return token.new_full(token.shape, stop_id)
        return token

    transformer.sample_token = watch
    try:
        yield
    finally:
        if shadowed:
            transformer.sample_token = previous
        else:
            del transformer.sample_token


def _drop_cut_tail(text: str) -> str:
    separators = list(_SEPARATORS.finditer(text))

    if not separators:
        return ""
    return text[:separators[-1].start()]


def propose(clip: Any, image: Any, prompt: str, instruction: str) -> str:
    """Ask the CLIP to write the image's candidate tags as comma separated text."""
    rendered = PROPOSE_TEMPLATE.format(instruction=build_instruction(prompt, instruction))
    tokens = None
    ids = None
    text = ""

    # A text starting with <|im_start|> skips core's template, at comfy/text_encoders/qwen3vl.py:167.
    tokens = clip.tokenize(rendered, images=[image])
    with cuda_graphs_disabled(), stop_on_repeat(clip):
        ids = clip.generate(tokens, do_sample=False, max_length=PROPOSE_MAX_TOKENS)
    text = clip.decode(ids)
    # Core stops early only on a stop token, so a decode that fills the budget ends mid tag.
    if len(ids) >= PROPOSE_MAX_TOKENS:
        return _drop_cut_tail(text)
    return text


def _clean_item(item: str) -> str:
    text = _BULLET.sub("", item.strip())

    text = text.strip(" \t" + _QUOTES).rstrip(". \t" + _QUOTES)
    return " ".join(text.lower().split())


def parse_candidates(text: str) -> list[str]:
    """Turn proposal text into unique candidate tags, in the order the CLIP wrote them."""
    body = _THINK_BLOCK.sub("", text).strip()
    lines = body.split("\n")
    candidates: list[str] = []
    seen: set[str] = set()

    # The stray role line core's empty think block provoked in the operator's probe.
    if lines and lines[0].strip() == "assistant":
        body = "\n".join(lines[1:])
    for item in _SEPARATORS.split(body):
        tag = _clean_item(item)
        if not tag or len(tag) > MAX_TAG_CHARS or len(tag.split()) > MAX_TAG_WORDS or tag in seen:
            continue
        seen.add(tag)
        candidates.append(tag)
        if len(candidates) == MAX_CANDIDATES:
            break
    return candidates


def build_classifier(clip: Any) -> Any:
    """One Classifier over the workflow's CLIP, with no prior drift and no calibration file."""
    try:
        from logit_classifier import Classifier, Config
        from logit_classifier.backends.comfy_clip import ComfyClipBackend
    except ImportError as error:
        raise RuntimeError(_LIBRARY_MISSING) from error

    backend = ComfyClipBackend(clip)
    if not backend.sees_images:
        raise ValueError(
            f"{NODE_NAME}: this CLIP was loaded without its vision tower, so it cannot read the "
            "image. Load a Qwen3-VL text encoder with the Load CLIP type set to krea2. The flux "
            "and flux2 types load it without the vision tower."
        )
    return Classifier(Config(use_prior_debias=False, calibration_path=None), backend)


def verify(classifier: Any, image: Any, candidates: list[str]) -> list[float]:
    """Each candidate's noul probability, from one request that asks about all of them."""
    questions: dict[str, Any] = {}

    if not candidates:
        return []
    from logit_classifier import NoulQuestion, SystemOneRequest

    questions = {
        f"tag{index}": NoulQuestion(instructions=VERIFY_STATEMENT.format(tag=tag))
        for index, tag in enumerate(candidates)
    }
    # The empty state is what the library renders as its image only state.
    response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions),
                                                 image=image)
    return [response.answers[qid].noul for qid in questions]


def filter_tags(candidates: list[str], probabilities: list[float], threshold: float) -> str:
    kept = [(tag, p) for tag, p in zip(candidates, probabilities, strict=True) if p >= threshold]

    # sorted is stable, so tied tags keep the order the CLIP proposed them in.
    kept = sorted(kept, key=lambda pair: pair[1], reverse=True)
    return ", ".join(tag for tag, _p in kept)


def tag_image(clip: Any, image: Any, prompt: str, *, threshold: float | None = None,
              instruction: str | None = None, classifier: Any = None) -> TagDetail:
    """Run every stage on one [1, H, W, C] image and report what each stage produced."""
    limit = NOUL_THRESHOLD if threshold is None else threshold
    wording = PROPOSE_INSTRUCTION if instruction is None else instruction
    resized = None
    proposal = ""
    candidates: list[str] = []
    probabilities: list[float] = []
    started = 0.0
    propose_ms = 0.0
    verify_ms = 0.0

    _check_images(image)
    if image.shape[0] != 1:
        raise ValueError(f"{NODE_NAME}: tag_image takes one image, got a batch of {image.shape[0]}")
    # Built before the propose stage, so a wrong CLIP fails before any generate runs.
    if classifier is None:
        classifier = build_classifier(clip)

    resized = resize_for_vl(image)
    started = time.perf_counter()
    proposal = propose(clip, resized, prompt, wording)
    propose_ms = (time.perf_counter() - started) * 1000.0
    candidates = parse_candidates(proposal)
    started = time.perf_counter()
    probabilities = verify(classifier, resized, candidates)
    verify_ms = (time.perf_counter() - started) * 1000.0

    return TagDetail(
        size_in=(int(image.shape[1]), int(image.shape[2])),
        size_out=(int(resized.shape[1]), int(resized.shape[2])),
        proposal=proposal,
        candidates=tuple(zip(candidates, probabilities, strict=True)),
        tags=filter_tags(candidates, probabilities, limit),
        propose_ms=propose_ms,
        verify_ms=verify_ms,
    )


def tag_images(clip: Any, images: Any, prompt: str,
               on_image_done: Callable[[int], None] | None = None) -> list[str]:
    """One tag string per image in the batch, in batch order."""
    classifier = None
    tags: list[str] = []

    _check_images(images)
    classifier = build_classifier(clip)
    for index in range(images.shape[0]):
        detail = tag_image(clip, images[index:index + 1], prompt, classifier=classifier)
        tags.append(detail.tags)
        if on_image_done is not None:
            on_image_done(index)
    return tags
