"""Logit Style Extractor's run: the medium and the style within it, from each item's prompt or its entire image.

No comfy, torch or logit_classifier module is imported at module scope, so the unit tests
run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import engine, tagging

STYLE_EXTRACTOR_NAME = "Logit Style Extractor"

STAGE_READING_PROMPT = tagging.STAGE_READING_PROMPT
STAGE_READING_STYLE = "reading style"

SOURCE_PROMPT = "prompt"
SOURCE_IMAGE = "image"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class StyleTrace:
    """One item's style and where it came from.

    `source` is "prompt", "image" or "none". `prompt_trace` is the toolkit's PromptStyleTrace of the item's prompt,
    or None for a blank prompt.
    """

    source: str
    style: str
    prompt_trace: Any


def extract_styles(clip: Any, images: list[Any], prompts: list[str], *,
                   on_progress: Callable[[int, str], None] | None = None) -> list[StyleTrace]:
    """One StyleTrace per prompt, in order, each from the style words its prompt names, or else from the
    [1, H, W, C] image at its index.

    An empty images list means no item has an image, so an item whose prompt names no style gets "".
    """
    announce = on_progress if on_progress is not None else (lambda _index, _stage: None)
    classifier = None
    settings: Any = None
    prompt_settings: Any = None
    style_cache: dict[str, float] = {}
    traces_by_prompt: dict[str, Any] = {}
    prompt_trace: Any = None
    style = ""
    source = SOURCE_NONE
    traces: list[StyleTrace] = []

    if images and len(images) != len(prompts):
        raise ValueError(
            f"{STYLE_EXTRACTOR_NAME}: got {len(images)} images and {len(prompts)} prompts. Give one prompt per image, "
            "an empty one for an image without a prompt."
        )
    for image in images:
        engine.check_images(image, STYLE_EXTRACTOR_NAME)
    # Built before any generate, so a wrong CLIP fails before the model runs.
    classifier = engine.build_classifier(clip, STYLE_EXTRACTOR_NAME, needs_images=bool(images))
    prompt_settings = tagging.tag_settings(node=STYLE_EXTRACTOR_NAME)
    from logit_classifier.toolkit.comfyui import StyleSettings, prompt_style, skip_resident_loads, style_line

    settings = StyleSettings()
    with skip_resident_loads(clip):
        for index, prompt in enumerate(prompts):
            # Read at its first item, so the prompt's generate runs under that item's progress share and status.
            if prompt.strip() and prompt not in traces_by_prompt:
                announce(index, STAGE_READING_PROMPT)
                traces_by_prompt[prompt] = prompt_style(clip, classifier, prompt, settings=settings,
                                                        tag_settings=prompt_settings, style_cache=style_cache)
            prompt_trace = traces_by_prompt.get(prompt)
            style = ", ".join(prompt_trace.style) if prompt_trace is not None else ""
            if style:
                source = SOURCE_PROMPT
            elif images:
                announce(index, STAGE_READING_STYLE)
                style = style_line(clip, images[index], settings=settings, node=STYLE_EXTRACTOR_NAME)
                source = SOURCE_IMAGE
            else:
                source = SOURCE_NONE
            traces.append(StyleTrace(source=source, style=style, prompt_trace=prompt_trace))
    return traces
