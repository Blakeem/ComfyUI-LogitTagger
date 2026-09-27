"""The pack's nodes, in ComfyUI's V1 node contract.

No comfy module is imported at module scope, so the unit tests run without ComfyUI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import engine, lists, progress, tagging

IMAGE_CHECK_NAME = "Logit Image Check"
TEXT_CHECK_NAME = "Logit Text Check"
CHECK_STAGE = "checking"
YES_THRESHOLD = 0.5
IMAGE_QUESTION = "Is there a person in this image?"
# On 80 labeled prompts this said no to every negated mention ("no cats") and decoy word ("dogwood"), where "Does this text mention" said yes.
TEXT_QUESTION = "Does this prompt ask for a person in the image?"


def format_display(tags: list[str]) -> str:
    """One line per image, with the batch index in front once the batch holds more than one."""
    if len(tags) == 1:
        return tags[0]
    return "\n".join(f"{index}: {text}" for index, text in enumerate(tags))


def _propose_instruction(values: list[str] | None) -> str | None:
    """The connected instruction, or None for the pipeline's own when unconnected or blank."""
    value = ""

    if not values:
        return None
    value = lists.single_value(values, tagging.NODE_NAME, "instruction")
    return value if value.strip() else None


def _image_progress(run: progress.RunProgress) -> Callable[[int, str], None]:
    """tag_images' on_progress, feeding run with each image's stages and closing each finished image."""

    def on_progress(index: int, stage: str) -> None:
        # tag_images starts an image only once the one before it returned, so its first stage closes that one.
        if index > 0 and stage == tagging.STAGE_WRITING:
            run.item_done(index - 1)
        run.item(index, stage)

    return on_progress


def _image_probability(classifier: Any, question: str, image: Any) -> float:
    return engine.nouls(classifier, [question], image=engine.resize_for_vl(image))[0]


def _text_probability(classifier: Any, question: str, text: str) -> float:
    return engine.nouls(classifier, [question], state=text)[0]


def _run_check(clip: Any, items: list[Any], question: list[str], unique_id: list[str] | None, *, node: str,
               noun: str, default_question: str, needs_images: bool,
               probability_of: Callable[[Any, str, Any], float]) -> dict[str, Any]:
    """One question for the run, one probability per item, then the answer, the yes and no split and the display."""
    from comfy_execution.graph_utils import ExecutionBlocker

    node_id = unique_id[0] if unique_id else None
    wording = ""
    classifier = None
    probabilities: list[float] = []
    verdicts: list[str] = []
    yes_items: list[Any] = []
    no_items: list[Any] = []
    display = ""

    if not items:
        raise ValueError(f"{node}: the {noun} input holds no {noun}s. Connect at least one {noun}.")
    wording = lists.single_value(question, node, "question")
    if not wording.strip():
        wording = default_question
    classifier = engine.build_classifier(clip, node, needs_images=needs_images)

    with progress.RunProgress(total_items=len(items), unique_id=node_id, noun=noun) as run:
        for index, item in enumerate(items):
            run.item(index, CHECK_STAGE)
            probabilities.append(float(probability_of(classifier, wording, item)))
            run.item_done(index)

    verdicts = [f"{'yes' if p >= YES_THRESHOLD else 'no'} {p:.3f}" for p in probabilities]
    yes_items = [item for item, p in zip(items, probabilities, strict=True) if p >= YES_THRESHOLD]
    no_items = [item for item, p in zip(items, probabilities, strict=True) if p < YES_THRESHOLD]
    display = f"{wording}\n{format_display(verdicts)}"
    # A silent blocker skips the nodes wired to an empty side instead of handing them no items.
    return {
        "ui": {"text": (display,)},
        "result": (not no_items, probabilities, yes_items or [ExecutionBlocker(None)],
                   no_items or [ExecutionBlocker(None)]),
    }


class LogitTagger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It proposes the tags and then checks each one against the image."}),
                "image": ("IMAGE", {"tooltip": "The images to tag, as a batch or a list of batches. Each image is tagged on its own."}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The things this prompt names join the candidate tags, and each is kept only where the image shows it. Connect one prompt for every image, or a list with one prompt per image."}),
                "instruction": ("STRING", {"forceInput": True, "tooltip": "This text replaces the built-in question that asks the encoder to list what each image holds. A blank text keeps the built-in question."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("tags", "images")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    # So the node runs and shows its tags with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "tag"
    CATEGORY = "Logit Tagger"
    DESCRIPTION = (
        "The node tags each image with the Qwen3-VL text encoder the workflow already holds. The encoder "
        "proposes candidate tags, and a logit classifier keeps only the tags the image shows. It returns "
        "each image beside its tag string so that Save Image-Text (to Folder) can save them as pairs."
    )

    def tag(self, clip: list[Any], image: list[Any], prompt: list[str] | None = None,
            instruction: list[str] | None = None, unique_id: list[str] | None = None) -> dict[str, Any]:
        # INPUT_IS_LIST delivers every input as a list, the CLIP and the hidden node id included.
        model = clip[0]
        node_id = unique_id[0] if unique_id else None
        images: list[Any] = []
        prompts: list[str] = []
        wording: str | None = None
        tags: list[str] = []

        images = lists.split_images(image, tagging.NODE_NAME)
        if not images:
            raise ValueError(f"{tagging.NODE_NAME}: the image input holds no images. Connect at least one image.")
        prompts = lists.pair_texts(prompt, len(images), tagging.NODE_NAME, "prompt")
        wording = _propose_instruction(instruction)

        with progress.RunProgress(total_items=len(images), unique_id=node_id, noun="image") as run:
            tags = [trace.tags for trace in tagging.tag_images(model, images, prompts, instruction=wording,
                                                              on_progress=_image_progress(run))]
            run.item_done(len(images) - 1)

        # The frontend joins a ui list's entries with blank lines, so the display travels as one string.
        return {"ui": {"text": (format_display(tags),)}, "result": (tags, images)}


class LogitImageCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It reads each image and answers the question."}),
                "image": ("IMAGE", {"tooltip": "The images to check, as a batch or a list of batches. Each image gets its own answer."}),
                "question": ("STRING", {"default": IMAGE_QUESTION, "multiline": False, "tooltip": "The yes or no question asked about each image. A blank question asks the default one."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("answer", "probability", "yes_images", "no_images")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True, True, True)
    # So the node runs and shows its answers with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "check"
    CATEGORY = "Logit Tagger"
    DESCRIPTION = (
        "The node answers a yes or no question about each image with the Qwen3-VL text encoder the workflow "
        "already holds. It reads the probability of yes from the logits instead of generating text. It sends "
        "each image to yes_images or no_images, and an empty side skips the nodes wired to it."
    )

    def check(self, clip: list[Any], image: list[Any], question: list[str],
              unique_id: list[str] | None = None) -> dict[str, Any]:
        images = lists.split_images(image, IMAGE_CHECK_NAME)

        return _run_check(clip[0], images, question, unique_id, node=IMAGE_CHECK_NAME, noun="image",
                          default_question=IMAGE_QUESTION, needs_images=True, probability_of=_image_probability)


class LogitTextCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It reads each text and answers the question."}),
                "text": ("STRING", {"forceInput": True, "tooltip": "The texts to check, such as prompts. Each text gets its own answer."}),
                "question": ("STRING", {"default": TEXT_QUESTION, "multiline": False, "tooltip": "The yes or no question asked about each text. A blank question asks the default one."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("answer", "probability", "yes_texts", "no_texts")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True, True, True)
    # So the node runs and shows its answers with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "check"
    CATEGORY = "Logit Tagger"
    DESCRIPTION = (
        "The node answers a yes or no question about each text with the Qwen3-VL text encoder the workflow "
        "already holds. It reads the probability of yes from the logits instead of generating text. It sends "
        "each text to yes_texts or no_texts, and an empty side skips the nodes wired to it."
    )

    def check(self, clip: list[Any], text: list[str], question: list[str],
              unique_id: list[str] | None = None) -> dict[str, Any]:
        return _run_check(clip[0], text, question, unique_id, node=TEXT_CHECK_NAME, noun="text",
                          default_question=TEXT_QUESTION, needs_images=False, probability_of=_text_probability)
