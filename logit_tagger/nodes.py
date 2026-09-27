"""The pack's nodes, in ComfyUI's V1 node contract.

No comfy module is imported at module scope, so the unit tests run without ComfyUI.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, ClassVar

from . import engine, lists, progress, tagging

IMAGE_CHECK_NAME = "Logit Image Check"
TEXT_CHECK_NAME = "Logit Text Check"
CHECK_STAGE = "checking"
YES_THRESHOLD = 0.5
IMAGE_QUESTION = "Is there a person in this image?"
# On 80 labeled prompts this said no to every negated mention ("no cats") and decoy word ("dogwood"),
# where "Does this text mention" said yes.
TEXT_QUESTION = "Does this prompt ask for a person in the image?"
NO_THINGS_LINE = "no things in the prompt"
CLASSIFY_NAME = "Logit Image Classify"
CLASSIFY_STAGE = "classifying"
CLASSIFY_QUESTION = "What is the art style of this image?"
# The 8 styles tests-AB/ab_release_gates.py measured on 72 Krea 2 renders, probe render-style.
CLASSIFY_OPTIONS = ("photograph, anime illustration, oil painting, 3D render, watercolor, pencil sketch, pixel art, "
                    "comic book")
MIN_OPTIONS = 2
# One library branch holds 52 labels, and its abstain label takes one of them.
MAX_OPTIONS = 51


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


def _tag_list(values: list[str] | None, prompts: list[str], instruction: str | None) -> tuple[str, ...]:
    """The connected tags to check, or () for tags the pipeline writes when unconnected or blank."""
    tag_list: tuple[str, ...] = ()

    if not values:
        return ()
    tag_list = tagging.parse_tag_list(lists.single_value(values, tagging.NODE_NAME, "tags"))
    if tag_list and any(prompt.strip() for prompt in prompts):
        raise ValueError(
            f"{tagging.NODE_NAME}: the prompt only guides the tags the node writes itself. Connect the prompt "
            "or the tags, not both."
        )
    if tag_list and instruction is not None:
        raise ValueError(
            f"{tagging.NODE_NAME}: the instruction only guides the tags the node writes itself. Connect the "
            "instruction or the tags, not both."
        )
    return tag_list


def _image_probability(classifier: Any, question: str, image: Any) -> float:
    from logit_classifier.toolkit.comfyui import fit_picture

    picture = fit_picture(image, engine.CHECK_MAX_PIXELS)

    return classifier.nouls([question], image=picture)[0]


def _text_probability(classifier: Any, question: str, text: str) -> float:
    return classifier.nouls([question], state=text)[0]


def _split_sides(items: list[Any], passed: list[bool]) -> tuple[list[Any], list[Any]]:
    """The items that passed and the items that failed, each in input order."""
    from comfy_execution.graph_utils import ExecutionBlocker

    yes_side = [item for item, verdict in zip(items, passed, strict=True) if verdict]
    no_side = [item for item, verdict in zip(items, passed, strict=True) if not verdict]

    # A silent blocker skips the nodes wired to an empty side instead of handing them no items.
    return yes_side or [ExecutionBlocker(None)], no_side or [ExecutionBlocker(None)]


def _run_check(clip: Any, items: list[Any], question: list[str], unique_id: list[str] | None, *, node: str,
               noun: str, default_question: str, needs_images: bool,
               probability_of: Callable[[Any, str, Any], float]) -> dict[str, Any]:
    """One question for the run, one probability per item, then the answer, the yes and no split and the display."""
    node_id = unique_id[0] if unique_id else None
    wording = ""
    classifier = None
    probabilities: list[float] = []
    verdicts: list[str] = []
    passed: list[bool] = []
    yes_side: list[Any] = []
    no_side: list[Any] = []
    display = ""

    if not items:
        raise ValueError(f"{node}: the {noun} input holds no {noun}s. Connect at least one {noun}.")
    wording = lists.single_value(question, node, "question")
    if not wording.strip():
        wording = default_question
    classifier = engine.build_classifier(clip, node, needs_images=needs_images)

    with progress.RunProgress(total_items=len(items), unique_id=node_id, noun=noun, node=node) as run:
        for index, item in enumerate(items):
            run.item(index, CHECK_STAGE)
            probabilities.append(float(probability_of(classifier, wording, item)))

    verdicts = [f"{'yes' if p >= YES_THRESHOLD else 'no'} {p:.3f}" for p in probabilities]
    passed = [p >= YES_THRESHOLD for p in probabilities]
    yes_side, no_side = _split_sides(items, passed)
    display = f"{wording}\n{format_display(verdicts)}"
    return {"ui": {"text": (display,)}, "result": (all(passed), probabilities, yes_side, no_side)}


def _prompt_check_display(checks: list[tagging.PromptCheck], missing_texts: list[str]) -> str:
    """Each image's score and missing things, under one line of things when every image was checked for the same
    things, otherwise each after its own things."""
    shared = len({check.things for check in checks}) == 1
    things_texts: list[str] = []
    verdicts: list[str] = []

    things_texts = [", ".join(check.things) or NO_THINGS_LINE for check in checks]
    verdicts = [f"{check.score:.2f} missing: {missing}" if missing else f"{check.score:.2f} all present"
                for check, missing in zip(checks, missing_texts, strict=True)]
    if shared:
        return f"{things_texts[0]}\n{format_display(verdicts)}"
    # One line of every prompt's things would name things an image was never checked for.
    return format_display([f"{things} | {verdict}" for things, verdict in zip(things_texts, verdicts, strict=True)])


def _widget_text(values: list[str], node: str, input_name: str, default: str) -> str:
    """The widget's one value, or default when that value is blank."""
    value = lists.single_value(values, node, input_name)

    return value if value.strip() else default


def _classify_options(text: str) -> list[str]:
    """The options of a comma or newline separated list, in order and spelled as typed."""
    options = [part.strip() for part in re.split(r"[,\n]", text) if part.strip()]
    seen: set[str] = set()

    for option in options:
        if option.casefold() in seen:
            raise ValueError(
                f"{CLASSIFY_NAME}: the options input names {option!r} more than once. Name each option once, so the "
                "index output matches its position."
            )
        seen.add(option.casefold())
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ValueError(
            f"{CLASSIFY_NAME}: the node takes {MIN_OPTIONS} to {MAX_OPTIONS} different options, and the options "
            f"input names {len(options)}. Separate the options with commas."
        )
    return options


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
                "tags": ("STRING", {"forceInput": True, "tooltip": "The node checks only these tags against each image instead of writing its own. Separate the tags with commas or new lines."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("tags", "images")
    OUTPUT_TOOLTIPS = (
        "One comma separated tag string per image, in image order.",
        "Each input image on its own, in the same order as the tags, so that Save Image-Text (to Folder) can save them as pairs.",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    # So the node runs and shows its tags with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "tag"
    CATEGORY = "Logit Tagger"
    SEARCH_ALIASES: ClassVar[list[str]] = ["caption", "auto tag", "image to tags", "dataset tags", "interrogate"]
    DESCRIPTION = (
        "The node tags each image with the Qwen3-VL text encoder the workflow already holds. The encoder "
        "proposes candidate tags, and a logit classifier keeps only the tags the image shows. It returns "
        "each image beside its tag string so that Save Image-Text (to Folder) can save them as pairs."
    )

    def tag(self, clip: list[Any], image: list[Any], prompt: list[str] | None = None,
            instruction: list[str] | None = None, tags: list[str] | None = None,
            unique_id: list[str] | None = None) -> dict[str, Any]:
        # INPUT_IS_LIST delivers every input as a list, the CLIP and the hidden node id included.
        model = clip[0]
        node_id = unique_id[0] if unique_id else None
        images: list[Any] = []
        prompts: list[str] = []
        wording: str | None = None
        tag_list: tuple[str, ...] = ()
        tag_texts: list[str] = []

        images = lists.split_images(image, tagging.NODE_NAME)
        if not images:
            raise ValueError(f"{tagging.NODE_NAME}: the image input holds no images. Connect at least one image.")
        prompts = lists.pair_texts(prompt, len(images), tagging.NODE_NAME, "prompt")
        wording = _propose_instruction(instruction)
        tag_list = _tag_list(tags, prompts, wording)

        with progress.RunProgress(total_items=len(images), unique_id=node_id, noun="image",
                                  node=tagging.NODE_NAME) as run:
            tag_texts = [trace.tags for trace in tagging.tag_images(model, images, prompts, instruction=wording,
                                                                   tag_list=tag_list, on_progress=run.item)]

        # The frontend joins a ui list's entries with blank lines, so the display travels as one string.
        return {"ui": {"text": (format_display(tag_texts),)}, "result": (tag_texts, images)}


class LogitImageCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It reads each image and answers the question."}),
                "image": ("IMAGE", {"tooltip": "The images to check, as a batch or a list of batches. Each image gets its own answer."}),
                "question": ("STRING", {"default": IMAGE_QUESTION, "multiline": False, "tooltip": "The yes or no question asked about each image. Since a statement such as \"This image contains an orange\" was answered about the color, word it as a question like the default. A blank question asks the default one."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("answer", "probability", "yes_images", "no_images")
    OUTPUT_TOOLTIPS = (
        "True only when every image answers yes. It is one value for the entire run.",
        "The probability of yes for each image, in image order.",
        "The images that answer yes. When no image answers yes, the nodes wired here are skipped.",
        "The images that answer no. When no image answers no, the nodes wired here are skipped.",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True, True, True)
    # So the node runs and shows its answers with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "check"
    CATEGORY = "Logit Tagger"
    SEARCH_ALIASES: ClassVar[list[str]] = ["image gate", "batch filter", "filter images", "yes no", "vqa"]
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
                "question": ("STRING", {"default": TEXT_QUESTION, "multiline": False, "tooltip": "The yes or no question asked about each text. Word it like the default so that a prompt that says \"no cats\" is answered no. A blank question asks the default one."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("answer", "probability", "yes_texts", "no_texts")
    OUTPUT_TOOLTIPS = (
        "True only when every text answers yes. It is one value for the entire run.",
        "The probability of yes for each text, in text order.",
        "The texts that answer yes. When no text answers yes, the nodes wired here are skipped.",
        "The texts that answer no. When no text answers no, the nodes wired here are skipped.",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True, True, True)
    # So the node runs and shows its answers with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "check"
    CATEGORY = "Logit Tagger"
    SEARCH_ALIASES: ClassVar[list[str]] = ["text gate", "prompt gate", "yes no text"]
    DESCRIPTION = (
        "The node answers a yes or no question about each text with the Qwen3-VL text encoder the workflow "
        "already holds. It reads the probability of yes from the logits instead of generating text. It sends "
        "each text to yes_texts or no_texts, and an empty side skips the nodes wired to it."
    )

    def check(self, clip: list[Any], text: list[str], question: list[str],
              unique_id: list[str] | None = None) -> dict[str, Any]:
        return _run_check(clip[0], text, question, unique_id, node=TEXT_CHECK_NAME, noun="text",
                          default_question=TEXT_QUESTION, needs_images=False, probability_of=_text_probability)


class LogitPromptCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It lists the things the prompt names and looks for each one in the image."}),
                "image": ("IMAGE", {"tooltip": "The images to check, as a batch or a list of batches. Each image is checked on its own."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The prompt the images were made from. Connect one prompt for every image, or a list with one prompt per image."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("all_present", "score", "missing", "images", "complete_images", "incomplete_images")
    OUTPUT_TOOLTIPS = (
        "True only when every image shows every thing its prompt names. It is one value for the entire run.",
        "The share of its prompt's things each image shows, from 0 to 1. A prompt that names no things scores 1.",
        "The things each image lacks, as one comma separated string per image. An image that shows every thing gets an empty string.",
        "Each input image on its own, in the same order as score and missing, so that Save Image-Text (to Folder) can save them as pairs.",
        "The images that show every thing their prompt names. When no image does, the nodes wired here are skipped.",
        "The images that lack a thing their prompt names. When every image shows every thing, the nodes wired here are skipped.",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True, True, True, True, True)
    # So the node runs and shows its scores with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "check"
    CATEGORY = "Logit Tagger"
    SEARCH_ALIASES: ClassVar[list[str]] = ["prompt adherence", "prompt following", "missing objects"]
    DESCRIPTION = (
        "The node checks each image for every physical thing its prompt names, with the Qwen3-VL text encoder "
        "the workflow already holds. The encoder lists the prompt's things, and a logit classifier reads whether "
        "the image shows each one. It returns the share of those things each image shows and the things it lacks. "
        "It sends each image to complete_images or incomplete_images."
    )

    def check(self, clip: list[Any], image: list[Any], prompt: list[str],
              unique_id: list[str] | None = None) -> dict[str, Any]:
        node_id = unique_id[0] if unique_id else None
        images: list[Any] = []
        prompts: list[str] = []
        checks: list[tagging.PromptCheck] = []
        missing_texts: list[str] = []
        passed: list[bool] = []
        complete_images: list[Any] = []
        incomplete_images: list[Any] = []

        images = lists.split_images(image, tagging.PROMPT_CHECK_NAME)
        if not images:
            raise ValueError(
                f"{tagging.PROMPT_CHECK_NAME}: the image input holds no images. Connect at least one image."
            )
        prompts = lists.pair_texts(prompt, len(images), tagging.PROMPT_CHECK_NAME, "prompt")

        with progress.RunProgress(total_items=len(images), unique_id=node_id, noun="image",
                                  node=tagging.PROMPT_CHECK_NAME) as run:
            checks = tagging.check_prompt(clip[0], images, prompts, on_progress=run.item)

        missing_texts = [", ".join(check.missing) for check in checks]
        passed = [not check.missing for check in checks]
        complete_images, incomplete_images = _split_sides(images, passed)
        return {
            "ui": {"text": (_prompt_check_display(checks, missing_texts),)},
            "result": (all(passed), [check.score for check in checks], missing_texts, images, complete_images,
                       incomplete_images),
        }


class LogitImageClassify:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It reads each image and picks one of the options."}),
                "image": ("IMAGE", {"tooltip": "The images to classify, as a batch or a list of batches. Each image gets its own choice."}),
                "question": ("STRING", {"default": CLASSIFY_QUESTION, "multiline": False, "tooltip": "The question the options answer, asked about each image. A blank question asks the default one."}),
                "options": ("STRING", {"default": CLASSIFY_OPTIONS, "multiline": False, "tooltip": f"The {MIN_OPTIONS} to {MAX_OPTIONS} answers each image picks one from, separated by commas. A blank widget uses the default art styles."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("choice", "index", "confidence", "images")
    OUTPUT_TOOLTIPS = (
        "The option each image picks, spelled as typed in options.",
        "The position of each image's choice in the options, counting from 0.",
        "The certainty of each choice, from 0 when every option is equally likely to 1 when one option takes all the probability.",
        "Each input image on its own, in the same order as the choices, so that Save Image-Text (to Folder) can save them as pairs.",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True)
    # So the node runs and shows its choices with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "classify"
    CATEGORY = "Logit Tagger"
    SEARCH_ALIASES: ClassVar[list[str]] = ["classifier", "choice", "zero-shot", "art style", "image category"]
    DESCRIPTION = (
        "The node picks one of the options for each image with the Qwen3-VL text encoder the workflow already "
        "holds. It reads the choice from the logits of one question instead of generating text. It returns each "
        "image's choice, the choice's position among the options and the confidence of the choice. It returns each "
        "image beside its choice so that Save Image-Text (to Folder) can save them as pairs."
    )

    def classify(self, clip: list[Any], image: list[Any], question: list[str], options: list[str],
                 unique_id: list[str] | None = None) -> dict[str, Any]:
        node_id = unique_id[0] if unique_id else None
        images: list[Any] = []
        wording = ""
        option_list: list[str] = []
        classifier = None
        answers: list[tuple[str, float]] = []
        choices: list[str] = []
        indexes: list[int] = []
        confidences: list[float] = []
        verdicts: list[str] = []

        images = lists.split_images(image, CLASSIFY_NAME)
        if not images:
            raise ValueError(f"{CLASSIFY_NAME}: the image input holds no images. Connect at least one image.")
        wording = _widget_text(question, CLASSIFY_NAME, "question", CLASSIFY_QUESTION)
        option_list = _classify_options(_widget_text(options, CLASSIFY_NAME, "options", CLASSIFY_OPTIONS))
        classifier = engine.build_classifier(clip[0], CLASSIFY_NAME, needs_images=True)
        from logit_classifier.toolkit.comfyui import fit_picture

        with progress.RunProgress(total_items=len(images), unique_id=node_id, noun="image", node=CLASSIFY_NAME) as run:
            for index, item in enumerate(images):
                run.item(index, CLASSIFY_STAGE)
                answers.append(engine.choose(classifier, wording, option_list,
                                             image=fit_picture(item, engine.CHECK_MAX_PIXELS)))

        choices = [choice for choice, _confidence in answers]
        indexes = [option_list.index(choice) for choice in choices]
        confidences = [float(confidence) for _choice, confidence in answers]
        verdicts = [f"{choice} {confidence:.2f}" for choice, confidence in zip(choices, confidences, strict=True)]
        return {"ui": {"text": (format_display(verdicts),)}, "result": (choices, indexes, confidences, images)}
