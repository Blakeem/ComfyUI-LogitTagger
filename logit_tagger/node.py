"""The Logit Tagger node, in ComfyUI's V1 node contract.

comfy is imported inside the node function only, so the unit tests run without ComfyUI.
"""

from __future__ import annotations

from typing import Any

from . import tagging


def format_display(tags: list[str]) -> str:
    """One line per image, with the batch index in front once the batch holds more than one."""
    if len(tags) == 1:
        return tags[0]
    return "\n".join(f"{index}: {text}" for index, text in enumerate(tags))


class LogitTagger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A Qwen3-VL text encoder, such as the one Krea 2 loads. It proposes the tags and then checks each one against the image."}),
                "image": ("IMAGE", {"tooltip": "The images to tag. Each image in the batch is tagged on its own."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional text, such as the prompt the image was made from. Its words set the vocabulary the tags use. Every tag is still checked against the image, so a prompt object the image lacks is dropped."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    # So the node runs and shows its tags with nothing wired to its output.
    OUTPUT_NODE = True
    FUNCTION = "tag"
    CATEGORY = "image/tagging"
    DESCRIPTION = (
        "Tags each image with the Qwen3-VL text encoder the workflow already holds. The encoder "
        "proposes candidate tags, and a logit classifier keeps only the tags the image shows. "
        "The output is one tag string per image."
    )

    def tag(self, clip: Any, image: Any, prompt: str) -> dict[str, Any]:
        import comfy.model_management
        import comfy.utils

        progress = comfy.utils.ProgressBar(image.shape[0])
        tags: list[str] = []

        def on_image_done(_index: int) -> None:
            progress.update(1)
            comfy.model_management.throw_exception_if_processing_interrupted()

        tags = tagging.tag_images(clip, image, prompt, on_image_done=on_image_done)
        # Core concatenates each ui value across calls, and the frontend joins a list's entries
        # with blank lines, so the display travels as one string in a one item tuple.
        return {"ui": {"text": (format_display(tags),)}, "result": (tags,)}
