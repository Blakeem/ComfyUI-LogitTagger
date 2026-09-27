"""Model access every node of the pack shares: image checks, the pixel caps, the classifier and choices.

No comfy, torch or logit_classifier module is imported at module scope, so the unit tests
run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

from typing import Any

# Core only resizes a VL image above 12.8 MP, so a 4k input would cost about 12k vision tokens.
# The Tagger's propose and verify stages share one vision encode at this cap.
VL_MAX_PIXELS = 1024 * 1024
# At 0.5 MP one question took 276 ms against 553 ms at 1 MP, and every logit probe stayed inside its 1 MP interval
# (tests-AB/ab_release_gates.py --max-pixels).
CHECK_MAX_PIXELS = 512 * 1024


def check_images(images: Any, node: str) -> None:
    """Reject anything but a ComfyUI IMAGE batch of shape [B, H, W, 3] or [B, H, W, 4]."""
    shape = tuple(getattr(images, "shape", ()))

    if len(shape) != 4:
        raise ValueError(
            f"{node}: the image input must be a ComfyUI IMAGE batch of shape [B, H, W, C], "
            f"got shape {shape}"
        )
    if shape[-1] < 3:
        raise ValueError(
            f"{node}: the image input must be an RGB or RGBA IMAGE of shape [B, H, W, 3] "
            f"or [B, H, W, 4], got shape {shape}"
        )


def library_error(node: str) -> RuntimeError:
    """The error for a missing or outdated logit-classifier, with the pip command that fixes it."""
    return RuntimeError(
        f"{node}: this node needs logit-classifier>=0.3.0, which provides the ComfyUI toolkit "
        "logit_classifier.toolkit. Install or upgrade it in the ComfyUI Python environment with: "
        "pip install -U logit-classifier"
    )


def build_classifier(clip: Any, node: str, *, needs_images: bool = True) -> Any:
    """One Classifier over the workflow's CLIP.

    The toolkit raises UnsupportedModelError or VisionUnsupportedError naming the node and the Load CLIP fix.
    """
    try:
        from logit_classifier.toolkit.comfyui import comfy_classifier
    except ImportError as error:
        raise library_error(node) from error

    return comfy_classifier(clip, node=node, needs_images=needs_images)


def choose(classifier: Any, question: str, options: list[str], *, image: Any) -> tuple[str, float]:
    """The option one choice question picks about image, with the answer's confidence."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    request = SystemOneRequest(state="", questions={
        "choice": ChoiceQuestion(instructions=question, criteria=dict.fromkeys(options)),
    })
    response, _diagnostics = classifier.classify(request, image=image)
    answer = response.answers["choice"]
    return answer.choice, answer.confidence
