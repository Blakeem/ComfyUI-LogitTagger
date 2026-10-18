"""Model access every node of the pack shares: image checks, the pixel caps, the library gate, the classifier
and choices.

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
# 0.6.0 adds the toolkit's style line, prompt style listing and tag strips.
LIBRARY_VERSION = "0.6.0"


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


def library_error(node: str, reason: str) -> RuntimeError:
    """The error for a missing or outdated logit-classifier, with the pip command that fixes it."""
    return RuntimeError(
        f"{node}: this node needs logit-classifier {LIBRARY_VERSION} or newer, and {reason}. Install or upgrade it "
        f'in the ComfyUI Python environment with: pip install -U "logit-classifier>={LIBRARY_VERSION}"'
    )


def toolkit(node: str) -> Any:
    """logit_classifier.toolkit.comfyui, or library_error under node's name when the library is missing, is older
    than LIBRARY_VERSION, or reports no version this gate can read."""
    installed = None
    installed_version = None

    try:
        import logit_classifier
        import logit_classifier.toolkit.comfyui as comfy_toolkit
    except ImportError as error:
        raise library_error(node, f"its toolkit cannot be imported ({error})") from error
    # Core requires transformers, which requires packaging, so pip's own version ordering is present wherever a node
    # runs.
    from packaging.version import InvalidVersion, Version

    # The module attribute, never importlib.metadata, since an editable install can keep a stale version in its dist
    # metadata.
    installed = getattr(logit_classifier, "__version__", None)
    if installed is None:
        raise library_error(node, "the installed library reports no version")
    try:
        installed_version = Version(installed)
    except (InvalidVersion, TypeError) as error:
        raise library_error(
            node, f"the installed library reports version {installed!r}, which is not a version number") from error
    if installed_version < Version(LIBRARY_VERSION):
        raise library_error(node, f"version {installed} is installed")
    return comfy_toolkit


def build_classifier(clip: Any, node: str, *, needs_images: bool = True) -> Any:
    """One Classifier over the workflow's CLIP.

    The toolkit raises UnsupportedModelError or VisionUnsupportedError naming the node and the Load CLIP fix.
    """
    return toolkit(node).comfy_classifier(clip, node=node, needs_images=needs_images)


def choose(classifier: Any, question: str, options: list[str], *, image: Any = None,
           state: str = "") -> tuple[str, float]:
    """The option one choice question picks about state, and about image when one is given, with the answer's
    confidence."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    request = SystemOneRequest(state=state, questions={
        "choice": ChoiceQuestion(instructions=question, criteria=dict.fromkeys(options)),
    })
    if image is None:
        response, _diagnostics = classifier.classify(request)
    else:
        response, _diagnostics = classifier.classify(request, image=image)
    answer = response.answers["choice"]
    return answer.choice, answer.confidence
