"""Model access every node of the pack shares: image checks, the classifier, generate, nouls, choices and the presence question.

No comfy, torch or logit_classifier module is imported at module scope, so the unit tests
run without ComfyUI and a missing library fails with its pip command.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

# Core only resizes a VL image above 12.8 MP, so a 4k input would cost about 12k vision tokens.
# The Tagger's propose and verify stages share one vision encode at this cap.
VL_MAX_PIXELS = 1024 * 1024
# At 0.5 MP one question took 276 ms against 553 ms at 1 MP, and every logit probe stayed inside its 1 MP interval
# (tests-AB/ab_release_gates.py --max-pixels).
CHECK_MAX_PIXELS = 512 * 1024

_IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

PRESENCE_QUESTION = "Is there {THING} in this image?"
ARTICLES = frozenset({"a", "an", "the", "some"})
NUMBER_WORDS = frozenset({"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"})
IRREGULAR_PLURALS = frozenset({"people", "children", "cacti", "mice"})
# Head nouns that take no article, so a question never reads "Is there a water in this image?".
MASS_NOUNS = frozenset({"water", "butter", "steam", "rain", "sunlight", "dust", "grass", "lightning", "armor",
                        "flour", "fruit", "syrup", "hair", "snow", "cheese", "linen", "dough", "spray"})


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


def resize_for_vl(image: Any, *, max_pixels: int | None = None) -> Any:
    """Downscale one [1, H, W, C] image to at most max_pixels, keeping the first three channels.

    None means VL_MAX_PIXELS, read at call time so a harness or test that sets the constant reaches every caller.
    """
    import torch

    height, width = image.shape[1], image.shape[2]
    cap = VL_MAX_PIXELS if max_pixels is None else max_pixels
    rgb = image[..., :3]
    scale = 0.0
    samples = None

    if height * width <= cap:
        return rgb
    scale = math.sqrt(cap / (height * width))
    # Floor, so the product can never land above the cap.
    size = (max(1, math.floor(height * scale)), max(1, math.floor(width * scale)))
    samples = torch.nn.functional.interpolate(rgb.movedim(-1, 1), size=size, mode="area")
    return samples.movedim(1, -1)


def library_error(node: str) -> RuntimeError:
    """The error for a missing or outdated logit-classifier, with the pip command that fixes it."""
    return RuntimeError(
        f"{node}: this node needs logit-classifier>=0.3.0, which provides the ComfyUI CLIP "
        "backend. Install or upgrade it in the ComfyUI Python environment with: "
        "pip install -U logit-classifier"
    )


def build_classifier(clip: Any, node: str, *, needs_images: bool = True) -> Any:
    """One Classifier over the workflow's CLIP, with no prior drift and no calibration file."""
    # shared_vision_encode is imported so a library older than 0.3.0 fails here with the pip command.
    try:
        from logit_classifier import Classifier, Config, UnsupportedModelError
        from logit_classifier.backends.comfy_clip import ComfyClipBackend, shared_vision_encode  # noqa: F401
    except ImportError as error:
        raise library_error(node) from error

    try:
        backend = ComfyClipBackend(clip)
    except UnsupportedModelError as error:
        # The library's own text also says to load a Qwen3-VL encoder, so it stays in the chained cause.
        raise ValueError(
            f"{node}: this CLIP is not a Qwen3-VL text encoder, so {node} cannot ask it questions. Load a Qwen3-VL "
            "text encoder, such as qwen3vl_4b_fp8_scaled.safetensors, with the Load CLIP type set to krea2."
        ) from error
    if needs_images and not backend.sees_images:
        raise ValueError(
            f"{node}: this CLIP was loaded without its vision tower, so {node} cannot read the "
            "image. Load a Qwen3-VL text encoder with the Load CLIP type set to krea2. The flux "
            "and flux2 types load it without the vision tower."
        )
    return Classifier(Config(use_prior_debias=False, calibration_path=None), backend)


def clip_generate(clip: Any, tokens: Any, **kwargs: Any) -> Any:
    """clip.generate, then core's cleanup of the CUDA graphs that generate captured."""
    # Core's Qwen3 CUDA graph decode keeps each layer's graph bound to the freed KV cache and asserts on the
    # next generate in the same node. The cleanup keeps graph decode on, which measured 100 to 16.5 ms per token.
    try:
        return clip.generate(tokens, **kwargs)
    finally:
        try:
            import comfy.model_prefetch as model_prefetch
        except ImportError:
            model_prefetch = None
        cleanup = getattr(model_prefetch, "cleanup_prefetch_queues", None)
        if cleanup is not None:
            cleanup()


def _transformer(clip: Any) -> Any:
    """Core's generating transformer, or None for a CLIP without one."""
    model = getattr(clip, "cond_stage_model", None)
    encoder_name = getattr(model, "clip", None)
    encoder = getattr(model, encoder_name, None) if isinstance(encoder_name, str) else None

    return getattr(encoder, "transformer", None)


@contextmanager
def stop_when(clip: Any, should_stop: Callable[[list[str]], bool]) -> Iterator[None]:
    """End core's greedy decode on a stop token once should_stop(the complete tags so far) is true.

    Core's generate loop takes no stop condition, so the token sample_token returns is replaced.
    Core copies that token into the sequence and breaks on it, at comfy/text_encoders/llama.py:1170-1187.
    """
    from logit_classifier.tags import complete_tags

    transformer = _transformer(clip)
    config = getattr(getattr(transformer, "model", None), "config", None)
    stop_tokens = getattr(config, "stop_tokens", None)
    history: list[int] = []

    if not callable(getattr(transformer, "sample_token", None)) or not stop_tokens:
        yield
        return
    stop_id = stop_tokens[0]
    original = transformer.sample_token
    shadowed = "sample_token" in vars(transformer)
    previous = vars(transformer).get("sample_token")

    def watch(*args: Any, **kwargs: Any) -> Any:
        token = original(*args, **kwargs)
        history.append(int(token.reshape(-1)[0]))
        if should_stop(complete_tags(clip.decode(history))):
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


def chat(user: str, *, image: bool = False) -> str:
    """The Qwen chat text for one user turn, with the image slot first when image is true."""
    # The tokenizer reads a <|...|> spelling in plain text as a control token, so an unescaped prompt could
    # close the user turn.
    safe_user = user.replace("<|", "<\u200b|")

    # A text starting with <|im_start|> skips core's template, whose empty think block made this model reply
    # empty or refuse.
    return f"<|im_start|>user\n{_IMAGE_TOKENS if image else ''}{safe_user}<|im_end|>\n<|im_start|>assistant\n"


def generate_text(clip: Any, user: str, *, image: Any = None, max_tokens: int,
                  should_stop: Callable[[list[str]], bool] | None = None, whole_tags: bool = True) -> str:
    """One greedy reply to user. With whole_tags, a reply that fills max_tokens is cut back to its last whole tag."""
    text = chat(user, image=image is not None)
    stopper: Any = nullcontext() if should_stop is None else stop_when(clip, should_stop)
    tokens = None
    ids = None
    reply = ""

    tokens = clip.tokenize(text) if image is None else clip.tokenize(text, images=[image])
    with stopper:
        ids = clip_generate(clip, tokens, do_sample=False, max_length=max_tokens)
    reply = clip.decode(ids)
    # Core stops early only on a stop token, so a decode that fills the budget ends mid tag.
    if whole_tags and len(ids) >= max_tokens:
        from logit_classifier.tags import drop_unfinished_tag

        return drop_unfinished_tag(reply)
    return reply


def nouls(classifier: Any, statements: list[str], *, image: Any = None, state: str = "") -> list[float]:
    """Each statement's noul probability, in order, from one request that asks about all of them."""
    questions: dict[str, Any] = {}

    if not statements:
        return []
    from logit_classifier import NoulQuestion, SystemOneRequest

    questions = {f"q{index}": NoulQuestion(instructions=statement) for index, statement in enumerate(statements)}
    # The empty state is what the library renders as its image only state.
    response, _diagnostics = classifier.classify(SystemOneRequest(state=state, questions=questions), image=image)
    return [response.answers[qid].noul for qid in questions]


def choose(classifier: Any, question: str, options: list[str], *, image: Any) -> tuple[str, float]:
    """The option one choice question picks about image, with the answer's confidence."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    request = SystemOneRequest(state="", questions={
        "choice": ChoiceQuestion(instructions=question, criteria=dict.fromkeys(options)),
    })
    response, _diagnostics = classifier.classify(request, image=image)
    answer = response.answers["choice"]
    return answer.choice, answer.confidence


def head_noun(phrase: str) -> str:
    """The noun a thing phrase names: its last word before 'of', 'reading' or an upper case word."""
    words: list[str] = []

    for word in phrase.split():
        if word in ("of", "reading") or word.isupper():
            break
        words.append(word.lower())
    return words[-1] if words else phrase.lower()


def reads_plural(phrase: str) -> bool:
    first = phrase.split()[0].lower()
    head = head_noun(phrase)

    if first in NUMBER_WORDS or first.isdigit() or head in IRREGULAR_PLURALS or head.endswith("men"):
        return True
    return head.endswith("s") and not head.endswith(("ss", "us"))


def with_article(phrase: str) -> str:
    """The phrase with a or an, unless it opens with an article or number, reads plural or names a mass noun."""
    first = phrase.split()[0].lower()

    if first in ARTICLES or first in NUMBER_WORDS or first.isdigit() or reads_plural(phrase):
        return phrase
    if head_noun(phrase) in MASS_NOUNS:
        return phrase
    return f"{'an' if first[0] in 'aeiou' else 'a'} {phrase}"


def presence_question(thing: str) -> str:
    """PRESENCE_QUESTION about thing, with the article its head noun takes."""
    return PRESENCE_QUESTION.replace("{THING}", with_article(thing))
