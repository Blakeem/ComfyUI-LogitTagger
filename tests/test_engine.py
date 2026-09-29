"""engine.py: the model access every node of the pack shares.

Opaque CLIPs and fake classifiers stand in for the Qwen3-VL text encoder and the library's
Classifier, so no comfy install, no GPU and no model are needed.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from logit_tagger import engine, nodes, progress, tagging

PACK_ROOT = Path(__file__).parent.parent.resolve()
NODE = "Some Node"


def an_image(height=8, width=8, channels=3, batch=1):
    torch = pytest.importorskip("torch")
    return torch.rand(batch, height, width, channels)


# --- module scope ---------------------------------------------------------------------


def test_module_scope_imports_no_comfy_no_torch_and_no_library():
    script = (
        "import sys\n"
        "import logit_tagger.engine\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'logit_classifier')]\n"
        "print(loaded)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], cwd=PACK_ROOT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_pixel_caps_are_1_mp_for_the_tagger_and_half_that_for_the_checks():
    assert engine.VL_MAX_PIXELS == 1024 * 1024
    assert engine.CHECK_MAX_PIXELS == 524288


# --- pixel cap at each call site ------------------------------------------------------


class Fitted(Exception):
    """Stops a node path right after its fit, so a call-site test needs no model fakes."""


class QuietRun:
    """Stands in for progress.RunProgress, which imports comfy."""

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def item(self, index, stage):
        pass


@pytest.fixture
def fit_calls(monkeypatch):
    """Replaces the toolkit's fit_image with one that records its pixel cap, then raises Fitted."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    calls = []

    def fit_image(image, max_pixels):
        calls.append(max_pixels)
        raise Fitted

    monkeypatch.setattr(toolkit, "fit_image", fit_image)
    return calls


def test_image_check_fits_to_the_check_cap(fit_calls):
    with pytest.raises(Fitted):
        nodes._image_probability("classifier", "Is there a cat in this image?", an_image())
    assert fit_calls == [engine.CHECK_MAX_PIXELS]


def test_image_classify_fits_to_the_check_cap(fit_calls, monkeypatch):
    monkeypatch.setattr(engine, "build_classifier", lambda *_args, **_kwargs: "classifier")
    monkeypatch.setattr(progress, "RunProgress", QuietRun)

    with pytest.raises(Fitted):
        nodes.LogitImageClassify().classify(clip=["clip"], image=[an_image()], question=[""], options=[""],
                                            unique_id=[NODE])
    assert fit_calls == [engine.CHECK_MAX_PIXELS]


def test_prompt_check_fits_to_the_check_cap(fit_calls):
    with pytest.raises(Fitted):
        tagging._presence_probabilities("classifier", an_image(), ("cat",))
    assert fit_calls == [engine.CHECK_MAX_PIXELS]


def test_tag_list_fits_to_the_check_cap(fit_calls):
    with pytest.raises(Fitted):
        tagging.check_tag_list("clip", "classifier", an_image(), ("cat",), tagging.tag_settings())
    assert fit_calls == [engine.CHECK_MAX_PIXELS]


def test_tagger_propose_and_verify_fit_to_the_tagger_cap(fit_calls):
    with pytest.raises(Fitted):
        tagging.tag_image("clip", "classifier", an_image(), thing_cache={})
    assert fit_calls == [engine.VL_MAX_PIXELS]


# --- check_images ---------------------------------------------------------------------


def test_check_images_accepts_rgb_and_rgba_batches():
    engine.check_images(an_image(batch=2), NODE)
    engine.check_images(an_image(channels=4), NODE)


def test_check_images_rejects_an_image_that_is_not_4d_naming_the_node():
    with pytest.raises(ValueError, match=f"^{NODE}: .*\\[B, H, W, C\\]"):
        engine.check_images(an_image()[0], NODE)


def test_check_images_rejects_fewer_than_3_channels_naming_the_node():
    with pytest.raises(ValueError, match=f"^{NODE}: .*RGB or RGBA"):
        engine.check_images(an_image(channels=1), NODE)


def test_check_images_rejects_an_object_without_a_shape():
    with pytest.raises(ValueError, match=NODE):
        engine.check_images(object(), NODE)


# --- build_classifier -----------------------------------------------------------------


def test_missing_library_names_the_node_the_version_the_toolkit_and_the_command(monkeypatch):
    # A module an earlier test imported would still import by its full name, so every one is blocked.
    for name in ["logit_classifier", *(module for module in sys.modules if module.startswith("logit_classifier."))]:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(RuntimeError) as raised:
        engine.build_classifier(object(), NODE)
    message = str(raised.value)
    assert message.startswith(f"{NODE}: ")
    assert "logit-classifier>=0.4.0" in message
    assert "logit_classifier.toolkit" in message
    assert "pip install -U logit-classifier" in message


def test_a_library_without_the_toolkit_gets_the_same_message(monkeypatch):
    pytest.importorskip("logit_classifier")
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)

    with pytest.raises(RuntimeError, match=rf"^{NODE}: .*logit-classifier>=0\.4\.0.*pip install -U logit-classifier"):
        engine.build_classifier(object(), NODE)


@pytest.fixture
def recording_classifier(monkeypatch):
    """Replaces the backend and the Classifier the toolkit builds, and records what the Classifier was built with."""
    toolkit_classifier = pytest.importorskip("logit_classifier.toolkit.comfyui.classifier")
    built = {}

    class Backend:
        sees_images = True

        def __init__(self, clip, **_kwargs):
            self.clip = clip

    class RecordingClassifier:
        def __init__(self, config, backend):
            built["config"] = config
            built["backend"] = backend

    monkeypatch.setattr(toolkit_classifier, "ComfyClipBackend", Backend)
    monkeypatch.setattr(toolkit_classifier, "Classifier", RecordingClassifier)
    built["backend_class"] = Backend
    return built


def test_build_classifier_turns_off_the_prior_and_the_calibration_file(recording_classifier):
    clip = object()

    engine.build_classifier(clip, NODE)
    assert recording_classifier["config"].use_prior_debias is False
    assert recording_classifier["config"].calibration_path is None
    assert recording_classifier["backend"].clip is clip


def test_a_clip_without_a_vision_tower_is_rejected_naming_the_node(recording_classifier):
    from logit_classifier import VisionUnsupportedError

    recording_classifier["backend_class"].sees_images = False

    with pytest.raises(VisionUnsupportedError, match=f"^{NODE}: .*vision tower.*krea2"):
        engine.build_classifier(object(), NODE)
    assert "config" not in recording_classifier


def test_a_text_only_node_accepts_a_clip_without_a_vision_tower(recording_classifier):
    recording_classifier["backend_class"].sees_images = False

    engine.build_classifier(object(), NODE, needs_images=False)
    assert recording_classifier["config"].use_prior_debias is False


def test_a_clip_that_is_not_qwen3_vl_is_rejected_naming_the_node_and_the_load_clip_fix(monkeypatch):
    toolkit_classifier = pytest.importorskip("logit_classifier.toolkit.comfyui.classifier")
    from logit_classifier import UnsupportedModelError

    class TextOnlyBackend:
        def __init__(self, clip, **_kwargs):
            raise UnsupportedModelError("no vision tower in qwen_3_4b")

    monkeypatch.setattr(toolkit_classifier, "ComfyClipBackend", TextOnlyBackend)
    with pytest.raises(UnsupportedModelError, match=f"^{NODE}: this CLIP is not a Qwen3-VL .*krea2") as raised:
        engine.build_classifier(object(), NODE)
    assert isinstance(raised.value.__cause__, UnsupportedModelError)


# --- choose ---------------------------------------------------------------------------


class ChoosingClassifier:
    """Answers every choice question with a fixed choice and confidence, recording each request."""

    def __init__(self, choice, confidence):
        self.answer = SimpleNamespace(choice=choice, confidence=confidence)
        self.calls = []

    def classify(self, request, *, image=None):
        self.calls.append((request, image))
        return SimpleNamespace(answers=dict.fromkeys(request.questions, self.answer)), None


def test_choose_sends_one_choice_question_over_the_options_in_order_and_returns_the_answer():
    pytest.importorskip("logit_classifier")
    classifier = ChoosingClassifier("oil painting", 0.83)
    image = an_image()

    result = engine.choose(classifier, "What is the style?", ["photograph", "oil painting", "pixel art"], image=image)
    (request, sent_image), = classifier.calls
    (question,) = request.questions.values()
    assert result == ("oil painting", 0.83)
    assert sent_image is image
    assert request.state == ""
    assert question.type == "choice"
    assert question.instructions == "What is the style?"
    assert question.criteria == {"photograph": None, "oil painting": None, "pixel art": None}
    assert list(question.criteria) == ["photograph", "oil painting", "pixel art"]
