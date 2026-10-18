"""nodes.py: the Logit Image Classify and Logit Text Classify nodes, their contract, options, choices, indexes and
display.

The engine, comfy and the server are stubbed, so no comfy install, no GPU and no model are needed.
"""

import re
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from logit_tagger import engine, progress
from logit_tagger.nodes import (
    CLASSIFY_OPTIONS,
    CLASSIFY_QUESTION,
    TEXT_CLASSIFY_QUESTION,
    LogitImageClassify,
    LogitTextClassify,
)

NODE_ID = "41"
DEFAULT_OPTIONS = ["photograph", "anime illustration", "oil painting", "3D render", "watercolor", "pencil sketch",
                   "pixel art", "comic book"]


def an_image(batch=1, value=0.0):
    return torch.full((batch, 8, 8, 3), value)


def a_batch(*values):
    return torch.stack([torch.full((8, 8, 3), value) for value in values])


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def classify(image, question=(CLASSIFY_QUESTION,), options=(CLASSIFY_OPTIONS,), clip="clip"):
    """The node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    return LogitImageClassify().classify(clip=[clip], image=image, question=list(question), options=list(options),
                                         unique_id=[NODE_ID])


class RecordingProgressBar:
    """Stands in for comfy.utils.ProgressBar and records every (value, total) it is given."""

    instances: ClassVar[list] = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.updates = []
        RecordingProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total))


class Interrupted(Exception):
    pass


@pytest.fixture
def fake_comfy(monkeypatch):
    """Fake comfy.utils, comfy.model_management and server modules."""
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
    server = types.ModuleType("server")
    state = SimpleNamespace(interrupted=False, sent=[])

    def throw_exception_if_processing_interrupted():
        if state.interrupted:
            raise Interrupted()

    RecordingProgressBar.instances = []
    utils.ProgressBar = RecordingProgressBar
    model_management.throw_exception_if_processing_interrupted = throw_exception_if_processing_interrupted
    model_management.InterruptProcessingException = Interrupted
    server.PromptServer = SimpleNamespace(instance=SimpleNamespace(
        send_progress_text=lambda text, node_id: state.sent.append((text, node_id))))
    comfy.utils = utils
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "server", server)
    return state


@pytest.fixture
def fake_engine(monkeypatch, resident_loads):
    """Replaces the classifier build, the toolkit's fit and resident load skip, and choose, which answers each call
    with the next queued answer."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = SimpleNamespace(answers=[], built=[], calls=[], resident_loads=resident_loads)

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node, needs_images))
        return "classifier"

    def fit_image(image, max_pixels):
        return SimpleNamespace(resized=image)

    def choose(classifier, question, options, *, image=None, state=""):
        record.calls.append(SimpleNamespace(classifier=classifier, question=question, options=options, image=image,
                                            state=state, resident=resident_loads.open))
        return record.answers.pop(0)

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    monkeypatch.setattr(toolkit, "fit_image", fit_image)
    monkeypatch.setattr(engine, "choose", choose)
    return record


# --- input and output contract --------------------------------------------------------


def test_inputs_are_clip_image_and_single_line_question_and_options_then_the_node_id():
    spec = LogitImageClassify.INPUT_TYPES()
    required = spec["required"]

    assert list(spec) == ["required", "hidden"]
    assert [(name, definition[0]) for name, definition in required.items()] == [
        ("clip", "CLIP"), ("image", "IMAGE"), ("question", "STRING"), ("options", "STRING")]
    assert required["question"][1]["default"] == CLASSIFY_QUESTION == "What is the art style of this image?"
    assert required["options"][1]["default"] == CLASSIFY_OPTIONS
    assert required["question"][1]["multiline"] is False
    assert required["options"][1]["multiline"] is False
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_output_contract():
    assert LogitImageClassify.RETURN_TYPES == ("STRING", "INT", "FLOAT", "IMAGE")
    assert LogitImageClassify.RETURN_NAMES == ("choice", "index", "confidence", "images")
    assert LogitImageClassify.INPUT_IS_LIST is True
    assert LogitImageClassify.OUTPUT_IS_LIST == (True, True, True, True)
    assert LogitImageClassify.OUTPUT_NODE is True
    assert LogitImageClassify.FUNCTION == "classify"
    assert LogitImageClassify.CATEGORY == "Logit Tagger"
    assert 1 <= len(sentences(LogitImageClassify.DESCRIPTION)) <= 4


@pytest.mark.parametrize("node_class", [LogitImageClassify, LogitTextClassify])
def test_every_input_has_a_tooltip_and_no_text_has_an_em_dash_or_semicolon(node_class):
    tooltips = [definition[1].get("tooltip", "") for definition in node_class.INPUT_TYPES()["required"].values()]

    for text in tooltips:
        assert 1 <= len(sentences(text)) <= 2, text
    for text in [node_class.DESCRIPTION, *node_class.OUTPUT_TOOLTIPS, *tooltips]:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- question and options -------------------------------------------------------------


def test_the_defaults_ask_the_art_style_question_over_the_eight_measured_styles(fake_comfy, fake_engine):
    fake_engine.answers = [("watercolor", 0.9)]

    classify([an_image()])

    (call,) = fake_engine.calls
    assert call.question == CLASSIFY_QUESTION
    assert call.options == DEFAULT_OPTIONS


@pytest.mark.parametrize("blank", ["", " \n"])
def test_blank_widgets_use_the_defaults(fake_comfy, fake_engine, blank):
    fake_engine.answers = [("watercolor", 0.9)]

    classify([an_image()], question=[blank], options=[blank])

    assert fake_engine.calls[0].question == CLASSIFY_QUESTION
    assert fake_engine.calls[0].options == DEFAULT_OPTIONS


def test_options_split_on_commas_and_newlines_keep_their_spelling_and_drop_blanks(fake_comfy, fake_engine):
    fake_engine.answers = [("Dog", 0.9)]

    classify([an_image()], question=["Which animal is this?"], options=[" Cat, Dog\r\nbird, ,\n"])

    assert fake_engine.calls[0].question == "Which animal is this?"
    assert fake_engine.calls[0].options == ["Cat", "Dog", "bird"]


def test_the_choice_output_keeps_the_typed_spelling_of_each_option(fake_comfy, fake_engine):
    fake_engine.answers = [("Photo", 0.9), ("3D Render", 0.8)]

    choices, indexes, _confidences, _images = classify([an_image(batch=2)], options=["Photo, 3D Render"])["result"]

    assert fake_engine.calls[0].options == ["Photo", "3D Render"]
    assert choices == ["Photo", "3D Render"]
    assert indexes == [0, 1]


def test_an_option_named_twice_is_an_error_before_the_run_starts(fake_comfy, fake_engine):
    with pytest.raises(ValueError, match=r"^Logit Image Classify: the options input names 'Cat' more than once\."):
        classify([an_image()], options=["cat, Cat"])

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize(("options", "count"), [
    ("cat", 1),
    ("cat, , \n", 1),
    (", ".join(f"option {index}" for index in range(52)), 52),
])
def test_an_option_count_outside_2_to_51_is_an_error_before_the_run_starts(fake_comfy, fake_engine, options, count):
    with pytest.raises(ValueError, match=rf"^Logit Image Classify: the node takes 2 to 51 different options, "
                                         rf"and the options input names {count}\."):
        classify([an_image()], options=[options])

    assert fake_engine.built == []
    assert fake_engine.calls == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize("count", [2, 51])
def test_2_and_51_options_are_accepted(fake_comfy, fake_engine, count):
    options = [f"option {index}" for index in range(count)]
    fake_engine.answers = [(options[-1], 0.5)]

    classify([an_image()], options=[", ".join(options)])

    assert fake_engine.calls[0].options == options


def test_a_missing_library_prints_the_pip_command_naming_the_node(fake_comfy, monkeypatch):
    # A module an earlier test imported would still import by its full name, so every one is blocked.
    for name in ["logit_classifier", *(module for module in sys.modules if module.startswith("logit_classifier."))]:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(RuntimeError, match=r'^Logit Image Classify: .*pip install -U "logit-classifier>=0\.6\.0"'):
        classify([an_image()])

    assert RecordingProgressBar.instances == []


def test_conflicting_options_are_an_error_before_the_run_starts(fake_comfy, fake_engine):
    with pytest.raises(ValueError, match="Logit Image Classify: the options input received 2 different values"):
        classify([an_image()], options=["cat, dog", "cat, bird"])

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize("image", [[], [torch.zeros(0, 8, 8, 3)]])
def test_no_images_is_an_error_before_the_run_starts(fake_comfy, fake_engine, image):
    with pytest.raises(ValueError, match="Logit Image Classify: the image input holds no images"):
        classify(image)

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


# --- choices, indexes and confidences -------------------------------------------------


def test_each_choice_maps_to_its_position_among_the_typed_options(fake_comfy, fake_engine):
    fake_engine.answers = [("comic book", 0.61), ("photograph", 0.97), ("3D render", 0.44)]

    choices, indexes, confidences, _images = classify([an_image(batch=3)])["result"]

    assert choices == ["comic book", "photograph", "3D render"]
    assert indexes == [7, 0, 3]
    assert confidences == [0.61, 0.97, 0.44]
    assert all(type(index) is int for index in indexes)
    assert all(type(confidence) is float for confidence in confidences)


def test_one_resident_load_skip_spans_every_choice_of_the_run(fake_comfy, fake_engine):
    clip = object()
    fake_engine.answers = [("cat", 0.9), ("dog", 0.8)]

    classify([a_batch(0.1, 0.2)], options=["cat, dog"], clip=clip)

    assert fake_engine.resident_loads.clips == [clip]
    assert [call.resident for call in fake_engine.calls] == [True, True]
    assert fake_engine.resident_loads.open is False


def test_a_list_of_two_batches_classifies_each_resized_image_once_in_order(fake_comfy, fake_engine):
    clip = object()
    fake_engine.answers = [("cat", 0.9), ("dog", 0.8), ("cat", 0.7)]

    choices, indexes, _confidences, images = classify([a_batch(0.1, 0.2), an_image(value=0.3)], options=["cat, dog"],
                                                      clip=clip)["result"]

    assert fake_engine.built == [(clip, "Logit Image Classify", True)]
    assert [call.classifier for call in fake_engine.calls] == ["classifier"] * 3
    assert [round(float(call.image.resized.mean()), 1) for call in fake_engine.calls] == [0.1, 0.2, 0.3]
    assert [tuple(call.image.resized.shape) for call in fake_engine.calls] == [(1, 8, 8, 3)] * 3
    assert choices == ["cat", "dog", "cat"]
    assert indexes == [0, 1, 0]
    assert [round(float(image.mean()), 1) for image in images] == [0.1, 0.2, 0.3]
    assert [tuple(image.shape) for image in images] == [(1, 8, 8, 3)] * 3


# --- display --------------------------------------------------------------------------


def test_one_image_shows_its_choice_and_confidence(fake_comfy, fake_engine):
    fake_engine.answers = [("oil painting", 0.8765)]

    output = classify([an_image()])

    assert set(output) == {"ui", "result"}
    assert output["ui"] == {"text": ("oil painting 0.88",)}


def test_many_images_show_one_indexed_choice_each(fake_comfy, fake_engine):
    fake_engine.answers = [("cat", 0.9), ("dog", 0.25)]

    output = classify([an_image(batch=2)], options=["cat, dog"])

    assert output["ui"] == {"text": ("0: cat 0.90\n1: dog 0.25",)}


# --- progress, status and interrupt wiring --------------------------------------------


def test_one_bar_closes_each_image_and_the_status_names_the_stage(fake_comfy, fake_engine):
    fake_engine.answers = [("cat", 0.9), ("dog", 0.1)]

    classify([an_image(batch=2)], options=["cat, dog"])

    assert len(RecordingProgressBar.instances) == 1
    bar = RecordingProgressBar.instances[0]
    assert bar.total == 2 * progress.ITEM_UNITS
    assert bar.updates[-1] == (200, 200)
    assert fake_comfy.sent[:-1] == [("image 1/2: classifying", NODE_ID), ("image 2/2: classifying", NODE_ID)]
    assert re.fullmatch(r"2 images in \d+\.\d s", fake_comfy.sent[-1][0])
    assert sys.modules["comfy.utils"].ProgressBar is RecordingProgressBar


def test_an_interrupt_stops_the_run_before_the_first_image(fake_comfy, fake_engine):
    fake_comfy.interrupted = True
    fake_engine.answers = [("cat", 0.9)]

    with pytest.raises(Interrupted):
        classify([an_image()])

    assert fake_engine.calls == []
    assert fake_comfy.sent == [("cancelled", NODE_ID)]


# --- Logit Text Classify --------------------------------------------------------------


def classify_texts(texts, question=(TEXT_CLASSIFY_QUESTION,), options=(CLASSIFY_OPTIONS,), clip="clip"):
    """The Text Classify node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    return LogitTextClassify().classify(clip=[clip], text=list(texts), question=list(question),
                                        options=list(options), unique_id=[NODE_ID])


class PickingClassifier:
    """Picks the first option the request's state names and records each request's state and keyword arguments.

    Its confidence rises by 0.1 with each request, so each answer is told apart by its confidence.
    """

    def __init__(self):
        self.calls = []

    def classify(self, request, **kwargs):
        (question,) = request.questions.values()
        choice = next(option for option in question.criteria if option.casefold() in request.state.casefold())

        self.calls.append((request.state, kwargs))
        answer = SimpleNamespace(choice=choice, confidence=round(0.5 + len(self.calls) / 10, 1))
        return SimpleNamespace(answers={"choice": answer}), None


@pytest.fixture
def picking_engine(monkeypatch, resident_loads):
    """Builds a PickingClassifier in place of the workflow's CLIP, and keeps engine.choose real."""
    pytest.importorskip("logit_classifier")
    record = SimpleNamespace(classifier=PickingClassifier(), built=[], resident_loads=resident_loads)

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node, needs_images))
        return record.classifier

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    return record


def test_text_classify_inputs_are_clip_a_text_socket_and_single_line_question_and_options_then_the_node_id():
    spec = LogitTextClassify.INPUT_TYPES()
    required = spec["required"]

    assert list(spec) == ["required", "hidden"]
    assert [(name, definition[0]) for name, definition in required.items()] == [
        ("clip", "CLIP"), ("text", "STRING"), ("question", "STRING"), ("options", "STRING")]
    assert required["text"][1]["forceInput"] is True
    assert required["question"][1]["default"] == TEXT_CLASSIFY_QUESTION == (
        "What is the art style of the image this prompt describes?")
    assert required["options"][1]["default"] == CLASSIFY_OPTIONS
    assert required["question"][1]["multiline"] is False
    assert required["options"][1]["multiline"] is False
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_text_classify_output_contract():
    assert LogitTextClassify.RETURN_TYPES == ("STRING", "INT", "FLOAT", "STRING")
    assert LogitTextClassify.RETURN_NAMES == ("choice", "index", "confidence", "texts")
    assert LogitTextClassify.INPUT_IS_LIST is True
    assert LogitTextClassify.OUTPUT_IS_LIST == (True, True, True, True)
    assert LogitTextClassify.OUTPUT_NODE is True
    assert LogitTextClassify.FUNCTION == "classify"
    assert LogitTextClassify.CATEGORY == "Logit Tagger"
    assert LogitTextClassify.SEARCH_ALIASES == ["text classifier", "prompt classifier", "route prompt",
                                                "prompt category"]
    assert 1 <= len(sentences(LogitTextClassify.DESCRIPTION)) <= 4


def test_text_classify_picks_one_option_per_text_from_the_classifier(fake_comfy, picking_engine):
    clip = object()
    texts = ["a watercolor of a harbor", "a pixel art castle", "a photograph of a fox"]

    choices, indexes, confidences, out_texts = classify_texts(texts, clip=clip)["result"]

    assert picking_engine.built == [(clip, "Logit Text Classify", False)]
    assert picking_engine.classifier.calls == [(text, {}) for text in texts]
    assert choices == ["watercolor", "pixel art", "photograph"]
    assert indexes == [4, 6, 0]
    assert confidences == [0.6, 0.7, 0.8]
    assert out_texts == texts


def test_text_classify_index_is_the_position_of_the_typed_option(fake_comfy, picking_engine):
    choices, indexes, _confidences, _texts = classify_texts(["a Dog", "a Cat", "a bird"],
                                                            options=[" Cat, Dog\nbird"])["result"]

    assert choices == ["Dog", "Cat", "bird"]
    assert indexes == [1, 0, 2]


@pytest.mark.parametrize("blank", ["", " \n"])
def test_text_classify_blank_widgets_use_the_defaults(fake_comfy, fake_engine, blank):
    fake_engine.answers = [("watercolor", 0.9)]

    classify_texts(["a watercolor of a harbor"], question=[blank], options=[blank])

    (call,) = fake_engine.calls
    assert call.question == TEXT_CLASSIFY_QUESTION
    assert call.options == DEFAULT_OPTIONS
    assert call.state == "a watercolor of a harbor"
    assert call.image is None


def test_text_classify_an_option_named_twice_is_an_error_naming_the_node(fake_comfy, fake_engine):
    with pytest.raises(ValueError, match=r"^Logit Text Classify: the options input names 'Cat' more than once\."):
        classify_texts(["a cat"], options=["cat, Cat"])

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


def test_text_classify_no_texts_is_an_error_before_the_run_starts(fake_comfy, fake_engine):
    with pytest.raises(ValueError, match=r"^Logit Text Classify: the text input holds no texts\. Connect at least one "
                                         r"text\.$"):
        classify_texts([])

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


def test_text_classify_one_resident_load_skip_spans_every_choice_and_the_bar_counts_texts(fake_comfy, fake_engine):
    clip = object()
    fake_engine.answers = [("cat", 0.9), ("dog", 0.25)]

    output = classify_texts(["a cat", "a dog"], options=["cat, dog"], clip=clip)

    assert fake_engine.resident_loads.clips == [clip]
    assert [call.resident for call in fake_engine.calls] == [True, True]
    assert fake_engine.resident_loads.open is False
    assert fake_comfy.sent[:-1] == [("text 1/2: classifying", NODE_ID), ("text 2/2: classifying", NODE_ID)]
    assert output["ui"] == {"text": ("0: cat 0.90\n1: dog 0.25",)}
