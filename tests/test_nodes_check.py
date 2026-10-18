"""nodes.py: the Logit Image Check and Logit Text Check nodes, their contract, answer, split and display.

The engine, comfy, comfy_execution and the server are stubbed, so no comfy install, no GPU and no model are needed.
"""

import re
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from logit_tagger import engine, progress
from logit_tagger.nodes import IMAGE_QUESTION, TEXT_QUESTION, LogitImageCheck, LogitTextCheck

NODE_ID = "21"
CHECKS = [LogitImageCheck, LogitTextCheck]


def an_image(batch=1, value=0.0):
    return torch.full((batch, 8, 8, 3), value)


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def check_images(image, question=(IMAGE_QUESTION,), clip="clip"):
    """The node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    return LogitImageCheck().check(clip=[clip], image=image, question=list(question), unique_id=[NODE_ID])


def check_texts(text, question=(TEXT_QUESTION,), clip="clip"):
    return LogitTextCheck().check(clip=[clip], text=text, question=list(question), unique_id=[NODE_ID])


class RecordingProgressBar:
    """Stands in for comfy.utils.ProgressBar and records every (value, total) it is given."""

    instances: ClassVar[list] = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.updates = []
        RecordingProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total))


class ExecutionBlocker:
    def __init__(self, message):
        self.message = message


class Interrupted(Exception):
    pass


@pytest.fixture
def fake_comfy(monkeypatch):
    """Fake comfy.utils, comfy.model_management, comfy_execution.graph_utils and server modules."""
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
    comfy_execution = types.ModuleType("comfy_execution")
    graph_utils = types.ModuleType("comfy_execution.graph_utils")
    server = types.ModuleType("server")
    state = SimpleNamespace(interrupt_checks=0, interrupted=False, sent=[])

    def throw_exception_if_processing_interrupted():
        state.interrupt_checks += 1
        if state.interrupted:
            raise Interrupted()

    RecordingProgressBar.instances = []
    utils.ProgressBar = RecordingProgressBar
    model_management.throw_exception_if_processing_interrupted = throw_exception_if_processing_interrupted
    model_management.InterruptProcessingException = Interrupted
    graph_utils.ExecutionBlocker = ExecutionBlocker
    server.PromptServer = SimpleNamespace(instance=SimpleNamespace(
        send_progress_text=lambda text, node_id: state.sent.append((text, node_id))))
    comfy.utils = utils
    comfy.model_management = model_management
    comfy_execution.graph_utils = graph_utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy_execution", comfy_execution)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph_utils", graph_utils)
    monkeypatch.setitem(sys.modules, "server", server)
    return state


@pytest.fixture
def fake_engine(monkeypatch, resident_loads):
    """Replaces the classifier build, the toolkit's fit and its resident load skip, with a classifier whose nouls
    answers each call with the next queued score."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = SimpleNamespace(scores=[], built=[], calls=[], resident_loads=resident_loads)

    class RecordingClassifier:
        def nouls(self, statements, *, image=None, state=""):
            record.calls.append(SimpleNamespace(classifier=self, statements=statements, image=image, state=state,
                                                resident=resident_loads.open))
            return [record.scores.pop(0)]

    record.classifier = RecordingClassifier()

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node, needs_images))
        return record.classifier

    def fit_image(image, max_pixels):
        return SimpleNamespace(resized=image)

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    monkeypatch.setattr(toolkit, "fit_image", fit_image)
    return record


def is_blocked(side):
    return len(side) == 1 and isinstance(side[0], ExecutionBlocker) and side[0].message is None


# --- input and output contract --------------------------------------------------------


@pytest.mark.parametrize(("node_class", "second", "second_type", "default"), [
    (LogitImageCheck, "image", "IMAGE", IMAGE_QUESTION),
    (LogitTextCheck, "text", "STRING", TEXT_QUESTION),
])
def test_inputs_are_clip_the_items_and_a_single_line_question(node_class, second, second_type, default):
    spec = node_class.INPUT_TYPES()

    assert list(spec) == ["required", "hidden"]
    assert [(name, definition[0]) for name, definition in spec["required"].items()] == [
        ("clip", "CLIP"), (second, second_type), ("question", "STRING")]
    assert spec["required"]["question"][1]["default"] == default
    assert spec["required"]["question"][1]["multiline"] is False
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_the_text_input_is_a_socket():
    assert LogitTextCheck.INPUT_TYPES()["required"]["text"][1]["forceInput"] is True


@pytest.mark.parametrize(("node_class", "return_names", "return_types"), [
    (LogitImageCheck, ("answer", "probability", "yes_images", "no_images"), ("BOOLEAN", "FLOAT", "IMAGE", "IMAGE")),
    (LogitTextCheck, ("answer", "probability", "yes_texts", "no_texts"), ("BOOLEAN", "FLOAT", "STRING", "STRING")),
])
def test_output_contract(node_class, return_names, return_types):
    assert return_types == node_class.RETURN_TYPES
    assert return_names == node_class.RETURN_NAMES
    assert node_class.INPUT_IS_LIST is True
    assert node_class.OUTPUT_IS_LIST == (False, True, True, True)
    assert node_class.OUTPUT_NODE is True
    assert node_class.FUNCTION == "check"
    assert node_class.CATEGORY == "Logit Tagger"
    assert 1 <= len(sentences(node_class.DESCRIPTION)) <= 3


@pytest.mark.parametrize("node_class", CHECKS)
def test_every_input_has_a_tooltip_and_no_text_has_an_em_dash_or_semicolon(node_class):
    tooltips = [definition[1].get("tooltip", "") for definition in node_class.INPUT_TYPES()["required"].values()]

    for text in tooltips:
        assert 1 <= len(sentences(text)) <= 3, text
    for text in [node_class.DESCRIPTION, *tooltips]:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- the question ---------------------------------------------------------------------


def test_the_default_question_is_asked(fake_comfy, fake_engine):
    fake_engine.scores = [0.9]

    check_images([an_image()])

    assert fake_engine.calls[0].statements == [IMAGE_QUESTION]


@pytest.mark.parametrize("blank", ["", " \n"])
def test_a_blank_question_asks_the_default(fake_comfy, fake_engine, blank):
    fake_engine.scores = [0.9, 0.9]

    check_images([an_image()], question=[blank])
    check_texts(["a cat"], question=[blank])

    assert fake_engine.calls[0].statements == [IMAGE_QUESTION]
    assert fake_engine.calls[1].statements == [TEXT_QUESTION]


def test_a_written_question_is_asked_of_every_item(fake_comfy, fake_engine):
    fake_engine.scores = [0.9, 0.1]

    check_images([an_image(batch=2)], question=["Is there a dog?", "Is there a dog?"])

    assert [call.statements for call in fake_engine.calls] == [["Is there a dog?"]] * 2


def test_conflicting_questions_are_an_error_before_the_run_starts(fake_comfy, fake_engine):
    with pytest.raises(ValueError, match="Logit Image Check: the question input received 2 different values"):
        check_images([an_image()], question=["Is there a dog?", "Is there a cat?"])

    assert fake_engine.built == []
    assert fake_engine.calls == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize(("run", "message"), [
    (lambda: check_images([]), "Logit Image Check: the image input holds no images"),
    (lambda: check_images([torch.zeros(0, 8, 8, 3)]), "Logit Image Check: the image input holds no images"),
    (lambda: check_texts([]), "Logit Text Check: the text input holds no texts"),
])
def test_no_items_is_an_error_before_the_run_starts(fake_comfy, fake_engine, run, message):
    with pytest.raises(ValueError, match=message):
        run()

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


# --- answer and split -----------------------------------------------------------------


@pytest.mark.parametrize(("score", "answer"), [(0.5, True), (0.4999, False)])
def test_the_threshold_is_yes_at_one_half(fake_comfy, fake_engine, score, answer):
    fake_engine.scores = [score]

    result = check_images([an_image()])["result"]

    assert result[0] is answer
    assert result[1] == [score]
    assert is_blocked(result[3] if answer else result[2])


def test_mixed_answers_split_in_input_order(fake_comfy, fake_engine):
    batch = torch.stack([torch.full((8, 8, 3), value) for value in (0.1, 0.2, 0.3, 0.4)])
    fake_engine.scores = [0.9, 0.2, 0.7, 0.1]

    answer, probabilities, yes_images, no_images = check_images([batch])["result"]

    assert answer is False
    assert probabilities == [0.9, 0.2, 0.7, 0.1]
    assert [tuple(image.shape) for image in yes_images + no_images] == [(1, 8, 8, 3)] * 4
    assert [round(float(image.mean()), 1) for image in yes_images] == [0.1, 0.3]
    assert [round(float(image.mean()), 1) for image in no_images] == [0.2, 0.4]


def test_every_yes_answers_true_and_blocks_the_no_side(fake_comfy, fake_engine):
    fake_engine.scores = [0.9, 0.6]

    answer, _probabilities, yes_images, no_images = check_images([an_image(batch=2)])["result"]

    assert answer is True
    assert len(yes_images) == 2
    assert is_blocked(no_images)


def test_every_no_answers_false_and_blocks_the_yes_side(fake_comfy, fake_engine):
    fake_engine.scores = [0.1, 0.3]

    answer, _probabilities, yes_images, no_images = check_images([an_image(batch=2)])["result"]

    assert answer is False
    assert is_blocked(yes_images)
    assert len(no_images) == 2


def test_a_list_of_two_batches_checks_each_resized_image_once(fake_comfy, fake_engine):
    first = torch.stack([torch.full((8, 8, 3), value) for value in (0.1, 0.2)])
    fake_engine.scores = [0.9, 0.8, 0.7]

    _answer, probabilities, yes_images, _no_images = check_images([first, an_image(value=0.3)])["result"]

    assert probabilities == [0.9, 0.8, 0.7]
    assert [round(float(call.image.resized.mean()), 1) for call in fake_engine.calls] == [0.1, 0.2, 0.3]
    assert [tuple(call.image.resized.shape) for call in fake_engine.calls] == [(1, 8, 8, 3)] * 3
    assert all(call.state == "" for call in fake_engine.calls)
    assert [round(float(image.mean()), 1) for image in yes_images] == [0.1, 0.2, 0.3]


def test_one_resident_load_skip_spans_every_check_of_the_run(fake_comfy, fake_engine):
    clip = object()
    fake_engine.scores = [0.9, 0.8]

    check_images([an_image(batch=2)], clip=clip)

    assert fake_engine.resident_loads.clips == [clip]
    assert [call.resident for call in fake_engine.calls] == [True, True]
    assert fake_engine.resident_loads.open is False


def test_the_image_check_reads_the_clip_from_its_list_and_needs_images(fake_comfy, fake_engine):
    clip = object()
    fake_engine.scores = [0.9]

    check_images([an_image()], clip=clip)

    assert fake_engine.built == [(clip, "Logit Image Check", True)]
    assert fake_engine.calls[0].classifier is fake_engine.classifier


def test_the_text_check_pairs_each_text_with_its_own_answer(fake_comfy, fake_engine):
    clip = object()
    fake_engine.scores = [0.95, 0.05, 0.6]

    answer, probabilities, yes_texts, no_texts = check_texts(["a cat", "no cats", "a man"], clip=clip)["result"]

    assert fake_engine.built == [(clip, "Logit Text Check", False)]
    assert [(call.state, call.image) for call in fake_engine.calls] == [("a cat", None), ("no cats", None),
                                                                          ("a man", None)]
    assert answer is False
    assert probabilities == [0.95, 0.05, 0.6]
    assert yes_texts == ["a cat", "a man"]
    assert no_texts == ["no cats"]


def test_a_text_check_with_every_yes_blocks_the_no_side(fake_comfy, fake_engine):
    fake_engine.scores = [0.8]

    answer, _probabilities, yes_texts, no_texts = check_texts(["a person"])["result"]

    assert answer is True
    assert yes_texts == ["a person"]
    assert is_blocked(no_texts)


# --- display --------------------------------------------------------------------------


def test_one_item_shows_the_question_then_its_verdict(fake_comfy, fake_engine):
    fake_engine.scores = [0.7314]

    output = check_images([an_image()])

    assert set(output) == {"ui", "result"}
    assert output["ui"] == {"text": (f"{IMAGE_QUESTION}\nyes 0.731",)}


def test_many_items_show_one_indexed_verdict_each_after_the_question(fake_comfy, fake_engine):
    fake_engine.scores = [0.9, 0.2, 0.5]

    output = check_texts(["a", "b", "c"], question=["Is it red?"])

    assert output["ui"] == {"text": ("Is it red?\n0: yes 0.900\n1: no 0.200\n2: yes 0.500",)}


# --- progress, status and interrupt wiring --------------------------------------------


def test_one_bar_closes_each_item_and_the_status_names_the_stage(fake_comfy, fake_engine):
    fake_engine.scores = [0.9, 0.1]

    check_images([an_image(batch=2)])

    assert len(RecordingProgressBar.instances) == 1
    bar = RecordingProgressBar.instances[0]
    assert bar.total == 2 * progress.ITEM_UNITS
    assert bar.updates[-1] == (200, 200)
    assert fake_comfy.interrupt_checks == 2
    assert fake_comfy.sent[:-1] == [("image 1/2: checking", NODE_ID), ("image 2/2: checking", NODE_ID)]
    assert re.fullmatch(r"2 images in \d+\.\d s", fake_comfy.sent[-1][0])
    assert sys.modules["comfy.utils"].ProgressBar is RecordingProgressBar


def test_an_interrupt_stops_the_run_before_the_first_check(fake_comfy, fake_engine):
    fake_comfy.interrupted = True
    fake_engine.scores = [0.9]

    with pytest.raises(Interrupted):
        check_texts(["a cat"])

    assert fake_engine.calls == []
    assert fake_comfy.sent == [("cancelled", NODE_ID)]
