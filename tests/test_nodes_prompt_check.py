"""nodes.py and tagging.check_prompt: the Logit Prompt Check node, its contract, scores, missing things and display.

The engine, the toolkit's prompt_tags and fit, comfy, comfy_execution and the server are stubbed, so no comfy
install, no GPU and no model are needed.
"""

import re
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from logit_tagger import engine, progress, tagging
from logit_tagger.nodes import NO_THINGS_LINE, LogitPromptCheck

NODE_ID = "31"


def an_image(batch=1, value=0.0):
    return torch.full((batch, 8, 8, 3), value)


def a_batch(*values):
    return torch.stack([torch.full((8, 8, 3), value) for value in values])


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def run_check(image, prompt, clip="clip"):
    """The node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    return LogitPromptCheck().check(clip=[clip], image=image, prompt=list(prompt), unique_id=[NODE_ID])


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
    state = SimpleNamespace(interrupted=False, sent=[])

    def throw_exception_if_processing_interrupted():
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
def fake_engine(monkeypatch):
    """Replaces the classifier build and the toolkit's fit, with a classifier whose nouls answers each call with the
    next queued list."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = SimpleNamespace(scores=[], built=[], calls=[])

    class RecordingClassifier:
        def nouls(self, statements, *, image=None, state=""):
            record.calls.append(SimpleNamespace(classifier=self, statements=statements, image=image, state=state))
            return record.scores.pop(0)

    record.classifier = RecordingClassifier()

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node, needs_images))
        return record.classifier

    def fit_picture(image, max_pixels):
        return SimpleNamespace(resized=image)

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    monkeypatch.setattr(toolkit, "fit_picture", fit_picture)
    return record


@pytest.fixture
def fake_things(monkeypatch):
    """Replaces the toolkit's prompt_tags with a lookup in things_of, recording each call."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = SimpleNamespace(things_of={}, calls=[])

    def prompt_tags(clip, classifier, prompt, settings, known):
        record.calls.append(SimpleNamespace(clip=clip, classifier=classifier, prompt=prompt, settings=settings,
                                            known=known))
        return SimpleNamespace(tags=record.things_of[prompt])

    monkeypatch.setattr(toolkit, "prompt_tags", prompt_tags)
    return record


def is_blocked(side):
    return len(side) == 1 and isinstance(side[0], ExecutionBlocker) and side[0].message is None


# --- input and output contract --------------------------------------------------------


def test_inputs_are_clip_image_and_a_prompt_socket_then_the_node_id():
    spec = LogitPromptCheck.INPUT_TYPES()

    assert list(spec) == ["required", "hidden"]
    assert [(name, definition[0]) for name, definition in spec["required"].items()] == [
        ("clip", "CLIP"), ("image", "IMAGE"), ("prompt", "STRING")]
    assert spec["required"]["prompt"][1]["forceInput"] is True
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_output_contract():
    assert LogitPromptCheck.RETURN_TYPES == ("BOOLEAN", "FLOAT", "STRING", "IMAGE", "IMAGE", "IMAGE")
    assert LogitPromptCheck.RETURN_NAMES == ("all_present", "score", "missing", "images", "complete_images",
                                             "incomplete_images")
    assert LogitPromptCheck.INPUT_IS_LIST is True
    assert LogitPromptCheck.OUTPUT_IS_LIST == (False, True, True, True, True, True)
    assert LogitPromptCheck.OUTPUT_NODE is True
    assert LogitPromptCheck.FUNCTION == "check"
    assert LogitPromptCheck.CATEGORY == "Logit Tagger"
    assert 1 <= len(sentences(LogitPromptCheck.DESCRIPTION)) <= 4


def test_every_input_has_a_tooltip_and_no_text_has_an_em_dash_or_semicolon():
    tooltips = [definition[1].get("tooltip", "") for definition in LogitPromptCheck.INPUT_TYPES()["required"].values()]

    for text in tooltips:
        assert 1 <= len(sentences(text)) <= 2, text
    for text in [LogitPromptCheck.DESCRIPTION, *tooltips]:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- scores and missing things --------------------------------------------------------


def test_every_thing_present_answers_true_with_a_full_score(fake_comfy, fake_engine, fake_things):
    clip = object()
    fake_things.things_of = {"a cat in a red hat": ("cat", "red hat")}
    fake_engine.scores = [[0.9, tagging.MISSING_THRESHOLD]]

    output = run_check([an_image()], ["a cat in a red hat"], clip=clip)

    all_present, scores, missing, images, complete_images, incomplete_images = output["result"]
    assert all_present is True
    assert scores == [1.0]
    assert missing == [""]
    assert [tuple(image.shape) for image in images] == [(1, 8, 8, 3)]
    assert len(complete_images) == 1 and complete_images[0] is images[0]
    assert is_blocked(incomplete_images)
    assert fake_engine.built == [(clip, "Logit Prompt Check", True)]
    assert [(call.clip, call.classifier) for call in fake_things.calls] == [(clip, fake_engine.classifier)]
    assert fake_things.calls[0].settings is tagging.tag_settings()
    assert fake_engine.calls[0].statements == ["This image visibly contains cat", "This image visibly contains red hat"]
    assert fake_engine.calls[0].classifier is fake_engine.classifier
    assert output["ui"] == {"text": ("cat, red hat\n1.00 all present",)}


def test_things_below_the_cutoff_are_missing_in_prompt_order(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat, a red hat and two dogs": ("cat", "red hat", "two dogs")}
    fake_engine.scores = [[0.9, 0.0099, 0.001]]

    output = run_check([an_image()], ["a cat, a red hat and two dogs"])

    all_present, scores, missing, images, complete_images, incomplete_images = output["result"]
    assert all_present is False
    assert scores == [pytest.approx(1 / 3)]
    assert missing == ["red hat, two dogs"]
    assert is_blocked(complete_images)
    assert len(incomplete_images) == 1 and incomplete_images[0] is images[0]
    assert output["ui"] == {"text": ("cat, red hat, two dogs\n0.33 missing: red hat, two dogs",)}


def test_a_prompt_with_no_things_scores_one_and_asks_the_image_nothing(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a calm mood": ()}

    output = run_check([an_image()], ["a calm mood"])

    assert output["result"][:3] == (True, [1.0], [""])
    assert fake_engine.calls == []
    assert output["ui"] == {"text": (f"{NO_THINGS_LINE}\n1.00 all present",)}


# --- pairing prompts with images ------------------------------------------------------


def test_one_prompt_for_three_images_lists_its_things_once(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat": ("cat",)}
    fake_engine.scores = [[0.9], [0.002], [0.8]]

    output = run_check([a_batch(0.1, 0.2), an_image(value=0.3)], ["a cat"])

    all_present, scores, missing, images, complete_images, incomplete_images = output["result"]
    assert [call.prompt for call in fake_things.calls] == ["a cat"]
    assert all_present is False
    assert scores == [1.0, 0.0, 1.0]
    assert missing == ["", "cat", ""]
    assert [round(float(call.image.resized.mean()), 1) for call in fake_engine.calls] == [0.1, 0.2, 0.3]
    assert [round(float(image.mean()), 1) for image in images] == [0.1, 0.2, 0.3]
    assert [tuple(image.shape) for image in images] == [(1, 8, 8, 3)] * 3
    assert [round(float(image.mean()), 1) for image in complete_images] == [0.1, 0.3]
    assert [round(float(image.mean()), 1) for image in incomplete_images] == [0.2]
    assert output["ui"] == {"text": ("cat\n0: 1.00 all present\n1: 0.00 missing: cat\n2: 1.00 all present",)}


def test_two_prompts_for_two_images_pair_in_order_and_share_one_thing_cache(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat": ("cat",), "a dog": ("dog",)}
    fake_engine.scores = [[0.9], [0.001]]

    output = run_check([a_batch(0.1, 0.2)], ["a cat", "a dog"])

    assert [call.prompt for call in fake_things.calls] == ["a cat", "a dog"]
    assert fake_things.calls[0].known is fake_things.calls[1].known
    assert [call.statements for call in fake_engine.calls] == [["This image visibly contains cat"],
                                                               ["This image visibly contains dog"]]
    assert output["result"][:3] == (False, [1.0, 0.0], ["", "dog"])
    assert output["ui"] == {"text": ("0: cat | 1.00 all present\n1: dog | 0.00 missing: dog",)}


def test_prompts_with_different_things_show_each_image_after_its_own_things(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat and a dog": ("cat", "dog"), "a car on a road": ("car", "road"), "calm": ()}
    fake_engine.scores = [[0.9, 0.8], [0.001, 0.9]]

    output = run_check([a_batch(0.1, 0.2, 0.3)], ["a cat and a dog", "a car on a road", "calm"])

    assert output["ui"] == {"text": (
        f"0: cat, dog | 1.00 all present\n1: car, road | 0.50 missing: car\n2: {NO_THINGS_LINE} | 1.00 all present",)}


def test_prompts_with_the_same_things_share_one_line_of_things(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat and a dog": ("cat", "dog"), "a dog and a cat": ("cat", "dog")}
    fake_engine.scores = [[0.9, 0.8], [0.001, 0.9]]

    output = run_check([a_batch(0.1, 0.2)], ["a cat and a dog", "a dog and a cat"])

    assert output["ui"] == {"text": ("cat, dog\n0: 1.00 all present\n1: 0.50 missing: cat",)}


def test_a_repeated_prompt_is_read_once(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat": ("cat",), "a dog": ("dog",)}
    fake_engine.scores = [[0.9], [0.9], [0.9]]

    run_check([a_batch(0.1, 0.2, 0.3)], ["a cat", "a dog", "a cat"])

    assert [call.prompt for call in fake_things.calls] == ["a cat", "a dog"]
    assert fake_engine.calls[2].statements == ["This image visibly contains cat"]


def test_a_wrong_prompt_count_is_an_error_before_the_run_starts(fake_comfy, fake_engine, fake_things):
    with pytest.raises(ValueError, match="Logit Prompt Check: the prompt input has 2 texts for 3 images"):
        run_check([a_batch(0.1, 0.2, 0.3)], ["a cat", "a dog"])

    assert fake_engine.built == []
    assert fake_things.calls == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize("image", [[], [torch.zeros(0, 8, 8, 3)]])
def test_no_images_is_an_error_before_the_run_starts(fake_comfy, fake_engine, fake_things, image):
    with pytest.raises(ValueError, match="Logit Prompt Check: the image input holds no images"):
        run_check(image, ["a cat"])

    assert fake_engine.built == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize(("images", "prompts", "message"), [
    ([an_image(), an_image()], ["a cat"], "Logit Prompt Check: got 2 images and 1 prompts"),
    ([an_image(batch=2)], ["a cat"], r"Logit Prompt Check: each image must be one image of shape \[1, H, W, C\]"),
])
def test_check_prompt_rejects_unpaired_or_batched_images_before_the_classifier(fake_engine, fake_things, images,
                                                                              prompts, message):
    with pytest.raises(ValueError, match=message):
        tagging.check_prompt("clip", images, prompts)

    assert fake_engine.built == []


# --- progress, status and interrupt wiring --------------------------------------------


def test_one_bar_closes_each_image_and_the_status_names_each_stage(fake_comfy, fake_engine, fake_things):
    fake_things.things_of = {"a cat": ("cat",), "a dog": ("dog",)}
    fake_engine.scores = [[0.9], [0.9], [0.9]]

    run_check([a_batch(0.1, 0.2, 0.3)], ["a cat", "a cat", "a dog"])

    assert len(RecordingProgressBar.instances) == 1
    bar = RecordingProgressBar.instances[0]
    assert bar.total == 3 * progress.ITEM_UNITS
    assert bar.updates[-1] == (300, 300)
    assert fake_comfy.sent[:-1] == [
        ("image 1/3: reading prompt", NODE_ID), ("image 1/3: checking", NODE_ID),
        ("image 2/3: checking", NODE_ID),
        ("image 3/3: reading prompt", NODE_ID), ("image 3/3: checking", NODE_ID),
    ]
    assert re.fullmatch(r"3 images in \d+\.\d s", fake_comfy.sent[-1][0])
    assert sys.modules["comfy.utils"].ProgressBar is RecordingProgressBar


def test_an_interrupt_stops_the_run_before_the_prompt_is_read(fake_comfy, fake_engine, fake_things):
    fake_comfy.interrupted = True
    fake_things.things_of = {"a cat": ("cat",)}

    with pytest.raises(Interrupted):
        run_check([an_image()], ["a cat"])

    assert fake_things.calls == []
    assert fake_engine.calls == []
    assert fake_comfy.sent == [("cancelled", NODE_ID)]
