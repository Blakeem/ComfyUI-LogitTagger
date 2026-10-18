"""nodes.py: the Logit Tag Check node's V1 contract, its tags pairing, its strict switch and its ui/result shape.

The tag check run or the classifier, comfy and the server are stubbed, so no comfy install, no GPU and no model are
needed.
"""

import re
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from logit_tagger import engine, progress, tagging
from logit_tagger.nodes import LogitTagCheck

NODE_ID = "44"


def an_image(batch=1, value=0.0):
    return torch.full((batch, 8, 8, 3), value)


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def run_check(image, tags, clip="clip", **optional):
    """The node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    return LogitTagCheck().check(clip=[clip], image=image, tags=list(tags), unique_id=[NODE_ID], **optional)


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
    """Fake comfy.utils, comfy.model_management and server modules, and the status texts they receive."""
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
    server = types.ModuleType("server")
    state = SimpleNamespace(sent=[])

    RecordingProgressBar.instances = []
    utils.ProgressBar = RecordingProgressBar
    model_management.throw_exception_if_processing_interrupted = lambda: None
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
def stub_check(monkeypatch):
    """Replaces check_tag_lists with a stub that keeps every listed tag and reports one check per image."""
    pytest.importorskip("logit_classifier.toolkit.comfyui")
    calls = []

    def check_tag_lists(clip, images, tag_lists, *, settings, on_progress=None, positions=False):
        calls.append(SimpleNamespace(clip=clip, images=images, tag_lists=tag_lists, settings=settings,
                                     positions=positions))
        for index in range(len(images)):
            if on_progress is not None:
                on_progress(index, tagging.STAGE_CHECKING)
        return [SimpleNamespace(tags=", ".join(tags)) for tags in tag_lists]

    monkeypatch.setattr(tagging, "check_tag_lists", check_tag_lists)
    return calls


@pytest.fixture
def fake_engine(monkeypatch, resident_loads):
    """Replaces the classifier build and the toolkit's shared vision scope with a classifier that sees only cats.

    nouls is the library's own, so a statement list reaches classify as requests the way it would on a real classifier.
    """
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = SimpleNamespace(built=[], requests=[], resident_loads=resident_loads)

    class CatClassifier:
        def classify(self, request, *, image=None):
            from logit_classifier.schema import NoulAnswer

            record.requests.append([question.instructions for question in request.questions.values()])
            answers = {qid: NoulAnswer(noul=1.0 if question.instructions.endswith(" cat") else 0.0)
                       for qid, question in request.questions.items()}
            return SimpleNamespace(answers=answers), None

        def nouls(self, statements, *, image=None, state=""):
            from logit_classifier import Classifier

            return Classifier.nouls(self, statements, image=image, state=state)

    @contextmanager
    def shared_vision_encode(clip):
        yield

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node))
        return CatClassifier()

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    monkeypatch.setattr(toolkit, "shared_vision_encode", shared_vision_encode)
    return record


# --- input and output contract --------------------------------------------------------


def test_inputs_are_clip_image_and_the_tags_socket_then_the_strict_and_positions_switches_and_the_node_id():
    spec = LogitTagCheck.INPUT_TYPES()

    assert list(spec) == ["required", "optional", "hidden"]
    assert [(name, definition[0]) for name, definition in spec["required"].items()] == [
        ("clip", "CLIP"), ("image", "IMAGE"), ("tags", "STRING")]
    assert spec["required"]["tags"][1]["forceInput"] is True
    assert list(spec["optional"]) == ["strict", "positions"]
    for switch in ("strict", "positions"):
        assert spec["optional"][switch][0] == "BOOLEAN"
        assert spec["optional"][switch][1]["default"] is False
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_every_input_has_a_tooltip_of_one_or_two_sentences():
    spec = LogitTagCheck.INPUT_TYPES()

    for group in ("required", "optional"):
        for name, definition in spec[group].items():
            assert 1 <= len(sentences(definition[1].get("tooltip", ""))) <= 2, name


def test_output_contract():
    assert LogitTagCheck.RETURN_TYPES == ("STRING", "IMAGE")
    assert LogitTagCheck.RETURN_NAMES == ("tags", "images")
    assert LogitTagCheck.INPUT_IS_LIST is True
    assert LogitTagCheck.OUTPUT_IS_LIST == (True, True)
    assert LogitTagCheck.OUTPUT_NODE is True
    assert LogitTagCheck.FUNCTION == "check"
    assert LogitTagCheck.CATEGORY == "Logit Tagger"
    assert LogitTagCheck.SEARCH_ALIASES == ["check tags", "verify tags", "filter tags", "declared tags"]
    assert 1 <= len(sentences(LogitTagCheck.DESCRIPTION)) <= 3


def test_user_facing_text_has_no_em_dash_or_semicolon():
    spec = LogitTagCheck.INPUT_TYPES()
    texts = [LogitTagCheck.DESCRIPTION, *LogitTagCheck.OUTPUT_TOOLTIPS]
    texts += [definition[1]["tooltip"] for group in ("required", "optional") for definition in spec[group].values()]

    for text in texts:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- tags pairing ---------------------------------------------------------------------


def test_one_tag_text_checks_every_image(fake_comfy, stub_check):
    tags, _images = run_check([an_image(batch=3)], ["Cat, dog\ncat"])["result"]

    assert stub_check[0].tag_lists == [("cat", "dog")] * 3
    assert tags == ["cat, dog"] * 3


def test_three_texts_pair_one_to_one_with_three_images_after_the_batches_split(fake_comfy, stub_check):
    tags, _images = run_check([an_image(batch=2), an_image()], ["cat", "dog, owl", "fox"])["result"]

    assert stub_check[0].tag_lists == [("cat",), ("dog", "owl"), ("fox",)]
    assert tags == ["cat", "dog, owl", "fox"]


def test_two_texts_for_three_images_are_an_error_before_the_run_starts(fake_comfy, stub_check):
    with pytest.raises(ValueError, match=r"^Logit Tag Check: the tags input has 2 texts for 3 images"):
        run_check([an_image(batch=3)], ["cat", "dog"])

    assert stub_check == []
    assert RecordingProgressBar.instances == []


def test_a_blank_text_gives_its_image_an_empty_tag_string(fake_comfy, fake_engine):
    tags, images = run_check([an_image(batch=2)], ["cat, dog", " \n "])["result"]

    assert tags == ["cat", ""]
    assert len(images) == 2
    assert fake_engine.requests == [["This image visibly contains cat", "This image visibly contains dog"]]
    assert fake_engine.built == [("clip", tagging.TAG_CHECK_NAME)]
    assert fake_engine.resident_loads.clips == ["clip"]


@pytest.mark.parametrize("image", [[], [torch.zeros(0, 8, 8, 3)]])
def test_no_images_is_an_error_before_the_run_starts(fake_comfy, stub_check, image):
    with pytest.raises(ValueError, match=r"^Logit Tag Check: the image input holds no images"):
        run_check(image, ["cat"])

    assert stub_check == []


def test_the_node_reads_the_clip_from_its_one_item_list(fake_comfy, stub_check):
    clip = object()

    run_check([an_image()], ["cat"], clip=clip)

    assert stub_check[0].clip is clip


# --- strict ---------------------------------------------------------------------------


@pytest.mark.parametrize(("optional", "strict"), [({}, False), ({"strict": [False]}, False), ({"strict": [True]}, True)])
def test_the_strict_switch_picks_the_settings_the_run_uses(fake_comfy, stub_check, optional, strict):
    run_check([an_image()], ["cat"], **optional)

    assert stub_check[0].settings is tagging.tag_settings(strict)


def test_differing_strict_values_fail_under_the_node_name(fake_comfy, stub_check):
    with pytest.raises(ValueError, match=r"^Logit Tag Check: the strict input received 2 different values"):
        run_check([an_image()], ["cat"], strict=[False, True])

    assert stub_check == []


# --- positions ------------------------------------------------------------------------


@pytest.mark.parametrize(("optional", "positions"), [({}, False), ({"positions": [False]}, False),
                                                     ({"positions": [True]}, True)])
def test_the_positions_switch_reaches_the_run(fake_comfy, stub_check, optional, positions):
    run_check([an_image()], ["cat"], **optional)

    assert stub_check[0].positions is positions


def test_differing_positions_values_fail_under_the_node_name(fake_comfy, stub_check):
    with pytest.raises(ValueError, match=r"^Logit Tag Check: the positions input received 2 different values"):
        run_check([an_image()], ["cat"], positions=[False, True])

    assert stub_check == []


def test_positions_on_returns_the_position_terms_of_the_classifier(fake_comfy, strip_engine):
    output = run_check([strip_engine.image()], ["cat, dog, owl, fox"], positions=[True])

    assert output["result"][0] == ["cat top-left, dog"]
    assert strip_engine.built == [("clip", tagging.TAG_CHECK_NAME)]
    assert ("placing tags", NODE_ID) in fake_comfy.sent


# --- ui, result and status ------------------------------------------------------------


def test_the_display_shows_the_kept_tags_and_the_status_names_each_check(fake_comfy, stub_check):
    first = an_image(value=0.1)

    output = run_check([first, an_image(batch=2)], ["cat", "dog", ""])

    assert set(output) == {"ui", "result"}
    assert output["ui"] == {"text": ("0: cat\n1: dog\n2: ",)}
    assert output["result"][1][0] is stub_check[0].images[0]
    assert [tuple(item.shape) for item in output["result"][1]] == [(1, 8, 8, 3)] * 3
    assert len(RecordingProgressBar.instances) == 1
    assert RecordingProgressBar.instances[0].total == 3 * progress.ITEM_UNITS
    assert fake_comfy.sent[:-1] == [
        ("image 1/3: checking tags", NODE_ID), ("image 2/3: checking tags", NODE_ID),
        ("image 3/3: checking tags", NODE_ID),
    ]
    assert re.fullmatch(r"3 images in \d+\.\d s", fake_comfy.sent[-1][0])
