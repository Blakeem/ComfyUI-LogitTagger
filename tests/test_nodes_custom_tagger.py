"""nodes.py: the Logit Custom Tagger node's V1 contract, its instruction default, its switches and its ui/result shape.

The tagging run, comfy and the server are stubbed, so no comfy install, no GPU and no model are needed.
"""

import re
import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from logit_tagger import progress, tagging
from logit_tagger.nodes import LogitCustomTagger

NODE_ID = "51"


def an_image(batch=1, value=0.0):
    return torch.full((batch, 8, 8, 3), value)


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def default_wording():
    return pytest.importorskip("logit_classifier.toolkit.comfyui").TagSettings().propose_instruction


def run_node(image, instruction=None, clip="clip", **optional):
    """The node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    wording = default_wording() if instruction is None else instruction

    return LogitCustomTagger().tag(clip=[clip], image=image, instruction=[wording], unique_id=[NODE_ID], **optional)


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
def stub_pipeline(monkeypatch):
    """Replaces tag_images with a stub that returns one canned trace per image and reports both stages per image."""
    pytest.importorskip("logit_classifier.toolkit.comfyui")
    calls = []

    def tag_images(clip, images, prompts, *, on_progress=None, settings=None, node=tagging.NODE_NAME,
                   positions=False):
        calls.append(SimpleNamespace(clip=clip, images=images, prompts=prompts, settings=settings, node=node,
                                     positions=positions))
        for index in range(len(images)):
            for stage in (tagging.STAGE_WRITING, tagging.STAGE_CHECKING):
                if on_progress is not None:
                    on_progress(index, stage)
        return [SimpleNamespace(tags=f"tag{index}") for index in range(len(images))]

    monkeypatch.setattr(tagging, "tag_images", tag_images)
    return calls


# --- input and output contract --------------------------------------------------------


def test_inputs_are_clip_image_and_the_instruction_then_the_three_switches_and_the_node_id():
    spec = LogitCustomTagger.INPUT_TYPES()

    assert list(spec) == ["required", "optional", "hidden"]
    assert [(name, definition[0]) for name, definition in spec["required"].items()] == [
        ("clip", "CLIP"), ("image", "IMAGE"), ("instruction", "STRING")]
    assert spec["required"]["instruction"][1]["multiline"] is True
    assert list(spec["optional"]) == ["strict", "things_only", "positions"]
    assert [definition[0] for definition in spec["optional"].values()] == ["BOOLEAN", "BOOLEAN", "BOOLEAN"]
    assert spec["optional"]["strict"][1]["default"] is False
    assert spec["optional"]["things_only"][1]["default"] is True
    assert spec["optional"]["positions"][1]["default"] is False
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_the_instruction_default_is_the_toolkit_propose_wording():
    assert LogitCustomTagger.INPUT_TYPES()["required"]["instruction"][1]["default"] == default_wording()


def test_the_instruction_default_is_empty_when_the_library_cannot_be_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)

    assert LogitCustomTagger.INPUT_TYPES()["required"]["instruction"][1]["default"] == ""


def test_every_input_has_a_tooltip_of_one_or_two_sentences():
    spec = LogitCustomTagger.INPUT_TYPES()

    for group in ("required", "optional"):
        for name, definition in spec[group].items():
            assert 1 <= len(sentences(definition[1].get("tooltip", ""))) <= 2, name


def test_the_instruction_tooltip_says_a_blank_instruction_asks_the_tagger_question():
    tooltip = LogitCustomTagger.INPUT_TYPES()["required"]["instruction"][1]["tooltip"]

    assert "A blank instruction asks the Tagger's own question." in tooltip


def test_output_contract():
    assert LogitCustomTagger.RETURN_TYPES == ("STRING", "IMAGE")
    assert LogitCustomTagger.RETURN_NAMES == ("tags", "images")
    assert LogitCustomTagger.INPUT_IS_LIST is True
    assert LogitCustomTagger.OUTPUT_IS_LIST == (True, True)
    assert LogitCustomTagger.OUTPUT_NODE is True
    assert LogitCustomTagger.FUNCTION == "tag"
    assert LogitCustomTagger.CATEGORY == "Logit Tagger"
    assert LogitCustomTagger.SEARCH_ALIASES == ["custom tags", "custom instruction", "tag instruction",
                                                "list with instruction"]
    assert 1 <= len(sentences(LogitCustomTagger.DESCRIPTION)) <= 3


def test_user_facing_text_has_no_em_dash_or_semicolon():
    spec = LogitCustomTagger.INPUT_TYPES()
    texts = [LogitCustomTagger.DESCRIPTION, *LogitCustomTagger.OUTPUT_TOOLTIPS]
    texts += [definition[1]["tooltip"] for group in ("required", "optional") for definition in spec[group].values()]

    for text in texts:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- settings -------------------------------------------------------------------------


def test_the_default_instruction_strict_off_and_things_only_on_run_the_tagger_settings(fake_comfy, stub_pipeline):
    run_node([an_image()], strict=[False], things_only=[True])

    assert stub_pipeline[0].settings == tagging.tag_settings()


def test_a_run_without_the_switches_runs_the_tagger_settings(fake_comfy, stub_pipeline):
    run_node([an_image()])

    assert stub_pipeline[0].settings == tagging.tag_settings()


def test_a_blank_instruction_proposes_with_the_default_wording(fake_comfy, stub_pipeline):
    run_node([an_image()], instruction=" \n ")

    assert stub_pipeline[0].settings.propose_instruction == default_wording()


def test_another_instruction_proposes_with_its_wording_and_no_echo_fallback(fake_comfy, stub_pipeline):
    run_node([an_image()], instruction="List the colors in this image.")

    assert stub_pipeline[0].settings.propose_instruction == "List the colors in this image."
    assert stub_pipeline[0].settings.echo_fallback_instruction is None


@pytest.mark.parametrize(("optional", "thing_check"), [({}, "all"), ({"things_only": [True]}, "all"),
                                                       ({"things_only": [False]}, "prompt")])
def test_things_only_picks_the_thing_check(fake_comfy, stub_pipeline, optional, thing_check):
    run_node([an_image()], **optional)

    assert stub_pipeline[0].settings.thing_check == thing_check


def test_strict_on_runs_the_strict_thresholds(fake_comfy, stub_pipeline):
    from logit_classifier.toolkit.tags import STRICT_THRESHOLDS

    run_node([an_image()], strict=[True])

    assert stub_pipeline[0].settings.verify_thresholds == STRICT_THRESHOLDS


@pytest.mark.parametrize(("input_name", "inputs"), [
    ("instruction", {"instruction": ["Name the animals.", "Name the plants."]}),
    ("strict", {"instruction": ["Name the animals."], "strict": [False, True]}),
    ("things_only", {"instruction": ["Name the animals."], "things_only": [True, False]}),
    ("positions", {"instruction": ["Name the animals."], "positions": [False, True]}),
])
def test_differing_widget_values_fail_under_the_node_name(fake_comfy, stub_pipeline, input_name, inputs):
    with pytest.raises(ValueError, match=rf"^Logit Custom Tagger: the {input_name} input received 2 different values"):
        LogitCustomTagger().tag(clip=["clip"], image=[an_image()], unique_id=[NODE_ID], **inputs)

    assert stub_pipeline == []


# --- run ------------------------------------------------------------------------------


def test_every_image_runs_with_an_empty_prompt_under_the_node_name(fake_comfy, stub_pipeline):
    clip = object()

    run_node([an_image(batch=2), an_image()], clip=clip)

    assert stub_pipeline[0].clip is clip
    assert stub_pipeline[0].prompts == ["", "", ""]
    assert stub_pipeline[0].node == tagging.CUSTOM_TAGGER_NAME


@pytest.mark.parametrize(("optional", "positions"), [({}, False), ({"positions": [False]}, False),
                                                     ({"positions": [True]}, True)])
def test_the_positions_switch_reaches_the_run(fake_comfy, stub_pipeline, optional, positions):
    run_node([an_image()], **optional)

    assert stub_pipeline[0].positions is positions


def test_positions_on_returns_the_position_terms_of_the_classifier(fake_comfy, strip_engine):
    output = run_node([strip_engine.image()], positions=[True])

    assert output["result"][0] == ["cat top-left, dog"]
    assert strip_engine.built == [("clip", tagging.CUSTOM_TAGGER_NAME)]
    assert ("placing tags", NODE_ID) in fake_comfy.sent


@pytest.mark.parametrize("image", [[], [torch.zeros(0, 8, 8, 3)]])
def test_no_images_is_an_error_before_the_run_starts(fake_comfy, stub_pipeline, image):
    with pytest.raises(ValueError, match=r"^Logit Custom Tagger: the image input holds no images"):
        run_node(image)

    assert stub_pipeline == []
    assert RecordingProgressBar.instances == []


def test_the_display_shows_the_tags_and_the_status_names_each_stage(fake_comfy, stub_pipeline):
    first = an_image(value=0.1)

    output = run_node([first, an_image()])

    assert set(output) == {"ui", "result"}
    assert output["ui"] == {"text": ("0: tag0\n1: tag1",)}
    assert output["result"][0] == ["tag0", "tag1"]
    assert output["result"][1][0] is stub_pipeline[0].images[0]
    assert len(RecordingProgressBar.instances) == 1
    assert RecordingProgressBar.instances[0].total == 2 * progress.ITEM_UNITS
    assert fake_comfy.sent[:-1] == [
        ("image 1/2: writing tags", NODE_ID), ("image 1/2: checking tags", NODE_ID),
        ("image 2/2: writing tags", NODE_ID), ("image 2/2: checking tags", NODE_ID),
    ]
    assert re.fullmatch(r"2 images in \d+\.\d s", fake_comfy.sent[-1][0])
