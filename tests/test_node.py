"""node.py: the Logit Tagger node's V1 contract, its ui/result shape and its registration.

The tagging pipeline and comfy are stubbed, so no comfy install, no GPU and no model are needed.
"""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from logit_tagger import node as node_module
from logit_tagger import tagging
from logit_tagger.node import LogitTagger

PACK_ROOT = Path(__file__).parent.parent.resolve()


def an_image(batch=1):
    return torch.zeros(batch, 8, 8, 3)


class RecordingProgressBar:
    """Stands in for comfy.utils.ProgressBar and records its total and every update."""

    instances: ClassVar[list] = []

    def __init__(self, total):
        self.total = total
        self.updates = []
        RecordingProgressBar.instances.append(self)

    def update(self, value):
        self.updates.append(value)


@pytest.fixture
def fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
    interrupt_checks = []

    RecordingProgressBar.instances = []
    utils.ProgressBar = RecordingProgressBar
    model_management.throw_exception_if_processing_interrupted = lambda: interrupt_checks.append(True)
    comfy.utils = utils
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    return interrupt_checks


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replaces tag_images with a stub that returns canned tags and calls the callback per image."""
    calls = []

    def tag_images(clip, images, prompt, on_image_done=None):
        tags = [f"tag{index}" for index in range(images.shape[0])]

        calls.append((clip, images, prompt))
        for index in range(images.shape[0]):
            if on_image_done is not None:
                on_image_done(index)
        return tags

    monkeypatch.setattr(tagging, "tag_images", tag_images)
    return calls


# --- input and output contract --------------------------------------------------------


def test_required_inputs_in_order_with_types():
    required = LogitTagger.INPUT_TYPES()["required"]

    assert list(required) == ["clip", "image", "prompt"]
    assert [required[name][0] for name in required] == ["CLIP", "IMAGE", "STRING"]
    assert list(LogitTagger.INPUT_TYPES()) == ["required"]


def test_prompt_is_a_multiline_string_with_an_empty_default():
    options = LogitTagger.INPUT_TYPES()["required"]["prompt"][1]

    assert options["multiline"] is True
    assert options["default"] == ""


def test_every_input_has_a_tooltip():
    required = LogitTagger.INPUT_TYPES()["required"]

    for name, definition in required.items():
        assert definition[1].get("tooltip", "").strip(), name


def test_prompt_tooltip_says_optional_vocabulary_and_image_check():
    tooltip = LogitTagger.INPUT_TYPES()["required"]["prompt"][1]["tooltip"].lower()

    assert "optional" in tooltip
    assert "vocabulary" in tooltip
    assert "checked against the image" in tooltip


def test_output_contract():
    assert LogitTagger.RETURN_TYPES == ("STRING",)
    assert LogitTagger.RETURN_NAMES == ("tags",)
    assert LogitTagger.OUTPUT_IS_LIST == (True,)
    assert LogitTagger.OUTPUT_NODE is True
    assert LogitTagger.FUNCTION == "tag"
    assert LogitTagger.CATEGORY == "image/tagging"
    assert LogitTagger.DESCRIPTION.strip()


def test_user_facing_text_has_no_em_dash_or_semicolon():
    texts = [LogitTagger.DESCRIPTION]
    texts += [definition[1]["tooltip"] for definition in LogitTagger.INPUT_TYPES()["required"].values()]

    for text in texts:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- ui and result shape --------------------------------------------------------------


def test_batch_of_one_shows_the_tags_alone(fake_comfy, stub_pipeline):
    clip = object()
    images = an_image(batch=1)

    output = getattr(LogitTagger(), LogitTagger.FUNCTION)(clip, images, "a cat")
    assert output == {"ui": {"text": ("tag0",)}, "result": (["tag0"],)}
    assert len(stub_pipeline) == 1
    assert stub_pipeline[0][0] is clip
    assert stub_pipeline[0][1] is images
    assert stub_pipeline[0][2] == "a cat"


def test_batch_of_three_shows_one_indexed_line_per_image(fake_comfy, stub_pipeline):
    output = LogitTagger().tag(object(), an_image(batch=3), "")

    assert output["result"] == (["tag0", "tag1", "tag2"],)
    assert output["ui"] == {"text": ("0: tag0\n1: tag1\n2: tag2",)}
    assert len(stub_pipeline) == 1


def test_display_keeps_an_image_with_no_tags_on_its_own_line():
    assert node_module.format_display(["cat", "", "dog"]) == "0: cat\n1: \n2: dog"
    assert node_module.format_display([""]) == ""


# --- progress and interrupt wiring ----------------------------------------------------


def test_callback_advances_a_batch_sized_bar_and_checks_for_interrupt(fake_comfy, stub_pipeline):
    LogitTagger().tag(object(), an_image(batch=3), "")

    assert len(RecordingProgressBar.instances) == 1
    assert RecordingProgressBar.instances[0].total == 3
    assert RecordingProgressBar.instances[0].updates == [1, 1, 1]
    assert fake_comfy == [True, True, True]


def test_an_interrupt_stops_the_run(fake_comfy, stub_pipeline, monkeypatch):
    class Interrupted(Exception):
        pass

    def interrupt():
        raise Interrupted

    monkeypatch.setattr(sys.modules["comfy.model_management"], "throw_exception_if_processing_interrupted",
                        interrupt)
    with pytest.raises(Interrupted):
        LogitTagger().tag(object(), an_image(batch=3), "")


# --- registration ---------------------------------------------------------------------


def test_loads_via_comfyui_directory_mechanism():
    # Replicates core's loader: a spec from the pack's __init__.py, registered in sys.modules
    # before exec_module so the relative import resolves.
    module_name = "ComfyUI-LogitTagger_x_test"
    spec = importlib.util.spec_from_file_location(module_name, PACK_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)

        node_class = module.NODE_CLASS_MAPPINGS["LogitTagger"]
        assert isinstance(node_class, type)
        assert node_class.__name__ == "LogitTagger"
        assert list(module.NODE_CLASS_MAPPINGS) == ["LogitTagger"]
        assert module.NODE_DISPLAY_NAME_MAPPINGS == {"LogitTagger": "Logit Tagger"}
        assert module.WEB_DIRECTORY == "./web"
        assert set(module.__all__) == {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"}
    finally:
        del sys.modules[module_name]


def test_web_directory_holds_the_extension():
    assert (PACK_ROOT / "web" / "logit_tagger.js").is_file()


def test_node_module_never_imports_comfy():
    # A subprocess pins the constraint independent of test order or prior imports.
    code = (
        "import sys\n"
        "import logit_tagger.node\n"
        "assert 'comfy' not in sys.modules, 'node.py imported comfy at module scope'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PACK_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
