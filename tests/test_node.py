"""nodes.py: the Logit Tagger node's V1 contract, its list inputs, its ui/result shape and its registration.

The tagging pipeline, comfy and the server are stubbed, so no comfy install, no GPU and no model are needed.
"""

import importlib.util
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from logit_tagger import nodes, progress, tagging
from logit_tagger.nodes import LogitTagger

PACK_ROOT = Path(__file__).parent.parent.resolve()
NODE_ID = "12"


def an_image(batch=1, value=0.0):
    return torch.full((batch, 8, 8, 3), value)


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def run_node(image, clip="clip", **optional):
    """The node called the way core calls an INPUT_IS_LIST node, with every input a list."""
    return LogitTagger().tag(clip=[clip], image=image, unique_id=[NODE_ID], **optional)


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
    """Fake comfy.utils, comfy.model_management and server modules, and the records they keep."""
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
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
    """Replaces tag_images with a stub that returns one canned trace per image and reports the stages it runs."""
    calls = []

    def tag_images(clip, images, prompts, *, instruction=None, tag_list=(), on_progress=None, settings=None):
        stages = (tagging.STAGE_CHECKING,) if tag_list else (tagging.STAGE_WRITING, tagging.STAGE_CHECKING)
        calls.append(SimpleNamespace(clip=clip, images=images, prompts=prompts, instruction=instruction,
                                     tag_list=tag_list, settings=settings,
                                     bar_class=sys.modules["comfy.utils"].ProgressBar))
        for index in range(len(images)):
            for stage in stages:
                if on_progress is not None:
                    on_progress(index, stage)
        return [SimpleNamespace(tags=f"tag{index}") for index in range(len(images))]

    monkeypatch.setattr(tagging, "tag_images", tag_images)
    return calls


# --- input and output contract --------------------------------------------------------


def test_inputs_are_clip_and_image_then_three_text_sockets_the_strict_switch_and_the_node_id():
    spec = LogitTagger.INPUT_TYPES()

    assert list(spec) == ["required", "optional", "hidden"]
    assert [(name, definition[0]) for name, definition in spec["required"].items()] == [
        ("clip", "CLIP"), ("image", "IMAGE")]
    assert list(spec["optional"]) == ["prompt", "instruction", "tags", "strict"]
    for name in ("prompt", "instruction", "tags"):
        assert spec["optional"][name][0] == "STRING", name
        assert spec["optional"][name][1]["forceInput"] is True, name
    assert spec["optional"]["strict"][0] == "BOOLEAN"
    assert spec["optional"]["strict"][1]["default"] is False
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_differing_strict_values_fail_with_one_value_asked(fake_comfy, stub_pipeline):
    pytest.importorskip("logit_classifier.toolkit.comfyui")

    with pytest.raises(ValueError, match=r"^Logit Tagger: the strict input received 2 different values"):
        run_node([an_image()], strict=[False, True])
    assert stub_pipeline == []


@pytest.mark.parametrize(("optional", "strict"), [({}, False), ({"strict": [False]}, False), ({"strict": [True]}, True)])
def test_the_strict_switch_picks_the_settings_the_run_uses(fake_comfy, stub_pipeline, optional, strict):
    pytest.importorskip("logit_classifier.toolkit.comfyui")

    run_node([an_image()], **optional)
    assert stub_pipeline[0].settings is tagging.tag_settings(strict)


def test_every_input_has_a_tooltip_of_one_or_two_sentences():
    spec = LogitTagger.INPUT_TYPES()

    for group in ("required", "optional"):
        for name, definition in spec[group].items():
            assert 1 <= len(sentences(definition[1].get("tooltip", ""))) <= 2, name


def test_output_contract():
    assert LogitTagger.RETURN_TYPES == ("STRING", "IMAGE")
    assert LogitTagger.RETURN_NAMES == ("tags", "images")
    assert LogitTagger.INPUT_IS_LIST is True
    assert LogitTagger.OUTPUT_IS_LIST == (True, True)
    assert LogitTagger.OUTPUT_NODE is True
    assert LogitTagger.FUNCTION == "tag"
    assert LogitTagger.CATEGORY == "Logit Tagger"
    assert 1 <= len(sentences(LogitTagger.DESCRIPTION)) <= 3


def test_user_facing_text_has_no_em_dash_or_semicolon():
    spec = LogitTagger.INPUT_TYPES()
    texts = [LogitTagger.DESCRIPTION]
    texts += [definition[1]["tooltip"] for group in ("required", "optional") for definition in spec[group].values()]

    for text in texts:
        assert chr(0x2014) not in text
        assert ";" not in text


# --- list inputs ----------------------------------------------------------------------


def test_one_prompt_applies_to_every_image(fake_comfy, stub_pipeline):
    run_node([an_image(batch=3)], prompt=["a cat"])

    assert stub_pipeline[0].prompts == ["a cat"] * 3


def test_no_prompt_gives_every_image_an_empty_prompt(fake_comfy, stub_pipeline):
    run_node([an_image(batch=2)])

    assert stub_pipeline[0].prompts == ["", ""]


def test_k_prompts_pair_with_k_images_after_the_batches_split(fake_comfy, stub_pipeline):
    run_node([an_image(batch=2), an_image()], prompt=["a cat", "a dog", "a fox"])

    assert stub_pipeline[0].prompts == ["a cat", "a dog", "a fox"]


def test_a_wrong_prompt_count_is_an_error_before_the_run_starts(fake_comfy, stub_pipeline):
    with pytest.raises(ValueError, match="Logit Tagger: the prompt input has 2 texts for 3 images"):
        run_node([an_image(batch=3)], prompt=["a cat", "a dog"])

    assert stub_pipeline == []
    assert RecordingProgressBar.instances == []


def test_conflicting_instructions_are_an_error_before_the_run_starts(fake_comfy, stub_pipeline):
    with pytest.raises(ValueError, match="Logit Tagger: the instruction input received 2 different values"):
        run_node([an_image()], instruction=["List the tags.", "Name the objects."])

    assert stub_pipeline == []
    assert RecordingProgressBar.instances == []


def test_a_connected_instruction_replaces_the_propose_instruction(fake_comfy, stub_pipeline):
    run_node([an_image(batch=2)], instruction=["List the tags.", "List the tags."])

    assert stub_pipeline[0].instruction == "List the tags."


@pytest.mark.parametrize("optional", [{}, {"instruction": [""]}, {"instruction": [" \n"]}])
def test_an_unconnected_or_blank_instruction_keeps_the_pipeline_instruction(fake_comfy, stub_pipeline, optional):
    run_node([an_image()], **optional)

    assert stub_pipeline[0].instruction is None


def test_connected_tags_reach_the_pipeline_as_the_parsed_tag_list(fake_comfy, stub_pipeline):
    pytest.importorskip("logit_classifier")

    run_node([an_image(batch=2)], prompt=[" "], tags=["Cat, dog\ncat", "Cat, dog\ncat"])

    assert stub_pipeline[0].tag_list == tagging.parse_tag_list("Cat, dog\ncat") == ("cat", "dog")


@pytest.mark.parametrize("optional", [{}, {"tags": [""]}, {"tags": [" \n"], "prompt": ["a cat"]}])
def test_unconnected_or_blank_tags_leave_the_pipeline_writing_tags(fake_comfy, stub_pipeline, optional):
    run_node([an_image()], **optional)

    assert stub_pipeline[0].tag_list == ()


def test_conflicting_tags_are_an_error_before_the_run_starts(fake_comfy, stub_pipeline):
    with pytest.raises(ValueError, match="Logit Tagger: the tags input received 2 different values"):
        run_node([an_image()], tags=["cat", "dog"])

    assert stub_pipeline == []
    assert RecordingProgressBar.instances == []


def test_tags_with_a_prompt_are_an_error_before_the_run_starts(fake_comfy, stub_pipeline):
    pytest.importorskip("logit_classifier")

    with pytest.raises(ValueError, match="Logit Tagger: the prompt only guides the tags the node writes itself"):
        run_node([an_image(batch=2)], prompt=["", "a cat"], tags=["cat"])

    assert stub_pipeline == []
    assert RecordingProgressBar.instances == []


def test_tags_with_an_instruction_are_an_error_before_the_run_starts(fake_comfy, stub_pipeline):
    pytest.importorskip("logit_classifier")

    with pytest.raises(ValueError, match="Logit Tagger: the instruction only guides the tags the node writes itself"):
        run_node([an_image()], instruction=["List the tags."], tags=["cat"])

    assert stub_pipeline == []
    assert RecordingProgressBar.instances == []


@pytest.mark.parametrize("image", [[], [torch.zeros(0, 8, 8, 3)]])
def test_no_images_is_an_error_before_the_run_starts(fake_comfy, stub_pipeline, image):
    with pytest.raises(ValueError, match="Logit Tagger: the image input holds no images"):
        run_node(image)

    assert stub_pipeline == []


def test_the_node_reads_the_clip_from_its_one_item_list(fake_comfy, stub_pipeline):
    clip = object()

    run_node([an_image()], clip=clip)

    assert stub_pipeline[0].clip is clip


# --- ui and result shape --------------------------------------------------------------


def test_a_batch_of_3_in_a_list_of_2_gives_4_aligned_outputs(fake_comfy, stub_pipeline):
    first = torch.stack([torch.full((8, 8, 3), value) for value in (0.1, 0.2, 0.3)])

    tags, images = run_node([first, an_image(value=0.4)])["result"]

    assert tags == ["tag0", "tag1", "tag2", "tag3"]
    assert [tuple(item.shape) for item in images] == [(1, 8, 8, 3)] * 4
    assert [round(float(item.mean()), 1) for item in images] == [0.1, 0.2, 0.3, 0.4]
    assert len(stub_pipeline[0].images) == 4
    assert all(out is tagged for out, tagged in zip(images, stub_pipeline[0].images, strict=True))


def test_one_image_shows_its_tags_alone(fake_comfy, stub_pipeline):
    image = an_image()

    output = run_node([image])

    assert set(output) == {"ui", "result"}
    assert output["ui"] == {"text": ("tag0",)}
    assert output["result"][0] == ["tag0"]
    assert torch.equal(output["result"][1][0], image)


def test_many_images_show_one_indexed_line_each(fake_comfy, stub_pipeline):
    output = run_node([an_image(batch=3)])

    assert output["ui"] == {"text": ("0: tag0\n1: tag1\n2: tag2",)}


def test_display_keeps_an_image_with_no_tags_on_its_own_line():
    assert nodes.format_display(["cat", "", "dog"]) == "0: cat\n1: \n2: dog"
    assert nodes.format_display([""]) == ""


# --- progress, status and interrupt wiring --------------------------------------------


def test_one_bar_closes_each_image_and_the_status_names_each_stage(fake_comfy, stub_pipeline):
    run_node([an_image(batch=3)])

    assert len(RecordingProgressBar.instances) == 1
    bar = RecordingProgressBar.instances[0]
    assert bar.total == 3 * progress.ITEM_UNITS
    assert [value for value, _total in bar.updates] == [0, 0, 0, 100, 100, 200, 200, 300]
    assert bar.updates[-1] == (300, 300)
    assert fake_comfy.interrupt_checks == 6
    assert fake_comfy.sent[:-1] == [
        ("image 1/3: writing tags", NODE_ID), ("image 1/3: checking tags", NODE_ID),
        ("image 2/3: writing tags", NODE_ID), ("image 2/3: checking tags", NODE_ID),
        ("image 3/3: writing tags", NODE_ID), ("image 3/3: checking tags", NODE_ID),
    ]
    assert re.fullmatch(r"3 images in \d+\.\d s", fake_comfy.sent[-1][0])


def test_a_tag_list_run_closes_each_image_and_the_status_names_the_check(fake_comfy, stub_pipeline):
    pytest.importorskip("logit_classifier")

    run_node([an_image(batch=3)], tags=["cat"])

    bar = RecordingProgressBar.instances[0]
    assert [value for value, _total in bar.updates] == [0, 0, 100, 200, 300]
    assert fake_comfy.sent[:-1] == [
        ("image 1/3: checking tags", NODE_ID), ("image 2/3: checking tags", NODE_ID),
        ("image 3/3: checking tags", NODE_ID),
    ]
    assert re.fullmatch(r"3 images in \d+\.\d s", fake_comfy.sent[-1][0])


def test_core_bars_route_into_the_run_only_while_the_pipeline_runs(fake_comfy, stub_pipeline):
    run_node([an_image()])

    assert stub_pipeline[0].bar_class is not RecordingProgressBar
    assert sys.modules["comfy.utils"].ProgressBar is RecordingProgressBar


def test_an_interrupt_stops_the_run_and_the_status_says_cancelled(fake_comfy, stub_pipeline):
    fake_comfy.interrupted = True

    with pytest.raises(Interrupted):
        run_node([an_image(batch=3)])

    assert fake_comfy.sent == [("cancelled", NODE_ID)]
    assert sys.modules["comfy.utils"].ProgressBar is RecordingProgressBar


def test_an_interrupt_during_the_prompt_generates_stops_the_run_before_the_first_image(fake_comfy, monkeypatch):
    tokens = []

    def tag_images(clip, images, prompts, *, instruction=None, tag_list=(), on_progress=None, settings=None):
        for prompt in dict.fromkeys(prompts):
            bar = sys.modules["comfy.utils"].ProgressBar(256)
            for _step in range(256):
                bar.update(1)
                tokens.append(prompt)
                fake_comfy.interrupted = True
        for index in range(len(images)):
            on_progress(index, tagging.STAGE_WRITING)
        return [SimpleNamespace(tags="") for _image in images]

    monkeypatch.setattr(tagging, "tag_images", tag_images)

    with pytest.raises(Interrupted):
        run_node([an_image(batch=3)], prompt=["cat", "dog", "fox"])

    assert tokens == ["cat"]
    assert fake_comfy.sent == [("cancelled", NODE_ID)]
    assert sys.modules["comfy.utils"].ProgressBar is RecordingProgressBar


# --- registration ---------------------------------------------------------------------


@pytest.fixture
def pack():
    # Replicates core's loader: a spec from the pack's __init__.py, registered in sys.modules
    # before exec_module so the relative import resolves.
    module_name = "ComfyUI-LogitTagger_x_test"
    spec = importlib.util.spec_from_file_location(module_name, PACK_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[module_name]


def test_loads_via_comfyui_directory_mechanism(pack):
    node_class = pack.NODE_CLASS_MAPPINGS["LogitTagger"]

    assert isinstance(node_class, type)
    assert node_class.__name__ == "LogitTagger"
    assert list(pack.NODE_CLASS_MAPPINGS) == ["LogitTagger", "LogitImageCheck", "LogitTextCheck", "LogitPromptCheck",
                                              "LogitImageClassify"]
    assert all(pack.NODE_CLASS_MAPPINGS[name].__name__ == name for name in pack.NODE_CLASS_MAPPINGS)
    assert pack.NODE_DISPLAY_NAME_MAPPINGS == {
        "LogitTagger": "Logit Tagger",
        "LogitImageCheck": "Logit Image Check",
        "LogitTextCheck": "Logit Text Check",
        "LogitPromptCheck": "Logit Prompt Check",
        "LogitImageClassify": "Logit Image Classify",
    }
    assert pack.WEB_DIRECTORY == "./web"
    assert set(pack.__all__) == {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"}


def test_every_node_has_one_output_tooltip_per_output_and_search_aliases(pack):
    # Core sends both to the frontend: output_tooltips pair with outputs by position, search_aliases feed node search.
    for name, node_class in pack.NODE_CLASS_MAPPINGS.items():
        assert len(node_class.OUTPUT_TOOLTIPS) == len(node_class.RETURN_TYPES), name
        assert isinstance(node_class.SEARCH_ALIASES, list) and node_class.SEARCH_ALIASES, name
        assert all(isinstance(alias, str) and alias.strip() for alias in node_class.SEARCH_ALIASES), name
        for text in node_class.OUTPUT_TOOLTIPS:
            assert 1 <= len(sentences(text)) <= 2, text
            assert chr(0x2014) not in text
            assert ";" not in text


def test_web_directory_holds_the_extension():
    assert (PACK_ROOT / "web" / "logit_tagger.js").is_file()


def test_the_extension_shows_the_text_of_every_output_node(pack):
    # Every output node here returns a ui text, and the frontend renders it only for the types the extension names.
    script = (PACK_ROOT / "web" / "logit_tagger.js").read_text(encoding="utf-8")
    declaration = re.search(r"const NODE_TYPES = new Set\(\[([^\]]*)\]\)", script)
    output_nodes = {name for name, node_class in pack.NODE_CLASS_MAPPINGS.items() if getattr(node_class, "OUTPUT_NODE", False)}

    assert declaration, "web/logit_tagger.js declares no NODE_TYPES set"
    assert set(re.findall(r'"([^"]+)"', declaration.group(1))) == output_nodes


def test_nodes_module_never_imports_comfy_torch_or_the_library():
    # A subprocess pins the constraint independent of test order or prior imports.
    code = (
        "import sys\n"
        "import logit_tagger.nodes\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'logit_classifier', 'server')]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PACK_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
