"""style.py: Logit Style Extractor's run, which takes each item's style from its prompt first and its image second.

The toolkit's prompt_style and style_line and the classifier build are faked, so no comfy install, no GPU and no
model are needed.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from logit_tagger import engine, style, tagging

PACK_ROOT = Path(__file__).parent.parent.resolve()
STYLED_PROMPT = "an oil painting of a cat, impressionist"
PLAIN_PROMPT = "a cat on a sofa"
IMAGE_LINE = "Oil painting with loose impressionist brushwork."


def an_image(value=0.0):
    return torch.full((1, 8, 8, 3), value)


@pytest.fixture
def fake_toolkit(monkeypatch, resident_loads):
    """Fakes the classifier build, prompt_style and style_line, and records every call with its scope state.

    prompt_style keeps the style words record.styles names for a prompt, and style_line writes record.line.
    """
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = SimpleNamespace(built=[], prompt_calls=[], line_calls=[], line=IMAGE_LINE,
                             styles={STYLED_PROMPT: ("oil painting", "impressionist")}, resident_loads=resident_loads)

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node, needs_images))
        return "classifier"

    def prompt_style(clip, classifier, prompt, *, settings, tag_settings, style_cache):
        words = record.styles.get(prompt, ())
        record.prompt_calls.append(SimpleNamespace(clip=clip, classifier=classifier, prompt=prompt, settings=settings,
                                                   tag_settings=tag_settings, style_cache=style_cache,
                                                   scoped=resident_loads.open))
        return toolkit.PromptStyleTrace(text=prompt, reply=", ".join(words), listed=words,
                                        p_style=(1.0,) * len(words), style=words)

    def style_line(clip, image, *, settings, node):
        record.line_calls.append(SimpleNamespace(clip=clip, image=image, settings=settings, node=node,
                                                 scoped=resident_loads.open))
        return record.line

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    monkeypatch.setattr(toolkit, "prompt_style", prompt_style)
    monkeypatch.setattr(toolkit, "style_line", style_line)
    return record


# --- module scope ---------------------------------------------------------------------


def test_module_scope_imports_no_comfy_and_no_library():
    script = (
        "import sys\n"
        "import logit_tagger.style\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'logit_classifier')]\n"
        "print(loaded)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], cwd=PACK_ROOT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- the prompt first rule ------------------------------------------------------------


def test_a_prompt_that_names_a_style_wins_over_the_image(fake_toolkit):
    traces = style.extract_styles("clip", [an_image()], [STYLED_PROMPT])

    assert [(trace.source, trace.style) for trace in traces] == [("prompt", "oil painting, impressionist")]
    assert traces[0].prompt_trace.style == ("oil painting", "impressionist")
    assert fake_toolkit.line_calls == []


def test_a_prompt_that_names_no_style_falls_back_to_the_image(fake_toolkit):
    image = an_image()

    traces = style.extract_styles("clip", [image], [PLAIN_PROMPT])

    assert [(trace.source, trace.style) for trace in traces] == [("image", IMAGE_LINE)]
    assert traces[0].prompt_trace.style == ()
    assert [call.prompt for call in fake_toolkit.prompt_calls] == [PLAIN_PROMPT]
    assert len(fake_toolkit.line_calls) == 1
    assert fake_toolkit.line_calls[0].image is image
    assert fake_toolkit.line_calls[0].node == style.STYLE_EXTRACTOR_NAME


@pytest.mark.parametrize("prompt", ["", "  \n"])
def test_a_blank_prompt_with_an_image_reads_the_image_and_lists_nothing(fake_toolkit, prompt):
    traces = style.extract_styles("clip", [an_image()], [prompt])

    assert [(trace.source, trace.style, trace.prompt_trace) for trace in traces] == [("image", IMAGE_LINE, None)]
    assert fake_toolkit.prompt_calls == []


def test_a_prompt_alone_that_names_no_style_gives_an_empty_style(fake_toolkit):
    traces = style.extract_styles("clip", [], [PLAIN_PROMPT, STYLED_PROMPT])

    assert [(trace.source, trace.style) for trace in traces] == [("none", ""), ("prompt", "oil painting, impressionist")]
    assert fake_toolkit.line_calls == []


def test_an_image_whose_style_line_is_empty_keeps_the_image_source(fake_toolkit):
    fake_toolkit.line = ""

    traces = style.extract_styles("clip", [an_image()], [""])

    assert [(trace.source, trace.style) for trace in traces] == [("image", "")]


def test_each_item_takes_its_own_source(fake_toolkit):
    images = [an_image(0.1), an_image(0.2), an_image(0.3)]

    traces = style.extract_styles("clip", images, [STYLED_PROMPT, PLAIN_PROMPT, ""])

    assert [trace.source for trace in traces] == ["prompt", "image", "image"]
    assert [call.image for call in fake_toolkit.line_calls] == images[1:]


# --- one listing per distinct prompt --------------------------------------------------


@pytest.mark.parametrize(("prompt", "line_count"), [(STYLED_PROMPT, 0), (PLAIN_PROMPT, 3)])
def test_one_distinct_prompt_over_three_images_runs_prompt_style_once(fake_toolkit, prompt, line_count):
    traces = style.extract_styles("clip", [an_image(), an_image(), an_image()], [prompt] * 3)

    assert [call.prompt for call in fake_toolkit.prompt_calls] == [prompt]
    assert len(fake_toolkit.line_calls) == line_count
    assert len({id(trace.prompt_trace) for trace in traces}) == 1


def test_one_style_cache_and_one_settings_serve_the_entire_run(fake_toolkit):
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")

    style.extract_styles("clip", [an_image(), an_image()], [STYLED_PROMPT, PLAIN_PROMPT])

    calls = fake_toolkit.prompt_calls
    assert calls[0].style_cache is calls[1].style_cache
    assert calls[0].settings is calls[1].settings is fake_toolkit.line_calls[0].settings
    assert calls[0].settings == toolkit.StyleSettings()
    assert calls[0].tag_settings is tagging.tag_settings()
    assert {(call.clip, call.classifier) for call in calls} == {("clip", "classifier")}


# --- classifier, scope and progress ---------------------------------------------------


@pytest.mark.parametrize(("images", "needs_images"), [([an_image()], True), ([], False)])
def test_the_classifier_is_built_once_and_needs_images_only_with_images(fake_toolkit, images, needs_images):
    style.extract_styles("clip", images, [PLAIN_PROMPT])

    assert fake_toolkit.built == [("clip", style.STYLE_EXTRACTOR_NAME, needs_images)]


def test_the_model_work_runs_inside_one_resident_load_scope(fake_toolkit):
    style.extract_styles("clip", [an_image(), an_image()], [STYLED_PROMPT, PLAIN_PROMPT])

    assert fake_toolkit.resident_loads.clips == ["clip"]
    assert all(call.scoped for call in fake_toolkit.prompt_calls + fake_toolkit.line_calls)
    assert fake_toolkit.resident_loads.open is False


def test_progress_hears_each_reading_stage_at_its_item(fake_toolkit):
    heard = []

    style.extract_styles("clip", [an_image(), an_image(), an_image()], [STYLED_PROMPT, PLAIN_PROMPT, PLAIN_PROMPT],
                         on_progress=lambda index, stage: heard.append((index, stage)))

    assert heard == [(0, "reading prompt"), (1, "reading prompt"), (1, "reading style"), (2, "reading style")]


def test_differing_image_and_prompt_counts_fail_before_the_classifier_is_built(fake_toolkit):
    with pytest.raises(ValueError, match=r"^Logit Style Extractor: got 2 images and 3 prompts"):
        style.extract_styles("clip", [an_image(), an_image()], [PLAIN_PROMPT] * 3)

    assert fake_toolkit.built == []


def test_a_tensor_that_is_not_an_image_batch_fails_under_the_node_name(fake_toolkit):
    with pytest.raises(ValueError, match=r"^Logit Style Extractor: the image input must be"):
        style.extract_styles("clip", [torch.zeros(8, 8, 3)], [PLAIN_PROMPT])

    assert fake_toolkit.built == []
