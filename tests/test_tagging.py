"""tagging.py: the Logit Tagger's settings and its runs through the toolkit's tagger, tag list check and prompt listing.

A duck typed CLIP stands in for the Qwen3-VL text encoder and a fake classifier stands in
for the library's Classifier, so no comfy install, no GPU and no model are needed.
"""

import dataclasses
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from logit_tagger import engine, tagging

PACK_ROOT = Path(__file__).parent.parent.resolve()


def settings():
    return tagging.tag_settings()


class FakeClip:
    """Records every tokenize and generate call, and decodes to the image reply or the prompt reply."""

    def __init__(self, reply="cat, dog", id_count=3, prompt_reply=""):
        self.reply = reply
        self.prompt_reply = prompt_reply
        self.id_count = id_count
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        # Core's tokenizer writes an image as a dict entry ahead of the text.
        image_entries = [({"type": "image", "data": image}, 1.0) for image in kwargs.get("images", [])]
        return {"qwen3vl_4b": [[*image_entries, (text, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return list(range(self.id_count))

    def decode(self, ids):
        _text, kwargs = self.tokenize_calls[-1]
        return self.reply if "images" in kwargs else self.prompt_reply

    def text_only_calls(self):
        return [text for text, kwargs in self.tokenize_calls if "images" not in kwargs]


class FakeClassifier:
    """Answers each presence noul from `probabilities` and each thing check's p("thing") from `thing`, both by tag.

    The language question gets `english`, so a prompt reads as English unless a test says otherwise.
    nouls is the library's own, so its requests reach classify as they would on a real classifier.
    """

    def __init__(self, probabilities=None, thing=None, english=1.0):
        self.probabilities = probabilities or {}
        self.thing = thing or {}
        self.english = english
        self.calls = []

    def classify(self, request, *, image=None):
        from logit_classifier.schema import ChoiceAnswer, NoulAnswer

        verify_prefix = settings().verify_statement.split("{TAG}")[0]
        answers = {}

        self.calls.append((request, image))
        for qid, question in request.questions.items():
            if question.type == "choice":
                p_thing = self.thing.get(question.instructions.split('"')[1], 1.0)
                answers[qid] = ChoiceAnswer(choice="thing" if p_thing >= 0.5 else "other", confidence=1.0,
                                            probabilities={"thing": p_thing, "other": 1.0 - p_thing})
            elif question.instructions == settings().language_question:
                answers[qid] = NoulAnswer(noul=self.english)
            else:
                tag = question.instructions.removeprefix(verify_prefix)
                answers[qid] = NoulAnswer(noul=self.probabilities.get(tag, 0.0))
        return SimpleNamespace(answers=answers), None

    def nouls(self, statements, *, image=None, state=""):
        from logit_classifier import Classifier

        return Classifier.nouls(self, statements, image=image, state=state)

    def calls_of(self, kind):
        return [(request, image) for request, image in self.calls
                if all(question.type == kind for question in request.questions.values())]


class ScriptedClip(FakeClip):
    """Decodes each generate to the next scripted reply of its kind, image or text only."""

    def __init__(self, replies=(), prompt_replies=()):
        super().__init__()
        self.replies = list(replies)
        self.prompt_replies = list(prompt_replies)

    def decode(self, ids):
        _text, kwargs = self.tokenize_calls[-1]
        return (self.replies if "images" in kwargs else self.prompt_replies).pop(0)


@pytest.fixture
def fake_classifier(monkeypatch, resident_loads):
    """Stands in for engine.build_classifier and for the toolkit's shared_vision_encode and resident load scopes."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    toolkit_tagger = pytest.importorskip("logit_classifier.toolkit.comfyui.tagger")
    classifier = FakeClassifier()
    built = []
    scopes = []

    def build(clip, node):
        built.append((clip, node))
        return classifier

    @contextmanager
    def shared_vision_encode(clip):
        scopes.append({"clip": clip, "open": True})
        try:
            yield
        finally:
            scopes[-1]["open"] = False

    monkeypatch.setattr(engine, "build_classifier", build)
    # The tag list check reads the scope from the package and the tagger from its own module.
    monkeypatch.setattr(toolkit, "shared_vision_encode", shared_vision_encode)
    monkeypatch.setattr(toolkit_tagger, "shared_vision_encode", shared_vision_encode)
    classifier.built = built
    classifier.scopes = scopes
    classifier.resident_loads = resident_loads
    return classifier


def an_image(height=8, width=8, channels=3, batch=1):
    torch = pytest.importorskip("torch")
    return torch.rand(batch, height, width, channels)


def tag_one(clip, classifier, **kwargs):
    return tagging.tag_image(clip, classifier, an_image(), thing_cache={}, **kwargs)


# --- module scope ---------------------------------------------------------------------


def test_module_scope_imports_no_comfy_and_no_library():
    script = (
        "import sys\n"
        "import logit_tagger.tagging\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'logit_classifier')]\n"
        "print(loaded)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], cwd=PACK_ROOT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- tag_settings ---------------------------------------------------------------------


def test_tag_settings_are_the_toolkit_defaults():
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")

    pack_settings = tagging.tag_settings()
    assert pack_settings == toolkit.TagSettings()
    assert pack_settings.order == "score"
    assert tagging.tag_settings() is pack_settings


def test_strict_tag_settings_hold_the_toolkit_strict_thresholds():
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    from logit_classifier.toolkit.tags import STRICT_THRESHOLDS

    strict_settings = tagging.tag_settings(strict=True)
    assert strict_settings == toolkit.TagSettings(verify_thresholds=STRICT_THRESHOLDS)
    assert tagging.tag_settings(True) is strict_settings
    assert tagging.tag_settings(False) is tagging.tag_settings()


def test_a_library_older_than_the_gate_fails_cached_tag_settings_with_the_version_message(monkeypatch):
    library = pytest.importorskip("logit_classifier")
    tagging.tag_settings()
    monkeypatch.setattr(library, "__version__", "0.5.0")

    with pytest.raises(RuntimeError, match=r'^Logit Tagger: .*0\.6\.0 or newer, and version 0\.5\.0 is installed'):
        tagging.tag_settings()


def test_tag_settings_without_the_library_names_the_node_that_asked(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)

    with pytest.raises(RuntimeError, match=r'^Logit Prompt Check: .*pip install -U "logit-classifier>=0\.6\.0"'):
        tagging.tag_settings(node=tagging.PROMPT_CHECK_NAME)


def test_check_tag_list_asks_in_the_settings_verify_statement(fake_classifier):
    settings = dataclasses.replace(tagging.tag_settings(), verify_statement="Seen: {TAG}")

    tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), ("cat",), settings)
    ((request, _image),) = fake_classifier.calls_of("noul")
    assert [q.instructions for q in request.questions.values()] == ["Seen: cat"]


def test_tag_settings_without_the_library_prints_the_pip_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)

    with pytest.raises(RuntimeError, match=r'^Logit Tagger: .*pip install -U "logit-classifier>=0\.6\.0"'):
        tagging.tag_settings()


# --- custom_settings ------------------------------------------------------------------


def test_custom_settings_with_the_default_wording_strict_off_and_things_only_on_are_the_tagger_settings():
    pytest.importorskip("logit_classifier.toolkit.comfyui")

    default_wording = tagging.tag_settings().propose_instruction
    assert tagging.custom_settings(default_wording, strict=False, things_only=True) == tagging.tag_settings()


@pytest.mark.parametrize("instruction", ["", " \n "])
def test_a_blank_instruction_proposes_with_the_default_wording_and_keeps_the_echo_fallback(instruction):
    pytest.importorskip("logit_classifier.toolkit.comfyui")

    custom = tagging.custom_settings(instruction, strict=False, things_only=True)
    assert custom.propose_instruction == tagging.tag_settings().propose_instruction
    assert custom.echo_fallback_instruction is not None
    assert custom.echo_fallback_instruction == tagging.tag_settings().echo_fallback_instruction


def test_another_instruction_proposes_with_its_own_wording_and_drops_the_echo_fallback():
    pytest.importorskip("logit_classifier.toolkit.comfyui")

    custom = tagging.custom_settings("List the colors in this image.", strict=False, things_only=True)
    assert custom.propose_instruction == "List the colors in this image."
    assert custom.echo_fallback_instruction is None


@pytest.mark.parametrize(("things_only", "thing_check"), [(True, "all"), (False, "prompt")])
def test_things_only_picks_the_thing_check(things_only, thing_check):
    pytest.importorskip("logit_classifier.toolkit.comfyui")

    assert tagging.custom_settings("Name the animals.", strict=False, things_only=things_only).thing_check == thing_check


@pytest.mark.parametrize("strict", [False, True])
def test_custom_settings_hold_the_strict_or_broad_thresholds(strict):
    pytest.importorskip("logit_classifier.toolkit.comfyui")
    from logit_classifier.toolkit.tags import BROAD_THRESHOLDS, STRICT_THRESHOLDS

    custom = tagging.custom_settings("Name the animals.", strict=strict, things_only=True)
    assert custom.verify_thresholds == (STRICT_THRESHOLDS if strict else BROAD_THRESHOLDS)


def test_custom_settings_without_the_library_name_the_custom_tagger(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)

    with pytest.raises(RuntimeError, match=r'^Logit Custom Tagger: .*pip install -U "logit-classifier>=0\.6\.0"'):
        tagging.custom_settings("", strict=False, things_only=True)


# --- tag_image ------------------------------------------------------------------------


def test_tag_image_reports_every_stage(fake_classifier, monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 100)
    fake_classifier.probabilities = {"cat": 0.9995, "dog": 0.3, "fox": 0.99}
    fake_classifier.thing = {"sunset": 0.05}

    trace = tagging.tag_image(FakeClip(reply="cat, objects, sunset, dog"), fake_classifier, an_image(40, 20),
                              prompt_tags=("dog", "fox"), thing_cache={})
    assert trace.size_in == (40, 20)
    assert trace.size_out == (14, 7)
    assert trace.reply == "cat, objects, sunset, dog"
    assert trace.echo_reply is None
    assert trace.proposed == ("cat", "objects", "sunset", "dog")
    assert trace.candidates == ("cat", "dog", "fox")
    assert trace.origins == ("model", "both", "prompt")
    assert trace.dropped == (("objects", "category noun"), ("sunset", "not a thing"))
    assert trace.scores == (0.9995, 0.3, 0.99)
    assert trace.kept == ("cat",)
    assert trace.tags == "cat"
    assert trace.propose_ms >= 0
    assert trace.verify_ms >= 0


def test_tag_image_verifies_every_candidate_on_the_fitted_image(fake_classifier):
    tag_one(FakeClip(reply="cat, sunset"), fake_classifier)
    ((request, image),) = fake_classifier.calls_of("noul")
    assert [q.instructions for q in request.questions.values()] == ["This image visibly contains cat",
                                                                     "This image visibly contains sunset"]
    assert request.state == ""
    assert tuple(image.shape) == (1, 8, 8, 3)


def test_tag_image_drops_a_model_tag_the_thing_check_calls_other(fake_classifier):
    fake_classifier.probabilities = {"watermark": 0.9995}
    fake_classifier.thing = {"watermark": 0.01}

    trace = tag_one(FakeClip(reply="watermark"), fake_classifier)
    assert trace.kept == ()
    assert trace.dropped == (("watermark", "not a thing"),)
    assert len(fake_classifier.calls_of("choice")) == 1


def test_tag_image_drops_a_tag_whose_words_all_sit_inside_a_longer_kept_tag(fake_classifier):
    fake_classifier.probabilities = dict.fromkeys(["hat", "red hat", "dog", "cat", "cat ear"], 0.9995)

    trace = tag_one(FakeClip(reply="hat, red hat, dog, cat, cat ear"), fake_classifier)
    assert trace.tags == "red hat, dog, cat ear"


def test_tag_image_drops_a_model_tag_below_the_proposed_threshold(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.999, "dog": 0.998}

    trace = tag_one(FakeClip(reply="cat, dog"), fake_classifier)
    assert trace.kept == ("cat",)


def test_tag_image_orders_by_probability_and_ties_keep_proposal_order(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9995, "dog": 0.9999, "fox": 0.9995, "owl": 0.99999}

    trace = tag_one(FakeClip(reply="cat, dog, fox, owl"), fake_classifier)
    assert trace.tags == "owl, dog, cat, fox"
    assert trace.kept == ("owl", "dog", "cat", "fox")


def test_tag_image_caps_the_proposed_tags(fake_classifier):
    cap = settings().propose_cap
    reply = ", ".join(f"item {index}" for index in range(cap + 5))

    trace = tag_one(FakeClip(reply=reply), fake_classifier)
    assert len(trace.proposed) == cap
    assert trace.proposed[-1] == f"item {cap - 1}"


def test_tag_image_skips_the_requests_when_nothing_parses(fake_classifier):
    trace = tag_one(FakeClip(reply="<think>hmm"), fake_classifier)

    assert trace.candidates == ()
    assert trace.tags == ""
    assert fake_classifier.calls == []


def test_tag_image_proposes_again_when_the_reply_echoes_only_category_nouns(fake_classifier):
    fake_classifier.probabilities = {"red car": 0.9995}
    clip = ScriptedClip(replies=["objects, people, animals", "red car, road"])

    trace = tag_one(clip, fake_classifier)
    assert trace.echo_reply == "objects, people, animals"
    assert trace.reply == "red car, road"
    assert "red car" in trace.candidates
    assert trace.tags == "red car"
    assert settings().echo_fallback_instruction in clip.tokenize_calls[1][0]


def test_settings_without_an_echo_fallback_never_propose_again(fake_classifier):
    clip = ScriptedClip(replies=["objects, people, animals", "red car, road"])
    custom = dataclasses.replace(settings(), propose_instruction="Name the things.", echo_fallback_instruction=None)

    trace = tag_one(clip, fake_classifier, settings=custom)
    assert len(clip.generate_calls) == 1
    assert trace.echo_reply is None
    assert trace.candidates == ()


def test_tag_image_hands_the_toolkit_the_given_settings_or_the_defaults(monkeypatch):
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    custom = dataclasses.replace(tagging.tag_settings(), propose_instruction="Name the animals.")
    received = []

    def tag_image(clip, classifier, image, *, prompt_tags=(), settings, thing_cache, on_stage=None):
        received.append(settings)
        return toolkit.ImageTagTrace(reply="", echo_reply=None, proposed=(), candidates=(), origins=(), dropped=(),
                                     scores=(), verified=(), kept=(), propose_ms=0.0, verify_ms=0.0)

    monkeypatch.setattr(toolkit, "tag_image", tag_image)
    tagging.tag_image("clip", "classifier", an_image(), thing_cache={}, settings=custom)
    tagging.tag_image("clip", "classifier", an_image(), thing_cache={})
    given, defaults = received
    assert given is custom
    assert defaults is tagging.tag_settings()


def test_tag_image_proposes_with_the_settings_instruction(fake_classifier):
    clip = FakeClip()
    custom = dataclasses.replace(settings(), propose_instruction="Name the animals.")

    tag_one(clip, fake_classifier, settings=custom)
    rendered = clip.tokenize_calls[0][0]
    assert "Name the animals." in rendered
    assert settings().propose_instruction not in rendered


def test_tag_image_calls_each_stage_before_its_model_call(fake_classifier):
    events = []

    class RecordingClip(FakeClip):
        def generate(self, tokens, **kwargs):
            events.append("generate")
            return super().generate(tokens, **kwargs)

    original_classify = fake_classifier.classify

    def classify(request, *, image=None):
        events.append(next(iter(request.questions.values())).type)
        return original_classify(request, image=image)

    fake_classifier.classify = classify
    tag_one(RecordingClip(reply="cat"), fake_classifier, on_stage=events.append)
    assert events == ["writing tags", "generate", "choice", "checking tags", "noul"]


def test_tag_image_proposes_and_verifies_inside_one_shared_vision_scope(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9995}
    seen = []

    class ScopeClip(FakeClip):
        def generate(self, tokens, **kwargs):
            seen.append(("propose", [scope["open"] for scope in fake_classifier.scopes]))
            return super().generate(tokens, **kwargs)

    original_classify = fake_classifier.classify

    def classify(request, *, image=None):
        seen.append(("check", [scope["open"] for scope in fake_classifier.scopes]))
        return original_classify(request, image=image)

    fake_classifier.classify = classify
    clip = ScopeClip(reply="cat")
    tag_one(clip, fake_classifier)
    assert seen == [("propose", [True]), ("check", [True]), ("check", [True])]
    assert [scope["clip"] for scope in fake_classifier.scopes] == [clip]
    assert fake_classifier.scopes[0]["open"] is False


def test_tag_image_rejects_a_batch(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match="Logit Tagger"):
        tagging.tag_image(clip, fake_classifier, an_image(batch=2), thing_cache={})
    assert clip.generate_calls == []


# --- tag_images -----------------------------------------------------------------------


def test_tag_images_runs_in_order_with_one_classifier(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9995, "dog": 0.999}
    clip = FakeClip(reply="cat, dog")
    images = [an_image(), an_image()]

    traces = tagging.tag_images(clip, images, ["", ""])
    sent = [image for _request, image in fake_classifier.calls_of("noul")]
    assert [trace.tags for trace in traces] == ["cat, dog", "cat, dog"]
    assert fake_classifier.built == [(clip, tagging.NODE_NAME)]
    assert [image.equal(images[index]) for index, image in enumerate(sent)] == [True, True]


def test_tag_images_reports_both_stages_per_image(fake_classifier):
    progress = []

    tagging.tag_images(FakeClip(reply="cat"), [an_image(), an_image()], ["", ""],
                       on_progress=lambda index, stage: progress.append((index, stage)))
    assert progress == [(0, "writing tags"), (0, "checking tags"), (1, "writing tags"), (1, "checking tags")]


def test_tag_images_reads_each_prompt_at_its_first_image(fake_classifier):
    progress = []

    tagging.tag_images(FakeClip(reply="cat", prompt_reply="fox"), [an_image(), an_image()], ["a fox", "a fox"],
                       on_progress=lambda index, stage: progress.append((index, stage)))
    assert progress == [(0, "reading prompt"), (0, "writing tags"), (0, "checking tags"), (1, "writing tags"),
                        (1, "checking tags")]


def test_tag_images_lists_each_distinct_prompt_once_and_asks_the_thing_check_once_per_tag(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9995, "fox": 0.99999}
    clip = FakeClip(reply="cat", prompt_reply="fox")

    traces = tagging.tag_images(clip, [an_image(), an_image(), an_image()], ["a fox", "a fox", " "])
    asked = [q.instructions for request, _image in fake_classifier.calls_of("choice")
             for q in request.questions.values()]
    assert len(clip.text_only_calls()) == 1
    assert asked == ['What does the image tag "fox" name', 'What does the image tag "cat" name']
    assert [trace.tags for trace in traces] == ["fox, cat", "fox, cat", "cat"]
    assert [trace.origins for trace in traces] == [("model", "prompt"), ("model", "prompt"), ("model",)]


def test_tag_images_drops_a_prompt_thing_the_thing_check_calls_other(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9995, "fox": 0.99999, "dusk": 0.99999}
    fake_classifier.thing = {"dusk": 0.05}
    clip = FakeClip(reply="cat", prompt_reply="fox, dusk")

    (trace,) = tagging.tag_images(clip, [an_image()], ["a fox at dusk"])
    assert trace.candidates == ("cat", "fox")
    assert trace.tags == "fox, cat"


def test_tag_images_passes_the_settings_to_the_prompt_listing_and_every_image_stage(fake_classifier):
    custom = dataclasses.replace(settings(), propose_cap=2, prompt_tags_max_tokens=7)
    clip = FakeClip(reply="cat, dog, owl", prompt_reply="fox")

    (trace,) = tagging.tag_images(clip, [an_image()], ["a fox"], settings=custom)
    (prompt_generate, image_generate) = [kwargs for _tokens, kwargs in clip.generate_calls]
    assert prompt_generate["max_length"] == 7
    assert image_generate["max_length"] == custom.propose_max_tokens
    assert trace.proposed == ("cat", "dog")


@pytest.mark.parametrize("run", ["tag_images", "check_tag_lists"])
def test_one_resident_load_skip_spans_every_model_call_of_the_run(fake_classifier, run):
    clip = FakeClip(reply="cat", prompt_reply="fox")
    resident = []
    classify = fake_classifier.classify

    def recording_classify(request, *, image=None):
        resident.append(fake_classifier.resident_loads.open)
        return classify(request, image=image)

    fake_classifier.classify = recording_classify
    if run == "tag_images":
        tagging.tag_images(clip, [an_image(), an_image()], ["a fox", "a fox"])
    else:
        tagging.check_tag_lists(clip, [an_image(), an_image()], [("cat",), ("cat",)], settings=settings())
    assert fake_classifier.resident_loads.clips == [clip]
    assert resident
    assert all(resident)
    assert fake_classifier.resident_loads.open is False


def test_tag_images_rejects_a_prompt_count_mismatch(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match="Logit Tagger: got 2 images and 1 prompts"):
        tagging.tag_images(clip, [an_image(), an_image()], [""])
    assert fake_classifier.built == []


def test_tag_images_rejects_an_image_that_is_not_4d(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match="Logit Tagger"):
        tagging.tag_images(clip, [an_image()[0]], [""])
    assert fake_classifier.built == []
    assert clip.generate_calls == []


def test_tag_images_rejects_an_image_with_fewer_than_3_channels(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match=tagging.NODE_NAME):
        tagging.tag_images(clip, [an_image(channels=1)], [""])
    assert clip.generate_calls == []


def test_tag_images_names_the_given_node_in_its_errors_and_its_classifier(fake_classifier):
    clip = FakeClip(reply="cat")

    with pytest.raises(ValueError, match=r"^Logit Custom Tagger: got 2 images and 1 prompts"):
        tagging.tag_images(clip, [an_image(), an_image()], [""], node=tagging.CUSTOM_TAGGER_NAME)
    with pytest.raises(ValueError, match=r"^Logit Custom Tagger: the image input must be"):
        tagging.tag_images(clip, [an_image()[0]], [""], node=tagging.CUSTOM_TAGGER_NAME)
    assert fake_classifier.built == []
    tagging.tag_images(clip, [an_image()], [""], node=tagging.CUSTOM_TAGGER_NAME)
    assert fake_classifier.built == [(clip, tagging.CUSTOM_TAGGER_NAME)]


def test_custom_settings_with_things_only_off_run_no_thing_check_without_a_prompt(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9995, "watercolor": 0.9995}
    fake_classifier.thing = {"watercolor": 0.05}
    clip = FakeClip(reply="cat, watercolor")
    custom = tagging.custom_settings("Tag this image with its things and its style.", strict=False, things_only=False)

    (trace,) = tagging.tag_images(clip, [an_image()], [""], settings=custom, node=tagging.CUSTOM_TAGGER_NAME)
    assert fake_classifier.calls_of("choice") == []
    assert "Tag this image with its things and its style." in clip.tokenize_calls[0][0]
    assert trace.candidates == ("cat", "watercolor")
    assert set(trace.kept) == {"cat", "watercolor"}


# --- tag list -------------------------------------------------------------------------


def listed(**probabilities):
    """The fake classifier's noul table, keyed by tag, since a listed tag is asked in the verify wording."""
    return {tag.replace("_", " "): p for tag, p in probabilities.items()}


def test_parse_tag_list_splits_on_commas_and_newlines_and_cleans_each_tag():
    pytest.importorskip("logit_classifier")

    assert tagging.parse_tag_list('Cat, "Red Car"\r\n- dog.\n\nbird') == ("cat", "red car", "dog", "bird")


def test_parse_tag_list_keeps_the_first_of_each_repeat_in_order():
    pytest.importorskip("logit_classifier")

    assert tagging.parse_tag_list("dog, Cat\ncat, DOG, fox") == ("dog", "cat", "fox")


@pytest.mark.parametrize("text", ["", " \n ", ", \n ,"])
def test_parse_tag_list_of_a_blank_list_is_empty(text):
    pytest.importorskip("logit_classifier")

    assert tagging.parse_tag_list(text) == ()


def test_parse_tag_list_with_a_library_older_than_the_gate_prints_the_pip_command(monkeypatch):
    library = pytest.importorskip("logit_classifier")
    monkeypatch.setattr(library, "__version__", "0.5.0")

    with pytest.raises(RuntimeError, match=r'^Logit Tagger: .*version 0\.5\.0 is installed.*pip install -U "logit-classifier>=0\.6\.0"'):
        tagging.parse_tag_list("cat")


def test_parse_tag_list_names_the_node_that_asked_when_the_library_is_old(monkeypatch):
    library = pytest.importorskip("logit_classifier")
    monkeypatch.setattr(library, "__version__", "0.5.0")

    with pytest.raises(RuntimeError, match=r"^Logit Tag Check: .*version 0\.5\.0 is installed"):
        tagging.parse_tag_list("cat", node=tagging.TAG_CHECK_NAME)


def test_parse_tag_list_has_no_cap():
    pytest.importorskip("logit_classifier")

    assert len(tagging.parse_tag_list(", ".join(f"tag {index}" for index in range(300)))) == 300


def test_check_tag_list_keeps_tags_at_or_above_the_threshold(fake_classifier):
    settings = tagging.tag_settings()
    threshold = settings.verify_thresholds.declared
    fake_classifier.probabilities = listed(cat=threshold, dog=threshold - 1e-6)

    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), ("cat", "dog"), settings)
    assert trace.kept == ("cat",)
    assert trace.scores == (threshold, threshold - 1e-6)


def test_check_tag_list_orders_by_probability_and_ties_keep_list_order(fake_classifier):
    fake_classifier.probabilities = listed(cat=0.99991, dog=0.99999, fox=0.99991, owl=0.5)

    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), ("cat", "dog", "fox", "owl"),
                                   tagging.tag_settings())
    assert trace.tags == "dog, cat, fox"


def test_check_tag_list_reports_the_listed_tags_as_its_only_candidates(fake_classifier):
    fake_classifier.probabilities = listed(cat=1.0)

    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(height=16), ("cat", "red car"),
                                   tagging.tag_settings())
    assert (trace.reply, trace.proposed, trace.dropped, trace.propose_ms) == ("", (), (), 0.0)
    assert trace.candidates == ("cat", "red car")
    assert trace.origins == ("list", "list")
    assert (trace.size_in, trace.size_out) == ((16, 8), (16, 8))


def test_check_tag_list_asks_the_presence_statement_about_the_fitted_image(fake_classifier, monkeypatch):
    monkeypatch.setattr(engine, "CHECK_MAX_PIXELS", 16)
    clip = FakeClip()

    tagging.check_tag_list(clip, fake_classifier, an_image(), ("cat", "red car"), tagging.tag_settings())
    ((request, image),) = fake_classifier.calls_of("noul")
    assert [q.instructions for q in request.questions.values()] == [
        "This image visibly contains cat", "This image visibly contains red car"]
    assert request.state == ""
    assert tuple(image.shape) == (1, 4, 4, 3)
    assert clip.generate_calls == []


def test_check_tag_list_asks_300_tags_inside_one_scope(fake_classifier):
    tags = tuple(f"tag {index}" for index in range(300))
    open_during: list[list[bool]] = []
    original_classify = fake_classifier.classify

    def classify(request, *, image=None):
        open_during.append([scope["open"] for scope in fake_classifier.scopes])
        return original_classify(request, image=image)

    fake_classifier.probabilities = listed(tag_0=1.0, tag_299=1.0)
    fake_classifier.classify = classify
    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), tags, tagging.tag_settings())
    assert [len(request.questions) for request, _image in fake_classifier.calls] == [256, 44]
    assert open_during == [[True], [True]]
    assert fake_classifier.scopes[0]["open"] is False
    assert len(trace.scores) == 300
    assert trace.kept == ("tag 0", "tag 299")


def test_check_tag_lists_checks_each_image_against_its_own_list_and_never_generates(fake_classifier):
    fake_classifier.probabilities = listed(cat=1.0, dog=1.0, owl=0.5)
    clip = FakeClip(reply="fox", prompt_reply="fox")
    progress = []

    traces = tagging.check_tag_lists(clip, [an_image(), an_image()], [("cat", "owl"), ("dog", "cat")],
                                     settings=settings(),
                                     on_progress=lambda index, stage: progress.append((index, stage)))
    asked = [[q.instructions for q in request.questions.values()] for request, _image in fake_classifier.calls]
    assert [trace.tags for trace in traces] == ["cat", "dog, cat"]
    assert [trace.candidates for trace in traces] == [("cat", "owl"), ("dog", "cat")]
    assert asked == [["This image visibly contains cat", "This image visibly contains owl"],
                     ["This image visibly contains dog", "This image visibly contains cat"]]
    assert progress == [(0, "checking tags"), (1, "checking tags")]
    assert clip.generate_calls == []
    assert fake_classifier.calls_of("choice") == []
    assert fake_classifier.built == [(clip, tagging.TAG_CHECK_NAME)]


def test_check_tag_lists_gives_an_empty_list_an_empty_tag_string_and_no_request(fake_classifier):
    fake_classifier.probabilities = listed(cat=1.0)

    traces = tagging.check_tag_lists(FakeClip(), [an_image(), an_image()], [(), ("cat",)], settings=settings())
    assert [trace.tags for trace in traces] == ["", "cat"]
    assert traces[0].scores == ()
    assert len(fake_classifier.calls) == 1


@pytest.mark.parametrize(("strict", "kept"), [(False, "cat, dog"), (True, "cat")])
def test_check_tag_lists_keeps_a_listed_tag_at_the_declared_threshold(fake_classifier, strict, kept):
    # fox clears the broad proposed threshold and misses the broad declared threshold.
    fake_classifier.probabilities = listed(cat=1.0, dog=0.99995, fox=0.9995)

    (trace,) = tagging.check_tag_lists(FakeClip(), [an_image()], [("cat", "dog", "fox")],
                                       settings=tagging.tag_settings(strict))
    assert trace.tags == kept


def test_check_tag_lists_rejects_a_tag_list_count_mismatch_before_building_the_classifier(fake_classifier):
    with pytest.raises(ValueError, match=r"^Logit Tag Check: got 2 images and 1 tag lists"):
        tagging.check_tag_lists(FakeClip(), [an_image(), an_image()], [("cat",)], settings=settings())
    assert fake_classifier.built == []


def test_check_tag_lists_rejects_a_batch_under_its_own_name(fake_classifier):
    with pytest.raises(ValueError, match=r"^Logit Tag Check: each image must be one image"):
        tagging.check_tag_lists(FakeClip(), [an_image(batch=2)], [("cat",)], settings=settings())
    assert fake_classifier.built == []


# --- positions ------------------------------------------------------------------------


@pytest.fixture
def located(monkeypatch):
    """Replaces the toolkit's locate_tags with a recorder that places cat top-left, gives dog no term and owl no strip."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    calls = []

    def locate_tags(classifier, image, tags, *, statement):
        terms = tuple("top-left" if tag == "cat" else "" for tag in tags)
        unplaced = tuple(tag for tag in tags if tag == "owl")
        placed = tuple((tag, term) for tag, term in zip(tags, terms, strict=True) if tag not in unplaced)

        calls.append(SimpleNamespace(classifier=classifier, image=image, tags=tuple(tags), statement=statement))
        return toolkit.TagPlaces(strips=(), terms=terms, unplaced=unplaced, placed=placed)

    monkeypatch.setattr(toolkit, "locate_tags", locate_tags)
    return calls


def three_kept(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.99999, "dog": 0.99995, "owl": 0.9995}
    return FakeClip(reply="cat, dog, owl")


def test_positions_off_makes_no_locate_call(fake_classifier, located):
    (tagged,) = tagging.tag_images(three_kept(fake_classifier), [an_image()], [""])
    (checked,) = tagging.check_tag_lists(FakeClip(), [an_image()], [("cat", "dog", "owl")], settings=settings())

    assert located == []
    assert (tagged.tags, tagged.terms, tagged.unplaced) == ("cat, dog, owl", (), ())
    assert checked.terms == ()


def test_tag_images_with_positions_places_the_kept_tags_on_the_full_resolution_image(fake_classifier, located,
                                                                                   monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 16)
    image = an_image()

    (trace,) = tagging.tag_images(three_kept(fake_classifier), [image], [""], positions=True)
    (call,) = located
    assert call.image is image
    assert call.classifier is fake_classifier
    assert call.tags == ("cat", "dog", "owl")
    assert call.statement == settings().verify_statement
    assert trace.size_out == (4, 4)
    assert trace.kept == ("cat", "dog", "owl")
    assert (trace.terms, trace.unplaced) == (("top-left", "", ""), ("owl",))
    assert trace.tags == "cat top-left, dog"


def test_check_tag_lists_with_positions_places_the_kept_tags_on_the_full_resolution_image(fake_classifier, located,
                                                                                        monkeypatch):
    monkeypatch.setattr(engine, "CHECK_MAX_PIXELS", 16)
    fake_classifier.probabilities = listed(cat=1.0, dog=1.0, owl=1.0, fox=0.5)
    strict = tagging.tag_settings(True)
    image = an_image()

    (trace,) = tagging.check_tag_lists(FakeClip(), [image], [("cat", "fox", "dog", "owl")], settings=strict,
                                       positions=True)
    (call,) = located
    assert call.image is image
    assert call.tags == ("cat", "dog", "owl")
    assert call.statement == strict.verify_statement
    assert trace.size_out == (4, 4)
    assert trace.tags == "cat top-left, dog"
    assert trace.unplaced == ("owl",)


@pytest.mark.parametrize("run", ["tag_images", "check_tag_lists"])
def test_an_image_with_no_kept_tags_makes_no_locate_call(fake_classifier, located, run):
    if run == "tag_images":
        (trace,) = tagging.tag_images(FakeClip(reply="cat"), [an_image()], [""], positions=True)
    else:
        (trace,) = tagging.check_tag_lists(FakeClip(), [an_image()], [("cat",)], settings=settings(), positions=True)

    assert located == []
    assert (trace.tags, trace.terms, trace.unplaced) == ("", (), ())


def test_the_status_names_the_placing_stage_after_each_image_stages(fake_classifier, located):
    tagged = []
    checked = []

    tagging.tag_images(three_kept(fake_classifier), [an_image(), an_image()], ["", ""], positions=True,
                       on_progress=lambda index, stage: tagged.append((index, stage)))
    tagging.check_tag_lists(FakeClip(), [an_image()], [("cat",)], settings=settings(), positions=True,
                            on_progress=lambda index, stage: checked.append((index, stage)))
    assert tagged == [(0, "writing tags"), (0, "checking tags"), (0, "placing tags"),
                      (1, "writing tags"), (1, "checking tags"), (1, "placing tags")]
    assert checked == [(0, "checking tags"), (0, "placing tags")]


@pytest.mark.parametrize(("height", "width"), [(2, 8), (8, 2)])
@pytest.mark.parametrize("run", ["tag_images", "check_tag_lists"])
def test_positions_on_an_image_under_3_pixels_fails_under_the_node_before_any_request(fake_classifier, located, run,
                                                                                     height, width):
    image = an_image(height=height, width=width)
    clip = FakeClip()
    node = tagging.NODE_NAME if run == "tag_images" else tagging.TAG_CHECK_NAME

    with pytest.raises(ValueError, match=rf"^{node}: positions needs an image at least 3 pixels on each side, got "
                                         rf"height {height} and width {width}\. Turn positions off or connect a "
                                         r"larger image\.$"):
        if run == "tag_images":
            tagging.tag_images(clip, [an_image(), image], ["", ""], positions=True)
        else:
            tagging.check_tag_lists(clip, [an_image(), image], [("cat",), ("cat",)], settings=settings(),
                                    positions=True)
    assert fake_classifier.built == []
    assert fake_classifier.calls == []
    assert clip.generate_calls == []


def test_tag_images_names_its_node_when_positions_reject_an_image(fake_classifier, located):
    with pytest.raises(ValueError, match=r"^Logit Custom Tagger: positions needs an image"):
        tagging.tag_images(FakeClip(), [an_image(height=2)], [""], positions=True, node=tagging.CUSTOM_TAGGER_NAME)


# --- library loading ------------------------------------------------------------------


def test_a_clip_that_is_not_a_qwen_model_fails_before_any_generate():
    pytest.importorskip("logit_classifier.toolkit.comfyui")
    from logit_classifier import UnsupportedModelError

    clip = FakeClip()

    with pytest.raises(UnsupportedModelError, match=r"^Logit Tagger: this CLIP is not a Qwen3-VL text encoder"):
        tagging.tag_images(clip, [an_image()], ["a fox"])
    assert clip.generate_calls == []


def test_a_clip_without_a_vision_tower_fails_before_any_generate(monkeypatch):
    toolkit_classifier = pytest.importorskip("logit_classifier.toolkit.comfyui.classifier")
    from logit_classifier import VisionUnsupportedError

    clip = FakeClip()

    class BlindBackend:
        sees_images = False

        def __init__(self, clip, **_kwargs):
            self.clip = clip

    monkeypatch.setattr(toolkit_classifier, "ComfyClipBackend", BlindBackend)
    with pytest.raises(VisionUnsupportedError, match=r"^Logit Tagger: .*vision tower"):
        tagging.tag_images(clip, [an_image()], ["a fox"])
    assert clip.generate_calls == []
