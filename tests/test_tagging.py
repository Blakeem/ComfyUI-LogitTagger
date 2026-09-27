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
    """Answers each verify noul from `probabilities` and each thing check from `other`, both by tag.

    The language question gets `english`, so a prompt reads as English unless a test says otherwise.
    nouls is the library's own, so its requests reach classify as they would on a real classifier.
    """

    def __init__(self, probabilities=None, other=None, english=1.0):
        self.probabilities = probabilities or {}
        self.other = other or {}
        self.english = english
        self.calls = []

    def classify(self, request, *, image=None):
        from logit_classifier.schema import ChoiceAnswer, NoulAnswer

        verify_prefix = settings().verify_statement.split("{TAG}")[0]
        answers = {}

        self.calls.append((request, image))
        for qid, question in request.questions.items():
            if question.type == "choice":
                p_other = self.other.get(question.instructions.split('"')[1], 0.0)
                answers[qid] = ChoiceAnswer(choice="other" if p_other >= 0.5 else "thing", confidence=1.0,
                                            probabilities={"thing": 1.0 - p_other, "other": p_other})
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
def fake_classifier(monkeypatch):
    """Stands in for engine.build_classifier and for the toolkit's shared_vision_encode scope."""
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
    return classifier


def an_image(height=8, width=8, channels=3, batch=1):
    torch = pytest.importorskip("torch")
    return torch.rand(batch, height, width, channels)


def tag_one(clip, classifier, **kwargs):
    return tagging.tag_image(clip, classifier, an_image(), known={}, **kwargs)


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


def test_tag_settings_are_the_toolkit_defaults_with_the_four_measured_overrides():
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")

    pack_settings = tagging.tag_settings()
    assert pack_settings.propose_cap == 40
    assert pack_settings.merge_cap == 80
    assert pack_settings.thing_check == "prompt"
    assert pack_settings.subsets_rule == "head-noun"
    assert pack_settings.order == "score"
    assert pack_settings == dataclasses.replace(toolkit.TagSettings(), propose_cap=40, merge_cap=80,
                                                thing_check="prompt", subsets_rule="head-noun", order="score")
    assert tagging.tag_settings() is pack_settings


def test_tag_settings_without_the_library_prints_the_pip_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)
    tagging.tag_settings.cache_clear()

    try:
        with pytest.raises(RuntimeError, match=r"^Logit Tagger: .*pip install -U logit-classifier"):
            tagging.tag_settings()
    finally:
        tagging.tag_settings.cache_clear()


# --- tag_image ------------------------------------------------------------------------


def test_tag_image_reports_every_stage(fake_classifier, monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 100)
    fake_classifier.probabilities = {"cat": 0.95, "dog": 0.3, "fox": 0.99}
    fake_classifier.other = {"sunset": 0.95}

    trace = tagging.tag_image(FakeClip(reply="cat, objects, sunset, dog"), fake_classifier, an_image(40, 20),
                              prompt_tags=("dog", "fox"), known={})
    assert trace.size_in == (40, 20)
    assert trace.size_out == (14, 7)
    assert trace.reply == "cat, objects, sunset, dog"
    assert trace.echo_reply is None
    assert trace.proposed == ("cat", "objects", "sunset", "dog")
    assert trace.candidates == ("cat", "sunset", "dog", "fox")
    assert trace.origins == ("model", "model", "both", "prompt")
    assert trace.dropped == (("objects", "category noun"),)
    assert trace.scores == (0.95, 0.0, 0.3, 0.99)
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


def test_tag_image_keeps_a_model_tag_the_thing_check_would_call_other(fake_classifier):
    fake_classifier.probabilities = {"watermark": 0.95}
    fake_classifier.other = {"watermark": 0.99}

    trace = tag_one(FakeClip(reply="watermark"), fake_classifier)
    assert trace.kept == ("watermark",)
    assert fake_classifier.calls_of("choice") == []


def test_tag_image_drops_a_subset_with_the_same_head_noun_and_keeps_a_whole_beside_its_part(fake_classifier):
    fake_classifier.probabilities = {"hat": 0.95, "red hat": 0.95, "dog": 0.95, "cat": 0.95, "cat ear": 0.95}

    trace = tag_one(FakeClip(reply="hat, red hat, dog, cat, cat ear"), fake_classifier)
    assert trace.tags == "red hat, dog, cat, cat ear"


def test_tag_image_orders_by_probability_and_ties_keep_proposal_order(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.95, "dog": 0.99, "fox": 0.95, "owl": 0.999}

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
    fake_classifier.probabilities = {"red car": 0.95}
    clip = ScriptedClip(replies=["objects, people, animals", "red car, road"])

    trace = tag_one(clip, fake_classifier)
    assert trace.echo_reply == "objects, people, animals"
    assert trace.reply == "red car, road"
    assert "red car" in trace.candidates
    assert trace.tags == "red car"
    assert settings().echo_fallback_instruction in clip.tokenize_calls[1][0]


def test_a_custom_instruction_never_triggers_the_echo_fallback(fake_classifier):
    clip = ScriptedClip(replies=["objects, people, animals", "red car, road"])

    trace = tag_one(clip, fake_classifier, instruction="Name the things.")
    assert len(clip.generate_calls) == 1
    assert trace.echo_reply is None
    assert trace.candidates == ()


def test_a_connected_instruction_reaches_tag_picture_with_the_echo_fallback_off(monkeypatch):
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    received = []

    def tag_picture(clip, classifier, picture, *, prompt_tags=(), settings, known, on_stage=None):
        received.append(settings)
        return toolkit.TagTrace(reply="", echo_reply=None, proposed=(), candidates=(), origins=(), dropped=(),
                                scores=(), verified=(), kept=(), propose_ms=0.0, verify_ms=0.0)

    monkeypatch.setattr(toolkit, "tag_picture", tag_picture)
    tagging.tag_image("clip", "classifier", an_image(), instruction="Name the animals.", known={})
    tagging.tag_image("clip", "classifier", an_image(), known={})
    connected, unconnected = received
    assert connected == dataclasses.replace(tagging.tag_settings(), propose_instruction="Name the animals.",
                                            echo_fallback_instruction=None)
    assert unconnected is tagging.tag_settings()


def test_tag_image_instruction_override(fake_classifier):
    clip = FakeClip()

    tag_one(clip, fake_classifier, instruction="Name the animals.")
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
    assert events == ["writing tags", "generate", "checking tags", "noul"]


def test_tag_image_proposes_and_verifies_inside_one_shared_vision_scope(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.95}
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
    assert seen == [("propose", [True]), ("check", [True])]
    assert [scope["clip"] for scope in fake_classifier.scopes] == [clip]
    assert fake_classifier.scopes[0]["open"] is False


def test_tag_image_rejects_a_batch(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match="Logit Tagger"):
        tagging.tag_image(clip, fake_classifier, an_image(batch=2), known={})
    assert clip.generate_calls == []


# --- tag_images -----------------------------------------------------------------------


def test_tag_images_runs_in_order_with_one_classifier(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.95, "dog": 0.9}
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


def test_tag_images_lists_each_distinct_prompt_once_and_checks_only_prompt_things(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.95, "fox": 0.99999}
    clip = FakeClip(reply="cat", prompt_reply="fox")

    traces = tagging.tag_images(clip, [an_image(), an_image(), an_image()], ["a fox", "a fox", " "])
    asked = [q.instructions for request, _image in fake_classifier.calls_of("choice")
             for q in request.questions.values()]
    assert len(clip.text_only_calls()) == 1
    assert asked == ['What does the image tag "fox" name']
    assert [trace.tags for trace in traces] == ["fox, cat", "fox, cat", "cat"]
    assert [trace.origins for trace in traces] == [("model", "prompt"), ("model", "prompt"), ("model",)]


def test_tag_images_drops_a_prompt_thing_the_thing_check_calls_other(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.95, "fox": 0.99999, "dusk": 0.99999}
    fake_classifier.other = {"dusk": 0.95}
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


def test_tag_images_passes_the_instruction_to_the_propose_stage(fake_classifier):
    clip = FakeClip()

    tagging.tag_images(clip, [an_image()], [""], instruction="Name the animals.")
    assert "Name the animals." in clip.tokenize_calls[0][0]


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


# --- tag list -------------------------------------------------------------------------


def listed(**probabilities):
    """The fake classifier's noul table, keyed by the presence question each tag asks."""
    from logit_classifier.toolkit.tags import presence_question

    return {presence_question(tag.replace("_", " ")): p for tag, p in probabilities.items()}


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


def test_parse_tag_list_without_the_library_prints_the_pip_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.tags", None)

    with pytest.raises(RuntimeError, match=r"^Logit Tagger: .*pip install -U logit-classifier"):
        tagging.parse_tag_list("cat")


def test_parse_tag_list_has_no_cap():
    pytest.importorskip("logit_classifier")

    assert len(tagging.parse_tag_list(", ".join(f"tag {index}" for index in range(300)))) == 300


def test_check_tag_list_keeps_tags_at_or_above_the_threshold(fake_classifier):
    threshold = tagging.TAG_LIST_THRESHOLD
    fake_classifier.probabilities = listed(cat=threshold, dog=threshold - 1e-6)

    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), ("cat", "dog"))
    assert trace.kept == ("cat",)
    assert trace.scores == (threshold, threshold - 1e-6)


def test_check_tag_list_orders_by_probability_and_ties_keep_list_order(fake_classifier):
    fake_classifier.probabilities = listed(cat=0.99991, dog=0.99999, fox=0.99991, owl=0.5)

    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), ("cat", "dog", "fox", "owl"))
    assert trace.tags == "dog, cat, fox"


def test_check_tag_list_reports_the_listed_tags_as_its_only_candidates(fake_classifier):
    fake_classifier.probabilities = listed(cat=1.0)

    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(height=16), ("cat", "red car"))
    assert (trace.reply, trace.proposed, trace.dropped, trace.propose_ms) == ("", (), (), 0.0)
    assert trace.candidates == ("cat", "red car")
    assert trace.origins == ("list", "list")
    assert (trace.size_in, trace.size_out) == ((16, 8), (16, 8))


def test_check_tag_list_asks_the_presence_question_about_the_fitted_image(fake_classifier, monkeypatch):
    monkeypatch.setattr(engine, "CHECK_MAX_PIXELS", 16)
    clip = FakeClip()

    tagging.check_tag_list(clip, fake_classifier, an_image(), ("cat", "red car"))
    ((request, image),) = fake_classifier.calls_of("noul")
    assert [q.instructions for q in request.questions.values()] == [
        "Is there a cat in this image?", "Is there a red car in this image?"]
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
    trace = tagging.check_tag_list(FakeClip(), fake_classifier, an_image(), tags)
    assert [len(request.questions) for request, _image in fake_classifier.calls] == [256, 44]
    assert open_during == [[True], [True]]
    assert fake_classifier.scopes[0]["open"] is False
    assert len(trace.scores) == 300
    assert trace.kept == ("tag 0", "tag 299")


def test_tag_images_with_a_tag_list_checks_every_image_and_never_generates(fake_classifier):
    fake_classifier.probabilities = listed(cat=1.0)
    clip = FakeClip(reply="dog", prompt_reply="fox")
    progress = []

    traces = tagging.tag_images(clip, [an_image(), an_image()], ["", ""], tag_list=("cat", "dog"),
                                on_progress=lambda index, stage: progress.append((index, stage)))
    assert [trace.tags for trace in traces] == ["cat", "cat"]
    assert progress == [(0, "checking tags"), (1, "checking tags")]
    assert clip.generate_calls == []
    assert fake_classifier.calls_of("choice") == []
    assert fake_classifier.built == [(clip, tagging.NODE_NAME)]


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
