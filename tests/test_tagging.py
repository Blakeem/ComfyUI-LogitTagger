"""tagging.py: the propose, merge, verify and clean pipeline behind the Logit Tagger node.

A duck typed CLIP stands in for the Qwen3-VL text encoder and a fake classifier stands in
for the library's Classifier, so no comfy install, no GPU and no model are needed.
"""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from logit_tagger import engine, tagging

PACK_ROOT = Path(__file__).parent.parent.resolve()
VERIFY_PREFIX = tagging.VERIFY_STATEMENT.split(tagging.TAG_PLACEHOLDER)[0]


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
        return {"qwen3vl_4b": [[(text, 1.0)]]}

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
    """

    def __init__(self, probabilities=None, other=None, english=1.0):
        self.probabilities = probabilities or {}
        self.other = other or {}
        self.english = english
        self.calls = []

    def classify(self, request, *, image=None):
        answers = {}

        self.calls.append((request, image))
        for qid, question in request.questions.items():
            if question.type == "choice":
                p_other = self.other.get(question.instructions.split('"')[1], 0.0)
                answers[qid] = SimpleNamespace(probabilities={"thing": 1.0 - p_other, "other": p_other})
            elif question.instructions == tagging.LANGUAGE_QUESTION:
                answers[qid] = SimpleNamespace(noul=self.english)
            else:
                tag = question.instructions.removeprefix(VERIFY_PREFIX)
                answers[qid] = SimpleNamespace(noul=self.probabilities.get(tag, 0.0))
        return SimpleNamespace(answers=answers), None

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
    """Stands in for engine.build_classifier and for the library's shared_vision_encode scope."""
    comfy_clip = pytest.importorskip("logit_classifier.backends.comfy_clip")
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
    monkeypatch.setattr(comfy_clip, "shared_vision_encode", shared_vision_encode)
    classifier.built = built
    classifier.scopes = scopes
    return classifier


def an_image(height=8, width=8, channels=3, batch=1):
    torch = pytest.importorskip("torch")
    return torch.rand(batch, height, width, channels)


def tag_one(clip, classifier, **kwargs):
    return tagging.tag_image(clip, classifier, an_image(), **kwargs)


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


# --- propose stage --------------------------------------------------------------------


def test_propose_sends_the_rendered_template_and_the_image():
    clip = FakeClip(reply="fox, snow")
    image = an_image()
    expected = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        + tagging.PROPOSE_INSTRUCTION
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

    text = tagging.propose(clip, image, tagging.PROPOSE_INSTRUCTION)
    (rendered, kwargs), = clip.tokenize_calls
    assert text == "fox, snow"
    assert rendered == expected
    assert kwargs.keys() == {"images"}
    assert kwargs["images"][0] is image
    assert len(kwargs["images"]) == 1
    assert "<think>" not in rendered


def test_propose_generates_greedy_within_the_budget():
    clip = FakeClip()

    tagging.propose(clip, an_image(), tagging.PROPOSE_INSTRUCTION)
    (_tokens, kwargs), = clip.generate_calls
    assert kwargs == {"do_sample": False, "max_length": tagging.PROPOSE_MAX_TOKENS}


def test_propose_drops_the_cut_tail_when_the_decode_fills_the_budget():
    clip = FakeClip(reply="fox, snow, red fire hyd", id_count=tagging.PROPOSE_MAX_TOKENS)

    assert tagging.propose(clip, an_image(), tagging.PROPOSE_INSTRUCTION) == "fox, snow"


STOP_ID = 0


class ScriptedTransformer:
    """Returns one scripted token per sample_token call, then a stop token, as core's sampler does."""

    def __init__(self, script):
        self.script = [*script, STOP_ID]
        self.model = SimpleNamespace(config=SimpleNamespace(stop_tokens=[STOP_ID]))

    def sample_token(self, logits, *args, **kwargs):
        torch = pytest.importorskip("torch")
        return torch.tensor([[self.script.pop(0)]])


class DecodingClip(FakeClip):
    """Runs core's decode loop, at comfy/text_encoders/llama.py:1147-1187, over a word per token id."""

    def __init__(self, words):
        super().__init__()
        self.vocabulary = ["<stop>", *dict.fromkeys(words)]
        self.transformer = ScriptedTransformer([self.vocabulary.index(word) for word in words])
        self.cond_stage_model = SimpleNamespace(clip="qwen3vl_4b",
                                                qwen3vl_4b=SimpleNamespace(transformer=self.transformer))

    def generate(self, tokens, **kwargs):
        ids = []

        self.generate_calls.append((tokens, kwargs))
        for _step in range(kwargs["max_length"]):
            ids.append(int(self.transformer.sample_token(None)[0, 0]))
            if ids[-1] == STOP_ID:
                break
        return ids

    def decode(self, ids):
        return "".join(self.vocabulary[token] for token in ids if token != STOP_ID)


def words_of(text):
    return [part for chunk in text.split(", ") for part in (chunk, ", ")][:-1]


def test_propose_stops_the_decode_after_the_tag_cap():
    tags = [f"item {index}" for index in range(tagging.MAX_PROPOSED_TAGS + 5)]
    clip = DecodingClip(words_of(", ".join(tags)))

    text = tagging.propose(clip, an_image(), tagging.PROPOSE_INSTRUCTION)
    assert text == ", ".join(tags[:tagging.MAX_PROPOSED_TAGS])


def test_propose_stops_the_decode_on_a_repeated_block():
    clip = DecodingClip([*words_of("sky, city lights, tower, city lights, tower, "), "moon"])

    text = tagging.propose(clip, an_image(), tagging.PROPOSE_INSTRUCTION)
    assert text == "sky, city lights, tower, city lights, tower"


def test_propose_keeps_a_repeat_that_is_not_back_to_back():
    clip = DecodingClip(words_of("rope, cart, rope, man, man with hat, manatee"))

    text = tagging.propose(clip, an_image(), tagging.PROPOSE_INSTRUCTION)
    assert text == "rope, cart, rope, man, man with hat, manatee"


def test_propose_restores_core_sample_token():
    clip = DecodingClip(words_of("cat, dog"))

    tagging.propose(clip, an_image(), tagging.PROPOSE_INSTRUCTION)
    assert "sample_token" not in vars(clip.transformer)


def test_propose_stop_predicate_at_the_cap_and_on_a_repeat():
    pytest.importorskip("logit_classifier.tags")
    below = [f"item {index}" for index in range(tagging.MAX_PROPOSED_TAGS - 1)]

    assert tagging._full_or_repeating(below) is False
    assert tagging._full_or_repeating([*below, "last"]) is True
    assert tagging._full_or_repeating(["sky", "tower", "moon", "tower", "moon"]) is True


# --- prompt grounding -----------------------------------------------------------------


def test_word_forms_keep_every_plural_cut():
    assert tagging._word_forms("Trees") == {"trees": {"trees", "tree", "tre"}}
    assert tagging._word_forms("berries") == {"berries": {"berries", "berrie", "berri", "berry"}}
    assert tagging._word_forms("gas") == {"gas": {"gas"}}


def test_grounded_matches_plurals_and_skips_stop_words():
    words = tagging.prompt_forms("Tall trees with red berries beside two cars")

    assert tagging.grounded("tree", words)
    assert tagging.grounded("red berry", words)
    assert tagging.grounded("a car with the trees", words)
    assert not tagging.grounded("red boat", words)
    assert not tagging.grounded("the of", words)


def test_prompt_list_stops_on_an_off_prompt_streak_or_a_repeat():
    pytest.importorskip("logit_classifier.tags")
    words = tagging.prompt_forms("a fox")

    assert tagging._off_prompt_or_repeating(["fox", "materials", "details"], words) is True
    assert tagging._off_prompt_or_repeating(["fox", "materials"], words) is False
    assert tagging._off_prompt_or_repeating(["materials"], words) is False
    assert tagging._off_prompt_or_repeating(["fox", "fox"], words) is True


def test_prompt_things_skip_the_generate_for_a_blank_prompt():
    clip = FakeClip()
    classifier = FakeClassifier()

    assert tagging.prompt_things(clip, classifier, "", {}) == ()
    assert tagging.prompt_things(clip, classifier, "  \n ", {}) == ()
    assert clip.generate_calls == []
    assert classifier.calls == []


def test_prompt_things_keep_grounded_things():
    pytest.importorskip("logit_classifier")
    clip = FakeClip(prompt_reply="red fox, tall trees, dusk, glowing aura")
    classifier = FakeClassifier(other={"dusk": 0.95})

    things = tagging.prompt_things(clip, classifier, "a red fox by {TAG} tall trees at dusk", {})
    (rendered,) = clip.text_only_calls()
    assert things == ("red fox", "tall trees")
    assert "a red fox by {TAG} tall trees at dusk" in rendered
    assert tagging.PROMPT_PLACEHOLDER not in rendered
    assert clip.generate_calls[0][1]["max_length"] == tagging.PROMPT_TAGS_MAX_TOKENS


# --- prompt language ------------------------------------------------------------------


def test_prompt_things_list_a_non_english_prompt_from_its_english_translation():
    pytest.importorskip("logit_classifier")
    prompt = "Eine Katze liegt auf einer Computertastatur."
    clip = ScriptedClip(prompt_replies=["A cat on a keyboard.", "cat, keyboard"])
    classifier = FakeClassifier(english=0.01)

    things = tagging.prompt_things(clip, classifier, prompt, {})
    translate, listing = clip.text_only_calls()
    (language_request, _image), = classifier.calls_of("noul")
    assert things == ("cat", "keyboard")
    assert [q.instructions for q in language_request.questions.values()] == [tagging.LANGUAGE_QUESTION]
    assert language_request.state == prompt
    assert tagging.TRANSLATE_INSTRUCTION.replace(tagging.PROMPT_PLACEHOLDER, prompt) in translate
    assert "A cat on a keyboard." in listing
    assert prompt not in listing
    assert clip.generate_calls[0][1]["max_length"] == tagging.TRANSLATE_MAX_TOKENS


def test_prompt_things_of_an_english_prompt_run_no_translation():
    pytest.importorskip("logit_classifier")
    clip = FakeClip(prompt_reply="fox")

    assert tagging.prompt_things(clip, FakeClassifier(english=0.99), "a fox", {}) == ("fox",)
    assert len(clip.generate_calls) == 1
    assert "Translate" not in clip.text_only_calls()[0]


def test_a_blank_translation_falls_back_to_the_prompt():
    pytest.importorskip("logit_classifier")
    clip = ScriptedClip(prompt_replies=[" \n", "fox"])

    assert tagging.prompt_things(clip, FakeClassifier(english=0.01), "a fox", {}) == ("fox",)
    assert "a fox" in clip.text_only_calls()[1]


# --- merge and thing check ------------------------------------------------------------


def test_merge_records_origins_category_nouns_and_repeats():
    pytest.importorskip("logit_classifier.tags")

    candidates, origins, dropped = tagging.merge_trace(("cat", "objects", "dog", "a cat"),
                                                       ("dog", "the fox", "fox"))
    assert candidates == ("cat", "dog", "fox")
    assert origins == ("model", "both", "prompt")
    assert dropped == (("objects", "category noun"), ("cat", "repeat"), ("fox", "repeat"))


def test_merge_caps_the_candidates_with_model_tags_first():
    pytest.importorskip("logit_classifier.tags")
    proposed = tuple(f"item {index}" for index in range(tagging.MAX_MERGED_TAGS))

    candidates, origins, dropped = tagging.merge_trace(proposed, ("extra",))
    assert candidates == proposed
    assert set(origins) == {"model"}
    assert dropped == (("extra", "over the cap"),)


def test_thing_scores_ask_each_distinct_tag_once_across_calls():
    pytest.importorskip("logit_classifier")
    classifier = FakeClassifier(other={"sky": 0.2, "dusk": 0.95})
    known = {}

    assert tagging.thing_scores(classifier, ("cat", "sky", "cat"), known) == (0.0, 0.2, 0.0)
    assert tagging.thing_scores(classifier, ("cat", "dusk"), known) == (0.0, 0.95)
    assert tagging.thing_scores(classifier, ("sky",), known) == (0.2,)
    asked = [[q.instructions for q in request.questions.values()] for request, _image in classifier.calls]
    assert asked == [['What does the image tag "cat" name', 'What does the image tag "sky" name'],
                     ['What does the image tag "dusk" name']]
    assert [image for _request, image in classifier.calls] == [None, None]


def test_is_thing_drops_only_a_confident_other():
    assert tagging.is_thing(tagging.THING_THRESHOLD - 0.01)
    assert not tagging.is_thing(tagging.THING_THRESHOLD)


# --- verify thresholds ----------------------------------------------------------------


def test_origin_thresholds():
    passed = tagging._passed(("cat", "dog", "fox", "owl", "elk"),
                             ("model", "both", "prompt", "prompt", "model"),
                             (0.9, 0.9, 0.9999, 0.9998, 0.8999))

    assert passed == ("cat", "dog", "fox")


def test_thresholds_are_read_at_call_time(monkeypatch):
    monkeypatch.setattr(tagging, "VERIFY_THRESHOLD", 0.95)

    assert tagging._passed(("cat",), ("model",), (0.9,)) == ()


# --- tag_image ------------------------------------------------------------------------


def test_tag_image_reports_every_stage(fake_classifier, monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 100)
    fake_classifier.probabilities = {"cat": 0.95, "dog": 0.3, "fox": 0.99}
    fake_classifier.other = {"sunset": 0.95}

    trace = tagging.tag_image(FakeClip(reply="cat, objects, sunset, dog"), fake_classifier, an_image(40, 20),
                              prompt_tags=("dog", "fox"))
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


def test_tag_image_verifies_every_candidate_on_the_resized_image(fake_classifier):
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


def test_tag_image_drops_subsets(fake_classifier):
    fake_classifier.probabilities = {"hat": 0.95, "red hat": 0.95, "dog": 0.95}

    assert tag_one(FakeClip(reply="hat, red hat, dog"), fake_classifier).tags == "red hat, dog"


def test_tag_image_orders_by_probability_and_ties_keep_proposal_order(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.95, "dog": 0.99, "fox": 0.95, "owl": 0.999}

    trace = tag_one(FakeClip(reply="cat, dog, fox, owl"), fake_classifier)
    assert trace.tags == "owl, dog, cat, fox"
    assert trace.kept == ("owl", "dog", "cat", "fox")


def test_tag_image_caps_the_proposed_tags(fake_classifier):
    reply = ", ".join(f"item {index}" for index in range(tagging.MAX_PROPOSED_TAGS + 5))

    trace = tag_one(FakeClip(reply=reply), fake_classifier)
    assert len(trace.proposed) == tagging.MAX_PROPOSED_TAGS
    assert trace.proposed[-1] == f"item {tagging.MAX_PROPOSED_TAGS - 1}"


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
    assert tagging.ECHO_FALLBACK_INSTRUCTION in clip.tokenize_calls[1][0]


def test_a_custom_instruction_never_triggers_the_echo_fallback(fake_classifier):
    clip = ScriptedClip(replies=["objects, people, animals", "red car, road"])

    trace = tag_one(clip, fake_classifier, instruction="Name the things.")
    assert len(clip.generate_calls) == 1
    assert trace.echo_reply is None
    assert trace.candidates == ()


def test_tag_image_instruction_override(fake_classifier):
    clip = FakeClip()

    tag_one(clip, fake_classifier, instruction="Name the animals.")
    rendered = clip.tokenize_calls[0][0]
    assert "Name the animals." in rendered
    assert tagging.PROPOSE_INSTRUCTION not in rendered


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
        tagging.tag_image(clip, fake_classifier, an_image(batch=2))
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
    return {engine.presence_question(tag.replace("_", " ")): p for tag, p in probabilities.items()}


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
    monkeypatch.setitem(sys.modules, "logit_classifier.tags", None)

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


def test_check_tag_list_asks_the_presence_question_about_the_resized_image(fake_classifier, monkeypatch):
    monkeypatch.setattr(engine, "CHECK_MAX_PIXELS", 16)
    clip = FakeClip()

    tagging.check_tag_list(clip, fake_classifier, an_image(), ("cat", "red car"))
    ((request, image),) = fake_classifier.calls_of("noul")
    assert [q.instructions for q in request.questions.values()] == [
        "Is there a cat in this image?", "Is there a red car in this image?"]
    assert request.state == ""
    assert tuple(image.shape) == (1, 4, 4, 3)
    assert clip.generate_calls == []


def test_check_tag_list_splits_300_tags_into_requests_of_256_inside_one_scope(fake_classifier):
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
    pytest.importorskip("logit_classifier.backends.comfy_clip")
    clip = FakeClip()

    with pytest.raises(ValueError, match=r"^Logit Tagger: this CLIP is not a Qwen3-VL text encoder"):
        tagging.tag_images(clip, [an_image()], ["a fox"])
    assert clip.generate_calls == []


def test_a_clip_without_a_vision_tower_fails_before_any_generate(monkeypatch):
    comfy_clip = pytest.importorskip("logit_classifier.backends.comfy_clip")
    clip = FakeClip()

    class BlindBackend:
        sees_images = False

        def __init__(self, clip):
            self.clip = clip

    monkeypatch.setattr(comfy_clip, "ComfyClipBackend", BlindBackend)
    with pytest.raises(ValueError, match="Logit Tagger"):
        tagging.tag_images(clip, [an_image()], ["a fox"])
    assert clip.generate_calls == []
