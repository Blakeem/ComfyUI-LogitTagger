"""tagging.py: the five stage tagging pipeline behind the Logit Tagger node.

A duck typed CLIP stands in for the Qwen3-VL text encoder and a fake classifier stands in
for the library's Classifier, so no comfy install, no GPU and no model are needed.
"""

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from logit_tagger import tagging

PACK_ROOT = Path(__file__).parent.parent.resolve()


class FakeClip:
    """Records every tokenize and generate call, and decodes to canned text."""

    def __init__(self, reply="cat, dog", id_count=3):
        self.reply = reply
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
        return self.reply


class FakeClassifier:
    """Answers each noul question with the probability its tag maps to."""

    def __init__(self, probabilities=None):
        self.probabilities = probabilities or {}
        self.calls = []

    def classify(self, request, *, image=None):
        answers = {}
        by_statement = {
            tagging.VERIFY_STATEMENT.format(tag=tag): p for tag, p in self.probabilities.items()
        }

        self.calls.append((request, image))
        for qid, question in request.questions.items():
            answers[qid] = SimpleNamespace(noul=by_statement.get(question.instructions, 0.0))
        return SimpleNamespace(answers=answers), None


@pytest.fixture
def fake_classifier(monkeypatch):
    classifier = FakeClassifier()
    built = []

    def build(clip):
        built.append(clip)
        return classifier

    monkeypatch.setattr(tagging, "build_classifier", build)
    classifier.built = built
    return classifier


def an_image(height=8, width=8, channels=3, batch=1):
    torch = pytest.importorskip("torch")
    return torch.rand(batch, height, width, channels)


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


# --- resize stage ---------------------------------------------------------------------


def test_resize_keeps_an_image_under_the_cap():
    image = an_image(16, 24)

    resized = tagging.resize_for_vl(image)
    assert resized.shape == (1, 16, 24, 3)
    assert resized.equal(image)


def test_resize_caps_pixels_and_keeps_aspect(monkeypatch):
    monkeypatch.setattr(tagging, "VL_MAX_PIXELS", 100)
    image = an_image(40, 20)

    resized = tagging.resize_for_vl(image)
    height, width = resized.shape[1], resized.shape[2]
    assert height * width <= 100
    assert (height, width) == (14, 7)


def test_resize_never_upscales(monkeypatch):
    monkeypatch.setattr(tagging, "VL_MAX_PIXELS", 10_000)
    image = an_image(4, 5)

    assert tagging.resize_for_vl(image).shape == (1, 4, 5, 3)


def test_resize_keeps_only_three_channels(monkeypatch):
    monkeypatch.setattr(tagging, "VL_MAX_PIXELS", 16)
    small = an_image(4, 4, channels=4)
    large = an_image(8, 8, channels=4)

    assert tagging.resize_for_vl(small).shape == (1, 4, 4, 3)
    assert tagging.resize_for_vl(large).shape == (1, 4, 4, 3)


# --- propose stage --------------------------------------------------------------------


def test_instruction_has_no_anchor_without_a_prompt():
    assert tagging.build_instruction("", tagging.PROPOSE_INSTRUCTION) == tagging.PROPOSE_INSTRUCTION
    assert tagging.build_instruction("  \n ", tagging.PROPOSE_INSTRUCTION) == tagging.PROPOSE_INSTRUCTION


def test_instruction_prepends_the_filled_anchor_for_a_prompt():
    anchor = tagging.PROMPT_ANCHOR.format(prompt="a red fox")

    instruction = tagging.build_instruction("  a red fox ", tagging.PROPOSE_INSTRUCTION)
    assert instruction == anchor + tagging.PROPOSE_INSTRUCTION


def test_propose_sends_the_rendered_template_and_the_image():
    clip = FakeClip(reply="fox, snow")
    image = an_image()
    expected = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        + tagging.PROPOSE_INSTRUCTION
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

    text = tagging.propose(clip, image, "", tagging.PROPOSE_INSTRUCTION)
    (rendered, kwargs), = clip.tokenize_calls
    assert text == "fox, snow"
    assert rendered == expected
    assert kwargs.keys() == {"images"}
    assert kwargs["images"][0] is image
    assert len(kwargs["images"]) == 1
    assert "<think>" not in rendered


def test_propose_generates_greedy_within_the_budget():
    clip = FakeClip()

    tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    (_tokens, kwargs), = clip.generate_calls
    assert kwargs == {
        "do_sample": False,
        "max_length": tagging.PROPOSE_MAX_TOKENS,
    }


def test_propose_drops_the_cut_tail_when_the_decode_fills_the_budget():
    clip = FakeClip(reply="fox, snow, red fire hyd", id_count=tagging.PROPOSE_MAX_TOKENS)

    assert tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION) == "fox, snow"


def test_propose_keeps_the_whole_reply_when_the_decode_stops_early():
    clip = FakeClip(reply="fox, snow, red fire hyd", id_count=tagging.PROPOSE_MAX_TOKENS - 1)

    text = tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    assert text == "fox, snow, red fire hyd"


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


@pytest.mark.parametrize(("tags", "size"), [
    (["man", "man"], 1),
    (["man", "man with hat"], 0),
    (["man", "manatee"], 0),
    (["sky", "a", "b", "c", "a", "b", "c"], 3),
    (["a", "b", "c", "d", "a", "b", "c", "d"], 4),
    (["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"], 0),
    (["a", "b", "a"], 0),
    ([], 0),
])
def test_repeated_block_finds_only_a_back_to_back_block(tags, size):
    assert tagging.repeated_block(tags) == size


def test_complete_tags_skip_the_tag_still_being_written():
    assert tagging.complete_tags("man, man") == ["man"]
    assert tagging.complete_tags("man, man,") == ["man", "man"]


def test_propose_stops_the_decode_on_a_repeated_tag():
    clip = DecodingClip(["cat", ", ", "man", ", ", "man", ", ", "dog", ", ", "fox"])

    text = tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    assert text == "cat, man, man"
    assert tagging.parse_candidates(text) == ["cat", "man"]


def test_propose_stops_the_decode_on_a_repeated_block():
    clip = DecodingClip([*words_of("sky, city lights, tower, city lights, tower, "), "moon"])

    text = tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    assert text == "sky, city lights, tower, city lights, tower"


def test_propose_keeps_a_repeat_that_is_not_back_to_back():
    clip = DecodingClip(words_of("rope, cart, rope, man, man with hat, manatee"))

    text = tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    assert text == "rope, cart, rope, man, man with hat, manatee"


def test_propose_restores_core_sample_token():
    clip = DecodingClip(words_of("cat, dog"))

    tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    assert "sample_token" not in vars(clip.transformer)


def test_repeat_stop_restores_sample_token_after_an_exception():
    clip = DecodingClip(words_of("cat, dog"))

    with pytest.raises(RuntimeError, match="boom"), tagging.stop_on_repeat(clip):
        raise RuntimeError("boom")
    assert "sample_token" not in vars(clip.transformer)


def test_repeat_stop_is_a_no_op_for_a_clip_without_core_sampler():
    clip = FakeClip()

    with tagging.stop_on_repeat(clip):
        pass
    assert not hasattr(clip, "cond_stage_model")


def test_propose_anchor_reaches_tokenize_only_with_a_prompt():
    clip = FakeClip()
    anchor = tagging.PROMPT_ANCHOR.format(prompt="a red fox")

    tagging.propose(clip, an_image(), "", tagging.PROPOSE_INSTRUCTION)
    tagging.propose(clip, an_image(), "a red fox", tagging.PROPOSE_INSTRUCTION)
    assert anchor not in clip.tokenize_calls[0][0]
    assert anchor in clip.tokenize_calls[1][0]


# --- CUDA graph guard -----------------------------------------------------------------


@pytest.fixture
def fake_comfy(monkeypatch):
    args = SimpleNamespace(disable_cuda_graphs=False)
    model_management = types.ModuleType("comfy.model_management")
    comfy = types.ModuleType("comfy")

    model_management.args = args
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    return args


def test_guard_sets_and_restores_the_flag(fake_comfy):
    with tagging.cuda_graphs_disabled():
        assert fake_comfy.disable_cuda_graphs is True
    assert fake_comfy.disable_cuda_graphs is False


def test_guard_keeps_a_flag_that_was_already_set(fake_comfy):
    fake_comfy.disable_cuda_graphs = True

    with tagging.cuda_graphs_disabled():
        assert fake_comfy.disable_cuda_graphs is True
    assert fake_comfy.disable_cuda_graphs is True


def test_guard_restores_the_flag_after_an_exception(fake_comfy):
    with pytest.raises(RuntimeError, match="boom"), tagging.cuda_graphs_disabled():
        raise RuntimeError("boom")
    assert fake_comfy.disable_cuda_graphs is False


def test_guard_is_a_no_op_without_comfy(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    ran = []

    with tagging.cuda_graphs_disabled():
        ran.append(True)
    assert ran == [True]


def test_propose_generates_inside_the_guard(fake_comfy):
    seen = []

    class RecordingClip(FakeClip):
        def generate(self, tokens, **kwargs):
            seen.append(fake_comfy.disable_cuda_graphs)
            return super().generate(tokens, **kwargs)

    tagging.propose(RecordingClip(), an_image(), "", tagging.PROPOSE_INSTRUCTION)
    assert seen == [True]
    assert fake_comfy.disable_cuda_graphs is False


# --- parse stage ----------------------------------------------------------------------


def test_parse_splits_on_commas_semicolons_and_newlines():
    assert tagging.parse_candidates("cat, dog; bird\nfish") == ["cat", "dog", "bird", "fish"]


def test_parse_strips_list_bullets():
    text = "- cat\n* dog\n• bird\n1. fish\n2) frog"

    assert tagging.parse_candidates(text) == ["cat", "dog", "bird", "fish", "frog"]


def test_parse_keeps_a_number_that_is_not_a_bullet():
    assert tagging.parse_candidates("3d render, 2 cats") == ["3d render", "2 cats"]


def test_parse_strips_quotes_and_trailing_periods():
    assert tagging.parse_candidates('"cat", \'dog\', bird.') == ["cat", "dog", "bird"]
    assert tagging.parse_candidates('"red hat."') == ["red hat"]


def test_parse_drops_a_think_block():
    assert tagging.parse_candidates("<think>\nmaybe a cat, a dog\n</think>\n\nfox, tree") == ["fox", "tree"]
    assert tagging.parse_candidates("<think>still thinking, cat") == []


def test_parse_drops_a_leading_assistant_line():
    assert tagging.parse_candidates("assistant\nfox, tree") == ["fox", "tree"]
    assert tagging.parse_candidates("fox\nassistant") == ["fox", "assistant"]


def test_parse_lowercases_and_collapses_whitespace():
    assert tagging.parse_candidates("Red   Wooden\tChair") == ["red wooden chair"]


def test_parse_drops_empty_and_long_items():
    seven_words = "one two three four five six seven"
    six_words = "one two three four five six"
    long_item = "a" * 61
    edge_item = "a" * 60

    candidates = tagging.parse_candidates(f"cat,, ,{seven_words},{six_words},{long_item},{edge_item}")
    assert candidates == ["cat", six_words, edge_item]


def test_parse_removes_duplicates_keeping_first_order():
    assert tagging.parse_candidates("dog, cat, Dog, cat.") == ["dog", "cat"]


def test_parse_caps_the_candidate_count():
    text = ", ".join(f"tag {index}" for index in range(tagging.MAX_CANDIDATES + 5))

    candidates = tagging.parse_candidates(text)
    assert len(candidates) == tagging.MAX_CANDIDATES
    assert candidates[-1] == f"tag {tagging.MAX_CANDIDATES - 1}"


# --- verify stage ---------------------------------------------------------------------


def test_verify_sends_one_request_with_one_noul_per_candidate():
    classifier = FakeClassifier({"cat": 0.9, "dog": 0.2})
    image = an_image()

    probabilities = tagging.verify(classifier, image, ["cat", "dog"])
    (request, sent_image), = classifier.calls
    questions = list(request.questions.values())
    assert probabilities == [0.9, 0.2]
    assert sent_image is image
    assert request.state == ""
    assert [q.type for q in questions] == ["noul", "noul"]
    assert [q.instructions for q in questions] == [
        "This image visibly contains cat",
        "This image visibly contains dog",
    ]


def test_verify_sends_no_request_without_candidates():
    classifier = FakeClassifier()

    assert tagging.verify(classifier, an_image(), []) == []
    assert classifier.calls == []


# --- filter stage ---------------------------------------------------------------------


def test_filter_keeps_a_probability_equal_to_the_threshold():
    kept = tagging.filter_tags(["cat", "dog"], [0.5, 0.4999], 0.5)

    assert kept == "cat"


def test_filter_orders_by_probability_and_ties_keep_proposal_order():
    kept = tagging.filter_tags(["a", "b", "c", "d"], [0.6, 0.9, 0.6, 0.95], 0.5)

    assert kept == "d, b, a, c"


def test_filter_yields_empty_text_when_nothing_survives():
    assert tagging.filter_tags(["cat"], [0.1], 0.5) == ""
    assert tagging.filter_tags([], [], 0.5) == ""


# --- entry and detail functions -------------------------------------------------------


def test_tag_images_runs_in_batch_order_with_one_callback_per_image(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9, "dog": 0.8}
    clip = FakeClip(reply="cat, dog")
    images = an_image(batch=3)
    done = []

    tags = tagging.tag_images(clip, images, "", on_image_done=done.append)
    sent = [image for _request, image in fake_classifier.calls]
    assert tags == ["cat, dog"] * 3
    assert done == [0, 1, 2]
    assert fake_classifier.built == [clip]
    assert [image.equal(images[index:index + 1]) for index, image in enumerate(sent)] == [True] * 3


def test_tag_images_without_a_callback(fake_classifier):
    tags = tagging.tag_images(FakeClip(reply="cat"), an_image(batch=2), "")

    assert tags == ["", ""]


def test_tag_images_rejects_an_image_that_is_not_4d(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match="Logit Tagger"):
        tagging.tag_images(clip, an_image()[0], "")
    assert fake_classifier.built == []
    assert clip.generate_calls == []


def test_tag_images_rejects_an_image_with_fewer_than_3_channels(fake_classifier):
    clip = FakeClip()

    with pytest.raises(ValueError, match=tagging.NODE_NAME):
        tagging.tag_images(clip, an_image(channels=1), "")
    assert clip.generate_calls == []


def test_tag_image_rejects_a_batch(fake_classifier):
    with pytest.raises(ValueError, match="Logit Tagger"):
        tagging.tag_image(FakeClip(), an_image(batch=2), "")


def test_tag_image_reports_sizes_text_candidates_and_timings(fake_classifier, monkeypatch):
    monkeypatch.setattr(tagging, "VL_MAX_PIXELS", 100)
    fake_classifier.probabilities = {"cat": 0.9, "dog": 0.3}

    detail = tagging.tag_image(FakeClip(reply="cat, dog"), an_image(40, 20), "")
    assert detail.size_in == (40, 20)
    assert detail.size_out == (14, 7)
    assert detail.proposal == "cat, dog"
    assert detail.candidates == (("cat", 0.9), ("dog", 0.3))
    assert detail.tags == "cat"
    assert isinstance(detail.propose_ms, float)
    assert isinstance(detail.verify_ms, float)
    assert detail.propose_ms >= 0
    assert detail.verify_ms >= 0


def test_tag_image_threshold_override(fake_classifier):
    fake_classifier.probabilities = {"cat": 0.9, "dog": 0.3}

    detail = tagging.tag_image(FakeClip(reply="cat, dog"), an_image(), "", threshold=0.2)
    assert detail.tags == "cat, dog"


def test_tag_image_reads_the_threshold_at_call_time(fake_classifier, monkeypatch):
    fake_classifier.probabilities = {"cat": 0.9, "dog": 0.3}
    monkeypatch.setattr(tagging, "NOUL_THRESHOLD", 0.95)

    detail = tagging.tag_image(FakeClip(reply="cat, dog"), an_image(), "")
    assert detail.tags == ""


def test_tag_image_instruction_override(fake_classifier):
    clip = FakeClip()

    tagging.tag_image(clip, an_image(), "", instruction="Name the animals.")
    rendered = clip.tokenize_calls[0][0]
    assert "Name the animals." in rendered
    assert tagging.PROPOSE_INSTRUCTION not in rendered


def test_tag_image_skips_the_request_when_nothing_parses(fake_classifier):
    detail = tagging.tag_image(FakeClip(reply="<think>hmm"), an_image(), "")

    assert detail.candidates == ()
    assert detail.tags == ""
    assert fake_classifier.calls == []


# --- library loading ------------------------------------------------------------------


def test_missing_library_names_the_node_the_version_and_the_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier", None)

    with pytest.raises(RuntimeError) as raised:
        tagging.build_classifier(FakeClip())
    message = str(raised.value)
    assert "Logit Tagger" in message
    assert "logit-classifier>=0.2.0" in message
    assert "pip install -U logit-classifier" in message


def test_library_without_the_comfy_backend_gets_the_same_message(monkeypatch):
    pytest.importorskip("logit_classifier")
    monkeypatch.setitem(sys.modules, "logit_classifier.backends.comfy_clip", None)

    with pytest.raises(RuntimeError, match=r"pip install -U logit-classifier"):
        tagging.build_classifier(FakeClip())


def test_a_clip_that_is_not_a_qwen_model_fails_before_any_generate():
    library = pytest.importorskip("logit_classifier.backends.base")
    clip = FakeClip()

    with pytest.raises(library.UnsupportedModelError):
        tagging.tag_images(clip, an_image(), "")
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
        tagging.tag_images(clip, an_image(), "")
    assert clip.generate_calls == []


def test_build_classifier_turns_off_the_prior_and_the_calibration_file(monkeypatch):
    comfy_clip = pytest.importorskip("logit_classifier.backends.comfy_clip")
    import logit_classifier

    built = {}

    class SeeingBackend:
        sees_images = True

        def __init__(self, clip):
            self.clip = clip

    class RecordingClassifier:
        def __init__(self, config, backend):
            built["config"] = config
            built["backend"] = backend

    monkeypatch.setattr(comfy_clip, "ComfyClipBackend", SeeingBackend)
    monkeypatch.setattr(logit_classifier, "Classifier", RecordingClassifier)
    clip = FakeClip()
    tagging.build_classifier(clip)
    assert built["config"].use_prior_debias is False
    assert built["config"].calibration_path is None
    assert built["backend"].clip is clip
